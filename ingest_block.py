# ingest_block.py (Optimized Version)

import os
import json
import hashlib
import logging
import argparse
import csv
import re
import warnings
import traceback
from pathlib import Path
import datetime as dt
from datetime import timezone
from typing import List, Dict, Optional, Tuple, Union, Any, Generator
from centroid_manager import CentroidManager
centroid_manager = CentroidManager()
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np
    


# PDF processing (ensure imports are correct)
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    logging.warning("pdfplumber not installed, table extraction in PDFs might be limited.")

# DOCX support
try:
    import docx
except ImportError:
    docx = None
    logging.warning("python-docx is not installed; DOCX files will not be processed.")

# Pandas for table formatting
try:
    import pandas as pd
except ImportError:
    pd = None
    logging.warning("pandas not installed, table formatting might be basic.")


from tenacity import retry, stop_after_attempt, wait_exponential

# LangChain components
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

# Weaviate v4 imports
import weaviate
# Remove unused Weaviate class imports if not needed here
# from weaviate.classes.config import Property, DataType, Configure,VectorDistances
from weaviate.classes.data import DataObject
# from weaviate.exceptions import WeaviateBaseError (already imported below)

# Local Configuration & Components
from config import cfg
# Import specific components needed from v7
from ingest_docs_v7 import PipelineConfig, RobustPDFLoaderV4
from calculate_centroid import should_recalculate_centroid


# Setup logging
logger = logging.getLogger(__name__)

def centroid_exists(centroid_path):
    return os.path.exists(centroid_path)


# Accepts an optional, already connected Weaviate client instance
def extract_text(filepath: Path, client_instance: Optional[weaviate.Client] = None) -> str:
    """Extracts text, reusing client if provided, especially for PDFs."""
    ext = filepath.suffix.lower()
    logger.debug(f"Extracting text from: {filepath.name} (type: {ext})")

    if ext == ".txt":
        return extract_text_from_txt(filepath)
    elif ext == ".docx":
        return extract_text_from_docx(filepath)
    elif ext == ".csv":
        return extract_text_from_csv(filepath)
    elif ext == ".pdf":
        temp_client_created = False # Flag to track if we created a temporary client
        loader_client = None
        try:
            # --- Use PASSED-IN CLIENT if available ---
            if client_instance and client_instance.is_connected(): # Check if passed client is usable
                 loader_client = client_instance
                 logger.debug(f"Reusing provided client instance for PDF extraction: {filepath.name}")
            else:
                # Fallback: Get a temporary client ONLY if none usable was passed
                if client_instance:
                     logger.warning(f"Passed client not connected for {filepath.name}, getting temporary one.")
                else:
                     logger.warning(f"No client passed to extract_text for {filepath.name}, getting temporary one.")

                loader_client = PipelineConfig.get_client() # Original problematic path
                if loader_client:
                    temp_client_created = True # Mark that we created one
                else:
                     logger.error(f"Failed to get fallback Weaviate client for PDF extraction: {filepath.name}")
                     return "" # Return empty on client failure

            # Use the determined client (passed-in or temporary)
            loader = RobustPDFLoaderV4(str(filepath), client=loader_client)
            docs = loader.load()
            return "\n".join(doc.page_content for doc in docs if doc.page_content)

        except Exception as e:
            logger.error(f"Error processing PDF file {filepath.name} with RobustLoader: {e}", exc_info=True)
            return ""
        finally:
            # --- Close ONLY if we created a temporary client ---
            if temp_client_created and loader_client and hasattr(loader_client, 'close'):
                 try:
                    if loader_client.is_connected():
                        loader_client.close()
                        logger.debug(f"Closed temporary client after PDF extraction: {filepath.name}")
                 except Exception as close_err:
                    logger.error(f"Error closing temporary client for {filepath.name}: {close_err}")
            # DO NOT close the client if it was passed in from the caller

    # Handle .md files
    elif ext == ".md":
        # Simple text reading for markdown, similar to txt
        return extract_text_from_txt(filepath)
    else:
        # logger.warning(f"Unsupported file type for extraction: {filepath.name}") # Already logged by caller
        return ""

class CleanableOllamaEmbeddings(OllamaEmbeddings):
    def __del__(self):
        """Ensure client is closed when object is garbage collected"""
        self.close()
        
    def close(self):
        """Close the underlying HTTP client"""
        if hasattr(self, 'client') and self.client:
            self.client.close()
            logger.info("Closed Ollama embeddings client")

# --- Incremental Ingestion Class ---
class IncrementalDocumentProcessorBlock:
    def __init__(self, data_dir: str, meta_filename="ingested_files.json"):
        self.data_dir = Path(data_dir).resolve()
        if not self.data_dir.is_dir():
            logger.warning(f"Data directory {self.data_dir} does not exist. Creating.")
            try:
                self.data_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                 logger.error(f"Failed to create data directory {self.data_dir}: {e}")
                 raise # Stop if directory cannot be ensured
        # Initialize domain_keywords as empty set
        self.domain_keywords = set()
        self.meta_file = self.data_dir.parent / meta_filename 
        self.processed_files = self.load_metadata()

        
        self.domain_keywords = set(getattr(cfg.env, 'merged_keywords', []))
        logger.info(f"Cached {len(self.domain_keywords)} domain keywords for quality scoring")         
        self.meta_file = self.data_dir.parent / meta_filename # Store metafile outside data dir
        self.processed_files = self.load_metadata()
        self._load_domain_keywords()
    
    def _load_domain_keywords(self):
        """Load domain keywords from config, if available"""
        try:
            from config import cfg
            if hasattr(cfg, 'env') and hasattr(cfg.env, 'merged_keywords'):
                self.domain_keywords = set(cfg.env.merged_keywords)
                logger.info(f"Cached {len(self.domain_keywords)} domain keywords for quality scoring")
            else:
                logger.warning("No merged_keywords found in configuration")
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not load domain keywords: {e}")
            
    def load_metadata(self) -> dict:
        """Load a JSON file mapping file paths to their last processed hash."""
        if self.meta_file.exists():
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading metadata from {self.meta_file}: {e}. Starting fresh.")
                return {}
        logger.info(f"Metadata file {self.meta_file} not found. Starting fresh.")
        return {}

    def save_metadata(self) -> None:
        """Save the updated metadata dictionary."""
        try:
            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump(self.processed_files, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving metadata to {self.meta_file}: {e}")

    def compute_hash(self, filepath: Path) -> str:
        """Compute the SHA-256 hash of a file’s contents."""
        hasher = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                while True:
                    buf = f.read(65536) # Read in chunks
                    if not buf:
                        break
                    hasher.update(buf)
            return hasher.hexdigest()
        except IOError as e:
            logger.error(f"Error reading file for hash {filepath.name}: {e}")
            return "" # Return empty string on error

    def load_new_documents(self, client_instance: weaviate.Client) -> List[Document]:
        """Iterate over files, process new/updated ones with parallel processing."""
        docs = []
        counter_lock = threading.Lock()
        logger.info(f"Scanning directory: {self.data_dir}")
        
        # Collect files to process
        files_to_process = []
        for filepath in self.data_dir.glob("*"):
            if not filepath.is_file():
                continue
                    
            # Check file type support
            supported_extensions = set(f".{ext.lower()}" for ext in getattr(PipelineConfig, 'FILE_TYPES', ['pdf', 'txt', 'csv', 'docx', 'md']))
            if filepath.suffix.lower() not in supported_extensions:
                continue
                
            current_hash = self.compute_hash(filepath)
            if not current_hash:
                continue
                
            stored_hash = self.processed_files.get(str(filepath))
            if stored_hash == current_hash:
                continue
                
            files_to_process.append(filepath)
        
        logger.info(f"Found {len(files_to_process)} new/modified files to process")
        
        # Process files in parallel
        max_workers = min(os.cpu_count() or 4, 8)  # Limit to 8 threads max
        logger.info(f"Starting parallel processing with {max_workers} worker threads")
        processed_count = 0
        error_count = 0
        # Thread-safe counter
        
        
        def process_file(filepath):
            nonlocal processed_count, error_count
            logger.info(f"Processing file: {filepath.name}")
            text = extract_text(filepath, client_instance=client_instance)
            
            if text is None or not text.strip():
                with counter_lock:
                    self.processed_files[str(filepath)] = self.compute_hash(filepath)
                return None
                
            try:
                file_path_obj = Path(filepath)
                stat = file_path_obj.stat()
                created_dt = dt.datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat().replace('+00:00', 'Z')
                modified_dt = dt.datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace('+00:00', 'Z')
                
                metadata = {
                    "source": str(filepath),
                    "filetype": file_path_obj.suffix.lower(),
                    "created_date": created_dt,
                    "modified_date": modified_dt,
                    "page": 0,
                    "content_flags": [],
                    "error_messages": [],
                    "quality_score": calculate_quality_score(text, self.domain_keywords)  # Pass cached keywords
                }
                
                doc = Document(page_content=text, metadata=metadata)
                
                with counter_lock:
                    self.processed_files[str(filepath)] = self.compute_hash(filepath)
                    processed_count += 1
                    
                return doc
            except Exception as e:
                with counter_lock:
                    error_count += 1
                logger.error(f"Error processing {filepath.name}: {e}")
                return None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_file, filepath): filepath for filepath in files_to_process}
            for future in as_completed(future_to_file):
                doc = future.result()
                if doc:
                    docs.append(doc)
    
        logger.info(f"Parallel processing complete. Used {max_workers} threads, Processed: {processed_count}, Errors: {error_count}")
        return docs

       
    def process(self, client_instance: weaviate.Client) -> List[Document]:
        """Load new documents, extract tables, calculate quality, and filter."""
        docs = []
        counter_lock = threading.Lock()

        # Initialize embeddings object
        embeddings = OllamaEmbeddings(
        model=PipelineConfig.EMBEDDING_MODEL,
        base_url=PipelineConfig.EMBEDDING_BASE_URL)


        logger.info(f"Scanning directory: {self.data_dir}")
        
        # Collect files to process
        files_to_process = []
        for filepath in self.data_dir.glob("*"):
            if not filepath.is_file():
                continue
                
            # Check file type support
            supported_extensions = set(f".{ext.lower()}" for ext in getattr(PipelineConfig, 'FILE_TYPES', ['pdf', 'txt', 'csv', 'docx', 'md']))
            if filepath.suffix.lower() not in supported_extensions:
                continue
                
            current_hash = self.compute_hash(filepath)
            if not current_hash:
                continue
                
            stored_hash = self.processed_files.get(str(filepath))
            if stored_hash == current_hash:
                continue
                
            files_to_process.append(filepath)
            
        logger.info(f"Found {len(files_to_process)} new/modified files to process")
        
        # Process files in parallel with adaptive batch sizing
        batch_size = self._calculate_optimal_batch_size(files_to_process)
        max_workers = min(os.cpu_count() or 4, 8)
        
        logger.info(f"Starting parallel processing with {max_workers} worker threads and batch size {batch_size}")
        
        processed_count = 0
        error_count = 0
        
        
        for batch in self._create_batches(files_to_process, batch_size):
            batch_docs = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(self._process_single_file, filepath, client_instance): filepath for filepath in batch}
                for future in as_completed(future_to_file):
                    doc = future.result()
                    if doc:
                        batch_docs.append(doc)
                        
            docs.extend(batch_docs)
            logger.info(f"Completed batch of {len(batch)} files, extracted {len(batch_docs)} documents")
        
        # Split documents
        if not docs:
            logger.info("No new documents loaded.")
            self.save_metadata()
            return []
            
        logger.info(f"Splitting {len(docs)} new documents...")
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=PipelineConfig.CHUNK_SIZE,
                chunk_overlap=PipelineConfig.CHUNK_OVERLAP
            )
            split_docs = splitter.split_documents(docs)
        except Exception as split_e:
            logger.error(f"Error during document splitting: {split_e}", exc_info=True)
            self.save_metadata()
            return []
            
        # Filter by minimum content length
        length_filtered_docs = [d for d in split_docs if len(d.page_content) >= PipelineConfig.MIN_CONTENT_LENGTH]
        
        # Filter by quality score
        min_quality_threshold = getattr(PipelineConfig, 'MIN_QUALITY_SCORE', 0.3)
        quality_filtered_docs = [
            d for d in length_filtered_docs
            if d.metadata.get('quality_score', 0) >= min_quality_threshold
        ]
        
        rejected_count = len(length_filtered_docs) - len(quality_filtered_docs)
        if rejected_count > 0:
            logger.info(f"Rejected {rejected_count} document chunks due to low quality scores (threshold: {min_quality_threshold})")
        
        logger.info(f"Prepared {len(quality_filtered_docs)} valid document chunks for insertion")
        self.save_metadata()
        
        # --- Centroid significance check and update ---
        try:
            if quality_filtered_docs and centroid_manager:
                # 1. Collect new vectors as numpy arrays
                new_vectors = []
                for doc in quality_filtered_docs:
                    embedding = embeddings.embed_query(doc.page_content)
                    new_vectors.append(np.array(embedding))

                # 2. Fetch all vectors and old centroid
                all_vectors = centroid_manager.get_all_vectors(client_instance, cfg.retrieval.COLLECTION_NAME)
                old_centroid = centroid_manager.get_centroid()
                
                # 3. Get thresholds from config
                threshold = getattr(cfg.ingestion, 'CENTROID_AUTO_THRESHOLD', 0.05)
                diversity_threshold = getattr(cfg.ingestion, 'CENTROID_DIVERSITY_THRESHOLD', 0.01)
                
                # 4. Decide and update
                if new_vectors:
                    if should_recalculate_centroid(new_vectors, all_vectors, old_centroid, threshold, diversity_threshold):
                        centroid_manager.update_centroid(all_vectors)
                        logger.info(f"Centroid recalculated and updated.")
                    else:
                        logger.info("Centroid update skipped (change not significant).")
        except Exception as e:
            logger.warning(f"Failed to update centroid: {e}")
        
        return quality_filtered_docs
        
           

    def _process_single_file(self, filepath, client_instance):
        """Process a single file with enhanced extraction."""
        
        counter_lock = threading.Lock()
        logger.info(f"Processing file: {filepath.name}")
        
        # Extract text content
        text = extract_text(filepath, client_instance=client_instance)
        if text is None or not text.strip():
            with counter_lock:
                self.processed_files[str(filepath)] = self.compute_hash(filepath)
            return None
        
        # Extract tables
        tables = extract_tables(filepath)
        
        try:
            file_path_obj = Path(filepath)
            stat = file_path_obj.stat()
            created_dt = dt.datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat().replace('+00:00', 'Z')
            modified_dt = dt.datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace('+00:00', 'Z')
            
            # Calculate quality score with tables and domain keywords
            quality_score = calculate_quality_score(
                text, 
                tables=tables,
                cached_domain_keywords=self.domain_keywords
            )
            
            metadata = {
                "source": str(filepath),
                "filetype": file_path_obj.suffix.lower(),
                "created_date": created_dt,
                "modified_date": modified_dt,
                "page": 0,
                "content_flags": [],
                "error_messages": [],
                "quality_score": quality_score,
                "has_tables": bool(tables),
                "table_count": len(tables) if tables else 0
            }
            
            # Add table data to metadata if present
            if tables:
                metadata["tables"] = tables
            
            doc = Document(page_content=text, metadata=metadata)
            
            with counter_lock:
                self.processed_files[str(filepath)] = self.compute_hash(filepath)
                
            return doc
            
        except Exception as e:
            with counter_lock:
                logger.error(f"Error processing {filepath.name}: {e}")
            return None

        
    # Enhance _calculate_optimal_batch_size in ingest_block.py:
    def _calculate_optimal_batch_size(self, files_to_process):
        """Calculate optimal batch size based on file types, sizes, and system resources"""
        if not files_to_process:
            return 10
            
        # Get system memory info
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        except ImportError:
            available_memory = 8  # Default assumption: 8GB
            
        # Analyze file sizes and types
        total_size = 0
        file_types = Counter()
        
        for filepath in files_to_process:
            file_types[filepath.suffix.lower()] += 1
            try:
                total_size += filepath.stat().st_size
            except OSError:
                pass
                
        # Adjust batch size based on average file size and available memory
        avg_size_mb = (total_size / len(files_to_process)) / (1024 * 1024) if files_to_process else 0
        pdf_ratio = file_types.get('.pdf', 0) / len(files_to_process) if files_to_process else 0
        
        # Memory-based calculation
        if available_memory < 2:  # Less than 2GB available
            base_size = 2
        elif available_memory < 4:  # 2-4GB available
            base_size = 4
        else:  # More than 4GB available
            base_size = 8
            
        # Adjust for file characteristics
        if avg_size_mb > 10 or pdf_ratio > 0.7:  # Large files or mostly PDFs
            return max(2, base_size // 2)
        elif avg_size_mb > 1 or pdf_ratio > 0.3:  # Medium files or some PDFs
            return max(4, base_size)
        else:  # Small files, few PDFs
            return max(6, base_size * 2)

        
    def _create_batches(self, files, batch_size):
        """Create batches of files for processing."""
        return [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

# --- Store Documents in Weaviate (Mostly Unchanged, Check Metadata) ---
def store_documents(docs: List[Document], client: weaviate.Client, embeddings_obj) -> None:
    """Stores documents in Weaviate using batch insertion."""
    if embeddings_obj is None:
        logger.error("store_documents: No embeddings object provided.")
        return

    if not docs:
        logger.info("store_documents: No documents to insert.")
        return

    collection_name = PipelineConfig.VECTORSTORE_COLLECTION
    try:
        collection = client.collections.get(collection_name)
    except Exception as e:
        logger.error(f"store_documents: Error fetching collection '{collection_name}': {e}", exc_info=True)
        return

    objects_to_insert: List[DataObject] = []
    logger.info(f"store_documents: Preparing {len(docs)} document chunks for insertion...")

    # --- Prepare objects with embeddings ---
    contents = [doc.page_content for doc in docs]
    embeddings = []
    if contents:
        try:
            # Embed in batches if embed_documents is efficient, otherwise loop embed_query
            if hasattr(embeddings_obj, 'embed_documents'):
                 logger.info(f"Embedding {len(contents)} chunks using embed_documents...")
                 embeddings = embeddings_obj.embed_documents(contents)
                 logger.info("Embeddings generated.")
            else:
                 logger.info(f"Embedding {len(contents)} chunks using embed_query loop...")
                 # Fallback to loop (potentially slower)
                 for i, content in enumerate(contents):
                     @retry(stop=stop_after_attempt(PipelineConfig.RETRY_ATTEMPTS),
                            wait=wait_exponential(multiplier=PipelineConfig.RETRY_WAIT_MULTIPLIER, max=PipelineConfig.RETRY_MAX_WAIT))
                     def get_single_embedding():
                         return embeddings_obj.embed_query(content)
                     embeddings.append(get_single_embedding())
                     if (i + 1) % 50 == 0: # Log progress
                          logger.info(f" Embedded {i+1}/{len(contents)} chunks...")
                 logger.info("Embeddings generated.")
        except Exception as embed_e:
             logger.error(f"Fatal error during embedding generation: {embed_e}", exc_info=True)
             return # Cannot proceed without embeddings

    if len(embeddings) != len(docs):
        logger.error(f"Mismatch between number of documents ({len(docs)}) and embeddings ({len(embeddings)}). Aborting insertion.")
        return

    # --- Create DataObject list ---
    for i, doc in enumerate(docs):
        # Ensure metadata keys required by Weaviate schema exist.
        # The keys were already prepared correctly in load_new_documents.
        properties = {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "filetype": doc.metadata.get("filetype", "unknown"),
            "page": doc.metadata.get("page", 0), # Splitter might add page info
            "created_date": doc.metadata.get("created_date"), # Already ISO string
            "modified_date": doc.metadata.get("modified_date"), # Already ISO string
            "content_flags": doc.metadata.get("content_flags", []), # Use defaults if missing
            "error_messages": doc.metadata.get("error_messages", [])
        }
        # Filter out None values before insertion if schema doesn't allow nulls
        properties_filtered = {k: v for k, v in properties.items() if v is not None}

        objects_to_insert.append(
            DataObject(
                properties=properties_filtered,
                vector=embeddings[i] # Get corresponding embedding
            )
        )

    if not objects_to_insert:
        logger.warning("store_documents: No valid objects prepared for insertion after embedding.")
        return

    # --- Perform Batch Insertion ---
    logger.info(f"store_documents: Attempting to insert {len(objects_to_insert)} objects...")
    try:
        # Use a context manager for batching if available and preferred
        # Or use insert_many directly
        response = collection.data.insert_many(objects=objects_to_insert)

        logger.info("store_documents: insert_many operation finished.")
        # Process response (check for errors)
        if response and hasattr(response, 'has_errors') and response.has_errors:
            error_count = 0
            for index, error in response.errors.items():
                logger.error(f"  Index {index}: {error.message}")
                error_count += 1
            logger.error(f"store_documents: Batch insertion completed with {error_count} errors.")
        elif response:
            # Log success count if possible from response details
            success_count = len(objects_to_insert) - (len(response.errors) if hasattr(response, 'errors') else 0)
            logger.info(f"store_documents: Successfully inserted {success_count} objects.")
        else:
             logger.warning("store_documents: insert_many response was None or unexpected.")

    except Exception as e: # Catch Weaviate errors and others
        logger.error(f"store_documents: Batch insertion failed: {e}", exc_info=True)
        # Consider adding more specific error handling if needed


# --- OPTIMIZED run_ingestion ---
def run_ingestion(folder: str) -> dict:
    """
    Optimized incremental ingestion: gets one client and passes it down.
    """
    client = None # Initialize client to None
    embeddings_obj = None # Initialize embeddings to None
    num_processed_chunks = 0 # Track processed CHUNKS

    try:
        # 1. Initialize the MAIN Weaviate client
        logger.info("run_ingestion: Attempting to get Weaviate client...")
        client = PipelineConfig.get_client()

        if client is None:
            logger.critical("run_ingestion: Failed to obtain Weaviate client.")
            return {"processed": 0, "message": "Incremental ingestion failed: Could not connect to Weaviate."}

        if not client.is_live():
            logger.critical("run_ingestion: Weaviate server is not responsive.")
            return {"processed": 0, "message": "Incremental ingestion failed: Weaviate server is not responsive."}
        logger.info("run_ingestion: Weaviate client is live.")

        # 2. Initialize embeddings
        logger.info("run_ingestion: Initializing embeddings...")
        try:
            embeddings_obj = OllamaEmbeddings(
                model=PipelineConfig.EMBEDDING_MODEL,
                base_url=PipelineConfig.EMBEDDING_BASE_URL
            )
            logger.info(f"run_ingestion: Embeddings initialized with model: {PipelineConfig.EMBEDDING_MODEL}")
        except Exception as embed_e:
            logger.error(f"run_ingestion: Error initializing embeddings: {embed_e}", exc_info=True)
            return {"processed": 0, "message": f"Incremental ingestion failed: Embeddings init error: {embed_e}"}

        # 3. Process documents incrementally, passing the main client
        logger.info(f"run_ingestion: Starting incremental processing for directory: {folder}")
        incremental_processor = IncrementalDocumentProcessorBlock(data_dir=folder)

        # --- PASS MAIN CLIENT to processor ---
        new_docs_chunks = incremental_processor.process(client_instance=client)
        num_processed_chunks = len(new_docs_chunks)

        # 4. Insert the resulting document chunks
        if new_docs_chunks:
            logger.info(f"run_ingestion: Storing {num_processed_chunks} new document chunks into Weaviate...")
            store_documents(new_docs_chunks, client, embeddings_obj)
        else:
            logger.info("run_ingestion: No new or modified documents found to process.")

        # 5. Log success
        logger.info(f"run_ingestion: Incremental ingestion completed: {num_processed_chunks} new document chunks processed.")

        # After storing documents successfully
        if new_docs_chunks and centroid_manager and hasattr(centroid_manager, 'update_centroid'):
            try:
                # Get embeddings for new documents
                vectors = []
                for doc in new_docs_chunks:
                    embedding = embeddings_obj.embed_query(doc.page_content)
                    vectors.append(embedding)
                
                # Update domain centroid with new vectors
                if vectors:
                    centroid_manager.update_centroid(vectors)
                    logger.info(f"run_ingestion: Updated domain centroid with {len(vectors)} new vectors")
            except Exception as centroid_err:
                logger.warning(f"run_ingestion: Failed to update centroid: {centroid_err}")


        return {
                    "processed": num_processed_chunks,
                    "message": "Incremental ingestion completed successfully."
        }

    except Exception as e:
        logger.critical(f"run_ingestion: Error during ingestion: {e}", exc_info=True)
        return {
            "processed": num_processed_chunks,  # Report chunks processed before error
            "message": f"Incremental ingestion failed: {str(e)}"
        }
    finally:
        # 6. Ensure the MAIN client connection is closed
        if client and hasattr(client, 'close') and callable(client.close):
            try:
                if client.is_connected():
                    client.close()
                    logger.info("run_ingestion: Closed Weaviate client connection.")
                # else: logger.info("run_ingestion: Weaviate client was already closed.") # Reduce noise
            except Exception as close_err:
                logger.error(f"run_ingestion: Error closing Weaviate client: {close_err}")
    # Close embeddings client
        if embeddings_obj and hasattr(embeddings_obj, 'client') and hasattr(embeddings_obj.client, 'close'):
            try:
                embeddings_obj.client.close()
                logger.info("run_ingestion: Closed embeddings client connection.")
            except Exception as close_err:
                logger.error(f"run_ingestion: Error closing embeddings client: {close_err}")



def after_ingestion(client, collection_name):
    """Update domain centroid after ingestion by averaging all document vectors."""
    logger.info(f"Updating domain centroid from collection '{collection_name}'...")
    
    # Use a lock to ensure thread safety
    with threading.Lock():
        try:
            vectors = []
            collection = client.collections.get(collection_name)
            
            # Log progress periodically
            processed = 0
            for obj in collection.iterator(include_vector=True, return_properties=[]):
                processed += 1
                if processed % 1000 == 0:
                    logger.info(f"Processed {processed} vectors...")
                
                if obj.vector and 'default' in obj.vector:
                    vectors.append(obj.vector['default'])
            
            if vectors:
                logger.info(f"Calculating centroid from {len(vectors)} vectors...")
                centroid = np.mean(np.array(vectors), axis=0)
                centroid_manager.save_centroid(centroid)
                logger.info(f"Domain centroid updated successfully with shape {centroid.shape}")
                return True
            else:
                logger.warning("No vectors found in collection. Centroid not updated.")
                return False
                
        except Exception as e:
            logger.error(f"Error updating centroid: {e}", exc_info=True)
            return False


def calculate_quality_score(text: str,cached_domain_keywords: set = None) -> float:
    """
    Calculate document quality score based on multiple factors:
    1. Information density (unique words / total words)
    2. Average sentence length (penalize very short or very long sentences)
    3. Keyword presence (domain-specific terms)
    4. Text structure (headings, paragraphs)
    
    Returns a score between 0.0 and 1.0
    """
    if not text or len(text) < 50:
        return 0.1  # Very short texts get low scores
    
    # Clean text
    clean_text = re.sub(r'\s+', ' ', text).strip()
    
    # 1. Information density
    words = re.findall(r'\b\w+\b', clean_text.lower())
    if not words:
        return 0.1
    
    unique_words = set(words)
    info_density = len(unique_words) / len(words)
    
    # 2. Sentence length analysis
    sentences = re.split(r'[.!?]+', clean_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.2
    
    avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
    # Penalize very short or very long sentences
    sentence_score = min(1.0, max(0.0, 1.0 - abs(avg_sentence_len - 15) / 20))
    
    # 3. Domain keywords presence
    if cached_domain_keywords:
        keyword_matches = sum(1 for word in unique_words if word.lower() in cached_domain_keywords)
        keyword_score = min(1.0, keyword_matches / 10)  # Cap at 10 keywords
    else:
        keyword_score = 0.5  # Neutral if no keywords defined
    
    # 4. Structure analysis
    has_headings = bool(re.search(r'\n[A-Z][^.!?]*\n', text))
    has_paragraphs = text.count('\n\n') > 0
    structure_score = 0.3 + (0.4 if has_paragraphs else 0) + (0.3 if has_headings else 0)
    
    # Calculate final score (weighted average)
    final_score = (
        info_density * 0.4 +
        sentence_score * 0.2 +
        keyword_score * 0.3 +
        structure_score * 0.1
    )
    
    return min(1.0, max(0.1, final_score))


def extract_text_from_txt(filepath: Path) -> str:
    try:
        import chardet
        with open(filepath, "rb") as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
        with open(filepath, "r", encoding=encoding) as f:
            content = f.read()
            logger.info(f"Successfully extracted {len(content)} characters from TXT file: {filepath.name}")
            return content
    except Exception as e:
        logger.error(f"Error reading TXT file {filepath.name}: {e}")
        return ""

# In ingest_block.py, enhance extract_text_from_docx:
def extract_text_from_docx(filepath: Path) -> str:
    if docx is None:
        logger.error(f"python-docx not installed. Cannot process DOCX: {filepath.name}")
        return ""
    
    try:
        doc = docx.Document(filepath)
        full_text = []
        
        # Extract main text with heading levels
        for para in doc.paragraphs:
            # Add heading level info if it's a heading
            if para.style.name.startswith('Heading'):
                level = para.style.name.replace('Heading ', '')
                full_text.append(f"{'#' * int(level)} {para.text}")
            else:
                full_text.append(para.text)
        
        # Extract tables with better formatting
        for table in doc.tables:
            table_text = []
            for i, row in enumerate(table.rows):
                cells = [cell.text for cell in row.cells]
                if i == 0:  # Header row
                    table_text.append("| " + " | ".join(cells) + " |")
                    table_text.append("| " + " | ".join(["---"] * len(cells)) + " |")
                else:
                    table_text.append("| " + " | ".join(cells) + " |")
            full_text.append("\n".join(table_text))
            
        return "\n\n".join(full_text)
    except Exception as e:
        logger.error(f"Error processing DOCX {filepath.name}: {e}")
        return ""


def extract_text_from_csv(filepath: Path) -> str:
    try:
        if pd is not None:
            df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
            # Convert to markdown table for better structure preservation
            return df.to_markdown(index=False)
        else:
            # Fallback to basic CSV reader
            with open(filepath, "r", encoding="utf-8", newline='') as f:
                reader = csv.reader(f)
                return "\n".join([", ".join(row) for row in reader])
    except Exception as e:
        logger.error(f"Error reading CSV file {filepath.name}: {e}")
        return ""

def extract_text_from_markdown(filepath: Path) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            # Preserve markdown structure
            return content
    except Exception as e:
        logger.error(f"Error reading Markdown file {filepath.name}: {e}")
        return ""
def extract_tables(filepath: Path) -> list:
    """Extract tables from various document formats."""
    ext = filepath.suffix.lower()
    tables = []
    
    try:
        if ext == '.pdf' and pdfplumber is not None:
            with pdfplumber.open(filepath) as pdf:
                for i, page in enumerate(pdf.pages):
                    for j, table in enumerate(page.extract_tables()):
                        if table:
                            tables.append({
                                "table_id": f"page_{i+1}_table_{j+1}",
                                "data": table,
                                "page": i+1
                            })
        elif ext == '.docx' and docx is not None:
            doc = docx.Document(filepath)
            for i, table in enumerate(doc.tables):
                data = []
                for row in table.rows:
                    data.append([cell.text for cell in row.cells])
                tables.append({
                    "table_id": f"table_{i+1}",
                    "data": data
                })
        elif ext == '.csv' and pd is not None:
            df = pd.read_csv(filepath)
            tables.append({
                "table_id": "csv_table",
                "data": df.values.tolist(),
                "headers": df.columns.tolist()
            })
        return tables
    except Exception as e:
        logger.error(f"Table extraction failed for {filepath.name}: {e}")
        return []
    
    
def calculate_quality_score(text: str, tables: list = None, cached_domain_keywords: set = None) -> float:
    """
    Calculate document quality score with enhanced metrics.
    """
    if not text or len(text) < 50:
        return 0.1
        
    # Clean text
    clean_text = re.sub(r'\s+', ' ', text).strip()
    
    # 1. Information density
    words = re.findall(r'\b\w+\b', clean_text.lower())
    if not words:
        return 0.1
    unique_words = set(words)
    info_density = len(unique_words) / len(words)
    
    # 2. Sentence length analysis
    sentences = re.split(r'[.!?]+', clean_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.2
    avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
    sentence_score = min(1.0, max(0.0, 1.0 - abs(avg_sentence_len - 15) / 20))
    
    # 3. Domain keywords presence
    keyword_score = 0.5
    if cached_domain_keywords and len(cached_domain_keywords) > 0:
        keyword_matches = sum(1 for word in unique_words if word.lower() in cached_domain_keywords)
        keyword_score = min(1.0, keyword_matches / 10)
    
    # 4. Structure analysis
    has_headings = bool(re.search(r'\n[A-Z][^.!?]*\n', text))
    has_paragraphs = text.count('\n\n') > 0
    structure_score = 0.3 + (0.4 if has_paragraphs else 0) + (0.3 if has_headings else 0)
    
    # 5. Table quality (new)
    table_score = 0.0
    if tables:
        table_score = min(1.0, len(tables) * 0.2)
    
    # 6. Domain relevance (if centroid available)
    domain_relevance = 0.0
    try:
        if centroid_manager and hasattr(centroid_manager, 'get_centroid'):
            domain_centroid = centroid_manager.get_centroid()
            if domain_centroid is not None and embeddings:
                text_vector = embeddings.embed_query(text)
                domain_relevance = cosine_similarity([text_vector], [domain_centroid])[0][0]
    except Exception as e:
        logger.warning(f"Error calculating domain relevance: {e}")
    
    # Calculate final score (weighted average)
    final_score = (
        info_density * 0.25 +
        sentence_score * 0.15 +
        keyword_score * 0.2 +
        structure_score * 0.1 +
        domain_relevance * 0.2 +
        table_score * 0.1
    )
    
    return min(1.0, max(0.1, final_score))


# --- Main Execution (for running script directly) ---
def main():
    parser = argparse.ArgumentParser(description="Run incremental document ingestion.")
    parser.add_argument("--folder", "-f", type=str, default=cfg.paths.DOCUMENT_DIR if cfg else "./data",
                        help="Folder containing documents to ingest.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    # Setup basic logging if run directly
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running ingest_block script directly...")

    run_ingestion(args.folder) # Call the main function

# Only call main() if this script is run directly.
if __name__ == "__main__":
    main()
