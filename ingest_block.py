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

# Setup logging
logger = logging.getLogger(__name__)


# --- Extraction Functions ---

def extract_text_from_txt(filepath: Path) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading TXT file {filepath.name}: {e}")
        return ""

def extract_text_from_docx(filepath: Path) -> str:
    if docx is None:
        logger.error(f"python-docx not installed. Cannot process DOCX: {filepath.name}")
        return ""
    try:
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs if para.text])
    except Exception as e:
        logger.error(f"Error reading DOCX file {filepath.name}: {e}")
        return ""

def extract_text_from_csv(filepath: Path) -> str:
    # Consider using pandas if available for more robust CSV handling
    if pd:
        try:
            df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
            # Represent as string - maybe join rows? Or just header and first few rows?
            # Simple approach: Convert to string representation
            return df.to_string(index=False)
        except Exception as e:
            logger.error(f"Error reading CSV file {filepath.name} with pandas: {e}")
            # Fallback to basic CSV reader if pandas fails
    # Basic CSV reader (fallback)
    try:
        with open(filepath, "r", encoding="utf-8", newline='') as f:
            reader = csv.reader(f)
            # Join rows, limit lines maybe?
            return "\n".join([", ".join(row) for row in reader])
    except Exception as e_basic:
        logger.error(f"Error reading CSV file {filepath.name} with basic reader: {e_basic}")
        return ""

# --- OPTIMIZED extract_text ---
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

        self.meta_file = self.data_dir.parent / meta_filename # Store metafile outside data dir
        self.processed_files = self.load_metadata()

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

    # --- OPTIMIZED load_new_documents ---
    # Accepts the main client instance
    def load_new_documents(self, client_instance: weaviate.Client) -> List[Document]:
        """Iterate over files, process new/updated ones, pass client to extract_text."""
        docs = []
        logger.info(f"Scanning directory: {self.data_dir}")
        processed_count = 0
        skipped_unchanged_count = 0
        skipped_unsupported_count = 0
        skipped_no_text_count = 0
        error_count = 0

        # Use rglob to include subdirectories if needed, otherwise keep glob
        # file_iterator = self.data_dir.rglob("*") if include_subdirs else self.data_dir.glob("*")
        file_iterator = self.data_dir.glob("*") # Current behavior

        for filepath in file_iterator:
            if not filepath.is_file():
                continue

            processed_count += 1

            # Check file type support
            supported_extensions = set(f".{ext.lower()}" for ext in getattr(PipelineConfig, 'FILE_TYPES', ['pdf', 'txt', 'csv', 'docx', 'md']))
            if filepath.suffix.lower() not in supported_extensions:
                skipped_unsupported_count += 1
                logger.debug(f"Skipping unsupported file type: {filepath.name}")
                continue

            current_hash = self.compute_hash(filepath)
            if not current_hash: # Skip if hashing failed
                error_count += 1
                continue

            stored_hash = self.processed_files.get(str(filepath))

            if stored_hash == current_hash:
                skipped_unchanged_count += 1
                # logger.debug(f"Skipping file (unchanged): {filepath.name}") # Reduce noise
                continue

            logger.info(f"Processing new/updated file: {filepath.name}")
            # --- PASS THE CLIENT ---
            text = extract_text(filepath, client_instance=client_instance)

            # Check if text is None OR empty/whitespace after stripping
            if text is None or not text.strip():
                logger.warning(f"No text extracted or text is empty/whitespace for {filepath.name}, skipping.")
                # Update hash to prevent retrying problematic files
                self.processed_files[str(filepath)] = current_hash
                skipped_no_text_count += 1
                continue

            # --- Document Creation ---
            try:
                file_path_obj = Path(filepath)
                stat = file_path_obj.stat()
                # Use timezone-aware datetime objects
                created_dt = dt.datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat().replace('+00:00', 'Z')
                modified_dt = dt.datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace('+00:00', 'Z')

                metadata = {
                    "source": str(filepath),
                    "filetype": file_path_obj.suffix.lower(),
                    "created_date": created_dt,
                    "modified_date": modified_dt,
                    "page": 0, # Default for whole-doc text
                    "content_flags": [],
                    "error_messages": []
                }

                doc = Document(page_content=text, metadata=metadata)
                docs.append(doc)

                # Update hash only after successful doc creation
                self.processed_files[str(filepath)] = current_hash

            except Exception as doc_create_e:
                logger.error(f"Error creating Document object for {filepath.name}: {doc_create_e}", exc_info=True)
                error_count += 1
                # Do NOT update hash, allow retry later
                continue
            # --- End Document Creation ---

        logger.info(
            f"Directory scan complete. Total files considered: {processed_count}. "
            f"New/Updated with text: {len(docs)}. "
            f"Skipped (unchanged): {skipped_unchanged_count}. "
            f"Skipped (unsupported): {skipped_unsupported_count}. "
            f"Skipped (no text): {skipped_no_text_count}. "
            f"Errors: {error_count}."
        )
        return docs

    # --- OPTIMIZED process ---
    # Accepts the main client instance
    def process(self, client_instance: weaviate.Client) -> List[Document]:
        """Load new documents using the client, split, filter, update metadata."""
        # --- PASS THE CLIENT ---
        docs = self.load_new_documents(client_instance=client_instance)
        if not docs:
            logger.info("No new documents loaded.")
            self.save_metadata() # Save metadata even if no docs, to record skips
            return []

        logger.info(f"Splitting {len(docs)} new documents...")
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=PipelineConfig.CHUNK_SIZE,
                chunk_overlap=PipelineConfig.CHUNK_OVERLAP
                # Add other params if needed (e.g., separators)
            )
            split_docs = splitter.split_documents(docs)
        except Exception as split_e:
             logger.error(f"Error during document splitting: {split_e}", exc_info=True)
             self.save_metadata() # Save hash updates even if splitting fails
             return [] # Return empty list if splitting fails

        valid_docs = [d for d in split_docs if len(d.page_content) >= PipelineConfig.MIN_CONTENT_LENGTH]
        logger.info(f"Prepared {len(valid_docs)} valid document chunks for insertion (Min length: {PipelineConfig.MIN_CONTENT_LENGTH}).")
        self.save_metadata() # Save updated hashes after successful processing/splitting
        return valid_docs


# --- Store Documents in Weaviate (Mostly Unchanged, Check Metadata) ---
def store_documents(docs: List[Document], client: weaviate.Client, embeddings_obj) -> None:
    """Stores documents in Weaviate using batch insertion."""
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
        return {
            "processed": num_processed_chunks,
            "message": "Incremental ingestion completed successfully."
        }

    except Exception as e:
        logger.critical(f"run_ingestion: Error during ingestion: {e}", exc_info=True)
        return {
            "processed": num_processed_chunks, # Report chunks processed before error
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
