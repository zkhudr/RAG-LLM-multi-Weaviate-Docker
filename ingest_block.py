#!/usr/bin/env python
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

# For PDF processing, use pdfplumber (as in ingest_docs_v7.py)
import pdfplumber
import PyPDF2
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

# LangChain components for Document and splitting
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_ollama import OllamaEmbeddings

# Import configuration from your project and pipeline settings from ingest_docs_v7.py
from config import cfg, AppConfig
#from ingest_docs_v7 import PipelineConfig, RobustPDFLoaderV4  #not need anymore for the updated model
from ingest_docs_v7 import RobustPDFLoaderV4 
############
import hashlib
import sys
# Weaviate v4 imports
import weaviate
from weaviate.classes.config import Property, DataType, Configure,VectorDistances
from weaviate.classes.data import DataObject
from weaviate.exceptions import (
    WeaviateConnectionError,
    WeaviateInvalidInputError,
    WeaviateBatchError,
    WeaviateBaseError
)

# Try to import DOCX support
try:
    import docx
except ImportError:
    docx = None
    logging.warning("python-docx is not installed; DOCX files will not be processed.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Extraction Functions for Various File Types ---

def extract_text_from_txt(filepath: Path) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error("Error reading TXT file %s: %s", filepath.name, e)
        return ""

def extract_text_from_docx(filepath: Path) -> str:
    if docx is None:
        logger.error("python-docx is not installed. Cannot process DOCX file: %s", filepath.name)
        return ""
    try:
        doc = docx.Document(filepath)
        paragraphs = [para.text for para in doc.paragraphs]
        return "\n".join(paragraphs)
    except Exception as e:
        logger.error("Error reading DOCX file %s: %s", filepath.name, e)
        return ""

def extract_text_from_csv(filepath: Path) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            # Join each row with commas and rows with newlines.
            return "\n".join([", ".join(row) for row in rows])
    except Exception as e:
        logger.error("Error reading CSV file %s: %s", filepath.name, e)
        return ""

def extract_text(filepath: Path) -> str:
    ext = filepath.suffix.lower()
    if ext == ".txt":
        return extract_text_from_txt(filepath)
    elif ext == ".docx":
        return extract_text_from_docx(filepath)
    elif ext == ".csv":
        return extract_text_from_csv(filepath)
    elif ext == ".pdf":
        # Use RobustPDFLoaderV4 for PDF extraction (with table extraction as defined in ingest_docs_v7.py)
        try:
            # Note: RobustPDFLoaderV4 requires a Weaviate client but for extraction we can pass None.
            loader = RobustPDFLoaderV4(str(filepath), client=None)
            docs = loader.load()
            # Combine all pages into one text block
            return "\n".join(doc.page_content for doc in docs if doc.page_content)
        except Exception as e:
            logger.error("Error processing PDF file %s: %s", filepath.name, e)
            return ""
    else:
        logger.warning("Unsupported file type: %s", filepath.name)
        return ""

def chunk_text(text: str, max_chunk_size: int) -> list:
    """Simple splitting of text into chunks up to max_chunk_size characters.
    For production, you might use a sentence or paragraph splitter instead."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks

# --- Incremental Ingestion Class ---
class IncrementalDocumentProcessorBlock:
    def __init__(self, data_dir: str, app_cfg: AppConfig, meta_filename="ingested_files.json"):
        self.data_dir = Path(data_dir).resolve()
        self.app_cfg = app_cfg 
        self.meta_file = self.data_dir / meta_filename
        self.processed_files = self.load_metadata()
    
    def load_metadata(self) -> dict:
        """Load a JSON file mapping file paths to their last processed hash."""
        if self.meta_file.exists():
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Error loading metadata: %s", e)
                return {}
        return {}
    
    def save_metadata(self) -> None:
        """Save the updated metadata dictionary."""
        try:
            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump(self.processed_files, f, indent=2)
        except Exception as e:
            logger.error("Error saving metadata: %s", e)
    
    def compute_hash(self, filepath: Path) -> str:
        """Compute the SHA-256 hash of a file’s contents."""
        hasher = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                while True:
                    buf = f.read(65536)
                    if not buf:
                        break
                    hasher.update(buf)
            return hasher.hexdigest()
        except Exception as e:
            logger.error("Error computing hash for %s: %s", filepath.name, e)
            return ""
    
    def load_new_documents(self) -> List[Document]:
        """Iterate over files in the data directory and return Document objects
        for new or updated files only."""
        docs = []
        for filepath in self.data_dir.glob("*"):
            if not filepath.is_file():
                continue
            current_hash = self.compute_hash(filepath)
            stored_hash = self.processed_files.get(str(filepath))
            if stored_hash == current_hash:
                logger.info("Skipping file (unchanged): %s", filepath.name)
                continue
            logger.info("Processing new/updated file: %s", filepath.name)
            text = extract_text(filepath)
            if not text.strip():
                logger.warning("No text extracted from %s, skipping.", filepath.name)
                continue
            # Create a Document object; we use basic metadata
            doc = Document(
                page_content=text,
                metadata={
                    "source": str(filepath),
                    "filetype": filepath.suffix.lower()
                }
            )
            docs.append(doc)
            # Update the metadata record for this file
            self.processed_files[str(filepath)] = current_hash
        return docs
    
    def process(self) -> List[Document]:
        """Load new documents, split them into chunks, filter out short chunks, and update metadata."""
        docs = self.load_new_documents()
        if not docs:
            logger.info("No new documents to process.")
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.app_cfg.document.CHUNK_SIZE,
            chunk_overlap=self.app_cfg.document.CHUNK_OVERLAP
        )
        split_docs = splitter.split_documents(docs)
        min_length = getattr(self.app_cfg.document, 'MIN_CONTENT_LENGTH', 50) # Example: Read from cfg if defined, else default 50
        valid_docs = [d for d in split_docs if len(d.page_content) >= min_length]
        logger.info("Prepared %d valid document chunks for insertion.", len(valid_docs))
        self.save_metadata()
        return valid_docs

# --- Store Documents in Weaviate --- now uses app_cfg 
def store_documents(docs: List[Document], client: weaviate.Client, embeddings_obj) -> None:
    if not docs:
        logger.info("No documents to insert.")
        return
    collection_name = app_cfg.retrieval.COLLECTION_NAME
    try:
        collection = client.collections.get(collection_name)
    except Exception as e:
          logger.error("Error fetching collection '%s': %s", collection_name, e)
          return {"inserted": 0, "errors": 0, "message": f"Failed to get collection {collection_name}"}
    objects = []
    logger.info("Preparing %d document chunks for insertion into '%s'.", len(docs), collection_name)
    retry_attempts = getattr(app_cfg.security, 'RETRY_ATTEMPTS', 3) # Example
    retry_wait_multiplier = getattr(app_cfg.security, 'RETRY_WAIT_MULTIPLIER', 2) # Example
    retry_max_wait = getattr(app_cfg.security, 'RETRY_MAX_WAIT', 30) # Example
    prepare_errors = 0

    for i, doc in enumerate(docs):
        try:
            @retry(stop=stop_after_attempt(retry_attempts),
                   wait=wait_exponential(multiplier=retry_wait_multiplier, max=retry_max_wait))
            def get_embedding():
                return embeddings_obj.embed_query(doc.page_content)

            embedding = get_embedding()

            properties = {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "filetype": doc.metadata.get("filetype", "unknown"),
                "page": doc.metadata.get("page", 0), # Consistent default
                "created_date": doc.metadata.get("created_date"),
                "modified_date": doc.metadata.get("modified_date")
            }
            # Ensure metadata matches collection schema

            objects.append(
                DataObject(
                    properties=properties,
                    vector=embedding
                )
            )
        except Exception as e:
            logger.error("Error preparing document %d (source: %s): %s", i, doc.metadata.get("source", "N/A"), e)
            prepare_errors += 1

    if not objects:
        logger.warning("No objects prepared for insertion.")
        return {"inserted": 0, "errors": prepare_errors, "message": "No objects prepared."}

    inserted_count = 0
    batch_errors = 0
    error_messages = []
    try:
        response = collection.data.insert_many(objects=objects)
        if response and response.has_errors:
            logger.error("Insertion completed with errors:")
            for index, error in response.errors.items():
                 logger.error(f"  Index {index}: {error.message}")
                 batch_errors += 1
                 error_messages.append(f"Index {index}: {error.message}")
            inserted_count = len(objects) - batch_errors
        elif response:
             inserted_count = len(objects)
             logger.info(f"{inserted_count} document chunks inserted successfully.")
        else:
            logger.warning("insert_many returned None or empty response.")

    except Exception as e:
        logger.error("Batch insertion failed: %s", e)
        batch_errors = len(objects) # Assume all failed if API call fails
        error_messages.append(f"Batch API Error: {str(e)}")

    total_errors = prepare_errors + batch_errors
    message = f"Insertion finished. Inserted: {inserted_count}, Errors: {total_errors}."
    if error_messages:
        message += " | Errors: " + "; ".join(error_messages[:3]) # Show first few errors

    return {"inserted": inserted_count, "errors": total_errors, "message": message}


# --- MODIFIED: run_ingestion uses app_cfg ---
def run_ingestion(folder: str, app_cfg: AppConfig) -> dict:
    """
    Run the incremental ingestion process on the specified folder using the provided config.
    Returns a dictionary with stats and a success message.
    """
    client = None
    try:
        # Initialize the Weaviate client using connection details from app_cfg
        # Use the static method from ingest_docs_v7 if preferred, or inline here
        host = app_cfg.retrieval.WEAVIATE_HOST
        http_port = app_cfg.retrieval.WEAVIATE_HTTP_PORT
        grpc_port = app_cfg.retrieval.WEAVIATE_GRPC_PORT
        logger.info(f"Incremental Ingest: Connecting to {host}:{http_port}...")
        client = weaviate.connect_to_local(host=host, port=http_port, grpc_port=grpc_port)
        if not client.is_ready():
            raise Exception(f"Weaviate server not responsive at {host}:{http_port}")

    except Exception as e:
        logger.critical("Error connecting to Weaviate for incremental ingest: %s", e)
        raise # Re-raise connection error

    try:
        # Initialize embeddings via OllamaEmbeddings using app_cfg
        embeddings_obj = OllamaEmbeddings(
            model=app_cfg.model.EMBEDDING_MODEL,
            base_url="http://localhost:11434" # Keep base URL or get from app_cfg if added
        )
    except Exception as e:
        logger.error("Error initializing embeddings for incremental ingest: %s", e)
        if client: client.close()
        raise # Re-raise embedding error

    try:
        # Process new/modified documents incrementally, passing app_cfg
        incremental_processor = IncrementalDocumentProcessorBlock(data_dir=folder, app_cfg=app_cfg)
        new_docs = incremental_processor.process()

        # Insert the resulting document chunks into Weaviate
        stats = store_documents(new_docs, client, embeddings_obj, app_cfg)

        num_processed = len(new_docs) # Number of docs loaded and split
        num_inserted = stats.get("inserted", 0)
        num_errors = stats.get("errors", 0)
        final_message = stats.get("message", "Incremental ingestion process finished.")

        logger.info(f"Incremental ingestion completed: {num_processed} chunks prepared, {num_inserted} inserted, {num_errors} errors.")

        return {
            "processed_chunks": num_processed,
            "inserted": num_inserted,
            "errors": num_errors,
            "message": final_message
        }
    except Exception as proc_e:
         logger.error(f"Error during incremental processing/storage: {proc_e}", exc_info=True)
         raise # Re-raise processing error
    finally:
        if client:
            client.close()
            logger.info(f"Incremental Ingest: Closed Weaviate client for {host}:{http_port}.")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, default=None,
                        help="Folder containing documents to ingest (overrides config).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Load main config
    try:
        from config import cfg, AppConfig
        if not cfg: raise RuntimeError("Failed to load main configuration (cfg).")
    except Exception as cfg_err:
        logger.critical(f"Could not load configuration: {cfg_err}", exc_info=True)
        return

    # Determine folder
    folder_to_ingest = args.folder if args.folder else cfg.paths.DOCUMENT_DIR

    try:
        # Run ingestion, passing the main cfg object
        result = run_ingestion(folder_to_ingest, cfg)
        logger.info(f"Incremental ingestion script finished. Result: {result}")
        print(f"✅ Success: {result.get('message')}")

    except Exception as e:
        logger.error("Incremental ingestion script failed: %s", str(e), exc_info=True)
        print(f"❌ Failure: {str(e)}")
    # --- END MODIFIED ---

# Only call main() if this script is run directly.
if __name__ == "__main__":
    main()