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
from typing import List, Dict, Optional, Tuple, Union

# For PDF processing, use pdfplumber (as in ingest_docs_v7.py)
import pdfplumber
import PyPDF2
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

# LangChain components for Document and splitting
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import configuration from your project and pipeline settings from ingest_docs_v7.py
from config import cfg
from ingest_docs_v7 import PipelineConfig, RobustPDFLoaderV4



############

import hashlib


import sys

from typing import Dict, List, Optional, Any, Generator


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

# LangChain Components
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_ollama import OllamaEmbeddings







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
    def __init__(self, data_dir: str, meta_filename="ingested_files.json"):
        self.data_dir = Path(data_dir).resolve()
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
            chunk_size=PipelineConfig.CHUNK_SIZE,
            chunk_overlap=PipelineConfig.CHUNK_OVERLAP
        )
        split_docs = splitter.split_documents(docs)
        valid_docs = [d for d in split_docs if len(d.page_content) >= PipelineConfig.MIN_CONTENT_LENGTH]
        logger.info("Prepared %d valid document chunks for insertion.", len(valid_docs))
        self.save_metadata()
        return valid_docs

# --- Store Documents in Weaviate ---
def store_documents(docs: List[Document], client: weaviate.Client, embeddings_obj) -> None:
    if not docs:
        logger.info("No documents to insert.")
        return
    collection_name = PipelineConfig.VECTORSTORE_COLLECTION
    try:
        collection = client.collections.get(collection_name)
    except Exception as e:
        logger.error("Error fetching collection '%s': %s", collection_name, e)
        return
    objects = []
    logger.info("Preparing %d document chunks for insertion.", len(docs))
    for i, doc in enumerate(docs):
        try:
            @retry(stop=stop_after_attempt(PipelineConfig.RETRY_ATTEMPTS),
                   wait=wait_exponential(multiplier=PipelineConfig.RETRY_WAIT_MULTIPLIER,
                                         max=PipelineConfig.RETRY_MAX_WAIT))
            def get_embedding():
                return embeddings_obj.embed_query(doc.page_content)
            embedding = get_embedding()
            properties = {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "filetype": doc.metadata.get("filetype", "unknown"),
                "page": 0,  # Not applicable here
                "created_date": None,
                "modified_date": None
            }
            objects.append(
                DataObject(
                    properties=properties,
                    vector=embedding
                )
            )
        except Exception as e:
            logger.error("Error preparing document %d (source: %s): %s", i, doc.metadata.get("source", "N/A"), e)
    if not objects:
        logger.warning("No objects prepared for insertion.")
        return
    try:
        response = collection.data.insert_many(objects=objects)
        if response and response.has_errors:
            logger.error("Insertion completed with errors: %s", response.errors)
        else:
            logger.info("Document chunks inserted successfully.")
    except Exception as e:
        logger.error("Batch insertion failed: %s", e)

def run_ingestion(folder: str) -> dict:
    """
    Run the incremental ingestion process on the specified folder.
    Returns a dictionary with stats and a success message.
    """
    try:
        # Initialize the Weaviate client using PipelineConfig
        client = PipelineConfig.get_client()
        if not client.is_live():
            raise Exception("Weaviate server is not responsive.")
    except Exception as e:
        logger.critical("Error connecting to Weaviate: %s", e)
        raise

    # Initialize embeddings via OllamaEmbeddings
    try:
        from langchain_ollama import OllamaEmbeddings
        embeddings_obj = OllamaEmbeddings(
            model=PipelineConfig.EMBEDDING_MODEL,
            base_url=PipelineConfig.EMBEDDING_BASE_URL
        )
    except Exception as e:
        logger.error("Error initializing embeddings: %s", e)
        raise

    # Process new/modified documents incrementally
    incremental_processor = IncrementalDocumentProcessorBlock(data_dir=folder)
    new_docs = incremental_processor.process()

    # Insert the resulting document chunks into Weaviate
    if embeddings_obj:
        store_documents(new_docs, client, embeddings_obj)
    else:
        logger.error("Embeddings object is not available; cannot compute embeddings.")
        raise Exception("Embeddings object is not available.")

    num_processed = len(new_docs)
    logger.info("Incremental ingestion completed: %d new document chunks processed.", num_processed)
    client.close()
    return {"processed": num_processed, "message": "Incremental ingestion completed successfully."}

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, default=cfg.paths.DOCUMENT_DIR,
                        help="Folder containing documents to ingest.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    # Initialize the Weaviate client using PipelineConfig
    try:
        client = PipelineConfig.get_client()
        if not client.is_live():
            raise Exception("Weaviate server is not responsive.")
    except Exception as e:
        logger.critical("Error connecting to Weaviate: %s", e)
        return
    # Initialize embeddings via OllamaEmbeddings
    try:
        from langchain_ollama import OllamaEmbeddings
        embeddings_obj = OllamaEmbeddings(
            model=PipelineConfig.EMBEDDING_MODEL,
            base_url=PipelineConfig.EMBEDDING_BASE_URL
        )
    except Exception as e:
        logger.error("Error initializing embeddings: %s", e)
        embeddings_obj = None
    # Process new/modified documents incrementally
    incremental_processor = IncrementalDocumentProcessorBlock(data_dir=args.folder)
    new_docs = incremental_processor.process()
    # Insert the resulting document chunks into Weaviate
    if embeddings_obj:
        store_documents(new_docs, client, embeddings_obj)
    else:
        logger.error("Embeddings object is not available; cannot compute embeddings.")
    logger.info("Incremental ingestion completed.")
    client.close()


# Only call main() if this script is run directly.
if __name__ == "__main__":
    main()


