# ========== DOCUMENT INGESTION PIPELINE (WEAVIATE v4) ==========

# ========== CORE IMPORTS ==========
import argparse
import os
import re
import json
import datetime as dt
from datetime import timezone
import traceback
import hashlib
import logging
import warnings
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
import pdfplumber
import PyPDF2
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential


# Weaviate v4 imports
import weaviate

from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
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
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Local Configuration
from config import cfg

# ========== LOGGING CONFIGURATION ==========
logger = logging.getLogger(__name__)
weaviate_logger = logging.getLogger("weaviate")
weaviate_logger.setLevel(logging.WARNING)
weaviate_logger.propagate = False

# ========== PIPELINE CONFIGURATION ==========
class PipelineConfig:
    """Centralized pipeline configuration for Weaviate v4"""
    # Document processing
    MIN_CONTENT_LENGTH = 50
    VECTORSTORE_COLLECTION = "industrial_tech"
    # Weaviate connection - NOW USING VALUES FROM YOUR config.py
    WEAVIATE_HOST = cfg.retrieval.WEAVIATE_HOST
    WEAVIATE_HTTP_PORT = cfg.retrieval.WEAVIATE_HTTP_PORT
    WEAVIATE_GRPC_PORT = cfg.retrieval.WEAVIATE_GRPC_PORT
    WEAVIATE_TIMEOUT = (30, 300)
    # Embedding configuration
    EMBEDDING_MODEL = cfg.model.EMBEDDING_MODEL
    EMBEDDING_TIMEOUT = 30
    EMBEDDING_BASE_URL = "http://localhost:11434"
    # Document processing
    PDF_TABLE_EXTRACTION = cfg.document.PARSE_TABLES
    CHUNK_SIZE = cfg.document.CHUNK_SIZE
    CHUNK_OVERLAP = cfg.document.CHUNK_OVERLAP
    # Retry configuration
    RETRY_ATTEMPTS = 3
    RETRY_WAIT_MULTIPLIER = 2
    RETRY_MAX_WAIT = 30

    @classmethod
    def get_client(cls): # Signature remains correct (only cls)
        """Weaviate v4 client initialization using official connection method.
        Reads CURRENT connection details from the global cfg object."""
        from weaviate import WeaviateClient
        from weaviate.connect import ConnectionParams
        from weaviate.config import AdditionalConfig
        from config import cfg # Ensure global cfg is accessible

        # --- READ CURRENT CONFIG FROM GLOBAL cfg ---
        if not cfg or not cfg.retrieval:
            logger.error("get_client: Global cfg or cfg.retrieval not available!")
            # Handle this case - maybe raise error or return None?
            raise RuntimeError("Configuration (cfg) not loaded or incomplete.")

        # Use cfg directly, not cls attributes
        host = cfg.retrieval.WEAVIATE_HOST
        http_port = cfg.retrieval.WEAVIATE_HTTP_PORT
        grpc_port = cfg.retrieval.WEAVIATE_GRPC_PORT
        timeout = cfg.retrieval.WEAVIATE_TIMEOUT
        # --- END READING CURRENT CONFIG ---

        logger.info(f"Connecting to {host}:{http_port} (from current cfg)") # Log source
        client = None
        try:
            connection_params = ConnectionParams.from_params(
                http_host=host,         # Use variable from cfg
                http_port=http_port,    # Use variable from cfg
                http_secure=False,
                grpc_host=host,         # Use variable from cfg
                grpc_port=grpc_port,    # Use variable from cfg
                grpc_secure=False
            )

            client = WeaviateClient(
                connection_params=connection_params,
                additional_config=AdditionalConfig(timeout=timeout) # Use variable from cfg
            )

            client.connect() # Connect

            if not client.is_live():
                logger.error(f"Weaviate client connected to {host}:{http_port} but reports not live.")
                # Close client before raising if possible
                if client and hasattr(client, 'close') and callable(client.close): client.close()
                raise WeaviateConnectionError(f"Weaviate server at {host}:{http_port} connected but is not live.")

            logger.info(f"Weaviate client connected successfully to {host}:{http_port} and is live.")
            return client

        except Exception as e:
            logger.error(f"Connection failed during get_client to {host}:{http_port}: {e}", exc_info=True)
            if client and hasattr(client, 'close') and callable(client.close):
                try:
                    if client.is_connected(): client.close()
                    logger.info("Closed partially created client after connection failure.")
                except Exception as close_err:
                    logger.error(f"Error closing client during exception handling: {close_err}")
            return None # Return None on failure

# ========== CORE COMPONENTS ==========
class RobustPDFLoaderV4:
    """Weaviate v4 compatible PDF loader with table handling"""
    
    SCHEMA_PROPERTIES = {
        "source": DataType.TEXT,
        "filetype": DataType.TEXT,
        "created_date": DataType.DATE,
        "modified_date": DataType.DATE,
        "page": DataType.INT,
        "content_flags": DataType.TEXT_ARRAY,
        "error_messages": DataType.TEXT_ARRAY
    }

    def __init__(self, file_path: str, client: weaviate.Client):
        self.file_path = Path(file_path).resolve()
        self.client = client
        self._validate_file()

    def _validate_file(self) -> None:
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")
        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid file type: {self.file_path.suffix}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    def load(self) -> List[Document]:
        try:
            return self._load_with_tables()
        except Exception as e:
            logger.error(f"PDF Load Failure: {str(e)}")
            return self._load_fallback()

    def _load_with_tables(self) -> List[Document]:
        documents = []
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for idx, page in enumerate(pdf.pages, 1):
                    if doc := self._process_page(page, idx):
                        documents.append(doc)
        except Exception as e:
            logger.error(f"Table extraction failed: {str(e)}")
        return documents

    def _process_page(self, page, page_num: int) -> Optional[Document]:
        try:
            metadata = self._get_base_metadata(page_num)
            content = self._extract_page_content(page)
            return Document(
                page_content=content,
                metadata=self._clean_metadata(metadata)
            )
        except Exception as e:
            logger.error(f"Page {page_num} error: {str(e)}")
            return None

    def _get_base_metadata(self, page_num: int) -> Dict[str, Any]:
        stat = self.file_path.stat()
        try:
            created_dt_utc = dt.datetime.fromtimestamp(stat.st_birthtime, tz=timezone.utc)
            modified_dt_utc = dt.datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            created_date_str = created_dt_utc.isoformat().replace('+00:00', 'Z')
            modified_date_str = modified_dt_utc.isoformat().replace('+00:00', 'Z')
            
        except OSError as e: # Handle potential errors getting timestamps on some systems
            logger.warning(f"Could not get file timestamps for {self.file_path}: {e}. Using current time.")
            now_dt_utc = dt.datetime.now(timezone.utc)
            created_date_str = now_dt_utc.isoformat().replace('+00:00', 'Z')
            modified_date_str = now_dt_utc.isoformat().replace('+00:00', 'Z')

        return {
            "source": str(self.file_path),
            "filetype": "pdf",
            "created_date": created_date_str, # Now RFC3339 compliant
            "modified_date": modified_date_str, # Now RFC3339 compliant
            "page": page_num,
            "content_flags": [],
            "error_messages": []
        }

    def _clean_metadata(self, metadata: Dict) -> Dict:
        cleaned = {}
        # Define expected types for basic validation/conversion
        expected_types = {
            "source": str,
            "filetype": str,
            "page": int,
            "content_flags": list, # Expects a list of strings
            "error_messages": list, # Expects a list of strings
            "created_date": str, # Expects ISO 8601 string
            "modified_date": str # Expects ISO 8601 string
        }

        for key, expected_type in expected_types.items():
            value = metadata.get(key)
            if value is None:
                # Handle missing keys based on type - empty string/list or skip?
                # For Weaviate, often better to omit the property if value is truly missing.
                # Let's try omitting for now.
                continue # Skip if value is None

            try:
                if expected_type == str:
                    cleaned[key] = str(value)
                elif expected_type == int:
                    cleaned[key] = int(value)
                elif expected_type == list:
                    # Ensure it's a list of strings
                    if isinstance(value, list):
                        cleaned[key] = [str(item) for item in value]
                    else:
                        # Handle case where it's not a list - maybe wrap it?
                        cleaned[key] = [str(value)] # Or log error/skip
                # Date types are already ISO strings from _get_base_metadata
                elif key in ["created_date", "modified_date"]:
                    # Basic check if it looks like ISO format (optional)
                    cleaned[key] = str(value) # Already should be string

            except (ValueError, TypeError) as e:
                logger.warning(f"Metadata cleaning error for key '{key}', value '{value}': {e}. Skipping key.")
                metadata.setdefault("error_messages", []).append(f"Cleaning error for {key}: {e}")

        # Add back error messages if any occurred
        if "error_messages" in metadata and metadata["error_messages"]:
            cleaned["error_messages"] = metadata["error_messages"]

        return cleaned

    def _convert_to_iso(self, value) -> str:
        try:
            # Assuming input 'value' is already a datetime object or ISO string
            if isinstance(value, dt.datetime):
                dt_obj = value
            else:
                dt_obj = dt.datetime.fromisoformat(str(value))

            # Make it UTC if naive
            if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            else:
                # Convert to UTC if it has another timezone
                dt_obj = dt_obj.astimezone(timezone.utc)

            return dt_obj.isoformat().replace('+00:00', 'Z')
        except Exception: # Broad except to catch parsing errors etc.
            logger.warning(f"Could not convert '{value}' to ISO UTC format. Using current time.")
            return dt.datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    def _extract_page_content(self, page) -> str:
        content_parts = []
        page_text = None
        try:
            # Extract tables (Ensure table formatting returns string)
            tables = page.extract_tables()
            if tables:
                for table_data in tables:
                    if table_data:
                        formatted_table = self._format_table(table_data) # Assume this returns str or empty str
                        if isinstance(formatted_table, str) and formatted_table: # Check type and non-empty
                            content_parts.append(formatted_table)

            # Extract text
            page_text = page.extract_text(x_tolerance=1, y_tolerance=3)
            if isinstance(page_text, str) and page_text.strip(): # Check type and non-empty/whitespace
                content_parts.append(page_text.strip())

        except Exception as e:
            logger.error(f"Content extraction error during processing: {str(e)}")
            # Attempt to recover page_text if possible
            if isinstance(page_text, str) and page_text.strip():
                if page_text.strip() not in content_parts:
                    content_parts.append(page_text.strip())

        # Final filter for only non-empty strings before joining
        final_content_parts = [part for part in content_parts if isinstance(part, str) and part]

        if not final_content_parts:
            try: # Try accessing page number safely
                page_num = page.page_number
            except AttributeError:
                page_num = "unknown"
            logger.warning(f"No content extracted for page {page_num}")
            return "" # Return empty string

        try:
            return "\n---\n".join(final_content_parts)
        except TypeError as join_e: # Catch the specific error just in case
            logger.error(f"JOIN FAILED! Parts: {final_content_parts} | Error: {join_e}")
            # Attempt recovery by joining only valid strings again
            safe_parts = [part for part in final_content_parts if isinstance(part, str)]
            return "\n---\n".join(safe_parts)

    def _format_table(self, table_data) -> str:
        try:
            df = pd.DataFrame(table_data[1:], columns=table_data[0])
            return f"TABLE:\n{df.to_markdown(index=False)}"
        except Exception as e:
            return "TABLE:\n" + "\n".join("|".join(row) for row in table_data)

    def _load_fallback(self) -> List[Document]:
        try:
            with open(self.file_path, 'rb') as f:
                return [
                    Document(
                        page_content=page.extract_text() or "",
                        metadata=self._clean_metadata(
                            self._get_base_metadata(idx+1))
                    )
                    for idx, page in enumerate(PyPDF2.PdfReader(f).pages)
                ]
        except Exception as e:
            logger.error(f"Fallback failed: {str(e)}")
            return []

class DocumentProcessor:
    """Weaviate v4 compatible document processing pipeline"""
    
    def __init__(self, data_dir: str, client: weaviate.Client):
        self.data_dir = Path(data_dir).resolve()
        self.weaviate_client = client
        self.embeddings = OllamaEmbeddings(
            model=PipelineConfig.EMBEDDING_MODEL,
            base_url=PipelineConfig.EMBEDDING_BASE_URL
        )
        self._init_collection()

    def _init_collection(self):
        """Initialize Weaviate collection with v4 settings"""
        collection_name = PipelineConfig.VECTORSTORE_COLLECTION
        if not self.weaviate_client.collections.exists(collection_name):
            logger.info(f"Collection '{collection_name}' does not exist. Creating...")
            try:
                self.weaviate_client.collections.create(
                    name=collection_name,
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="source", data_type=DataType.TEXT),
                        Property(name="filetype", data_type=DataType.TEXT),
                        # Add other properties based on your SCHEMA_PROPERTIES if needed
                        Property(name="page", data_type=DataType.INT),
                        Property(name="created_date", data_type=DataType.DATE),
                        Property(name="modified_date", data_type=DataType.DATE),
                        Property(name="content_flags", data_type=DataType.TEXT_ARRAY),
                        Property(name="error_messages", data_type=DataType.TEXT_ARRAY),
                    ],
                    vectorizer_config=Configure.Vectorizer.none(), # Correct for external embeddings
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE, # Use the enum
                        quantizer=Configure.VectorIndex.Quantizer.pq() # Keep if desired
                    ),
                    inverted_index_config=Configure.inverted_index(
                        bm25_b=0.75,
                        bm25_k1=1.2
                    )
                )
                logger.info(f"Collection '{collection_name}' created successfully.")
            except Exception as e:
                logger.error(f"Failed to create collection '{collection_name}': {e}", exc_info=True)
                raise # Re-raise after logging
        else:
            logger.info(f"Collection '{collection_name}' already exists.")

    def execute(self):
        """Main processing pipeline"""
        documents = self._load_documents()
        split_docs = self._split_documents(documents)
        filtered = self._filter_documents(split_docs)
        self._store_results(filtered)

    def _load_documents(self) -> List[Document]:
        documents = []
        # PDF processing
        for pdf_path in self.data_dir.glob("**/*.pdf"):
            try:
                loader = RobustPDFLoaderV4(str(pdf_path), self.weaviate_client)
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"PDF load error {pdf_path}: {e}")
        
        # Add TXT and CSV processing here...
        
        return documents

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=PipelineConfig.CHUNK_SIZE,
            chunk_overlap=PipelineConfig.CHUNK_OVERLAP
        )
        return splitter.split_documents(documents)

    def _filter_documents(self, documents: List[Document]) -> List[Document]:
        return [doc for doc in documents if self._is_valid(doc)]

    def _is_valid(self, doc: Document) -> bool:
        return len(doc.page_content) >= PipelineConfig.MIN_CONTENT_LENGTH

    def _store_results(self, documents: List[Document]) -> None:
        """Weaviate v4 batch insertion with embeddings"""
        collection = self.weaviate_client.collections.get(
            PipelineConfig.VECTORSTORE_COLLECTION
        )

        objects = []
        logger.info(f"Preparing {len(documents)} documents for insertion...")
        for i, doc in enumerate(documents):
            try:
                # Add retry mechanism for embedding generation
                @retry(stop=stop_after_attempt(PipelineConfig.RETRY_ATTEMPTS), wait=wait_exponential(multiplier=PipelineConfig.RETRY_WAIT_MULTIPLIER, max=PipelineConfig.RETRY_MAX_WAIT))
                def get_embedding():
                    return self.embeddings.embed_query(doc.page_content)

                embedding = get_embedding()

                # Prepare properties, ensuring all required keys exist even if None initially
                properties = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "filetype": doc.metadata.get("filetype", "unknown"),
                    "page": doc.metadata.get("page", 0), # Default to 0 if missing
                    "created_date": doc.metadata.get("created_date"), # Pass as string or None
                    "modified_date": doc.metadata.get("modified_date") # Pass as string or None
                    # Add other properties if defined in schema
                }
                # Optionally filter out None properties if Weaviate handles that better
                # properties = {k: v for k, v in properties.items() if v is not None}

                objects.append(DataObject(
                    properties=properties,
                    vector=embedding
                ))
            except Exception as e:
                logger.error(f"Failed to prepare object for document {i} (Source: {doc.metadata.get('source', 'N/A')}): {e}", exc_info=True)
                # Decide whether to skip this doc or halt entirely

        if not objects:
            logger.warning("No valid objects prepared for insertion.")
            return

        logger.info(f"Attempting to insert {len(objects)} objects...")
        response = None # Initialize response to None
        try:
            response = collection.data.insert_many(
                objects=objects
            )
            logger.info(f"Insert_many operation finished. Checking response...")

            if response and response.has_errors: # Check if response is not None
                logger.error(f"Batch insertion completed with errors:")
                error_count = 0
                for index, error in response.errors.items():
                    logger.error(f"  Index {index}: {error.message}")
                    error_count += 1
                logger.error(f"Total errors in batch: {error_count}")
            elif response: # Check if response is not None
                logger.info(f"Successfully inserted objects. (Result details may vary)")
                # Log success count if available in response structure, e.g.
                # success_count = len(objects) - len(response.errors)
                # logger.info(f"Successfully inserted {success_count} objects.")

        except WeaviateBaseError as e: # Catches all Weaviate errors
            logger.error(f"Batch insertion API call failed: {str(e)}", exc_info=True)
            # response remains None if API call itself fails
            raise e



    def __del__(self):
        if self.weaviate_client.is_connected():
            self.weaviate_client.close()

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    try:
        # Initialize components
        parser = argparse.ArgumentParser()
        parser.add_argument("--folder", "-f", type=str)
        parser.add_argument("--verbose", "-v", action="store_true")
        args = parser.parse_args()
        client = PipelineConfig.get_client()
        
        # Validate connection
        if not client.is_live():
            raise WeaviateConnectionError("Weaviate server not responsive")
        
        # Process documents, passing the CORRECT v4 client
        processor = DocumentProcessor(
            data_dir=args.folder or cfg.paths.DOCUMENT_DIR,
            client=client  # Pass the v4 client created above
        )
        processor.execute()
        
        logger.info("Processing completed successfully")
        print("✅ Success!")
        
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}",exc_info=True)
        print(f"❌ Failure: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)
        
    finally:
        if 'client' in locals() and client.is_connected():
            client.close()