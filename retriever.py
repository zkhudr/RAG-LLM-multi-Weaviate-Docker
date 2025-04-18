# retriever.py (Modified for Weaviate with Flexible Parameters)

# --- Weaviate Imports ---
import weaviate
from weaviate.exceptions import WeaviateConnectionError
# Corrected Import based on previous error
from langchain_weaviate.vectorstores import WeaviateVectorStore
# -----------------------

# --- Original Imports ---
# Corrected Ollama Import (Address Deprecation Warning)
from langchain_ollama import OllamaEmbeddings
# --- Was: from langchain_community.embeddings import OllamaEmbeddings ---
from config import cfg  # Keep using central config
import logging
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
# -----------------------

# Configure logging
logger = logging.getLogger(__name__)

# Now moved to config.py
#DEFAULT_WEAVIATE_HOST = "localhost"
#DEFAULT_WEAVIATE_HTTP_PORT = 8080
#DEFAULT_WEAVIATE_GRPC_PORT = 50051
#DEFAULT_COLLECTION_NAME = "industrial_tech"

class TechnicalRetriever:

    def __init__(self):
        self.cfg = cfg
        self.logger = logger
        self.logger.info("Initializing TechnicalRetriever for Weaviate...")

        try:
            # Using corrected import
            self.embeddings = OllamaEmbeddings(
                model=self.cfg.model.EMBEDDING_MODEL,
                base_url="http://localhost:11434"
            )
            self.logger.info(f"OllamaEmbeddings initialized with model: {self.cfg.model.EMBEDDING_MODEL}")
        except Exception as e:
            self.logger.error(f"Failed to initialize OllamaEmbeddings: {str(e)}", exc_info=True)
            raise

        self.weaviate_client: weaviate.Client = None
        # Corrected type hint
        self.vectorstore: WeaviateVectorStore = None
        self._init_vectorstore()

    def _init_vectorstore(self):
        """Initializes the Weaviate client and Langchain vector store wrapper."""
        try:
            # --- MODIFIED: Get connection details from self.cfg ---
            host = self.cfg.retrieval.WEAVIATE_HOST
            http_port = self.cfg.retrieval.WEAVIATE_HTTP_PORT
            grpc_port = self.cfg.retrieval.WEAVIATE_GRPC_PORT
            collection_name = self.cfg.retrieval.COLLECTION_NAME # Get collection name from cfg

            self.logger.info(f"Attempting to connect to Weaviate at {host}:{http_port} (gRPC: {grpc_port})...")

            # Close existing client if it exists and is connected
            if self.weaviate_client and self.weaviate_client.is_connected():
                self.logger.warning("Existing Weaviate client found. Closing before reconnecting.")
                try:
                    self.weaviate_client.close()
                except Exception as close_err:
                    self.logger.error(f"Error closing existing client: {close_err}")

            # Connect using details from cfg
            self.weaviate_client = weaviate.connect_to_local(
                host=host,
                port=http_port,
                grpc_port=grpc_port
            )
            if not self.weaviate_client.is_connected():
                raise WeaviateConnectionError(f"Weaviate client failed to connect to {host}:{http_port}.")

            self.logger.info("Weaviate client connected successfully.")

            self.logger.info(f"Initializing Weaviate vector store wrapper for collection: '{collection_name}'")

            # Corrected Instantiation using collection_name from cfg
            self.vectorstore = WeaviateVectorStore(
                client=self.weaviate_client,
                index_name=collection_name,
                text_key="content",
                embedding=self.embeddings,
                attributes=["source", "page", "filetype", "created_date", "modified_date"] # Adjust if schema differs
            )
            self.logger.info("Weaviate vector store wrapper initialized.")

            # Optional: Check count
            try:
                count = self.vectorstore._collection.aggregate.over_all(total_count=True).total_count
                if count == 0:
                    self.logger.warning(f"Weaviate collection '{collection_name}' at {host}:{http_port} appears to be empty.")
                else:
                     self.logger.info(f"Collection '{collection_name}' has {count} items.")
            except Exception as count_err:
                 self.logger.error(f"Could not get item count for collection '{collection_name}': {count_err}")

        except Exception as e:
            self.logger.error(f"Weaviate vector store initialization failed: {str(e)}", exc_info=True)
            # Set client and vectorstore to None on failure to prevent usage
            self.weaviate_client = None
            self.vectorstore = None
            raise RuntimeError("Failed to initialize Weaviate connection") from e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_context(self, query: str) -> str:
        """Retrieves context from Weaviate based on the query and config settings."""
        if not self.vectorstore:
             self.logger.error("Vectorstore not initialized. Cannot retrieve context.")
             return ""
        try:
            search_type = self.cfg.retrieval.SEARCH_TYPE.lower() # Use lower case for reliable comparison
            k_value = self.cfg.retrieval.K_VALUE
            score_threshold = self.cfg.retrieval.SCORE_THRESHOLD
            lambda_mult = self.cfg.retrieval.LAMBDA_MULT

            self.logger.info(f"Retrieving context for query snippet: '{query[:50]}...'")
            self.logger.info(f"Using search_type='{search_type}', k={k_value}, score_threshold={score_threshold}")

            # --- *** MODIFIED: Flexible Parameter Mapping *** ---
            search_kwargs = {
                "k": k_value,
            }

            if search_type == 'mmr':
                search_kwargs["alpha"] = lambda_mult
                self.logger.info(f"MMR alpha (diversity): {search_kwargs['alpha']}")
                # score_threshold is NOT added for MMR as it's incompatible with Weaviate's hybrid query

            elif search_type == 'similarity_score_threshold':
                search_kwargs["score_threshold"] = score_threshold
                self.logger.info(f"Similarity score threshold: {search_kwargs['score_threshold']}")


            elif search_type == 'similarity':
                # Basic similarity search just uses 'k'
                self.logger.info("Using basic similarity search.")
                # Do not add score_threshold unless specifically intended and tested for post-filtering behavior

            else:
                # Handle unknown search types if necessary
                self.logger.warning(f"Unknown search_type '{search_type}'. Using default similarity search arguments.")
                search_type = 'similarity' # Fallback to similarity

            # --- End of Modified Parameter Mapping ---

            # --- Get Retriever ---
            retriever = self.vectorstore.as_retriever(
                search_type=search_type, # Pass the validated/lowercase search_type
                search_kwargs=search_kwargs
            )

            # --- Invoke Retrieval ---
            docs = retriever.invoke(query)
            self.logger.info(f"Retrieved {len(docs)} documents.")

            return self._format_docs(docs)

        except TypeError as te:
             # Catch potential TypeErrors specifically related to kwargs
             if "got an unexpected keyword argument" in str(te):
                  self.logger.error(f"Mismatch in search arguments for search_type='{search_type}'. Args passed: {search_kwargs}. Error: {te}", exc_info=True)
                  return f"Error: Configuration mismatch for search type '{search_type}'. Check parameters."
             else:
                  # Re-raise other TypeErrors
                  raise
        except Exception as e:
            self.logger.error(f"Retrieval failed during invoke/format: {str(e)}", exc_info=True)
            return "" # Return empty string on general failure

    def _format_docs(self, docs: List[Any]) -> str:
        # ... (Same as before) ...
        formatted_docs = []
        for idx, doc in enumerate(docs):
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content
            truncated_content = f"{content[:500]}{'...' if len(content) > 500 else ''}"
            formatted_docs.append(
                f"DOCUMENT {idx+1}: Source='{source}', Page={page}\n"
                f"CONTENT: {truncated_content}"
            )
        return "\n\n".join(formatted_docs)

    def close(self):
        # ... (Same as before) ...
        if self.weaviate_client and self.weaviate_client.is_connected():
            self.logger.info("Closing Weaviate client connection.")
            self.weaviate_client.close()

# Example usage (for testing retriever.py directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        retriever = TechnicalRetriever()

        # Test MMR (should work now)
        cfg.retrieval.SEARCH_TYPE = 'mmr'
        cfg.retrieval.SCORE_THRESHOLD = 0.6 # Keep it in config, but code should ignore it for MMR
        test_query_mmr = "Explain LCN architecture"
        print(f"\n--- Testing MMR (Score Threshold {cfg.retrieval.SCORE_THRESHOLD} should be ignored) ---")
        context_mmr = retriever.get_context(test_query_mmr)
        print(context_mmr)

        # Test Similarity with Threshold (will use score_threshold)
        cfg.retrieval.SEARCH_TYPE = 'similarity_score_threshold'
        cfg.retrieval.SCORE_THRESHOLD = 0.7 # Example higher threshold
        test_query_thresh = "details about cybersecurity mss"
        print(f"\n--- Testing Similarity + Threshold (Score Threshold {cfg.retrieval.SCORE_THRESHOLD} should be applied) ---")
        context_thresh = retriever.get_context(test_query_thresh)
        print(context_thresh)

        retriever.close()
    except Exception as main_e:
        print(f"Error during direct test: {main_e}")