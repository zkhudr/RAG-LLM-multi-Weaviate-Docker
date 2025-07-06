# retriever.py

import logging
from typing import Optional

from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from weaviate.config import AdditionalConfig, Timeout
from langchain_weaviate import WeaviateVectorStore
from langchain_ollama import OllamaEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np

from config import cfg

logger = logging.getLogger(__name__)

class VectorStoreInitError(Exception):
    """Raised when Weaviate or vector store initialization fails."""
    pass

class TechnicalRetriever:
    """
    Retrieves context documents from a Weaviate vector store, via gRPC.
    Supports similarity, MMR, and hybrid searches.
    """
    def __init__(self):
        self.cfg = cfg
        self.logger = logger
        self._init_embeddings()
        self._init_client_and_store()
        # legacy alias used by app.py’s select_weaviate_instance route
        self.weaviate_client = self.client

    def _init_embeddings(self) -> None:
        try:
            model_name = self.cfg.model.EMBEDDING_MODEL
            self.embeddings = OllamaEmbeddings(model=model_name)
            self.logger.info(f"OllamaEmbeddings initialized: {model_name}")
        except Exception:
            self.logger.exception("Failed to initialize embeddings")
            raise

    def _init_client_and_store(self) -> None:
        host       = getattr(self.cfg.retrieval, "WEAVIATE_HOST", "127.0.0.1")
        http_port  = self.cfg.retrieval.WEAVIATE_HTTP_PORT
        grpc_port  = self.cfg.retrieval.WEAVIATE_GRPC_PORT
        collection = self.cfg.retrieval.COLLECTION_NAME
        conn_to, call_to = getattr(self.cfg.retrieval, "WEAVIATE_TIMEOUT", (5, 30))

        self.logger.info(f"Initializing Weaviate gRPC client at {host}:{grpc_port}")
        try:
            params = ConnectionParams.from_params(
                http_host=host,
                http_port=http_port,
                http_secure=getattr(self.cfg.retrieval, "WEAVIATE_HTTP_SECURE", False),
                grpc_host=host,
                grpc_port=grpc_port,
                grpc_secure=False,
            )

            self.client = WeaviateClient(
                connection_params=params,
                additional_config=AdditionalConfig(
                    timeout=Timeout(init=conn_to, call=call_to)
                ),
                additional_headers=getattr(self.cfg.retrieval, "WEAVIATE_HEADERS", {})
            )

            self.client.connect()

            col_configs = self.client.collections.list_all()
            self.logger.info(
                f"Weaviate connection OK, found {len(col_configs)} collections"
            )

            self.vectorstore = WeaviateVectorStore(
                client=self.client,
                index_name=collection,
                text_key="page_content",
                embedding=self.embeddings,
            )
            self.logger.info(f"Vector store ready for collection '{collection}'")

        except Exception:
            self.logger.exception("Failed to initialize Weaviate gRPC client or vector store")
            raise VectorStoreInitError("Could not connect to Weaviate via gRPC")



    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_context(self, query: str) -> str:
        """
        Retrieve document contexts based on SEARCH_TYPE in config.
        Returns concatenated page_content strings.
        """
        vs = getattr(self, 'vectorstore', None)
        if not vs:
            self.logger.error("Vector store not initialized.")
            return ""

        try:
            if not self.client.is_live():
                self.logger.warning("Weaviate client was closed. Reconnecting…")
                self.client.connect()
        except Exception:
            self.logger.exception("Failed to reconnect Weaviate client")
            return ""

        s_type = self.cfg.retrieval.SEARCH_TYPE.lower()
        k      = self.cfg.retrieval.K_VALUE
        α      = self.cfg.retrieval.LAMBDA_MULT

        try:
            query_vector = self.embeddings.embed_query(query)
            norm = float(np.linalg.norm(query_vector))
            if norm == 0:
                self.logger.error(f"Query vector is zero! Query='{query}'")
            else:
                self.logger.info(f"[DEBUG] Query vector norm: {norm:.6f}")
        except Exception:
            self.logger.exception("Failed to compute query embedding vector.")
            return ""

        self.logger.info(f"Retrieving context: type={s_type}, k={k}")
        try:
            if s_type == 'mmr':
                docs = vs.max_marginal_relevance_search(
                    query, k=k, fetch_k=k * 2, lambda_mult=α
                )
            elif s_type == 'similarity':
                docs = vs.similarity_search(query, k=k)
            elif s_type == 'similarity_score_threshold':
                thresh = self.cfg.retrieval.SCORE_THRESHOLD
                docs = vs.similarity_search_with_score_threshold(
                    query, k=k, score_threshold=thresh
                )
            elif s_type == 'hybrid':
                docs = vs.hybrid_search(query, k=k, alpha=α)
            else:
                self.logger.warning(f"Unknown search type '{s_type}', falling back to similarity.")
                docs = vs.similarity_search(query, k=k)

            if not docs:
                self.logger.warning("No documents returned from vectorstore search.")
                return ""

            return "\n\n".join(
                getattr(doc, "page_content", "") for doc in docs
                if getattr(doc, "page_content", "").strip()
            )

        except Exception:
            self.logger.exception("Error during context retrieval")
            return ""



    def close(self) -> None:
        """Gracefully close the Weaviate client."""
        client = getattr(self, 'client', None)
        if client:
            try:
                self.logger.info("Closing Weaviate client.")
                client.close()
            except Exception:
                self.logger.warning("Error closing Weaviate client", exc_info=True)


if __name__ == '__main__':
    def setup_logging(force_utf8=False, level=logging.INFO):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        if force_utf8:
            try:
                stream_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
            except Exception:
                pass
        logging.basicConfig(handlers=[stream_handler], level=level, force=True)

    setup_logging(force_utf8=True)

    try:
        retriever = TechnicalRetriever()
        print("Context:", retriever.get_context("test query"))
    except Exception as e:
        print("Error:", e)
    finally:
        try:
            retriever.close()
        except Exception:
            pass



class RetrieverSingleton:
    _instance: Optional[TechnicalRetriever] = None

    @classmethod
    def get(cls) -> TechnicalRetriever:
        if cls._instance is None:
            cls._instance = TechnicalRetriever()
        return cls._instance
