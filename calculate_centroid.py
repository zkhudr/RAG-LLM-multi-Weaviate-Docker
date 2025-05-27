# calculate_centroid.py

import os
import sys
import time
import logging
from typing import List

import numpy as np
import weaviate

from config import cfg
from centroid_manager import CentroidManager, should_recalculate_centroid

# Optional import for dimension probe
try:
    from langchain_ollama import OllamaEmbeddings
    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_all_vectors(
    client: weaviate.Client,
    collection_name: str
) -> List[np.ndarray]:
    """
    Fetch all vectors from a Weaviate collection as numpy arrays.
    """
    vectors: List[np.ndarray] = []
    processed_count = 0
    try:
        collection = client.collections.get(collection_name)
    except Exception as e:
        logger.error(f"Collection '{collection_name}' not found: {e}")
        return vectors

    for obj in collection.iterator(include_vector=True, return_properties=[]):
        vec = obj.vector.get('default')
        if vec:
            vectors.append(np.array(vec))
            processed_count += 1
            if processed_count % 1000 == 0:
                logger.info(f"Fetched {processed_count} vectors...")
    logger.info(f"Total fetched vectors: {processed_count}")
    return vectors


def calculate_and_save_centroid(
    client: weaviate.Client,
    collection_name: str,
    save_path: str,
    force: bool = False
) -> None:
    """Fetch vectors, compute centroid if needed, and save it."""
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    # 1) Fetch everything
    all_vectors = fetch_all_vectors(client, collection_name)

    # 2) If empty, write a zero-vector fallback
    if not all_vectors:
        logger.warning("No vectors found; creating zero-vector centroid fallback.")
        # Determine embedding dimension
        dim = None
        if _HAS_OLLAMA:
            try:
                emb = OllamaEmbeddings(
                    model=cfg.model.EMBEDDING_MODEL,
                    base_url=getattr(cfg.retrieval, 'EMBEDDING_BASE_URL', None)
                )
                dim = len(emb.embed_query(""))
            except Exception as e:
                logger.error(f"Failed to probe embedding dimension via OllamaEmbeddings: {e}")
        if dim is None:
            logger.error("Cannot determine embedding dimension; aborting centroid save.")
            return

        zero_centroid = np.zeros(dim, dtype=float)
        cm = CentroidManager(centroid_path=save_path)
        try:
            cm.save_centroid(zero_centroid)
            logger.info(f"Saved zero-vector centroid to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save zero-vector centroid: {e}")
        return

    # 3) Load existing centroid
    cm = CentroidManager(centroid_path=save_path)
    old_centroid = cm.get_centroid()

    # 4) Decide if recalculation is needed
    if not force and not should_recalculate_centroid(
        new_vectors=all_vectors,
        all_vectors=all_vectors,
        old_centroid=old_centroid,
        threshold=cfg.ingestion.CENTROID_AUTO_THRESHOLD,
        diversity_threshold=cfg.ingestion.CENTROID_DIVERSITY_THRESHOLD
    ):
        logger.info("Centroid up-to-date; no recalculation needed.")
        return

    # 5) Compute new centroid
    centroid_vector = np.mean(np.vstack(all_vectors), axis=0)
    logger.info(f"Calculated new centroid with shape: {centroid_vector.shape}")

    # 6) Save centroid
    try:
        cm.save_centroid(centroid_vector)
        logger.info(f"Saved centroid to: {cm.centroid_path}")
    except Exception as e:
        logger.error(f"Failed to save centroid: {e}")


if __name__ == "__main__":
    # Read settings from config
    host = cfg.retrieval.WEAVIATE_HOST
    http_port = cfg.retrieval.WEAVIATE_HTTP_PORT
    grpc_port = cfg.retrieval.WEAVIATE_GRPC_PORT
    collection_name = cfg.retrieval.COLLECTION_NAME
    centroid_save_path = getattr(cfg.paths, 'DOMAIN_CENTROID_PATH', './domain_centroid.npy')

    # Connect to Weaviate
    logger.info(f"Connecting to Weaviate at {host}:{http_port} (gRPC:{grpc_port})...")
    try:
        client = weaviate.connect_to_local(
            host=host,
            port=http_port,
            grpc_port=grpc_port
        )
        if not client.is_connected():
            raise ConnectionError("Could not connect to Weaviate.")
        logger.info("Weaviate connection established.")
    except Exception as e:
        logger.critical(f"Connection failed: {e}")
        sys.exit(1)

    # Optionally check collection existence and count
    try:
        stats = client.collections.get(collection_name).aggregate.over_all(total_count=True)
        logger.info(f"Collection '{collection_name}' contains {stats.total_count} objects.")
    except Exception as e:
        logger.error(f"Error querying collection '{collection_name}': {e}")
        sys.exit(1)

    # Run calculation
    calculate_and_save_centroid(
        client,
        collection_name,
        centroid_save_path,
        force=False
    )

    # Close connection
    try:
        client.close()
        logger.info("Weaviate connection closed.")
    except Exception:
        pass
