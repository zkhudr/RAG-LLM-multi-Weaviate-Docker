# calculate_centroid.py

import os
import sys
import logging
from typing import List

import numpy as np
import weaviate

from config import cfg
from centroid_manager import CentroidManager, should_recalculate_centroid, get_centroid_stats




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


def fetch_all_vectors(client: weaviate.Client, collection_name: str) -> List[np.ndarray]:
    vectors = []
    processed = 0
    try:
        collection = client.collections.get(collection_name)
    except Exception as e:
        logger.error(f"Collection '{collection_name}' not found: {e}")
        return []

    for obj in collection.iterator(include_vector=True, return_properties=[]):
        vec = obj.vector.get("default")
        if vec:
            vectors.append(np.array(vec))
            processed += 1
            if processed % 1000 == 0:
                logger.info(f"Fetched {processed} vectors...")
    logger.info(f"Total fetched vectors: {processed}")
    return vectors


def calculate_and_save_centroid(client, collection_name, save_path, force=False) -> dict:
    """
    Calculates and saves centroid for the given Weaviate collection.
    Returns a dict with status and optional error.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    vectors = fetch_all_vectors(client, collection_name)

    if not vectors:
        # No vectors: create zero centroid (dimension from metadata, probe, or config)
        
        # 1) Try existing metadata for shape
        dim = None
        cm_tmp = CentroidManager(centroid_path=save_path)
        meta = cm_tmp.get_metadata()
        if "shape" in meta and isinstance(meta["shape"], (list, tuple)):
            try:
                dim = int(meta["shape"][0])
            except Exception:
                dim = None
        # 2) Probe via embedding model
        if dim is None and _HAS_OLLAMA:
            try:
                emb = OllamaEmbeddings(
                    model=cfg.model.EMBEDDING_MODEL,
                    base_url=getattr(cfg.retrieval, 'EMBEDDING_BASE_URL', None)
                )
                dim = len(emb.embed_query(""))
            except Exception as e:
                logger.error(f"Failed to probe embedding dimension: {e}")
        # 3) Fallback to configured dimension
        if dim is None:
            dim = getattr(cfg.model, 'EMBEDDING_DIMENSION', None)
            if dim is None:
                raise RuntimeError("Cannot determine embedding dimension for zero‐vector centroid.")

        # Build and save zero vector
        zero_centroid = np.zeros(dim, dtype=float)
        cm = CentroidManager(centroid_path=save_path)
        cm.save_centroid(zero_centroid)
        logger.info(f"Saved zero‐vector centroid of dim {dim} to {save_path}")
        return {"ok": True, "shape": zero_centroid.shape, "path": str(cm.centroid_path)}


    cm = CentroidManager(centroid_path=save_path)
    old = cm.get_centroid()

    if not force and not should_recalculate_centroid(
        new_vectors=vectors,
        all_vectors=vectors,
        old_centroid=old,
        threshold=cfg.ingestion.CENTROID_AUTO_THRESHOLD,
        diversity_threshold=cfg.ingestion.CENTROID_DIVERSITY_THRESHOLD
    ):
        logger.info("Centroid already up-to-date. Saving stats anyway.")
        stats = get_centroid_stats(old)
        cm.save_metadata(stats)
        return {
            "ok": True,
            "skipped": True,
            "message": "Centroid up-to-date.",
            "shape": old.shape,
            "path": str(cm.centroid_path)
        }

    centroid = np.mean(np.vstack(vectors), axis=0)
    logger.info(f"Calculated new centroid. Shape: {centroid.shape}")

    try:
        cm.save_centroid(centroid)
        stats = get_centroid_stats(centroid)
        cm.save_metadata(stats)
        logger.info(f"Saved centroid to: {cm.centroid_path}")
        return {
            "ok": True,
            "shape": centroid.shape,
            "path": str(cm.centroid_path)
        }
    except Exception as e:
        logger.error(f"Failed to save centroid: {e}")
        return { "ok": False, "error": str(e) }




if __name__ == "__main__":
    host = cfg.retrieval.WEAVIATE_HOST
    http_port = cfg.retrieval.WEAVIATE_HTTP_PORT
    grpc_port = cfg.retrieval.WEAVIATE_GRPC_PORT
    collection = cfg.retrieval.COLLECTION_NAME
    centroid_dir = getattr(cfg.paths, 'CENTROID_DIR', './centroids')
    os.makedirs(centroid_dir, exist_ok=True)
    save_path = os.path.join(centroid_dir, f"{collection}_centroid.npy")

    logger.info(f"Connecting to Weaviate at {host}:{http_port} (gRPC:{grpc_port})...")
    try:
        client = weaviate.connect_to_local(host=host, port=http_port, grpc_port=grpc_port)
        if not client.is_connected():
            raise ConnectionError("Could not connect to Weaviate.")
    except Exception as e:
        logger.critical(f"Weaviate connection failed: {e}")
        sys.exit(1)

    try:
        stats = client.collections.get(collection).aggregate.over_all(total_count=True)
        logger.info(f"Collection '{collection}' contains {stats.total_count} objects.")
    except Exception as e:
        logger.error(f"Error accessing collection stats: {e}")
        sys.exit(1)

    calculate_and_save_centroid(client, collection, save_path, force=False)

    try:
        client.close()
    except Exception:
        pass
