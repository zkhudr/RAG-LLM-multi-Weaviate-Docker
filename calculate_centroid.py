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


logger = logging.getLogger(__name__)

def fetch_all_vectors(client: weaviate.Client, collection_name: str, retries=5, delay=2) -> List[np.ndarray]:
    """
    Fetches all vectors for a collection, with retries if gRPC schema isn't ready yet.
    """
    vectors = []
    processed = 0
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            collection = client.collections.get(collection_name)
            for obj in collection.iterator(include_vector=True, return_properties=[]):
                vec = obj.vector.get("default")
                if vec:
                    vectors.append(np.array(vec))
                    processed += 1
                    if processed % 1000 == 0:
                        logger.info(f"Fetched {processed} vectors...")
            logger.info(f"Total fetched vectors: {processed}")
            return vectors
        except weaviate.exceptions.WeaviateQueryError as e:
            if "could not find class" in str(e):
                logger.warning(f"[FetchVectors] Attempt {attempt}/{retries}: Collection not yet visible in gRPC. Retrying in {delay}s...")
                time.sleep(delay)
                last_exc = e
                continue
            else:
                raise
    # Exhausted retries
    logger.error(f"[FetchVectors] Exhausted retries. Last error: {last_exc}")
    raise last_exc

def calculate_and_save_centroid(client, collection_name, instance_alias, base_path, force=False) -> dict:
    alias_safe = instance_alias.replace('.', '_').replace(':', '_')
    col_safe = collection_name.replace(' ', '_')
    save_path = os.path.join(base_path, f"{alias_safe}_{col_safe}_centroid.npy")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    vectors = fetch_all_vectors(client, collection_name)

    if not vectors:
        # Instead of creating zero vector, fail gracefully
        msg = (
            f"The collection '{collection_name}' in instance '{instance_alias}' contains no vectors. "
            "Centroid cannot be calculated. Please ingest documents before computing a centroid."
        )
        logger.warning(msg)
        return {
            "ok": False,
            "skipped": True,
            "error": msg
        }

    cm = CentroidManager(instance_alias=instance_alias, collection_name=collection_name, base_path=base_path)
    old = cm.get_centroid()

    if not force and not should_recalculate_centroid(
        new_vectors=vectors,
        all_vectors=vectors,
        old_centroid=old,
        threshold=cfg.ingestion.CENTROID_AUTO_THRESHOLD,
        diversity_threshold=cfg.ingestion.CENTROID_DIVERSITY_THRESHOLD
    ):
        logger.info("Centroid already up-to-date. Saving stats anyway.")
        if old is not None:
            stats = get_centroid_stats(old)
            cm.save_metadata(stats)
            return {
                "ok": True,
                "skipped": True,
                "message": "Centroid up-to-date.",
                "shape": old.shape,
                "path": str(cm.centroid_path)
            }
        else:
            return {
                "ok": False,
                "error": "Centroid metadata indicates up-to-date, but no existing centroid found."
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
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    host = cfg.retrieval.WEAVIATE_HOST
    http_port = cfg.retrieval.WEAVIATE_HTTP_PORT
    grpc_port = cfg.retrieval.WEAVIATE_GRPC_PORT
    alias = cfg.retrieval.WEAVIATE_ALIAS
    collection = cfg.retrieval.COLLECTION_NAME
    centroid_dir = getattr(cfg.paths, 'CENTROID_DIR', './centroids')

    os.makedirs(centroid_dir, exist_ok=True)

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

    calculate_and_save_centroid(client, collection_name=collection, instance_alias=alias, base_path=centroid_dir, force=False)

    try:
        client.close()
    except Exception:
        pass
