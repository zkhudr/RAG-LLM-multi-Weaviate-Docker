# calculate_centroid.py

import numpy as np
import weaviate
import logging
import sys
from config import cfg # Use config for collection name, centroid path, and connection details
from centroid_manager import should_recalculate_centroid

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_and_save_centroid(client: weaviate.Client, collection_name: str, save_path: str):
    """Fetches all vectors from a Weaviate collection, calculates the centroid (mean), and saves it."""
    try:
        collection = client.collections.get(collection_name)
        logger.info(f"Fetching vectors from collection '{collection_name}' to calculate centroid...")

        all_vectors = []
        processed_count = 0

        # Iterate through all objects, fetching only the vector
        for obj in collection.iterator(include_vector=True, return_properties=[]):
            if obj.vector and 'default' in obj.vector:  # Adjust 'default' if using named vectors
                all_vectors.append(obj.vector['default'])
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Fetched {processed_count} vectors...")

        if not all_vectors:
            logger.error("No vectors found in the collection. Cannot calculate centroid.")
            return

        logger.info(f"Fetched a total of {len(all_vectors)} vectors.")

        # Calculate the mean vector (centroid)
        centroid_vector = np.mean(np.array(all_vectors), axis=0)
        logger.info(f"Calculated centroid vector with shape: {centroid_vector.shape}")

        # Save the centroid vector using NumPy
        np.save(save_path, centroid_vector)
        logger.info(f"Domain centroid vector saved successfully to: {save_path}")

        # Example of using should_recalculate_centroid function after fetching vectors
        if should_recalculate_centroid(new_vectors=all_vectors, all_vectors=all_vectors, old_centroid=None):
            logger.info("Centroid needs recalculation.")
        else:
            logger.info("No need to recalculate centroid.")

    except Exception as e:
        logger.error(f"Failed to calculate or save centroid: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    client = None
    try:
        # Get connection details from config
        host = cfg.retrieval.WEAVIATE_HOST
        http_port = cfg.retrieval.WEAVIATE_HTTP_PORT
        grpc_port = cfg.retrieval.WEAVIATE_GRPC_PORT
        collection_name = cfg.retrieval.COLLECTION_NAME
        centroid_save_path = getattr(cfg.paths, 'DOMAIN_CENTROID_PATH', "./domain_centroid.npy")
        
        logger.info(f"Connecting to Weaviate at {host}:{http_port} (gRPC: {grpc_port})...")
        
        client = weaviate.connect_to_local(
            host=host,
            port=http_port,
            grpc_port=grpc_port
        )

        if not client.is_connected():
            raise ConnectionError(f"Failed to connect to Weaviate at {host}:{http_port}.")
            
        logger.info("Weaviate connection successful.")
        
        # Optional: Check collection exists
        try:
            collection_check = client.collections.get(collection_name)
            count = collection_check.aggregate.over_all(total_count=True).total_count
            logger.info(f"Collection '{collection_name}' has {count} items.")
        except Exception as e:
            logger.error(f"Error checking collection '{collection_name}': {e}")
            raise
            
        calculate_and_save_centroid(client, collection_name, centroid_save_path)
        
    except Exception as e:
        logger.critical(f"Centroid calculation failed: {e}")
        sys.exit(1)
    finally:
        if client and client.is_connected():
            client.close()
            logger.info("Weaviate connection closed.")
