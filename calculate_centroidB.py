# calculate_centroid.py
import numpy as np
import weaviate
import logging
import sys
from config import config as cfg # Use config for collection name and centroid path

# --- Configuration ---
# Get Weaviate connection details and centroid path from config
DEFAULT_WEAVIATE_HOST = "localhost" # Or get from config if moved there
DEFAULT_WEAVIATE_HTTP_PORT = 8080
DEFAULT_WEAVIATE_GRPC_PORT = 50051
COLLECTION_NAME = getattr(cfg.retrieval, 'COLLECTION_NAME', "industrial_tech")
CENTROID_SAVE_PATH = getattr(cfg.paths, 'DOMAIN_CENTROID_PATH', "./domain_centroid.npy")

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
            if obj.vector and 'default' in obj.vector: # Adjust 'default' if using named vectors
                all_vectors.append(obj.vector['default'])
                processed_count += 1
                if processed_count % 1000 == 0:
                    logger.info(f"Fetched {processed_count} vectors...")
            # else:
            #    logger.warning(f"Object UUID {obj.uuid} missing vector.")

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

    except Exception as e:
        logger.error(f"Failed to calculate or save centroid: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    client = None
    try:
        logger.info("Connecting to Weaviate...")
        client = weaviate.connect_to_local(
            host=DEFAULT_WEAVIATE_HOST,
            port=DEFAULT_WEAVIATE_HTTP_PORT,
            grpc_port=DEFAULT_WEAVIATE_GRPC_PORT
        )
        if not client.is_connected():
            raise ConnectionError("Failed to connect to Weaviate.")
        logger.info("Weaviate connection successful.")

        calculate_and_save_centroid(client, COLLECTION_NAME, CENTROID_SAVE_PATH)

    except Exception as e:
        logger.critical(f"Centroid calculation failed: {e}")
        sys.exit(1)
    finally:
        if client and client.is_connected():
            client.close()
            logger.info("Weaviate connection closed.")
