import weaviate
import logging
import json
import sys
from weaviate.exceptions import WeaviateConnectionError

# --- Configuration ---
WEAVIATE_HOST = "localhost"
WEAVIATE_HTTP_PORT = 8080
WEAVIATE_GRPC_PORT = 50051
COLLECTION_NAME = "industrial_tech" # Same as in your ingestion script
NUM_OBJECTS_TO_FETCH = 50 # How many objects to retrieve for verification

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Connects to Weaviate, fetches objects, and prints them."""
    client = None # Initialize client to None for finally block
    try:
        # --- Connect to Weaviate ---
        logging.info(f"Attempting to connect to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_HTTP_PORT}...")
        client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_HTTP_PORT,
            grpc_port=WEAVIATE_GRPC_PORT
        )
        if not client.is_ready():
            raise WeaviateConnectionError("Client connected but not ready.")
        logging.info("Weaviate connection successful and ready.")

        # --- Get Collection Reference ---
        try:
            collection = client.collections.get(COLLECTION_NAME)
            logging.info(f"Successfully got collection reference: '{COLLECTION_NAME}'")
        except Exception as e:
            logging.error(f"Could not get collection '{COLLECTION_NAME}'. Does it exist? Error: {e}", exc_info=True)
            sys.exit(1)

        # --- Fetch Objects ---
        logging.info(f"Fetching first {NUM_OBJECTS_TO_FETCH} objects from '{COLLECTION_NAME}'...")
        try:
            response = collection.query.fetch_objects(
                limit=NUM_OBJECTS_TO_FETCH,
                # --- Request other valid metadata fields ---
                return_metadata=["creation_time", "last_update_time"], # Example
                # --------------------------------------------
                include_vector=False
            )
            logging.info(f"Query successful. Retrieved {len(response.objects)} objects.")

        except Exception as e:
            logging.error(f"Failed to query objects: {e}", exc_info=True)
            sys.exit(1)

        # --- Print Results ---
        if not response.objects:
            logging.warning("No objects found in the collection. Was the ingestion successful?")
        else:
            print("\n--- Fetched Objects ---")
            for i, obj in enumerate(response.objects):
                print(f"\n--- Object {i+1} ---")
                print(f"Weaviate UUID: {obj.uuid}") 
                print("Properties:")
                # Use json.dumps for pretty printing the properties dictionary
                # default=str handles potential date/datetime objects if they weren't stored as strings
                print(json.dumps(obj.properties, indent=2, default=str))
            print("\n---------------------\n")
            logging.info(f"Successfully printed {len(response.objects)} objects.")

    except WeaviateConnectionError as e:
        logging.error(f"Could not connect to Weaviate: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # --- Close Connection ---
        if client and client.is_connected():
            logging.info("Closing Weaviate client connection.")
            client.close()

if __name__ == "__main__":
    main()
