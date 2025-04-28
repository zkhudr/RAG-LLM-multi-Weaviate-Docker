import weaviate
import logging
import json
import sys
from weaviate.exceptions import WeaviateConnectionError

# --- Configuration ---
WEAVIATE_HOST = "localhost"
WEAVIATE_HTTP_PORT = 8080
WEAVIATE_GRPC_PORT = 50051
COLLECTION_NAME = "industrial_tech" # The collection you ingested into

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Connects to Weaviate and fetches various stats."""
    client = None # Initialize client to None for finally block
    collection_count = None
    nodes_info = None

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

        # --- 1. Get Collection Object Count using Aggregate ---
        try:
            collection = client.collections.get(COLLECTION_NAME)
            # Use aggregate.over_all to get the total count for the collection
            response = collection.aggregate.over_all(
                total_count=True
            )
            collection_count = response.total_count
            logging.info(f"Successfully retrieved aggregate count for '{COLLECTION_NAME}'.")
        except Exception as e:
            logging.error(f"Could not aggregate count for collection '{COLLECTION_NAME}'. Does it exist? Error: {e}", exc_info=True)
            # Continue to try getting node stats even if aggregation fails

        # --- 2. Get Node-Level Stats ---
        try:
            logging.info("Fetching node information (verbose)...")
            # Use output="verbose" to include shard details
            nodes_info = client.cluster.nodes(output="verbose")
            logging.info(f"Successfully retrieved information for {len(nodes_info)} node(s).")
        except Exception as e:
            logging.error(f"Failed to get node information: {e}", exc_info=True)
            # Continue to print whatever stats were successfully gathered

        # --- Print Collected Stats ---
        print("\n--- Weaviate Instance Stats ---")

        if collection_count is not None:
            print(f"\nCollection '{COLLECTION_NAME}':")
            print(f"  - Total Objects (from Aggregate): {collection_count}")
            print(f"  - Total Vectors (estimated*):    {collection_count}")
            print("     *Assumes 1 vector per object (no named vectors used)")
        else:
            print(f"\nCollection '{COLLECTION_NAME}':")
            print("  - Could not retrieve object count via Aggregate.")

        if nodes_info:
            print("\nNode Information:")
            for i, node in enumerate(nodes_info):
                print(f"\n  --- Node {i+1} ({node.name}) ---")
                print(f"    Status:      {node.status}")
                print(f"    Version:     {node.version} (Git Hash: {node.git_hash})")
                print(f"    Total Shards on Node: {node.stats.shard_count}")
                print(f"    Total Objects on Node: {node.stats.object_count}")
                # In a single-node setup, this node count should ideally match the aggregate count

                print("    Shards Detail:")
                if node.shards:
                    shards_for_collection = 0
                    objects_in_collection_shards = 0
                    for shard in node.shards:
                         print(f"      - Shard Name:  {shard.name}")

                         # --- Attempt to get collection name for the shard ---
                         try:
                             # Try accessing 'collection' attribute
                             collection_name = shard.collection
                             print(f"        Collection:  {collection_name}") # Print the found collection name
                             # Check if it matches the target collection and update counts
                             if collection_name == COLLECTION_NAME:
                                  shards_for_collection += 1
                                  objects_in_collection_shards += shard.object_count
                         except AttributeError:
                             logger.warning(f"Could not determine collection name for shard {shard.name} using 'shard.collection'. Attribute missing?")
                             # Optionally try getattr(shard, 'class', 'Unknown') here as a fallback if needed
                         except Exception as e:
                             logger.error(f"Error processing collection name for shard {shard.name}: {e}")
                         # ----------------------------------------------------

                         print(f"        Object Count:{shard.object_count}") # Print object count for the shard

                    # --- End of loop over shards ---

                    # --- This part outside the loop remains the same ---
                    print(f"    -> Shards for '{COLLECTION_NAME}' on this node: {shards_for_collection}")
                    print(f"    -> Total Objects in '{COLLECTION_NAME}' shards on this node: {objects_in_collection_shards}")
                    if collection_count is not None and objects_in_collection_shards != collection_count:
                         logging.warning(f"Node shard object count ({objects_in_collection_shards}) does not match aggregate count ({collection_count}). This might happen during indexing or if collection spans nodes.")
                else:
                    print("      (No shard details available - output might not be 'verbose' or node has no shards)")
        else:
            print("\nNode Information: Could not be retrieved.")

        print("\n-----------------------------\n")
        print("Note: For detailed disk usage, memory stats, and performance metrics,")
        print("consider setting up Prometheus monitoring as described in Weaviate docs.")
        print("-----------------------------\n")


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
