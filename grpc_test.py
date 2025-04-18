import grpc
import logging
import time

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TARGET_ADDRESS = 'localhost:50051'
TIMEOUT_SECONDS = 7 # Increased timeout slightly

if __name__ == "__main__":
    logging.info(f"Attempting to create insecure gRPC channel to {TARGET_ADDRESS}")
    channel = None # Define channel outside try block for finally
    try:
        # Create an insecure channel (no TLS)
        channel = grpc.insecure_channel(TARGET_ADDRESS)
        logging.info("Channel created. Waiting for channel to become READY...")
        try:
            # Wait for the channel to be ready, with a timeout
            # This will block until connection succeeds or timeout expires
            grpc.channel_ready_future(channel).result(timeout=TIMEOUT_SECONDS)
            logging.info(f"SUCCESS: gRPC channel reached READY state within {TIMEOUT_SECONDS} seconds.")
        except grpc.FutureTimeoutError:
            # This is the most likely error if connection fails
            logging.error(f"FAILURE: gRPC channel did NOT become READY within {TIMEOUT_SECONDS} seconds (Timeout).")
        except Exception as e: # Catch other potential errors during wait
            # Examples: DNS resolution errors, connection refused errors handled by grpc internal mechanisms
            logging.error(f"FAILURE: Error waiting for channel readiness: {e}", exc_info=True)

    except Exception as e:
        # Errors during channel creation itself
        logging.error(f"Failed to create channel: {e}", exc_info=True)
    finally:
        if channel:
            # Ensure channel is closed
            channel.close()
            logging.info("Channel closed.")

    logging.info("Test finished.")

