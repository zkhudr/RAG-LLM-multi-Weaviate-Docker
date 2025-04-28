import weaviate

client = weaviate.Client(
    "http://localhost:8091",  # The URL of the Weaviate instance
    grpc_url="grpc://localhost:50062"  # gRPC URL for communication (optional)
)

# Check if the client is connected
if client.is_ready():
    print("Weaviate client initialized successfully!")
else:
    print("Failed to connect to Weaviate.")
