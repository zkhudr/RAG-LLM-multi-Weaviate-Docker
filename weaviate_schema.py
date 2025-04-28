import weaviate
client = weaviate.connect_to_local(
    host="localhost",
    port=8090,  # Use the port of your first Weaviate instance
    grpc_port=50061
)
schema = client.schema.get()
print(json.dumps(schema, indent=2))
