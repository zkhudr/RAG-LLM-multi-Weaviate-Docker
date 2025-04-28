import weaviate

# Connect to your local instance (adjust host/port if needed)
client = weaviate.connect_to_local(
    host="localhost",
    port=8091,
    grpc_port=51002,
)

# List all collections (classes)
collections = client.collections.list_all(simple=True)
print("Collections:", collections)

# To get detailed schema for all collections
collections_detailed = client.collections.list_all(simple=False)
print("Detailed Collections Schema:", collections_detailed)

# To get document count for 'Industrial_tech'
if "Industrial_tech" in collections:
    collection = client.collections.get("Industrial_tech")
    count = collection.aggregate().count()
    print(f"Document count in Industrial_tech: {count}")

    # Fetch and print a few sample objects (adjust property names as needed)
    objs = collection.query.fetch_objects(limit=3)
    print("Sample objects:", objs)
else:
    print("Collection 'Industrial_tech' not found!")

client.close()
