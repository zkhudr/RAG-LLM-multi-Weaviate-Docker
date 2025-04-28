import weaviate

client = weaviate.connect_to_local(host="localhost", port=8091, grpc_port=51002)
collection = client.collections.get("Industrial_tech")
objs = collection.query.fetch_objects(limit=10)
for obj in objs.objects:
    print(obj.properties.get('content', ''), obj.properties.get('text', ''))
client.close()
