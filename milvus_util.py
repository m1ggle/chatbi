from pymilvus import MilvusClient

client = MilvusClient(".milvus.db")
print(client.list_collections())

