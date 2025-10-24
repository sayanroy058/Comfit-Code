from llama_index.vector_stores.duckdb import DuckDBVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding, HuggingFaceEmbedding

# ---- HuggingFace embeddings ----
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

db_path = r"D:\Sahithi\9_3_2025_ComFit\ComFit\vector_store\extracted_images.duckdb"

# Load DuckDB store
store = DuckDBVectorStore.from_local(db_path)
index = VectorStoreIndex.from_vector_store(store, embed_model=embed_model)

# Use retriever (no LLM involved)
retriever = index.as_retriever(similarity_top_k=3)

query = "An example of a scanned 3D head model with data quality issues"
nodes = retriever.retrieve(query)

print("\n✅ Retrieved Results:")
for node in nodes:
    name = node.metadata.get("name")
    caption = node.metadata.get("caption")
    page = node.metadata.get("page")
    print(f"🖼️ File: {name}\n📖 Page: {page}\n📝 Caption: {caption}\n")
