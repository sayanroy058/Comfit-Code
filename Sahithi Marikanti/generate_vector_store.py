from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Optional: Disable OpenAI and use local embedding
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")  # or "all-MiniLM-L6-v2"

# Load docs
documents = SimpleDirectoryReader(input_dir="./ebooks").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Save
index.storage_context.persist(persist_dir="./my_vector_store")

print("✅ Vector store saved")
