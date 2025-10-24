import duckdb
import json

# Connect to the DuckDB database
db_path = r"D:\Sahithi\9_3_2025_ComFit\ComFit\vector_store\extracted_images.duckdb"
con = duckdb.connect(db_path)

# Fetch all rows from the 'documents' table as a dataframe
df = con.execute("SELECT * FROM documents").fetchdf()

# Function to parse and pretty-print JSON content in metadata_ column
def parse_json_column(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return json_string  # In case it's not a valid JSON

# Display the content with formatted columns
for idx, row in df.iterrows():
    node_id = row['node_id']
    text = row['text']
    embedding = row['embedding']  # The 'embedding' column is likely a list
    metadata = parse_json_column(row['metadata_'])  # Parse the JSON content in 'metadata_'
    
    # Print the first few elements of the embedding to avoid large output
    embedding_preview = embedding[:10] if isinstance(embedding, list) else embedding
    
    print(f"--- Row {idx + 1} ---")
    print(f"Node ID: {node_id}")
    print(f"Text: {text[:100]}...")  # Print first 100 characters of text to avoid large output
  #  print(f"Embedding (Preview): {embedding_preview}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")  # Pretty-print metadata
    print("-" * 50)
