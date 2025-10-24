import duckdb
import os
import json
import shutil

# Path to your DuckDB file - update exact path!
duckdb_path = r"D:\Sahithi\9_3_2025_ComFit\ComFit\vector_store\extracted_images.duckdb"
output_folder = r"D:\Sahithi\9_3_2025_ComFit\ComFit\extracted_images_for_upload"

os.makedirs(output_folder, exist_ok=True)

conn = duckdb.connect(duckdb_path)
rows = conn.execute("SELECT metadata_ FROM documents").fetchall()

for i, (metadata_json,) in enumerate(rows):
    try:
        meta = json.loads(metadata_json)
        if meta.get('type') == 'image' and 'path' in meta:
            src_path = meta['path']
            file_name = meta.get('name', f'image_{i}.png')
            if os.path.isfile(src_path):
                dst_path = os.path.join(output_folder, file_name)
                shutil.copy(src_path, dst_path)
                print(f"Copied '{src_path}' to '{dst_path}'")
            else:
                print(f"Image file not found at '{src_path}'")
    except Exception as e:
        print(f"Error processing metadata at index {i}: {e}")

conn.close()
