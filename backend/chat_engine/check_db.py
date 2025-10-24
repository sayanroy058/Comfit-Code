import duckdb

# Use forward slashes for maximum compatibility in Python
DB_PATH = 'D:/Sahithi/9_3_2025_ComFit/ComFit/vector_store/extracted_images.duckdb'


try:
    con = duckdb.connect(database=DB_PATH, read_only=True)
    
    # 1. Find the table name
    print("--- Available Tables ---")
    tables = con.execute("PRAGMA show_tables;").fetchall()
    print(tables)
    
    # 2. ASSUME your vector table is the first one, or look for a likely name
    # We'll use the first table found, or 'knowledge_vectors' as a fallback
    VECTOR_TABLE_NAME = tables[0][0] if tables else 'knowledge_vectors'
    
    # Re-verify the table name assumption:
    if not tables:
        print("!! WARNING: No tables found. Cannot proceed. !!")
        if 'con' in locals(): con.close()
        exit()

    print(f"\n--- Schema for Table: {VECTOR_TABLE_NAME} ---")
    schema = con.execute(f"PRAGMA table_info('{VECTOR_TABLE_NAME}');").fetchall()
    
    # Print the column names and types
    column_names = [col[1] for col in schema]
    print("Columns:", column_names)

    # 3. Sample the data to check for the image URL
    print(f"\n--- Sample Data (LIMIT 2) from {VECTOR_TABLE_NAME} ---")
    
    # Search for a column containing 'url', 'image', or 'link' in the name
    url_column = next((col for col in column_names if 'url' in col.lower() or 'image' in col.lower() or 'link' in col.lower()), None)
    
    # Also look for a generic metadata column which might contain the URL as a JSON field
    metadata_column = next((col for col in column_names if 'metadata' in col.lower() or 'meta' in col.lower()), None)
    
    columns_to_select = []
    
    if url_column:
        print(f"** Found direct URL column: {url_column} **")
        columns_to_select.append(url_column)
    elif metadata_column:
        print(f"** Found generic metadata column: {metadata_column} **")
        columns_to_select.append(metadata_column)
    else:
        print("!! Could not find a specific column for URL or Metadata. !!")

    # Assuming a text content column is present for context
    text_column = next((col for col in column_names if 'text' in col.lower() and 'vector' not in col.lower()), None)
    if text_column and text_column not in columns_to_select:
         columns_to_select.append(text_column)
    
    if columns_to_select:
        sample_query = f"SELECT {', '.join(columns_to_select)} FROM '{VECTOR_TABLE_NAME}' LIMIT 2;"
        sample_data = con.execute(sample_query).fetchall()
        
        for i, row in enumerate(sample_data):
            print(f"--- Row {i+1} ---")
            for col_name, value in zip(columns_to_select, row):
                if 'text' in col_name.lower():
                     print(f"{col_name}: {str(value)[:100]}...") # Print first 100 chars of text
                else:
                    print(f"{col_name}: {value}")
            print("-" * 20)
    else:
        print("!! Unable to construct a meaningful sample query. Please examine the 'Columns:' list above. !!")
        
except Exception as e:
    print(f"An error occurred: {type(e).__name__}: {e}")
    print("\n!! Ensure your DB_PATH variable is set correctly and the file exists. !!")
finally:
    if 'con' in locals():
        con.close()