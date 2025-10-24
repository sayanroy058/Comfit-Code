import os
from supabase import create_client

# 🔑 Directly set Supabase project credentials
SUPABASE_URL = "https://tyswhhteurchuzkngqja.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR5c3doaHRldXJjaHV6a25ncWphIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzE5MzQxMCwiZXhwIjoyMDY4NzY5NDEwfQ.Oot7sxH3lxDeXxQyrkGM-NrkhDRkWmxr8GBTnWcGiVU"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

bucket_name = "comfit_images"
images_folder = r"D:\Sahithi\9_3_2025_ComFit\ComFit\extracted_images_for_upload"

def upload_all_images(folder):
    if not os.path.exists(folder):
        print(f"❌ Folder does not exist: {folder}")
        return
    
    files = os.listdir(folder)
    for file_name in files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder, file_name)
            with open(file_path, "rb") as f:
                print(f"⬆️ Uploading {file_name}...")
                response = supabase.storage.from_(bucket_name).upload(file_name, f)
                # Construct public URL
                public_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{file_name}"
                print(f"✅ Uploaded {file_name} → {public_url}")
    print("🎉 All images uploaded successfully.")

if __name__ == "__main__":
    upload_all_images(images_folder)
