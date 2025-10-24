import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Get environment variables with fallbacks
url: str = os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
key: str = os.environ.get("NEXT_PUBLIC_SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
role: str = os.environ.get("NEXT_PUBLIC_SUPABASE_ROLE") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

# Validate required environment variables
if not url:
    print("⚠️  Warning: NEXT_PUBLIC_SUPABASE_URL or SUPABASE_URL not found in environment variables")
    print("   Please set one of these in your .env file")
    url = "http://localhost:54321"  # Fallback for testing

if not key:
    print("⚠️  Warning: NEXT_PUBLIC_SUPABASE_KEY or SUPABASE_ANON_KEY not found in environment variables")
    print("   Please set one of these in your .env file")
    key = "dummy-key"  # Fallback for testing

if not role:
    print("⚠️  Warning: NEXT_PUBLIC_SUPABASE_ROLE or SUPABASE_SERVICE_ROLE_KEY not found in environment variables")
    print("   Please set one of these in your .env file")
    role = "dummy-role"  # Fallback for testing

# Create Supabase clients
try:
    #for database
    supabase: Client = create_client(url, role)
    #for auth
    supabase_auth: Client = create_client(url, key)
    print("✅ Supabase clients created successfully")
except Exception as e:
    print(f"❌ Failed to create Supabase clients: {e}")
    # Create dummy clients for testing
    supabase = None
    supabase_auth = None

async def create_tables():
    """
    Initialize database tables. This function is called on startup.
    Note: With Supabase, tables are typically created through migrations,
    but this function can be used to verify table existence or create
    any missing tables if needed.
    """
    if not supabase:
        print("❌ Cannot test database connection - Supabase client not initialized")
        return False
        
    try:
        # Test connection
        test_query = supabase.table("conversations").select("id").limit(1).execute()
        print("✅ Database connection successful - tables exist")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("Please ensure your Supabase database has the required tables:")
        print("- conversations")
        print("- messages") 
        print("- branches")
        return False
