import os
import json

# Conditional import — supabase is NOT installed on Vercel to keep bundle small
try:
    from supabase import create_client, Client
    _SUPABASE_AVAILABLE = True
except ImportError:
    _SUPABASE_AVAILABLE = False
    Client = None

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://vwzwulyfvfdxjbadacvo.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZ3end1bHlmdmZkeGpiYWRhY3ZvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTExMTExNsQsImV4cCI6MjEwNjY4NzE2NH0.fUPNjqxjigk2WmM68_rVciW6tQDL2yqAqmZne2i_dmc")

def get_supabase_client():
    if not _SUPABASE_AVAILABLE:
        return None
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return client
    except Exception as e:
        print(f"[Supabase] Connection error: {e}")
        return None

def upload_file_to_supabase(file_path: str, bucket_name: str = "churn-datasets") -> str:
    """
    Uploads a local file to Supabase Storage and returns the public URL.
    """
    client = get_supabase_client()
    if not client:
        return None

    try:
        # Check if bucket exists, create if missing
        existing_buckets = [b.name for b in client.storage.list_buckets()]
        if bucket_name not in existing_buckets:
            client.storage.create_bucket(bucket_name, options={"public": True})

        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            client.storage.from_(bucket_name).upload(
                path=file_name,
                file=f,
                file_options={"cache-control": "3600", "upsert": "true"}
            )
        
        public_url = client.storage.from_(bucket_name).get_public_url(file_name)
        print(f"[Supabase] File uploaded to {bucket_name}: {public_url}")
        return public_url
    except Exception as e:
        print(f"[Supabase] Upload warning: {e}")
        return None

def log_campaign_to_supabase(campaign_data: dict, table_name: str = "campaign_logs"):
    """
    Logs campaign optimization results to Supabase database.
    """
    client = get_supabase_client()
    if not client:
        return None

    try:
        res = client.table(table_name).insert(campaign_data).execute()
        print(f"[Supabase] Logged campaign to database table '{table_name}'")
        return res
    except Exception as e:
        print(f"[Supabase] Database log notice: {e}")
        return None
