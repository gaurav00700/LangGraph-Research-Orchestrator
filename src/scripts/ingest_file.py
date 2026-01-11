"""
CLI Utility for ingesting files into the VIBE Vector Store via the server's /upload endpoint.
Supports PDF, TXT, and MD files.
"""
import argparse
import requests
import sys
import os

UPLOAD_URL = "http://localhost:8000/upload"

def ingest_file(filepath):
    """Uploads a file to the VIBE Agent Vector Store."""
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found at {filepath}")
        return

    filename = os.path.basename(filepath)
    print(f"🚀 Uploading {filename}...")
    
    try:
        with open(filepath, "rb") as f:
            files = {"file": (filename, f, "application/octet-stream")}
            response = requests.post(UPLOAD_URL, files=files)
            
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Encoded {data['size']} characters.")
            print(f"📄 Result: {data['result']}")
        else:
            print(f"❌ Failed (Status {response.status_code}): {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("Ensure the server is running on localhost:8000")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a file (PDF/MD/TXT) into the VIBE Vector Store.")
    parser.add_argument("-f", "--file-path", required=True, help="Path to the file to ingest.")
    
    args = parser.parse_args()
    ingest_file(args.file_path)
