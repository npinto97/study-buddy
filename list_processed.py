import json
from pathlib import Path

DOCS_FILE = Path("data/processed/processed_docs.json")

def list_processed():
    if not DOCS_FILE.exists():
        print(f"File not found: {DOCS_FILE}")
        return

    try:
        with open(DOCS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"Total processed docs: {len(data)}")
        # We can't easily map hash back to filename unless we compute hash again.
        # But we can verify if the file is empty or not.

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    list_processed()
