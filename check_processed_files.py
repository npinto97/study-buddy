import json
from pathlib import Path

def check_processed_files():
    processed_file = Path("data/processed/processed_docs.json")
    if not processed_file.exists():
        print("processed_docs.json not found.")
        return

    try:
        with open(processed_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"Total processed files: {len(data)}")
        
        extensions = {}
        for entry in data:
            ext = Path(entry['file_path']).suffix
            extensions[ext] = extensions.get(ext, 0) + 1
            
        print("Processed file extensions:")
        for ext, count in extensions.items():
            print(f"  {ext}: {count}")
            
        # Check for specific types
        print(f"DOCX files: {extensions.get('.docx', 0)}")
        print(f"DOC files: {extensions.get('.doc', 0)}")
        print(f"IPYNB files: {extensions.get('.ipynb', 0)}")
        
    except Exception as e:
        print(f"Error reading processed_docs.json: {e}")

if __name__ == "__main__":
    check_processed_files()
