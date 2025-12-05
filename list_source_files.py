import os
from pathlib import Path

SOURCE_DIR = Path(r"C:\Users\Utente\OneDrive - Università degli Studi di Bari\Universita\Magistrale\II Anno\I Semestre\Semantics\Materiale didattico")
OUTPUT_FILE = Path("source_files.txt")

def list_files():
    if not SOURCE_DIR.exists():
        print(f"Directory not found: {SOURCE_DIR}")
        return

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for root, dirs, files in os.walk(SOURCE_DIR):
                for file in files:
                    f.write(os.path.join(root, file) + "\n")
        print(f"File list saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error listing files: {e}")

if __name__ == "__main__":
    list_files()
