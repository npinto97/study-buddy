import os
from pathlib import Path

EXTRACTED_DIR = Path("data/processed/extracted_text")

def search_extracted_files():
    if not EXTRACTED_DIR.exists():
        print(f"Directory not found: {EXTRACTED_DIR}")
        return

    keywords = ["criteria", "grade", "valutazione", "attribuzione"]
    found_any = False

    for file_path in EXTRACTED_DIR.glob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Check if this looks like the SIIA syllabus (e.g. contains "SIIA" or "Semantic")
            if "SIIA" in content or "Semantic Intelligent Information Access" in content:
                print(f"\nScanning potential SIIA file: {file_path.name}")
                
                for kw in keywords:
                    if kw in content.lower():
                        print(f"  Found keyword '{kw}'")
                        idx = content.lower().find(kw)
                        print(f"  Snippet: {content[idx:idx+200].replace(chr(10), ' ')}")
                        found_any = True
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not found_any:
        print("No grading keywords found in any potential SIIA extracted files.")

if __name__ == "__main__":
    search_extracted_files()
