import json
from collections import Counter
from pathlib import Path

DATA_FILE = Path("data/processed/parsed_course_data.json")

def check_duplicates():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        paths = [c.get("path") for c in data if c.get("path")]
        counts = Counter(paths)
        
        duplicates = {p: c for p, c in counts.items() if c > 1}
        
        print(f"Total paths: {len(paths)}")
        print(f"Unique paths: {len(counts)}")
        print(f"Duplicate paths: {len(duplicates)}")
        
        with open("duplicates.txt", "w", encoding="utf-8") as f:
            f.write(f"Total paths: {len(paths)}\n")
            f.write(f"Unique paths: {len(counts)}\n")
            f.write(f"Duplicate paths: {len(duplicates)}\n\n")
            if duplicates:
                f.write("Duplicates:\n")
                for p, c in duplicates.items():
                    f.write(f"{p}: {c}\n")
                    print(f"  {p}: {c}")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    check_duplicates()
