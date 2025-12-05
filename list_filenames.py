import json
from pathlib import Path

DATA_FILE = Path("data/processed/parsed_course_data.json")

def list_expected_files():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        filenames = set()
        for course in data:
            path = course.get("path", "")
            if path:
                filenames.add(Path(path).name)
        
        print(f"Found {len(filenames)} unique expected filenames.")
        for name in list(filenames)[:20]:
            print(name)

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    list_expected_files()
