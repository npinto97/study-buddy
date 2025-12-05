import json
from pathlib import Path

DATA_FILE = Path("data/processed/parsed_course_data.json")

def inspect_paths():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Total entries: {len(data)}")
        for i, course in enumerate(data[:5]):
            print(f"Entry {i}: {course.get('path', 'NO_PATH')}")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect_paths()
