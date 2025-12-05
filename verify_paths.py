import json
import random
from pathlib import Path

DATA_FILE = Path("data/processed/parsed_course_data.json")

def verify_random_paths():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Total entries: {len(data)}")
        sample = random.sample(data, 5)

        for i, course in enumerate(sample):
            path = course.get("path", "")
            exists = Path(path).exists()
            print(f"Entry {i}:")
            print(f"  Path: {path}")
            print(f"  Exists: {exists}")
            if not exists:
                print(f"  (File not found on disk)")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    verify_random_paths()
