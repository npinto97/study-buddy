import json
from pathlib import Path

DATA_FILE = Path("data/processed/parsed_course_data.json")

def check_siia_path():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        for course in data:
            if "SIIA" in course.get("course_name", ""):
                print(f"Path: {course.get('path')}")
                # return # Print all if duplicates

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    check_siia_path()
