import json
from pathlib import Path

DATA_FILE = Path("data/processed/parsed_course_data.json")

def print_siia_entry():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for course in data:
        if "SIIA" in course.get("course_name", ""):
            print(json.dumps(course, indent=4))
            break

if __name__ == "__main__":
    print_siia_entry()
