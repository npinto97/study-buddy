import json
from pathlib import Path

DATA_FILE = Path("data/processed/parsed_course_data.json")

def find_syllabus_entry():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        found = False
        for entry in data:
            path = entry.get("path", "")
            if "SIIA_syllabus.txt" in path:
                print(f"Found entry for SIIA_syllabus.txt:")
                print(f"  Path: {path}")
                print(f"  Exists: {Path(path).exists()}")
                found = True
                break
        
        if not found:
            print("Entry for SIIA_syllabus.txt NOT found in parsed_course_data.json")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    find_syllabus_entry()
