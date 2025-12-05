import json
from pathlib import Path

DATA_FILE = Path("data/processed/parsed_course_data.json")
RAW_DIR = Path("data/raw/syllabuses").resolve()

def update_paths():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        updated_count = 0
        for course in data:
            old_path = course.get("path", "")
            if not old_path:
                continue
            
            filename = Path(old_path).name
            new_path = RAW_DIR / filename
            
            if new_path.exists():
                course['path'] = str(new_path)
                updated_count += 1
            else:
                # Try checking if it's the SIIA syllabus we manually created/copied
                if "SIIA" in filename and (RAW_DIR / "SIIA_syllabus.pdf").exists():
                     course['path'] = str(RAW_DIR / "SIIA_syllabus.pdf")
                     updated_count += 1

        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"Updated {updated_count} paths in parsed_course_data.json")

    except Exception as e:
        print(f"Error updating file: {e}")

if __name__ == "__main__":
    update_paths()
