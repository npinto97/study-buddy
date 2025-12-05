import json
from pathlib import Path

DATA_FILE = Path("data/processed/parsed_course_data.json")
NEW_PATH = str(Path(r"C:\Users\Utente\OneDrive - Università degli Studi di Bari\Universita\Magistrale\II Anno\I Semestre\Semantics\Project\study-buddy\data\raw\syllabuses\SIIA_syllabus.txt"))

def update_json():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        updated = False
        for course in data:
            if "SIIA_syllabus.pdf" in course.get("path", ""):
                print(f"Updating path for: {course.get('course_name')}")
                print(f"Old path: {course['path']}")
                course['path'] = NEW_PATH
                print(f"New path: {course['path']}")
                updated = True
                break
        
        if updated:
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            print("Successfully updated parsed_course_data.json")
        else:
            print("SIIA syllabus entry not found.")

    except Exception as e:
        print(f"Error updating file: {e}")

if __name__ == "__main__":
    update_json()
