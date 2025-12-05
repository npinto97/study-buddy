import json
import os
from pathlib import Path

DATA_FILE = Path("data/processed/parsed_course_data.json")
RAW_DIR = Path("data/raw").resolve()

def update_paths_recursive():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return

    # 1. Index all files in data/raw
    print("Indexing files in data/raw...")
    file_map = {}
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            file_map[file] = str(Path(root) / file)
    
    print(f"Indexed {len(file_map)} files.")

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        updated_count = 0
        not_found_count = 0
        
        for course in data:
            old_path = course.get("path", "")
            if not old_path:
                continue
            
            filename = Path(old_path).name
            
            if filename in file_map:
                course['path'] = file_map[filename]
                updated_count += 1
            else:
                # Special case for SIIA syllabus if we renamed it or it's different
                if "SIIA" in filename and "SIIA_syllabus.txt" in file_map:
                     course['path'] = file_map["SIIA_syllabus.txt"]
                     updated_count += 1
                else:
                    # print(f"File not found: {filename}")
                    not_found_count += 1

        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"Updated {updated_count} paths.")
        print(f"Not found: {not_found_count}")

    except Exception as e:
        print(f"Error updating file: {e}")

if __name__ == "__main__":
    update_paths_recursive()
