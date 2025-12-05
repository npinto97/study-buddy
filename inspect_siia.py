import json
from pathlib import Path

DATA_FILE = Path("data/processed/parsed_course_data.json")

def inspect_siia_content():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Total entries: {len(data)}")
        for course in data:
            print(f"Entry: {course.get('course_name', 'Unknown')} | Type: {course.get('type', 'Unknown')} | Path: {course.get('path', 'Unknown')}")
            
            # Check for SIIA in course name or content
            if "SIIA" in course.get("course_name", "") or "Sviluppo di Interfacce" in course.get("course_name", "") or "SIIA" in course.get("path", ""):
                print(f"Found SIIA Course: {course.get('course_name')}")
                content = course.get("content", "")
                print(f"Content length: {len(content)}")
                
                # Look for "Criteria" or "Valutazione"
                keywords = ["criteria", "grade", "valutazione", "attribuzione"]
                found = False
                for kw in keywords:
                    idx = content.lower().find(kw)
                    if idx != -1:
                        print(f"\n--- Found keyword '{kw}' at index {idx} ---")
                        start = max(0, idx - 500)
                        end = min(len(content), idx + 1000)
                        print(f"Snippet:\n{content[start:end]}")
                        found = True
                
                if not found:
                    print("WARNING: No grading keywords found in SIIA content.")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect_siia_content()
