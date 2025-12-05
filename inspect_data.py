import json
from pathlib import Path

DATA_FILE = Path("data/processed/parsed_course_data.json")

def inspect_data():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Total courses found: {len(data)}")
        
        for i, course in enumerate(data):
            print(f"\n--- Course {i+1} ---")
            # Print keys to understand structure
            print(f"Keys: {list(course.keys())}")
            
            # Print specific fields if they exist
            if "course_name" in course:
                print(f"Name: {course['course_name']}")
            
            # Check for grading criteria in content
            content = course.get("content", "")
            print(f"Content length: {len(content)} chars")
            
            if "criteria" in content.lower() or "grade" in content.lower() or "valutazione" in content.lower():
                print("Found keywords 'criteria'/'grade'/'valutazione' in content.")
                # Print a snippet around the keyword
                idx = content.lower().find("valutazione")
                if idx == -1: idx = content.lower().find("grade")
                if idx == -1: idx = content.lower().find("criteria")
                
                start = max(0, idx - 200)
                end = min(len(content), idx + 500)
                print(f"Snippet:\n...{content[start:end]}...")
            else:
                print("WARNING: No grading keywords found in content.")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect_data()
