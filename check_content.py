import json
from pathlib import Path

DATA_FILE = Path("data/processed/parsed_course_data.json")

def check_any_content():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        count_with_content = 0
        total = len(data)
        
        for course in data:
            content = course.get("content", "")
            if len(content) > 0:
                count_with_content += 1
                if count_with_content <= 3:
                    print(f"Found content in: {course.get('course_name')} (Length: {len(content)})")

        print(f"\nTotal entries: {total}")
        print(f"Entries with content: {count_with_content}")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    check_any_content()
