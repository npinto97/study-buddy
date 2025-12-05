import json
from pathlib import Path

DATA_FILE = Path("data/processed/parsed_course_data.json")

def inspect_siia_entries():
    if not DATA_FILE.exists():
        print(f"File not found: {DATA_FILE}")
        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Total entries: {len(data)}")
        siia_entries = [c for c in data if "SIIA" in c.get("course_name", "")]
        print(f"SIIA entries found: {len(siia_entries)}")

        for i, entry in enumerate(siia_entries):
            print(f"Entry {i}:")
            print(f"  Path: {entry.get('path')}")
            print(f"  Content Length: {len(entry.get('content', ''))}")
            print("-" * 20)

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect_siia_entries()
