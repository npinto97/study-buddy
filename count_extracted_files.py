from pathlib import Path

def count_extracted_files():
    extracted_dir = Path("data/processed/extracted_text")
    if not extracted_dir.exists():
        print("extracted_text directory not found.")
        return

    files = list(extracted_dir.glob("*.txt"))
    print(f"Total extracted text files: {len(files)}")

if __name__ == "__main__":
    count_extracted_files()
