import sys
import os
sys.path.append(os.getcwd())

from study_buddy.utils.tools import DocumentProcessor

def main():
    file_path = r"data\raw\supplementary_materials\semantic-web.pdf"
    processor = DocumentProcessor()
    
    print(f"Extracting text from: {file_path}")
    text = processor.extract_text(file_path)
    
    print("\n--- EXTRACTED TEXT (First 2000 chars) ---\n")
    print(text[:2000])
    
    print("\n--- SEARCHING FOR KEYWORDS ---\n")
    keywords = ["Tim Berners-Lee", "Web", "machine", "process", "extend", "meaning", "version", "vision"]
    
    lower_text = text.lower()
    for kw in keywords:
        count = lower_text.count(kw.lower())
        print(f"Keyword '{kw}': found {count} times")

    # Find the specific definition if possible
    sentences = text.split('.')
    print("\n--- RELEVANT SENTENCES ---\n")
    for s in sentences:
        if "Tim Berners-Lee" in s or "Semantic Web" in s:
            print(f"- {s.strip()}")

if __name__ == "__main__":
    main()
