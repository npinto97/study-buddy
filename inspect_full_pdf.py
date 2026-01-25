import sys
import os
import fitz

def main():
    file_path = r"data\raw\supplementary_materials\semantic-web.pdf"
    print(f"Opening: {file_path}")
    
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    print(f"Extracted {len(full_text)} chars.")
    
    # Search for keywords with context
    keywords = ["machine", "process", "readable"]
    lower_text = full_text.lower()
    
    for kw in keywords:
        indices = [i for i in range(len(lower_text)) if lower_text.startswith(kw, i)]
        print(f"\n--- Matches for '{kw}' ({len(indices)}) ---")
        for i in indices:
            start = max(0, i - 100)
            end = min(len(full_text), i + 100)
            print(f"...{full_text[start:end].replace(chr(10), ' ')}...")
            
    print("\n--- Tim Berners-Lee Quote ---")
    if "Tim Berners-Lee" in full_text:
        idx = full_text.find("The Semantic Web is an")
        if idx != -1:
             print(full_text[idx:idx+300].replace('\n', ' '))

if __name__ == "__main__":
    main()
