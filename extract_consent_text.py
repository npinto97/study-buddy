
import sys
import os
from study_buddy.utils.tools import DocumentProcessor

def main():
    file_path = "doc_privacy/Consenso al trattamento dei dati.pdf"
    processor = DocumentProcessor()
    
    try:
        text = processor.extract_text(file_path)
        print("--- START PDF CONTENT ---")
        print(text)
        print("--- END PDF CONTENT ---")
    except Exception as e:
        print(f"Error extracting text: {e}")

if __name__ == "__main__":
    main()
