from pathlib import Path
from study_buddy.vectorstore_pipeline.document_loader import load_document

SIIA_SYLLABUS_PATH = Path(r"C:\Users\Utente\OneDrive - Università degli Studi di Bari\Universita\Magistrale\II Anno\I Semestre\Semantics\Project\study-buddy\data\raw\syllabuses\SIIA_syllabus.pdf")

def test_extraction():
    if not SIIA_SYLLABUS_PATH.exists():
        print(f"File not found: {SIIA_SYLLABUS_PATH}")
        return

    print(f"Attempting to load: {SIIA_SYLLABUS_PATH}")
    try:
        docs = load_document(SIIA_SYLLABUS_PATH)
        if not docs:
            print("No documents returned.")
            return

        print(f"Extracted {len(docs)} document chunks.")
        full_text = "\n".join([d.page_content for d in docs])
        print(f"Total text length: {len(full_text)}")
        
        # Check for grading criteria
        keywords = ["criteria", "grade", "valutazione", "attribuzione"]
        for kw in keywords:
            if kw in full_text.lower():
                print(f"Found keyword '{kw}'")
                idx = full_text.lower().find(kw)
                print(f"Snippet: {full_text[idx:idx+200]}")
            else:
                print(f"Keyword '{kw}' NOT found.")

    except Exception as e:
        print(f"Error loading document: {e}")

if __name__ == "__main__":
    test_extraction()
