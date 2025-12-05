from langchain_community.document_loaders import TextLoader
from pathlib import Path

FILE_PATH = Path(r"C:\Users\Utente\OneDrive - Università degli Studi di Bari\Universita\Magistrale\II Anno\I Semestre\Semantics\Project\study-buddy\data\raw\syllabuses\SIIA_syllabus.txt")

def test_loader():
    if not FILE_PATH.exists():
        print(f"File not found: {FILE_PATH}")
        return

    try:
        loader = TextLoader(str(FILE_PATH))
        docs = loader.load()
        print(f"Successfully loaded {len(docs)} docs.")
        print(f"Content preview: {docs[0].page_content[:100]}")
    except Exception as e:
        print(f"Error loading: {e}")

if __name__ == "__main__":
    test_loader()
