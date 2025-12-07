import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from study_buddy.utils.embeddings import get_embeddings
from study_buddy.config import FAISS_INDEX_DIR, logger

SYLLABUS_PATH = r"c:\Users\Utente\OneDrive - Università degli Studi di Bari\Universita\Magistrale\II Anno\I Semestre\Semantics\Project\study-buddy\data\raw\syllabuses\MRI_syllabus.pdf"

def force_index():
    print(f"Loading index from {FAISS_INDEX_DIR}...")
    try:
        vector_store = FAISS.load_local(
            str(FAISS_INDEX_DIR), 
            get_embeddings(), 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    print(f"Loading syllabus from {SYLLABUS_PATH}...")
    if not os.path.exists(SYLLABUS_PATH):
        print("Syllabus file not found!")
        return

    try:
        loader = PyPDFLoader(SYLLABUS_PATH)
        docs = loader.load()
        print(f"Loaded {len(docs)} pages.")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        print(f"Split into {len(splits)} chunks.")
        
        # Add metadata source if missing
        for doc in splits:
             doc.metadata["source"] = "MRI_syllabus.pdf"
        
        print("Adding to vector store...")
        vector_store.add_documents(splits)
        
        print("Saving vector store...")
        vector_store.save_local(str(FAISS_INDEX_DIR))
        print("✅ DONE! Syllabus force-indexed.")
        
    except Exception as e:
        print(f"Error during forcing: {e}")

if __name__ == "__main__":
    force_index()
