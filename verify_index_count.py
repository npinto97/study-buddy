from langchain_community.vectorstores import FAISS
from study_buddy.utils.embeddings import initialize_gpu_embeddings
import os

def verify_index_count():
    embeddings = initialize_gpu_embeddings()
    if os.path.exists("faiss_index"):
        try:
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            count = vector_store.index.ntotal
            print(f"Total documents in FAISS index: {count}")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
    else:
        print("faiss_index directory not found.")

if __name__ == "__main__":
    verify_index_count()
