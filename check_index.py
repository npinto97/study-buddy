from langchain_community.vectorstores import FAISS
from study_buddy.utils.embeddings import embeddings
from study_buddy.config import FAISS_INDEX_DIR

try:
    vector_store = FAISS.load_local(
        str(FAISS_INDEX_DIR), 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    print(f"Index loaded. Total documents: {vector_store.index.ntotal}")
except Exception as e:
    print(f"Error: {e}")
