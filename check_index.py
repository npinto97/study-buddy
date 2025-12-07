from langchain_community.vectorstores import FAISS
from study_buddy.utils.embeddings import get_embeddings
from study_buddy.config import FAISS_INDEX_DIR

try:
    vector_store = FAISS.load_local(
        str(FAISS_INDEX_DIR), 
        get_embeddings(), 
        allow_dangerous_deserialization=True
    )
    print(f"Index loaded. Total documents: {vector_store.index.ntotal}")
    
    found = False
    count = 0
    for doc_id, doc in vector_store.docstore._dict.items():
        if "MRI_syllabus.pdf" in doc.metadata.get("source", ""):
            found = True
            count += 1
            if count == 1:
                print(f"First match found: {doc.metadata}")
                print(f"Sample content: {doc.page_content[:200]}")
    
    if found:
        print(f"✅ MRI_syllabus.pdf found! Total chunks: {count}")
    else:
        print("❌ MRI_syllabus.pdf NOT found in index.")

except Exception as e:
    print(f"Error: {e}")
