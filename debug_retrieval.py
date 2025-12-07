from langchain_community.vectorstores import FAISS
from study_buddy.utils.embeddings import get_embeddings
from study_buddy.config import FAISS_INDEX_DIR

def debug_retrieval():
    try:
        print("Loading vector store...")
        vector_store = FAISS.load_local(
            str(FAISS_INDEX_DIR), 
            get_embeddings(), 
            allow_dangerous_deserialization=True
        )
        print(f"Index loaded. Total docs: {vector_store.index.ntotal}")
        
        query = "Quali sono gli orari di ricevimento del Professor Pasquale Lops elencati nel programma MRI?"
        print(f"\nQuery: {query}")
        
        results = vector_store.similarity_search_with_score(query, k=5)
        
        print(f"\nFound {len(results)} results:")
        for doc, score in results:
            source = doc.metadata.get("source", "Unknown")
            print(f"\n--- Score: {score:.4f} ---")
            print(f"Source: {source}")
            print(f"Content Preview: {doc.page_content[:200]}...")
            
            if "MRI_syllabus.pdf" in source:
                print("✅ SENTINEL MATCH: MRI_syllabus.pdf found in top results!")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_retrieval()
