import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from study_buddy.utils.retriever import HybridRetriever
from study_buddy.config import logger

def test_retrieval():
    print("Initializing HybridRetriever...")
    retriever = HybridRetriever(k=5, fetch_k=20)
    
    queries = [
        "According to the MRI_syllabus.pdf, who is the professor for the 'Metodi per il Ritrovamento dell'Informazione' course?",
        "Find the email address for Professor Giovanni Semeraro in the SIIA_syllabus.pdf.",
        "What are the office hours for Professor Pasquale Lops as listed in the MRI syllabus?",
        "What are the five criteria for the final grade attribution in the SIIA course syllabus?"
    ]
    
    for query in queries:
        print(f"\n--- Query: {query} ---")
        docs = retriever.invoke(query)
        print(f"Retrieved {len(docs)} documents.")
        for i, doc in enumerate(docs):
            print(f"[{i+1}] Source: {doc.metadata.get('source', 'Unknown')} | Score: {doc.metadata.get('relevance_score', 'N/A')}")
            print(f"    Content snippet: {doc.page_content[:200]}...")

if __name__ == "__main__":
    test_retrieval()
