import sys
import os
sys.path.append(os.getcwd())

from study_buddy.vectorstore_pipeline.vector_store_builder import get_vector_store
from study_buddy.config import FAISS_INDEX_DIR

# Load FAISS vector store
print(f"Loading FAISS index from: {FAISS_INDEX_DIR}")
vector_store = get_vector_store(FAISS_INDEX_DIR)

print(f"Total documents in index: {vector_store.index.ntotal}")

# Get a sample of documents
print("\n" + "="*80)
print("SAMPLE DOCUMENTS (first 10):")
print("="*80)

docstore = vector_store.docstore._dict
for i, (doc_id, doc) in enumerate(list(docstore.items())[:10]):
    print(f"\nDoc {i+1}:")
    print(f"  ID: {doc_id}")
    print(f"  Metadata: {doc.metadata}")
    print(f"  Content preview: {doc.page_content[:200]}...")

# Search for documents containing "semeraro" and "email" or "@"
print("\n" + "="*80)
print("SEARCHING FOR SEMERARO + EMAIL/CONTACT INFO:")
print("="*80)

semeraro_docs = []
for doc_id, doc in docstore.items():
    content_lower = doc.page_content.lower()
    if 'semeraro' in content_lower:
        # Check if it contains email-like patterns
        if '@' in doc.page_content or 'mail' in content_lower or 'email' in content_lower or 'contatto' in content_lower:
            semeraro_docs.append(doc)

print(f"\nFound {len(semeraro_docs)} documents with Semeraro + potential contact info")

for i, doc in enumerate(semeraro_docs[:5]):
    print(f"\n--- Document {i+1} ---")
    print(f"Source: {doc.metadata.get('file_path', 'Unknown')}")
    print(f"Type: {doc.metadata.get('type', 'Unknown')}")
    print(f"Content:\n{doc.page_content}")
    print("-" * 80)
