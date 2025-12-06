from study_buddy.utils.tools import VectorStoreRetriever

query = "Secondo le slide introduttive del corso MRI, quale problema risolvono i sistemi di Information Filtering e Recommender Systems?"

print(f"Query: {query}")
print("-" * 50)

try:
    retriever = VectorStoreRetriever()
    results, docs, paths = retriever.retrieve(query, k=6, min_score=0.1)
    
    print(f"Retrieved {len(docs)} documents.")
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1} (Score: N/A):")
        print(doc.page_content[:300] + "...")
        
    if "Information Overload" in results or "information overload" in results:
        print("\n✅ SUCCESS: 'Information Overload' found in retrieved context.")
    else:
        print("\n❌ FAILURE: 'Information Overload' NOT found in retrieved context.")

except Exception as e:
    print(f"Error: {e}")
