from study_buddy.agent import compiled_graph
from study_buddy.config import logger
import sys

# Query from Scenario 1
query = "Secondo le slide introduttive del corso MRI, quale problema risolvono i sistemi di Information Filtering e Recommender Systems?"

print(f"Testing retrieval for query: {query}")

try:
    # Use the retrieve_knowledge tool directly or simulate the graph
    # For now, let's try to use the vector store directly if possible, 
    # but the graph is better to test the full chain.
    # However, graph is async.
    
    # Let's import the tool directly
    from study_buddy.utils.tools import retrieve_knowledge
    
    print("Calling retrieve_knowledge...")
    result = retrieve_knowledge(query)
    print("\n--- Retrieval Result ---")
    print(result)
    
except Exception as e:
    print(f"Error: {e}")
