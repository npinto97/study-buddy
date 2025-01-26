from study_buddy.modules.vector_store import get_vector_store

vector_store = get_vector_store()
retrieved_docs = vector_store.similarity_search("Conversational Recommender Systems", k=1)
print("Documenti trovati", retrieved_docs)