from study_buddy.modules.vector_store import get_vector_store

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

vector_store = get_vector_store()
# # retrieved_docs = vector_store.similarity_search("Linked open data", k=1)
# retrieved_docs_1 = vector_store.similarity_search("Knowledge aware RS", k=1)
# retrieved_docs_2 = vector_store.similarity_search("Large language models", k=1)
# retrieved_docs_3 = vector_store.similarity_search("Netflix", k=1)
# docs = [retrieved_docs_1, retrieved_docs_2, retrieved_docs_3]
# for i, doc in enumerate(docs, 1):
#     print(f"Retrieved docs {i}: {doc}")


retrieved_docs = vector_store.similarity_search("Giovanni semeraro", k=2)
print(f"Retrieved docs: {retrieved_docs}")
