from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from study_buddy.embeddings import embeddings
from study_buddy.config import CONFIG, logger

logger.info(f"Initializing Vector Store of type: {CONFIG.vector_store.type}")

if CONFIG.vector_store.type == "in_memory":
    vector_store = InMemoryVectorStore(embeddings)
elif CONFIG.vector_store.type == "faiss":
    vector_store = FAISS.from_embeddings(embeddings)
else:
    raise ValueError(f"Vector store type '{CONFIG.vector_store.type}' not supported.")


def add_documents_to_store(docs):
    logger.info(f"Adding {len(docs)} documents to vector store.")
    vector_store.add_documents(docs)
    logger.info("Documents successfully indexed.")