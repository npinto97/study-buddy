from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from study_buddy.embeddings import embeddings
from study_buddy.config import CONFIG, logger
from study_buddy.document_loader import all_splits

logger.info(f"Initializing Vector Store of type: {CONFIG.vector_store.type}")

if CONFIG.vector_store.type == "in_memory":
    vector_store = InMemoryVectorStore(embeddings)
elif CONFIG.vector_store.type == "faiss":
    vector_store = FAISS.from_embeddings(embeddings)
else:
    raise ValueError(f"Vector store type '{CONFIG.vector_store.type}' not supported.")


def is_vector_store_empty():
    """Check if the Vector Store is empty."""
    return vector_store.similarity_search("", k=1) == []


def add_documents_to_store(docs):
    """Adds documents to the Vector Store only if they are not already present."""
    if is_vector_store_empty():
        logger.info(f"Adding {len(docs)} documents to vector store.")
        vector_store.add_documents(docs)
        logger.info("Documents successfully indexed.")
    else:
        logger.info("Vector store already contains documents. Skipping indexing.")


# Index only new documents if present
if all_splits:
    add_documents_to_store(all_splits)
    logger.info("New documents indexed successfully.")
else:
    logger.info("No new documents to index.")
