import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from study_buddy.utils.embeddings import embeddings
from study_buddy.vectorstore_pipeline.document_loader import scan_directory_for_new_documents
from study_buddy.config import logger, PROCESSED_DOCS_FILE, FAISS_INDEX_DIR
from typing import Optional, Set

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def initialize_faiss_store() -> Optional[FAISS]:
    """
    Initialize the FAISS vector store or update an existing one.

    Returns:
        FAISS instance or None if initialization fails.
    """
    faiss_file_path = Path(FAISS_INDEX_DIR)
    global vector_store

    # If the vector store file exists, load it
    if faiss_file_path.exists():
        logger.info(f"Loading FAISS vector store from {faiss_file_path}")
        try:
            vector_store = FAISS.load_local(str(faiss_file_path), embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS vector store loaded successfully.")

            # Scan for new documents
            new_docs, new_hashes = scan_directory_for_new_documents(load_processed_hashes(PROCESSED_DOCS_FILE))
            if not new_docs:
                logger.info("No new documents found to update the vector store.")
                return vector_store

            logger.info(f"Updating vector store with {len(new_docs)} new documents...")
            vector_store = index_documents(new_docs, new_hashes, load_processed_hashes(PROCESSED_DOCS_FILE), vector_store)
            logger.info("Vector store updated successfully.")

            return vector_store

        except Exception as e:
            logger.error(f"Error loading FAISS vector store: {e}")
            return None

    # If the file does not exist, create and populate it
    else:
        logger.info(f"FAISS store not found at {faiss_file_path}. Attempting to create and populate it.")

        # Scan for any documents
        new_docs, new_hashes = scan_directory_for_new_documents(load_processed_hashes(PROCESSED_DOCS_FILE))

        # If there are no documents to process, log the error
        if not new_docs:
            logger.error("No documents found. FAISS cannot be initialized.")
            return None

        vector_store = FAISS.from_documents(new_docs, embeddings)

        vector_store = index_documents(new_docs, new_hashes, load_processed_hashes(PROCESSED_DOCS_FILE), vector_store)
        logger.info("FAISS vector store created and populated with documents.")

        return vector_store


def load_processed_hashes(file_path: Path) -> Set[str]:
    """Load hashes of already processed documents."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()
    except Exception as e:
        logger.error(f"Error loading processed hashes: {e}")
        return set()


def save_processed_hashes(file_path: Path, hashes: Set[str]):
    """Save the hashes of processed documents, creating the directory if needed."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(list(hashes), f, indent=4)
        logger.info("Processed hashes saved successfully.")
    except Exception as e:
        logger.error(f"Error saving processed hashes: {e}")


# def is_vector_store_empty(vector_store: FAISS) -> bool:
#     """Check if the vector store is empty."""
#     try:
#         return not vector_store.similarity_search("", k=1)
#     except Exception as e:
#         logger.error(f"Error checking vector store: {e}")
#         return True


def add_documents_to_store(vector_store: FAISS, docs):
    """
    Add documents to the vector store if it is empty.

    Args:
        vector_store: The vector store instance.
        docs: List of documents to add.
    """
    if not docs:
        logger.info("No documents to add to the vector store.")
        return

    try:
        vector_store.add_documents(docs)
        logger.info("Documents successfully indexed.")
        _save_faiss_store(vector_store)
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")


def _save_faiss_store(vector_store: FAISS):
    """Save the FAISS vector store."""
    faiss_file_path = Path(FAISS_INDEX_DIR)
    try:
        vector_store.save_local(str(faiss_file_path))
        logger.info(f"FAISS vector store saved to {faiss_file_path}")
    except Exception as e:
        logger.error(f"Error saving FAISS vector store: {e}")


def index_documents(new_docs: list, new_hashes: set, processed_hashes: set, vector_store: Optional[FAISS] = None) -> Optional[FAISS]:
    """
    Index new documents without overwriting the existing vector store.

    Args:
        new_docs: List of new documents to index.
        processed_hashes: Set of hashes that have already been processed.
        vector_store: Optional; an existing vector store instance to update.

    Returns:
        The updated or newly created vector store.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_splits = text_splitter.split_documents(new_docs)

    if all_splits:
        add_documents_to_store(vector_store, all_splits)
        logger.info(f"Indexed {len(all_splits)} document chunks successfully.")
        processed_hashes.update(new_hashes)
        save_processed_hashes(PROCESSED_DOCS_FILE, processed_hashes)
    else:
        logger.info("No document chunks generated from the new documents.")

    return vector_store


def get_vector_store(faiss_file_path) -> Optional[FAISS]:
    """Retrieve the initialized and populated FAISS vector store."""
    vector_store = FAISS.load_local(str(faiss_file_path), embeddings, allow_dangerous_deserialization=True)
    logger.info("FAISS vector store loaded successfully.")

    return vector_store


# Global vector store instance
vector_store: Optional[FAISS] = None
