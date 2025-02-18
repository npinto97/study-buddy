import json
import os
from pathlib import Path
from time import sleep
from typing import Optional, Set

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from study_buddy.utils.embeddings import embeddings
from study_buddy.vectorstore_pipeline.document_loader import scan_directory_for_new_documents
from study_buddy.config import logger, PROCESSED_DOCS_FILE, FAISS_INDEX_DIR, PARSED_COURSES_DATA_FILE, TEMP_DOCS_FILE

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 200  # Numero di documenti per batch
RATE_LIMIT_DELAY = 60  # Secondi di attesa se viene raggiunto il TPM massimo


def save_temp_docs(docs, hashes):
    """Save temporary documents and their hashes in case of failure."""
    try:
        with open(TEMP_DOCS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "docs": [doc.dict() for doc in docs],
                "hashes": list(hashes)
            }, f, indent=4)
        logger.info("Temporary documents and hashes saved successfully.")
    except Exception as e:
        logger.error(f"Error saving temporary documents: {e}")


def load_temp_docs():
    """Load temporary documents and their hashes if they exist."""
    try:
        with open(TEMP_DOCS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Temporary documents and hashes loaded successfully.")
        return data.get("docs", []), set(data.get("hashes", []))
    except FileNotFoundError:
        return [], set()
    except Exception as e:
        logger.error(f"Error loading temporary documents: {e}")
        return [], set()


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

            # Prova a caricare i documenti temporanei
            new_docs, new_hashes = load_temp_docs()
            if not new_docs:
                new_docs, new_hashes = scan_directory_for_new_documents(load_processed_hashes(PROCESSED_DOCS_FILE), PARSED_COURSES_DATA_FILE)

            if not new_docs:
                logger.info("No new documents found to update the vector store.")
                return vector_store

            save_temp_docs(new_docs, new_hashes)  # se fallisce l'aggiornamento dell'indice, non devo aspettare altre X ore... :-(

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

        new_docs, new_hashes = load_temp_docs()
        if not new_docs:
            new_docs, new_hashes = scan_directory_for_new_documents(load_processed_hashes(PROCESSED_DOCS_FILE), PARSED_COURSES_DATA_FILE)

        # If there are no documents to process, log the error
        if not new_docs:
            logger.error("No documents found. FAISS cannot be initialized.")
            return None

        save_temp_docs(new_docs, new_hashes)

        vector_store = FAISS.from_documents(new_docs, embeddings)
        vector_store = index_documents(load_temp_docs(), new_hashes, load_processed_hashes(PROCESSED_DOCS_FILE), vector_store)
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


def batch_process_documents(docs, batch_size=BATCH_SIZE):
    """Generator that yields documents in batches."""
    for i in range(0, len(docs), batch_size):
        yield docs[i:i + batch_size]


def index_documents(new_docs: list, new_hashes: set, processed_hashes: set, vector_store: Optional[FAISS] = None) -> Optional[FAISS]:
    """
    Index new documents in batches to avoid exceeding TPM limits.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    successfully_processed_hashes = set()

    for batch in batch_process_documents(new_docs):
        all_splits = text_splitter.split_documents(batch)

        if all_splits:
            try:
                add_documents_to_store(vector_store, all_splits)
                logger.info(f"Indexed {len(all_splits)} document chunks successfully.")

                successfully_processed_hashes.update(new_hashes)
            except Exception as e:
                logger.error(f"Error indexing batch: {e}")
                logger.info(f"Waiting {RATE_LIMIT_DELAY} seconds before retrying...")
                sleep(RATE_LIMIT_DELAY)  # Attendi per evitare il rate limit

                # Continua il loop senza salvare gli hash prematuramente
                continue

    # Dopo il completamento di tutti i batch, aggiorniamo gli hash definitivamente
    if successfully_processed_hashes:
        processed_hashes.update(successfully_processed_hashes)
        save_processed_hashes(PROCESSED_DOCS_FILE, processed_hashes)

    # Remove temporary file after successful indexing
    if os.path.exists(TEMP_DOCS_FILE):
        os.remove(TEMP_DOCS_FILE)
        logger.info("Temporary documents file deleted successfully.")

    return vector_store


def get_vector_store(faiss_file_path) -> Optional[FAISS]:
    """Retrieve the initialized and populated FAISS vector store."""
    vector_store = FAISS.load_local(str(faiss_file_path), embeddings, allow_dangerous_deserialization=True)
    logger.info("FAISS vector store loaded successfully.")

    return vector_store


# Global vector store instance
vector_store: Optional[FAISS] = None
