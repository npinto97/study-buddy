import json
from pathlib import Path
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from study_buddy.modules.embeddings import embeddings
from study_buddy.modules.document_loader import scan_directory_for_new_documents
from study_buddy.config import logger, CONFIG, PROCESSED_DOCS_FILE, RAW_DATA_DIR, SUPPORTED_EXTENSIONS, FAISS_INDEX_DIR

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

vector_store = None  # Variabile globale per mantenere il vector store


def initialize_vector_store(CONFIG):
    """
    Initialize the appropriate Vector Store based on the configuration.

    Args:
        CONFIG: Configuration object with vector_store.type.

    Returns:
        An initialized vector store instance.
    """
    logger.info(f"Initializing Vector Store of type: {CONFIG.vector_store.type}")

    if CONFIG.vector_store.type == "in_memory":
        return InMemoryVectorStore(embeddings)
    
    elif CONFIG.vector_store.type == "faiss":
        faiss_file_path = Path(FAISS_INDEX_DIR)
        if faiss_file_path.exists():
            logger.info(f"Loading FAISS vector store from {faiss_file_path}")
            try:
                return FAISS.load_local(str(faiss_file_path), embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                logger.error(f"Error loading FAISS vector store: {e}")
                raise
        else:
            logger.info(f"FAISS vector store not found at {faiss_file_path}. Creating a new one.")
            
            new_docs, _ = scan_directory_for_new_documents(RAW_DATA_DIR, SUPPORTED_EXTENSIONS, set())

            if not new_docs:
                logger.warning("Nessun documento trovato. FAISS non può essere inizializzato.")
                return None
            
            return FAISS.from_documents(new_docs, embeddings)
        

def is_vector_store_empty(vector_store):
    """
    Check if the Vector Store is empty.

    Args:
        vector_store: The vector store instance.

    Returns:
        bool: True if the vector store is empty, False otherwise.
    """
    try:
        return vector_store.similarity_search("", k=1) == []
    except Exception as e:
        logger.error(f"Error checking vector store: {e}")
        return True


def add_documents_to_store(vector_store, docs):
    """
    Adds documents to the Vector Store if it is empty.

    Args:
        vector_store: The vector store instance.
        docs: The documents to add.
    """
    if not docs:
        logger.info("No documents to add to the vector store.")
        return

    if is_vector_store_empty(vector_store):
        logger.info(f"Adding {len(docs)} documents to the vector store.")
        try:
            vector_store.add_documents(docs)
            logger.info("Documents successfully indexed.")

            # Save the vector store if it is FAISS
            if isinstance(vector_store, FAISS):
                faiss_file_path = Path(FAISS_INDEX_DIR)
                try:
                    vector_store.save_local(str(faiss_file_path))
                    logger.info(f"FAISS vector store saved to {faiss_file_path}")
                except Exception as e:
                    logger.error(f"Error saving FAISS vector store: {e}")
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
    else:
        logger.info("Vector store already contains documents. Skipping indexing.")


def load_processed_hashes(file_path: Path) -> set:
    """
    Load hashes of already processed documents.

    Args:
        file_path (Path): Path to the processed hashes file.

    Returns:
        set: A set of processed document hashes.
    """
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception as e:
            logger.error(f"Error loading processed hashes: {e}")
    return set()


def save_processed_hashes(file_path: Path, hashes: set):
    """
    Save the hashes of processed documents.

    Args:
        file_path (Path): Path to save the processed hashes.
        hashes (set): Set of hashes to save.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(list(hashes), f, indent=4)
        logger.info("Processed hashes saved successfully.")
    except Exception as e:
        logger.error(f"Error saving processed hashes: {e}")


def index_documents():
    """
    Indicizza nuovi documenti senza sovrascrivere il vector store esistente.
    """
    global vector_store  # Usa la variabile globale per mantenerlo persistente

    if vector_store is None:
        vector_store = initialize_vector_store(CONFIG)

    processed_hashes = load_processed_hashes(PROCESSED_DOCS_FILE)

    new_docs, new_hashes = scan_directory_for_new_documents(RAW_DATA_DIR, SUPPORTED_EXTENSIONS, processed_hashes)

    if not new_docs:
        logger.info("No new documents to process. Keeping the existing vector store.")
        return vector_store  # Restituisci il vector_store esistente

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


def get_vector_store():
    """Recupera il vector store inizializzato e popolato."""
    global vector_store  # Mantiene il vector store persistente

    if is_vector_store_empty(vector_store):
        logger.info("⚠️ Vector Store è vuoto! Lo stiamo caricando...")
        vector_store = index_documents()

    logger.info(f"📂 Vector Store caricato: {vector_store}")
    return vector_store
