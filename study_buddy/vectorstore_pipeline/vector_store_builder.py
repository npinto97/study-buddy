import json
import os
from pathlib import Path
from time import sleep
from typing import Optional, Set
import torch
import pickle

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

from study_buddy.utils.embeddings import embeddings
from study_buddy.vectorstore_pipeline.document_loader import scan_directory_for_new_documents
from study_buddy.config import logger, PROCESSED_DOCS_FILE, FAISS_INDEX_DIR, PARSED_COURSES_DATA_FILE, TEMP_DOCS_FILE

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 64
EMBEDDING_BATCH_SIZE = 32
RATE_LIMIT_DELAY = 60
GPU_MEMORY_FRACTION = 0.85

BM25_INDEX_FILE = FAISS_INDEX_DIR / "bm25_retriever.pkl"

def optimize_gpu_memory():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        
        # Set a memory limit to avoid OOM
        memory_limit = int(total_memory * GPU_MEMORY_FRACTION)
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        
        logger.info(f"GPU Memory optimized:")
        logger.info(f"   - Memory limit: {memory_limit / 1e9:.1f}GB / {total_memory / 1e9:.1f}GB")
        logger.info(f"   - Batch size embeddings: {EMBEDDING_BATCH_SIZE}")
        
        torch.cuda.empty_cache()


def save_temp_docs(docs, hashes):
    """Save temporary documents and their hashes in case of failure."""
    try:
        # Crea la cartella se non esiste
        temp_dir = os.path.dirname(TEMP_DOCS_FILE)
        if temp_dir and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
            logger.info(f"Created directory: {temp_dir}")
        
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

        docs = [Document(**doc) for doc in data.get("docs", [])]
        hashes = set(data.get("hashes", []))

        logger.info("Temporary documents and hashes loaded successfully.")
        return docs, hashes

    except FileNotFoundError:
        return [], set()
    except Exception as e:
        logger.error(f"Error loading temporary documents: {e}")
        return [], set()


def get_embedding_dimensions(embeddings_instance) -> int:
    """Get embedding dimensions from HuggingFaceEmbeddings instance."""
    try:
        logger.info("Detecting embedding dimensions...")
        test_embedding = embeddings_instance.embed_query("test")
        dimensions = len(test_embedding)
        logger.info(f"Embedding dimensions: {dimensions}")
        
        return dimensions
        
    except Exception as e:
        logger.error(f"Error getting embedding dimensions: {e}")
        logger.warning("Using fallback dimensions: 1024 (BGE-M3 default)")
        return 1024


def initialize_faiss_store() -> Optional[FAISS]:
    """Initialize the FAISS vector store with GPU optimizations."""
    optimize_gpu_memory()
    
    faiss_file_path = Path(FAISS_INDEX_DIR)
    global vector_store

    if faiss_file_path.exists():
        logger.info(f"Loading FAISS vector store from {faiss_file_path}")
        try:
            vector_store = FAISS.load_local(
                str(faiss_file_path), 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS vector store loaded successfully.")

            new_docs, new_hashes = load_temp_docs()
            if not new_docs:
                new_docs, new_hashes = scan_directory_for_new_documents(
                    load_processed_hashes(PROCESSED_DOCS_FILE), 
                    PARSED_COURSES_DATA_FILE
                )

            if not new_docs:
                logger.info("No new documents found to update the vector store.")
                return vector_store

            save_temp_docs(new_docs, new_hashes)

            logger.info(f"Updating vector store with {len(new_docs)} new documents...")
            vector_store = index_documents(
                new_docs, 
                new_hashes, 
                load_processed_hashes(PROCESSED_DOCS_FILE), 
                vector_store
            )
            logger.info("Vector store updated successfully.")
            return vector_store

        except Exception as e:
            logger.error(f"Error loading FAISS vector store: {e}")
            return None

    else:
        logger.info(f"Creating new FAISS store at {faiss_file_path}")

        new_docs, new_hashes = load_temp_docs()
        if not new_docs:
            new_docs, new_hashes = scan_directory_for_new_documents(
                load_processed_hashes(PROCESSED_DOCS_FILE), 
                PARSED_COURSES_DATA_FILE
            )

        if not new_docs:
            logger.error("No documents found. FAISS cannot be initialized.")
            return None

        save_temp_docs(new_docs, new_hashes)

        embedding_dim = get_embedding_dimensions(embeddings)
        vector_store = FAISS(
            embedding_function=embeddings,
            index=IndexFlatL2(embedding_dim),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        
        vector_store = index_documents(
            new_docs, 
            new_hashes, 
            load_processed_hashes(PROCESSED_DOCS_FILE), 
            vector_store
        )
        logger.info("FAISS vector store created and populated successfully")
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
    """Save the hashes of processed documents."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(list(hashes), f, indent=4)
        logger.info("Processed hashes saved successfully.")
    except Exception as e:
        logger.error(f"Error saving processed hashes: {e}")


def add_documents_to_store(vector_store: FAISS, docs):
    """Add documents to vector store with GPU batch processing."""
    if not docs:
        logger.info("No documents to add to the vector store.")
        return

    try:
        total_docs = len(docs)
        logger.info(f"Processing {total_docs} documents on RTX 5080...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for i in range(0, total_docs, EMBEDDING_BATCH_SIZE):
            batch = docs[i:i + EMBEDDING_BATCH_SIZE]
            batch_size = len(batch)
            
            logger.info(f"Processing batch {i//EMBEDDING_BATCH_SIZE + 1}/{(total_docs-1)//EMBEDDING_BATCH_SIZE + 1} ({batch_size} docs)")
            
            vector_store.add_documents(batch)
            
            if torch.cuda.is_available() and i % (EMBEDDING_BATCH_SIZE * 4) == 0:
                torch.cuda.empty_cache()
        
        logger.info(f"All {total_docs} documents successfully indexed on GPU")
        _save_faiss_store(vector_store)
        
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


def _save_faiss_store(vector_store: FAISS):
    """Save the FAISS vector store."""
    faiss_file_path = Path(FAISS_INDEX_DIR)
    try:
        vector_store.save_local(str(faiss_file_path))
        logger.info(f"FAISS vector store saved to {faiss_file_path}")
    except Exception as e:
        logger.error(f"Error saving FAISS vector store: {e}")


def batch_process_documents(docs, batch_size=BATCH_SIZE):
    total_batches = (len(docs) - 1) // batch_size + 1
    logger.info(f"Processing {len(docs)} documents in {total_batches} batches (size: {batch_size})")
    
    for i in range(0, len(docs), batch_size):
        batch_num = i // batch_size + 1
        batch = docs[i:i + batch_size]
        logger.info(f"Batch {batch_num}/{total_batches}: {len(batch)} documents")
        yield batch


def index_documents(new_docs: list, new_hashes: set, processed_hashes: set, 
                   vector_store: Optional[FAISS] = None) -> Optional[FAISS]:
    
    logger.info(f"Starting GPU-accelerated document indexing...")
    logger.info(f"   - Total documents: {len(new_docs)}")
    logger.info(f"   - Chunk size: {CHUNK_SIZE}")
    logger.info(f"   - Batch size: {BATCH_SIZE}")
    logger.info(f"   - GPU batch size: {EMBEDDING_BATCH_SIZE}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    successfully_processed_hashes = set()
    total_chunks_processed = 0

    for batch_num, batch in enumerate(batch_process_documents(new_docs), 1):
        try:
            all_splits = text_splitter.split_documents(batch)
            chunk_count = len(all_splits)
            
            if all_splits:
                logger.info(f"Batch {batch_num}: {chunk_count} chunks to index...")
                
                add_documents_to_store(vector_store, all_splits)
                
                total_chunks_processed += chunk_count
                logger.info(f"Batch {batch_num} completed. Total chunks: {total_chunks_processed}")
                
                successfully_processed_hashes.update(new_hashes)

        except Exception as e:
            logger.error(f"Error indexing batch {batch_num}: {e}")
            logger.info(f"Waiting {RATE_LIMIT_DELAY} seconds before retrying...")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            sleep(RATE_LIMIT_DELAY)
            continue

    # Save processed hashes
    if successfully_processed_hashes:
        processed_hashes.update(successfully_processed_hashes)
        save_processed_hashes(PROCESSED_DOCS_FILE, processed_hashes)
        logger.info(f"Indexing completed. {total_chunks_processed} total chunks processed.")

    # Create and save BM25 retriever if we have documents
    if new_docs:
        all_splits = []
        # Re-collect all splits for BM25
        full_text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        all_splits = full_text_splitter.split_documents(new_docs)
        
        if all_splits:
            logger.info("Building BM25 index...")
            bm25_retriever = BM25Retriever.from_documents(all_splits)
            # Save BM25 retriever
            with open(BM25_INDEX_FILE, "wb") as f:
                pickle.dump(bm25_retriever, f)
    else:
        logger.info(f"Creating new FAISS store at {faiss_file_path}")

        new_docs, new_hashes = load_temp_docs()
        if not new_docs:
            new_docs, new_hashes = scan_directory_for_new_documents(
                load_processed_hashes(PROCESSED_DOCS_FILE), 
                PARSED_COURSES_DATA_FILE
            )

        if not new_docs:
            logger.error("No documents found. FAISS cannot be initialized.")
            return None

        save_temp_docs(new_docs, new_hashes)

        embedding_dim = get_embedding_dimensions(embeddings)
        vector_store = FAISS(
            embedding_function=embeddings,
            index=IndexFlatL2(embedding_dim),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        
        vector_store = index_documents(
            new_docs, 
            new_hashes, 
            load_processed_hashes(PROCESSED_DOCS_FILE), 
            vector_store
        )
        logger.info("FAISS vector store created and populated successfully")
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
    """Save the hashes of processed documents."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(list(hashes), f, indent=4)
        logger.info("Processed hashes saved successfully.")
    except Exception as e:
        logger.error(f"Error saving processed hashes: {e}")


def add_documents_to_store(vector_store: FAISS, docs):
    """Add documents to vector store with GPU batch processing."""
    if not docs:
        logger.info("No documents to add to the vector store.")
        return

    try:
        total_docs = len(docs)
        logger.info(f"Processing {total_docs} documents on RTX 5080...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for i in range(0, total_docs, EMBEDDING_BATCH_SIZE):
            batch = docs[i:i + EMBEDDING_BATCH_SIZE]
            batch_size = len(batch)
            
            logger.info(f"Processing batch {i//EMBEDDING_BATCH_SIZE + 1}/{(total_docs-1)//EMBEDDING_BATCH_SIZE + 1} ({batch_size} docs)")
            
            vector_store.add_documents(batch)
            
            if torch.cuda.is_available() and i % (EMBEDDING_BATCH_SIZE * 4) == 0:
                torch.cuda.empty_cache()
        
        logger.info(f"All {total_docs} documents successfully indexed on GPU")
        _save_faiss_store(vector_store)
        
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


def _save_faiss_store(vector_store: FAISS):
    """Save the FAISS vector store."""
    faiss_file_path = Path(FAISS_INDEX_DIR)
    try:
        vector_store.save_local(str(faiss_file_path))
        logger.info(f"FAISS vector store saved to {faiss_file_path}")
    except Exception as e:
        logger.error(f"Error saving FAISS vector store: {e}")


def batch_process_documents(docs, batch_size=BATCH_SIZE):
    total_batches = (len(docs) - 1) // batch_size + 1
    logger.info(f"Processing {len(docs)} documents in {total_batches} batches (size: {batch_size})")
    
    for i in range(0, len(docs), batch_size):
        batch_num = i // batch_size + 1
        batch = docs[i:i + batch_size]
        logger.info(f"Batch {batch_num}/{total_batches}: {len(batch)} documents")
        yield batch


def index_documents(new_docs: list, new_hashes: set, processed_hashes: set, 
                   vector_store: Optional[FAISS] = None) -> Optional[FAISS]:
    
    logger.info(f"Starting GPU-accelerated document indexing...")
    logger.info(f"   - Total documents: {len(new_docs)}")
    logger.info(f"   - Chunk size: {CHUNK_SIZE}")
    logger.info(f"   - Batch size: {BATCH_SIZE}")
    logger.info(f"   - GPU batch size: {EMBEDDING_BATCH_SIZE}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    successfully_processed_hashes = set()
    total_chunks_processed = 0

    for batch_num, batch in enumerate(batch_process_documents(new_docs), 1):
        try:
            all_splits = text_splitter.split_documents(batch)
            chunk_count = len(all_splits)
            
            if all_splits:
                logger.info(f"Batch {batch_num}: {chunk_count} chunks to index...")
                
                add_documents_to_store(vector_store, all_splits)
                
                total_chunks_processed += chunk_count
                logger.info(f"Batch {batch_num} completed. Total chunks: {total_chunks_processed}")
                
                successfully_processed_hashes.update(new_hashes)

        except Exception as e:
            logger.error(f"Error indexing batch {batch_num}: {e}")
            logger.info(f"Waiting {RATE_LIMIT_DELAY} seconds before retrying...")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            sleep(RATE_LIMIT_DELAY)
            continue

    # Save processed hashes
    if successfully_processed_hashes:
        processed_hashes.update(successfully_processed_hashes)
        save_processed_hashes(PROCESSED_DOCS_FILE, processed_hashes)
        logger.info(f"Indexing completed. {total_chunks_processed} total chunks processed.")

    # Create and save BM25 retriever if we have documents
    if new_docs:
        all_splits = []
        # Re-collect all splits for BM25
        full_text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        all_splits = full_text_splitter.split_documents(new_docs)
        
        if all_splits:
            logger.info("Building BM25 index...")
            bm25_retriever = BM25Retriever.from_documents(all_splits)
            # Save BM25 retriever
            with open(BM25_INDEX_FILE, "wb") as f:
                pickle.dump(bm25_retriever, f)
            logger.info(f"BM25 index saved to {BM25_INDEX_FILE}")

    return vector_store


def get_vector_store(faiss_file_path) -> Optional[FAISS]:
    """Retrieve the FAISS vector store."""
    vector_store = FAISS.load_local(
        str(faiss_file_path), 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    logger.info("FAISS vector store loaded successfully.")
    return vector_store


def get_gpu_memory_info():
    """Mostra informazioni memoria GPU."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        
        logger.info(f"GPU Memory Status:")
        logger.info(f"   - Allocated: {allocated:.2f}GB")
        logger.info(f"   - Reserved: {reserved:.2f}GB") 
        logger.info(f"   - Total: {total:.2f}GB")
        logger.info(f"   - Free: {total - reserved:.2f}GB")


vector_store: Optional[FAISS] = None

if __name__ == "__main__":
    initialize_faiss_store()