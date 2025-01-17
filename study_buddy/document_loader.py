# import nltk
# nltk.download('it_core_news_sm')
# nltk.download('averaged_perceptron_tagger_eng')
import hashlib
import json
from pathlib import Path
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, PDFPlumberLoader
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from study_buddy.config import RAW_DATA_DIR, PROCESSED_DOCS_FILE, SUPPORTED_EXTENSIONS, logger


FILE_LOADERS = {
    ".pdf": PDFPlumberLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": DoclingLoader
}


def compute_document_hash(filepath: Path) -> str:
    """Generate a SHA-256 hash for the content of a document."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_processed_hashes() -> set:
    """Load hashes of already processed documents."""
    if PROCESSED_DOCS_FILE.exists():
        with open(PROCESSED_DOCS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_processed_hashes(hashes: set):
    """Save the hashes of processed documents."""
    with open(PROCESSED_DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(hashes), f, indent=4)


def load_document(filepath: Path):
    """Load a document using the appropriate loader based on its extension."""
    file_extension = filepath.suffix.lower()

    if file_extension in SUPPORTED_EXTENSIONS:
        loader_class = FILE_LOADERS[file_extension]
        loader = loader_class(str(filepath))
        documents = loader.load()

        return documents
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    

logger.info("Loading previously processed document hashes...")
processed_hashes = load_processed_hashes()
logger.info(f"Loaded {len(processed_hashes)} processed document hashes.")


logger.info("Scanning directory for new documents...")
new_docs = []
new_hashes = set()


for filepath in RAW_DATA_DIR.rglob("*"):
    if filepath.is_file() and filepath.suffix.lower() in SUPPORTED_EXTENSIONS:
        doc_hash = compute_document_hash(filepath)

        if doc_hash not in processed_hashes:
            logger.info(f"Loading new document: {filepath}")

            documents = load_document(filepath)

            for doc in documents:
                new_docs.append(doc)
            new_hashes.add(doc_hash)

logger.info(f"New documents to process: {len(new_docs)}")

if not new_docs:
    logger.info("No new documents to process. Exiting.")
    all_splits = []
else:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(new_docs)

    logger.info(f"Split documents into {len(all_splits)} chunks.")

processed_hashes.update(new_hashes)
save_processed_hashes(processed_hashes)
