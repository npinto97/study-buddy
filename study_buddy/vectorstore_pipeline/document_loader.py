# import nltk
# nltk.download('it_core_news_sm')
# nltk.download('averaged_perceptron_tagger_eng')
import hashlib
from pathlib import Path

from study_buddy.config import logger, SUPPORTED_EXTENSIONS, FILE_LOADERS, AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from study_buddy.vectorstore_pipeline.audio_handler import transcribe_audio
from study_buddy.vectorstore_pipeline.video_handler import transcribe_video


def compute_document_hash(filepath: Path) -> str:
    """
    Generate a SHA-256 hash for the content of a document.

    Args:
        filepath (Path): Path to the file.

    Returns:
        str: SHA-256 hash of the file content.
    """
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {filepath}: {e}")
        raise


def load_document(filepath: Path):
    """
    Load a document using the appropriate loader based on its extension.

    Args:
        filepath (Path): Path to the file.

    Returns:
        list: List of documents loaded.
    """
    file_extension = filepath.suffix.lower()

    if file_extension in SUPPORTED_EXTENSIONS:
        try:
            if file_extension in AUDIO_EXTENSIONS:
                return transcribe_audio(filepath)
            
            if file_extension in VIDEO_EXTENSIONS:
                return transcribe_video(filepath)

            loader_class = FILE_LOADERS[file_extension]
            loader = loader_class(str(filepath))
            documents = loader.load()
            return documents

        except Exception as e:
            logger.error(f"Error loading document {filepath}: {e}")
            raise
    else:
        logger.warning(f"Unsupported file format: {file_extension}")
        raise ValueError(f"Unsupported file format: {file_extension}")


def scan_directory_for_new_documents(raw_data_dir: Path, processed_hashes: set):
    """
    Scans a directory for unprocessed documents and loads them.

    Args:
        raw_data_dir (Path): Directory containing the raw documents.
        processed_hashes (set): Set of hashes for already processed documents.

    Returns:
        tuple: A list of new documents and a set of their hashes.
    """
    new_docs = []
    new_hashes = set()

    try:
        for filepath in raw_data_dir.rglob("*"):
            if filepath.is_file() and filepath.suffix.lower() in SUPPORTED_EXTENSIONS:
                doc_hash = compute_document_hash(filepath)

                if doc_hash not in processed_hashes:
                    logger.info(f"Found new document: {filepath}")
                    try:
                        documents = load_document(filepath)
                        new_docs.extend(documents)
                        new_hashes.add(doc_hash)
                    except Exception as e:
                        logger.error(f"Error processing document {filepath}: {e}")

        logger.info(f"New documents to process: {len(new_docs)}")
    except Exception as e:
        logger.error(f"Error scanning directory {raw_data_dir}: {e}")
        raise

    return new_docs, new_hashes
