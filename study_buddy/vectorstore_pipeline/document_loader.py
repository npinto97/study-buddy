# import nltk
# nltk.download('it_core_news_sm')
# nltk.download('averaged_perceptron_tagger_eng')
import hashlib
from pathlib import Path
import json
from langchain.schema import Document

from study_buddy.config import logger, SUPPORTED_EXTENSIONS, FILE_LOADERS, AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, RAW_DATA_DIR, METADATA_DIR, EXTRACTED_TEXT_DIR
from study_buddy.vectorstore_pipeline.audio_handler import transcribe_audio
from study_buddy.vectorstore_pipeline.video_handler import transcribe_video
from study_buddy.vectorstore_pipeline.external_resources_handler import (
    extract_text_from_url,
    extract_readme_from_repo,
    extract_transcript_from_youtube
)


def save_extracted_text(doc_hash: str, content: str):
    """
    Save extracted text to a file inside the `extracted_texts` folder.
    """
    EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = EXTRACTED_TEXT_DIR / f"{doc_hash}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Extracted text saved: {file_path}")


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


def scan_directory_for_new_documents(processed_hashes: set):
    """
    Scans a directory for unprocessed documents/external resources and loads them.

    Args:
        raw_data_dir (Path): Directory containing the raw documents.
        lesson_json_dir (Path): Directory contenente i file JSON delle lezioni.
        processed_hashes (set): Set of hashes for already processed documents.

    Returns:
        tuple: A list of new documents and a set of their hashes.
    """
    new_docs = []
    new_hashes = set()

    # Scansiona i file locali
    try:
        for filepath in RAW_DATA_DIR.rglob("*"):
            if filepath.is_file() and filepath.suffix.lower() in SUPPORTED_EXTENSIONS:
                doc_hash = compute_document_hash(filepath)

                if doc_hash not in processed_hashes:
                    logger.info(f"Found new document: {filepath}")
                    try:
                        documents = load_document(filepath)
                        for doc in documents:
                            save_extracted_text(doc_hash, doc.page_content)
                        new_docs.extend(documents)
                        new_hashes.add(doc_hash)
                    except Exception as e:
                        logger.error(f"Error processing document {filepath}: {e}")
    except Exception as e:
        logger.error(f"Errore nella scansione delle risorse locali: {e}")
        raise

    # Scansione di tutti i file JSON nella cartella delle lezioni
    try:
        for json_file in METADATA_DIR.rglob("*.json"):
            logger.info(f"Lettura file JSON: {json_file}")

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    lesson_data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Errore nel parsing di {json_file}: {e}")
                continue

            resources = lesson_data.get("external_resources", [])
            if not resources:
                logger.warning(f"Nessuna risorsa trovata in {json_file}")

            for resource in resources:
                url = resource.get("url")
                logger.info(f"Estrazione da: {url}")
                content = None

                try:
                    if "github.com" in url:
                        logger.info(f"[GitHub] Estrazione README da: {url}")
                        content = extract_readme_from_repo(url)
                    elif "youtube.com" in url or "youtu.be" in url:
                        logger.info(f"[YouTube] Estrazione trascrizione da: {url}")
                        content = extract_transcript_from_youtube(url)
                    else:
                        logger.info(f"[Web] Estrazione testo da: {url}")
                        content = extract_text_from_url(url)

                    if content:
                        doc_hash = hashlib.sha256(content.encode()).hexdigest()
                        if doc_hash not in processed_hashes:
                            metadata = {key: value for key, value in resource.items()}
                            new_docs.append(Document(page_content=content, metadata=metadata))
                            new_hashes.add(doc_hash)
                            save_extracted_text(doc_hash, content)
                            logger.info(f"Documento aggiunto: {url}")
                        else:
                            logger.info(f"Documento già processato: {url}")
                    else:
                        logger.warning(f"Nessun contenuto estratto da: {url}")
                except Exception as e:
                    logger.error(f"Errore nell'estrazione da {url}: {e}")

    except Exception as e:
        logger.error(f"Errore nella scansione della cartella metadata: {e}")

    logger.info(f"Nuovi documenti trovati: {len(new_docs)}")

    return new_docs, new_hashes
