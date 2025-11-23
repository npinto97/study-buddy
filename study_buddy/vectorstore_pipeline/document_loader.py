# import nltk
# nltk.download('it_core_news_sm')
# nltk.download('averaged_perceptron_tagger_eng')
import hashlib
from pathlib import Path
import json
from langchain_core.documents import Document

from study_buddy.config import (
    logger, SUPPORTED_EXTENSIONS, FILE_LOADERS, AUDIO_EXTENSIONS, VIDEO_EXTENSIONS,
    EXTRACTED_TEXT_DIR
)
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


def compute_document_hash(content: str) -> str:
    """Genera un hash SHA-256 per una stringa di contenuto."""
    return hashlib.sha256(content.encode()).hexdigest()


def compute_file_hash(filepath: Path) -> str:
    """Genera un hash SHA-256 per un file."""
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
    """Upload a document using the appropriate loader based on the extension."""
    file_extension = filepath.suffix.lower()

    if file_extension in SUPPORTED_EXTENSIONS:
        try:
            if file_extension in AUDIO_EXTENSIONS:
                return transcribe_audio(filepath)

            if file_extension in VIDEO_EXTENSIONS:
                return transcribe_video(filepath)

            loader_class = FILE_LOADERS[file_extension]
            loader = loader_class(str(filepath))
            return loader.load()

        except Exception as e:
            logger.error(f"Error loading document {filepath}: {e}")
            raise
    else:
        logger.warning(f"Unsupported file format: {file_extension}")
        raise ValueError(f"Unsupported file format: {file_extension}")


def scan_directory_for_new_documents(processed_hashes: set, parsed_data_file: Path):
    """Scans files in `parsed_data_path`, adds metadata, and saves transcripts."""
    new_docs = []
    new_hashes = set()

    if not parsed_data_file.exists():
        logger.error(f"File {parsed_data_file} non trovato!")
        return new_docs, new_hashes

    # Carica il JSON dei file elaborati
    with open(parsed_data_file, "r", encoding="utf-8") as f:
        parsed_data = json.load(f)
    
    print(f"\n🔍 DEBUG: Total entries in parsed_course_data.json: {len(parsed_data)}")
    
    external_count = sum(1 for e in parsed_data if e.get("type") == "external_resource")
    local_count = sum(1 for e in parsed_data if e.get("type") != "external_resource")
    print(f"   - External resources: {external_count}")
    print(f"   - Local files: {local_count}")
    
    for entry in parsed_data:
        metadata = {
            "course_name": entry.get("course_name", "Unknown Course"),
            "course_description": entry.get("course_description", "No description available"),
            "lesson_number": entry.get("lesson_number"),
            "lesson_title": entry.get("lesson_title"),
            "type": entry.get("type")
        }

        if entry["type"] == "external_resource":
            # Gestisce risorse esterne (YouTube, GitHub, siti web)
            url = entry.get("url")
            metadata["source_url"] = url  # Aggiunge l'URL ai metadati
            logger.info(f"Extraction from external resource: {url}")
            content = None

            try:
                if "github.com" in url:
                    content = extract_readme_from_repo(url)
                elif "youtube.com" in url or "youtu.be" in url:
                    content = extract_transcript_from_youtube(url)
                else:
                    content = extract_text_from_url(url)

                if content:
                    doc_hash = compute_document_hash(content)

                    if doc_hash not in processed_hashes:
                        new_docs.append(Document(page_content=content, metadata=metadata))
                        new_hashes.add(doc_hash)
                        save_extracted_text(doc_hash, content)  # solo per debug
                        logger.info(f"External document added: {url}")
                    else:
                        logger.info(f"External document already processed: {url}")
                else:
                    logger.warning(f"No content extracted from: {url}")

            except Exception as e:
                logger.error(f"Error in extraction from {url}: {e}")

        else:
            # Gestisce file locali
            filepath = Path(entry["path"])

            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                continue

            doc_hash = compute_file_hash(filepath)

            if doc_hash in processed_hashes:
                logger.info(f"Local document already processed: {filepath}")
                continue

            logger.info(f"Uploading new document: {filepath}")

            try:
                documents = load_document(filepath)
                for doc in documents:
                    # Aggiunge i metadati al documento
                    metadata["file_path"] = str(filepath)
                    
                    new_doc = Document(page_content=doc.page_content, metadata=metadata)
                    save_extracted_text(doc_hash, doc.page_content)
                    new_docs.append(new_doc)
                    new_hashes.add(doc_hash)

            except Exception as e:
                logger.error(f"Error in processing file {filepath}: {e}")

    logger.info(f"New docs found: {len(new_docs)}")

    return new_docs, new_hashes
