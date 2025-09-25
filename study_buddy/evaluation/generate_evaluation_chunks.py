import os
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from study_buddy.vectorstore_pipeline.document_loader import scan_directory_for_new_documents
from study_buddy.config import logger, PARSED_COURSES_DATA_FILE

# Ensure these values are IDENTICAL to those used in 'study_buddy\vectorstore_pipeline\vector_store_builder.py'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

OUTPUT_DIR = Path("study_buddy/evaluation/chunked_documents")

def sanitize_filename(filename: str) -> str:
    """Removes invalid characters to create a safe filename."""
    # Removes the extension and problematic characters
    base_name = Path(filename).stem
    return re.sub(r'[\\/*?:"<>|]', "_", base_name)

def generate_chunks_for_evaluation():
    """
    Loads all documents, splits them into chunks, and saves each chunk
    to a separate text file with a unique ID. This is used to create
    the 'gold standard' evaluation dataset.
    """
    logger.info("Starting the chunk generation process for evaluation...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Chunks will be saved in: {OUTPUT_DIR.resolve()}")

    # Load ALL documents, ignoring the cache of already processed files.
    #  We pass an empty set to force the reprocessing of everything.
    all_docs, _ = scan_directory_for_new_documents(
        processed_hashes=set(),
        parsed_data_file=PARSED_COURSES_DATA_FILE
    )

    if not all_docs:
        logger.warning("No documents found. The process is terminating.")
        return

    logger.info(f"Found {len(all_docs)} documents to process.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    total_chunks_saved = 0
    for doc_index, doc in enumerate(all_docs):
        source_path = doc.metadata.get("file_path") or doc.metadata.get("source_url", f"external_doc_{doc_index}")
        base_filename = sanitize_filename(source_path)
        
        logger.info(f"Processing doc [{doc_index + 1}/{len(all_docs)}]: {Path(source_path).name}")

        chunks = text_splitter.split_documents([doc])

        for chunk_index, chunk in enumerate(chunks):
            # Create a unique ID for the chunk
            chunk_id = f"{base_filename}_chunk_{chunk_index + 1}"
            output_filename = f"{chunk_id}.txt"
            output_path = OUTPUT_DIR / output_filename

            # Save the chunk to a file
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"--- CHUNK METADATA ---\n")
                    f.write(f"CHUNK_ID: {chunk_id}\n")
                    f.write(f"SOURCE_FILE: {source_path}\n")
                    f.write(f"METADATA: {chunk.metadata}\n")
                    f.write("--- CONTENT ---\n\n")
                    f.write(chunk.page_content)
                total_chunks_saved += 1
            except Exception as e:
                logger.error(f"Could not save chunk {chunk_id}: {e}")

    logger.info(f"Process complete! Total chunks saved: {total_chunks_saved}")
    logger.info(f"You can now inspect the files in '{OUTPUT_DIR.resolve()}' to populate your evaluation dataset.")

if __name__ == "__main__":
    generate_chunks_for_evaluation()