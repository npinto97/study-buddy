# `vectorstore_pipeline/`

This directory contains the pipeline responsible for processing study materials and building or updating the **FAISS vector store**. This vector store is a core component of the RAG (Retrieval-Augmented Generation) system, enabling the AI to retrieve and synthesize information from the provided course content.

The pipeline is designed to be **efficient and scalable**, handling various document types, including local files and external resources. It also features **GPU acceleration** for faster embedding generation and an **incremental update** mechanism to process only new or modified content.


## Main Goal

The primary function of this pipeline is to convert raw course materials (like PDFs, audio, video, and web pages) into a searchable FAISS index. This index stores vector embeddings of the content, allowing for semantic search capabilities.

The entire process is automated and can be initiated with a single command. The system automatically detects new documents and updates the index without rebuilding it from scratch.


## How to Run

To build the vector store from scratch or to update it with new course materials, simply run the main script:

```bash
python update_faiss_index.py
```

This script will handle the entire process:

1.  **Parse Metadata:** It first scans the `metadata` directory to parse all course and lesson information from JSON files.
2.  **Identify Changes:** It compares the hashes of the files to be processed with a list of already-processed documents, identifying only the new or modified materials.
3.  **Process Documents:** For each new document, the system determines the file type and uses the appropriate handler (e.g., `document_loader`, `audio_handler`, `video_handler`, `external_resources_handler`) to extract its content. This includes:
      * **Transcribing** audio and video files.
      * **Extracting text** from PDFs, websites, and GitHub repositories.
4.  **Chunk and Embed:** The extracted text is then split into smaller, manageable chunks. These chunks are converted into numerical vector embeddings using a HuggingFace model, with optimizations for **GPU batch processing**.
5.  **Update FAISS:** The new embeddings are added to the FAISS index, either creating a new one if it doesn't exist or incrementally updating the existing store.

After completion, the updated FAISS index and a list of processed document hashes are saved to disk, ensuring that future runs only process new content.


## Components of the Pipeline

  - `update_faiss_index.py`: The main entry point. It orchestrates the entire process of parsing metadata and updating the vector store.
  - `parse_course_metadata.py`: Responsible for reading and parsing the course and lesson metadata from JSON files. It creates a structured list of all documents to be processed.
  - `document_loader.py`: Handles the loading and preprocessing of various file types, including local documents and external resources. It uses file hashes to manage which documents have already been processed.
  - `audio_handler.py`: Contains functions to transcribe audio files (`.mp3`, `.wav`) into text using the Whisper model.
  - `video_handler.py`: Transcribes video files (`.mp4`, `.mov`) by first extracting the audio and then using the Whisper model for transcription.
  - `external_resources_handler.py`: Manages the extraction of content from external sources like YouTube videos (by fetching transcripts), GitHub repositories (by getting READMEs), and general websites.
  - `vector_store_builder.py`: The core component for building and updating the FAISS index. It manages document chunking, GPU-accelerated embedding, and saving the final index. It also includes functions for optimizing GPU memory usage.