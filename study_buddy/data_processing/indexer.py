import json
import hashlib
import logging
from pathlib import Path
from typing import List, Generator, Optional, Set
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from study_buddy.config import EXTRACTED_TEXT_DIR, FAISS_INDEX_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Indexer:
    def __init__(self, json_dir: Path, index_path: Path, batch_size: int = 10, model: str = "text-embedding-3-small"):
        """
        Initialize the Indexer module.

        Args:
            json_dir (Path): Path to the directory containing JSON files.
            index_path (Path): Path to save the FAISS index.
            batch_size (int): Number of documents per batch for indexing.
            model (str): Name of the embedding model to use.
        """
        self.json_dir = json_dir
        self.index_path = index_path
        self.batch_size = batch_size
        self.model = model

    def _batch_documents(self, documents: List[Document]) -> Generator[List[Document], None, None]:
        """
        Divide documents into smaller batches.

        Args:
            documents (List[Document]): List of Document objects.

        Yields:
            List[Document]: A batch of documents.
        """
        for i in range(0, len(documents), self.batch_size):
            yield documents[i:i + self.batch_size]

    def _load_existing_index(self, embeddings: OpenAIEmbeddings) -> Optional[FAISS]:
        """
        Load the existing FAISS index if it exists, otherwise return None.

        Args:
            embeddings (OpenAIEmbeddings): Embedding model to use with the FAISS index.

        Returns:
            Optional[FAISS]: Loaded FAISS index or None if it doesn't exist.
        """
        if self.index_path.exists():
            logging.info("Loading existing index...")
            return FAISS.load_local(str(self.index_path), embeddings, allow_dangerous_deserialization=True)
        return None

    def _calculate_hash(self, document: Document) -> str:
        """
        Calculate a SHA256 hash of the document's content and metadata.

        Args:
            document (Document): Document object.

        Returns:
            str: SHA256 hash of the document.
        """
        content = document.page_content
        metadata = str(document.metadata)
        return hashlib.sha256((content + metadata).encode("utf-8")).hexdigest()

    def _filter_duplicates(self, documents: List[Document], vector_store: Optional[FAISS]) -> List[Document]:
        """
        Filter out duplicate documents based on their hash, including those already in the vector_store.

        Args:
            documents (List[Document]): List of Document objects.
            vector_store (Optional[FAISS]): Existing FAISS vector store.

        Returns:
            List[Document]: List of unique Document objects.
        """
        seen_hashes: Set[str] = set()

        # Check hashes of documents already in the vector_store
        if vector_store:
            logging.info("Fetching existing hashes from vector store...")
            for doc in vector_store.similarity_search("", k=vector_store.index.ntotal):
                seen_hashes.add(self._calculate_hash(doc))

        # Filter out duplicates in the new documents
        unique_documents = []
        for doc in documents:
            doc_hash = self._calculate_hash(doc)
            if doc_hash not in seen_hashes:
                unique_documents.append(doc)
                seen_hashes.add(doc_hash)

        return unique_documents

    def _load_documents(self) -> List[Document]:
        """
        Load JSON files and extract text and metadata as Document objects.

        Returns:
            List[Document]: List of Document objects loaded from JSON files.
        """
        documents = []
        for json_file in self.json_dir.rglob("*.json"):
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    documents.append(Document(page_content=data["content"], metadata=data["metadata"]))
            except Exception as e:
                logging.error(f"Error processing file {json_file}: {e}")
        return documents

    def index_documents(self):
        """
        Load JSON files, extract text and metadata, and create a FAISS index with batching.
        """
        logging.info("Loading documents...")
        documents = self._load_documents()

        # Create embeddings and load or initialize FAISS index
        embeddings = OpenAIEmbeddings(model=self.model)
        vector_store = self._load_existing_index(embeddings)

        # Filter duplicates, including those already in the vector_store
        documents = self._filter_duplicates(documents, vector_store)
        logging.info(f"Filtered down to {len(documents)} unique documents.")

        for batch_idx, batch in enumerate(self._batch_documents(documents), start=1):
            logging.info(f"Processing batch {batch_idx} with {len(batch)} documents...")

            if vector_store:
                vector_store.add_documents(batch)
            else:
                vector_store = FAISS.from_documents(batch, embeddings)

            # Save the updated index
            vector_store.save_local(str(self.index_path))
            logging.info(f"Batch {batch_idx} indexed and saved to {self.index_path}")

        logging.info("Indexing completed successfully!")


if __name__ == "__main__":
    json_dir = Path(EXTRACTED_TEXT_DIR)
    index_path = Path(FAISS_INDEX_DIR)

    # Set batch size to a reasonable number to prevent rate limits
    indexer = Indexer(json_dir=json_dir, index_path=index_path, batch_size=10)
    indexer.index_documents()
