import json
import hashlib
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


class Indexer:
    def __init__(self, json_dir, index_path, batch_size=10):
        """
        Initialize the Indexer module.

        Args:
            json_dir (str): Path to the directory containing JSON files.
            index_path (str): Path to save the FAISS index.
            batch_size (int): Number of documents per batch for indexing.
        """
        self.json_dir = Path(json_dir)
        self.index_path = index_path
        self.batch_size = batch_size

    def _batch_documents(self, documents):
        """
        Divide documents into smaller batches.

        Args:
            documents (list): List of Document objects.

        Returns:
            list: A list of batches, each containing a subset of documents.
        """
        for i in range(0, len(documents), self.batch_size):
            yield documents[i:i + self.batch_size]

    def _load_existing_index(self, embeddings):
        """
        Load the existing FAISS index if it exists, otherwise return None.

        Args:
            embeddings (OpenAIEmbeddings): Embedding model to use with the FAISS index.

        Returns:
            FAISS: Loaded FAISS index or None if it doesn't exist.
        """
        if Path(self.index_path).exists():
            print("Loading existing index...")
            return FAISS.load_local(self.index_path, embeddings, allow_dangerous_deserialization=True)
        return None

    def _calculate_hash(self, document):
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

    def _filter_duplicates(self, documents, vector_store):
        """
        Filter out duplicate documents based on their hash, including those already in the vector_store.

        Args:
            documents (list): List of Document objects.
            vector_store (FAISS): Existing FAISS vector store.

        Returns:
            list: List of unique Document objects.
        """
        seen_hashes = set()

        # Check hashes of documents already in the vector_store
        if vector_store:
            print("Fetching existing hashes from vector store...")
            for doc in vector_store.similarity_search("", k=vector_store.index.ntotal):
                doc_hash = self._calculate_hash(doc)
                seen_hashes.add(doc_hash)

        # Filter out duplicates in the new documents
        unique_documents = []
        for doc in documents:
            doc_hash = self._calculate_hash(doc)
            if doc_hash not in seen_hashes:
                unique_documents.append(doc)
                seen_hashes.add(doc_hash)

        return unique_documents

    def index_documents(self):
        """
        Load JSON files, extract text and metadata, and create a FAISS index with batching.
        """
        documents = []
        for json_file in self.json_dir.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    documents.append(
                        Document(
                            page_content=data["content"],
                            metadata=data["metadata"]
                        )
                    )
            except Exception as e:
                print(f"Error processing file {json_file}: {e}")
                continue  # Skip problematic files

        # Create embeddings and load or initialize FAISS index
        embeddings = OpenAIEmbeddings()  # model="text-embedding-3-small"
        vector_store = self._load_existing_index(embeddings)

        # Filter duplicates, including those already in the vector_store
        documents = self._filter_duplicates(documents, vector_store)
        print(f"Filtered down to {len(documents)} unique documents.")

        for batch_idx, batch in enumerate(self._batch_documents(documents)):
            print(f"Processing batch {batch_idx + 1} with {len(batch)} documents...")

            if vector_store:
                # Add documents to existing index
                vector_store.add_documents(batch)
            else:
                # Create a new index
                vector_store = FAISS.from_documents(batch, embeddings)

            # Save the updated index
            vector_store.save_local(self.index_path)
            print(f"Batch {batch_idx + 1} indexed and saved to {self.index_path}")

        print("Indexing completed successfully!")


if __name__ == "__main__":
    json_dir = "./processed_data"
    index_path = "./faiss_index"

    # Set batch size to a reasonable number to prevent rate limits
    indexer = Indexer(json_dir, index_path, batch_size=10)
    indexer.index_documents()
