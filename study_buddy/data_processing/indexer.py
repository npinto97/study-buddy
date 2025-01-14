from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import json
from pathlib import Path
from study_buddy.config import EXTRACTED_TEXT_DIR, FAISS_INDEX_DIR


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

    def index_documents(self):
        """
        Load JSON files, extract text and metadata, and create or update a FAISS index.
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

        # Create or update the FAISS index
        embeddings = OpenAIEmbeddings()

        # Check if index already exists
        if Path(self.index_path).exists():
            print("Loading existing index...")
            vector_store = FAISS.load_local(self.index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            print("Creating a new index...")
            vector_store = FAISS(embeddings=embeddings)

        for batch_idx, batch in enumerate(self._batch_documents(documents)):
            print(f"Processing batch {batch_idx + 1} with {len(batch)} documents...")
            vector_store.add_documents(batch)  # Add new batch to the existing index
            vector_store.save_local(self.index_path)  # Save the updated index
            print(f"Batch {batch_idx + 1} indexed and saved to {self.index_path}")

        print("Indexing completed successfully!")


if __name__ == "__main__":
    json_dir = EXTRACTED_TEXT_DIR
    index_path = FAISS_INDEX_DIR / "index.faiss"

    indexer = Indexer(json_dir, index_path, batch_size=10)
    indexer.index_documents()
