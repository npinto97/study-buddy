import pickle
from pathlib import Path
from typing import List, Optional, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from flashrank import Ranker, RerankRequest
from pydantic import PrivateAttr

from study_buddy.config import logger, FAISS_INDEX_DIR
from study_buddy.vectorstore_pipeline.vector_store_builder import get_vector_store

BM25_INDEX_FILE = FAISS_INDEX_DIR / "bm25_retriever.pkl"

class HybridRetriever(BaseRetriever):
    """
    Hybrid Retriever that combines BM25 (keyword) and FAISS (vector) search,
    followed by a Re-ranking step using Flashrank.
    """
    k: int = 5
    fetch_k: int = 20  # Number of docs to fetch from each retriever before re-ranking
    
    _vector_store: Any = PrivateAttr()
    _bm25_retriever: Optional[Any] = PrivateAttr(default=None)
    _reranker: Any = PrivateAttr()

    def __init__(self, k: int = 5, fetch_k: int = 20, **kwargs):
        super().__init__(k=k, fetch_k=fetch_k, **kwargs)
        
        # Load Vector Store
        self._vector_store = get_vector_store(FAISS_INDEX_DIR)
        
        # Load BM25 Retriever
        if BM25_INDEX_FILE.exists():
            try:
                with open(BM25_INDEX_FILE, "rb") as f:
                    self._bm25_retriever = pickle.load(f)
                logger.info("BM25 Retriever loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load BM25 Retriever: {e}")
                self._bm25_retriever = None
        else:
            logger.warning(f"BM25 Index not found at {BM25_INDEX_FILE}. Hybrid search will fall back to vector-only.")

        # Initialize Re-ranker (Flashrank is lightweight and fast)
        # Using a small but effective model
        self._reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=str(FAISS_INDEX_DIR / "flashrank_cache"))

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve documents using Hybrid Search + Re-ranking.
        """
        logger.info(f"Hybrid Retrieval for query: '{query}'")
        
        # 1. Retrieve from Vector Store
        vector_docs = self._vector_store.similarity_search(query, k=self.fetch_k)
        logger.info(f"Vector search returned {len(vector_docs)} docs.")

        # 2. Retrieve from BM25 (if available)
        bm25_docs = []
        if self._bm25_retriever:
            bm25_docs = self._bm25_retriever.invoke(query)
            # Limit BM25 results to fetch_k
            bm25_docs = bm25_docs[:self.fetch_k]
            logger.info(f"BM25 search returned {len(bm25_docs)} docs.")

        # 3. Combine and Deduplicate
        # Use a dict keyed by source/content hash to deduplicate
        unique_docs = {}
        for doc in vector_docs + bm25_docs:
            # Create a unique key based on content or metadata
            # Assuming file_path and chunk index or just content hash
            doc_key = doc.page_content # Simple content-based deduplication
            if doc_key not in unique_docs:
                unique_docs[doc_key] = doc
        
        combined_docs = list(unique_docs.values())
        logger.info(f"Combined unique docs: {len(combined_docs)}")

        if not combined_docs:
            return []

        # 4. Re-ranking
        logger.info("Re-ranking documents...")
        
        # Prepare requests for Flashrank
        passages = [
            {"id": str(i), "text": doc.page_content, "meta": doc.metadata} 
            for i, doc in enumerate(combined_docs)
        ]
        
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self._reranker.rerank(rerank_request)
        
        # Sort results by score and take top k
        # Flashrank returns a list of dicts with 'score', 'id', 'text', 'meta'
        # We need to convert back to Documents
        
        final_docs = []
        for res in results[:self.k]:
            # Reconstruct Document
            doc = Document(
                page_content=res["text"],
                metadata=res["meta"]
            )
            # Optionally add score to metadata
            doc.metadata["relevance_score"] = res["score"]
            final_docs.append(doc)
            
        logger.info(f"Re-ranking complete. Returning top {len(final_docs)} docs.")
        return final_docs
