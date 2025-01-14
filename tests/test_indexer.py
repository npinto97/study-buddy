from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def test_index_loading(index_path):
    """
    Test if the FAISS index can be loaded successfully.
    """
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded successfully!")
        return vector_store
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None


if __name__ == "__main__":
    index_path = "./faiss_index"
    vector_store = test_index_loading(index_path)
    print(f"Number of documents in the index: {vector_store.index.ntotal}")
