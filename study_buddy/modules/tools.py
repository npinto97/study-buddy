from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from study_buddy.modules.vector_store import vector_store


@tool(response_format="content_and_artifact")
def retrieve_tool(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


web_search_tool = TavilySearchResults(max_results=2)
