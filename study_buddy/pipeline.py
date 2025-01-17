from study_buddy.document_loader import all_splits
from study_buddy.vector_store import add_documents_to_store
from study_buddy.vector_store import vector_store
from study_buddy.llm import generate_response
from langchain import hub
from typing import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START
from study_buddy.config import logger


add_documents_to_store(all_splits)

logger.info("Loading prompt from LangChain Hub...")
prompt = hub.pull("rlm/rag-prompt")
logger.info("Prompt loaded successfully.")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    logger.info(f"Retrieving context for question: {state['question']}")
    retrieved_docs = vector_store.similarity_search(state["question"])
    logger.info(f"Retrieved {len(retrieved_docs)} relevant documents.")
    return {"context": retrieved_docs}


def generate(state: State):
    logger.info(f"Generating answer for question: {state['question']}")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = generate_response(messages)
    logger.info("Answer generated successfully.")
    return {"answer": response}


logger.info("Building the state graph for the pipeline...")
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
logger.info("Pipeline compiled successfully.")
