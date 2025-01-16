from langchain_openai import ChatOpenAI
from study_buddy.config import CONFIG, logger

logger.info(f"Initializing LLM with model: {CONFIG.llm.model}")

llm = ChatOpenAI(model=CONFIG.llm.model)

def generate_response(messages):
    logger.debug(f"LLM invoked with messages: {messages}")
    response = llm.invoke(messages)
    logger.info(f"LLM response: {response.content}")
    return response.content
