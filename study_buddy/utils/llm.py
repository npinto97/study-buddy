from langchain_openai import ChatOpenAI
from study_buddy.config import CONFIG, logger

logger.info(f"Initializing LLM with model: {CONFIG.llm.model}")

llm = ChatOpenAI(model=CONFIG.llm.model)
