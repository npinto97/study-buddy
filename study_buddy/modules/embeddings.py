from langchain_openai import OpenAIEmbeddings
from study_buddy.config import CONFIG, logger

logger.info(f"Initializing Embeddings with model: {CONFIG.embeddings.model}")

embeddings = OpenAIEmbeddings(model=CONFIG.embeddings.model)

logger.info(f"Embeddings with model {CONFIG.embeddings.model} successfully initialized.")
