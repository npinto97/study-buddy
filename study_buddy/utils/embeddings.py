from langchain_huggingface import HuggingFaceEmbeddings
from study_buddy.config import CONFIG, logger

logger.info(f"Initializing Embeddings with model: {CONFIG.embeddings.model}")

embeddings = HuggingFaceEmbeddings(model_name=CONFIG.embeddings.model)

logger.info(f"Embeddings with model {CONFIG.embeddings.model} successfully initialized.")
