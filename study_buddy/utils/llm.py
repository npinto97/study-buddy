from langchain_together import ChatTogether
from study_buddy.config import CONFIG, logger

logger.info(f"Initializing LLM with model: {CONFIG.llm.model}")

llm = ChatTogether(
    model=CONFIG.llm.model,
    temperature=0.3,
    streaming= True
)

logger.info(f"LLM with model {CONFIG.llm.model} successfully initialized.")
