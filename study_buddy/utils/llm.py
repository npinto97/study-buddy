from study_buddy.config import CONFIG, logger
import os

provider = CONFIG.llm.provider.lower()

if provider == "together":
    model_name = CONFIG.llm.together.model
    logger.info(f"Initializing Together AI LLM with model: {model_name}")
    try:
        from langchain_together import ChatTogether
    except ModuleNotFoundError:
        logger.error("Provider 'together' selected but package 'langchain_together' is not installed.")
        raise

    llm = ChatTogether(
        model=model_name,
        temperature=0.3,
        streaming=True
    )
    logger.info(f"Together AI LLM with model {model_name} successfully initialized.")

elif provider == "gemini":
    model_name = CONFIG.llm.gemini.model
    logger.info(f"Initializing Google Gemini LLM with model: {model_name}")

    # Check for Google API key
    if not os.environ.get("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY not found in environment variables")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ModuleNotFoundError:
        logger.error("Provider 'gemini' selected but package 'langchain_google_genai' is not installed.")
        raise

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        streaming=True,
        convert_system_message_to_human=True
    )
    logger.info(f"Google Gemini LLM with model {model_name} successfully initialized.")

else:
    logger.error(f"Unknown LLM provider: {provider}. Supported providers: 'together', 'gemini'")
    raise ValueError(f"Unknown LLM provider: {provider}")
