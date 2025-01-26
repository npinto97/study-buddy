from pathlib import Path
import os
import getpass
import yaml
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
import sys


# Load environment variables from .env
load_dotenv()

logger.add(sys.stderr, level="INFO", format="{time} {level} {message}")
logger.add("logs/study_buddy.log", level="DEBUG", rotation="10 MB", compression="zip")

# logger.info("Logging initialized.")

# Main paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"Project root path resolved to: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
# RAW_DATA_DIR = PROJ_ROOT / "notebooks" / "demo_material"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
METADATA_DIR = DATA_DIR / "metadata"

FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

PROCESSED_DOCS_FILE = PROCESSED_DATA_DIR / "processed_docs.json"
# PROCESSED_STATUS_FILE = PROCESSED_DATA_DIR / "processed_status.json"
# logger.info(f"Processed documents file set to: {PROCESSED_DOCS_FILE}")


IMAGES_DIR = PROJ_ROOT / "images"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


# Definition of configuration models
class LLMConfig(BaseModel):
    model: str


class EmbeddingsConfig(BaseModel):
    model: str


class VectorStoreConfig(BaseModel):
    type: str  # "in_memory" o "faiss"


class AppConfig(BaseModel):
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    vector_store: VectorStoreConfig


# Loading configuration from YAML
CONFIG_PATH = PROJ_ROOT / "config.yaml"


def load_config(path: Path) -> AppConfig:
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)
    return AppConfig(**config_data)


CONFIG = load_config(CONFIG_PATH)

# Handling the OpenAI API Key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Advanced configuration of loguru for tqdm (if installed)
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(
        lambda msg: tqdm.write(msg, end="") if tqdm else logger.error(f"Failed to write log message: {msg}"),
        colorize=True
    )
except ModuleNotFoundError as e:
    logger.warning(f"tqdm is not installed: {e}")

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false") == "true"
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

logger.info(f"LangSmith Tracing: {LANGSMITH_TRACING}")
logger.info(f"LangSmith Endpoint: {LANGSMITH_ENDPOINT}")
