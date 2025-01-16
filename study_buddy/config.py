from pathlib import Path
import os
import getpass
import yaml
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
import sys

logger.add(sys.stderr, level="INFO", format="{time} {level} {message}")
logger.add("logs/study_buddy.log", level="DEBUG", rotation="10 MB", compression="zip")

logger.info("Logging initialized.")

# Load environment variables from .env, if it exists
load_dotenv()

# Main paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"Project root path resolved to: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
METADATA_DIR = DATA_DIR / "metadata"

EXTRACTED_TEXT_DIR = PROCESSED_DATA_DIR / "extracted_text"
FAISS_INDEX_DIR = PROCESSED_DATA_DIR / "faiss_index"


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
