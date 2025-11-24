from pathlib import Path
import os
import getpass
import yaml
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
import sys
import pytesseract
from PIL import Image


from langchain_community.document_loaders import (TextLoader,
                                                  UnstructuredMarkdownLoader,
                                                  UnstructuredPDFLoader,
                                                  UnstructuredCSVLoader,
                                                  UnstructuredExcelLoader,
                                                  UnstructuredPowerPointLoader,
                                                  UnstructuredHTMLLoader,
                                                  UnstructuredEPubLoader,
                                                  UnstructuredWordDocumentLoader)


def image_loader(filepath: str):
    """ Extracts text from an image using OCR."""
    image = Image.open(filepath)
    text = pytesseract.image_to_string(image)
    return [text]


# Define a mapping for file loaders
FILE_LOADERS = {
    ".pdf": UnstructuredPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".csv": UnstructuredCSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".html": UnstructuredHTMLLoader,
    ".epub": UnstructuredEPubLoader,
}

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SUPPORTED_EXTENSIONS = set(FILE_LOADERS.keys()).union(AUDIO_EXTENSIONS, VIDEO_EXTENSIONS)

# Load environment variables from .env
load_dotenv()

logger.remove()
# Add colorful console output with detailed formatting
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
)

# Add detailed logging to file with process ID
logger.add(
    "logs/study_buddy_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {process} | {thread} | {name}:{function}:{line} | {message}",
    rotation="00:00",  # Create new file at midnight
    retention="30 days",  # Keep logs for 30 days
    compression="zip",
    enqueue=True  # Thread-safe logging
)

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
# METADATA_DIR = PROJ_ROOT / "notebooks" / "demo_material" / "metadata"
TEMP_DATA_DIR = DATA_DIR / "temp"
EXTRACTED_TEXT_DIR = PROCESSED_DATA_DIR / "extracted_text"
FAISS_INDEX_DIR = PROJ_ROOT / "faiss_index"
IMAGES_DIR = PROJ_ROOT / "images"

PARSED_COURSES_DATA_FILE = PROCESSED_DATA_DIR / "parsed_course_data.json"
PROCESSED_DOCS_FILE = PROCESSED_DATA_DIR / "processed_docs.json"
TEMP_DOCS_FILE = TEMP_DATA_DIR / "temp_extracted_documents.json"


# Definition of configuration models
class TogetherConfig(BaseModel):
    model: str

class GeminiConfig(BaseModel):
    model: str

class LLMConfig(BaseModel):
    provider: str
    together: TogetherConfig
    gemini: GeminiConfig


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
# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Advanced configuration of loguru for tqdm (if installed)
try:
    from tqdm import tqdm

    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True
    )
except ModuleNotFoundError as e:
    logger.warning(f"tqdm is not installed: {e}")

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false") == "true"
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

logger.info(f"LangSmith Tracing: {LANGSMITH_TRACING}")
logger.info(f"LangSmith Endpoint: {LANGSMITH_ENDPOINT}")
