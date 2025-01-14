from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"Project root path resolved to: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
METADATA_DIR = DATA_DIR / "metadata"

EXTRACTED_TEXT_DIR = PROCESSED_DATA_DIR / "extracted_text"
FAISS_INDEX_DIR = PROCESSED_DATA_DIR / "faiss_index"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: (tqdm.write(msg, end="") if tqdm else logger.error(f"Failed to write log message: {msg}")), colorize=True)
except ModuleNotFoundError as e:
    logger.warning(f"tqdm is not installed: {e}")