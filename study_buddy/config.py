from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"Project root path resolved to: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
logger.info(f"Data directory path resolved to: {DATA_DIR}")

RAW_DATA_DIR = DATA_DIR / "raw"
logger.info(f"Raw data directory path resolved to: {RAW_DATA_DIR}")

PROCESSED_DATA_DIR = DATA_DIR / "processed"
logger.info(f"Processed data directory path resolved to: {PROCESSED_DATA_DIR}")

EXTERNAL_DATA_DIR = DATA_DIR / "external"
logger.info(f"External data directory path resolved to: {EXTERNAL_DATA_DIR}")

METADATA_DIR = DATA_DIR / "metadata"
logger.info(f"Metadata directory path resolved to: {METADATA_DIR}")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: (tqdm.write(msg, end="") if tqdm else logger.error(f"Failed to write log message: {msg}")), colorize=True)
except ModuleNotFoundError as e:
    logger.warning(f"tqdm is not installed: {e}")