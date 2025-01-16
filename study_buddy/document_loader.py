# import nltk
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from study_buddy.config import RAW_DATA_DIR, logger

logger.info("Loading documents from directory...")

text_loader_kwargs = {"autodetect_encoding": True}
loader = DirectoryLoader(
    RAW_DATA_DIR,
    silent_errors=True,
    show_progress=True,
    use_multithreading=True,
    loader_kwargs=text_loader_kwargs,
    recursive=True
)

docs = loader.load()
logger.info(f"Loaded {len(docs)} documents.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

logger.info(f"Split documents into {len(all_splits)} chunks.")