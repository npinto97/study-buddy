from study_buddy.utils.vector_store import initialize_faiss_store
from study_buddy.config import logger


def main():
    logger.info("Starting the FAISS index update process...")

    vector_store = initialize_faiss_store()

    if vector_store:
        logger.info("The FAISS index has been successfully updated.")
    else:
        logger.error("There was an issue updating the FAISS index.")


if __name__ == "__main__":
    main()
