from study_buddy.pipeline import graph
from study_buddy.config import logger
from study_buddy.utils.graph_generator import generate_graph

logger.info("Starting Study Buddy Application...")

if __name__ == "__main__":
    question = input("Enter your question: ")
    state = {"question": question, "context": [], "answer": ""}

    logger.info(f"Received user question: {question}")
    result = graph.invoke(state)

    logger.info(f"Final answer: {result['answer']}")
    print("Answer:", result["answer"])

    generate_graph(graph=graph)

    logger.info("Application finished execution.")
