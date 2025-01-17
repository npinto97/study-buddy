# from study_buddy.pipeline import graph
from study_buddy.config import IMAGES_DIR
import os
from loguru import logger


def generate_graph(graph):
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

    output_file_path = os.path.join(IMAGES_DIR, "pipeline_graph.png")
    graph.get_graph().draw_mermaid_png(output_file_path=output_file_path)

    logger.info(f"Graph saved to: {output_file_path}")
