import os
from loguru import logger
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from study_buddy.modules.memory import MemorySaver
from study_buddy.config import IMAGES_DIR


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[Sequence[BaseMessage], add_messages]


def build_graph(node, node_tools):
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", node)

    chatbot_node = ToolNode(tools=node_tools)
    graph_builder.add_node("tools", chatbot_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("tools", "chatbot")

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)


def print_graph(graph):
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

    output_file_path = os.path.join(IMAGES_DIR, "agents_graph.png")
    graph.get_graph().draw_mermaid_png(output_file_path=output_file_path)

    logger.info(f"Graph saved to: {output_file_path}")
