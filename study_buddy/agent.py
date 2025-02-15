from loguru import logger

from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

from study_buddy.utils.memory import MemorySaver
from study_buddy.config import IMAGES_DIR

from study_buddy.utils.state import AgentState

from study_buddy.utils.nodes import call_model, tool_node

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def build_compiled_graph():
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("agent", call_model)
    graph_builder.add_node("tools", tool_node)

    graph_builder.set_entry_point("agent")

    graph_builder.add_conditional_edges(
        "agent",
        tools_condition
    )

    graph_builder.add_edge("tools", "agent")

    memory = MemorySaver()

    return graph_builder.compile(checkpointer=memory)


compiled_graph = build_compiled_graph()

if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

output_file_path = os.path.join(IMAGES_DIR, "agents_graph.png")
compiled_graph.get_graph().draw_mermaid_png(output_file_path=output_file_path)

logger.info(f"Graph saved to: {output_file_path}")


# # Run chatbot loop
# config = {"configurable": {"thread_id": "8"}}

# while True:
#     user_input = input("User: ")
#     if user_input.lower() in ["quit", "exit", "q"]:
#         print("Goodbye!")
#         break

#     events = compiled_graph.stream(
#         {"messages": [{"role": "user", "content": user_input}]},
#         config,
#         stream_mode="values",
#     )
#     for event in events:
#         # print("DEBUG: Messages:", event["messages"])
#         event["messages"][-1].pretty_print()

#     for message_chunk, metadata in compiled_graph.stream(
#         {"messages": [{"role": "user", "content": user_input}]},
#         config,
#         stream_mode="messages",
#     ):
#         if message_chunk.content:
#             print(message_chunk.content, end="", flush=True)
