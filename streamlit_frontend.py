import streamlit as st
from PIL import Image
import os

# Import your LangGraph application logic
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from study_buddy.utils.memory import MemorySaver
from study_buddy.utils.state import AgentState
from study_buddy.utils.nodes import call_model, tool_node
from study_buddy.config import IMAGES_DIR

# Set up the environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def build_graph():
    """Builds and compiles the LangGraph state graph."""
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", call_model)
    graph_builder.add_node("tools", tool_node)
    graph_builder.set_entry_point("agent")
    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")

    # memory = MemorySaver()
    # compiled_graph = graph_builder.compile(checkpointer=memory)

    if "memory" not in st.session_state:
        st.session_state.memory = MemorySaver()

    compiled_graph = graph_builder.compile(checkpointer=st.session_state.memory)

    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

    output_file_path = os.path.join(IMAGES_DIR, "agents_graph.png")
    compiled_graph.get_graph().draw_mermaid_png(output_file_path=output_file_path)

    return compiled_graph, output_file_path


# Build the graph and generate the visualization
compiled_graph, graph_image_path = build_graph()


# Streamlit app layout
st.set_page_config(page_title="LangGraph Interface", layout="wide")
st.title("LangGraph Application")

# Sidebar for user input
with st.sidebar:
    st.header("Configuration")
    user_input = st.text_area("Enter your input:", placeholder="Type your message here...")
    config_thread_id = st.text_input("Thread ID:", value="7", help="Specify the thread ID for the configuration.")
    submit_button = st.button("Submit")

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Graph Visualization")
    if os.path.exists(graph_image_path):
        graph_image = Image.open(graph_image_path)
        st.image(graph_image, caption="Compiled LangGraph", use_container_width=True)
    else:
        st.error("Graph image not found. Please check the path.")

with col2:
    st.header("Chatbot Response")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if submit_button:
        if not user_input.strip():
            st.warning("Please enter a valid input.")
        else:
            config = {"configurable": {"thread_id": config_thread_id}}

            if not st.session_state.chat_history or st.session_state.chat_history[-1]["content"] != user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})

            try:
                events = compiled_graph.stream(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config,
                    stream_mode="values",
                )
                st.success("Response received:")

                for event in events:
                    ai_messages = [
                        msg.content for msg in event.get("messages", []) if type(msg).__name__ == "AIMessage"
                    ]

                    for ai_message in ai_messages:
                        if not st.session_state.chat_history or st.session_state.chat_history[-1]["content"] != ai_message:
                            st.write(ai_message)
                            st.session_state.chat_history.append({"role": "bot", "content": ai_message})

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.caption("LangGraph Application - Built with Streamlit")
