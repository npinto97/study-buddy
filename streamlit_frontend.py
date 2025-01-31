import streamlit as st
from PIL import Image
import os

from study_buddy.agent import compiled_graph

# Set up the environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
    if os.path.exists("images/agents_graph.png"):
        graph_image = Image.open("images/agents_graph.png")
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

                last_ai_message = None

                for event in events:
                    ai_messages = [
                        msg.content for msg in event.get("messages", []) if type(msg).__name__ == "AIMessage"
                    ]

                    for ai_message in ai_messages:
                        if ai_message != last_ai_message and (
                            len(st.session_state.chat_history) < 2 or ai_message != st.session_state.chat_history[-2]["content"]
                        ):
                            st.write(ai_message)
                            st.session_state.chat_history.append({"role": "bot", "content": ai_message})
                            last_ai_message = ai_message 

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.caption("LangGraph Application - Built with Streamlit")
