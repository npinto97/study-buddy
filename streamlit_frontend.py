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

            # Save user message if not already present as the last message
            if not st.session_state.chat_history or st.session_state.chat_history[-1]["role"] != "user" or st.session_state.chat_history[-1]["content"] != user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})

            try:
                events = compiled_graph.stream(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config,
                    stream_mode="values",
                )
                st.success("Response received:")

                last_event = None
                count = 0

                # iterate over the events, considering the last one as the final response
                for event in events:
                    count += 1
                    print(f"event {count}: {event}\n")
                    last_event = event

                # After processing all events, extract the last AI message
                if last_event is not None:
                    ai_messages = [
                        msg.content for msg in last_event.get("messages", [])
                        if type(msg).__name__ == "AIMessage"
                    ]
                    if ai_messages:
                        final_ai_message = ai_messages[-1]
                        st.session_state.chat_history.append({"role": "bot", "content": final_ai_message})
                    else: 
                        st.write("No AI message found.")
                else:
                    st.write("No stream events received.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Show conversation history
    st.markdown("---")
    st.markdown("### Conversation")
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"**Utente:** {chat['content']}")
        else:
            st.markdown(f"**Bot:** {chat['content']}")


# Footer
st.markdown("---")
st.caption("LangGraph Application - Built with Streamlit")
