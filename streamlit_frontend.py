import streamlit as st
from PIL import Image
import os

from study_buddy.agent import compiled_graph

# Set up the environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Streamlit app layout
st.set_page_config(page_title="LangGraph Interface", layout="wide")

def initialize_session():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def sidebar_configuration():
    """Render the sidebar for user input and configuration."""
    with st.sidebar:
        st.header("Configuration")
        user_input = st.text_area("Enter your input:", placeholder="Type your message here...")
        config_thread_id = st.text_input("Thread ID:", value="7", help="Specify the thread ID for the configuration.")
        submit_button = st.button("Submit")
    
    return user_input, config_thread_id, submit_button

def display_graph():
    """Display the compiled LangGraph image."""
    st.header("Graph Visualization")
    graph_path = "images/agents_graph.png"
    
    if os.path.exists(graph_path):
        graph_image = Image.open(graph_path)
        st.image(graph_image, caption="Compiled LangGraph", use_container_width=True)
    else:
        st.error("Graph image not found. Please check the path.")

def handle_chatbot_response(user_input, config_thread_id):
    """Handle chatbot response and update conversation history."""
    if not user_input.strip():
        st.warning("Please enter a valid input.")
        return

    config = {"configurable": {"thread_id": config_thread_id}}

    # Save user message if not already the last one
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

        # Iterate over events and store the last one
        for count, event in enumerate(events, start=1):
            print(f"event {count}: {event}\n")
            last_event = event

        # Extract the last AI message
        if last_event is not None:
            ai_messages = [
                msg.content for msg in last_event.get("messages", [])
                if type(msg).__name__ == "AIMessage"
            ]
            if ai_messages:
                st.session_state.chat_history.append({"role": "bot", "content": ai_messages[-1]})
            else:
                st.write("No AI message found.")
        else:
            st.write("No stream events received.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def display_chat_history():
    """Display the conversation history."""
    st.markdown("---")
    st.markdown("### Conversation")
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"**Utente:** {chat['content']}")
        else:
            st.markdown(f"**Bot:** {chat['content']}")

def main():
    """Main function to render the Streamlit app."""
    st.title("LangGraph Application")

    # Initialize session state
    initialize_session()

    # Sidebar configuration
    user_input, config_thread_id, submit_button = sidebar_configuration()

    # Define layout columns
    col1, col2 = st.columns([1, 2])

    # Display graph in column 1
    with col1:
        display_graph()

    # Chatbot response in column 2
    with col2:
        st.header("Chatbot Response")

        if submit_button:
            handle_chatbot_response(user_input, config_thread_id)

        display_chat_history()

    # Footer
    st.markdown("---")
    st.caption("LangGraph Application - Built with Streamlit")

if __name__ == "__main__":
    main()
