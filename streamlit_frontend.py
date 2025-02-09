import streamlit as st
from PIL import Image
import os

from study_buddy.agent import compiled_graph

# Set up the environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Streamlit app layout
st.set_page_config(page_title="LangGraph Interface", layout="wide")


def initialize_session():
    """If not exists, initialize chat histories dictionary."""
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}


def get_chat_history(thread_id):
    """Retrieves the chat history associated to a specific thread_id.
    If not exists, it is initialized as an empty list"""
    if thread_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[thread_id] = []
    return st.session_state.chat_histories[thread_id]


def sidebar_configuration():
    """Render the sidebar for user input and configuration."""
    with st.sidebar:
        st.header("Configuration")
        user_input = st.text_area("Enter your input:", placeholder="Type your message here...")
        config_thread_id = st.text_input("Thread ID:", value="7", help="Specify the thread ID for the configuration.")
        submit_button = st.button("Submit")
        config_complexity_level = st.selectbox(
            "Level of complexity of responses",
            ('Base', 'Intermediate', 'Advanced')
        )
        config_language = st.selectbox(
            "Responses language",
            ("Italian", "English")
        )
        config_course = st.selectbox(
            "Which course do you want to delve in?",
            ("None", "Semantics in Intelligent Information Access")
        )
    
    return user_input, config_thread_id, submit_button, config_complexity_level, config_language, config_course


def display_graph():
    """Display the compiled LangGraph image."""
    st.header("Graph Visualization")
    graph_path = "images/agents_graph.png"
    
    if os.path.exists(graph_path):
        graph_image = Image.open(graph_path)
        st.image(graph_image, caption="Compiled LangGraph", use_container_width=True)
    else:
        st.error("Graph image not found. Please check the path.")


#TODO è una funzione per provare, ha il problema di dipendere dalla lingua
def enhance_user_input(config_complexity_level, config_language, config_course, user_input):

    select_complexity_string = f"Answer with a response with {config_complexity_level} level of complexity.\n"
    select_language_string = f"The answer must be in {config_language}.\n"
    if config_course == "None":
        select_course_string = ""
    else:
        select_course_string = f"The question is about the course of {config_course}.\n"

    enhanced_user_input = select_complexity_string + select_language_string + select_course_string + user_input

    # if config_complexity_level == "Base":
    #     enhanced_user_input = "Se è una domanda inerente al contenuto del corso, risponi in maniera basilare.\n" + user_input
    # elif config_complexity_level == "Intermediate":
    #     enhanced_user_input = "Se è una domanda inerente al contenuto del corso, rispondi in maniera intermedia.\n" + user_input
    # elif config_complexity_level == "Advanced":
    #     enhanced_user_input = "Se è una domanda inerente al contenuto del corso, rispondi in maniera dettagliata.\n" + user_input
    # else:
    #     raise ValueError('Selected complexity level not recognized')

    return enhanced_user_input


def handle_chatbot_response(user_input, thread_id, config_complexity_level, config_language, config_course):
    """Handle chatbot response and update conversation history."""
    if not user_input.strip():
        st.warning("Please enter a valid input.")
        return
    
    # modify user input based on config_complexity_level
    enhanced_user_input = enhance_user_input(config_complexity_level, config_language, config_course, user_input)
    config = {"configurable": {"thread_id": thread_id}}
    chat_history = get_chat_history(thread_id)

    # Save user message if not already the last one
    if not chat_history or chat_history[-1]["role"] != "user" or chat_history[-1]["content"] != user_input:
        chat_history.append({"role": "user", "content": user_input})

    try:
        events = compiled_graph.stream(
            {"messages": [{"role": "user", "content": enhanced_user_input}]},
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
                chat_history.append({"role": "bot", "content": ai_messages[-1]})
            else:
                st.write("No AI message found.")
        else:
            st.write("No stream events received.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def display_chat_history(thread_id):
    """Display the conversation history."""
    st.markdown("---")
    st.markdown("### Conversation")
    chat_history = get_chat_history(thread_id)
    for chat in chat_history:
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
    user_input, config_thread_id, submit_button, config_complexity_level, config_language, config_course = sidebar_configuration()

    # Define layout columns
    col1, col2 = st.columns([1, 2])

    # Display graph in column 1
    with col1:
        display_graph()

    # Chatbot response in column 2
    with col2:
        st.header("Chatbot Response")

        if submit_button:
            handle_chatbot_response(user_input, config_thread_id, config_complexity_level, config_language, config_course)

        display_chat_history(config_thread_id)

    # Footer
    st.markdown("---")
    st.caption("LangGraph Application - Built with Streamlit")

if __name__ == "__main__":
    main()
