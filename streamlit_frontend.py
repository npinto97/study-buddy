from pydub import AudioSegment
import streamlit as st
from PIL import Image
import os
from pathlib import Path
import time
import tempfile 

from study_buddy.agent import compiled_graph
from study_buddy.utils.tools import OpenAITTSWrapper, OpenAISpeechToText

# Set up the environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Streamlit app layout
st.set_page_config(page_title="LangGraph Interface", layout="wide")


st.logo(image=Path("images\\logo.png"), size="large", icon_image=Path("images\\logo_icon.png"))


class ConfigChat:
    def __init__(self, complexity_level="None", language="Engligh", course="None"):
        self.complexity_level = complexity_level
        self.language = language
        self.course = course


def initialize_session():
    """If not exists, initialize chat histories dictionary."""
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    if "chat_list" not in st.session_state:
        st.session_state.chat_list = []


def get_chat_history(thread_id):
    """Retrieves the chat history associated to a specific thread_id.
    If not exists, it is initialized as an empty list"""
    if thread_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[thread_id] = []
    return st.session_state.chat_histories[thread_id]


def sidebar_configuration():
    """Render the sidebar for user input and configuration."""
    with st.sidebar:

        user_input = st.text_area(":pencil2: Enter your input:", placeholder="Type your message here...")

        user_file_input = st.file_uploader(":paperclip: Attach multimedial content")

        user_audio_input = st.audio_input(":studio_microphone: Use voice mode")

        # config_thread_id = st.text_input("Thread ID:", value="7", help="Specify the thread ID for the configuration.")

        new_thread_id = st.text_input(
            "Thread ID:",
            value="7",
            help="Specify the thread ID for the configuration"
        )

        if new_thread_id and new_thread_id not in st.session_state.chat_list:
            st.session_state.chat_list.append(new_thread_id)

        # submit_button = st.button("Submit")
        select_thread_id = st.selectbox(
            "Existing chats",
            st.session_state.chat_list,
            key="select_thread_id",
            help="Configure one of the following by selecting the corresponding Thread ID"
        )

        config_thread_id = select_thread_id if select_thread_id and select_thread_id == new_thread_id else new_thread_id

        with st.expander(":gear: Chat configuration"):

            config_complexity_level = st.selectbox(
                "Level of complexity of responses",
                ('None', 'Base', 'Intermediate', 'Advanced')
            )
            config_language = st.selectbox(
                "Responses language",
                ("Italian", "English")
            )
            config_course = st.selectbox(
                "Which course do you want to delve in?",
                ("None", "Semantics in Intelligent Information Access")
            )

            config_chat = ConfigChat(
                complexity_level=config_complexity_level,
                language=config_language,
                course=config_course
            )

        submit_button = st.button("Submit")

    return user_input, user_file_input, user_audio_input, config_thread_id, submit_button, config_chat


def display_graph():
    """Display the compiled LangGraph image."""
    st.header("Graph Visualization")
    graph_path = "images/agents_graph.png"

    if os.path.exists(graph_path):
        graph_image = Image.open(graph_path)
        st.image(graph_image, caption="Compiled LangGraph", use_container_width=True)
    else:
        st.error("Graph image not found. Please check the path.")


def enhance_user_input(config_chat, user_input):

    if config_chat.complexity_level == "None":
        select_complexity_string = ""
    else:
        select_complexity_string = f"Answer with a response with {config_chat.complexity_level} level of complexity.\n"

    select_language_string = f"The answer must be in {config_chat.language}.\n"

    if config_chat.course == "None":
        select_course_string = ""
    else:
        select_course_string = f"The question is about the course of {config_chat.course}.\n"

    enhanced_user_input = select_complexity_string + select_language_string + select_course_string + user_input

    return enhanced_user_input


def handle_chatbot_response(user_input, thread_id, config_chat):
    """Handle chatbot response and update conversation history."""
    if not user_input.strip():
        st.warning("Please enter a valid input.")
        return

    # modify user input based on config_complexity_level
    enhanced_user_input = enhance_user_input(config_chat, user_input)
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


def transcribe_audio(audio_file):
    """Use OpenAI's Whisper to transcribe audio."""
    try:
        # Salva il file audio in un file temporaneo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_audio_file:
            tmp_audio_file.write(audio_file.getvalue())
            tmp_audio_file.close()
            wav_audio_path = tmp_audio_file.name

            audio = AudioSegment.from_file(tmp_audio_file.name)
            wav_audio_path = tmp_audio_file.name.replace(".tmp", ".wav")

            audio.export(wav_audio_path, format="wav")

            transcriber = OpenAISpeechToText()
            transcribed_audio = transcriber.transcribe_audio(wav_audio_path)

            return transcribed_audio
    except Exception as e:
        st.error(f"Audio transcription error: {str(e)}")
        return ""


def play_text_to_speech(text, key=None):
    """Call OpenAI's TTS tool and play the generated speech."""
    if st.button("🔊", key=key):
        tts_wrapper = OpenAITTSWrapper()
        audio_path = tts_wrapper.text_to_speech(text)
        st.audio(audio_path, format="audio/mp3")


def display_chat_history(thread_id, chunk_last_message=False):
    """Display the conversation history.

    If `chunk_last_message` is True, the last bot message is shown as generated in chunks.
    """
    st.markdown("---")
    st.markdown("### Conversation")
    chat_history = get_chat_history(thread_id)

    for i, chat in enumerate(chat_history):
        if chat["role"] == "user":
            st.markdown(
                f'<p style="color: #613980;"><strong>Utente:</strong> {chat["content"]}</p>',
                unsafe_allow_html=True,
            )
        else:
            # Se il submit è stato cliccato ed è l'ultimo messaggio del bot, mostralo a chunk
            if chunk_last_message and i == len(chat_history) - 1:
                placeholder = st.empty()
                content_so_far = ""
                # Simula la generazione a chunk (ad esempio, aggiornando parola per parola)
                for word in chat["content"].split():
                    content_so_far += word + " "
                    placeholder.markdown(f"**Bot:** {content_so_far}")
                    time.sleep(0.1)  # Ritardo per simulare la generazione graduale
                # NON stampare un ulteriore messaggio finale: il placeholder è già aggiornato.
            else:
                st.markdown(f"**Bot:** {chat['content']}")

            play_text_to_speech(chat["content"], key=f"tts_button_{i}")


def main():
    """Main function to render the Streamlit app."""
    st.title(":sparkles: UniChat: ci sono domande:question:")

    # Initialize session state
    initialize_session()

    # Sidebar configuration
    user_input, user_file_input, user_audio_input, config_thread_id, submit_button, config_chat = sidebar_configuration()

    transcribed_text = ""
    if user_audio_input is not None:
        st.info("🎙️ Trascrivendo audio...")
        transcribed_text = transcribe_audio(user_audio_input)
        if transcribed_text:
            user_input = transcribed_text
            st.success(f"Transcribed text: {user_input}")

    # Define layout columns
    col1, col2 = st.columns([1, 2])

    # Display graph in column 1
    with col1:
        # display_graph()
        st.header(":space_invader: Our Purpose")
        st.text("Inserire descrizione del progetto")

    # Chatbot response in column 2
    with col2:
        st.header(":crystal_ball: Chatbot Response")

        if submit_button:
            handle_chatbot_response(user_input, config_thread_id, config_chat)
            display_chat_history(config_thread_id, chunk_last_message=True)
        else:
            display_chat_history(config_thread_id)

    # Footer
    st.markdown("---")
    st.caption("LangGraph Application - Built with Streamlit")


if __name__ == "__main__":
    main()
