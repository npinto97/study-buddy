from pydub import AudioSegment
import streamlit as st
from PIL import Image
import os
from pathlib import Path
import time
import tempfile 

from study_buddy.agent import compiled_graph
from study_buddy.utils.tools import OpenAITTSWrapper, OpenAISpeechToText

CHUNK_MAX_LENGTH = 16

# Set up the environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Streamlit app layout
st.set_page_config(page_title="LangGraph Interface", layout="wide")
st.logo(image=Path("images\\logo.png"), size="large", icon_image=Path("images\\logo_icon.png"))

# =============================================================================
# CONFIGURATION CLASSES & SESSION STATE
# =============================================================================

class ConfigChat:
    def __init__(self, complexity_level="None", language="English", course="None"):
        self.complexity_level = complexity_level
        self.language = language
        self.course = course

def initialize_session():
    """Inizializza la chat history e la lista dei thread se non già presenti."""
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    if "chat_list" not in st.session_state:
        st.session_state.chat_list = []
    if "audio_key" not in st.session_state:
        st.session_state["audio_key"] = 0 

def get_chat_history(thread_id):
    """Recupera (o crea) la cronologia per il thread_id."""
    if thread_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[thread_id] = []
    return st.session_state.chat_histories[thread_id]

tts_wrapper = OpenAITTSWrapper()

def play_text_to_speech(text, key):
    """Call OpenAI's TTS tool and play the generated speech."""
    if st.button("🔊", key=key):
        audio_path = tts_wrapper.text_to_speech(text)
        st.audio(audio_path, format="audio/mp3")

# =============================================================================
# SIDEBAR
# =============================================================================

def sidebar_configuration():
    """Renderizza la sidebar per l’input e la configurazione della chat."""
    with st.sidebar:
        user_input = st.text_area(":pencil2: Enter your input:", placeholder="Type your message here...")
        user_file_input = st.file_uploader(":paperclip: Attach multimedial content")
        user_audio_input = st.audio_input(":studio_microphone: Record a voice message", key=f"user_audio_{st.session_state['audio_key']}")

        new_thread_id = st.text_input("Thread ID:", value="7", help="Specify the thread ID for the configuration")
        if new_thread_id and new_thread_id not in st.session_state.chat_list:
            st.session_state.chat_list.append(new_thread_id)
        select_thread_id = st.selectbox(
            "Existing chats",
            st.session_state.chat_list,
            key="select_thread_id",
            help="Select one of the thread IDs"
        )
        config_thread_id = select_thread_id if select_thread_id and select_thread_id == new_thread_id else new_thread_id

        with st.expander(":gear: Chat configuration"):
            config_complexity_level = st.selectbox(
                "Level of complexity of responses",
                ('None', 'Base', 'Intermediate', 'Advanced')
            )
            config_language = st.selectbox("Responses language", ("Italian", "English"))
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

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def enhance_user_input(config_chat, user_input):
    """Modifica l'input dell'utente in base alle impostazioni di configurazione."""
    complexity_str = "" if config_chat.complexity_level == "None" else f"Answer with a response with {config_chat.complexity_level} level of complexity.\n"
    language_str = f"The answer must be in {config_chat.language}.\n"
    course_str = "" if config_chat.course == "None" else f"The question is about the course of {config_chat.course}.\n"
    return complexity_str + language_str + course_str + user_input

def render_chat_history_upto(thread_id, count, container):
    """
    Renderizza i messaggi nella cronologia fino all'indice 'count' (esclusi quelli successivi).
    Solo i messaggi del bot (assistant) mostrano il pulsante per generare l'audio.
    """
    chat_history = get_chat_history(thread_id)
    for idx, msg in enumerate(chat_history[:count]):
        if msg["role"] == "user":
            with container.chat_message("user"):
                st.markdown(msg["content"])
        else:  # Messaggio del bot
            with container.chat_message("assistant"):
                st.markdown(msg["content"])
                play_text_to_speech(text=msg["content"], key=f"audio_{idx}")


def render_chat_history(thread_id, container):
    """Renderizza l'intera cronologia."""
    chat_history = get_chat_history(thread_id)
    render_chat_history_upto(thread_id, len(chat_history), container)


def handle_chatbot_response(user_input, thread_id, config_chat):
    """
    Gestisce la coppia nuovo messaggio utente / risposta bot:
      - Visualizza il messaggio dell'utente (senza pulsante audio)
      - Streaming della risposta del bot, visualizzata subito sotto.
      - Al termine, mostra il pulsante TTS per il messaggio del bot.
      
    Questa coppia NON verrà renderizzata da render_chat_history (per evitare duplicazioni),
    ma è mostrata in fondo alla chat.
    """
    if not user_input.strip():
        st.warning("Please enter a valid input.")
        return

    enhanced_input = enhance_user_input(config_chat, user_input)
    config = {"configurable": {"thread_id": thread_id}}
    chat_history = get_chat_history(thread_id)

    chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    bot_response = ""
    bot_placeholder = st.empty()
    try:
        for message_chunk, metadata in compiled_graph.stream(
            {"messages": [{"role": "user", "content": enhanced_input}]},
            config,
            stream_mode="messages",
        ):
            if message_chunk.content and len(message_chunk.content) < CHUNK_MAX_LENGTH:
                bot_response += message_chunk.content
                bot_placeholder.chat_message("assistant").markdown(bot_response)
                time.sleep(0.05)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    play_text_to_speech(text=bot_response, key=f"new_bot_audio_{time.time()}")
    
    chat_history.append({"role": "bot", "content": bot_response})


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


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.title(":sparkles: UniChat: ci sono domande:question:")
    initialize_session()
    user_input, user_file_input, user_audio_input, config_thread_id, submit_button, config_chat = sidebar_configuration()
    col1, col2 = st.columns([1, 2])

    transcribed_text = ""
    if user_audio_input is not None:
        st.info("🎙️ Trascrivendo audio...")
        transcribed_text = transcribe_audio(user_audio_input)
        if transcribed_text:
            user_input = transcribed_text
            st.success(f"Transcribed text: {user_input}")

    with col1:
        st.header(":space_invader: Our Purpose")
        st.text("Inserire descrizione del progetto")

    with col2:
        st.header(":crystal_ball: Chatbot Response")
        chat_container = st.container()
        if submit_button:
            old_count = len(get_chat_history(config_thread_id))
            render_chat_history_upto(config_thread_id, old_count, chat_container)
            handle_chatbot_response(user_input, config_thread_id, config_chat)

            st.session_state["audio_key"] += 1 
        else:
            render_chat_history(config_thread_id, chat_container)

    st.markdown("---")
    st.caption("LangGraph Application - Built with Streamlit")

if __name__ == "__main__":
    main()
