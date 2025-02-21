from pydub import AudioSegment
import streamlit as st
from PIL import Image
import mimetypes
import os
import re
from pathlib import Path
import tempfile
import json
import torch

from study_buddy.agent import compiled_graph
from study_buddy.utils.tools import OpenAITTSWrapper, OpenAISpeechToText

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# Set up the environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Streamlit app layout
st.set_page_config(page_title="LangGraph Interface", layout="wide")


st.logo(image=Path("images\\new_unichat_icon_hd.png"), size="large", icon_image=Path("images\\new_unichat_icon_hd.png"))


class ConfigChat:
    def __init__(self, complexity_level="None", language="Italian", course="None"):
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


def enhance_user_input(config_chat, user_input, file_path):
    """
    Generates an optimized prompt for the chatbot, ensuring:
    - A hierarchical retrieval strategy (first within the selected course, then other courses, finally the web).
    - Security measures to prevent harmful or unethical responses.
    - A clear and structured response according to user preferences.
    """

    # Complexity level customization
    complexity_instructions = {
        "Basic": "Keep the response simple and easy to understand, using everyday language and minimal technical jargon. Provide basic explanations with common examples.",
        "Intermediate": "Provide a moderate level of detail, including some technical terms where appropriate. Use examples that demonstrate the concept but remain accessible.",
        "Advanced": "Deliver an in-depth and comprehensive response, incorporating technical jargon, citations, and advanced examples when relevant. Provide critical analysis where applicable."
    }

    select_complexity_string = (
        f"{complexity_instructions.get(config_chat.complexity_level, '')}\n"
        if config_chat.complexity_level in complexity_instructions
        else ""
    )

    # Set response language
    select_language_string = f"The answer must be in {config_chat.language}.\n"

    # Define course context
    if config_chat.course == "None":
        select_course_string = ""
        course_info_string = ""
    else:
        select_course_string = f"The user is studying the course '{config_chat.course}'.\n"
        course_info_string = (
            f"Prioritize retrieving information from course materials related to '{config_chat.course}'.\n"
            "If no relevant course materials are found, expand the search to other courses.\n"
            "If there are still no relevant documents, use an appropriate web search tool.\n"
            "Structure your response with clarity, using examples where needed.\n"
        )

    # Handle attached files
    file_path_string = (
        f"The user has uploaded a file for analysis: {file_path}.\n"
        if file_path else ""
    )

    # Retrieval preference instructions
    retrieval_instruction = (
        "For document-related queries, prioritize using the 'retrieve_tool' to find relevant information.\n"
        "Always provide a source or a link where the user can access the referenced material.\n"
        "If the user's question relates to study efficiency, accessibility, or mental health, consider recommending the appropriate tool.\n"
        "Examples of tool recommendations:\n"
        "- If the user is struggling with complex concepts, suggest 'code_interpreter' for demonstrations.\n"
        "- If the user expresses stress, analyze their sentiment and provide study wellness tips, including recommending relaxing music or music they enjoy.\n"
        "- If accessibility is a concern, offer text-to-speech or document summarization options.\n"
        "- If the user is working on research, suggest retrieving academic papers via Google Scholar.\n"
    )

    # Interdisciplinary awareness
    interdisciplinary_instruction = (
        "Encourage an interdisciplinary approach when beneficial.\n"
        "For example, if a student is studying Natural Language Processing but struggles with mathematical models, offer insights from linear algebra.\n"
    )

    # # Security and ethical guidelines
    # security_guidelines = (
    #     "Ensure that all responses adhere to ethical and safety standards.\n"
    #     "DO NOT generate harmful, misleading, biased, or illegal content.\n"
    #     "DO NOT provide personal, medical, financial, or legal advice.\n"
    #     "If the request is ambiguous, ask for clarification before responding.\n"
    #     "If a query violates ethical guidelines, politely refuse to provide an answer.\n"
    # )

    # # Response optimization guidelines
    # response_guidelines = (
    #     "Structure the response in a clear and logical manner.\n"
    #     "Provide concise and informative answers while avoiding unnecessary verbosity.\n"
    #     "Use examples when helpful and cite sources when applicable.\n"
    #     "If additional context is required, prompt the user for clarification.\n"
    # )

    # Compose the final meta-prompt
    enhanced_user_input = (
        # security_guidelines +
        retrieval_instruction +
        course_info_string +
        select_complexity_string +
        select_language_string +
        select_course_string +
        # response_guidelines +
        file_path_string +
        interdisciplinary_instruction +
        user_input
    )

    return enhanced_user_input


def handle_chatbot_response(user_input, thread_id, config_chat, user_file_path):
    """Handle chatbot response and update conversation history."""
    if not (user_input.strip() or user_file_path):
        st.warning("Please enter a valid input.")
        return

    # modify user input based on config_complexity_level
    enhanced_user_input = enhance_user_input(config_chat, user_input, user_file_path)
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

            # Controlla se l'evento ha un messaggio ToolMessage che contiene il percorso dell'immagine
            tool_messages = [
                msg.content for msg in last_event.get("messages", [])
                if type(msg).__name__ == "ToolMessage" and msg.name == "image_generator"
            ]

            if ai_messages:
                chat_history.append({"role": "bot", "content": ai_messages[-1]})
            else:
                st.write("No AI message found.")

            image_path = None

            if tool_messages:
                try:
                    tool_output = json.loads(tool_messages[-1])
                    image_path = tool_output[0]

                    if image_path:
                        # Controlla se l'immagine è già stata aggiunta alla cronologia
                        last_image = next((msg for msg in reversed(chat_history) if msg.get("role") == "tool"), None)

                        # Aggiunge solo se è una nuova immagine e non è già presente
                        if not last_image or last_image.get("image") != image_path:
                            chat_history.append({"role": "tool", "image": image_path})
                except Exception as e:
                    st.error(f"Errore nell'estrazione dell'immagine: {str(e)}")

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


def save_file(user_file_input):
    """Salva il file caricato in base al tipo (audio, immagine, video, testo)"""
    try:
        file_name = user_file_input.name
        file_type, encoding = mimetypes.guess_type(file_name)

        # Crea una cartella temporanea per il salvataggio dei file
        temp_dir = tempfile.mkdtemp()

        # Verifica se il tipo di file è audio, immagine, video o testo
        if file_type:
            if file_type.startswith('audio'):
                # Salva i file audio con estensione .wav o altro
                tmp_file_path = save_audio_file(user_file_input, temp_dir)
            elif file_type.startswith('image'):
                # Salva i file immagine con estensione .jpg o altro
                tmp_file_path = save_image_file(user_file_input, temp_dir)
            elif file_type.startswith('video'):
                # Salva i file video con estensione .mp4 o altro
                tmp_file_path = save_video_file(user_file_input, temp_dir)
            elif file_type.startswith('text'):
                # Salva i file testuali (txt, csv, json, xml, ecc.)
                tmp_file_path = save_text_file(user_file_input, temp_dir)
            elif file_type == 'application/pdf':
                # Salva i file PDF
                tmp_file_path = save_pdf_file(user_file_input, temp_dir)
            else:
                st.error(f"Tipo di file non supportato: {file_type}")
                return ""
        else:
            st.error("Tipo di file non riconosciuto.")
            return ""

        return tmp_file_path
    except Exception as e:
        st.error(f"Errore durante il salvataggio del file: {str(e)}")
        return ""


def save_audio_file(user_file_input, temp_dir):
    """Salva i file audio nel formato appropriato"""
    try:
        tmp_file_path = os.path.join(temp_dir, user_file_input.name)
        with open(tmp_file_path, 'wb') as f:
            f.write(user_file_input.getvalue())
        st.success("File audio caricato 📤")
        return tmp_file_path
    except Exception as e:
        st.error(f"Errore durante il salvataggio del file audio: {str(e)}")
        return ""


def save_image_file(user_file_input, temp_dir):
    """Salva i file immagine nel formato appropriato"""
    try:
        tmp_file_path = os.path.join(temp_dir, user_file_input.name)
        with open(tmp_file_path, 'wb') as f:
            f.write(user_file_input.getvalue())
        st.success(f"File immagine caricato 📤 {tmp_file_path}")
        return tmp_file_path
    except Exception as e:
        st.error(f"Errore durante il salvataggio del file immagine: {str(e)}")
        return ""


def save_video_file(user_file_input, temp_dir):
    """Salva i file video nel formato appropriato"""
    try:
        tmp_file_path = os.path.join(temp_dir, user_file_input.name)
        with open(tmp_file_path, 'wb') as f:
            f.write(user_file_input.getvalue())
        st.success(f"File video caricato 📤 {tmp_file_path}")
        return tmp_file_path
    except Exception as e:
        st.error(f"Errore durante il salvataggio del file video: {str(e)}")
        return ""


def save_text_file(user_file_input, temp_dir):
    """Salva i file testuali (txt, csv, json, xml)"""
    try:
        tmp_file_path = os.path.join(temp_dir, user_file_input.name)
        with open(tmp_file_path, 'wb') as f:
            f.write(user_file_input.getvalue())
        st.success("File testuale caricato 📤")
        return tmp_file_path
    except Exception as e:
        st.error(f"Errore durante il salvataggio del file testuale: {str(e)}")
        return ""


def save_pdf_file(user_file_input, temp_dir):
    """Salva i file PDF"""
    try:
        tmp_file_path = os.path.join(temp_dir, user_file_input.name)
        with open(tmp_file_path, 'wb') as f:
            f.write(user_file_input.getvalue())
        st.success("File PDF caricato 📤")
        return tmp_file_path
    except Exception as e:
        st.error(f"Errore durante il salvataggio del file PDF: {str(e)}")
        return ""


def play_text_to_speech(text, key=None):
    """Call OpenAI's TTS tool and play the generated speech."""
    if st.button("🔊", key=key):
        tts_wrapper = OpenAITTSWrapper()
        audio_path = tts_wrapper.text_to_speech(text)
        st.audio(audio_path, format="audio/mp3")


def display_chat_history(thread_id):
    """Display the conversation history."""

    st.markdown("---")
    st.markdown("### Conversation")
    chat_history = get_chat_history(thread_id)

    download_counter = 0

    for i, chat in enumerate(chat_history):
        # print(f"-->{chat}")
        role = chat["role"]
        content = chat.get("content", None)
        image_path = chat.get("image", None)

        # Se il messaggio contiene un link Markdown a un'immagine, lo rimuoviamo
        if content:
            content = re.sub(r"!\[.*?\]\(sandbox:.*?\)", "", content).strip()

        # Messaggio dell'utente
        if role == "user":
            st.markdown(
                f'<p style="color: #613980;"><strong>Utente:</strong> {content if content else "[Messaggio vuoto]"}</p>',
                unsafe_allow_html=True,
            )

        # Messaggio del bot
        elif role == "bot" and content:
            st.markdown(f"**Bot:** {content}", unsafe_allow_html=True)

            # Cerca link file:/// e crea pulsanti di download
            if "file:///" in content:
                file_links = re.findall(r'\[([^\]]+)\]\((file:///.*?)\)', content)
                for text, link in file_links:
                    file_path = link.replace("file:///", "")  # Rimuove il prefisso per ottenere il percorso locale

                    if os.path.exists(file_path):
                        with open(file_path, "rb") as file:
                            file_bytes = file.read()

                        st.download_button(
                            label=f"📥 Scarica {text}",
                            data=file_bytes,
                            file_name=os.path.basename(file_path),
                            mime="application/octet-stream",
                            key=f"download_{download_counter}"
                        )
                        download_counter += 1
                    else:
                        st.error(f"❌ File non trovato: {file_path}")

            play_text_to_speech(content, key=f"tts_button_{i}")

        elif role == "tool" and image_path:
            if os.path.exists(image_path):
                st.image(image_path, caption="Immagine generata", width=500)
            else:
                st.error(f"Errore: immagine non trovata in {image_path}")


def main():
    """Main function to render the Streamlit app."""
    st.title(":crystal_ball: UNIVOX: University Virtual Orchestrated eXpert")

    # Initialize session state
    initialize_session()

    # Sidebar configuration
    user_input, user_file_input, user_audio_input, config_thread_id, submit_button, config_chat = sidebar_configuration()

    transcribed_text = ""
    if user_audio_input is not None:
        st.info("✍️ Trascrivendo audio...")
        transcribed_text = transcribe_audio(user_audio_input)
        if transcribed_text:
            user_input = transcribed_text
            st.success(f"Transcribed text: {user_input}")

    user_file_path = None
    if user_file_input is not None:
        st.info("📤 Caricando il file...")
        user_file_path = save_file(user_file_input)

    # Define layout columns
    col1, col2 = st.columns([1, 2])

    # Display graph in column 1
    with col1:
        st.header("🚀 Come utilizzare UNIVOX")

        with st.expander("📖 Clicca qui per vedere la guida"):
            st.markdown(
                """
                Questo tutor virtuale basato sull'intelligenza artificiale è progettato per supportarti nel tuo percorso accademico.  
                Ecco come puoi interagire con il chatbot:

                🔹 **Fai domande**: Digita la tua domanda nella casella di input per ricevere una risposta generata dall'AI.

                🔹 **Carica file**: Condividi PDF o immagini per estrarre e analizzare contenuti.

                🔹 **Usa l'input vocale**: Carica un file audio per trascrivere e processare la tua richiesta.

                🔹 **Supporto multi-tool**: Il sistema integra diversi strumenti AI per attività come riassunti, analisi del sentiment e riconoscimento testuale.  

                🔹 **Monitora la conversazione**: Visualizza le interazioni precedenti per tenere traccia delle discussioni.
                
                **Gestione delle conversazioni**  
                - **Thread ID**: Ogni chat ha un identificativo univoco (*Thread ID*).  
                - **Nuova chat**: Cambia il numero del *Thread ID* per avviare una nuova conversazione.  
                - **Riapri una chat precedente**: Inserisci un *Thread ID* già usato per riprendere la discussione da dove l'avevi lasciata.  


                **Consigli per un'esperienza ottimale**
                - Sii chiaro e specifico nelle tue domande.
                - Se necessario, fornisci contesto o documenti di supporto.
                - Sperimenta diverse formulazioni per esplorare al meglio le funzionalità.

                **Limitazioni**
                - Le risposte possono variare a causa della natura non deterministica dell'AI.
                - Alcune funzionalità avanzate sono state disabilitate per motivi di stabilità.
                - Il sistema è in continuo miglioramento: il tuo feedback è prezioso!

                🏆 *Migliora la tua esperienza di studio con UNIVOX!*
                """
            )

    # Chatbot response in column 2
    with col2:
        st.header("👩🏻‍🏫 Chatbot Response")

        if submit_button:
            handle_chatbot_response(user_input, config_thread_id, config_chat, user_file_path)
            display_chat_history(config_thread_id)
        else:
            display_chat_history(config_thread_id)

    # Footer
    st.markdown("---")
    st.caption("LangGraph Application - Built with Streamlit")


if __name__ == "__main__":
    main()
