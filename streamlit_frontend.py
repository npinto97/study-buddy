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
import asyncio
from typing import AsyncGenerator

from study_buddy.agent import compiled_graph
from study_buddy.utils.tools import ElevenLabsTTSWrapper, AssemblyAISpeechToText

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# Set up the environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.logo(image=Path("images\\univox_logo2.png"), size="large", icon_image=Path("images\\univox_logo2.png"))

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
    if "voice_mode" not in st.session_state:
        st.session_state.voice_mode = False
    if "streaming_enabled" not in st.session_state:
        st.session_state.streaming_enabled = True

def get_chat_history(thread_id):
    """Retrieves the chat history associated to a specific thread_id.
    If not exists, it is initialized as an empty list"""
    if thread_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[thread_id] = []
    return st.session_state.chat_histories[thread_id]

def format_message_content(message):
    """Formatta il contenuto del messaggio per la visualizzazione"""
    if hasattr(message, 'content'):
        content = message.content
    elif isinstance(message, dict):
        content = message.get('content', None)
    else:
        content = str(message)
    
    # Rimuove Markdown immagini da contenuto testuale
    if content:
        content = re.sub(r"!\[.*?\]\(sandbox:.*?\)", "", content).strip()
    
    return content

def format_tool_calls(message):
    """Formatta le chiamate ai tool per la visualizzazione"""
    if hasattr(message, 'tool_calls') and message.tool_calls:
        tool_info = []
        for tool_call in message.tool_calls:
            tool_info.append(f"🔧 **{tool_call['name']}**")
            if 'args' in tool_call:
                for key, value in tool_call['args'].items():
                    tool_info.append(f"  - {key}: `{value}`")
        return "\n".join(tool_info)
    return None

def display_images_and_files(content, image_paths_list=None, message_index=0):
    """Gestisce la visualizzazione di immagini e file"""
    download_counter = 0
    
    # Gestisce immagini dal contenuto markdown
    if content:
        image_markdown_matches = re.findall(r'!\[.*?\]\((.*?)\)', content)
        
        for img_path in image_markdown_matches:
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.warning(f"⚠️ Immagine non trovata: {img_path}")
    
    # Mostra immagini da image_paths
    if image_paths_list:
        for img_path in image_paths_list:
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.warning(f"⚠️ Immagine non trovata: {img_path}")
    
    # Cerca percorsi file (es. C:\Users...) e file:///...
    if content:
        file_paths = re.findall(r'([A-Za-z]:\\[^\s]+)', content)
        file_paths += re.findall(r'\[([^\]]+)\]\((file:///.*?)\)', content)

        for file_path in file_paths:
            file_path = file_path.strip().rstrip(").,]\"'")

            if isinstance(file_path, tuple):
                text, link = file_path
                file_path = link.replace("file:///", "")

            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    file_bytes = file.read()

                st.download_button(
                    label=f"📥 Scarica {os.path.basename(file_path)}",
                    data=file_bytes,
                    file_name=os.path.basename(file_path),
                    mime="application/octet-stream",
                    key=f"download_{message_index}_{download_counter}"
                )
                download_counter += 1
            else:
                st.error(f"❌ File non trovato: {file_path}")

def display_chat_history(thread_id):
    """Display the conversation history with streaming support."""
    chat_history = get_chat_history(thread_id)
    
    # Container per i messaggi
    chat_container = st.container()
    
    # Visualizzazione dei messaggi
    with chat_container:
        for i, message in enumerate(chat_history):
            role = message.get("role", "unknown")
            content = format_message_content(message)
            
            # Messaggio utente
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content if content else "[Messaggio vuoto]")
            
            # Messaggio bot/assistant
            elif role == "bot" or role == "assistant":
                with st.chat_message("assistant"):
                    # Contenuto principale (senza immagini markdown)
                    if content:
                        cleaned_content = re.sub(r'!\[.*?\]\((.*?)\)', '', content).strip()
                        if cleaned_content:
                            st.markdown(cleaned_content)
                    
                    # Gestisci immagini e file
                    image_paths_list = message.get("image_paths", [])
                    display_images_and_files(content, image_paths_list, i)
                    
                    # Mostra le tool calls se presenti (per compatibilità con LangChain)
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        tool_calls_formatted = format_tool_calls(message)
                        if tool_calls_formatted:
                            with st.expander("🔧 Strumenti utilizzati"):
                                st.markdown(tool_calls_formatted)
                    
                    # TTS button
                    play_text_to_speech(content or "", key=f"tts_button_{i}")
            
            # Messaggio tool (per LangChain ToolMessage)
            elif hasattr(message, 'name') and message.name:
                with st.chat_message("assistant"):
                    with st.expander(f"📋 Risultato: {message.name}"):
                        st.code(format_message_content(message), language="text")

# Funzione asincrona per gestire lo streaming della risposta
async def stream_agent_response(user_input: str, thread_id: str, config_chat, user_file_paths=None) -> AsyncGenerator[str, None]:
    """Gestisce lo streaming della risposta dell'agente"""
    file_path_for_prompt = user_file_paths[0] if user_file_paths else None
    enhanced_user_input = enhance_user_input(config_chat, user_input, file_path_for_prompt)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Stream della risposta dall'agente
        async for event in compiled_graph.astream(
            {"messages": [{"role": "user", "content": enhanced_user_input}]}, 
            config=config
        ):
            # Gestisci diversi tipi di eventi
            for node_name, node_output in event.items():
                if node_name == "agent" and "messages" in node_output:
                    for message in node_output["messages"]:
                        # Per AIMessage di LangChain
                        if hasattr(message, 'content') and message.content:
                            # Stream del contenuto character by character
                            for char in message.content:
                                yield char
                                await asyncio.sleep(0.01)  # Piccolo delay per effetto typing
                        
                        # Per messaggi dictionary
                        elif isinstance(message, dict) and message.get('content'):
                            for char in message['content']:
                                yield char
                                await asyncio.sleep(0.01)
    
    except Exception as e:
        yield f"\n❌ **Errore:** {str(e)}"

# Funzione di supporto per gestire lo streaming in Streamlit
async def handle_streaming_response_async(user_input: str, thread_id: str, config_chat):
    """Gestisce la risposta streaming in Streamlit"""
    message_placeholder = st.empty()
    full_response = ""
    
    try:
        async for chunk in stream_agent_response(user_input, thread_id, config_chat):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        
        # Rimuovi il cursore alla fine
        message_placeholder.markdown(full_response)
        return full_response
        
    except Exception as e:
        error_msg = f"Errore durante lo streaming: {str(e)}"
        message_placeholder.markdown(error_msg)
        return error_msg

def sidebar_configuration():
    """Render the sidebar for configuration and guide."""
    with st.sidebar:
        # Sezione Thread ID
        new_thread_id = st.text_input(
            "Thread ID:",
            value="7",
            help="Specify the thread ID for the configuration"
        )

        if new_thread_id and new_thread_id not in st.session_state.chat_list:
            st.session_state.chat_list.append(new_thread_id)

        select_thread_id = st.selectbox(
            "Existing chats",
            st.session_state.chat_list,
            key="select_thread_id",
            help="Configure one of the following by selecting the corresponding Thread ID"
        )

        config_thread_id = select_thread_id if select_thread_id and select_thread_id == new_thread_id else new_thread_id

        # Sezione configurazione chat
        with st.expander(":gear: Chat configuration"):
            config_complexity_level = st.selectbox(
                "Level of complexity of responses",
                ('None', 'Base', 'Intermediate', 'Advanced')
            )
            config_course = st.selectbox(
                "Which course do you want to delve in?",
                ("None", "Semantics in Intelligent Information Access")
            )
            
            # Opzione per abilitare/disabilitare streaming
            streaming_enabled = st.checkbox(
                "Abilita streaming delle risposte",
                value=st.session_state.streaming_enabled,
                help="Mostra le risposte in tempo reale carattere per carattere"
            )
            st.session_state.streaming_enabled = streaming_enabled

            config_chat = ConfigChat(
                complexity_level=config_complexity_level,
                course=config_course
            )

        # Sezione guida spostata qui
        st.divider()  # Separatore visivo
        
        with st.expander("🚀 Come utilizzare UNIVOX"):
            st.markdown(
                """
                Questo tutor virtuale basato sull'intelligenza artificiale è progettato per supportarti nel tuo percorso accademico.  
                Ecco come puoi interagire con il chatbot:

                🔹 **Fai domande**: Digita la tua domanda nella casella di input per ricevere una risposta generata dall'AI.

                🔹 **Carica file**: Trascina file PDF o immagini direttamente nella chat per estrarre e analizzare contenuti.

                🔹 **Usa l'input vocale**: Clicca sul pulsante del microfono per registrare la tua voce.

                🔹 **Supporto multi-tool**: Il sistema integra diversi strumenti AI per attività come riassunti, analisi del sentiment e riconoscimento testuale.  

                🔹 **Monitora la conversazione**: Visualizza le interazioni precedenti per tenere traccia delle discussioni.
                
                **Gestione delle conversazioni**  
                - **Thread ID**: Ogni chat ha un identificativo univoco (*Thread ID*).  
                - **Nuova chat**: Cambia il numero del *Thread ID* per avviare una nuova conversazione.  
                - **Riapri una chat precedente**: Inserisci un *Thread ID* già usato per riprendere la discussione da dove l'avevi lasciata.  

                **Modalità Streaming**
                - **Abilita streaming**: Nelle impostazioni puoi attivare le risposte in tempo reale
                - **Effetto typing**: Vedi il testo apparire carattere per carattere
                - **Indicatori tool**: Visualizza quando vengono utilizzati gli strumenti

                **Consigli per un'esperienza ottimale**
                - Sii chiaro e specifico nelle tue domande.
                - Se necessario, fornisci contesto o documenti di supporto.
                - Sperimenta diverse formulazioni per esplorare al meglio le funzionalità.

                **Limitazioni**
                - Le risposte possono variare a causa della natura non deterministica dell'AI.
                - Alcune funzionalità avanzate sono state disabilitate per motivi di stabilità.
                - Il sistema è in continuo miglioramento: il tuo feedback è prezioso!

                🏆 *Migliora la tua esperienza di studio con UNIVOX!*
                """,
                unsafe_allow_html=True
            )

    return config_thread_id, config_chat

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
    file_path_string = ""
    if file_path:
        file_path_string = f"The user has uploaded a file for analysis: {file_path}.\n"

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

    # Compose the final meta-prompt
    enhanced_user_input = (
        retrieval_instruction +
        course_info_string +
        select_complexity_string +
        select_language_string +
        select_course_string +
        file_path_string +
        interdisciplinary_instruction +
        user_input
    )

    return enhanced_user_input

def save_chat_history(thread_id, chat_history):
    # Crea la cartella se non esiste
    os.makedirs("history", exist_ok=True)

    with open(f"history/{thread_id}.json", "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

def handle_chatbot_response(user_input, thread_id, config_chat, user_files=None):
    """Handle chatbot response with streaming support."""
    # Gestisci il caso in cui user_input sia un oggetto ChatInputValue
    if hasattr(user_input, 'text'):
        input_text = user_input.text
    elif isinstance(user_input, str):
        input_text = user_input
    else:
        input_text = str(user_input) if user_input else ""
    
    if not (input_text.strip() or user_files):
        st.warning("Please enter a valid input.")
        return

    chat_history = get_chat_history(thread_id)

    # Save user message if not already the last one
    if not chat_history or chat_history[-1]["role"] != "user" or chat_history[-1]["content"] != input_text:
        chat_history.append({"role": "user", "content": input_text})

    # Process uploaded files if any
    user_file_paths = []
    if user_files:
        for uploaded_file in user_files:
            file_path = save_file(uploaded_file)
            if file_path:
                user_file_paths.append(file_path)
                st.success(f"File caricato: {uploaded_file.name}")

    try:
        if st.session_state.streaming_enabled:
            # Modalità streaming
            file_path_for_prompt = user_file_paths[0] if user_file_paths else None
            enhanced_user_input = enhance_user_input(config_chat, input_text, file_path_for_prompt)
            config = {"configurable": {"thread_id": thread_id}}
            
            full_response = ""
            message_placeholder = st.empty()
            
            # Raccoglie tutti gli eventi per processare i tool messages
            all_events = []
            has_image_tools = False  # Flag per controllare se ci sono strumenti che generano immagini
            
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            async def run_streaming():
                nonlocal full_response, all_events, has_image_tools
                
                async for event in compiled_graph.astream(
                    {"messages": [{"role": "user", "content": enhanced_user_input}]}, 
                    config=config
                ):
                    all_events.append(event)
                    
                    # Controlla se ci sono tool che generano immagini
                    for node_name, node_output in event.items():
                        if node_name == "tools" and "messages" in node_output:
                            for message in node_output["messages"]:
                                if (hasattr(message, 'name') and 
                                    message.name in ["image_generator", "data_viz_tool"]):
                                    has_image_tools = True
                    
                    for node_name, node_output in event.items():
                        if node_name == "agent" and "messages" in node_output:
                            for message in node_output["messages"]:
                                if hasattr(message, 'content') and message.content:
                                    # Se ci sono strumenti che generano immagini, non fare streaming del testo
                                    # ma aspetta di mostrare tutto insieme
                                    if not has_image_tools:
                                        for char in message.content:
                                            full_response += char
                                            message_placeholder.markdown(full_response + "▌")
                                            await asyncio.sleep(0.01)
                                    else:
                                        full_response += message.content
                                
                                if hasattr(message, 'tool_calls') and message.tool_calls:
                                    tool_info = f"\n\n🔧 **Utilizzo strumenti:** {len(message.tool_calls)} tool(s)"
                                    if not has_image_tools:
                                        full_response += tool_info
                                        message_placeholder.markdown(full_response + "▌")
                
                # Se non ci sono strumenti immagine, mostra solo il testo finale
                if not has_image_tools:
                    message_placeholder.markdown(full_response)
                
                return full_response
            
            # Esegui lo streaming
            full_response = loop.run_until_complete(run_streaming())
            
            # Process tool messages for images after streaming is complete
            tool_messages = []
            for event in all_events:
                for node_name, node_output in event.items():
                    if node_name == "tools" and "messages" in node_output:
                        for message in node_output["messages"]:
                            if (hasattr(message, 'name') and 
                                message.name in ["image_generator", "data_viz_tool"]):
                                tool_messages.append(message)
            
            # Se ci sono strumenti che generano immagini, mostra tutto insieme
            if tool_messages:
                try:
                    all_image_paths = []
                    
                    for tool_msg in tool_messages:
                        tool_output = json.loads(tool_msg.content)
                        image_paths = tool_output.get("image_paths")

                        if image_paths is None:  # fallback per retrocompatibilità
                            image_path = tool_output.get("image_path") or tool_output.get("path")
                            if image_path:
                                image_paths = [image_path]
                            elif isinstance(tool_output, list):
                                image_paths = tool_output
                            else:
                                image_paths = []

                        if image_paths:
                            all_image_paths.extend(image_paths)
                    
                    if all_image_paths:
                        # Verifica se le immagini esistono effettivamente
                        valid_image_paths = [img_path for img_path in all_image_paths if os.path.exists(img_path)]
                        
                        if valid_image_paths:
                            # Se i chart sono stati generati correttamente, mostra solo le immagini
                            message_placeholder.markdown("Ecco le visualizzazioni generate:")
                            
                            # Display images
                            for img_path in valid_image_paths:
                                st.image(img_path, use_container_width=True)
                            
                            # Add only image message to chat history
                            chat_history.append({
                                "role": "bot",
                                "content": "Ecco le visualizzazioni generate:",
                                "image_paths": valid_image_paths
                            })
                        else:
                            # Le immagini non esistono, mostra la risposta testuale
                            message_placeholder.markdown(full_response)
                            if full_response and not full_response.startswith("❌"):
                                chat_history.append({"role": "bot", "content": full_response})
                    else:
                        # Nessuna immagine nei tool messages, mostra la risposta testuale
                        message_placeholder.markdown(full_response)
                        if full_response and not full_response.startswith("❌"):
                            chat_history.append({"role": "bot", "content": full_response})

                except Exception as e:
                    st.error(f"Errore nell'estrazione dell'immagine: {str(e)}")
                    # Fallback: salva solo la risposta testuale
                    if full_response and not full_response.startswith("❌"):
                        chat_history.append({"role": "bot", "content": full_response})
            else:
                # Nessuno strumento immagine, salva solo la risposta testuale
                if full_response and not full_response.startswith("❌"):
                    chat_history.append({"role": "bot", "content": full_response})
                
        else:
            # Modalità tradizionale (senza streaming) - rimane uguale
            file_path_for_prompt = user_file_paths[0] if user_file_paths else None
            enhanced_user_input = enhance_user_input(config_chat, input_text, file_path_for_prompt)
            config = {"configurable": {"thread_id": thread_id}}

            events = compiled_graph.stream(
                {"messages": [{"role": "user", "content": enhanced_user_input}]},
                config,
                stream_mode="values",
            )

            last_event = None

            for count, event in enumerate(events, start=1):
                print(f"event {count}: {event}\n")
                last_event = event

            if last_event is not None:
                ai_messages = [
                    msg.content for msg in last_event.get("messages", [])
                    if type(msg).__name__ == "AIMessage"
                ]

                tool_messages = [
                    msg for msg in last_event.get("messages", [])
                    if type(msg).__name__ == "ToolMessage" and msg.name in ["image_generator", "data_viz_tool"]
                ]

                if ai_messages:
                    bot_response = ai_messages[-1]
                    
                    # Process image from tool message
                    if tool_messages:
                        try:
                            all_image_paths = []
                            
                            for tool_msg in tool_messages:
                                tool_output = json.loads(tool_msg.content)
                                image_paths = tool_output.get("image_paths")

                                if image_paths is None:  # fallback per retrocompatibilità
                                    image_path = tool_output.get("image_path") or tool_output.get("path")
                                    if image_path:
                                        image_paths = [image_path]
                                    elif isinstance(tool_output, list):
                                        image_paths = tool_output
                                    else:
                                        image_paths = []

                                if image_paths:
                                    all_image_paths.extend(image_paths)

                            # Verifica se le immagini esistono effettivamente
                            valid_image_paths = [img_path for img_path in all_image_paths if os.path.exists(img_path)]
                            
                            if valid_image_paths:
                                # Se i chart sono stati generati correttamente, mostra solo le immagini
                                st.markdown("Ecco le visualizzazioni generate:")
                                
                                # Display images
                                for img_path in valid_image_paths:
                                    st.image(img_path, use_container_width=True)
                                
                                chat_history.append({
                                    "role": "bot",
                                    "content": "Ecco le visualizzazioni generate:",
                                    "image_paths": valid_image_paths
                                })
                            else:
                                # Le immagini non esistono, mostra la risposta testuale
                                st.markdown(bot_response)
                                chat_history.append({"role": "bot", "content": bot_response})

                        except Exception as e:
                            st.error(f"Errore nell'estrazione dell'immagine: {str(e)}")
                            # Fallback: mostra solo il testo
                            st.markdown(bot_response)
                            chat_history.append({"role": "bot", "content": bot_response})
                    else:
                        # Solo testo, nessuna immagine
                        st.markdown(bot_response)
                        chat_history.append({"role": "bot", "content": bot_response})
                else:
                    st.write("No AI message found.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    # Salva la cronologia aggiornata
    save_chat_history(thread_id, chat_history)


def transcribe_audio(audio_file):
    """Use AssemblyAI to transcribe audio."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_audio_file:
            tmp_audio_file.write(audio_file.getvalue())
            tmp_audio_file.close()

            audio = AudioSegment.from_file(tmp_audio_file.name)
            wav_audio_path = tmp_audio_file.name.replace(".tmp", ".wav")
            audio.export(wav_audio_path, format="wav")

            transcriber = AssemblyAISpeechToText()
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
            elif file_type.startswith('text') or file_type in ['application/vnd.ms-excel', 'text/csv']: 
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

        return repr(os.path.normpath(tmp_file_path))  # return tmp_file_path 

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

def play_text_to_speech(text, key):
    """Call ElevenLabs TTS tool and play the generated speech."""
    if st.button("🔊", key=key):
        audio_path = ElevenLabsTTSWrapper().text_to_speech(text)
        st.audio(audio_path, format="audio/mp3")


def voice_chat_input():
    """Chat input con microfono posizionata in basso"""
    
    # CSS per styling personalizzato e posizionamento in basso
    st.markdown("""
    <style>
    /* Container principale fisso in basso */
    .bottom-chat-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(0, 0, 0, 0.1);
        padding: 15px 20px;
        z-index: 1000;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .chat-container {
        display: flex;
        align-items: center;
        gap: 12px;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Bottone microfono minimale */
    .voice-button {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        width: 44px !important;
        height: 44px !important;
        border-radius: 50% !important;
        font-size: 20px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        color: #666 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .voice-button:hover {
        background: rgba(0, 0, 0, 0.05) !important;
        color: #333 !important;
        transform: scale(1.1) !important;
    }
    
    .voice-button.active {
        color: #ff4444 !important;
        animation: pulse-mic 1.5s infinite !important;
    }
    
    @keyframes pulse-mic {
        0% { 
            color: #ff4444;
            transform: scale(1);
        }
        50% { 
            color: #ff6666;
            transform: scale(1.1);
        }
        100% { 
            color: #ff4444;
            transform: scale(1);
        }
    }
    
    /* Nascondere i bottoni Streamlit predefiniti nel microfono */
    .stButton > button {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Spazio per il contenuto principale */
    .main .block-container {
        padding-bottom: 100px !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .bottom-chat-container {
            padding: 12px 15px;
        }
        
        .chat-container {
            gap: 8px;
        }
        
        .voice-button {
            width: 40px !important;
            height: 40px !important;
            font-size: 18px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Container per input e microfono
    chat_container = st.container()
    
    with chat_container:
        col1, col2 = st.columns([10, 1])
        
        with col1:
            user_submission = st.chat_input(
                placeholder="💬 Scrivi qui o usa il microfono →",
                key="main_chat_input_bottom",
                accept_file="multiple",
                file_type=['txt', 'pdf', 'png', 'jpg', 'jpeg', 'mp3', 'wav', 'mp4', 'csv', 'json', 'xml']
            )
        
        with col2:
            # Bottone microfono minimale
            voice_active = st.session_state.get('voice_mode', False)
            button_emoji = "🔴" if voice_active else "🎤"
            
            voice_button = st.button(
                button_emoji,
                key="voice_button_bottom",
                help="🎤 Registra messaggio vocale" if not voice_active else "🔴 Modalità vocale attiva",
                use_container_width=True
            )
    
    # Posiziona il container in basso usando float
    try:
        from streamlit_float import float_init
        chat_container.float(
            css_string="""
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(0, 0, 0, 0.1);
            padding: 15px 20px;
            z-index: 1000;
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
            """
        )
    except:
        # Fallback se float non funziona - usa solo CSS
        pass
    
    return user_submission, voice_button


def main():
    """Main function con una sola chat input in basso"""

    # Layout logo + titolo affiancati
    col1, col2 = st.columns([1, 6])
    
    with col1:
        try:
            st.image("images/new_unichat_icon_hd.png", width=70)
        except:
            st.markdown("🤖")  # Emoji fallback se l'immagine non esiste
    
    with col2:
        st.markdown("""
        <h1 style='margin-top: 15px; color: #667eea; font-size: 2.2rem; font-weight: bold;'>
            UNIVOX: University Virtual Orchestrated eXpert
        </h1>
        """, unsafe_allow_html=True)
    
    st.markdown("---")  # Linea separatrice

    # Initialize session state
    initialize_session()
    
    try:
        from streamlit_float import float_init
        float_init()
    except:
        pass

    # Sidebar configuration
    config_thread_id, config_chat = sidebar_configuration()
   
    # Display chat history
    st.header("💬 Chat")
    display_chat_history(config_thread_id)

    # UNICA chat input con microfono in basso
    user_submission, voice_button = voice_chat_input()

    # Gestisci il toggle della modalità vocale
    if voice_button:
        st.session_state.voice_mode = not st.session_state.get('voice_mode', False)
        
        if st.session_state.voice_mode:
            st.toast("🎤 Modalità vocale attivata", icon="🔴")
        else:
            st.toast("⏹️ Modalità vocale disattivata", icon="✅")
        
        st.rerun()

    # Voice input quando attivata
    if st.session_state.get('voice_mode', False):
        # Mostra l'audio input nella parte principale (non in basso)
        with st.container():
            st.info("🎤 Modalità vocale attivata - Registra il tuo messaggio")
            user_audio_input = st.audio_input(
                "Registra il tuo messaggio vocale",
                key="voice_recorder"
            )
            
            if user_audio_input is not None:
                with st.spinner("✍️ Trascrivendo audio..."):
                    transcribed_text = transcribe_audio(user_audio_input)
                    if transcribed_text:
                        st.success(f"Testo trascritto: {transcribed_text}")
                        
                        # Mostra il messaggio dell'utente
                        with st.chat_message("user"):
                            st.write(f"🎤 {transcribed_text}")
                        
                        # Processa la risposta del chatbot
                        with st.chat_message("assistant"):
                            handle_chatbot_response(transcribed_text, config_thread_id, config_chat, None)
                        
                        # Disattiva modalità vocale dopo l'uso
                        st.session_state.voice_mode = False
                        st.rerun()
    
    # Gestisci l'input testuale dalla chat in basso
    if user_submission:
        # Estrai testo e file dalla submission
        if hasattr(user_submission, 'text') and hasattr(user_submission, 'files'):
            # È un oggetto ChatInputValue
            user_input = user_submission.text
            user_files = user_submission.files
        elif isinstance(user_submission, str):
            # È una stringa semplice
            user_input = user_submission
            user_files = []
        else:
            # Fallback
            user_input = str(user_submission) if user_submission else ""
            user_files = []
        
        # Mostra il messaggio dell'utente
        with st.chat_message("user"):
            if user_input:
                st.write(user_input)
            # Mostra i file caricati se presenti
            if user_files:
                for file in user_files:
                    st.write(f"📎 {file.name}")
        
        # Processa la risposta del chatbot
        with st.chat_message("assistant"):
            if st.session_state.get('streaming_enabled', True):
                with st.spinner("🤔 Sto pensando..."):
                    handle_chatbot_response(user_input, config_thread_id, config_chat, user_files)
            else:
                handle_chatbot_response(user_input, config_thread_id, config_chat, user_files)
        
        st.rerun()

if __name__ == "__main__":
    main()