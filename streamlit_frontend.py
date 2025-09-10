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
from typing import AsyncGenerator, List, Optional

from streamlit_float import *
from streamlit_extras.bottom_container import bottom

from study_buddy.agent import compiled_graph
from study_buddy.utils.tools import AudioProcessor

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
    """Retrieves the chat history associated to a specific thread_id."""
    if thread_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[thread_id] = []
    return st.session_state.chat_histories[thread_id]

def add_message_to_history(thread_id: str, role: str, content: str, file_paths: Optional[List[str]] = None):
    """Centralizes adding messages to chat history with enhanced file path support (versione corretta)."""
    chat_history = get_chat_history(thread_id)
    message = {"role": role, "content": content}
    
    if file_paths:
        # Rimuovi duplicati mantenendo l'ordine
        unique_file_paths = []
        seen = set()
        for fp in file_paths:
            normalized_fp = fp.replace("\\\\", "\\").replace("\\", os.sep).strip("'\"")
            if normalized_fp not in seen:
                seen.add(normalized_fp)
                unique_file_paths.append(normalized_fp)
        
        # Separa immagini da altri file
        image_paths = [fp for fp in unique_file_paths if fp.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'))]
        other_files = [fp for fp in unique_file_paths if not fp.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'))]
        
        if image_paths:
            message["image_paths"] = image_paths
        if other_files:
            message["file_paths"] = other_files
    
    chat_history.append(message)
    save_chat_history(thread_id, chat_history)


def format_message_content(message):
    """Formats the message content for display"""
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
    """Formats tool calls for display"""
    if hasattr(message, 'tool_calls') and message.tool_calls:
        tool_info = []
        for tool_call in message.tool_calls:
            tool_info.append(f"🔧 **{tool_call['name']}**")
            if 'args' in tool_call:
                for key, value in tool_call['args'].items():
                    tool_info.append(f"  - {key}: `{value}`")
        return "\n".join(tool_info)
    return None

def display_images_and_files(content, file_paths_list=None, message_index=0):
    """Displays images inline and download buttons for any file - IMPROVED VERSION."""
    if not file_paths_list:
        print("[DEBUG] No file paths provided to display_images_and_files")
        return
    
    print(f"[DEBUG] display_images_and_files called with {len(file_paths_list)} files")
    
    download_counter = 0
    processed_files = set()  # Per evitare duplicati
    
    # Mostra i file passati esplicitamente dal tool
    for raw_path in file_paths_list:
        # Normalizza il path rimuovendo escape characters
        norm_path = raw_path.replace("\\\\", "\\").replace("\\", os.sep).replace("/", os.sep)
        # Rimuovi anche eventuali apici singoli o doppi all'inizio e fine
        norm_path = norm_path.strip("'\"")
        # Converti in percorso assoluto
        abs_path = os.path.abspath(norm_path)
        
        print(f"[DEBUG] Processing file for display: {raw_path} -> {abs_path}")
        
        if abs_path in processed_files:
            print(f"[DEBUG] Skipping duplicate file: {abs_path}")
            continue
        processed_files.add(abs_path)
        
        if os.path.exists(abs_path):
            ext = abs_path.lower()
            file_name = os.path.basename(abs_path)
            
            print(f"[DEBUG] File exists, displaying: {file_name}")
            
            # Display image inline if it's an image file
            if ext.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg")):
                try:
                    st.image(abs_path, caption=file_name, width='content')
                    print(f"[DEBUG] Successfully displayed image: {file_name}")
                except Exception as e:
                    st.error(f"Error displaying image {file_name}: {str(e)}")
                    print(f"[DEBUG] Error displaying image: {e}")
            
            # Always provide download button
            mime_type = mimetypes.guess_type(abs_path)[0] or "application/octet-stream"
            
            try:
                with open(abs_path, "rb") as f:
                    file_bytes = f.read()
                
                # Choose appropriate icon
                if ext.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg")):
                    icon = "🖼️"
                elif ext.endswith((".pdf",)):
                    icon = "📄"
                elif ext.endswith((".csv", ".xlsx", ".xls")):
                    icon = "📊"
                elif ext.endswith((".txt", ".md")):
                    icon = "📝"
                elif ext.endswith((".json", ".xml")):
                    icon = "🗂️"
                else:
                    icon = "📥"
                
                # Create unique key for download button
                file_hash = hash(abs_path) % 10000
                unique_key = f"download_{message_index}_{file_hash}_{download_counter}"
                
                st.download_button(
                    label=f"{icon} Scarica {file_name}",
                    data=file_bytes,
                    file_name=file_name,
                    mime=mime_type,
                    key=unique_key,
                    width='content'
                )
                download_counter += 1
                print(f"[DEBUG] Created download button for: {file_name}")
                
            except Exception as e:
                st.error(f"Error reading file {file_name}: {str(e)}")
                print(f"[DEBUG] Error reading file: {e}")
        else:
            st.error(f"File not found: {abs_path}")
            print(f"[DEBUG] File not found: {abs_path}")


def display_chat_history(thread_id):
    """Display the conversation history with enhanced file download support (versione corretta)."""
    chat_history = get_chat_history(thread_id)
    
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(chat_history):
            role = message.get("role", "unknown")
            content = format_message_content(message)
            
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content if content else "[Messaggio vuoto]")
            
            elif role == "bot" or role == "assistant":
                with st.chat_message("assistant"):
                    if content:
                        cleaned_content = re.sub(r'!\[.*?\]\((.*?)\)', '', content).strip()
                        if cleaned_content:
                            st.markdown(cleaned_content)
                    
                    all_file_paths = []
                    
                    # Aggiungi image_paths se esistono
                    if "image_paths" in message:
                        all_file_paths.extend(message["image_paths"])
                    
                    # Aggiungi file_paths se esistono
                    if "file_paths" in message:
                        all_file_paths.extend(message["file_paths"])
                    
                    # Rimuovi duplicati
                    unique_file_paths = []
                    seen = set()
                    for fp in all_file_paths:
                        normalized = fp.replace("\\\\", "\\").replace("\\", os.sep).strip("'\"")
                        if normalized not in seen:
                            seen.add(normalized)
                            unique_file_paths.append(normalized)
                    
                    if unique_file_paths:
                        display_images_and_files(content, unique_file_paths, i)
                    
                    # Mostra informazioni sui tool utilizzati
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        tool_calls_formatted = format_tool_calls(message)
                        if tool_calls_formatted:
                            with st.expander("🔧 Strumenti utilizzati"):
                                st.markdown(tool_calls_formatted)
                    
                    if content:
                        play_text_to_speech(content, key=f"tts_button_{i}")
            
            elif hasattr(message, 'name') and message.name:
                with st.chat_message("assistant"):
                    with st.expander(f"📋 Risultato: {message.name}"):
                        st.code(format_message_content(message), language="text")



def process_tool_messages_for_images(tool_messages):
    """Processes tool messages to extract valid image and file paths - IMPROVED VERSION."""
    all_file_paths = []
    
    for tool_msg in tool_messages:
        try:
            # Get content from tool message
            if hasattr(tool_msg, 'content'):
                content = tool_msg.content
            else:
                content = str(tool_msg)
            
            print(f"[DEBUG] Processing tool message content: {content[:200]}...")
            
            try:
                # Try to parse as JSON first
                tool_output = json.loads(content)
                print(f"[DEBUG] Parsed JSON tool output: {tool_output}")
            except json.JSONDecodeError:
                # If not JSON, search for file paths in the text
                print(f"[DEBUG] Not JSON, searching for file paths in text")
                
                # Look for common path patterns (both Windows and Unix)
                path_patterns = re.findall(r'([A-Za-z]:[\\\/][^"\'\s]+\.[a-zA-Z0-9]+|\/[^"\'\s]+\.[a-zA-Z0-9]+|[^"\'\s]*visualizations[\\\/][^"\'\s]+\.[a-zA-Z0-9]+)', content)
                
                for path in path_patterns:
                    normalized_path = path.replace("\\\\", "\\").replace("\\", os.sep).replace("/", os.sep).strip("'\"")
                    abs_path = os.path.abspath(normalized_path)
                    print(f"[DEBUG] Found path pattern: {path} -> normalized: {normalized_path} -> absolute: {abs_path}")
                    
                    if os.path.exists(abs_path):
                        all_file_paths.append(abs_path)
                        print(f"[DEBUG] Added existing file: {abs_path}")
                    else:
                        print(f"[DEBUG] File doesn't exist: {abs_path}")
                
                # Also look for quoted paths
                quoted_patterns = re.findall(r'["\']([^"\']+\.[a-zA-Z0-9]+)["\']', content)
                for path in quoted_patterns:
                    normalized_path = path.replace("\\\\", "\\").replace("\\", os.sep).replace("/", os.sep)
                    abs_path = os.path.abspath(normalized_path)
                    
                    if os.path.exists(abs_path) and abs_path not in all_file_paths:
                        all_file_paths.append(abs_path)
                        print(f"[DEBUG] Added quoted path: {abs_path}")
                
                continue
            
            # Process JSON output
            file_paths = []
            
            # Check various possible keys in the JSON response
            for key in ["file_paths", "image_paths", "file_path", "image_path", "path", "saved_paths", "output_files"]:
                if key in tool_output:
                    value = tool_output[key]
                    if isinstance(value, list):
                        file_paths.extend(value)
                    elif isinstance(value, str):
                        file_paths.append(value)
            
            # If tool_output is directly a list of paths
            if isinstance(tool_output, list):
                file_paths.extend(tool_output)
            
            # If tool_output is directly a string path
            elif isinstance(tool_output, str) and ("." in tool_output and ("/" in tool_output or "\\" in tool_output)):
                file_paths.append(tool_output)
            
            # Process all found file paths
            for file_path in file_paths:
                if file_path:
                    # Clean and normalize the path
                    normalized_path = str(file_path).replace("\\\\", "\\").replace("\\", os.sep).replace("/", os.sep).strip("'\"")
                    abs_path = os.path.abspath(normalized_path)
                    
                    print(f"[DEBUG] Processing file path: {file_path} -> {abs_path}")
                    
                    if os.path.exists(abs_path):
                        all_file_paths.append(abs_path)
                        print(f"[DEBUG] Successfully added file: {abs_path}")
                    else:
                        print(f"[DEBUG] File not found: {abs_path}")
                        
        except Exception as e:
            print(f"[ERROR] Error processing tool message: {e}")
            print(f"[ERROR] Tool message content: {content if 'content' in locals() else 'No content'}")
            continue
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in all_file_paths:
        normalized = os.path.abspath(path.replace("\\\\", "\\").replace("\\", os.sep).strip("'\""))
        if normalized not in seen:
            seen.add(normalized)
            unique_paths.append(normalized)
    
    print(f"[DEBUG] Final unique paths: {unique_paths}")
    return unique_paths


async def handle_streaming_events(events_generator):
    """Handles real-time streaming of agent events with improved download support - IMPROVED VERSION."""
    full_response = ""
    tool_messages = []
    message_placeholder = st.empty()
    
    async for event in events_generator:
        for node_name, node_output in event.items():
            print(f"[DEBUG] Processing event from node: {node_name}")
            
            if node_name == "tools" and "messages" in node_output:
                print(f"[DEBUG] Found {len(node_output['messages'])} tool messages")
                for message in node_output["messages"]:
                    if hasattr(message, 'name'):
                        tool_messages.append(message)
                        print(f"[DEBUG] Added tool message: {message.name}")
            
            elif node_name == "agent" and "messages" in node_output:
                for message in node_output["messages"]:
                    if hasattr(message, 'content') and message.content:
                        for char in message.content:
                            full_response += char
                            message_placeholder.markdown(full_response + "▌")
                            await asyncio.sleep(0.01)
    
    if full_response:
        message_placeholder.markdown(full_response)
    
    print(f"[DEBUG] Processing {len(tool_messages)} tool messages for file extraction")
    valid_file_paths = process_tool_messages_for_images(tool_messages)
    
    if valid_file_paths:
        print(f"[DEBUG] Found {len(valid_file_paths)} valid file paths")
        st.markdown("---")
        st.markdown("📁 **File generati:**")
        display_images_and_files("", valid_file_paths, 999)
        
        return full_response, valid_file_paths
    else:
        print("[DEBUG] No valid file paths found")
    
    return full_response, None


def handle_non_streaming_events(events):
    """Handles agent events in non-streaming mode with improved download support - IMPROVED VERSION."""
    full_response = ""
    tool_messages = []
    
    for event in events:
        for node_name, node_output in event.items():
            print(f"[DEBUG] Processing event from node: {node_name}")
            
            if node_name == "tools" and "messages" in node_output:
                print(f"[DEBUG] Found {len(node_output['messages'])} tool messages")
                for message in node_output["messages"]:
                    if hasattr(message, 'name'):
                        tool_messages.append(message)
                        print(f"[DEBUG] Added tool message: {message.name}")
            
            elif node_name == "agent" and "messages" in node_output:
                for message in node_output["messages"]:
                    if hasattr(message, 'content') and message.content:
                        full_response += message.content
    
    if full_response:
        st.markdown(full_response)
    
    print(f"[DEBUG] Processing {len(tool_messages)} tool messages for file extraction")
    valid_file_paths = process_tool_messages_for_images(tool_messages)
    
    if valid_file_paths:
        print(f"[DEBUG] Found {len(valid_file_paths)} valid file paths")
        st.markdown("---")
        st.markdown("📁 **File generati:**")
        display_images_and_files("", valid_file_paths, 999)
        
        return full_response, valid_file_paths
    else:
        print("[DEBUG] No valid file paths found")
    
    return full_response, None


def sidebar_configuration():
    """Render the sidebar for configuration and guide."""
    with st.sidebar:
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

        with st.expander("Chat configuration"):
            config_complexity_level = st.selectbox(
                "Level of complexity of responses",
                ('None', 'Base', 'Intermediate', 'Advanced')
            )
            config_course = st.selectbox(
                "Which course do you want to delve in?",
                ("None", "Semantics in Intelligent Information Access", "Metodi per il Ritrovamento dell'Informazione")
            )
            
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

        st.divider()
        
        with st.expander("Cosa puoi fare con UNIVOX"):
            st.markdown(
                """
                **UNIVOX** è un agente AI con oltre 20 strumenti integrati per supportarti nel tuo percorso di studio e ricerca.

                ## **Ricerca e Studio**
                - **Ricerca web avanzata** con Tavily per informazioni aggiornate
                - **Ricerca accademica** su ArXiv, Google Scholar e PubMed
                - **Consultazione** di Wikipedia, Wikidata e Google Books
                - **Recupero documenti** dal tuo archivio personale indicizzato

                ## **Elaborazione Documenti**
                - **Riassunti automatici** di PDF e file di testo
                - **Estrazione testo** da PDF scansionati e immagini (OCR)
                - **Analisi del sentiment** per valutare il tono emotivo dei contenuti
                - **Analisi CSV** con statistiche descrittive e insight semantici

                ## **Visualizzazione Dati**
                - **Generazione grafici** automatica da dataset CSV
                - **Codice Python** eseguibile in sandbox sicuro
                - **Interpretazione** di query in linguaggio naturale per creare visualizzazioni

                ## **Strumenti Multimodali**
                - **Riconoscimento immagini** con Google Lens
                - **Text-to-Speech** con voci naturali ElevenLabs
                - **Speech-to-Text** per trascrizioni accurate
                - **Input vocale** integrato nell'interfaccia

                ## **Funzionalità Avanzate**
                - **Esecuzione codice** in tempo reale per calcoli e analisi
                - **Ricerca musicale** su Spotify per il benessere mentale
                - **Ricerca video** su YouTube per contenuti educativi
                - **Supporto multilingua** per utenti internazionali

                ## **Come Utilizzare UNIVOX**

                ### **Interazione Base**
                - Digita domande in linguaggio naturale
                - Carica file trascinandoli nella chat
                - Usa comandi vocali cliccando il microfono

                ### **Gestione Conversazioni**
                - **Thread ID**: Identificativo univoco per ogni conversazione
                - **Nuova chat**: Modifica il Thread ID per iniziare da capo
                - **Riprendi chat**: Inserisci un Thread ID esistente per continuare

                ### **Suggerimenti per Risultati Ottimali**
                - Sii **specifico** nelle richieste ("Analizza questo PDF e creami un riassunto di 200 parole")
                - **Combina strumenti** ("Cerca su ArXiv articoli su ML e riassumi i primi 3")
                - **Fornisci contesto** per domande complesse
                - **Sperimenta** con diverse formulazioni

                ## **Note Tecniche**
                - Sandbox isolato per l'esecuzione sicura del codice
                - Archivio vettoriale per ricerca semantica nei documenti
                - API multiple integrate per massima copertura informativa
                - Risposte non deterministiche per natura dell'AI
                """,
                unsafe_allow_html=True
            )

    return config_thread_id, config_chat

def enhance_user_input(config_chat, user_input, file_path):
    """Versione semplificata che non interferisce con il sistema prompt principale."""
    context_instructions = []
    
    # Lingua
    if config_chat.language:
        context_instructions.append(f"Respond in {config_chat.language}.")
    
    # Livello di complessità
    complexity_map = {
        "Basic": "Use simple language and basic explanations.",
        "Intermediate": "Provide moderate detail with some technical terms.",
        "Advanced": "Give comprehensive analysis with technical depth."
    }
    
    if config_chat.complexity_level in complexity_map:
        context_instructions.append(complexity_map[config_chat.complexity_level])
    
    # Contesto del corso
    if config_chat.course and config_chat.course != "None":
        context_instructions.append(f"Context: User is studying {config_chat.course}. Prioritize course-related materials, then expand if needed.")
    
    # File da analizzare
    if file_path:
        context_instructions.append(f"User has uploaded a file for analysis: {file_path}")
    
    # Istruzioni per formule matematiche
    context_instructions.append("For mathematical formulas, use LaTeX notation: inline formulas with $formula$ and display formulas with $$formula$$.")
    
    # Costruisci il prompt finale
    if context_instructions:
        context_block = "\n".join(f"• {instr}" for instr in context_instructions)
        return f"{context_block}\n\nUser Query: {user_input}"
    
    return user_input

def save_chat_history(thread_id, chat_history):
    """Save chat history to file."""
    os.makedirs("history", exist_ok=True)
    with open(f"history/{thread_id}.json", "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

def save_uploaded_file(user_file_input):
    """Saves any file uploaded by the user to a dedicated folder and returns a clean path."""
    try:
        upload_dir = os.path.join(os.getcwd(), "uploaded_files")
        os.makedirs(upload_dir, exist_ok=True)

        tmp_file_path = os.path.join(upload_dir, user_file_input.name)
        print(f"Saving uploaded file to: {tmp_file_path}")

        with open(tmp_file_path, 'wb') as f:
            f.write(user_file_input.getvalue())

        # Controlliamo il tipo di file per mostrare un messaggio coerente
        file_type, _ = mimetypes.guess_type(user_file_input.name)
        if file_type:
            if file_type.startswith('audio'):
                st.success("File audio caricato")
            elif file_type.startswith('image'):
                st.success("File immagine caricato")
            elif file_type.startswith('video'):
                st.success("File video caricato")
            elif file_type == 'application/pdf':
                st.success("File PDF caricato")
            else:
                st.success("File caricato")
        else:
            st.success("File caricato")

        return os.path.normpath(tmp_file_path)

    except Exception as e:
        st.error(f"Error in saving the file: {str(e)}")
        return ""


def get_clean_path(file_path: str) -> str:
    """Returns an absolute, normalized, and safe path."""
    # Rimuove eventuali apici e normalizza i separatori
    cleaned_path = file_path.replace("\\\\", "\\").replace("\\", os.sep).strip("'\"")
    # Converte sempre in percorso assoluto
    abs_path = os.path.abspath(cleaned_path)
    return abs_path


def handle_chatbot_response(user_input, thread_id, config_chat, user_files=None):
    """Handle chatbot response with unified streaming/non-streaming logic."""
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
        add_message_to_history(thread_id, "user", input_text)

    # Process uploaded files if any
    user_file_paths = []
    if user_files:
        for uploaded_file in user_files:
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                cleaned_path = get_clean_path(file_path)
                if os.path.exists(cleaned_path):
                    user_file_paths.append(cleaned_path)
                else:
                    st.error(f"File not found: {cleaned_path}")


    try:
        file_path_for_prompt = get_clean_path(user_file_paths[0]) if user_file_paths else None
        enhanced_user_input = enhance_user_input(config_chat, input_text, file_path_for_prompt)
        config = {"configurable": {"thread_id": thread_id}}
        
        if st.session_state.streaming_enabled:
            # Modalità streaming - streaming in tempo reale
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            async def run_streaming():
                events_generator = compiled_graph.astream(
                    {"messages": [{"role": "user", "content": enhanced_user_input}]}, 
                    config=config
                )
                return await handle_streaming_events(events_generator)
            
            bot_response, image_paths = loop.run_until_complete(run_streaming())
        else:
            # Modalità normale - tutto insieme
            events = list(compiled_graph.stream(
                {"messages": [{"role": "user", "content": enhanced_user_input}]},
                config,
                stream_mode="values",
            ))
            
            bot_response, image_paths = handle_non_streaming_events(events)

        if bot_response and not bot_response.startswith("❌"):
            add_message_to_history(thread_id, "bot", bot_response, image_paths)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def transcribe_audio(audio_file):
    """Use AssemblyAI to transcribe audio."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_audio_file:
            tmp_audio_file.write(audio_file.getvalue())
            tmp_audio_file.close()

            # Convert to WAV format
            audio = AudioSegment.from_file(tmp_audio_file.name)
            wav_audio_path = tmp_audio_file.name.replace(".tmp", ".wav")
            audio.export(wav_audio_path, format="wav")

            # Use the new AudioProcessor class
            processor = AudioProcessor()
            transcribed_text = processor.speech_to_text(wav_audio_path)

            return transcribed_text
    except ValueError as e:
        st.error(f"Audio transcription setup error: {str(e)}")
        return ""
    except Exception as e:
        st.error(f"Audio transcription error: {str(e)}")
        return ""

def play_text_to_speech(text, key):
    """Call ElevenLabs TTS tool and play the generated speech."""
    if st.button("🔊", key=key):
        try:
            # Use the new AudioProcessor class
            processor = AudioProcessor()
            audio_path = processor.text_to_speech(text)
            
            if audio_path and not audio_path.startswith("Error"):
                st.audio(audio_path, format="audio/mp3")
            else:
                st.error(f"TTS error: {audio_path}")
        except ValueError as e:
            st.error(f"TTS setup error: {str(e)}")
        except Exception as e:
            st.error(f"TTS error: {e}")

def voice_chat_input():
    """Handles voice and text input at the bottom of the interface."""
    float_init()

    with bottom():
        col1, col2 = st.columns([10, 1])
        
        with col1:
            user_submission = st.chat_input(
                placeholder="Scrivi qui o usa il microfono",
                key="main_chat_input_bottom",
                accept_file="multiple",
                file_type=['txt', 'pdf', 'png', 'jpg', 'jpeg', 'mp3', 'wav', 'mp4', 'csv', 'json', 'xml']
            )
        
        with col2:
            voice_active = st.session_state.get('voice_mode', False)
            button_emoji = "🔴" if voice_active else "🎤"
            
            voice_button = st.button(
                button_emoji,
                key="voice_button_bottom",
                help="Registra messaggio vocale" if not voice_active else "Modalità vocale attiva",
                width='content'
            )

    return user_submission, voice_button

def main():
    initialize_session()
    
    try:
        from streamlit_float import float_init
        float_init()
    except:
        pass

    config_thread_id, config_chat = sidebar_configuration()
    chat_history = get_chat_history(config_thread_id)

    if not chat_history:
        st.title("Come posso aiutarti oggi?")
    else:
        display_chat_history(config_thread_id)

    user_submission, voice_button = voice_chat_input()

    if voice_button:
        st.session_state.voice_mode = not st.session_state.get('voice_mode', False)
        
        if st.session_state.voice_mode:
            st.toast("🎤 Modalità vocale attivata", icon="🔴")
        else:
            st.toast("⏹️ Modalità vocale disattivata", icon="✅")
        
        st.rerun()

    if st.session_state.get('voice_mode', False):
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
                        
                        with st.chat_message("user"):
                            st.write(f"🎤 {transcribed_text}")
                        
                        with st.chat_message("assistant"):
                            handle_chatbot_response(transcribed_text, config_thread_id, config_chat, None)
                        
                        st.session_state.voice_mode = False
                        st.rerun()
    
    if user_submission:
        # Estrai testo e file dalla submission
        if hasattr(user_submission, 'text') and hasattr(user_submission, 'files'):
            user_input = user_submission.text
            user_files = user_submission.files
        elif isinstance(user_submission, str):
            user_input = user_submission
            user_files = []
        else:
            user_input = str(user_submission) if user_submission else ""
            user_files = []
        
        with st.chat_message("user"):
            if user_input:
                st.write(user_input)
            if user_files:
                for file in user_files:
                    st.write(f"📎 {file.name}")
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Sto pensando..."):
                handle_chatbot_response(user_input, config_thread_id, config_chat, user_files)
        
        st.rerun()

if __name__ == "__main__":
    main()