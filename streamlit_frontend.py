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

def display_images_and_files(content, file_paths_list=None, message_index=0):
    """Mostra immagini inline e pulsanti di download per qualsiasi file (versione corretta)."""
    if not file_paths_list:
        return
    
    download_counter = 0
    processed_files = set()  # Per evitare duplicati
    
    # Mostra i file passati esplicitamente dal tool
    for raw_path in file_paths_list:
        # Normalizza il path rimuovendo escape characters
        norm_path = raw_path.replace("\\\\", "\\").replace("\\", os.sep)
        # Rimuovi anche eventuali apici singoli o doppi all'inizio e fine
        norm_path = norm_path.strip("'\"")
        
        if norm_path in processed_files:
            continue
        processed_files.add(norm_path)
        
        if os.path.exists(norm_path):
            ext = norm_path.lower()
            file_name = os.path.basename(norm_path)
            
            if ext.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg")):
                st.image(norm_path, caption=file_name, use_container_width=True)
            
            mime_type = mimetypes.guess_type(norm_path)[0] or "application/octet-stream"
            
            try:
                with open(norm_path, "rb") as f:
                    file_bytes = f.read()
                
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
                
                # Key unica basata su hash del path per evitare conflitti
                file_hash = hash(norm_path) % 10000  # Limita la lunghezza dell'hash
                unique_key = f"download_{message_index}_{file_hash}_{download_counter}"
                
                st.download_button(
                    label=f"{icon} Scarica {file_name}",
                    data=file_bytes,
                    file_name=file_name,
                    mime=mime_type,
                    key=unique_key,
                    use_container_width=True
                )
                download_counter += 1
                
            except Exception as e:
                st.error(f"Errore nella lettura del file {file_name}: {str(e)}")
        else:
            st.error(f"File non trovato: {norm_path}")


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
    """Processa i messaggi dei tool per estrarre percorsi delle immagini e file validi (versione corretta)."""
    all_file_paths = []
    
    for tool_msg in tool_messages:
        try:
            # Prova a parsare come JSON
            if hasattr(tool_msg, 'content'):
                content = tool_msg.content
            else:
                content = str(tool_msg)
            
            try:
                tool_output = json.loads(content)
            except json.JSONDecodeError:
                # Se non è JSON, tratta come stringa e cerca pattern di path
                path_patterns = re.findall(r'["\']([^"\']+\.[a-zA-Z0-9]+)["\']', content)
                for path in path_patterns:
                    normalized_path = path.replace("\\\\", "\\").replace("\\", os.sep).strip("'\"")
                    if os.path.exists(normalized_path):
                        all_file_paths.append(normalized_path)
                continue
            
            file_paths = []
            
            if "file_paths" in tool_output:
                file_paths.extend(tool_output["file_paths"])
            
            if "image_paths" in tool_output:
                file_paths.extend(tool_output["image_paths"])
            
            if "file_path" in tool_output:
                file_paths.append(tool_output["file_path"])
            elif "image_path" in tool_output:
                file_paths.append(tool_output["image_path"])
            elif "path" in tool_output:
                file_paths.append(tool_output["path"])
            
            elif isinstance(tool_output, list):
                file_paths.extend(tool_output)
            
            elif isinstance(tool_output, str) and "." in tool_output:
                file_paths.append(tool_output)
            
            for file_path in file_paths:
                if file_path:
                    normalized_path = str(file_path).replace("\\\\", "\\").replace("\\", os.sep).strip("'\"")
                    if os.path.exists(normalized_path):
                        all_file_paths.append(normalized_path)
                        
        except Exception as e:
            st.error(f"Errore nel parsing del tool message: {e}")
            continue
    
    seen = set()
    unique_paths = []
    for path in all_file_paths:
        normalized = path.replace("\\\\", "\\").replace("\\", os.sep).strip("'\"")
        if normalized not in seen:
            seen.add(normalized)
            unique_paths.append(normalized)
    
    return unique_paths


async def handle_streaming_events(events_generator):
    """Gestisce lo streaming in tempo reale degli eventi dell'agente con download migliorato."""
    full_response = ""
    tool_messages = []
    message_placeholder = st.empty()
    
    async for event in events_generator:
        for node_name, node_output in event.items():
            if node_name == "tools" and "messages" in node_output:
                for message in node_output["messages"]:
                    if hasattr(message, 'name'):
                        tool_messages.append(message)
            
            elif node_name == "agent" and "messages" in node_output:
                for message in node_output["messages"]:
                    if hasattr(message, 'content') and message.content:
                        for char in message.content:
                            full_response += char
                            message_placeholder.markdown(full_response + "▌")
                            await asyncio.sleep(0.01)  # Effetto typing
    
    if full_response:
        message_placeholder.markdown(full_response)
    
    valid_file_paths = process_tool_messages_for_images(tool_messages)
    
    if valid_file_paths:
        st.markdown("---")
        st.markdown("📁 **File generati:**")
        display_images_and_files("", valid_file_paths, 999)
        
        return full_response, valid_file_paths
    
    return full_response, None

def handle_non_streaming_events(events):
    """Gestisce gli eventi dell'agente in modalità non-streaming con download migliorato."""
    full_response = ""
    tool_messages = []
    
    for event in events:
        for node_name, node_output in event.items():
            if node_name == "tools" and "messages" in node_output:
                for message in node_output["messages"]:
                    if hasattr(message, 'name'):
                        tool_messages.append(message)
            
            elif node_name == "agent" and "messages" in node_output:
                for message in node_output["messages"]:
                    if hasattr(message, 'content') and message.content:
                        full_response += message.content
    
    if full_response:
        st.markdown(full_response)
    
    valid_file_paths = process_tool_messages_for_images(tool_messages)
    
    if valid_file_paths:
        st.markdown("---")
        st.markdown("📁 **File generati:**")
        display_images_and_files("", valid_file_paths, 999)
        
        return full_response, valid_file_paths
    
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

        with st.expander(":gear: Chat configuration"):
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
    """Generates an optimized prompt for the chatbot with focused instructions."""
    core_instructions = []
    
    if config_chat.language:
        core_instructions.append(f"Respond in {config_chat.language}.")
    
    complexity_map = {
        "Basic": "Use simple language and basic explanations.",
        "Intermediate": "Provide moderate detail with some technical terms.",
        "Advanced": "Give comprehensive analysis with technical depth."
    }
    
    if config_chat.complexity_level in complexity_map:
        core_instructions.append(complexity_map[config_chat.complexity_level])
    
    if config_chat.course and config_chat.course != "None":
        core_instructions.append(f"Context: User is studying {config_chat.course}. Prioritize course-related materials, then expand if needed.")
    
    if file_path:
        core_instructions.append(f"Analyze uploaded file: {file_path}")
    
    core_instructions.extend([
        "Academic support agent: Never invent information. Use retrieve_tool for documents, web_search for current info. Always provide files paths if available.",
        "For mathematical formulas, use LaTeX notation: inline formulas with $formula$ and display formulas with $$formula$$.",
        "If no reliable sources found, clearly state limitations rather than guessing."
    ])
    
    if core_instructions:
        instruction_block = "\n".join(f"• {instr}" for instr in core_instructions)
        return f"{instruction_block}\n\nQuery: {user_input}"
    
    return user_input

def save_chat_history(thread_id, chat_history):
    """Save chat history to file."""
    os.makedirs("history", exist_ok=True)
    with open(f"history/{thread_id}.json", "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

def save_uploaded_file(user_file_input):
    """Salva qualsiasi tipo di file caricato dall'utente."""
    try:
        temp_dir = tempfile.mkdtemp()
        tmp_file_path = os.path.join(temp_dir, user_file_input.name)
        
        with open(tmp_file_path, 'wb') as f:
            f.write(user_file_input.getvalue())
        
        file_type, _ = mimetypes.guess_type(user_file_input.name)
        
        # Messaggio di successo basato sul tipo
        if file_type:
            if file_type.startswith('audio'):
                st.success("File audio caricato 📤")
            elif file_type.startswith('image'):
                st.success(f"File immagine caricato 📤")
            elif file_type.startswith('video'):
                st.success(f"File video caricato 📤")
            elif file_type == 'application/pdf':
                st.success("File PDF caricato 📤")
            else:
                st.success("File caricato 📤")
        else:
            st.success("File caricato 📤")
        
        return repr(os.path.normpath(tmp_file_path))
        
    except Exception as e:
        st.error(f"Errore durante il salvataggio del file: {str(e)}")
        return ""

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
                user_file_paths.append(file_path)

    try:
        file_path_for_prompt = user_file_paths[0] if user_file_paths else None
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

        # Salva la risposta del bot
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

            audio = AudioSegment.from_file(tmp_audio_file.name)
            wav_audio_path = tmp_audio_file.name.replace(".tmp", ".wav")
            audio.export(wav_audio_path, format="wav")

            transcriber = AssemblyAISpeechToText()
            transcribed_audio = transcriber.transcribe_audio(wav_audio_path)

            return transcribed_audio
    except Exception as e:
        st.error(f"Audio transcription error: {str(e)}")
        return ""

def play_text_to_speech(text, key):
    """Call ElevenLabs TTS tool and play the generated speech."""
    if st.button("🔊", key=key):
        audio_path = ElevenLabsTTSWrapper().text_to_speech(text)
        st.audio(audio_path, format="audio/mp3")

def voice_chat_input():
    """Gestisce l'input vocale e testuale nella parte inferiore."""
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
                help="🎤 Registra messaggio vocale" if not voice_active else "🔴 Modalità vocale attiva",
                use_container_width=True
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