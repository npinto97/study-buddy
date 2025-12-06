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
import sys
from datetime import datetime
import uuid 

from streamlit_float import *
from streamlit_extras.bottom_container import bottom

from study_buddy.agent import compiled_graph
from study_buddy.utils.tools import AudioProcessor
import yaml
from loguru import logger

# Fix per classi torch su Windows/alcuni ambienti
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# Set up the environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def update_llm_provider(provider: str):
    """Updates the LLM provider in config.yaml"""
    config_path = Path("config.yaml")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        config_data['llm']['provider'] = provider
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        
        return True
    except Exception as e:
        st.error(f"Errore nell'aggiornamento del provider: {e}")
        return False

# Setup session log file
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Conversation log setup
conversation_log = log_dir / f"conversation_{datetime.now().strftime('%Y-%m-%d')}.log"

def log_conversation(question: str, answer: str, sources: list = None, retrieved_docs: list = None):
    with open(conversation_log, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"\n{'='*80}\n[{timestamp}]\n\n❓ DOMANDA:\n{question}\n")
        f.write(f"\n📚 FONTI UTILIZZATE:\n")
        if sources:
            for i, source in enumerate(sources, 1):
                f.write(f"  {i}. {source}\n")
        else:
            f.write(f"  • Nessuno strumento utilizzato\n")
        
        if retrieved_docs:
            f.write(f"\n📄 DOCUMENTI RECUPERATI:\n")
            for i, doc in enumerate(retrieved_docs, 1):
                doc_preview = doc.replace('\n', ' ').strip()[:200]
                f.write(f"  {i}. {doc_preview}...\n")
        
        f.write(f"\n💬 RISPOSTA:\n{answer}\n{'='*80}\n\n")

    # Console Logging (Loguru) - Replicating file log for terminal visibility
    console_msg = f"\n❓ QUESTION:\n{question}\n"
    
    if sources:
        console_msg += "\n📚 SOURCES:\n" + "\n".join([f"  {i}. {s}" for i, s in enumerate(sources, 1)]) + "\n"
    
    if retrieved_docs:
        console_msg += "\n📄 RETRIEVED DOCS:\n" + "\n".join([f"  {i}. {d.replace(chr(10), ' ').strip()[:150]}..." for i, d in enumerate(retrieved_docs, 1)]) + "\n"
        
    console_msg += f"\n💬 ANSWER:\n{answer}\n"
    logger.info(console_msg)

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

try:
    st.logo(image=str(Path("images/univox_logo2.png")), size="large", icon_image=str(Path("images/univox_logo2.png")))
except:
    pass

class ConfigChat:
    def __init__(self, complexity_level="None", language="Italian", course="None"):
        self.complexity_level = complexity_level
        self.language = language
        self.course = course

# --- GESTIONE METADATI CHAT (Nomi vs ID) ---
METADATA_FILE = "history/metadata.json"

def load_chat_metadata():
    """Carica il mapping ID -> Nome. Se non esiste, crea dai file esistenti."""
    os.makedirs("history", exist_ok=True)
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        metadata = {}
        files = [f for f in os.listdir("history") if f.endswith(".json") and f != "metadata.json"]
        for f in files:
            thread_id = f.replace(".json", "")
            metadata[thread_id] = f"Chat {thread_id}"
        save_chat_metadata(metadata)
        return metadata

def save_chat_metadata(metadata):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def create_new_chat(name):
    new_id = str(uuid.uuid4())
    metadata = st.session_state.chat_metadata
    metadata[new_id] = name
    st.session_state.chat_metadata = metadata
    save_chat_metadata(metadata)
    return new_id

# -------------------------------------------

def initialize_session():
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    
    if "chat_metadata" not in st.session_state:
        st.session_state.chat_metadata = load_chat_metadata()
        
    if "voice_mode" not in st.session_state:
        st.session_state.voice_mode = False
    if "streaming_enabled" not in st.session_state:
        st.session_state.streaming_enabled = True
    if "active_thread_id" not in st.session_state:
        if st.session_state.chat_metadata:
            st.session_state.active_thread_id = list(st.session_state.chat_metadata.keys())[-1]
        else:
            st.session_state.active_thread_id = None

    os.makedirs(os.path.join(os.getcwd(), "uploaded_files", "audio"), exist_ok=True)

def get_chat_history(thread_id):
    if not thread_id: return []
    if thread_id not in st.session_state.chat_histories:
        history_path = f"history/{thread_id}.json"
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    st.session_state.chat_histories[thread_id] = json.load(f)
            except:
                st.session_state.chat_histories[thread_id] = []
        else:
            st.session_state.chat_histories[thread_id] = []
    return st.session_state.chat_histories[thread_id]

def add_message_to_history(thread_id: str, role: str, content: str, file_paths: Optional[List[str]] = None, audio_path: Optional[str] = None):
    chat_history = get_chat_history(thread_id)
    message = {"role": role, "content": content}
    
    if file_paths:
        unique_paths = list(dict.fromkeys([fp.replace("\\\\", "\\").replace("\\", "/").strip("'\"") for fp in file_paths]))
        image_paths = [fp for fp in unique_paths if fp.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
        other_files = [fp for fp in unique_paths if fp not in image_paths]
        
        if image_paths: message["image_paths"] = image_paths
        if other_files: message["file_paths"] = other_files
    
    if audio_path: message["audio_path"] = audio_path
    
    chat_history.append(message)
    save_chat_history(thread_id, chat_history)

def format_message_content(message):
    content = message.content if hasattr(message, 'content') else message.get('content', '') if isinstance(message, dict) else str(message)
    if content: content = re.sub(r"!\[.*?\]\(sandbox:.*?\)", "", content).strip()
    return content

def format_tool_calls(message):
    if hasattr(message, 'tool_calls') and message.tool_calls:
        return "\n".join([f"🔧 **{tc['name']}**" + (f" - Args: `{tc['args']}`" if 'args' in tc else "") for tc in message.tool_calls])
    return None

def display_images_and_files(content, file_paths_list=None, message_index=0):
    if not file_paths_list: return
    
    download_counter = 0
    processed_files = set()
    
    for raw_path in file_paths_list:
        abs_path = os.path.abspath(raw_path.replace("\\\\", "\\").replace("\\", "/").strip("'\""))
        if abs_path in processed_files: continue
        processed_files.add(abs_path)
        
        if os.path.exists(abs_path):
            file_name = os.path.basename(abs_path)
            ext = abs_path.lower()
            
            if ext.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                try: st.image(abs_path, caption=file_name, width='content')
                except Exception as e: st.error(f"Error image: {e}")
            
            mime_type = mimetypes.guess_type(abs_path)[0] or "application/octet-stream"
            try:
                with open(abs_path, "rb") as f:
                    icon = "🖼️" if ext.endswith((".png", ".jpg")) else "📊" if ext.endswith(".csv") else "📄" if ext.endswith(".pdf") else "📥"
                    st.download_button(
                        label=f"{icon} Scarica {file_name}",
                        data=f.read(),
                        file_name=file_name,
                        mime=mime_type,
                        key=f"dl_{message_index}_{download_counter}",
                        width='content'
                    )
                    download_counter += 1
            except Exception as e: st.error(f"Error file: {e}")

def display_chat_history(thread_id):
    chat_history = get_chat_history(thread_id)
    with st.container():
        for i, message in enumerate(chat_history):
            role = message.get("role", "unknown")
            content = format_message_content(message)
            
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content if content else "...")
                    if message.get("audio_path") and os.path.exists(message["audio_path"]):
                        st.audio(message["audio_path"], format="audio/wav")
                        st.caption("🎤 Vocale")
            
            elif role in ["bot", "assistant"]:
                with st.chat_message("assistant"):
                    if content:
                        st.markdown(re.sub(r'!\[.*?\]\((.*?)\)', '', content).strip())
                    
                    files = message.get("image_paths", []) + message.get("file_paths", [])
                    if files: display_images_and_files(content, files, i)
                    
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        with st.expander("🔧 Strumenti"):
                            st.markdown(format_tool_calls(message))
                    
                    if content: play_text_to_speech(content, key=f"tts_{i}")

def process_tool_messages_for_images(tool_messages):
    all_file_paths = []
    for tool_msg in tool_messages:
        try:
            if getattr(tool_msg, 'name', '') == 'extract_text': continue
            content = getattr(tool_msg, 'content', str(tool_msg))
            
            # Try JSON
            try:
                out = json.loads(content)
                for k in ["file_paths", "image_paths", "saved_paths"]:
                    if k in out: all_file_paths.extend(out[k] if isinstance(out[k], list) else [out[k]])
            except: pass
            
            # Regex paths
            patterns = re.findall(r'([A-Za-z]:[\\\/][^"\'\s]+\.[a-zA-Z0-9]+|[^"\'\s]*visualizations[\\\/][^"\'\s]+\.[a-zA-Z0-9]+)', content)
            all_file_paths.extend([p for p in patterns if os.path.exists(os.path.abspath(p.replace("\\\\", "\\").strip("'\"")))])
            
        except: continue
    
    return list(dict.fromkeys([os.path.abspath(p.replace("\\\\", "\\").strip("'\"")) for p in all_file_paths]))

async def handle_streaming_events(events_generator):
    full_response = ""
    tool_msgs = []
    srcs, docs = [], []
    placeholder = st.empty()
    
    async for event in events_generator:
        for node, output in event.items():
            if node == "tools" and "messages" in output:
                for msg in output["messages"]:
                    if hasattr(msg, 'name'):
                        tool_msgs.append(msg)
                        info = f"Tool: {msg.name}"
                        if msg.name == "retrieve_knowledge":
                            docs.append(str(msg.content)[:500])
                            info += f" ({len(docs)} docs)"
                        if info not in srcs: srcs.append(info)
            
            elif node == "agent" and "messages" in output:
                for msg in output["messages"]:
                    if hasattr(msg, 'content') and msg.content:
                        text = msg.content
                        if isinstance(text, list): text = "".join([x.get('text','') for x in text if x.get('type')=='text'])
                        for char in str(text):
                            full_response += char
                            placeholder.markdown(full_response + "▌")
                            await asyncio.sleep(0.005)
    
    placeholder.markdown(full_response)
    valid_files = process_tool_messages_for_images(tool_msgs)
    if valid_files:
        st.markdown("---")
        st.markdown("📁 **File generati:**")
        display_images_and_files("", valid_files, 999)
    
    return full_response, valid_files, srcs, docs

def handle_non_streaming_events(events):
    full_response = ""
    tool_msgs = []
    srcs, docs = [], []
    
    for event in events:
        for node, output in event.items():
            if node == "tools" and "messages" in output:
                for msg in output["messages"]:
                    if hasattr(msg, 'name'):
                        tool_msgs.append(msg)
                        info = f"Tool: {msg.name}"
                        if msg.name == "retrieve_knowledge":
                            docs.append(str(msg.content)[:500])
                        if info not in srcs: srcs.append(info)
            
            elif node == "agent" and "messages" in output:
                for msg in output["messages"]:
                    if hasattr(msg, 'content') and msg.content:
                        text = msg.content
                        if isinstance(text, list): text = "".join([x.get('text','') for x in text if x.get('type')=='text'])
                        full_response += str(text)
    
    if full_response: st.markdown(full_response, unsafe_allow_html=True)
    
    valid_files = process_tool_messages_for_images(tool_msgs)
    if valid_files:
        st.markdown("---")
        st.markdown("📁 **File generati:**")
        display_images_and_files("", valid_files, 999)
    
    return full_response, valid_files, srcs, docs

def sidebar_configuration():
    with st.sidebar:
        # --- SEZIONE NUOVA CHAT (Corretta) ---
        
        # Funzione di callback per gestire creazione e pulizia
        def on_submit_new_chat():
            name = st.session_state.new_chat_input
            if name:
                # Controllo duplicati solo visivo (il dizionario usa ID univoci)
                if name in st.session_state.chat_metadata.values():
                    st.toast("Esiste già una chat con questo nome!", icon="⚠️")
                else:
                    new_id = create_new_chat(name)
                    st.session_state.active_thread_id = new_id
                    st.toast(f"Chat '{name}' creata!", icon="✅")
                    
                # QUESTO È IL TRUCCO: Svuota il campo di testo usando la session state
                st.session_state.new_chat_input = ""

        # Input collegato alla callback
        st.text_input(
            "Nome nuova chat:",
            placeholder="Es. Lezione Storia...",
            key="new_chat_input", # Chiave fondamentale per la pulizia
            on_change=on_submit_new_chat
        )

        # --- LISTA CHAT ESISTENTI ---
        metadata = st.session_state.chat_metadata
        
        # Ordina chat (più recenti in alto)
        chat_ids = list(metadata.keys())[::-1] 
        chat_names = [metadata[id] for id in chat_ids]
        
        current_id = st.session_state.active_thread_id
        
        # Trova l'indice corretto
        try:
            current_index = chat_ids.index(current_id)
        except (ValueError, IndexError):
            current_index = 0
            if chat_ids:
                st.session_state.active_thread_id = chat_ids[0]

        # Callback per il cambio chat
        def on_chat_change():
            selected_name = st.session_state.chat_selection
            # Trova ID corrispondente al nome
            for cid, cname in metadata.items():
                if cname == selected_name:
                    st.session_state.active_thread_id = cid
                    break

        if chat_names:
            st.selectbox(
                "Le tue chat:",
                options=chat_names,
                index=current_index,
                key="chat_selection",
                on_change=on_chat_change,
                help="Seleziona una chat per caricarne la cronologia"
            )
        else:
            st.info("Nessuna chat esistente.")

        config_thread_id = st.session_state.active_thread_id

        with st.expander("Configurazione"):
            config_path = Path("config.yaml")
            current_provider = "together"
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    current_provider = config_data.get('llm', {}).get('provider', 'together')
            except: pass
            
            provider_map = {"Together AI": "together", "Google Gemini": "gemini"}
            
            llm_provider = st.selectbox("Provider", list(provider_map.keys()), index=list(provider_map.values()).index(current_provider) if current_provider in provider_map.values() else 0)
            
            if provider_map[llm_provider] != current_provider:
                if update_llm_provider(provider_map[llm_provider]):
                    st.toast("Provider aggiornato! Ricarica pagina.")

            config_complexity = st.selectbox("Livello", ('None', 'Base', 'Intermediate', 'Advanced'))
            config_course = st.selectbox("Corso", ("None", "Semantics in Intelligent Information Access", "Metodi per il Ritrovamento dell'Informazione"))
            st.session_state.streaming_enabled = st.checkbox("Streaming", value=st.session_state.streaming_enabled)
            
            config_chat = ConfigChat(complexity_level=config_complexity, course=config_course)
            
        st.divider()
        with st.expander("Info"):
            st.markdown("**UNIVOX** Assistant v2.0")

    return config_thread_id, config_chat

def enhance_user_input(config_chat, user_input, file_path):
    instr = []
    if config_chat.language: instr.append(f"Respond in {config_chat.language}.")
    if config_chat.course != "None": instr.append(f"Context: User is studying {config_chat.course}.")
    
    if file_path:
        rel_path = file_path.replace('\\', '/').split('uploaded_files')[-1] if 'uploaded_files' in file_path else file_path
        instr.append(f"User uploaded file: uploaded_files{rel_path}")
    
    instr.append("Use LaTeX for math.")
    instr.append("IMPORTANT: For analyzing images (png, jpg, jpeg), use 'google_lens_analyze' to describe the visual content.")
    instr.append("IMPORTANT: For summaries use 'summarize_document'. For reading text use 'extract_text'.")
    
    return "\n".join(f"• {i}" for i in instr) + f"\n\nQuery: {user_input}"

def save_chat_history(thread_id, chat_history):
    os.makedirs("history", exist_ok=True)
    with open(f"history/{thread_id}.json", "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

def save_uploaded_file(user_file_input):
    try:
        upload_dir = os.path.join(os.getcwd(), "uploaded_files")
        os.makedirs(upload_dir, exist_ok=True)
        path = os.path.join(upload_dir, user_file_input.name)
        with open(path, 'wb') as f: f.write(user_file_input.getvalue())
        return os.path.normpath(path).replace('\\', '/')
    except: return ""

def get_clean_path(file_path):
    return os.path.abspath(file_path.replace("\\\\", "\\").strip("'\""))

def handle_chatbot_response(user_input, thread_id, config_chat, user_files=None):
    text = user_input.text if hasattr(user_input, 'text') else str(user_input) if user_input else ""
    if not (text.strip() or user_files):
        st.warning("Input vuoto."); return

    chat_hist = get_chat_history(thread_id)
    if not chat_hist or chat_hist[-1]["role"] != "user" or chat_hist[-1]["content"] != text:
        add_message_to_history(thread_id, "user", text)

    paths = []
    if user_files:
        for f in user_files:
            p = save_uploaded_file(f)
            if p: paths.append(get_clean_path(p))

    try:
        prompt_path = paths[0] if paths else None
        enhanced = enhance_user_input(config_chat, text, prompt_path)
        config = {"configurable": {"thread_id": thread_id}}
        
        if st.session_state.streaming_enabled:
            try: loop = asyncio.get_event_loop()
            except: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            
            async def run():
                gen = compiled_graph.astream({"messages": [{"role": "user", "content": enhanced}]}, config=config)
                return await handle_streaming_events(gen)
            
            resp, imgs, srcs, docs = loop.run_until_complete(run())
        else:
            events = list(compiled_graph.stream({"messages": [{"role": "user", "content": enhanced}]}, config, stream_mode="values"))
            resp, imgs, srcs, docs = handle_non_streaming_events(events)

        if resp and not resp.startswith("❌"):
            log_conversation(text, resp, srcs, docs)
            add_message_to_history(thread_id, "bot", resp, imgs)

    except Exception as e:
        st.error(f"Errore: {e}")

def transcribe_audio(audio_file):
    try:
        d = os.path.join(os.getcwd(), "uploaded_files", "audio")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp:
            tmp.write(audio_file.getvalue()); tmp.close()
            AudioSegment.from_file(tmp.name).export(path, format="wav")
            os.remove(tmp.name)
            
        return AudioProcessor().speech_to_text(path), path
    except Exception as e: return f"Error: {e}", None

def play_text_to_speech(text, key):
    if st.button("🔊", key=key):
        try:
            path = AudioProcessor().text_to_speech(text)
            if path and not path.startswith("Error"): st.audio(path, format="audio/mp3")
        except: st.error("TTS Error")

def voice_chat_input():
    float_init()
    with bottom():
        c1, c2 = st.columns([10, 1])
        with c1: sub = st.chat_input("Scrivi qui...", key="main_in", accept_file="multiple", file_type=['txt','pdf','png','jpg','csv','pptx','docx'])
        with c2: 
            act = st.session_state.get('voice_mode', False)
            if st.button("🔴" if act else "🎤", key="vb"): return sub, True
    return sub, False

def main():
    initialize_session()
    try: float_init()
    except: pass

    tid, cfg = sidebar_configuration()
    
    if not tid:
        st.title("Benvenuto in UNIVOX 👋")
        st.info("👈 Crea una nuova chat dalla barra laterale per iniziare!")
    else:
        chat_name = st.session_state.chat_metadata.get(tid, f"Chat {tid}")
        st.title(f"{chat_name}") 
        display_chat_history(tid)

    sub, v_btn = voice_chat_input()

    if v_btn:
        st.session_state.voice_mode = not st.session_state.get('voice_mode', False)
        st.rerun()

    if st.session_state.get('voice_mode', False):
        with st.container():
            st.info("🎤 Registrazione attiva")
            aud = st.audio_input("Registra", key="vr")
            if aud:
                with st.spinner("Trascrizione..."):
                    txt, path = transcribe_audio(aud)
                    if txt and "Error" not in txt:
                        add_message_to_history(tid, "user", txt, audio_path=path)
                        with st.chat_message("user"): st.write(txt)
                        with st.chat_message("assistant"): handle_chatbot_response(txt, tid, cfg, None)
                        st.session_state.voice_mode = False
                        st.rerun()

    if sub:
        txt = sub.text if hasattr(sub, 'text') else str(sub)
        files = sub.files if hasattr(sub, 'files') else []
        with st.chat_message("user"):
            st.write(txt)
            for f in files: st.write(f"📎 {f.name}")
        with st.chat_message("assistant"):
            with st.spinner("..."): handle_chatbot_response(txt, tid, cfg, files)
        st.rerun()

if __name__ == "__main__":
    main()