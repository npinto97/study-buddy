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
import hashlib
import csv

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

# --- CONFIGURAZIONE STUDIO E PERCORSI ---
STUDY_LOG_DIR = Path("data/study_logs")
STUDY_RES_DIR = Path("data/study_results")
STUDY_LOG_DIR.mkdir(parents=True, exist_ok=True)
STUDY_RES_DIR.mkdir(parents=True, exist_ok=True)

# --- SCENARI DI STUDIO ---
# --- SCENARI DI STUDIO (DOMANDE VERIFICATE DA EVALUATION) ---
SCENARIOS = {
    "1": {
        "title": "The Exam Prep",
        "role": "Sei uno studente che ripassa le slide introduttive.",
        "context": "Devi capire le motivazioni alla base dei Recommender Systems.",
        "goal": "Usa Study Buddy per:\n1. Chiedere (in Italiano o Inglese): 'Secondo le slide introduttive del corso MRI, quale problema risolvono i sistemi di Information Filtering e Recommender Systems?' (Originale: 'According to the introductory slides of the MRI course, what problem do Information Filtering and Recommender Systems solve?')\n2. Verificare se la risposta corrisponde a: 'Information Overload'"
    },
    "2": {
        "title": "The Practical Student",
        "role": "Devi contattare il Prof. Lops.",
        "context": "Non sai quando è disponibile per il ricevimento.",
        "goal": "Usa Study Buddy per:\n1. Chiedere (in Italiano o Inglese): 'Quali sono gli orari di ricevimento del Professor Lops?' (Originale: 'What are the office hours for Professor Lops?')\n2. Verificare se la risposta corrisponde a: 'Martedì 10:00-12:00'"
    },
    "3": {
        "title": "The Curriculum Analyst",
        "role": "Stai decidendo quale corso frequentare.",
        "context": "Vuoi confrontare lo stile di insegnamento di due corsi.",
        "goal": "Usa Study Buddy per:\n1. Chiedere (in Italiano o Inglese): 'Confronta i metodi di insegnamento usati nei corsi MRI e SIIA.' (Originale: 'Compare the teaching methods used in the MRI and SIIA courses.')\n2. Verificare se la risposta spiega che: 'MRI ha esercitazioni guidate vs SIIA ha sessioni di laboratorio.'"
    }
}

def get_user_data_dir():
    """Returns the effective data directory based on mode."""
    # MODALITÀ STUDIO: Cartella isolata per utente
    if st.session_state.get("app_mode") == "study":
        if "study_id" not in st.session_state:
            st.session_state.study_id = str(uuid.uuid4())
        user_dir = Path("history") / st.session_state.study_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    
    # MODALITÀ DEV: Cartella history/ classica condivisa
    else:
        return Path("history")

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

def log_study_interaction(session_id: str, role: str, content: str, metadata: dict = None):
    """Logs interactions for the user study in JSONL format."""
    # Logga solo se siamo in study mode
    if st.session_state.get("app_mode") != "study": return

    entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "role": role,
        "content": content,
        "metadata": metadata or {}
    }
    file_path = STUDY_LOG_DIR / f"session_{session_id}.jsonl"
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def log_conversation(question: str, answer: str, sources: list = None, retrieved_docs: list = None):
    # Legacy debug logging
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

    # Console Logging (Loguru)
    console_msg = f"\n❓ QUESTION:\n{question}\n"
    if sources:
        console_msg += "\n📚 SOURCES:\n" + "\n".join([f"  {i}. {s}" for i, s in enumerate(sources, 1)]) + "\n"
    if retrieved_docs:
        console_msg += "\n📄 RETRIEVED DOCS:\n" + "\n".join([f"  {i}. {d.replace(chr(10), ' ').strip()[:150]}..." for i, d in enumerate(retrieved_docs, 1)]) + "\n"
    console_msg += f"\n💬 ANSWER:\n{answer}\n"
    logger.info(console_msg)

st.set_page_config(page_title="Study Buddy", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

try:
    st.logo(image=str(Path("images/univox_logo2.png")), size="large", icon_image=str(Path("images/univox_logo2.png")))
except:
    pass

class ConfigChat:
    def __init__(self, complexity_level="None", language="Italian", course="None"):
        self.complexity_level = complexity_level
        self.language = language
        self.course = course

# --- GESTIONE METADATI CHAT ---
def get_metadata_file():
    return get_user_data_dir() / "metadata.json"

def load_chat_metadata():
    """Carica il mapping ID -> Nome specifico per la sessione corrente."""
    meta_file = get_metadata_file()
    
    # Se in DEV mode, crea se non esiste
    if not meta_file.exists():
        new_meta = {}
        # In Dev Mode, recupera vecchi file json nella cartella
        if st.session_state.get("app_mode") == "dev":
            files = [f for f in os.listdir("history") if f.endswith(".json") and f != "metadata.json" and os.path.isfile(os.path.join("history", f))]
            for f in files:
                tid = f.replace(".json", "")
                new_meta[tid] = f"Chat {tid}"
        
        save_chat_metadata(new_meta)
        return new_meta

    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)

def save_chat_metadata(metadata):
    meta_file = get_metadata_file()
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def create_new_chat(name):
    new_id = str(uuid.uuid4())
    metadata = st.session_state.chat_metadata
    metadata[new_id] = name
    st.session_state.chat_metadata = metadata
    save_chat_metadata(metadata)
    return new_id

# -------------------------------------------

def show_mode_selection():
    st.title("Benvenuto in Study Buddy 👋")
    st.subheader("Seleziona la modalità di avvio:")
    
    c1, c2 = st.columns(2)
    
    with c1:
        with st.container(border=True):
            st.markdown("### 🔬 Partecipante allo Studio")
            st.write("Scegli questa opzione se sei uno studente che partecipa al test di valutazione.")
            st.write("- Ambiente isolato")
            st.write("- Dati anonimizzati")
            if st.button("Avvia Studio", type="primary", use_container_width=True):
                st.session_state.app_mode = "study"
                st.rerun()

    with c2:
        with st.container(border=True):
            st.markdown("### 🛠️ Modalità Sviluppatore")
            st.write("Accesso completo per debug e manutenzione del sistema.")
            st.write("- Accesso storico completo")
            st.write("- Nessun log di studio")
            
            pwd = st.text_input("Password Amministratore", type="password", key="dev_pwd")
            
            if st.button("Avvia Dev Mode", use_container_width=True):
                if pwd == "admin123":
                    st.session_state.app_mode = "dev"
                    st.rerun()
                else:
                    st.error("❌ Password errata!")

def initialize_session():
    # Se non è stata scelta la modalità, non inizializzare altro
    if "app_mode" not in st.session_state:
        return

    if "study_id" not in st.session_state:
        st.session_state.study_id = str(uuid.uuid4())
        
    if "consent_given" not in st.session_state:
        st.session_state.consent_given = False
    if "study_completed" not in st.session_state:
        st.session_state.study_completed = False
    
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    
    if "chat_metadata" not in st.session_state:
        st.session_state.chat_metadata = load_chat_metadata()
        
    if "voice_mode" not in st.session_state:
        st.session_state.voice_mode = False
    if "streaming_enabled" not in st.session_state:
        st.session_state.streaming_enabled = True
    
    # ACTIVE THREAD LOGIC
    if "active_thread_id" not in st.session_state:
        if st.session_state.chat_metadata:
             st.session_state.active_thread_id = list(st.session_state.chat_metadata.keys())[-1]
        else:
             st.session_state.active_thread_id = None
             
    # Auto-create chat if missing
    if not st.session_state.active_thread_id:
        if st.session_state.app_mode == "study":
            new_id = create_new_chat("Sessione Studio")
        else:
            new_id = create_new_chat("Nuova Chat")
        st.session_state.active_thread_id = new_id

    os.makedirs(os.path.join(os.getcwd(), "uploaded_files", "audio"), exist_ok=True)

def show_consent_screen():
    st.subheader("📋 Consenso Informato")
    st.markdown("""
    **Benvenuto allo studio di valutazione di "Study Buddy".**
    
    Prima di iniziare, ti chiediamo di leggere e accettare quanto segue:
    
    1.  **Obiettivo**: Valutare l'efficacia di un assistente AI per lo studio.
    2.  **Privacy**: Le tue conversazioni saranno registrate in forma **anonima** solo a fini di ricerca. Nessun dato personale identificativo verrà associato alle chat. I tuoi dati sono isolati e non visibili ad altri utenti.
    3.  **Durata**: Circa 10-15 minuti.
    4.  **Volontarietà**: Puoi abbandonare in qualsiasi momento.
    5.  **Premio**: Al termine, riceverai un codice per confermare la tua partecipazione.
    
    Cliccando su "Accetto e Inizio", confermi di aver letto e compreso le condizioni, di avere almeno 18 anni e acconsenti all'uso dei dati anonimi.
    """)
    if st.button("✅ Accetto e Inizio", type="primary"):
        st.session_state.consent_given = True
        st.rerun()

def show_scenario_selection():
    st.subheader("🎯 Seleziona la tua Missione")
    st.write("Per testare il sistema in un contesto realistico, scegli uno dei seguenti scenari:")
    
    cols = st.columns(3)
    
    for i, (sid, scen) in enumerate(SCENARIOS.items()):
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"### {scen['title']}")
                st.info(f"**Ruolo**: {scen['role']}")
                st.caption(scen['context'])
                if st.button(f"Scegli {scen['title']}", key=f"btn_scen_{sid}", use_container_width=True):
                    st.session_state.selected_scenario = sid
                    st.rerun()

def save_questionnaire_results(data):
    file_path = STUDY_RES_DIR / "results.csv"
    file_exists = file_path.exists()
    
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'session_id', 'q1', 'q2', 'q3', 'q4', 'q5', 'comments'])
        
        writer.writerow([
            datetime.now().isoformat(),
            st.session_state.study_id,
            data['q1'], data['q2'], data['q3'], data['q4'], data['q5'],
            data['comments']
        ])

def register_completion_code(session_id, code):
    """Saves the generated code to a secure registry CSV."""
    file_path = STUDY_RES_DIR / "codes.csv"
    file_exists = file_path.exists()
    
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'session_id', 'completion_code'])
        writer.writerow([datetime.now().isoformat(), session_id, code])

def generate_completion_code(session_id):
    h = hashlib.sha256(session_id.encode()).hexdigest()
    code = f"SB-{h[:4].upper()}-{h[-4:].upper()}"
    register_completion_code(session_id, code)
    return code

def show_questionnaire_screen():
    st.subheader("🎓 Questionario Finale")
    st.write("Grazie per aver provato Study Buddy! Rispondi a queste brevi domande per ricevere il tuo codice.")
    
    with st.form("sus_form"):
        q1 = st.slider("1. Penso che userei frequentemente questo sistema.", 1, 5, 3)
        q2 = st.slider("2. Ho trovato il sistema inutilmente complesso.", 1, 5, 3)
        q3 = st.slider("3. Ho trovato il sistema facile da usare.", 1, 5, 3)
        q4 = st.slider("4. Penso che la maggior parte delle persone imparerebbe a usare questo sistema molto rapidamente.", 1, 5, 3)
        q5 = st.slider("5. Mi sono sentito molto sicuro usando il sistema.", 1, 5, 3)
        comments = st.text_area("Eventuali commenti o suggerimenti:")
        
        submitted = st.form_submit_button("Invia e Ottieni Codice")
        
        if submitted:
            save_questionnaire_results({
                'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4, 'q5': q5, 'comments': comments
            })
            st.session_state.study_completed = True
            st.rerun()

def show_completion_screen():
    st.balloons()
    if "final_code" not in st.session_state:
        st.session_state.final_code = generate_completion_code(st.session_state.study_id)
        
    code = st.session_state.final_code
    st.success("🎉 Studio Completato con Successo!")
    st.markdown(f"""
    ### Il tuo codice di completamento è:
    # `{code}`
    
    **Copia questo codice e invialo al responsabile della ricerca per ricevere il tuo premio.**
    
    Il codice è stato registrato nel sistema. Puoi chiudere questa pagina.
    """)
    st.info(f"Session ID: {st.session_state.study_id}")
    if st.button("Riavvia (Nuovo Utente)"):
        st.session_state.clear()
        st.rerun()

def get_chat_history(thread_id):
    if not thread_id: return []
    if thread_id not in st.session_state.chat_histories:
        history_path = get_user_data_dir() / f"{thread_id}.json"
        
        if history_path.exists():
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
    print("DEBUG: Entered handle_streaming_events")
    full_response = ""
    tool_msgs = []
    srcs, docs = [], []
    placeholder = st.empty()
    
    try:
        async for event in events_generator:
            print(f"DEBUG: Streaming event received: {type(event)} - Keys: {event.keys() if isinstance(event, dict) else 'N/A'}")
            for node, output in event.items():
                print(f"DEBUG: Processing node: {node}")
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
    except Exception as e:
        print(f"DEBUG: Error in streaming loop: {e}")
        st.error(f"Errore nello streaming: {e}")
    
    print(f"DEBUG: Streaming finished. Full response length: {len(full_response)}")
    print(f"DEBUG: RESPONSE CONTENT: {repr(full_response)}")
    if not full_response:
        print("DEBUG: WARNING - No response text generated!")
        st.warning("Nessuna risposta generata dal modello.")

    placeholder.markdown(full_response)
    valid_files = process_tool_messages_for_images(tool_msgs)
    if valid_files:
        st.markdown("---")
        st.markdown("📁 **File generati:**")
        display_images_and_files("", valid_files, 999)
    
    return full_response, valid_files, srcs, docs

def handle_non_streaming_events(events):
    print("DEBUG: Entered handle_non_streaming_events")
    full_response = ""
    tool_msgs = []
    srcs, docs = [], []
    
    for event in events:
        print(f"DEBUG: Processing event: {type(event)}")
        for node, output in event.items():
            print(f"DEBUG: Node: {node}")
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
                        print(f"DEBUG: Accumulated response length: {len(full_response)}")
    
    if full_response: 
        print(f"DEBUG: Final response: {full_response[:50]}...")
        st.markdown(full_response, unsafe_allow_html=True)
    else:
        print("DEBUG: No full_response generated!")
    
    valid_files = process_tool_messages_for_images(tool_msgs)
    if valid_files:
        st.markdown("---")
        st.markdown("📁 **File generati:**")
        display_images_and_files("", valid_files, 999)
    
    return full_response, valid_files, srcs, docs

def sidebar_configuration():
    with st.sidebar:
        # BOTTONE RICARICA (utile in Dev)
        if st.button("🔄 Riavvia App"):
            st.rerun()
            
        if st.button("🏠 Cambia Modalità"):
            st.session_state.clear()
            st.rerun()
            
        # MODALITÀ STUDIO: Niente lista chat globale
        if st.session_state.app_mode == "study":
            st.header("🔬 Studio Utente")
            
            # MOSTRA OBIETTIVO SCENARIO
            if "selected_scenario" in st.session_state:
                scen = SCENARIOS.get(st.session_state.selected_scenario)
                if scen:
                    with st.expander("🎯 La tua Missione", expanded=True):
                        st.markdown(f"**{scen['title']}**")
                        st.success(scen['goal'])
            
            if st.button("📝 Concludi e Valuta", type="primary", use_container_width=True):
                st.session_state.show_questionnaire = True
                st.rerun()
            st.divider()
        
        # MODALITÀ DEV: Lista chat normale + Nuova Chat
        else:
            st.header("🛠️ Dev Mode")
            # --- SEZIONE NUOVA CHAT ---
            def on_submit_new_chat():
                name = st.session_state.new_chat_input
                if name:
                    if name in st.session_state.chat_metadata.values():
                        st.toast("Esiste già una chat con questo nome!", icon="⚠️")
                    else:
                        new_id = create_new_chat(name)
                        st.session_state.active_thread_id = new_id
                        st.toast(f"Chat '{name}' creata!", icon="✅")
                    st.session_state.new_chat_input = ""

            st.text_input(
                "Nome nuova chat:",
                placeholder="Es. Lezione Storia...",
                key="new_chat_input", 
                on_change=on_submit_new_chat
            )

            # --- LISTA CHAT ---
            metadata = st.session_state.chat_metadata
            chat_ids = list(metadata.keys())[::-1] 
            chat_names = [metadata[id] for id in chat_ids]
            current_id = st.session_state.active_thread_id
            
            # Safe index finding
            current_index = 0
            if current_id in chat_ids:
                current_index = chat_ids.index(current_id)
            elif chat_ids:
                 st.session_state.active_thread_id = chat_ids[0]

            def on_chat_change():
                selected_name = st.session_state.chat_selection
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
                    on_change=on_chat_change
                )
            else:
                st.info("Nessuna chat esistente.")
        
        # --- CONFIG COMUNE ---
        config_thread_id = st.session_state.get("active_thread_id")

        with st.expander("Configurazione Avanzata"):
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
            st.markdown(f"**UNIVOX** Assistant v2.1")
            if st.session_state.app_mode == "study":
                st.code(f"Study ID: {st.session_state.get('study_id', '')[:8]}")

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
    # SALVATAGGIO ISOLATO o CLASSICO (Gestito da get_user_data_dir)
    hist_file = get_user_data_dir() / f"{thread_id}.json"
    with open(hist_file, "w", encoding="utf-8") as f:
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

    # Ensure thread_id exists in session, otherwise likely cleared
    if not thread_id:
        st.error("Errore sessione: Ricarica la pagina.")
        return

    chat_hist = get_chat_history(thread_id)
    if not chat_hist or chat_hist[-1]["role"] != "user" or chat_hist[-1]["content"] != text:
        add_message_to_history(thread_id, "user", text)
        log_study_interaction(st.session_state.get("study_id", "dev"), "user", text)

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
            async def run():
                gen = compiled_graph.astream({"messages": [{"role": "user", "content": enhanced}]}, config=config)
                return await handle_streaming_events(gen)
            
            resp, imgs, srcs, docs = asyncio.run(run())
        else:
            events = list(compiled_graph.stream({"messages": [{"role": "user", "content": enhanced}]}, config, stream_mode="values"))
            resp, imgs, srcs, docs = handle_non_streaming_events(events)

        if resp and not resp.startswith("❌"):
            log_conversation(text, resp, srcs, docs)
            log_study_interaction(st.session_state.get("study_id", "dev"), "bot", resp, {"sources": srcs, "docs_count": len(docs)})
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
    if "app_mode" not in st.session_state:
        show_mode_selection()
        return

    initialize_session()
    
    # --- FLUSSO STUDIO ---
    if st.session_state.app_mode == "study":
        if not st.session_state.consent_given:
            show_consent_screen()
            return
        
        # SCENARIO SELECTION
        if "selected_scenario" not in st.session_state:
            show_scenario_selection()
            return
        
        if st.session_state.study_completed:
            show_completion_screen()
            return

        if st.session_state.get("show_questionnaire", False):
            if st.button("🔙 Torna alla Chat"):
                st.session_state.show_questionnaire = False
                st.rerun()
            show_questionnaire_screen()
            return
    # ---------------------

    try: float_init()
    except: pass

    tid, cfg = sidebar_configuration()
    
    # Se per qualche motivo tid è None (es. dev mode prima run), mostra messaggio
    if not tid and st.session_state.app_mode == "dev":
        st.title("Benvenuto in UNIVOX (Dev Mode) �️")
        st.info("👈 Seleziona o crea una chat per iniziare.")
    elif tid:
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
                        log_study_interaction(st.session_state.get("study_id","dev"), "user", txt, {"mode": "voice"})
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