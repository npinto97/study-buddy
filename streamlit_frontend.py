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
import random

from streamlit_float import *
from streamlit_extras.bottom_container import bottom

from study_buddy.agent import compiled_graph
from study_buddy.utils.tools import AudioProcessor
import yaml
from loguru import logger

# Fix per classi torch su Windows/alcuni ambienti
try:
    torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
except:
    pass

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
        "title": "RecSys Fundamentals (MRI)",
        "subject": "MRI",
        "role": "Sei uno studente del corso MRI che sta studiando i Recommender Systems.",
        "context": "Hai bisogno di chiarire il problema fondamentale che questi sistemi risolvono.",
        "goal": "Usa Study Buddy per:\n1. Chiedere: 'Secondo le slide introduttive del corso MRI, quale problema principale risolvono i sistemi di Information Filtering e Recommender Systems?'\n2. Verificare se la risposta menziona l'**Information Overload**."
    },
    "2": {
        "title": "Content-Based Filtering (MRI)",
        "subject": "MRI",
        "role": "Stai approfondendo i metodi di raccomandazione Content-Based.",
        "context": "Vuoi capire come vengono rappresentati gli item in questo approccio.",
        "goal": "Usa Study Buddy per:\n1. Chiedere: 'In un sistema Content-Based, come viene tipicamente rappresentato il profilo di un item?'\n2. Verificare se la risposta fa riferimento a **feature** o **attributi** (es. keyword, metadati)."
    },
    "3": {
        "title": "Evaluation Metrics (MRI)",
        "subject": "MRI",
        "role": "Devi valutare le performance di un sistema di raccomandazione.",
        "context": "Ti serve sapere quali metriche usare per misurare l'accuratezza.",
        "goal": "Usa Study Buddy per:\n1. Chiedere: 'Quali sono le principali metriche di errore utilizzate per valutare l'accuratezza di predizione nei Recommender Systems?'\n2. Verificare se la risposta include **MAE** (Mean Absolute Error) o **RMSE** (Root Mean Squared Error)."
    },
    "4": {
        "title": "Semantic Web Vision (SIIA)",
        "subject": "SIIA",
        "role": "Sei uno studente del corso SIIA che inizia a studiare il Semantic Web.",
        "context": "Vuoi comprendere la visione originale di Tim Berners-Lee.",
        "goal": "Usa Study Buddy per:\n1. Chiedere: 'Qual è l'idea centrale del Semantic Web secondo la visione di Tim Berners-Lee?'\n2. Verificare se la risposta parla di estendere il Web per rendere i dati **leggibili e processabili dalle macchine**."
    },
    "5": {
        "title": "RDF Triples (SIIA)",
        "subject": "SIIA",
        "role": "Stai studiando il modello dei dati RDF.",
        "context": "Hai bisogno di un esempio concreto di come sono strutturate le informazioni.",
        "goal": "Usa Study Buddy per:\n1. Chiedere: 'Spiegami la struttura di una tripla RDF e fammi un esempio.'\n2. Verificare se la risposta descrive la struttura **Soggetto-Predicato-Oggetto**."
    },
    "6": {
        "title": "Python Typing (LP)",
        "subject": "LP",
        "role": "Sei uno studente di Linguaggi di Programmazione interessato a Python.",
        "context": "Vuoi capire come Python gestisce i tipi rispetto a linguaggi come Java.",
        "goal": "Usa Study Buddy per:\n1. Chiedere: 'Che tipo di tipizzazione utilizza Python e in cosa differisce dalla tipizzazione statica?'\n2. Verificare se la risposta menziona la **tipizzazione dinamica** o **duck typing**."
    },
    "7": {
        "title": "Java Interfaces vs Abstract Classes (LP)",
        "subject": "LP",
        "role": "Stai ripassando i concetti di OOP in Java.",
        "context": "Ti serve chiarire quando usare un'interfaccia o una classe astratta.",
        "goal": "Usa Study Buddy per:\n1. Chiedere: 'Qual è la differenza principale tra una classe astratta e un'interfaccia in Java (pre-Java 8)?'\n2. Verificare se la risposta spiega che una classe può implementare **molteplici interfacce** ma estendere **una sola classe**."
    },
    "8": {
        "title": "Haskell Functions (LP)",
        "subject": "LP",
        "role": "Sei curioso riguardo alla programmazione funzionale in Haskell.",
        "context": "Hai sentito parlare di funzioni di ordine superiore.",
        "goal": "Usa Study Buddy per:\n1. Chiedere: 'Cos'è una funzione di ordine superiore (Higher-Order Function) in Haskell?'\n2. Verificare se la risposta dice che è una funzione che prende **altre funzioni come argomenti** o restituisce una funzione."
    },
    "9": {
        "title": "Collaborative Filtering (MRI)",
        "subject": "MRI",
        "role": "Stai studiando i metodi di filtraggio collaborativo.",
        "context": "Vuoi capire la differenza tra User-Based e Item-Based.",
        "goal": "Usa Study Buddy per:\n1. Chiedere: 'Qual è la differenza principale tra User-Based e Item-Based Collaborative Filtering?'\n2. Verificare se la risposta menziona la **similarità tra utenti** vs **similarità tra item**."
    },
    "10": {
        "title": "Matrix Factorization (MRI)",
        "subject": "MRI",
        "role": "Sei interessato alle tecniche avanzate di Recommender Systems.",
        "context": "Hai letto di SVD e fattorizzazione di matrici.",
        "goal": "Usa Study Buddy per:\n1. Chiedere: 'In che modo la Matrix Factorization aiuta a risolvere il problema della scarsità dei dati (sparsity) nei Recommender Systems?'\n2. Verificare se la risposta spiega che riduce la dimensionalità catturando **feature latenti**."
    },
    "11": {
        "title": "SPARQL Queries (SIIA)",
        "subject": "SIIA",
        "role": "Devi estrarre dati da un knowledge graph.",
        "context": "Stai imparando il linguaggio di query per RDF.",
        "goal": "Usa Study Buddy per:\n1. Chiedere: 'A cosa serve la keyword OPTIONAL in una query SPARQL?'\n2. Verificare se la risposta dice che permette di recuperare dati anche se un **pattern (tripla) non corrisponde** (gestione dati mancanti)."
    },
    "12": {
        "title": "OWL Ontologies (SIIA)",
        "subject": "SIIA",
        "role": "Stai progettando un'ontologia per il Web Semantico.",
        "context": "Ti serve capire la differenza tra le varie versioni di OWL.",
        "goal": "Usa Study Buddy per:\n1. Chiedere: 'Qual è la differenza principale tra OWL Lite, OWL DL e OWL Full?'\n2. Verificare se la risposta menziona il trade-off tra **espressività** e **decidibilità/efficienza computazionale**."
    },
    "13": {
        "title": "Linked Data Principles (SIIA)",
        "subject": "SIIA",
        "role": "Vuoi pubblicare dati aperti sul Web.",
        "context": "Devi seguire i principi dei Linked Data.",
        "goal": "Usa Study Buddy per:\n1. Chiedere: 'Quali sono i 4 principi dei Linked Data definiti da Tim Berners-Lee?'\n2. Verificare se la risposta li elenca, incluso l'uso di **URI** per identificare cose e **link RDF** per collegarle."
    },
    "14": {
        "title": "Prolog Unification (LP)",
        "subject": "LP",
        "role": "Stai studiando la programmazione logica.",
        "context": "Il concetto di unificazione non ti è chiaro.",
        "goal": "Usa Study Buddy per:\n1. Chiedere: 'Spiega il processo di unificazione in Prolog.'\n2. Verificare se la risposta lo descrive come il meccanismo per rendere **identici due termini** trovando le sostituzioni per le variabili."
    },
    "15": {
        "title": "Polymorphism Types (LP)",
        "subject": "LP",
        "role": "Approfondisci i sistemi di tipi nei linguaggi moderni.",
        "context": "Vuoi distinguere tra polimorfismo parametrico e ad-hoc.",
        "goal": "Usa Study Buddy per:\n1. Chiedere: 'Qual è la differenza tra polimorfismo parametrico (es. Generics in Java) e polimorfismo ad-hoc (es. Overloading)?'\n2. Verificare se la risposta spiega che il parametrico usa lo **stesso codice per tipi diversi**, mentre l'ad-hoc usa **codice diverso** per ogni tipo."
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

def render_provider_selector():
    """Renders the UI for selecting the LLM provider."""
    # Read current config
    config_path = Path("config.yaml")
    current_provider = "together"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            current_provider = config_data.get('llm', {}).get('provider', 'together')
    except: pass
    
    provider_map = {"Together AI": "together", "Google Gemini": "gemini"}
    
    # Determine index
    try:
        curr_idx = list(provider_map.values()).index(current_provider)
    except ValueError:
        curr_idx = 0

    llm_provider = st.selectbox(
        "Seleziona Modello AI", 
        list(provider_map.keys()), 
        index=curr_idx,
        key="llm_provider_shared_selector"
    )
    
    if provider_map[llm_provider] != current_provider:
        if update_llm_provider(provider_map[llm_provider]):
            st.toast(f"Provider impostato su {llm_provider}. Ricarica...", icon="✅")
            # Optional: st.rerun()

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
                # Preserva la struttura ma limita la lunghezza del contenuto
                # doc è già formattato con newlines da tools.py
                f.write(f"  {i}. {doc.strip()[:600]}...\n")
        
        f.write(f"\n💬 RISPOSTA:\n{answer}\n{'='*80}\n\n")

    # Console Logging (Loguru)
    console_msg = f"\n❓ QUESTION:\n{question}\n"
    if sources:
        console_msg += "\n📚 SOURCES:\n" + "\n".join([f"  {i}. {s}" for i, s in enumerate(sources, 1)]) + "\n"
    if retrieved_docs:
        # Anche qui aumentiamo il limite per vedere il path
        console_msg += "\n📄 RETRIEVED DOCS:\n" + "\n".join([f"  {i}. {d.strip()[:600]}..." for i, d in enumerate(retrieved_docs, 1)]) + "\n"
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
    st.subheader("Modulo di Consenso Informato")
    st.markdown("""
    **Benvenuto allo studio di valutazione di "Study Buddy".**
    
    Prima di procedere, è necessario prendere visione e accettare le seguenti condizioni:
    
    1.  **Obiettivo dello Studio**: Valutare l'efficacia e l'usabilità di un assistente AI di supporto allo studio.
    2.  **Privacy e Trattamento Dati**: Le interazioni con il sistema saranno registrate in forma **strettamente anonima** ed esclusivamente per fini di ricerca accademica. Nessun dato personale identificativo verrà associato alle sessioni di chat. I dati sono isolati e non accessibili ad altri partecipanti.
    3.  **Durata Stimata**: La sessione ha una durata di circa 5-10 minuti.
    4.  **Partecipazione Volontaria**: La partecipazione è interamente facoltativa e non prevede alcuna valutazione o beneficio ai fini dell'esame. È possibile interrompere lo studio in qualsiasi momento.
    """)
    
    st.markdown("### Dichiarazione di Consenso")

    # Download Informativa
    privacy_pdf_path = Path("doc_privacy/Informativa trattamento dati personali StudyBuddy.pdf")
    if privacy_pdf_path.exists():
        with open(privacy_pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Scarica Informativa completa (PDF)",
                data=pdf_file,
                file_name="Informativa_Privacy_StudyBuddy.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.warning("File informativa non trovato.")
    
    st.write("") # Spacer

    # --- INTRO TEXT ---
    st.markdown("""
    ai sensi delle disposizioni del Regolamento (UE) 2016/679 e del Decreto Legislativo 196/2003 e 
    successive modifiche e integrazioni e avendo letto le “Informazioni sul trattamento dei dati personali”.
    """)
    
    # --- CONSENT 1 (MANDATORY) ---
    # User specified: "acconsento al primo blocco obbligatorio"
    q1 = st.radio(
        "q1_label_hidden",
        options=["acconsento", "non acconsento"],
        index=None,
        horizontal=True,
        label_visibility="collapsed",
        key="consent_q1"
    )

    st.write("") # Spacer

    # --- TEXT 1 (NECESSARY) ---
    # User specified: "si presenta dopo acconsento"
    st.markdown("""
    al trattamento - **necessario** ai fini della partecipazione alla ricerca in questione - dei miei dati personali 
    per scopi di ricerca scientifica e statistica nel modo e per i motivi descritti nella sezione intitolata 
    “Finalità e modalità del trattamento”.
    """)

    # --- CONSENT 2 (OPTIONAL) ---
    # User specified: "acconsento al secondo blocco che non e obbligatorio"
    q2 = st.radio(
        "q2_label_hidden",
        options=["acconsento", "non acconsento"],
        index=None,
        horizontal=True,
        label_visibility="collapsed",
        key="consent_q2"
    )

    st.write("") # Spacer

    # --- TEXT 2 (NON NECESSARY) ---
    # User specified: "si presenta dopo acconsento"
    st.markdown("""
    alla conservazione e ulteriore utilizzo – **non necessario** ai fini della partecipazione allo studio in questione - 
    dei miei dati personali per successive attività di ricerca e per essere eventualmente ricontattato per studi ulteriori.
    """)
    
    # Blocking logic: Only Q1 (First Block) is mandatory as per user instruction
    # "primo blocco obbligatorio", "secondo blocco non obbligatorio"
    consent_given = (q1 == "acconsento")
    
    # Log Q2 (Optional re-contact)
    if q2 == "acconsento":
        st.session_state.consent_recontact = True
    else:
        st.session_state.consent_recontact = False

    if st.button("Accetto e Inizio Sessione", type="primary", disabled=not consent_given, use_container_width=True):
        st.session_state.consent_given = True
        st.rerun()

def show_scenario_selection():
    # 1. Subject Selection Phase
    if "selected_study_subject" not in st.session_state:
        st.subheader("Selezione Materia")
        st.write("Per iniziare, seleziona la materia su cui vuoi verta la sessione di studio:")
        
        cols = st.columns(3)
        subjects = ["MRI", "SIIA", "LP"]
        
        for i, subj in enumerate(subjects):
            with cols[i]:
                if st.button(subj, key=f"btn_subj_{subj}", use_container_width=True, type="primary"):
                    st.session_state.selected_study_subject = subj
                    st.rerun()
        return

    # 2. Random Scenario Assignment Phase
    st.subheader("Scenario di Studio Assegnato")
    st.write("Il sistema ha assegnato il seguente scenario per la sessione corrente:")
    
    if "proposed_scenario_id" not in st.session_state:
        subj = st.session_state.selected_study_subject
        # Filter scenarios by subject
        candidates = [k for k, v in SCENARIOS.items() if v.get("subject") == subj]
        
        if not candidates:
             st.error(f"Nessuno scenario trovato per {subj}")
             return
             
        st.session_state.proposed_scenario_id = random.choice(candidates)
    
    sid = st.session_state.proposed_scenario_id
    scen = SCENARIOS[sid]
    
    with st.container(border=True):
        st.markdown(f"### {scen['title']}")
        st.caption(f"Materia: {scen.get('subject', 'N/A')}")
        st.info(f"**Ruolo**: {scen['role']}")
        st.markdown(f"**Contesto**: {scen['context']}")
        st.success(f"**Obiettivo**:\n{scen['goal']}")
        
        st.write("") # Spacer
        if st.button("Avvia Sessione", type="primary", use_container_width=True):
            st.session_state.selected_scenario = sid
            st.rerun()

def save_questionnaire_results(data):
    file_path = STUDY_RES_DIR / "results.csv"
    file_exists = file_path.exists()
    
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'timestamp', 'session_id', 
                'age', 'gender', 'enrollment',
                'sus_q1', 'sus_q2', 'sus_q3', 'sus_q4', 'sus_q5', 'sus_q6', 'sus_q7', 'sus_q8', 'sus_q9', 'sus_q10',
                'qual_completeness', 'qual_clarity', 'qual_utility', 'qual_trust', 'qual_sources',
                'nps_score',
                'comments'
            ])
        
        writer.writerow([
            datetime.now().isoformat(),
            st.session_state.study_id,
            data.get('age', ''), data.get('gender', ''), data.get('enrollment', ''),
            data['sus_q1'], data['sus_q2'], data['sus_q3'], data['sus_q4'], data['sus_q5'], 
            data['sus_q6'], data['sus_q7'], data['sus_q8'], data['sus_q9'], data['sus_q10'],
            data['qual_completeness'], data['qual_clarity'], data['qual_utility'], 
            data['qual_trust'], data['qual_sources'],
            data['nps_score'],
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
    st.write("Grazie per aver provato Study Buddy! La tua opinione è fondamentale per la nostra ricerca.")
    
    with st.form("study_evaluation_form"):
        st.markdown("### 0. Profilazione (Opzionale ma gradita)")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            age = st.number_input("Età", min_value=18, max_value=100, step=1, value=None)
        with col_p2:
            gender = st.selectbox("Genere", ["Seleziona...", "Uomo", "Donna", "Non binario", "Preferisco non specificare", "Altro"])
        with col_p3:
            enrollment = st.selectbox("Iscrizione", ["Seleziona...", "Triennale", "Magistrale", "Dottorato", "Altro"])

        st.divider()
        st.markdown("### 1. Esperienza Utente (SUS)")
        st.caption("Valuta da 1 (Per nulla d'accordo) a 5 (Completamente d'accordo)")
        
        col1, col2 = st.columns(2)
        with col1:
            sus_q1 = st.slider("1. Penso che userei frequentemente questo sistema.", 1, 5, 3)
            sus_q2 = st.slider("2. Ho trovato il sistema inutilmente complesso.", 1, 5, 3)
            sus_q3 = st.slider("3. Ho trovato il sistema facile da usare.", 1, 5, 3)
            sus_q4 = st.slider("4. Penso che avrei bisogno del supporto di una persona tecnica per usare questa app.", 1, 5, 3)
            sus_q5 = st.slider("5. Ho trovato le varie funzioni dell'app ben integrate tra loro.", 1, 5, 3)
        with col2:
            sus_q6 = st.slider("6. Ho trovato troppa incoerenza in questa app.", 1, 5, 3)
            sus_q7 = st.slider("7. Immagino che la maggior parte delle persone imparerebbe a usare questa app molto rapidamente.", 1, 5, 3)
            sus_q8 = st.slider("8. Ho trovato l'app molto macchinosa da usare.", 1, 5, 3)
            sus_q9 = st.slider("9. Mi sono sentito molto sicuro nell'usare l'app.", 1, 5, 3)
            sus_q10 = st.slider("10. Ho dovuto imparare molte cose prima di poter iniziare a usare l'app.", 1, 5, 3)

        st.divider()
        st.markdown("### 2. Qualità delle Risposte")
        st.caption("Valuta la qualità delle risposte fornite dall'IA (1=Scarso, 5=Eccellente)")
        
        qc1, qc2 = st.columns(2)
        with qc1:
            qual_completeness = st.slider("Completezza: La risposta ha coperto tutti i punti richiesti?", 1, 5, 3)
            qual_clarity = st.slider("Fluidità: Il linguaggio era chiaro e comprensibile?", 1, 5, 3)
            qual_utility = st.slider("Utilità: Il tono era appropriato per l'apprendimento?", 1, 5, 3)
        with qc2:
            qual_trust = st.slider("Fiducia: Ti fideresti per un esame reale?", 1, 5, 3)
            qual_sources = st.slider("Fonti: I riferimenti erano chiari e corretti?", 1, 5, 3)

        st.divider()
        st.markdown("### 3. Conclusione")
        nps_score = st.slider("Consiglieresti Study Buddy a un collega universitario? (0=No, 10=Assolutamente sì)", 0, 10, 5)
        
        comments = st.text_area("Eventuali commenti o suggerimenti:")
        
        submitted = st.form_submit_button("Invia Risposte")
        
        if submitted:
            save_questionnaire_results({
                'age': age,
                'gender': gender,
                'enrollment': enrollment,
                'sus_q1': sus_q1, 'sus_q2': sus_q2, 'sus_q3': sus_q3, 'sus_q4': sus_q4, 'sus_q5': sus_q5,
                'sus_q6': sus_q6, 'sus_q7': sus_q7, 'sus_q8': sus_q8, 'sus_q9': sus_q9, 'sus_q10': sus_q10,
                'qual_completeness': qual_completeness,
                'qual_clarity': qual_clarity, 
                'qual_utility': qual_utility,
                'qual_trust': qual_trust,
                'qual_sources': qual_sources,
                'nps_score': nps_score,
                'comments': comments
            })
            st.session_state.study_completed = True
            st.rerun()

def show_completion_screen():
    st.balloons()
    
    st.success("🎉 Studio Completato con Successo!")
    st.markdown("""
    ### Grazie per la tua partecipazione!
    
    I tuoi dati sono stati registrati in forma anonima.
    Puoi chiudere questa pagina.
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
    
    unique_files = list(dict.fromkeys([os.path.abspath(p.replace("\\\\", "\\").replace("\\", "/").strip("'\"")) for p in file_paths_list]))
    
    # 1. Images - Show directly
    images = [p for p in unique_files if p.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
    for img_path in images:
        if os.path.exists(img_path):
            try: st.image(img_path, caption=os.path.basename(img_path), width='content')
            except Exception as e: st.error(f"Error image: {e}")
            
    # 2. Documents - Lazy Load via Selectbox
    docs = [p for p in unique_files if p not in images and os.path.exists(p)]
    
    if docs:
        label = "📂 File allegati" if len(docs) > 1 else f"📂 File: {os.path.basename(docs[0])}"
        with st.expander(f"{label} ({len(docs)})", expanded=False):
            # Selectbox for lazy loading
            options = sorted(docs, key=lambda x: os.path.basename(x))
            selected_doc = st.selectbox(
                "Scegli un file da scaricare:",
                options=options,
                format_func=lambda x: os.path.basename(x),
                key=f"sel_msg_{message_index}"
            )
            
            if selected_doc:
                try:
                    mime_type = mimetypes.guess_type(selected_doc)[0] or "application/octet-stream"
                    file_name = os.path.basename(selected_doc)
                    with open(selected_doc, "rb") as f:
                        icon = "📊" if file_name.endswith(".csv") else "📄" if file_name.endswith(".pdf") else "📥"
                        st.download_button(
                            label=f"{icon} Scarica {file_name}",
                            data=f.read(),
                            file_name=file_name,
                            mime=mime_type,
                            key=f"dl_msg_{message_index}_{file_name}",
                            width='content'
                        )
                except Exception as e: st.error(f"Error reading file: {e}")

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

def show_documents_sidebar():
    """Mostra una sezione nella sidebar per scaricare i documenti del corso, filtrati per materia attiva."""
    st.sidebar.divider()
    st.sidebar.header("📚 Materiale Didattico")
    
    docs_root = Path("data/raw")
    if not docs_root.exists():
        st.sidebar.info("Nessun documento trovato.")
        return

    # Categories (Folders)
    categories = {
        "Syllabus": "syllabuses",
        "Slides": "slides", 
        "Libri": "books",
        "Materiale Integrativo": "supplementary_materials",
        "Esercizi": "exercises",
        "Riferimenti": "references",
        "Multimedia": "multimedia"
    }

    # Classification Rules (Heuristic)
    subject_keywords = {
        "LP": ["LP", "Compilatore", "Analizzatore", "Lezione 1-Presentazione", "1. Introduzione"],
        "MRI": [
            "MRI", "InformationRetrieval", "InformationFiltering", "Lucene", "Page_Rank", 
            "Information Retrieval", "Information Filtering", "Information_Retrieval", "Information_Extraction", "Information_Filtering",
            "Text_Categorization", "Text Categorization", "Collaborative", "Content-Based", "Vector_Space", 
            "Relevance_Feedback", "Valutation", "Evaluation"
        ],
        "SIIA": [
            "SIIA", "Lesson_", "Semantics", "Linked_Data", "RecSys", "Recommender", "Knowledge", 
            "LLM", "Learning", "WEB", "NETFLIX", "Neuro-symbolic", "Ontology"
        ],
    }
    
    # Files to exclude globally
    excluded_files = [
        "en_2025_10_15_PINTO_807348.pdf", 
        "SIIA_syllabus.txt"
    ]

    # Helper to classify file
    def get_subject(filename, specific_keywords):
        for subj, keywords in specific_keywords.items():
            for kw in keywords:
                if kw.lower() in filename.lower():
                    return subj
        return "Altro"

    # --- DETERMINE ACTIVE SUBJECT ---
    # User Request: Always show ALL subjects (Folders for Subjects)
    active_subjects = list(subject_keywords.keys())
    
    # Pre-load and organize all files
    # buckets for active subjects + Altro
    categorized_files = {subj: {cat: [] for cat in categories} for subj in active_subjects}
    categorized_files["Altro"] = {cat: [] for cat in categories}

    total_files = 0
    
    # Scan files
    for cat_label, folder_name in categories.items():
        folder_path = docs_root / folder_name
        if folder_path.exists():
            for f in folder_path.glob("*.pdf"):
                if f.name in excluded_files: continue
                
                # Check against ALL keywords to find its true subject first
                true_subj = "Altro"
                for s_key, kw_list in subject_keywords.items():
                    for kw in kw_list:
                        if kw.lower() in f.name.lower():
                            true_subj = s_key
                            break
                    if true_subj != "Altro": break
                
                # Logic:
                # If matches active subject -> Add to Subject
                # Else -> Add to Altro (so it's accessible)
                if true_subj in active_subjects:
                    categorized_files[true_subj][cat_label].append(f)
                    total_files += 1
                else:
                    # It matches another subject OR is generic Altro.
                    # We put it in Altro so user can find it if they really want.
                    categorized_files["Altro"][cat_label].append(f)
                    total_files += 1

    if total_files == 0:
        st.sidebar.info("Nessun documento trovato.")
        return

    # Display in Sidebar
    # 1. Active Subjects
    for subj in active_subjects:
        files_map = categorized_files[subj]
        has_files = any(files_map.values())
        
        if has_files:
            container = st.sidebar
            if len(active_subjects) > 1:
                container = st.sidebar.expander(f"📘 {subj}", expanded=False)
            else:
                st.sidebar.markdown(f"### 📘 {subj}")
            
            with container:
                for cat_label in categories.keys():
                    files = files_map[cat_label]
                    if files:
                        with st.expander(f"📂 {cat_label} ({len(files)})", expanded=False):
                            # Optimization: Use selectbox to avoid loading ALL files into memory
                            file_options = sorted(files, key=lambda x: x.name)
                            selected_file_path = st.selectbox(
                                "Seleziona un file da scaricare:",
                                options=file_options,
                                format_func=lambda x: x.name,
                                key=f"sel_{subj}_{cat_label}"
                            )
                            
                            if selected_file_path:
                                try:
                                    with open(selected_file_path, "rb") as pdf_file:
                                        st.download_button(
                                            label=f"⬇️ Scarica {selected_file_path.name}",
                                            data=pdf_file,
                                            file_name=selected_file_path.name,
                                            mime="application/pdf",
                                            key=f"dl_{subj}_{cat_label}"
                                        )
                                except Exception as e:
                                    st.error(f"Errore caricamento file: {e}")
            if len(active_subjects) == 1:
                st.sidebar.divider()

    # 2. Others / Remaining Files (Always show if populated)
    # This catches "missed" files or files from other subjects
    has_other = any(categorized_files["Altro"].values())
    if has_other:
        with st.sidebar.expander("📂 Altri Documenti", expanded=False):
            st.caption("Documenti non classificati o di altre materie")
            for cat_label in categories.keys():
                files = categorized_files["Altro"][cat_label]
                if files:
                    with st.expander(f"📂 {cat_label} ({len(files)})", expanded=False):
                        file_options = sorted(files, key=lambda x: x.name)
                        selected_file_path = st.selectbox(
                            "Seleziona un file:",
                            options=file_options,
                            format_func=lambda x: x.name,
                            key=f"sel_Altro_{cat_label}"
                        )
                        
                        if selected_file_path:
                             try:
                                with open(selected_file_path, "rb") as pdf_file:
                                    st.download_button(
                                        label=f"⬇️ Scarica {selected_file_path.name}",
                                        data=pdf_file,
                                        file_name=selected_file_path.name,
                                        mime="application/pdf",
                                        key=f"dl_Altro_{cat_label}"
                                    )
                             except: pass

def sidebar_configuration():
    with st.sidebar:
        # BOTTONE RICARICA (utile in Dev)
        if st.button("🔄 Riavvia App"):
            st.rerun()
            
        if st.button("🏠 Cambia Modalità"):
            st.session_state.clear()
            st.rerun()
            
        if st.session_state.app_mode == "study":
            st.header("Sessione di Studio")
            
            # MOSTRA OBIETTIVO SCENARIO
            course_for_chat = "None"
            if "selected_scenario" in st.session_state:
                scen = SCENARIOS.get(st.session_state.selected_scenario)
                if scen:
                    with st.expander("Dettagli Scenario", expanded=True):
                        st.markdown(f"**{scen['title']}**")
                        st.info(scen['goal'])
                    
                    # Map subject code to full course name for the agent
                    subj_map = {
                        "SIIA": "Semantics in Intelligent Information Access",
                        "MRI": "Metodi per il Ritrovamento dell'Informazione",
                        "LP": "Linguaggi di programmazione (LP)"
                    }
                    course_for_chat = subj_map.get(scen.get("subject"), "None")
            
            if st.button("Concludi e Compila Questionario", type="primary", use_container_width=True):
                st.session_state.show_questionnaire = True
                st.rerun()
            
            # Show shared documents
            show_documents_sidebar()
            
            st.divider()
            
            # --- MODEL SELECTION (STUDY MODE) ---
            with st.expander("⚙️ Impostazioni Modello"):
                render_provider_selector()
                st.info("Modificare il provider solo se necessario.")

            st.divider()
            
            # Initialize config for Student Mode
            config_thread_id = st.session_state.get("active_thread_id")
            config_chat = ConfigChat(complexity_level="Intermediate", course=course_for_chat)
            return config_thread_id, config_chat
        
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
                render_provider_selector() # Replaced inline logic with helper

                config_complexity = st.selectbox("Livello", ('None', 'Base', 'Intermediate', 'Advanced'))
                config_course = st.selectbox("Corso", ("None", "Semantics in Intelligent Information Access", "Metodi per il Ritrovamento dell'Informazione", "Linguaggi di programmazione (LP)"))
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
    instr.append("IMPORTANT: Do NOT mention absolute file paths in your response. Reference documents by filename or title only.")
    
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