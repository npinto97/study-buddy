import os
import json
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from study_buddy.config import CONFIG, logger

# Carica la configurazione API da .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") 

# Directory dei file JSON
json_folder = "data\\processed"
documents = []

# Limite per il numero di richieste API
count = 0

# Lettura dei file JSON
for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        with open(os.path.join(json_folder, filename), 'r', encoding="utf-8") as f:
            data = json.load(f)
            doc = {
                "filename": filename,
                "content": data["content"],
                "lesson_title": data["metadata"].get("lesson_title", ""),
                "lesson_number": data["metadata"].get("lesson_number", ""),
                "keywords": data["metadata"].get("keywords", []),
                "references": [ref["title"] for ref in data["metadata"].get("references", [])],
                "supplementary_materials": [mat["title"] for mat in data["metadata"].get("supplementary_materials", [])]
            }
            documents.append(doc)
    count += 1
    if count > 3:
        break

# Funzione per generare una domanda
def generate_question(text, keywords):
    keyword_str = ", ".join(keywords) if keywords else "no keywords"
    prompt = f"""
    Analizza il seguente testo e genera una domanda basata sul contenuto.
    
    Testo: "{text}"
    Parole chiave: {keyword_str}
    
    Restituisci solo la domanda senza altro testo.
    """
    response = openai.chat.completions.create(
        model=CONFIG.llm.model,
        messages=[{"role": "system", "content": "Sei un assistente che genera domande per la valutazione della RAG."},
                  {"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content if response.choices else None

# Funzione per generare la risposta
def generate_response(question, answer):
    prompt = f"""
    Usa "domanda" per estrarre la risposta alla domanda da "risposta attesa".
    
    domanda: "{question}"
    risposta attesa: "{answer}"
    
    Restituisci solo la risposta, in italiano, senza altro testo.
    """
    response = openai.chat.completions.create(
        model=CONFIG.llm.model, 
        messages=[{"role": "system", "content": "Sei un assistente che genera risposte in base alle domande e al testo fornito."},
                  {"role": "user", "content": prompt}]
    )
    
    # Verifica che la risposta esista
    if response.choices:
        return response.choices[0].message.content.strip()
    else:
        return None

# Creazione del dataset di test
test_set = []
for doc in documents:
    # Filtra documenti con contenuti troppo brevi
    if len(doc["content"]) < 50:
        continue

    # Limita la lunghezza del contenuto per evitare richieste troppo grandi
    MAXLEN = min(len(doc["content"]), 5000)

    # Genera la domanda
    question = generate_question(doc["content"][:MAXLEN], doc["keywords"])
    if not question:
        continue  # Se non è stata generata una domanda, passa al prossimo documento

    # Genera la risposta
    response = generate_response(question, doc["content"][:MAXLEN])
    if not response:
        continue  # Se non è stata generata una risposta, passa al prossimo documento

    # Aggiungi al set di test
    test_set.append({"question": question, "expected_answer": response, "filename": doc["filename"]})

# Salva il risultato in un file JSON
with open("test_set.json", "w", encoding="utf-8") as f:
    json.dump(test_set, f, indent=4, ensure_ascii=False)

logger.info("Test set creato e salvato in 'test_set_rag.json'")
