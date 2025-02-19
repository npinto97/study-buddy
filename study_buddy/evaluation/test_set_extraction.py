import os
import json
import openai
import logging
from dotenv import load_dotenv
from study_buddy.config import CONFIG, logger
from study_buddy.config import PROCESSED_DATA_DIR, EVAL_DATA_DIR

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") 

# path to the folder containing the documents used for vectorial indexing
json_folder = PROCESSED_DATA_DIR
documents = []

# counter for limiting API request (for testing the functions)
count = 0
MAXDOCS = 100 # limit of processed documents
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
    logging.info(f"Processed document: {filename}")
    count += 1
    if count >= MAXDOCS: # this limit must be removed for creating the final test set
        break

# Function for generating a question starting from the file content
def generate_question(text, keywords):
    keyword_str = ", ".join(keywords) if keywords else "no keywords"
    prompt = f"""
    Analizza il seguente testo e genera una domanda basata sul contenuto.
    La domanda generata deve essere simile a quelle che farebbe uno studente universitario, di diverso livello di preparazione.
    Le domande devono essere generiche e non devono fare direttamente riferimento al testo contenente la risposta.
    
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

# Function to generate an answer starting from the question and the file content
def generate_response(question, content):
    prompt = f"""
    Usa "domanda" per estrarre la risposta alla domanda da "risposta attesa".
    
    domanda: "{question}"
    risposta attesa: "{content}"
    
    Restituisci solo la risposta come la restituirebbe un tutor didattico, in italiano, senza altro testo.
    """
    response = openai.chat.completions.create(
        model=CONFIG.llm.model, 
        messages=[{"role": "system", "content": "Sei un assistente che genera risposte in base alle domande e al testo fornito."},
                  {"role": "user", "content": prompt}]
    )
    
    if response.choices:
        return response.choices[0].message.content.strip()
    else:
        return None

# Creating test dataset
test_set = []
for doc in documents:
    
    #  Filter on documents with reduced content (if any) 
    if len(doc["content"]) < 50:
        continue

    # To avoid request fails
    MAXLEN = min(len(doc["content"]), 5000)

    # Generate question
    question = generate_question(doc["content"][:MAXLEN], doc["keywords"])
    if not question:
        continue  

    # Generate answer starting from the generated question
    response = generate_response(question, doc["content"][:MAXLEN])
    if not response:
        continue  

    test_set.append({"question": question, "expected_answer": response, "filename": doc["filename"]})

    logger.info(f"Question-answer correctly generated")


test_set_path = EVAL_DATA_DIR / "test_set.json"
with open(test_set_path, "w", encoding="utf-8") as f:
    json.dump(test_set, f, indent=4, ensure_ascii=False)

logger.info(f"Test set saved in: {test_set_path}.")
