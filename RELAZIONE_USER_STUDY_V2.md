# Relazione Tecnica: Studio Utente e Aggiornamenti Sistema Study Buddy (UniVox)

Questo documento riassume il lavoro svolto per l'implementazione dello **User Study** e le ottimizzazioni apportate nell'ultima sessione di sviluppo (Dicembre 2025).

---

## 1. Studio Utente (User Study)

È stata implementata una modalità dedicata per consentire la valutazione del sistema da parte di studenti in un ambiente controllato e isolato.

### 1.1 Modalità Studio vs. Modalità Dev
All'avvio dell'applicazione, l'utente può scegliere tra:
-   **Partecipante allo Studio**: Ambiente isolato. I dati sono anonimizzati e le sessioni di chat vengono salvate in una cartella specifica basata su un ID univoco (`study_id`).
-   **Modalità Sviluppatore**: Accesso completo per debug, richiede una password amministratore.

### 1.2 Scenari di Test
Sono stati definiti tre scenari realistici ("Missioni") per guidare l'utente:
1.  **Preparation (The Exam Prep)**: Recupero di informazioni specifiche dalle slide introduttive (es. Information Overload).
2.  **Practical (The Practical Student)**: Ricerca di informazioni logistiche (orari di ricevimento dei docenti).
3.  **Curriculum (The Curriculum Analyst)**: Analisi comparativa tra diversi corsi (MRI vs SIIA).

### 1.3 Raccolta Dati e Tracciamento
Il sistema registra automaticamente:
-   **Interazioni (JSONL)**: Tutte le domande e risposte durante la modalità studio vengono salvate in `data/study_logs/`.
-   **Metadiche di Risposta**: Inclusi gli strumenti (tools) utilizzati e il numero di documenti recuperati.
-   **Questionario Finale (SUS)**: Un modulo finale raccoglie il feedback dell'utente su usabilità e sicurezza. I risultati vengono salvati in `data/study_results/results.csv`.
-   **Codice di Completamento**: Generato tramite hashing dell'ID di sessione per confermare la partecipazione. Salvato in `data/study_results/codes.csv`.

---

## 2. Ultimi Aggiornamenti Tecnici (Ultima Commit)

Il sistema ha subito un profondo refactoring per migliorare stabilità, prestazioni e qualità delle risposte (RAG).

### 2.1 Supporto Documentale e Correzioni Path
-   **Supporto DOCX**: Implementato il caricamento e l'estrazione di testo da file Microsoft Word tramite `UnstructuredWordDocumentLoader`.
-   **Sanitizzazione Percorsi**: Introdotta una logica robusta per gestire i percorsi Windows con caratteri speciali (accenati, spazi) e trasformarli in percorsi portabili (relativi).

### 2.2 Rebuild della Base di Conoscenza (FAISS)
La "memoria" del sistema è stata ricostruita da zero:
-   **Da 598 a 22.414 chunk**: Un incremento massiccio della granularità delle informazioni.
-   **Nuovo Modello di Embedding**: Utilizzo di `BAAI/bge-m3` per una migliore comprensione semantica.
-   **Pulizia della Cache**: Eliminati i vecchi metadati corrotti che puntavano a percorsi non esistenti.

### 2.3 Ottimizzazione Token e "Budgeting"
Per evitare risposte troncate o errori nelle chiamate API:
-   **Max Tokens**: Incrementato il limite di generazione da 256 a **2048 token**.
-   **Memoria Conversazionale**: Estesa la cronologia da 2 a **6 messaggi**.
-   **Gestione Dinamica**: Implementato un algoritmo che calcola il budget di token residui per l'input e l'output, troncando i messaggi più lunghi se necessario per non superare il limite del modello.

### 2.4 Funzionalità Avanzate
-   **Google Lens con Auto-Upload**: Le immagini caricate vengono ora caricate automaticamente su un server temporaneo (imgbb) per permettere all'API di Google Lens di analizzarle correttamente tramite URL pubblico.
-   **Miglioramento Audio**: Registrazione e riproduzione audio più stabili nel frontend, con salvataggio persistente dei file WAV.
-   **Loguru Logging**: Implementato un sistema di logging colorato ed emoji-coded per il debug in tempo reale nel terminale.

---

## 3. Risultati e Metriche

| Metrica | Prima | Dopo | Miglioramento |
| :--- | :--- | :--- | :--- |
| **Chunk nel Vector Store** | 598 | 22.414 | **+3.649%** |
| **Capacità Risposta (Token)** | 256 | 2.048 | **+700%** |
| **Supporto DOCX** | No | Sì | **Implementato** |
| **Blocchi Sistema (Hanging)** | Frequenti | Assenti | **Risolto** |
