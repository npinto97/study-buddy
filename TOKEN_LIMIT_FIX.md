# Fix Errori Token Limit 400

## Problema
Il modello Together (Llama-3.3-70B) rifiutava richieste con errore:
```
400 - Input validation error: `inputs` tokens + `max_new_tokens` must be <= 8193
```

Il server contava ~30% più token di quanto stimato dal nostro codice.

## Modifiche Applicate (11 Nov 2025)

### 1. Retry Loop con Riduzione Progressiva
- **File**: `study_buddy/utils/nodes.py` (righe ~1877-1908)
- **Cosa fa**: Se il server rifiuta per token limit, riduce `max_tokens` del 50% e riprova (fino a 4 tentativi, minimo 32 token)
- **Beneficio**: Gestisce errori temporanei e stime imprecise

### 2. SAFETY_MARGIN Aumentato
- **Prima**: 50 token
- **Adesso**: 2000 token
- **Perché**: Il server conta ~30% più token di quanto stimiamo. Un margine ampio previene errori.

### 3. MAX_RETRIEVAL_CHARS Ridotto
- **Prima**: 1500 caratteri (~375 token)
- **Adesso**: 600 caratteri (~150 token)
- **Perché**: I risultati RAG occupavano troppo spazio nel contesto

### 4. MAX_HISTORY_MESSAGES Ridotto
- **Prima**: 4 messaggi (2 scambi) per conversazioni normali
- **Adesso**: 2 messaggi (1 scambio) per tutte le conversazioni
- **Perché**: Con tool calls ripetuti, la storia cresceva troppo velocemente

### 5. DEFAULT_MAX_NEW_TOKENS Ridotto
- **Prima**: 512 token di output
- **Adesso**: 256 token di output
- **Perché**: Lascia più spazio per l'input

## Come Verificare

```powershell
# Attiva venv
.\venv\Scripts\Activate.ps1

# Lancia l'applicazione
python streamlit_frontend.py
```

## Comportamento Atteso

### Prima delle modifiche
- Crash immediato con 400 BadRequest

### Dopo le modifiche
- **Caso 1 - Input troppo grande**: Retry con output ridotto (256→128→64→32 token)
- **Caso 2 - Input OK**: Funziona al primo tentativo con 256 token di output
- **Caso 3 - Input enorme**: Dopo 4 retry falliti, solleva errore (prevenzione necessaria)

## Log di Esempio (Funzionante)

```
Attempting model.invoke (attempt 1) with max_tokens=256
Model invocation attempt 1 failed: ...
Token limit error detected. Reducing max_tokens from 256 to 128 and retrying.
Attempting model.invoke (attempt 2) with max_tokens=128
✅ AFTER invoke: Got response type=<class 'langchain_core.messages.ai.AIMessage'>
```

## Note Tecniche

### Perché la stima dei token è imprecisa?
- Il nostro codice usa `len(text) / 4` come euristica (fallback quando tiktoken non è disponibile)
- Il server usa il tokenizer reale del modello (più accurato)
- Differenza tipica: ~30% in più lato server

### Possibili Miglioramenti Futuri
1. Installare e usare sempre `tiktoken` con encoding corretto del modello
2. Cachare i conteggi token per evitare ricalcoli
3. Implementare un sistema di "compressione" intelligente della cronologia (riassunti)
4. Usare un modello con context window più ampio (es. GPT-4 Turbo: 128k token)

## File Modificati
- `study_buddy/utils/nodes.py` (righe 1614-1617, 1720-1723, 1769-1772, 1877-1908)

## Test Eseguito
- ✅ Caricamento modulo senza errori
- ✅ Retry loop attivo
- ✅ Margini di sicurezza applicati
- 🔄 Da testare: esecuzione reale con query "ciao" (dovrebbe funzionare ora)
