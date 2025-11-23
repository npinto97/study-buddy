# 🔊 Funzionalità Riascolto Audio

## ✨ Miglioramenti Implementati

### 1. **Salvataggio Permanente delle Registrazioni**
- Le registrazioni vocali vengono ora salvate permanentemente in `uploaded_files/audio/`
- Ogni file audio ha un nome unico basato sul timestamp: `audio_YYYYMMDD_HHMMSS.wav`
- I file rimangono disponibili anche dopo il riavvio dell'applicazione

### 2. **Player Audio nella Chat**
- Quando invii un messaggio vocale, vedrai un player audio sotto il testo trascritto
- Il player ti permette di:
  - ▶️ Riascoltare la registrazione
  - ⏸️ Mettere in pausa
  - 🔊 Regolare il volume
  - ⏩ Avanzare/riavvolgere

### 3. **Cronologia Persistente**
- Tutte le registrazioni vocali vengono salvate nella cronologia della chat
- Quando ricarichi la pagina o cambi thread, potrai ancora riascoltare le vecchie registrazioni
- Ogni messaggio vocale mostra sia il testo trascritto che il player audio

## 📋 Come Funziona

### Registrazione e Ascolto
1. **Clicca sul pulsante** 🎤 in basso a destra
2. **Registra** il tuo messaggio vocale
3. **Riascolta immediatamente** la registrazione appena sopra il testo trascritto
4. **Rivedi la cronologia** - scorri in alto per riascoltare vecchie registrazioni

### Dove Vengono Salvati i File
```
univox/
├── uploaded_files/
│   └── audio/
│       ├── audio_20251109_143022.wav
│       ├── audio_20251109_143145.wav
│       └── audio_20251109_144301.wav
```

### Formato Audio
- **Formato**: WAV (non compresso)
- **Qualità**: Alta qualità per trascrizione accurata
- **Compatibilità**: Funziona su tutti i browser moderni

## 🔧 Modifiche Tecniche

### 1. `transcribe_audio()` - Nuova Funzionalità
```python
# Prima: cancellava il file dopo la trascrizione
return transcribed_text

# Ora: salva il file e restituisce anche il percorso
return transcribed_text, audio_path
```

### 2. `add_message_to_history()` - Supporto Audio
- Nuovo parametro `audio_path` opzionale
- Salva il percorso nel JSON della cronologia
- Esempio JSON:
```json
{
  "role": "user",
  "content": "Ciao, come stai?",
  "audio_path": "uploaded_files/audio/audio_20251109_143022.wav"
}
```

### 3. `display_chat_history()` - Visualizzazione Player
- Controlla se il messaggio ha un `audio_path`
- Mostra automaticamente il player audio se il file esiste
- Gestisce gracefully i file mancanti

## ⚠️ Note Importanti

1. **Spazio su Disco**: Le registrazioni occupano spazio. Considera di eliminare periodicamente i vecchi file audio se non servono più.

2. **Privacy**: I file audio vengono salvati localmente sul server. Non condividere la cartella `uploaded_files/audio/` pubblicamente.

3. **Backup**: Se fai backup della chat history, ricorda di includere anche la cartella `uploaded_files/audio/`.

4. **Performance**: I file WAV non compressi possono essere grandi. Per chat molto lunghe con molte registrazioni, considera di:
   - Eliminare vecchie registrazioni
   - Comprimere i file in MP3 (richiede modifiche al codice)

## 🎯 Prossimi Passi Suggeriti

### Funzionalità Aggiuntive Possibili
- [ ] Pulsante per scaricare le registrazioni vocali
- [ ] Compressione automatica in MP3 per risparmiare spazio
- [ ] Pulizia automatica di registrazioni vecchie (es. > 30 giorni)
- [ ] Indicatore visivo per messaggi con audio (icona 🎤)
- [ ] Statistiche: durata totale registrazioni, numero di messaggi vocali, ecc.

## 🐛 Troubleshooting

### Il player non mostra l'audio
- Verifica che il file esista in `uploaded_files/audio/`
- Controlla i permessi della cartella
- Verifica nella console del browser per errori

### L'audio non si sente
- Controlla il volume del browser
- Verifica che il file WAV sia valido
- Prova a scaricare il file e aprirlo con un player esterno

### Errore durante la registrazione
- Verifica i permessi del microfono
- Controlla che `pydub` e `assemblyai` siano installati
- Verifica la chiave API di AssemblyAI nel file `.env`

## 📝 Log delle Modifiche

### Versione 2.0 - 9 Novembre 2025
- ✅ Implementato salvataggio permanente delle registrazioni
- ✅ Aggiunto player audio nella chat
- ✅ Integrato supporto audio nella cronologia
- ✅ Creata directory dedicata per gli audio
- ✅ Aggiornata documentazione

---

Per domande o problemi, contatta il team di sviluppo! 🚀
