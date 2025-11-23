# Google Lens Setup Guide

## Overview

Il tool `google_lens_analyze` analizza immagini utilizzando Google Lens per identificare oggetti, scene, persone, luoghi e landmark.

## Funzionalità

### Per URL pubblici
Se fornisci un URL pubblico (http:// o https://), l'immagine viene analizzata direttamente da Google Lens.

### Per file locali
Google Lens richiede un URL pubblico accessibile. Quando fornisci un file locale:

1. **Upload automatico** (se configurato): Il file viene automaticamente caricato su imgbb.com (hosting temporaneo gratuito) per 10 minuti, poi analizzato da Google Lens
2. **Fallback OCR** (se upload non disponibile): Viene eseguito OCR locale per estrarre il testo dall'immagine

## Configurazione API Keys

Aggiungi le seguenti chiavi al tuo file `.env`:

### 1. Google Lens API (OBBLIGATORIO)

Ottieni una chiave API da [SerpAPI](https://serpapi.com/):

1. Vai su https://serpapi.com/users/sign_up
2. Crea un account gratuito (100 ricerche/mese gratis)
3. Copia la tua API key dalla dashboard
4. Aggiungi al `.env`:

```bash
# Preferito (nome specifico)
GOOGLE_LENS_API_KEY=your_serpapi_key_here

# Oppure (nome generico, compatibile con altri tool SERP)
SERP_API_KEY=your_serpapi_key_here
```

### 2. ImgBB API (OPZIONALE - per upload automatico)

Per abilitare l'analisi di file locali senza limitazioni, ottieni una chiave ImgBB:

1. Vai su https://api.imgbb.com/
2. Crea un account gratuito
3. Copia la tua API key
4. Aggiungi al `.env`:

```bash
IMGBB_API_KEY=your_imgbb_key_here
```

**Nota**: Senza IMGBB_API_KEY, i file locali verranno analizzati solo tramite OCR (estrazione testo), non con analisi visuale completa.

## Esempio di utilizzo

### Con Streamlit UI
1. Carica un'immagine tramite l'interfaccia
2. Chiedi: "Cosa raffigura questa immagine?"
3. Il sistema utilizzerà automaticamente il tool `google_lens_analyze`

### Con URL pubblico
```
Utente: Analizza questa immagine: https://example.com/image.jpg
```

### Con file locale
```
Utente: Cosa vedi in questa immagine? [allega file]
```

## Comportamento per scenario

| Scenario | GOOGLE_LENS_API_KEY | IMGBB_API_KEY | Risultato |
|----------|---------------------|---------------|-----------|
| URL pubblico | ✅ | - | ✅ Analisi Google Lens completa |
| File locale | ✅ | ✅ | ✅ Upload automatico + analisi completa |
| File locale | ✅ | ❌ | ⚠️ Fallback OCR (solo testo) |
| Qualsiasi | ❌ | - | ⚠️ Fallback OCR (solo testo) |

## Limitazioni

- **File locali senza upload**: Google Lens API non supporta percorsi `file:///` locali. Senza IMGBB_API_KEY, viene eseguito solo OCR
- **Formati supportati**: PNG, JPG, JPEG, TIFF, BMP, GIF, WEBP
- **Dimensioni**: ImgBB supporta fino a 32MB per immagine
- **Scadenza upload**: Le immagini caricate su ImgBB scadono dopo 10 minuti (per privacy)

## Troubleshooting

### "Google Lens is not configured"
- Verifica che `GOOGLE_LENS_API_KEY` o `SERP_API_KEY` sia presente nel file `.env`
- Riavvia l'applicazione dopo aver modificato `.env`

### "Image upload failed"
- Verifica che `IMGBB_API_KEY` sia corretta
- Controlla la dimensione del file (max 32MB)
- Verifica connessione internet

### "OCR fallback"
- Indica che l'upload non è configurato o è fallito
- Il sistema ha comunque estratto il testo presente nell'immagine
- Per analisi visuale completa, configura `IMGBB_API_KEY`

## Privacy e sicurezza

- **ImgBB**: Le immagini caricate sono temporanee (10 minuti) e poi eliminate automaticamente
- **SerpAPI**: Le richieste a Google Lens sono anonimizzate tramite SerpAPI
- **Dati locali**: L'OCR viene eseguito localmente quando utilizzato come fallback

## Costi

- **SerpAPI**: Piano gratuito include 100 ricerche/mese
- **ImgBB**: Gratuito per upload illimitati (con rate limiting)
- **OCR locale**: Gratuito, utilizza Tesseract
