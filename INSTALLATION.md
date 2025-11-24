# Study Buddy - Installation Guide

Guida completa all'installazione di Study Buddy con supporto GPU.

## Prerequisiti

### Software Richiesto

1. **Python 3.10+**
   - Scarica da: https://www.python.org/downloads/
   - Durante l'installazione, seleziona "Add Python to PATH"

2. **NVIDIA GPU Drivers** (opzionale, per supporto GPU)
   - Scarica da: https://www.nvidia.com/Download/index.aspx
   - Verifica l'installazione con: `nvidia-smi`

3. **FFmpeg** (per elaborazione audio/video)
   - Scarica da: https://www.gyan.dev/ffmpeg/builds/
   - Estrai e aggiungi la cartella `bin` al PATH di sistema

4. **Tesseract OCR** (per OCR di immagini)
   - Scarica da: https://github.com/UB-Mannheim/tesseract/wiki
   - Aggiungi al PATH di sistema

5. **Poppler** (opzionale, per OCR di PDF scansionati)
   - Scarica da: https://github.com/oschwartz10612/poppler-windows/releases/
   - Estrai e aggiungi la cartella `bin` al PATH

## Installazione Automatica (Consigliata)

### Windows

1. Apri PowerShell nella cartella del progetto
2. Esegui lo script di setup:
   ```powershell
   .\setup.ps1
   ```

Lo script:
- Crea automaticamente l'ambiente virtuale
- Rileva la presenza di GPU NVIDIA
- Installa la versione corretta di PyTorch (CPU o CUDA)
- Installa tutte le dipendenze necessarie

## Installazione Manuale

### 1. Crea l'ambiente virtuale

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Installa PyTorch

**Con GPU (CUDA 12.8):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Solo CPU:**
```powershell
pip install torch torchvision torchaudio
```

### 3. Installa le dipendenze

```powershell
pip install -r requirements.txt
```

### 4. Installa pacchetti aggiuntivi

```powershell
pip install ffmpeg-python youtube-transcript-api wikipedia google-search-results
```

## Configurazione

### 1. Crea il file `.env`

Copia `.env.example` in `.env` e configura le tue API keys:

```bash
# LLM Providers
TOGETHER_API_KEY=your_together_api_key
GOOGLE_API_KEY=your_google_api_key

# Audio Services
ELEVEN_API_KEY=your_elevenlabs_api_key
ASSEMBLYAI_API_KEY=your_assemblyai_api_key

# Search & Tools
SERP_API_KEY=your_serpapi_key
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_LENS_API_KEY=your_google_lens_key
IMGBB_API_KEY=your_imgbb_key

# Code Execution
E2B_API_KEY=your_e2b_api_key
```

### 2. Verifica l'installazione

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Avvio dell'Applicazione

```powershell
streamlit run streamlit_frontend.py
```

L'applicazione sarà disponibile su: http://localhost:8502

## Risoluzione Problemi

### GPU non rilevata

Se hai una GPU NVIDIA ma PyTorch usa la CPU:

1. Verifica i driver NVIDIA: `nvidia-smi`
2. Reinstalla PyTorch con CUDA:
   ```powershell
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

### Errori di importazione

Se ricevi errori `ModuleNotFoundError`:

```powershell
pip install <nome_pacchetto_mancante>
```

Pacchetti comuni che potrebbero mancare:
- `ffmpeg-python`
- `youtube-transcript-api`
- `wikipedia`
- `google-search-results`

### Errori OCR

Se l'OCR non funziona:

1. Verifica che Tesseract sia installato: `tesseract --version`
2. Verifica che sia nel PATH di sistema
3. Per PDF scansionati, installa Poppler

### Ctrl+C non funziona

Streamlit a volte non risponde immediatamente a Ctrl+C su Windows. Soluzioni:

1. Premi Ctrl+C più volte
2. Chiudi il terminale
3. Usa Task Manager per terminare il processo Python

## Aggiornamento

Per aggiornare le dipendenze:

```powershell
pip install --upgrade -r requirements.txt
```

## Supporto

Per problemi o domande, consulta:
- README.md del progetto
- Documentazione di Streamlit: https://docs.streamlit.io/
- Documentazione PyTorch: https://pytorch.org/docs/
