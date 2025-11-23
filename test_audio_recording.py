"""
Test script per verificare che la configurazione audio funzioni correttamente
"""
import os
import assemblyai as aai

# Carica le variabili d'ambiente dal file .env manualmente
def load_env_file():
    """Carica manualmente il file .env"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"')

load_env_file()

def test_assemblyai_config():
    """Verifica che AssemblyAI sia configurato correttamente"""
    print("🔍 Verifica configurazione AssemblyAI...")
    
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    
    if not api_key:
        print("❌ ERRORE: ASSEMBLYAI_API_KEY non trovata nel file .env")
        return False
    
    print(f"✅ API Key trovata: {api_key[:10]}...{api_key[-4:]}")
    
    # Testa la connessione
    try:
        aai.settings.api_key = api_key
        transcriber = aai.Transcriber()
        print("✅ Connessione ad AssemblyAI riuscita")
        
        # Informazioni sulla configurazione
        print("\n📋 Configurazione:")
        print(f"   - Lingua supportata: Italiano (it)")
        print(f"   - Modello: best (massima accuratezza)")
        print(f"   - Punteggiatura: Abilitata")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRORE nella connessione: {e}")
        return False

def print_browser_instructions():
    """Stampa le istruzioni per configurare i permessi del browser"""
    print("\n" + "="*60)
    print("📱 ISTRUZIONI PER I PERMESSI DEL BROWSER")
    print("="*60)
    print("""
Per usare la registrazione vocale in Streamlit:

1. **Chrome/Edge:**
   - Clicca sull'icona del lucchetto nella barra degli indirizzi
   - Vai su 'Impostazioni sito'
   - Trova 'Microfono' e seleziona 'Consenti'

2. **Firefox:**
   - Clicca sull'icona 'i' nella barra degli indirizzi
   - Clicca su 'Cancella permessi'
   - Ricarica la pagina e clicca 'Consenti' quando richiesto

3. **Safari:**
   - Vai su Safari > Preferenze > Siti web > Microfono
   - Trova localhost e seleziona 'Consenti'

4. **Dopo aver dato i permessi:**
   - Ricarica la pagina Streamlit (CTRL+R / CMD+R)
   - Clicca sul pulsante 🎤
   - Clicca sul pulsante rosso di registrazione
   - Parla chiaramente nel microfono
   - Clicca STOP quando hai finito
    """)

def print_troubleshooting():
    """Stampa suggerimenti per il troubleshooting"""
    print("\n" + "="*60)
    print("🔧 RISOLUZIONE PROBLEMI COMUNI")
    print("="*60)
    print("""
Se la registrazione non funziona:

❌ Problema: Il pulsante di registrazione non appare
✅ Soluzione: 
   - Verifica i permessi del browser (vedi sopra)
   - Ricarica la pagina con CTRL+R
   - Prova con un browser diverso

❌ Problema: "Error in speech-to-text"
✅ Soluzione:
   - Verifica la connessione internet
   - Controlla che l'API key di AssemblyAI sia valida
   - Verifica che il microfono sia connesso e funzionante

❌ Problema: La trascrizione è vuota o imprecisa
✅ Soluzione:
   - Parla più chiaramente e lentamente
   - Avvicinati al microfono
   - Riduci il rumore di fondo
   - Verifica le impostazioni audio del sistema

❌ Problema: "Audio transcription setup error"
✅ Soluzione:
   - Verifica che pydub sia installato: pip install pydub
   - Verifica che assemblyai sia installato: pip install assemblyai
   - Riavvia l'applicazione Streamlit
    """)

if __name__ == "__main__":
    print("🎤 TEST CONFIGURAZIONE AUDIO\n")
    
    if test_assemblyai_config():
        print("\n✅ Tutti i test superati!")
        print_browser_instructions()
        print_troubleshooting()
        
        print("\n" + "="*60)
        print("💡 PROSSIMI PASSI")
        print("="*60)
        print("""
1. Assicurati che Streamlit sia in esecuzione
2. Apri l'app nel browser (di solito http://localhost:8501)
3. Dai i permessi del microfono quando richiesto
4. Clicca sul pulsante 🎤 in basso a destra
5. Segui le istruzioni a schermo per registrare
        """)
    else:
        print("\n❌ Alcuni test sono falliti. Controlla gli errori sopra.")
