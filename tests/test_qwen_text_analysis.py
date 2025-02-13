from study_buddy.utils.tools import QwenTextAnalysis

print("Test manuale di QwenTextAnalysis...")
test_file_path = "C:/Users/Ningo/Desktop/sample-1.pdf"

analyzer = QwenTextAnalysis()
try:
    result = analyzer.analyze_text(text="", files=[test_file_path])
    print(f"Testo OK! Risultato: {result}")
except Exception as e:
    print(f"Errore: {e}")
    # Stampa ulteriori dettagli se possibile
    if hasattr(e, 'response'):
        print(f"Risposta API: {e.response.text}")
    if hasattr(e, 'status_code'):
        print(f"Codice di stato API: {e.status_code}")
