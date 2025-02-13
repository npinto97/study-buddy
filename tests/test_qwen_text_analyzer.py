from study_buddy.utils.tools import QwenTextAnalysis
import os
import requests

import base64

test_file_path = "C:/Users/Ningo/Desktop/basic-text.pdf"


# Leggi il file e codificalo in Base64
with open(test_file_path, "rb") as f:
    encoded_file = base64.b64encode(f.read()).decode("utf-8")

# Esegui l'analisi passando il file codificato
analyzer = QwenTextAnalysis()
result = analyzer.analyze_text(text="Questo è un testo: a e i o u", files=[encoded_file])

print(f"Testo OK! Risultato: {result}")


analyzer = QwenTextAnalysis()
result = analyzer.analyze_text(text="Questo è un test.", files=[])
print(f"Testo OK! Risultato: {result}")
