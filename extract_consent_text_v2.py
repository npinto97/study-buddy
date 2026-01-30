
from pypdf import PdfReader
import sys

try:
    reader = PdfReader("doc_privacy/Consenso al trattamento dei dati.pdf")
    print("--- START PDF CONTENT ---")
    for page in reader.pages:
        print(page.extract_text())
    print("--- END PDF CONTENT ---")
except Exception as e:
    print(f"Error: {e}")
