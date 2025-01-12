import os
import PyPDF2
import docx
from pathlib import Path

# Configurazione dei percorsi
DATA_PATH = Path("data")
OUTPUT_PATH = Path("data/processed_texts")

# Creazione directory per i file processati
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    """Estrae il testo da un file PDF."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Errore nell'estrazione del testo da {pdf_path}: {e}")
    return text


def extract_text_from_word(doc_path):
    """Estrae il testo da un file Word (.docx)."""
    text = ""
    try:
        doc = docx.Document(doc_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Errore nell'estrazione del testo da {doc_path}: {e}")
    return text


def process_files(input_path, output_path):
    """Elabora i file PDF e Word in una directory."""
    for root, _, files in os.walk(input_path):
        for file in files:
            file_path = Path(root) / file
            if file.lower().endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif file.lower().endswith(".docx"):
                text = extract_text_from_word(file_path)
            else:
                continue

            # Salva il testo estratto in un file .txt
            output_file = output_path / f"{file_path.stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as out_file:
                out_file.write(text)
            print(f"File processato: {output_file}")


if __name__ == "__main__":
    process_files(DATA_PATH, OUTPUT_PATH)
