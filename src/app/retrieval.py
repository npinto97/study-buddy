import os
from PyPDF2 import PdfReader

def search_json_metadata(query, metadata_dir="data/metadata/"):
    results = []
    for root, _, files in os.walk(metadata_dir):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    data = f.read()
                    if query.lower() in data.lower():
                        results.append({"file": file, "path": os.path.join(root, file)})
    return results

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "".join([page.extract_text() for page in reader.pages])
    return text
