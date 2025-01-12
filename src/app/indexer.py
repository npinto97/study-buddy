import os
import json
from PyPDF2 import PdfReader


def index_data(data_dir="data/"):
    index = []

    # Walk through the directory
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".json"):
                # Index JSON files
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        index.append({
                            "file_type": "json",
                            "title": data.get("course_name") 
                            or data.get("title"),
                            "keywords": data.get("keywords", []),
                            "content": data,  # Store full JSON for flexibility
                            "path": file_path
                        })
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON: {file_path}")
            elif file.endswith(".pdf"):
                # Index PDF files
                try:
                    reader = PdfReader(file_path)
                    content = "".join([page.extract_text() for
                                       page in reader.pages])
                    index.append({
                        "file_type": "pdf",
                        "title": os.path.basename(file),
                        "keywords": [],
                        # Store first 500 chars for search
                        "content": content[:500],  
                        "path": file_path
                    })
                except Exception as e:
                    print(f"Error reading PDF: {file_path} - {str(e)}")

    return index


def save_index(index, output_file="data/index.json"):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=4, ensure_ascii=False)


def load_index(index_file="data/index.json"):
    with open(index_file, "r", encoding="utf-8") as f:
        return json.load(f)
