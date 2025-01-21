import os
import json
from pathlib import Path
import hashlib
import whisper

whisper_model = whisper.load_model("base")


def transcribe(file_path, output_text_path):
    """
    Creates a .txt file with transcription of an audio file.
    """
    result = whisper_model.transcribe(file_path, fp16=False)
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(result["text"])


def extract_lesson_folders(folder_path):
    """
    Extracts and returns a list of lesson folders from the given directory.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder path {folder_path} does not exist.")

    lesson_folders = [
        folder for folder in os.listdir(folder_path)
        if folder.startswith("lesson") and os.path.isdir(os.path.join(folder_path, folder))
    ]

    return lesson_folders


def extract_lesson_list(course_path):

    # open the metadata.json file in course folder
    metadata_path = os.path.join(course_path, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Error: no metadata.json found in {metadata_path}")

    # open the metadata file
    with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
        metadata = json.load(metadata_file)

    # from each lesson extract the folder name
    lesson_list = [lesson.get("folder") for lesson in metadata.get("lessons", [])]

    return lesson_list


# folder_path = percorso della lezione di cui il contenuto è parte
# text_path = percorso della trascrizione del contenuto
def create_content_metadata(folder_path, text_path):
    try:
        metadata_path = os.path.join(folder_path, "metadata.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Error: no metadata.json found in {metadata_path}")
        
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Error: no textual content available in {text_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
            metadata = json.load(metadata_file)

        with open(text_path, 'r', encoding='utf-8') as text_file:
            text_content = text_file.read()

        selected_metadata = {
            "lesson_title": metadata.get("title", ""),
            "lesson_number": metadata.get("lesson_number", ""),
            "keywords": metadata.get("keywords", []),
            "references": [
                {
                    "title": ref.get("title", ""),
                    "description": ref.get("description", "")
                }
                for ref in metadata.get("references", [])
            ],
            "supplementary_materials": [
                {
                    "title": sup.get("title", ""),
                    "description": sup.get("description", "")
                }
                for sup in metadata.get("supplementary_materials", [])
            ]
        }

        combined_content = {
            "content": text_content,
            "metadata": selected_metadata
        }

        return combined_content
    
    except Exception as e:
        print(e)
        return None


def remove_non_json_files(folder_path):
    foder_path = Path(folder_path)

    if not folder_path.is_dir():
        print(f"Error: {folder_path} is not a directory.")
        return
    
    for file in folder_path.iterdir():
        if file.is_file() and not file.name.endswith(".json"):
            try:
                file.unlink()
                print(f"File {file} removed.")

            except Exception as e:
                print(f"Error removing file {file}: {e}")


def calculate_file_hash(file_path):
    hash_sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(4096):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def remove_duplicates(folder_path):
    folder_path = Path(folder_path)

    if not folder_path.is_dir():
        print(f"Error: {folder_path} is not a directory.")
        return
    
    file_hashes = {}

    for file in folder_path.iterdir():
        if file.is_file():
            try:
                file_hash = calculate_file_hash(file)

                if file_hash in file_hashes:
                    file.unlink()
                    print(f"Duplicate named {file} removed.")
                else:
                    file_hashes[file_hash] = file
            
            except Exception as e:
                print(f"Error removing file {file}: {e}")

    print(f"Duplicate removal for {folder_path} completed.")