import os
import json
from pathlib import Path
import hashlib
import whisper

whisper_model = whisper.load_model("base")


def transcribe(file_path, output_text_path):
    """
    Transcribes an audio file and saves the transcription to a .txt file.

    Args:
        file_path (str): The path to the audio file to be transcribed.
        output_text_path (str): The path where the transcription text file will be saved.

    Returns:
        None
    """
    result = whisper_model.transcribe(file_path, fp16=False)
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(result["text"])


# TODO verify if this function can be removed
def extract_lesson_folders(folder_path):
    """
    Extracts and returns a list of lesson folders from the given course directory.

    Args:
        folder_path (str): The path to the directory containing lesson folders.
    Returns:
        list: A list of folder names that start with "lesson" and are directories.
    Raises:
        FileNotFoundError: If the specified folder path does not exist.    
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder path {folder_path} does not exist.")

    lesson_folders = [
        folder for folder in os.listdir(folder_path)
        if folder.startswith("lesson") and os.path.isdir(os.path.join(folder_path, folder))
    ]

    return lesson_folders


def extract_lesson_list(course_path):
    """
    Extracts a list of lesson folder names from the metadata.json file in the specified course directory.
    Args:
        course_path (str): The path to the course directory containing the metadata.json file.
    Returns:
        list: A list of folder names for each lesson in the course.
    Raises:
        FileNotFoundError: If the metadata.json file does not exist in the specified course directory.
    """
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


def create_content_metadata(folder_path, text_path):
    """
    Creates a combined content dictionary containing text content and selected metadata.
    Args:
        folder_path (str): The path to the folder containing the metadata.json file.
        text_path (str): The path to the text file containing the textual content.
    Returns:
        dict: A dictionary with the combined content and selected metadata, or None if an error occurs.
    Raises:
        FileNotFoundError: If the metadata.json file or the text file does not exist.
        Exception: For any other exceptions that may occur during the process.
    """
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
    """
    Removes all non-JSON files from the specified folder.
    Args:
        folder_path (str or Path): The path to the folder from which non-JSON files should be removed.
    Returns:
        None
    Prints:
        Error messages if the specified path is not a directory or if there is an error removing a file.
        Success messages for each file that is successfully removed.
    """
    folder_path = Path(folder_path)

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


# this function helps in removing possible duplicates in a folder
def calculate_file_hash(file_path):
    """
    Calculate the SHA-256 hash of a file.
    Args:
        file_path (Path): The path to the file to be hashed.
    Returns:
        str: The SHA-256 hash of the file in hexadecimal format.
    """
    hash_sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(4096):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def remove_duplicates(folder_path):
    """
    Removes duplicate files in the specified folder based on their hash values.
    Args:
        folder_path (str or Path): The path to the folder where duplicate files need to be removed.
    Returns:
        None
    Prints:
        Error messages if the specified path is not a directory or if there are issues removing files.
        Messages indicating which duplicate files have been removed.
        A completion message when the duplicate removal process is finished.
    """
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
