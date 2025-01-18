import os
import json
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


# TODO: the function should create the list starting from the metadata.json file of the course
def extract_lesson_list(course_path):
        """
        Extracts and returns a list of lesson folders from the given course directory.
        """
        lesson_folders = extract_lesson_folders(course_path)
        lesson_list = [folder for folder in lesson_folders if folder.startswith("lesson")]

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
            metadata_content = json.load(metadata_file)

        print(f"metadata_content: {metadata_content}")

        with open(text_path, 'r', encoding='utf-8') as text_file:
            text_content = text_file.read()

        print(f"text_content: {text_content}")
        
        combined_content = {
            "content": text_content,
            **metadata_content
        }

        print(f"\n\ncombined_content: {combined_content}")

        # output_json_path = os.path.join(folder_path, "combined_content.json")
        # with open(output_json_path, 'w', encoding='utf-8') as output_file:
        #     json.dump(combined_content, output_file, indent=4, ensure_ascii=False)

        # print(f"JSON file correctly created: {output_json_path}")
        return combined_content
    
    except Exception as e:
        print(e)
        return None
