from pathlib import Path
from mutagen.mp3 import MP3
from study_buddy.data_processing.utils import transcribe


def extract_metadata_local(file_path):
    """
    Extracts metadata from a local audio file.
    Args:
        file_path (str): The path to the audio file.
    Returns:
        dict: A dictionary containing the metadata of the audio file, including:
            - filename (str): The name of the file.
            - filepath (str): The full path to the file.
            - duration (float, optional): The duration of the audio file in seconds (if applicable).
            - type (str, optional): The type of the file (if applicable).
    Raises:
        Exception: If there is an error in extracting metadata.
    """
    
    file_path = Path(file_path)
    metadata = {"filename": file_path.name, "filepath": str(file_path)}
    try:
        if file_path.suffix.lower() in ['.mp3', '.wav']:
            audio = MP3(file_path)
            metadata.update({
                "duration": audio.info.length,
                "type": "audio"
            })
        else:
            print("file format not recognized as audio")
    except Exception as e:
        print(f"Error in metadata extraction of {file_path}: {e}")
    return metadata


def transcribe_audio(file_path, output_text_path):
    """
    Transcribes audio from the given file path and saves the transcription to the specified output text path.
    Args:
        file_path (str): The path to the audio file to be transcribed.
        output_text_path (str): The path where the transcribed text will be saved.
    Raises:
        Exception: If an error occurs during the transcription process, it will be caught and printed.
    """
    
    try:
        transcribe(file_path, output_text_path)

    except Exception as e:
        print(f"Error during {file_path} transcription: {e}")
