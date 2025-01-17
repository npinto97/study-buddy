from pathlib import Path
from mutagen.mp3 import MP3
from study_buddy.data_processing.utils import transcribe


def extract_metadata_local(file_path):
    """
    Extracts metadata from a local audio file.
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


# TODO insert this logic in an utils.py script, add try-catch block here
# def transcribe_audio(file_path, output_text_path):
#     """
#     Creates a .txt file with transcription of an audio file.
#     """
#     result = whisper_model.transcribe(file_path, fp16=False)
#     with open(output_text_path, "w", encoding="utf-8") as f:
#         f.write(result["text"])


# TODO: audio_path must be modified
def transcribe_audio(file_path, output_text_path):
    """
    Creates a .txt file with transcription of a video file.
    """
    try:
        transcribe(file_path, output_text_path)

    except Exception as e:
        print(f"Error during {file_path} transcription: {e}")
