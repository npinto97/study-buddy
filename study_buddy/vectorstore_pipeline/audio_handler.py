from pathlib import Path
from typing import Dict, Optional, List, Union
import logging
from mutagen import File
from langchain.schema import Document
from study_buddy.config import TEMP_DATA_DIR

import assemblyai as aai
import os

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")


def get_audio_metadata(file_path: Path) -> Dict[str, Optional[str]]:
    """
    Extract metadata from an audio file.
    Args:
        file_path (Path): Path to the audio file.
    Returns:
        dict: Dictionary containing metadata (filename, filepath, duration, type).
    """
    file_path = Path(file_path)
    metadata = {"filename": file_path.name,
                "filepath": str(file_path),
                "type": "audio"}

    try:
        audio = File(file_path)  # Supporta più formati (MP3, WAV, FLAC, etc.)
        if audio and audio.info:
            metadata["duration"] = round(audio.info.length, 2)

    except Exception as e:
        logging.error(f"Error extracting metadata from {file_path}: {e}")

    return metadata


def transcribe_audio(file_path: Path) -> List[Document]:
    """
    Transcribes an audio file using AssemblyAI and returns a list of Documents for FAISS.

    Args:
        file_path (Path): Path to the audio file to be transcribed.

    Returns:
        list: List containing a single Document object with the transcription.
    """
    try:
        temp_dir = Path(TEMP_DATA_DIR)
        temp_dir.mkdir(parents=True, exist_ok=True)

        transcriber = aai.Transcriber()
        config = aai.TranscriptionConfig(
            language_code="it",
            punctuate=True,
            format_text=True,
            speech_model=aai.SpeechModel.best
        )

        transcript = transcriber.transcribe(str(file_path), config)

        if transcript.status == aai.TranscriptStatus.error:
            logging.error(f"Transcription failed for {file_path}: {transcript.error}")
            return []

        transcription_text = transcript.text

        if not transcription_text.strip():
            logging.warning(f"Transcription for {file_path} is empty.")
            return []

        doc = Document(
            page_content=transcription_text,
            metadata=get_audio_metadata(file_path)
        )

        logging.info(f"Transcription processed for {file_path}")
        return [doc]

    except Exception as e:
        logging.error(f"Error transcribing {file_path}: {e}")
        return []
