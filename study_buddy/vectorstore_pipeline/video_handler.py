from pathlib import Path
from typing import Dict, Optional, List
import logging
import ffmpeg
from langchain.schema import Document
from study_buddy.config import TEMP_DATA_DIR
import assemblyai as aai
import os

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")


def extract_audio_from_video(video_path: Path) -> Path:
    """
    Extracts audio from a video file and saves it as a temporary WAV file.
    """
    temp_audio_path = TEMP_DATA_DIR / f"{video_path.stem}.wav"
    temp_audio_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ffmpeg.input(str(video_path)).output(str(temp_audio_path), format="wav").run(overwrite_output=True)
        logging.info(f"Audio extracted from {video_path} and saved in {temp_audio_path}")
        return temp_audio_path
    except Exception as e:
        logging.error(f"Error during audio extraction: {e}")
        raise


def get_video_metadata(file_path: Path) -> Dict[str, Optional[str]]:
    """
    Extracts metadata from a video file.
    """
    metadata = {"filename": file_path.name, "filepath": str(file_path), "type": "video"}

    try:
        probe = ffmpeg.probe(str(file_path))
        video_streams = [stream for stream in probe["streams"] if stream["codec_type"] == "video"]

        if video_streams:
            metadata["duration"] = round(float(video_streams[0]["duration"]), 2)

    except Exception as e:
        logging.error(f"Error extracting metadata: {e}")

    return metadata


def transcribe_video(file_path: Path) -> List[Document]:
    """
    Extracts audio from a video, transcribes it using AssemblyAI
    and returns the metadata.
    """
    try:
        temp_audio = extract_audio_from_video(file_path)

        transcriber = aai.Transcriber()
        config = aai.TranscriptionConfig(
            language_code="it",
            punctuate=True,
            format_text=True,
            speech_model=aai.SpeechModel.best
        )

        transcript = transcriber.transcribe(str(temp_audio), config)

        if transcript.status == aai.TranscriptStatus.error:
            logging.error(f"Transcription failed for {file_path}: {transcript.error}")
            return []

        transcription_text = transcript.text

        if not transcription_text.strip():
            logging.warning(f"Transcription empty for {file_path}")
            return []

        doc = Document(
            page_content=transcription_text,
            metadata=get_video_metadata(file_path)
        )

        logging.info(f"Transcription completed for {file_path}")
        return [doc]

    except Exception as e:
        logging.error(f"Error in transcription for {file_path}: {e}")
        return []

    finally:
        # Rimuove il file audio temporaneo
        if temp_audio.exists():
            temp_audio.unlink()
