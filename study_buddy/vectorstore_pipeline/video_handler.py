from pathlib import Path
from typing import Dict, Optional, List
import logging
import whisper
import ffmpeg
from langchain.schema import Document
from study_buddy.config import TEMP_DATA_DIR


def extract_audio_from_video(video_path: Path) -> Path:
    """
    Estrae l'audio da un file video e lo salva come file temporaneo.
    """
    temp_audio_path = TEMP_DATA_DIR / f"{video_path.stem}.wav"

    try:
        ffmpeg.input(str(video_path)).output(str(temp_audio_path), format="wav").run(overwrite_output=True)
        logging.info(f"Audio estratto da {video_path} e salvato in {temp_audio_path}")
        return temp_audio_path
    except Exception as e:
        logging.error(f"Errore durante l'estrazione audio: {e}")
        raise


def get_video_metadata(file_path: Path) -> Dict[str, Optional[str]]:
    """
    Estrae i metadati di un file video.
    """
    metadata = {"filename": file_path.name, "filepath": str(file_path), "type": "video"}

    try:
        probe = ffmpeg.probe(str(file_path))
        video_streams = [stream for stream in probe["streams"] if stream["codec_type"] == "video"]

        if video_streams:
            metadata["duration"] = round(float(video_streams[0]["duration"]), 2)

    except Exception as e:
        logging.error(f"Errore nell'estrazione dei metadati: {e}")

    return metadata


def transcribe_video(file_path: Path) -> List[Document]:
    """
    Estrae l'audio da un video, lo trascrive e restituisce i metadati.
    """
    try:
        temp_audio = extract_audio_from_video(file_path)
        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(str(temp_audio), fp16=False)

        transcription_text = result["text"]
        if not transcription_text.strip():
            logging.warning(f"Trascrizione vuota per {file_path}")
            return []

        doc = Document(
            page_content=transcription_text,
            metadata=get_video_metadata(file_path)
        )

        logging.info(f"Trascrizione completata per {file_path}")
        return [doc]

    except Exception as e:
        logging.error(f"Errore nella trascrizione di {file_path}: {e}")
        return []
