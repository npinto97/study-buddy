import os
import json
from yt_dlp import YoutubeDL
import whisper
from pathlib import Path
from mutagen.mp3 import MP3
from moviepy.editor import VideoFileClip

# Configurazione dei percorsi
LOCAL_PATH = Path("data//multimedia//")
OUTPUT_PATH = Path("data//processed_media//")
METADATA_PATH = Path("data//metadata//SIIA//")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Configurazione Whisper
whisper_model = whisper.load_model("base")


def extract_metadata_local(file_path):
    """
    Estrae i metadati da un file locale (video o audio).
    """
    metadata = {"filename": file_path.name, "filepath": str(file_path)}
    try:
        if file_path.suffix.lower() in ['.mp4', '.mkv', '.webm']:
            clip = VideoFileClip(str(file_path))
            metadata.update({
                "duration": clip.duration,
                "resolution": clip.size,
                "fps": clip.fps,
                "type": "video"
            })
            clip.close()
        elif file_path.suffix.lower() in ['.mp3', '.wav']:
            audio = MP3(file_path)
            metadata.update({
                "duration": audio.info.length,
                "type": "audio"
            })
    except Exception as e:
        print(f"Errore nell'estrazione dei metadati per {file_path}: {e}")
    return metadata


def extract_video_metadata_youtube(video_url):
    """
    Estrae i metadati di un video da un URL di YouTube.
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'format': 'best',
        'skip_download': True,
        'noplaylist': True,
        'dump_single_json': True
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            metadata = {
                "title": info_dict.get("title"),
                "description": info_dict.get("description"),
                "duration": info_dict.get("duration"),
                "upload_date": info_dict.get("upload_date"),
                "channel": info_dict.get("uploader"),
                "tags": info_dict.get("tags", []),
                "url": video_url
            }
            return metadata
    except Exception as e:
        print(f"Errore nell'estrazione dei metadati da {video_url}: {e}")
        return None


def transcribe_audio(file_path):
    """
    Trascrive il contenuto audio di un file locale usando Whisper.
    """
    try:
        result = whisper_model.transcribe(str(file_path))
        return result['text']
    except Exception as e:
        print(f"Errore nella trascrizione di {file_path}: {e}")
        return None


def process_local_files(input_path, output_path):
    """
    Processa i file multimediali locali nella directory specificata.
    """
    for root, _, files in os.walk(input_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() not in ['.mp4', '.mkv', '.webm', '.mp3', '.wav']:
                continue

            print(f"Processando file locale: {file_path}")
            metadata = extract_metadata_local(file_path)

            # Trascrizione dell'audio (se applicabile)
            if metadata.get("type") in ["audio", "video"]:
                transcript = transcribe_audio(file_path)
                metadata['transcription'] = transcript

            # Salvataggio dei metadati
            output_file = output_path / f"{file_path.stem}_metadata.json"
            with open(output_file, 'w', encoding='utf-8') as out_file:
                json.dump(metadata, out_file, indent=4, ensure_ascii=False)
            print(f"Metadati salvati in: {output_file}")


def process_external_resources(external_resources, output_path):
    """
    Processa i video esterni elencati nei metadati delle lezioni.
    """
    for resource in external_resources:
        if resource.get("type") == "video":
            video_url = resource.get("url")
            print(f"Processando video esterno: {video_url}")
            metadata = extract_video_metadata_youtube(video_url)

            if metadata:
                # Salvataggio dei metadati
                output_file = output_path / f"{metadata['title'].replace(' ', '_')}_metadata.json"
                with open(output_file, 'w', encoding='utf-8') as out_file:
                    json.dump(metadata, out_file, indent=4, ensure_ascii=False)
                print(f"Metadati salvati in: {output_file}")


def process_lesson_metadata(lesson_metadata_path, output_path):
    """
    Process lesson metadata to handle external resources like videos.
    """
    try:
        with open(lesson_metadata_path, 'r', encoding='utf-8') as file:
            lesson_metadata = json.load(file)

        external_resources = lesson_metadata.get("external_resources", [])
        process_external_resources(external_resources, output_path)
    except Exception as e:
        print(f"Error processing lesson metadata {lesson_metadata_path}: {e}")


if __name__ == "__main__":
    # Processa i file locali
    print("Processamento dei file locali...")
    process_local_files(LOCAL_PATH, OUTPUT_PATH)

    # Processa risorse esterne dai metadati delle lezioni
    print("Processamento delle risorse esterne dai metadati delle lezioni...")
    for metadata_file in METADATA_PATH.glob("lesson*_metadata.json"):
        print(f"Processing lesson metadata: {metadata_file}")
        process_lesson_metadata(metadata_file, OUTPUT_PATH)
