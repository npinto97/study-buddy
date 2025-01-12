import os
import json
from pathlib import Path
from mutagen import File as MutagenFile
from whisper import load_model

# Load the Whisper model globally to reuse for transcriptions
whisper_model = load_model("base")


def extract_audio_metadata(file_path):
    """
    Extract metadata from an audio file.
    """
    metadata = {"filename": file_path.name, "filepath": str(file_path)}
    try:
        audio = MutagenFile(file_path)
        if audio is not None:
            metadata.update({
                "duration": audio.info.length if
                hasattr(audio.info, "length") else None,
                "bitrate": audio.info.bitrate if
                hasattr(audio.info, "bitrate") else None,
                "type": "audio"
            })
    except Exception as e:
        print(f"Error extracting metadata for {file_path}: {e}")
    return metadata


def transcribe_audio(file_path):
    """
    Transcribe audio content using Whisper.
    """
    try:
        print(f"Transcribing: {file_path}")
        result = whisper_model.transcribe(str(file_path))
        return result.get("text", "")
    except Exception as e:
        print(f"Error transcribing audio for {file_path}: {e}")
        return None


def process_audio_files(input_path, output_path):
    """
    Process all audio files in the specified directory.
    """
    for root, _, files in os.walk(input_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() not in ['.mp3', '.wav', '.ogg', '.flac']:
                continue

            print(f"Processing audio file: {file_path}")
            metadata = extract_audio_metadata(file_path)

            # Transcription
            if metadata.get("type") == "audio":
                transcript = transcribe_audio(file_path)
                metadata['transcription'] = transcript

            # Save metadata to JSON
            output_file = output_path / f"{file_path.stem}_metadata.json"
            with open(output_file, 'w', encoding='utf-8') as out_file:
                json.dump(metadata, out_file, indent=4, ensure_ascii=False)
            print(f"Metadata saved to: {output_file}")


if __name__ == "__main__":
    LOCAL_AUDIO_PATH = Path("data/multimedia/audio/")
    OUTPUT_PATH = Path("output/audio_metadata/")

    # Ensure the output directory exists
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    print("Starting audio file processing...")
    process_audio_files(LOCAL_AUDIO_PATH, OUTPUT_PATH)
    print("Audio file processing completed.")
