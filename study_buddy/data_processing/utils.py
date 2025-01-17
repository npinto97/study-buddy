import whisper

whisper_model = whisper.load_model("base")


def transcribe(file_path, output_text_path):
    """
    Creates a .txt file with transcription of an audio file.
    """
    result = whisper_model.transcribe(file_path, fp16=False)
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
