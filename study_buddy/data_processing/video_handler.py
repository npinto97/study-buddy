import os
from pathlib import Path
from yt_dlp import YoutubeDL
from moviepy.video.io.VideoFileClip import VideoFileClip
from study_buddy.data_processing.utils import transcribe
from study_buddy.config import EXTERNAL_DATA_DIR


def extract_metadata_local(file_path):
    """
    Extracts metadata from a local video file.
    """
    file_path = Path(file_path)
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
        else:
            print("File format not recognized as video")
    except Exception as e:
        print(f"Error in metadata extraction of {file_path}: {e}")
    return metadata


def extract_metadata_youtube(video_url):
    """
    Extracts metadata from a youtube video.
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


# TODO: audio_path must be modified
def transcribe_video(file_path, output_text_path):
    """
    Creates a .txt file with transcription of a video file.
    """
    try:
        clip = VideoFileClip(file_path)
        audio_path = os.path.join(EXTERNAL_DATA_DIR, "temp_audio.mp3")
        clip.audio.write_audiofile(audio_path)

        transcribe(audio_path, output_text_path)

    except Exception as e:
        print(f"Error during {file_path} transcription: {e}")

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def transcribe_youtube_video(url_link, output_text_path):
    """
    Creates a .txt file with transcription of a YouTube video.
    """
    audio_dir = EXTERNAL_DATA_DIR
    audio_filename = "temp_audio.mp3"
    audio_path = audio_dir / audio_filename

    if not audio_dir.exists():
        audio_dir.mkdir(parents=True)

    try:
        # Download audio using yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(audio_path),
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url_link])

        if not audio_path.exists():
            print(f"Error: audio file not created in {audio_path}")
            return

        # Transcribe the audio
        transcribe(str(audio_path), output_text_path)

    except Exception as e:
        print(f"Error during {audio_path} transcription: {e}")

    finally:
        if audio_path.exists():
            os.remove(audio_path)
