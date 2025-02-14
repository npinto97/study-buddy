import os
from pathlib import Path
from yt_dlp import YoutubeDL
from moviepy.video.io.VideoFileClip import VideoFileClip
from study_buddy.data_processing.utils import transcribe
from study_buddy.config import EXTERNAL_DATA_DIR


def extract_metadata_local(file_path):        
    """
    Extracts metadata from a local video file.

    Parameters:
        file_path (str): The path to the local video file.

        Returns:
        dict: A dictionary containing metadata of the video file, including:
            - filename (str): The name of the video file.
            - filepath (str): The full path to the video file.
            - duration (float): The duration of the video in seconds (if applicable).
            - resolution (tuple): The resolution of the video as (width, height) (if applicable).
            - fps (float): The frames per second of the video (if applicable).
            - type (str): The type of the file, which is "video" for recognized video formats.
        
        Note:
        - Recognized video formats are .mp4, .mkv, and .webm.
        - If the file format is not recognized as a video, a message will be printed.
        - In case of an error during metadata extraction, an error message will be printed.
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
    Extract metadata from a YouTube video URL using youtube-dl.
    Args:
        video_url (str): The URL of the YouTube video.
    Returns:
        dict: A dictionary containing the video's metadata, including:
            - title (str): The title of the video.
            - description (str): The description of the video.
            - duration (int): The duration of the video in seconds.
            - upload_date (str): The upload date of the video in YYYYMMDD format.
            - channel (str): The name of the channel that uploaded the video.
            - tags (list): A list of tags associated with the video.
            - url (str): The URL of the video.
        None: If an error occurs during metadata extraction.
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


def transcribe_video(file_path, output_text_path):
    """
    Transcribes the audio from a video file and saves the transcription to a text file.
    Args:
        file_path (str): The path to the video file to be transcribed.
        output_text_path (str): The path where the transcribed text will be saved.
    Raises:
        Exception: If an error occurs during the transcription process, it will be caught and printed.
    Notes:
        A temporary audio file will be created during the process and deleted afterwards.
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
    Downloads the audio from a YouTube video, transcribes it, and saves the transcription to a file.
    Args:
        url_link (str): The URL of the YouTube video to transcribe.
        output_text_path (str): The file path where the transcription text will be saved.
    Raises:
        Exception: If there is an error during the audio download or transcription process.
    Notes:
        - This function uses yt-dlp to download the audio from the YouTube video.
        - The audio is temporarily saved as an MP3 file in the EXTERNAL_DATA_DIR directory.
        - The audio file is deleted after transcription is completed.
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
            'outtmpl': str(audio_path).replace('.mp3', ''),
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