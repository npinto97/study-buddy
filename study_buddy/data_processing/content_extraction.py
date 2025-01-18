import os
import json
from pathlib import Path
from study_buddy.data_processing.video_handler import transcribe_video
from study_buddy.data_processing.external_resource_handler import extract_external_video_from_lesson, extract_external_text_from_lesson, extract_external_repo_from_lesson, extract_external_webinfo_from_lesson
from study_buddy.data_processing.utils import extract_lesson_folders, create_content_metadata, extract_lesson_list
from study_buddy.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR

folder_path = Path("C:\\Users\\e1204\\Downloads\\SIIA\\SIIA")
lesson_folders = extract_lesson_folders(folder_path)
temp_output_path = os.path.join(EXTERNAL_DATA_DIR, "temp.txt")
# video_folder_suffix = "\\multimedia\\videos"

# input: path della cartella della lezione
def extract_videos_from_lesson(lesson_path):
    
    metadata_path = lesson_path / "metadata.json"
    
    try:
        with metadata_path.open('r', encoding='utf-8') as metadata_file:
            metadata = json.load(metadata_file)
    except json.JSONDecodeError:
        print(f"Error reading {metadata_path}. Skip ...")

    videos = metadata.get("multimedia", {}).get("videos", [])
    if not videos:
        print(f"No video found in {lesson_path}.")
    else:
        video_folder_path = lesson_path / "multimedia" / "videos"
        for video in videos:
            video_path = video_folder_path / video
            if not video_path.exists():
                print(f"Video {video} not found in {video_folder_path}. Skip ...")
            else:
                transcribe_video(video_path, temp_output_path)
                combined_content = create_content_metadata(lesson_path, temp_output_path)

                print(f"\n{combined_content}")

                output_json_path = PROCESSED_DATA_DIR / ((video.split(".")[0]) + ".json")
                with open(output_json_path, 'w', encoding='utf-8') as output_file:
                    json.dump(combined_content, output_file, indent=4, ensure_ascii=False)
                

def extract_videos_from_course(course_path):

    lesson_list = extract_lesson_list(course_path)

    for lesson in lesson_list:
        lesson_path = course_path / lesson
        extract_videos_from_lesson(lesson_path)
    

def extract_external_resource_from_lesson(lesson_path):
    count = 0
    metadata_path = lesson_path / "metadata.json"

    try:
        with metadata_path.open('r', encoding='utf-8') as metadata_file:
            metadata = json.load(metadata_file)
    except json.JSONDecodeError:
        print(f"Error reading {metadata_path}. Skip ...")
        return  # Uscita in caso di errore nel file JSON

    # Estrarre il contenuto del campo "external_resources"
    external_resources = metadata.get("external_resources", [])

    for resource in external_resources:
        count += 1
        resource_url = resource.get("url")
        resource_type = resource.get("type")

        print(f"\nResource url: {resource_url}\nResource type: {resource_type}\n")

        if resource_type == "video":
            combined_content = extract_external_video_from_lesson(resource_url, lesson_path)
        elif resource_type in ["article", "articles", "model", "blog", "paper", "announcement"]:
            combined_content = extract_external_text_from_lesson(resource_url, lesson_path)
        elif resource_type == "repo":
            combined_content = extract_external_repo_from_lesson(resource_url, lesson_path)
        elif resource_type in ["dataset", "website"]:
            combined_content = extract_external_webinfo_from_lesson(resource_url, lesson_path)
        else:
            print(f"Resource type {resource_type} not recognized. Skip ...")
        
        if combined_content is not None:
            output_json_path = PROCESSED_DATA_DIR / (resource_type + str(count) + ".json")
            with open(output_json_path, 'w', encoding='utf-8') as output_file:
                json.dump(combined_content, output_file, indent=4, ensure_ascii=False)
            output_file.close()
        else:
            print(f"Failed to process resource {count}. Skipping...")



def extract_external_resource_from_course(course_path):

    # ottieni la lista contenente le lezioni del corso (usa la funzione creata)

    # per ogni lezione, esegui extract_external_resource_from_lesson

    pass



# def extract_content_from_videos(lesson_folders):
    
#     json_files = []

#     for folder in lesson_folders:
#         folder_path = Path(folder)
#         metadata_path = folder_path / "metadata.json"

#         if not metadata_path.exists():
#             print(f"metadata.json not found in {folder_path}. Skip ...")
#             continue

#         try:
#             with metadata_path.open('r', encoding='utf-8') as metadata_file:
#                 metadata = json.load(metadata_file)
#         except json.JSONDecodeError:
#             print(f"Error reading {metadata_path}. Skip ...")
#             continue

#         videos = metadata.get("multimedia", {}).get("videos", [])
#         if not videos:
#             print(f"No video found in {folder_path}.")
#             continue

#         video_folder_path = folder_path / video_folder_suffix.strip("\\").replace("\\", os.sep)

#         for video in videos:
#             video_path = video_folder_path / video

#             if not video_path.exists():
#                 print(f"Video {video} not found in {video_folder_path}. Skip ...")
#                 continue

#             try:
#                 transcribe_video(video_path, temp_output_path)
#             except Exception as e:
#                 print(f"Error during transcription of {video_path}: {e}")
#                 continue

#             try:
#                 combined_json = create_content_metadata(folder_path, temp_output_path)
#                 json_files.append(combined_json)
#             except Exception as e:
#                 print(f"Error during JSON creation for {video}: {e}")
#                 continue
    
#     return json_files
