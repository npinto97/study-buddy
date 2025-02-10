import os
import re
import json
from pathlib import Path
from study_buddy.data_processing.text_handler import TextExtractor
from study_buddy.data_processing.video_handler import transcribe_video
from study_buddy.data_processing.external_resource_handler import extract_external_video_from_lesson, extract_external_text_from_lesson, extract_external_repo_from_lesson, extract_external_webinfo_from_lesson
from study_buddy.data_processing.utils import extract_lesson_folders, create_content_metadata, extract_lesson_list
from study_buddy.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR

folder_path = Path("C:\\Users\\e1204\\Downloads\\SIIA\\SIIA")
lesson_folders = extract_lesson_folders(folder_path)
temp_output_path = os.path.join(EXTERNAL_DATA_DIR, "temp.txt")


def extract_videos_from_lesson(lesson_path):
    """
    Extracts video metadata from a lesson directory, transcribes the videos, and saves the combined content metadata.
    Args:
        lesson_path (Path): The path to the lesson directory containing the metadata and multimedia files.
    Raises:
        json.JSONDecodeError: If there is an error reading the metadata JSON file.
    Notes:
        - The function expects the lesson directory to contain a "metadata.json" file with video information.
        - The function transcribes each video found in the metadata and creates combined content metadata.
        - The combined content metadata is saved as a JSON file in the processed data directory.
    """
    
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

                output_dir = PROCESSED_DATA_DIR / lesson_path.name / "videos"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                output_json_path = PROCESSED_DATA_DIR / output_dir / (f"{video.split('.')[0]}.json")
                with open(output_json_path, 'w', encoding='utf-8') as output_file:
                    json.dump(combined_content, output_file, indent=4, ensure_ascii=False)
    

def extract_external_resource_from_lesson(lesson_path):
    """
    Extracts external resources from a lesson's metadata file and processes them based on their type.
    Args:
        lesson_path (Path): The path to the lesson directory containing the metadata.json file.
    Returns:
        None
    The function reads the metadata.json file from the given lesson path, extracts external resources,
    and processes each resource based on its type. Supported resource types include:
    - "video": Processes the resource as a video.
    - "article", "articles", "model", "blog", "paper", "announcement": Processes the resource as text.
    - "repository": Processes the resource as a repository.
    - "dataset", "website": Processes the resource as web information.
    Processed resources are saved as JSON files in the "external_resources" directory within the lesson path.
    If a resource type is not recognized or processing fails, the function skips that resource.
    """
    count = 0
    metadata_path = lesson_path / "metadata.json"

    try:
        with metadata_path.open('r', encoding='utf-8') as metadata_file:
            metadata = json.load(metadata_file)
    except json.JSONDecodeError:
        print(f"Error reading {metadata_path}. Skip ...")
        return  

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
        elif resource_type == "repository":
            combined_content = extract_external_repo_from_lesson(resource_url, lesson_path)
        elif resource_type in ["dataset", "website"]:
            combined_content = extract_external_webinfo_from_lesson(resource_url, lesson_path)
        else:
            print(f"Resource type {resource_type} not recognized. Skip ...")
        
        if combined_content is not None:
            output_dir = PROCESSED_DATA_DIR / lesson_path.name / "external_resources"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_json_path = output_dir / (f"resource_{count}.json")
            with open(output_json_path, 'w', encoding='utf-8') as output_file:
                json.dump(combined_content, output_file, indent=4, ensure_ascii=False)
            output_file.close()
        else:
            print(f"Failed to process resource {count}. Skipping...")


def extract_references_from_lesson(lesson_path: Path, extractor: TextExtractor):
    """
    Extracts references from a lesson's metadata and processes them.
    This function reads the metadata file from the given lesson path to extract references.
    It then processes each reference by extracting its text content using the provided
    TextExtractor, and saves the extracted content along with additional metadata to a JSON file.
    Args:
        lesson_path (Path): The path to the lesson directory containing the metadata and references.
        extractor (TextExtractor): An instance of TextExtractor used to extract text from reference files.
    Raises:
        FileNotFoundError: If the metadata file is not found in the lesson path.
        Exception: If there is an error during the extraction or processing of references.
    Returns:
        None
    """
    
    try:
        metadata_path = lesson_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found in {lesson_path}.")
        
        with metadata_path.open('r', encoding='utf-8') as metadata_file:
            metadata = json.load(metadata_file)

        references = metadata.get("references", [])
        if not references:
            print(f"No references found in {lesson_path}.")
            return
        
        output_dir = PROCESSED_DATA_DIR / lesson_path.name / "references"
        output_dir.mkdir(parents=True, exist_ok=True)

        for reference in references:
            ref_title = re.sub(r'[^\w]', '', reference.get("title", "unknown").replace(" ", "_"))
            ref_filename = reference.get("filename")

            if not ref_filename:
                print("No filename found in reference. Skip ...")
                continue

            ref_file_path = lesson_path / "references" / ref_filename
            if not ref_file_path.exists():
                print(f"File {ref_filename} not found in {lesson_path}. Skip ...")
                continue

            try:
                print(f"Extracting text from {ref_file_path} ...")
                ref_content = extractor.extract_text(file_path=Path(ref_file_path))

                # create a txt file containing ref content
                ref_output_path = output_dir / (ref_title + ".txt")
                with ref_output_path.open('w', encoding='utf-8') as ref_output_file:
                    ref_output_file.write(ref_content)
            except Exception as e:
                print(f"Error extracting text from {ref_file_path}: {e}")
                continue

            combined_data = create_content_metadata(lesson_path, ref_output_path)
            os.remove(ref_output_path)

            output_file_path = output_dir / (ref_title + ".json")
            with output_file_path.open('w', encoding='utf-8') as output_file:
                json.dump(combined_data, output_file, indent=4, ensure_ascii=False)

            print(f"Content correctly saved to {output_file_path}.")

    except Exception as e:
        print(f"Error processing references in {lesson_path}: {e}")


def extract_slides_from_lesson(lesson_path: Path, extractor: TextExtractor):
    """
    Extracts text from slides in a lesson and saves the content to JSON files.
    Args:
        lesson_path (Path): The path to the lesson directory containing the metadata and slides.
        extractor (TextExtractor): An instance of TextExtractor used to extract text from slide files.
    Raises:
        FileNotFoundError: If the metadata file is not found in the lesson directory.
        Exception: If there is an error processing the slides or extracting text.
    The function performs the following steps:
        1. Reads the metadata file from the lesson directory.
        2. Extracts the list of slides from the metadata.
        3. Creates an output directory for the processed slides.
        4. Iterates through each slide, extracts text using the provided extractor, and saves the content to a JSON file.
        5. Handles errors gracefully and prints appropriate error messages.
    """

    try:
        metadata_path = lesson_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found in {lesson_path}.")
        
        with metadata_path.open('r', encoding='utf-8') as metadata_file:
            metadata = json.load(metadata_file)

        slides = metadata.get("multimedia", {}).get("slides", [])
        if not slides:
            print(f"No slides found in {lesson_path}.")
            return
    
        output_dir = PROCESSED_DATA_DIR / lesson_path.name / "slides"
        output_dir.mkdir(parents=True, exist_ok=True)

        for slide in slides:
            slide_title = slide.get("title", "unknown").replace(" ", "_")
            slide_filename = slide.get("filename")

            if not slide_filename:
                print("No filename found in slide. Skip ...")
                continue

            slide_file_path = lesson_path / "slides" / slide_filename
            if not slide_file_path.exists():
                print(f"File {slide_filename} not found in {lesson_path}. Skip ...")
                continue

            try:
                print(f"Extracting text from {slide_file_path} ...")
                slide_content = extractor.extract_text(file_path=Path(slide_file_path))

                # create a txt file containing slide content
                slide_output_path = output_dir / (slide_title + ".txt")
                with slide_output_path.open('w', encoding='utf-8') as slide_output_file:
                    slide_output_file.write(slide_content)
            except Exception as e:
                print(f"Error extracting text from {slide_file_path}: {e}")
                continue

            combined_data = create_content_metadata(lesson_path, slide_output_path)
            os.remove(slide_output_path)

            output_file_path = output_dir / (slide_title + ".json")
            with output_file_path.open('w', encoding='utf-8') as output_file:
                json.dump(combined_data, output_file, indent=4, ensure_ascii=False)

            print(f"Content correctly saved to {output_file_path}.")
    
    except Exception as e:
        print(f"Error processing slides in {lesson_path}: {e}")


def extract_supplementary_materials_from_lesson(lesson_path: Path, extractor: TextExtractor):
    """
    Extracts supplementary materials from a lesson directory and processes them.
    This function reads the metadata file from the given lesson directory to find supplementary materials.
    It then extracts text from each supplementary material file using the provided text extractor,
    saves the extracted content to a text file, and then creates a JSON file with combined metadata and content.
    Args:
        lesson_path (Path): The path to the lesson directory containing the metadata and supplementary materials.
        extractor (TextExtractor): An instance of a text extractor used to extract text from supplementary material files.
    Raises:
        FileNotFoundError: If the metadata file is not found in the lesson directory.
        Exception: If there is an error during the extraction or processing of supplementary materials.
    Returns:
        None
    """

    try:
        metadata_path = lesson_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found in {lesson_path}.")
        
        with metadata_path.open('r', encoding='utf-8') as metadata_file:
            metadata = json.load(metadata_file)

        supp_materials = metadata.get("supplementary_materials", [])
        if not supp_materials:
            print(f"No supplementary materials found in {lesson_path}.")
            return

        output_dir = PROCESSED_DATA_DIR / lesson_path.name / "supplementary_materials" 
        output_dir.mkdir(parents=True, exist_ok=True)

        for supp_material in supp_materials:
            supp_title = re.sub(r'[^\w]', '', supp_material.get("title", "unknown").replace(" ", "_"))
            supp_filename = supp_material.get("filename")

            if not supp_filename:
                print("No filename found in supplementary material. Skip ...")
                continue

            supp_file_path = lesson_path / "supplementary_materials" / supp_filename
            if not supp_file_path.exists():
                print(f"File {supp_filename} not found in {lesson_path}. Skip ...")
                continue

            try:
                print(f"Extracting text from {supp_file_path} ...")
                supp_content = extractor.extract_text(file_path=Path(supp_file_path))

                # create a txt file containing supplementary material content
                supp_output_path = output_dir / (supp_title + ".txt")
                with supp_output_path.open('w', encoding='utf-8') as supp_output_file:
                    supp_output_file.write(supp_content)
            except Exception as e:
                print(f"Error extracting text from {supp_file_path}: {e}")
                continue

            combined_data = create_content_metadata(lesson_path, supp_output_path)
            os.remove(supp_output_path)

            output_file_path = output_dir / (supp_title + ".json")
            with output_file_path.open('w', encoding='utf-8') as output_file:
                json.dump(combined_data, output_file, indent=4, ensure_ascii=False)

            print(f"Content correctly saved to {output_file_path}.")

    except Exception as e:
        print(f"Error processing supplementary materials in {lesson_path}: {e}")


def extract_content_from_course(course_path):
    """
    Extracts content from all lessons in a given course.
    Args:
        course_path (str): The path to the course directory.
    Returns:
        None
    """
    lesson_list = extract_lesson_list(course_path)
    for lesson in lesson_list:
        lesson_path = Path(lesson)
        lesson_path = course_path / lesson_path
        extractor = TextExtractor(data_dir=lesson_path, output_dir=PROCESSED_DATA_DIR, metadata_dir=lesson_path)
        extract_references_from_lesson(lesson_path, extractor)
        extract_slides_from_lesson(lesson_path, extractor)
        extract_supplementary_materials_from_lesson(lesson_path, extractor)
        extract_external_resource_from_lesson(lesson_path)
        extract_videos_from_lesson(lesson_path)
