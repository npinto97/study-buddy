import os
import json
import requests
from bs4 import BeautifulSoup
from study_buddy.config import EXTERNAL_DATA_DIR
from study_buddy.data_processing.utils import create_content_metadata
from study_buddy.data_processing.video_handler import transcribe_youtube_video


# Function for extracting metadata from a URL
def extract_metadata(url, output_path):
    """
    Extracts metadata from a web page starting from the URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        metadata = {}

        title_tag = soup.find('title')
        metadata['title'] = title_tag.text if title_tag else None

        meta_tags = soup.find_all('meta')

        for tag in meta_tags:
            if tag.get('name'):
                metadata[tag['name']] = tag.get('content', None)
            elif tag.get('property'):
                metadata[tag['property']] = tag.get('content', None)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    except Exception as e:
        print(f"Error during metadata extraction: {e}")


# Function for extracting textual information in .txt format from content in HTML
def extract_text_content(url, output_path):
    """
    Extracts content in paragraphs <p> corresponding to a URL path  
    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        article_body = soup.find_all('p')

        article_text = "\n".join([paragraph.get_text()
                                for paragraph in article_body])

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(article_text)

        print(f"Article correctly extracted in file {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")

    except Exception as e:
        print(f"error: {e}")


# Function for extracting a .txt file from a repository's README.md
def extract_readme_from_repo(repo_url, output_path):
    """
    Extract README.md file content from a GitHub repository.
    """
    try:
        if not repo_url.endswith('/'):
            repo_url += '/'
        raw_readme_url = repo_url.replace("github.com", "raw.githubusercontent.com") + "main/README.md"

        response = requests.get(raw_readme_url)
        response.raise_for_status()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"Information from repo {repo_url} correctly extracted")

    except requests.exceptions.RequestException as e:
        print(f"Error during extracting repo content: {e}")

    except Exception as e:
        print(f"Error: {e}")


# Function for extracting information from metadata.json for content of type 'dataset' and 'website'
# folder_path refers to the course path
def extract_metadata_json(url_path, output_path, folder_path):
    """
    Extracts external resource information from metadata file
    """
    try:

        lesson_folders = [
            folder for folder in os.listdir(folder_path)
            if folder.startswith("lesson") and os.path.isdir(os.path.join(folder_path, folder))
        ]

        for subfolder in lesson_folders:
            subfolder_path = os.path.join(folder_path, subfolder)
            metadata_path = os.path.join(subfolder_path, "metadata.json")

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as file:
                    metadata = json.load(file)

                    for resource in metadata.get("external_resources", []):
                        if resource.get("url") == url_path:

                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(f"Title: {resource.get('title')}\n")
                                f.write(f"Description: {resource.get('description')}\n")
                            print(f"Information saved in {output_path}")
                            return

        raise ValueError(f"There are no external resources corresponding to {url_path} in this course")

    except ValueError as e:
        print(e)

    except Exception as e:
        print(f"Error during the operation: {e}")


# Writes the .json file representing metadata of the content of a lesson
def extract_external_video_from_lesson(video_url, lesson_path):
    """
    Returns JSON variable containing video transcription and lesson metadata.
    """
    temp_output_path = os.path.join(EXTERNAL_DATA_DIR, "temp_output.txt")
    transcribe_youtube_video(video_url, temp_output_path)

    combined_content = create_content_metadata(lesson_path, temp_output_path)

    return combined_content


def extract_external_text_from_lesson(url_path, lesson_path):
    """
    Returns JSON variable containing text content and lesson metadata.
    """
    temp_output_path = os.path.join(EXTERNAL_DATA_DIR, "temp_output.txt")
    extract_text_content(url_path, temp_output_path)

    combined_content = create_content_metadata(lesson_path, temp_output_path)

    return combined_content


def extract_external_repo_from_lesson(url_path, lesson_path):
    """
    Returns JSON variable containing repo content and lesson metadata.
    """
    temp_output_path = os.path.join(EXTERNAL_DATA_DIR, "temp_output.txt")
    extract_readme_from_repo(url_path, temp_output_path)

    combined_content = create_content_metadata(lesson_path, temp_output_path)

    return combined_content


def extract_external_webinfo_from_lesson(url_path, lesson_path):
    """
    Returns JSON variable containing web content and lesson metadata.
    """
    # from lesson_path obtain course_path
    course_path = os.path.dirname(lesson_path)

    temp_output_path = os.path.join(EXTERNAL_DATA_DIR, "temp_output.txt")
    extract_metadata_json(url_path, temp_output_path, course_path)

    combined_content = create_content_metadata(lesson_path, temp_output_path)

    return combined_content
