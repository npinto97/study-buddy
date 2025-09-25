import json
import os
from pathlib import Path

from study_buddy.config import METADATA_DIR, PARSED_COURSES_DATA_FILE, RAW_DATA_DIR, logger


def parse_course_metadata(course_metadata_path: Path, course: Path):
    with open(course_metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    course_name = data.get("course_name", "Unknown Course")
    course_description = data.get("description", "No description available")
    extracted_data = []

    # Extracting syllabus
    if "syllabus" in data:
        extracted_data.append({
            "path": os.path.join(RAW_DATA_DIR, data["syllabus"]),
            "type": "syllabus",
            "course_name": course_name,
            "course_description": course_description
        })

    # Extracting books
    for book in data.get("books", []):
        extracted_data.append({
            "path": os.path.join(RAW_DATA_DIR, book["filename"]),
            "type": "book",
            "course_name": course_name,
            "course_description": course_description,
            "title": book.get("title"),
            "author": book.get("author"),
            "year": book.get("year"),
            "isbn": book.get("isbn")
        })

    # Extracting notes
    for note in data.get("notes", []):
        extracted_data.append({
            "path": os.path.join(RAW_DATA_DIR, note),
            "type": "note",
            "course_name": course_name,
            "course_description": course_description
        })

    # Extracting lessons and their components
    for lesson in data.get("lessons", []):
        lesson_number = lesson.get("lesson_number")
        lesson_metadata_path = os.path.join(course, lesson.get("metadata"))
        lesson_title = "No title"

        if os.path.exists(lesson_metadata_path):
            with open(lesson_metadata_path, "r", encoding="utf-8") as f:
                lesson_data = json.load(f)
            lesson_title = lesson_data.get("title", "No title")
            lesson_keywords = lesson_data.get("keywords", [])

            # Extracting slides
            for slide in lesson_data.get("slides", []):
                extracted_data.append({
                    "path": os.path.join(RAW_DATA_DIR, slide),
                    "type": "slide",
                    "course_name": course_name,
                    "course_description": course_description,
                    "lesson_number": lesson_number,
                    "lesson_title": lesson_title,
                    "keywords": lesson_keywords
                })

            # Extracting lesson notes
            for note in lesson_data.get("notes", []):
                extracted_data.append({
                    "path": os.path.join(RAW_DATA_DIR, note),
                    "type": "lesson_note",
                    "course_name": course_name,
                    "course_description": course_description,
                    "lesson_number": lesson_number,
                    "lesson_title": lesson_title,
                    "keywords": lesson_keywords
                })

            # Extracting references
            for ref in lesson_data.get("references", []):
                extracted_data.append({
                    "path": os.path.join(RAW_DATA_DIR, ref["filename"]),
                    "type": "reference",
                    "course_name": course_name,
                    "course_description": course_description,
                    "lesson_number": lesson_number,
                    "lesson_title": lesson_title,
                    "keywords": lesson_keywords,
                    "title": ref.get("title"),
                    "description": ref.get("description")
                })

            # Extracting supplementary materials
            for material in lesson_data.get("supplementary_materials", []):
                extracted_data.append({
                    "path": os.path.join(RAW_DATA_DIR, material["filename"]),
                    "type": "supplementary_material",
                    "course_name": course_name,
                    "course_description": course_description,
                    "lesson_number": lesson_number,
                    "lesson_title": lesson_title,
                    "keywords": lesson_keywords,
                    "title": material.get("title"),
                    "description": material.get("description")
                })

            # Extracting exercises
            for exercise in lesson_data.get("exercises", []):
                # The exercise can have multiple files
                filenames = exercise.get("filename", [])
                if isinstance(filenames, str):
                    filenames = [filenames]
                
                for filename in filenames:
                    extracted_data.append({
                        "path": os.path.join(RAW_DATA_DIR, filename),
                        "type": "exercise",
                        "course_name": course_name,
                        "course_description": course_description,
                        "lesson_number": lesson_number,
                        "lesson_title": lesson_title,
                        "keywords": lesson_keywords,
                        "title": exercise.get("title"),
                        "description": exercise.get("description")
                    })

            # Extracting multimedia content
            multimedia = lesson_data.get("multimedia", {})
            
            # Video
            for video in multimedia.get("videos", []):
                extracted_data.append({
                    "path": os.path.join(RAW_DATA_DIR, video),
                    "type": "video",
                    "course_name": course_name,
                    "course_description": course_description,
                    "lesson_number": lesson_number,
                    "lesson_title": lesson_title,
                    "keywords": lesson_keywords
                })
            
            # Images
            for image in multimedia.get("images", []):
                extracted_data.append({
                    "path": os.path.join(RAW_DATA_DIR, image),
                    "type": "image",
                    "course_name": course_name,
                    "course_description": course_description,
                    "lesson_number": lesson_number,
                    "lesson_title": lesson_title,
                    "keywords": lesson_keywords
                })
            
            # Audio
            for audio in multimedia.get("audio", []):
                extracted_data.append({
                    "path": os.path.join(RAW_DATA_DIR, audio),
                    "type": "audio",
                    "course_name": course_name,
                    "course_description": course_description,
                    "lesson_number": lesson_number,
                    "lesson_title": lesson_title,
                    "keywords": lesson_keywords
                })

            # Extracting external resources
            for resource in lesson_data.get("external_resources", []):
                extracted_data.append({
                    "url": resource["url"],
                    "type": "external_resource",
                    "course_name": course_name,
                    "course_description": course_description,
                    "lesson_number": lesson_number,
                    "lesson_title": lesson_title,
                    "keywords": lesson_keywords,
                    "resource_type": resource.get("type"),
                    "title": resource.get("title"),
                    "description": resource.get("description")
                })
        else:
            logger.warning(f"Lesson file not found: {lesson_metadata_path}")

    return extracted_data


def parse_all_courses_metadata():
    """Scans all courses in the METADATA_DIR folder and saves the data to a JSON file."""
    all_parsed_data = []

    metadata_dir = Path(METADATA_DIR)
    if not metadata_dir.exists():
        logger.error(f"The folder METADATA_DIR '{metadata_dir}' doesn't exists.")
        return

    for course in metadata_dir.iterdir():
        course_metadata_path = course / "course_metadata.json"
        if course.is_dir() and course_metadata_path.exists():
            logger.info(f"Parsing metadata for course: {course.name}")
            all_parsed_data.extend(parse_course_metadata(course_metadata_path, course))
        else:
            logger.warning(f"No metadata found for course: {course.name}")

    os.makedirs(os.path.dirname(PARSED_COURSES_DATA_FILE), exist_ok=True)

    with open(PARSED_COURSES_DATA_FILE, "w", encoding="utf-8") as out_file:
        json.dump(all_parsed_data, out_file, indent=4, ensure_ascii=False)

    logger.info(f"Parsing completed! Data saved in '{PARSED_COURSES_DATA_FILE}'.")
    logger.info(f"Total items processed: {len(all_parsed_data)}")


if __name__ == "__main__":
    parse_all_courses_metadata()