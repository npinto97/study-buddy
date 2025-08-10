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

    # Estrarre syllabus
    if "syllabus" in data:
        extracted_data.append({
            "path": os.path.join(RAW_DATA_DIR, data["syllabus"]),
            "type": "syllabus",
            "course_name": course_name,
            "course_description": course_description
        })

    # Estrarre libri
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

    # Estrarre note
    for note in data.get("notes", []):
        extracted_data.append({
            "path": os.path.join(RAW_DATA_DIR, note),
            "type": "note",
            "course_name": course_name,
            "course_description": course_description
        })

    # Estrarre informazioni dalle lezioni
    for lesson in data.get("lessons", []):
        lesson_number = lesson.get("lesson_number")
        lesson_metadata_path = os.path.join(course, lesson.get("metadata"))
        lesson_title = "No title"

        if os.path.exists(lesson_metadata_path):
            with open(lesson_metadata_path, "r", encoding="utf-8") as f:
                lesson_data = json.load(f)
            lesson_title = lesson_data.get("title", "No title")

            for slide in lesson_data.get("slides", []):
                extracted_data.append({
                    "path": os.path.join(RAW_DATA_DIR, slide),
                    "type": "slide",
                    "course_name": course_name,
                    "course_description": course_description,
                    "lesson_number": lesson_number,
                    "lesson_title": lesson_title
                })

            for ref in lesson_data.get("references", []):
                extracted_data.append({
                    "path": os.path.join(RAW_DATA_DIR, ref["filename"]),
                    "type": "reference",
                    "course_name": course_name,
                    "course_description": course_description,
                    "lesson_number": lesson_number,
                    "lesson_title": lesson_title
                })

            for material in lesson_data.get("supplementary_materials", []):
                extracted_data.append({
                    "path": os.path.join(RAW_DATA_DIR, material["filename"]),
                    "type": "supplementary_material",
                    "course_name": course_name,
                    "course_description": course_description,
                    "lesson_number": lesson_number,
                    "lesson_title": lesson_title
                })

            for video in lesson_data.get("multimedia", {}).get("videos", []):
                extracted_data.append({
                    "path": os.path.join(RAW_DATA_DIR, video),
                    "type": "video",
                    "course_name": course_name,
                    "course_description": course_description,
                    "lesson_number": lesson_number,
                    "lesson_title": lesson_title
                })

            for resource in lesson_data.get("external_resources", []):
                extracted_data.append({
                    "url": resource["url"],
                    "type": "external_resource",
                    "course_name": course_name,
                    "course_description": course_description,
                    "lesson_number": lesson_number,
                    "lesson_title": lesson_title
                })
        else:
            print(f"Errore: File della lezione non trovato -> {lesson_metadata_path}")

    return extracted_data


def parse_all_courses_metadata():
    """Scansiona tutti i corsi nella cartella METADATA_DIR e salva i dati in un file JSON."""
    all_parsed_data = []

    metadata_dir = Path(METADATA_DIR)
    if not metadata_dir.exists():
        logger.error(f"La cartella METADATA_DIR '{metadata_dir}' non esiste.")
        return

    for course in metadata_dir.iterdir():
        course_metadata_path = course / "course_metadata.json"
        if course.is_dir() and course_metadata_path.exists():
            logger.info(f"Parsing metadati per il corso: {course.name}")
            all_parsed_data.extend(parse_course_metadata(course_metadata_path, course))
        else:
            logger.warning(f"Nessun metadato trovato per il corso: {course.name}")

    # Salvataggio del file JSON finale
    os.makedirs(os.path.dirname(PARSED_COURSES_DATA_FILE), exist_ok=True)

    with open(PARSED_COURSES_DATA_FILE, "w", encoding="utf-8") as out_file:
        json.dump(all_parsed_data, out_file, indent=4, ensure_ascii=False)

    logger.info(f"Parsing completato! Dati salvati in '{PARSED_COURSES_DATA_FILE}'.")


# Esegui il parsing di tutti i corsi
if __name__ == "__main__":
    parse_all_courses_metadata()
