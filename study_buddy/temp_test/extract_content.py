from pathlib import Path
from study_buddy.data_processing.content_extraction import extract_content_from_course, extract_supplementary_materials_from_lesson, extract_references_from_lesson
from study_buddy.data_processing.text_handler import TextExtractor
from study_buddy.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR

path = Path("C:\\Users\\e1204\\Downloads\\SIIA\\SIIA")
lesson_path = Path("C:\\Users\\e1204\\Downloads\\SIIA\\SIIA\\lesson1")

extractor = TextExtractor(data_dir=lesson_path, output_dir=PROCESSED_DATA_DIR, metadata_dir=lesson_path)

extract_content_from_course(path)