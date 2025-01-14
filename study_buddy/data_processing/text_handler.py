import os
import json
from pathlib import Path
import docx
import ebooklib
from ebooklib import epub
from PyPDF2 import PdfReader
from study_buddy.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR


class TextExtractor:
    def __init__(self, data_dir, output_dir, metadata_dir):
        """
        Initialize the TextExtractor module.
        
        Args:
            data_dir (str): Path to the directory containing the input files.
            output_dir (str): Path to the directory where extracted text will be saved.
            metadata_dir (str): Path to the directory containing lesson metadata.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.metadata_dir = Path(metadata_dir)
        self.supported_formats = {".txt", ".docx", ".epub", ".pdf"}
        
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_text(self, file_path):
        """
        Extract text based on file format.
        
        Args:
            file_path (Path): Path to the file to process.
        
        Returns:
            str: Extracted text content.
        """
        ext = file_path.suffix.lower()
        if ext == ".txt":
            return self._extract_text_from_txt(file_path)
        elif ext == ".docx":
            return self._extract_text_from_docx(file_path)
        elif ext == ".epub":
            return self._extract_text_from_epub(file_path)
        elif ext == ".pdf":
            return self._extract_text_from_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _extract_text_from_txt(self, file_path):
        """
        Extract text from a .txt file.
        
        Args:
            file_path (Path): Path to the .txt file.
        
        Returns:
            str: Extracted text content.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _extract_text_from_docx(self, file_path):
        """
        Extract text from a .docx file.
        
        Args:
            file_path (Path): Path to the .docx file.
        
        Returns:
            str: Extracted text content.
        """
        doc = docx.Document(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    def _extract_text_from_epub(self, file_path):
        """
        Extract text from an .epub file.
        
        Args:
            file_path (Path): Path to the .epub file.
        
        Returns:
            str: Extracted text content.
        """
        book = epub.read_epub(file_path)
        text = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                text.append(item.get_content().decode("utf-8"))
        return "\n".join(text)

    def _extract_text_from_pdf(self, file_path):
        """
        Extract text from a .pdf file.
        
        Args:
            file_path (Path): Path to the .pdf file.
        
        Returns:
            str: Extracted text content.
        """
        text = []
        reader = PdfReader(file_path)
        for page in reader.pages:
            text.append(page.extract_text())
        return "\n".join(text)

    def load_lesson_metadata(self, lesson_metadata_path):
        """
        Load lesson metadata from a JSON file.
        
        Args:
            lesson_metadata_path (str): Path to the lesson metadata JSON file.
        
        Returns:
            dict: Parsed metadata.
        """
        if not os.path.exists(lesson_metadata_path):
            raise FileNotFoundError(f"Metadata not found: {lesson_metadata_path}")
        with open(lesson_metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def process_all_files(self):
        """
        Process all files in the data directory, extracting text and metadata.
        """
        for lesson_metadata_file in self.metadata_dir.rglob("lesson*_metadata.json"):
            lesson_metadata = self.load_lesson_metadata(lesson_metadata_file)
            for file_path in self.data_dir.rglob("*"):
                if file_path.suffix.lower() in self.supported_formats:
                    try:
                        print(f"Processing file: {file_path}")
                        text = self.extract_text(file_path)
                        output_file = self.output_dir / f"{file_path.stem}.json"
                        self._save_text_to_json(file_path.name, text, output_file, lesson_metadata)
                        print(f"Text and metadata saved to {output_file}")
                    except Exception as e:
                        print(f"Failed to process file {file_path}: {e}")

    def _save_text_to_json(self, file_name, text, output_file, metadata):
        """
        Save extracted text and selected metadata to a JSON file.
        
        Args:
            file_name (str): Name of the original file.
            text (str): Extracted text content.
            output_file (Path): Path to the JSON file where text will be saved.
            metadata (dict): Full lesson metadata from which selected fields will be extracted.
        """
        selected_metadata = {
            "lesson_title": metadata.get("title", ""),
            "lesson_number": metadata.get("lesson_number", ""),
            "keywords": metadata.get("keywords", []),
            "references": [
                {
                    "title": ref.get("title", ""),
                    "description": ref.get("description", "")
                }
                for ref in metadata.get("references", [])
            ],
            "supplementary_materials": [
                {
                    "title": sup.get("title", ""),
                    "description": sup.get("description", "")
                }
                for sup in metadata.get("supplementary_materials", [])
            ]
        }

        data = {
            "file_name": file_name,
            "content": text,
            "metadata": selected_metadata
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    data_directory = RAW_DATA_DIR
    output_directory = PROCESSED_DATA_DIR
    metadata_dir = METADATA_DIR

    extractor = TextExtractor(data_directory, output_directory, metadata_dir)
    extractor.process_all_files()
