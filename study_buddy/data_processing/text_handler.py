import json
from pathlib import Path
from typing import Union, Dict
import docx
import ebooklib
from ebooklib import epub
from PyPDF2 import PdfReader
from study_buddy.config import RAW_DATA_DIR, EXTRACTED_TEXT_DIR, METADATA_DIR


class TextExtractor:
    SUPPORTED_FORMATS = {".txt", ".docx", ".epub", ".pdf"}

    def __init__(self, data_dir: Union[str, Path], output_dir: Union[str, Path], metadata_dir: Union[str, Path]):
        """
        Initialize the TextExtractor module.
        
        Args:
            data_dir (Union[str, Path]): Path to the directory containing the input files.
            output_dir (Union[str, Path]): Path to the directory where extracted text will be saved.
            metadata_dir (Union[str, Path]): Path to the directory containing lesson metadata.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.metadata_dir = Path(metadata_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_text(self, file_path: Path) -> str:
        """
        Extract text based on file format.
        
        Args:
            file_path (Path): Path to the file to process.
        
        Returns:
            str: Extracted text content.
        """
        ext = file_path.suffix.lower()
        extractor_methods = {
            ".txt": self._extract_text_from_txt,
            ".docx": self._extract_text_from_docx,
            ".epub": self._extract_text_from_epub,
            ".pdf": self._extract_text_from_pdf
        }
        
        extractor = extractor_methods.get(ext)
        if not extractor:
            raise ValueError(f"Unsupported file format: {ext}")

        return extractor(file_path)

    def _extract_text_from_txt(self, file_path: Path) -> str:
        """Extract text from a .txt file."""
        with file_path.open("r", encoding="utf-8") as f:
            return f.read()

    def _extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from a .docx file."""
        doc = docx.Document(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    def _extract_text_from_epub(self, file_path: Path) -> str:
        """Extract text from an .epub file."""
        book = epub.read_epub(file_path)
        text = [item.get_content().decode("utf-8")
                for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT]
        return "\n".join(text)

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from a .pdf file."""
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages)

    def load_lesson_metadata(self, lesson_metadata_path: Path) -> Dict:
        """
        Load lesson metadata from a JSON file.
        
        Args:
            lesson_metadata_path (Path): Path to the lesson metadata JSON file.
        
        Returns:
            Dict: Parsed metadata.
        """
        if not lesson_metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {lesson_metadata_path}")

        with lesson_metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def process_all_files(self):
        """
        Process all files in the data directory, extracting text and metadata.
        """
        for lesson_metadata_file in self.metadata_dir.rglob("lesson*_metadata.json"):
            lesson_metadata = self.load_lesson_metadata(lesson_metadata_file)

            for file_path in self.data_dir.rglob("*"):
                if file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                    try:
                        print(f"Processing file: {file_path}")
                        text = self.extract_text(file_path)
                        output_file = self.output_dir / f"{file_path.stem}.json"
                        self._save_text_to_json(file_path.name, text, output_file, lesson_metadata)
                        print(f"Text and metadata saved to {output_file}")
                    except Exception as e:
                        print(f"Failed to process file {file_path}: {e}")

    def _save_text_to_json(self, file_name: str, text: str, output_file: Path, metadata: Dict):
        """
        Save extracted text and selected metadata to a JSON file.
        
        Args:
            file_name (str): Name of the original file.
            text (str): Extracted text content.
            output_file (Path): Path to the JSON file where text will be saved.
            metadata (Dict): Full lesson metadata from which selected fields will be extracted.
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

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    extractor = TextExtractor(RAW_DATA_DIR, EXTRACTED_TEXT_DIR, METADATA_DIR)
    extractor.process_all_files()
