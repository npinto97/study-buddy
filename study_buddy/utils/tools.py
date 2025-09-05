import tempfile
import os
import base64
import uuid
import requests
from typing import Union, List, Optional, Dict, Any
from abc import ABC, abstractmethod

import pandas as pd
import fitz  # pip install PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from textblob import TextBlob

from pydantic import BaseModel, Field
from langchain_core.tools import Tool
from langchain.tools import StructuredTool
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.llms import Together
from langchain_community.tools.google_lens import GoogleLensQueryRun
from langchain_community.utilities.google_lens import GoogleLensAPIWrapper
from langchain_community.tools import YouTubeSearchTool, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.google_books import GoogleBooksQueryRun
from langchain_community.utilities.google_books import GoogleBooksAPIWrapper
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_tavily import TavilySearch
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain.text_splitter import CharacterTextSplitter

from e2b_code_interpreter import Sandbox
from elevenlabs.client import ElevenLabs
from elevenlabs import save
import assemblyai as aai
from together import Together as TogetherClient

from study_buddy.vectorstore_pipeline.vector_store_builder import get_vector_store
from study_buddy.config import FAISS_INDEX_DIR


# =============================================================================
# Configuration and Constants
# =============================================================================

class Config:
    """Centralized configuration management."""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
    SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
    E2B_API_KEY = os.getenv("E2B_API_KEY")
    
    # Default models and settings
    DEFAULT_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    DEFAULT_TTS_MODEL = "eleven_multilingual_v2"
    DEFAULT_VOICE_ID = "paraDwhkbSkX4FhBkAzc"
    
    @classmethod
    def validate_key(cls, key_name: str) -> str:
        """Validate that required API key exists."""
        key = getattr(cls, key_name, None)
        if not key:
            raise ValueError(f"{key_name} must be set in environment variables")
        return key


# =============================================================================
# Base Classes and Utilities
# =============================================================================

class BaseWrapper(ABC):
    """Base class for all tool wrappers."""
    
    def __init__(self, **kwargs):
        self.validate_dependencies()
    
    @abstractmethod
    def validate_dependencies(self):
        """Validate required dependencies and API keys."""
        pass


class FileProcessor:
    """Utility class for file operations."""
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """Get normalized file extension."""
        return os.path.splitext(file_path)[1].lower()
    
    @staticmethod
    def create_temp_file(suffix: str = "") -> str:
        """Create temporary file and return path."""
        return tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
    
    @staticmethod
    def download_file(url: str, suffix: str = "") -> str:
        """Download file from URL to temporary location."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        temp_path = FileProcessor.create_temp_file(suffix)
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return temp_path


class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create_together_llm(
        model: str = Config.DEFAULT_LLM_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 1024
    ):
        """Create Together LLM instance."""
        return Together(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            together_api_key=Config.validate_key("TOGETHER_API_KEY")
        )
        

def resolve_file_path(file_path: str) -> str:
    """
    Risolve automaticamente il percorso del file.
    Se riceve solo il nome del file, lo cerca nella cartella uploaded_files.
    """
    if os.path.isabs(file_path) and os.path.exists(file_path):
        return file_path
    
    if not os.path.isabs(file_path):
        upload_dir = os.path.join(os.getcwd(), "uploaded_files")
        potential_path = os.path.join(upload_dir, file_path)
        if os.path.exists(potential_path):
            return potential_path
    
    return file_path


# =============================================================================
# Pydantic Models for Input Validation
# =============================================================================

class FilePathInput(BaseModel):
    """Base model for tools that require file paths."""
    file_path: str = Field(description="Path to the input file")


class QueryInput(BaseModel):
    """Base model for tools that require queries."""
    query: str = Field(description="Search or analysis query")


class CodeInterpreterInput(BaseModel):
    """Input schema for code interpreter."""
    code: str = Field(description="Python code to execute")


class DataVizInput(BaseModel):
    """Input schema for data visualization."""
    csv_path: str = Field(description="Absolute path to the CSV file")
    query: str = Field(description="Natural language question about the dataset")


class AudioInput(BaseModel):
    """Input schema for audio processing."""
    audio_input: Union[str, bytes] = Field(description="Audio file path, URL, or raw bytes")


# =============================================================================
# Core Tool Implementations
# =============================================================================

class VectorStoreRetriever(BaseWrapper):
    """Enhanced vector store retrieval tool."""
    
    def validate_dependencies(self):
        """Validate vector store availability."""
        if not os.path.exists(FAISS_INDEX_DIR):
            raise ValueError(f"FAISS index directory not found: {FAISS_INDEX_DIR}")
    
    def retrieve(self, query: str, k: int = 4) -> tuple[str, list, list]:
        """Retrieve information with validation and error handling."""
        try:
            vector_store = get_vector_store(FAISS_INDEX_DIR)
            
            # Validate embedding dimensions
            embedding_dim = vector_store.index.d
            test_embedding = vector_store.embeddings.embed_query("test")
            if len(test_embedding) != embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: {len(test_embedding)} vs {embedding_dim}"
                )
            
            retrieved_docs = vector_store.similarity_search(query, k=k)
            
            file_paths = []
            serialized_parts = []
            
            for doc in retrieved_docs:
                path = doc.metadata.get("file_path")
                if path:
                    normalized_path = os.path.normpath(path)
                    file_paths.append(normalized_path)
                
                serialized_parts.append(
                    f"Source: {doc.metadata}\nContent: {doc.page_content}"
                )
            
            return "\n\n".join(serialized_parts), retrieved_docs, file_paths
            
        except Exception as e:
            return f"Error during retrieval: {str(e)}", [], []


class DocumentProcessor(BaseWrapper):
    """Unified document processing for various file types."""
    
    def validate_dependencies(self):
        """Validate document processing dependencies."""
        pass
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from various document types."""
        
        resolved_path = resolve_file_path(file_path)
        
        ext = FileProcessor.get_file_extension(resolved_path)
        
        if ext == ".pdf":
            return self._extract_from_pdf(resolved_path)
        elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            return self._extract_from_image(resolved_path)
        elif ext == ".txt":
            return self._extract_from_text(resolved_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with OCR fallback."""
        try:
            doc = fitz.open(file_path)
            text = "\n".join([page.get_text("text") for page in doc])
            doc.close()
            
            # Use OCR if no text found (scanned PDF)
            if not text.strip():
                return self._extract_from_scanned_pdf(file_path)
            return text
        except Exception as e:
            return f"Error extracting from PDF: {str(e)}"
    
    def _extract_from_scanned_pdf(self, file_path: str) -> str:
        """Extract text from scanned PDF using OCR."""
        try:
            images = convert_from_path(file_path)
            return "\n".join([pytesseract.image_to_string(img) for img in images])
        except Exception as e:
            return f"Error in PDF OCR: {str(e)}"
    
    def _extract_from_image(self, file_path: str) -> str:
        """Extract text from image using OCR."""
        try:
            image = Image.open(file_path)
            return pytesseract.image_to_string(image)
        except Exception as e:
            return f"Error extracting from image: {str(e)}"
    
    def _extract_from_text(self, file_path: str) -> str:
        """Load text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading text file: {str(e)}"


class DocumentSummarizer(BaseWrapper):
    """Document summarization using Gradio API."""
    
    def validate_dependencies(self):
        """Validate Gradio client dependencies."""
        try:
            from gradio_client import Client, handle_file
        except ImportError:
            raise ValueError("gradio_client not installed. Run: pip install gradio_client")
    
    def __init__(self):
        super().__init__()
        from gradio_client import Client
        self.client = Client("Tulika2000/ai-pdf-summarizer")
    
    def summarize(self, file_path: str, length: str = "Medium", style: str = "Key Takeaways") -> str:
        """
        Summarize document content using Gradio API.
        
        Args:
            file_path: Path to the PDF file
            length: Summary length ("Short", "Medium", "Long")  
            style: Summary style ("Key Takeaways", "Executive Summary", "Detailed Analysis")
        """
        try:
            from gradio_client import handle_file
            
            print(f"Summarizing document: {file_path}")
            resolved_path = resolve_file_path(file_path)
            print(f"Resolved file path: {resolved_path}")
            
            # Verify file exists and is PDF
            if not os.path.exists(resolved_path):
                return f"Error: File not found at {resolved_path}"
            
            ext = FileProcessor.get_file_extension(resolved_path)
            if ext != ".pdf":
                return f"Error: Only PDF files are supported. Got: {ext}"
            
            # Call Gradio API
            result = self.client.predict(
                file=handle_file(resolved_path),
                length=length,
                style=style,
                api_name="/predict"
            )
            
            return result
            
        except Exception as e:
            return f"Error summarizing document: {str(e)}"


class SentimentAnalyzer(BaseWrapper):
    """Text sentiment analysis."""
    
    def validate_dependencies(self):
        """Validate sentiment analysis dependencies."""
        pass
    
    def analyze(self, file_path: str) -> str:
        """Analyze sentiment from file."""
        try:
            resolved_path = resolve_file_path(file_path)
            
            processor = DocumentProcessor()
            text = processor.extract_text(resolved_path)
            
            if text.startswith("Error"):
                return text
            
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0:
                sentiment = "Positive"
            elif polarity < 0:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            return (
                f"Sentiment: {sentiment}\n"
                f"Polarity: {polarity:.3f}\n"
                f"Subjectivity: {subjectivity:.3f}"
            )
            
        except Exception as e:
            return f"Error in sentiment analysis: {str(e)}"


class AudioProcessor(BaseWrapper):
    """Audio processing for TTS and STT."""
    
    def validate_dependencies(self):
        """Validate audio processing dependencies."""
        Config.validate_key("ELEVEN_API_KEY")
        Config.validate_key("ASSEMBLYAI_API_KEY")
    
    def text_to_speech(self, text: str) -> str:
        """Convert text to speech using ElevenLabs."""
        try:
            client = ElevenLabs(api_key=Config.ELEVEN_API_KEY)
            
            audio = client.text_to_speech.convert(
                text=text,
                voice_id=Config.DEFAULT_VOICE_ID,
                model_id=Config.DEFAULT_TTS_MODEL,
                output_format="mp3_44100_128"
            )
            
            temp_path = FileProcessor.create_temp_file(".mp3")
            save(audio, temp_path)
            return temp_path
            
        except Exception as e:
            return f"Error in text-to-speech: {str(e)}"
    
    def speech_to_text(self, audio_input: Union[str, bytes]) -> str:
        """Convert speech to text using AssemblyAI with Italian language support."""
        try:
            import assemblyai as aai
            import os

            # Imposta la chiave API
            aai.settings.api_key = Config.ASSEMBLYAI_API_KEY
            transcriber = aai.Transcriber()

            # Prepara il file audio
            audio_path = self._prepare_audio_path(audio_input)

            # Configurazione della trascrizione
            config = aai.TranscriptionConfig(
                language_code="it",                     
                punctuate=True,                         
                format_text=True,                       
                speech_model=aai.SpeechModel.best       
            )

            # Avvia la trascrizione
            transcript = transcriber.transcribe(audio_path, config)

            # Controlla se c'è stato un errore
            if transcript.status == aai.TranscriptStatus.error:
                raise RuntimeError(f"Transcription failed: {transcript.error}")

            return transcript.text

        except Exception as e:
            return f"Error in speech-to-text: {str(e)}"

        finally:
            # Rimuove il file temporaneo se esiste
            if isinstance(audio_input, (str, bytes)) and os.path.exists(audio_path):
                os.remove(audio_path)

    
    def _prepare_audio_path(self, audio_input: Union[str, bytes]) -> str:
        """Prepare audio file path from various input types."""
        if isinstance(audio_input, str):
            if audio_input.startswith("http"):
                return FileProcessor.download_file(audio_input, ".mp3")
            elif os.path.exists(audio_input):
                return audio_input
            else:
                raise ValueError("Invalid audio path")
        elif isinstance(audio_input, bytes):
            temp_path = FileProcessor.create_temp_file(".mp3")
            with open(temp_path, "wb") as f:
                f.write(audio_input)
            return temp_path
        else:
            raise TypeError("Unsupported audio input type")


class SpotifySearcher(BaseWrapper):
    """Spotify music search functionality."""
    
    def validate_dependencies(self):
        """Validate Spotify API credentials."""
        Config.validate_key("SPOTIFY_CLIENT_ID")
        Config.validate_key("SPOTIFY_CLIENT_SECRET")
    
    def __init__(self):
        super().__init__()
        self.access_token = self._get_access_token()
    
    def _get_access_token(self) -> str:
        """Get Spotify access token."""
        auth_url = "https://accounts.spotify.com/api/token"
        response = requests.post(
            auth_url,
            data={"grant_type": "client_credentials"},
            auth=(Config.SPOTIFY_CLIENT_ID, Config.SPOTIFY_CLIENT_SECRET),
        )
        response.raise_for_status()
        return response.json()["access_token"]
    
    def search(self, query: str, search_type: str = "track", limit: int = 5) -> str:
        """Search for music on Spotify."""
        try:
            url = f"https://api.spotify.com/v1/search"
            params = {"q": query, "type": search_type, "limit": limit}
            headers = {"Authorization": f"Bearer self.access_token"}
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            items = data.get(f"{search_type}s", {}).get("items", [])
            
            results = []
            for item in items:
                name = item.get("name", "Unknown")
                artist = item.get("artists", [{}])[0].get("name", "Unknown Artist")
                link = item.get("external_urls", {}).get("spotify", "")
                results.append(f"{name} by {artist} → [Listen]({link})")
            
            return "\n".join(results) if results else "No results found"
            
        except Exception as e:
            return f"Error searching Spotify: {str(e)}"


class CodeInterpreter:
    """Enhanced code interpreter with E2B sandbox management."""

    def __init__(self):
        self._initialize_sandbox()

    def _initialize_sandbox(self):
        """Initialize E2B sandbox with error handling."""
        try:
            self.sandbox = Sandbox.create()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize E2B sandbox: {e}")

    def run_code(self, code: str) -> dict:
        """Execute code in sandbox with basic error handling."""
        try:
            execution = self.sandbox.run_code(code)
            return {
                "results": getattr(execution, "results", []),
                "stdout": getattr(execution.logs, "stdout", "") if hasattr(execution, "logs") else "",
                "stderr": getattr(execution.logs, "stderr", "") if hasattr(execution, "logs") else "",
                "error": getattr(execution, "error", None),
            }
        except Exception as e:
            raise RuntimeError(f"Code execution failed: {e}")

    def close(self):
        """Clean up sandbox resources."""
        try:
            self.sandbox.kill()
        except:
            pass


class CSVAnalyzer:
    """Hybrid CSV analysis: statistical and semantic analysis."""

    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)
        self.df = pd.read_csv(self.file_path)
        self.documents = self._load_documents()

    def _load_documents(self) -> List:
        loader = CSVLoader(file_path=self.file_path)
        raw_docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        return splitter.split_documents(raw_docs)

    def get_statistical_analysis(self) -> str:
        parts = [
            f"Columns: {list(self.df.columns)}",
            f"Rows: {len(self.df)}",
            f"Data types:\n{self.df.dtypes.to_string()}",
            f"\nFirst 3 rows:\n{self.df.head(3).to_string(index=False)}",
            f"\nDescriptive statistics:\n{self.df.describe(include='all').fillna('').to_string()}",
        ]
        return "\n\n".join(parts)

    def get_semantic_analysis(self) -> str:
        try:
            llm = LLMFactory.create_together_llm()
            content = "\n\n".join([doc.page_content for doc in self.documents[:10]])
            prompt = f"Analyze this CSV dataset content and provide key insights:\n\n{content}"
            return llm.invoke(prompt)
        except Exception as e:
            return f"Error in semantic analysis: {str(e)}"

    def full_analysis(self) -> str:
        return f"[STATISTICAL ANALYSIS]\n{self.get_statistical_analysis()}\n\n[SEMANTIC ANALYSIS]\n{self.get_semantic_analysis()}"


class DataVisualizer:
    """Data visualization tool using sandboxed Python execution."""

    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.sandbox = Sandbox.create()
        self.llm = TogetherClient(api_key=Config.TOGETHER_API_KEY)

    def create_visualization(self, csv_path: str, query: str) -> dict:
        """Generate visualizations from natural language query."""
        try:
            resolved_path = os.path.abspath(csv_path)
            with open(resolved_path, "rb") as f:
                sandbox_path = self.sandbox.files.write("dataset.csv", f).path

            code = self._generate_code(query, sandbox_path, resolved_path)
            image_paths = self._execute_and_save(code)
            return {"image_paths": image_paths, "success": True}

        except Exception as e:
            return {"error": str(e), "success": False}

    def _generate_code(self, query: str, sandbox_path: str, local_path: str) -> str:
        """Generate Python code for visualization."""
        df_sample = pd.read_csv(local_path, nrows=300)
        column_info = df_sample.dtypes.astype(str).to_dict()

        prompt = f"""Generate Python code for visualization.
Dataset path in sandbox: {sandbox_path}
Columns and types: {column_info}
User query: "{query}"
Requirements:
- Load data: pd.read_csv("{sandbox_path}")
- Use matplotlib/seaborn
- Save plots: plt.savefig("chart-<name>.png")
- Call plt.close() after saving
- Output ONLY Python code, without ``` and useless spaces
Code:"""

        response = self.llm.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are a Python data visualization expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.4
        )

        code = response.choices[0].message.content.strip()
        return self._clean_code(code)

    def _clean_code(self, code: str) -> str:
        """Remove markdown formatting from generated code."""
        if code.startswith("```"):
            lines = code.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)
        return code.strip()

    def _execute_and_save(self, code: str) -> List[str]:
        """Execute code in sandbox and save images."""
        
        execution = self.sandbox.run_code(code)
        print("Code generated by LLM:", code)
        
        if execution.error:
            raise RuntimeError(f"Code execution failed: {execution.error.value}")

        saved_paths = []
        for result in getattr(execution, "results", []):
            if getattr(result, "png", None):
                filename = f"{uuid.uuid4().hex}.png"
                file_path = os.path.join(self.output_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(base64.b64decode(result.png))
                saved_paths.append(os.path.abspath(file_path))

        if not saved_paths:
            raise RuntimeError("No visualizations were generated")

        return saved_paths

    def close(self):
        """Clean up sandbox resources."""
        try:
            self.sandbox.kill()
        except:
            pass


# =============================================================================
# Custom API Wrappers with Error Handling
# =============================================================================

class RobustGoogleBooksWrapper(GoogleBooksAPIWrapper):
    """Enhanced Google Books wrapper with better error handling."""
    
    def _format(self, query: str, books: list) -> str:
        """Format search results with graceful handling of missing fields."""
        if not books:
            return f"No books found for query: {query}"
        
        results = [f"Found {len(books)} books for '{query}':"]
        
        for i, book in enumerate(books, start=1):
            info = book.get("volumeInfo", {})
            title = info.get("title", "Title unavailable")
            authors = self._format_authors(info.get("authors", ["Unknown author"]))
            summary = info.get("description", "No description")
            link = info.get("infoLink", "No link available")
            
            result = f'{i}. "{title}" by {authors}\n   {summary}\n   More info: {link}'
            results.append(result)
        
        return "\n\n".join(results)


class EnhancedGoogleScholarWrapper(GoogleScholarAPIWrapper):
    """Enhanced Google Scholar wrapper with URL inclusion."""
    
    def run(self, query: str) -> str:
        """Run query with enhanced result formatting."""
        total_results = []
        page = 0
        
        while page < max((self.top_k_results - 20), 1):
            results = (
                self.google_scholar_engine({
                    "q": query,
                    "start": page,
                    "hl": self.hl,
                    "num": min(self.top_k_results, 20),
                    "lr": self.lr,
                })
                .get_dict()
                .get("organic_results", [])
            )
            
            total_results.extend(results)
            if not results:
                break
            page += 20
        
        if not total_results:
            return "No Google Scholar results found"
        
        formatted_results = []
        for result in total_results:
            title = result.get('title', 'No title')
            authors = result.get('publication_info', {}).get('authors', [])
            author_names = ', '.join([author.get('name', '') for author in authors])
            summary = result.get('publication_info', {}).get('summary', 'No summary')
            citations = result.get('inline_links', {}).get('cited_by', {}).get('total', 'N/A')
            url = result.get('link', 'No URL available')
            
            formatted_result = (
                f"Title: {title}\n"
                f"Authors: {author_names or 'Unknown'}\n"
                f"Summary: {summary}\n"
                f"Citations: {citations}\n"
                f"URL: {url}"
            )
            formatted_results.append(formatted_result)
        
        return "\n\n".join(formatted_results)


class WikidataSearcher(BaseWrapper):
    """Wikidata SPARQL query wrapper."""
    
    def validate_dependencies(self):
        """Validate Wikidata access."""
        pass
    
    def search(self, query: str) -> str:
        """Search Wikidata using SPARQL."""
        try:
            url = "https://query.wikidata.org/sparql"
            headers = {"User-Agent": "StudyBuddyBot/1.0"}
            
            sparql_query = f"""
            SELECT ?item ?itemLabel ?description ?alias ?propertyLabel ?propertyValueLabel WHERE {{
              ?item ?label "{query}"@en.
              OPTIONAL {{ ?item skos:altLabel ?alias FILTER (LANG(?alias) = "en") }}
              OPTIONAL {{ ?item schema:description ?description FILTER (LANG(?description) = "en") }}
              OPTIONAL {{ 
                ?item ?property ?propertyValue.
                ?property wikibase:directClaim ?propClaim.
                ?propClaim rdfs:label ?propertyLabel.
                ?propertyValue rdfs:label ?propertyValueLabel.
                FILTER(LANG(?propertyLabel) = "en" && LANG(?propertyValueLabel) = "en") 
              }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT 10
            """
            
            response = requests.get(
                url, 
                params={"query": sparql_query, "format": "json"}, 
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            return self._format_results(response.json(), query)
            
        except Exception as e:
            return f"Error searching Wikidata: {str(e)}"
    
    def _format_results(self, data: dict, query: str) -> str:
        """Format Wikidata SPARQL results."""
        results = data.get("results", {}).get("bindings", [])
        
        if not results:
            return f"No Wikidata results for '{query}'"
        
        entities = {}
        for result in results:
            entity_id = result["item"]["value"].split("/")[-1]
            
            if entity_id not in entities:
                entities[entity_id] = {
                    "label": result.get("itemLabel", {}).get("value", "Unknown"),
                    "description": result.get("description", {}).get("value", "No description"),
                    "aliases": set(),
                    "properties": {}
                }
            
            # Collect aliases and properties
            alias = result.get("alias", {}).get("value")
            if alias:
                entities[entity_id]["aliases"].add(alias)
            
            prop_label = result.get("propertyLabel", {}).get("value")
            prop_value = result.get("propertyValueLabel", {}).get("value")
            if prop_label and prop_value:
                if prop_label not in entities[entity_id]["properties"]:
                    entities[entity_id]["properties"][prop_label] = set()
                entities[entity_id]["properties"][prop_label].add(prop_value)
        
        # Format final output
        formatted_results = []
        for entity_id, details in entities.items():
            result_str = [
                f"Entity {entity_id}:",
                f"Label: {details['label']}",
                f"Description: {details['description']}"
            ]
            
            if details["aliases"]:
                result_str.append(f"Aliases: {', '.join(sorted(details['aliases']))}")
            
            for prop, values in details["properties"].items():
                result_str.append(f"{prop}: {', '.join(sorted(values))}")
            
            formatted_results.append("\n".join(result_str))
        
        return "\n\n".join(formatted_results)


# =============================================================================
# Tool Creation Functions
# =============================================================================

def create_basic_tools() -> List[Tool]:
    """Create basic search and information tools."""
    # Initialize API wrappers
    web_search = TavilySearch(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
        include_images=True
    )
    
    youtube_search = YouTubeSearchTool()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    arxiv = ArxivQueryRun()
    
    google_books = None
    google_lens = None
    
    try:
        google_books = GoogleBooksQueryRun(
            api_wrapper=RobustGoogleBooksWrapper(
                google_books_api_key=Config.GOOGLE_API_KEY
            )
        )
    except ValueError:
        print("Warning: Google Books API key not found")
    
    try:
        google_lens = GoogleLensQueryRun(api_wrapper=GoogleLensAPIWrapper())
    except ValueError:
        print("Warning: Google Lens API key not found")
    
    # Create enhanced wrappers
    google_scholar = GoogleScholarQueryRun(api_wrapper=EnhancedGoogleScholarWrapper())
    
    # Vector store retriever
    retriever = VectorStoreRetriever()
    
    # Wikidata searcher
    wikidata_searcher = WikidataSearcher()
    
    tools = [
        Tool(
            name="retrieve_knowledge",
            description="Retrieve information from local knowledge base using semantic search",
            func=lambda query: retriever.retrieve(query)[0]
        ),
        Tool(
            name="web_search",
            description="Search the web for current information and news",
            func=web_search.run
        ),
        Tool(
            name="youtube_search",
            description="Search for educational videos on YouTube",
            func=youtube_search.run
        ),
        Tool(
            name="wikipedia_search",
            description="Search Wikipedia for encyclopedic information",
            func=wikipedia.run
        ),
        Tool(
            name="arxiv_search",
            description="Search academic papers on arXiv",
            func=arxiv.run
        ),
        Tool(
            name="google_scholar_search",
            description="Search academic literature on Google Scholar",
            func=google_scholar.run
        ),
        Tool(
            name="wikidata_search",
            description="Search structured data on Wikidata",
            func=wikidata_searcher.search
        )
    ]
    
    if google_books:
        tools.append(Tool(
            name="google_books_search",
            description="Search for books on Google Books",
            func=google_books.run
        ))
    
    if google_lens:
        tools.append(Tool(
            name="google_lens_analyze",
            description="Analyze images using Google Lens",
            func=google_lens.run
        ))
    
    return tools


def create_document_tools() -> List[Tool]:
    """Create document processing tools."""
    processor = DocumentProcessor()
    summarizer = DocumentSummarizer()
    sentiment_analyzer = SentimentAnalyzer()
    
    return [
        Tool(
            name="extract_text",
            description="Extract text from PDF, image, or text files using OCR if needed",
            func=processor.extract_text
        ),
        Tool(
            name="summarize_document",
            description="Generate a summary of a PDF document with Medium length and Key Takeaways style",
            func=summarizer.summarize
        ),
        Tool(
            name="analyze_sentiment",
            description="Analyze sentiment, polarity, and subjectivity of document content",
            func=sentiment_analyzer.analyze
        )
    ]


def create_audio_tools() -> List[Tool]:
    """Create audio processing tools."""
    try:
        audio_processor = AudioProcessor()
        
        return [
            Tool(
                name="text_to_speech",
                description="Convert text to speech using ElevenLabs TTS",
                func=audio_processor.text_to_speech
            ),
            Tool(
                name="speech_to_text",
                description="Transcribe audio files to text using AssemblyAI",
                func=audio_processor.speech_to_text
            )
        ]
    except ValueError as e:
        print(f"Warning: Audio tools not available - {e}")
        return []


def create_multimedia_tools() -> List[Tool]:
    """Create multimedia and entertainment tools."""
    tools = []
    
    # Spotify search
    try:
        spotify = SpotifySearcher()
        tools.append(Tool(
            name="spotify_search",
            description="Search for music on Spotify and get listening links",
            func=spotify.search
        ))
    except ValueError as e:
        print(f"Warning: Spotify search not available - {e}")
    
    return tools



def create_data_analysis_tools() -> List[Tool]:
    """Create data analysis and visualization tools."""
    tools = []

    # Code interpreter 
    try:
        code_interpreter = CodeInterpreter()
        
        def execute_code(code: str) -> str:
            """Safely execute code and return formatted results."""
            try:
                result = code_interpreter.run_code(code)
                
                # Format the execution results
                output_parts = []
                
                if result.get("stdout"):
                    output_parts.append(f"Output:\n{result['stdout']}")
                
                if result.get("stderr"):
                    output_parts.append(f"Errors:\n{result['stderr']}")
                
                if result.get("error"):
                    error_msg = result['error']
                    if hasattr(error_msg, 'message'):
                        error_text = error_msg.message
                    elif hasattr(error_msg, 'value'):
                        error_text = error_msg.value
                    else:
                        error_text = str(error_msg)
                    output_parts.append(f"Execution Error:\n{error_text}")
                
                if result.get("results"):
                    # Handle different types of results (charts, data, etc.)
                    for i, res in enumerate(result["results"]):
                        if hasattr(res, 'png') and res.png:
                            output_parts.append(f"Generated visualization {i+1} (PNG image)")
                        elif hasattr(res, 'text') and res.text:
                            output_parts.append(f"Text output {i+1}:\n{res.text}")
                        elif hasattr(res, 'json') and res.json:
                            output_parts.append(f"JSON output {i+1}:\n{res.json}")
                        else:
                            output_parts.append(f"Result {i+1}: {str(res)}")
                
                return "\n\n".join(output_parts) if output_parts else "Code executed successfully with no output"
                
            except Exception as e:
                return f"Error executing code: {str(e)}"
        
        tools.append(StructuredTool.from_function(
            func=execute_code,
            name="execute_code",
            description="Execute Python code in a secure E2B sandbox environment. Supports data analysis, visualization, file operations, and more.",
            args_schema=CodeInterpreterInput
        ))
        
    except ValueError as e:
        print(f"Warning: Code interpreter not available - {e}")
    except Exception as e:
        print(f"Warning: Failed to initialize code interpreter - {e}")

    # CSV analysis tool
    def analyze_csv(file_path: str) -> str:
        """Safely analyze CSV files."""
        try:
            resolved_path = resolve_file_path(file_path)
            if not os.path.exists(resolved_path):
                return f"Error: File not found at {resolved_path}"
            
            analyzer = CSVAnalyzer(resolved_path)
            return analyzer.full_analysis()
        except Exception as e:
            return f"Error analyzing CSV: {str(e)}"

    tools.append(StructuredTool.from_function(
        func=analyze_csv,
        name="analyze_csv",
        description="Perform statistical and semantic analysis on CSV files. Provides data overview, statistics, and AI-generated insights.",
        args_schema=FilePathInput
    ))

    # Data visualization tool
    try:
        visualizer = DataVisualizer()

        def create_visualization(csv_path: str, query: str) -> str:
            """Safely create data visualizations."""
            try:
                print(f"Creating visualization for file: {csv_path} with query: {query}")
                resolved_path = resolve_file_path(csv_path)
                print(f"Resolved file path: {resolved_path}")
                
                if not os.path.exists(resolved_path):
                    return f"Error: The file '{resolved_path}' doesn't exist. Please upload a valid CSV file."
                
                result = visualizer.create_visualization(resolved_path, query)
                
                if result.get("success"):
                    image_paths = result.get("image_paths", [])
                    if image_paths:
                        paths_str = "\n".join([f"- {path}" for path in image_paths])
                        return f"Successfully created {len(image_paths)} visualization(s):\n{paths_str}"
                    else:
                        return "Visualization completed but no image files were generated."
                else:
                    return f"Visualization failed: {result.get('error', 'Unknown error')}"
                    
            except Exception as e:
                return f"Error creating visualization: {str(e)}"

        tools.append(StructuredTool.from_function(
            func=create_visualization,
            name="create_visualization",
            description="Generate data visualizations from CSV files using natural language queries. Creates charts, graphs, and plots based on your description.",
            args_schema=DataVizInput
        ))

    except ValueError as e:
        print(f"Warning: Data visualizer not available - {e}")
    except Exception as e:
        print(f"Warning: Failed to initialize data visualizer - {e}")

    return tools



# =============================================================================
# Main Tool Assembly
# =============================================================================

def get_all_tools() -> List[Tool]:
    """Assemble all available tools with error handling."""
    all_tools = []
    
    print("Initializing tools...")
    
    try:
        basic_tools = create_basic_tools()
        all_tools.extend(basic_tools)
        print(f"Loaded {len(basic_tools)} basic tools")
    except Exception as e:
        print(f"Error loading basic tools: {e}")
    
    try:
        doc_tools = create_document_tools()
        all_tools.extend(doc_tools)
        print(f"Loaded {len(doc_tools)} document tools")
    except Exception as e:
        print(f"Error loading document tools: {e}")
    
    try:
        audio_tools = create_audio_tools()
        all_tools.extend(audio_tools)
        print(f"Loaded {len(audio_tools)} audio tools")
    except Exception as e:
        print(f"Error loading audio tools: {e}")
    
    try:
        multimedia_tools = create_multimedia_tools()
        all_tools.extend(multimedia_tools)
        print(f"Loaded {len(multimedia_tools)} multimedia tools")
    except Exception as e:
        print(f"Error loading multimedia tools: {e}")
    
    try:
        data_tools = create_data_analysis_tools()
        all_tools.extend(data_tools)
        print(f"Loaded {len(data_tools)} data analysis tools")
    except Exception as e:
        print(f"Error loading data analysis tools: {e}")
    
    print(f"Total tools loaded: {len(all_tools)}")
    return all_tools
