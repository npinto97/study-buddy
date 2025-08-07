import tempfile, os, base64
import pandas as pd
import fitz  # PyMuPDF per estrazione testo da PDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import requests
import tempfile
from typing import Union
from textblob import TextBlob
import matplotlib.pyplot as plt

import os
import base64
import uuid
from typing import List
from together import Together

from pydantic import BaseModel, Field
from langchain_together import ChatTogether

from langchain_core.tools import Tool
# from langchain_community.agent_toolkits.load_tools import load_tools

from langchain.tools import StructuredTool

# from langchain_community.agent_toolkits import O365Toolkit
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.llms import Together
from langchain_community.tools.google_lens import GoogleLensQueryRun
from langchain_community.utilities.google_lens import GoogleLensAPIWrapper
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.tools.google_books import GoogleBooksQueryRun
from langchain_community.utilities.google_books import GoogleBooksAPIWrapper
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_tavily import TavilySearch
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from e2b_code_interpreter import Sandbox

from langchain.schema import HumanMessage
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_core.language_models import BaseChatModel

from langchain.tools import StructuredTool

from elevenlabs.client import ElevenLabs
from elevenlabs import save
import assemblyai as aai


# from gradio_tools import (StableDiffusionTool,
#                           ImageCaptioningTool,
#                           StableDiffusionPromptGeneratorTool,
#                           TextToVideoTool,
#                           WhisperAudioTranscriptionTool)

from gradio_client import Client

from study_buddy.vectorstore_pipeline.vector_store_builder import get_vector_store
from study_buddy.config import FAISS_INDEX_DIR

google_api_key = os.getenv("GOOGLE_API_KEY")


# ------------------------------------------------------------------------------
# Basic tools
# base_tool = load_tools(["human", "pubmed"])


def retrieve_tool(query: str):
    """Retrieve information related to a query."""
    vector_store = get_vector_store(FAISS_INDEX_DIR)

    embedding_dim = vector_store.index.d
    test_embedding = vector_store.embeddings.embed_query("test")
    if len(test_embedding) != embedding_dim:
        raise ValueError(
            f"Dimensione embedding ({len(test_embedding)}) diversa da quella dell'indice FAISS ({embedding_dim}). "
            "Rigenera l'indice FAISS con lo stesso modello di embedding."
        )

    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


web_search_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True
)


class CustomGoogleBooksAPIWrapper(GoogleBooksAPIWrapper):
    """Custom Google Books API Wrapper to handle missing fields gracefully."""

    def _format(self, query: str, books: list) -> str:
        if not books:
            return f"Sorry, no books could be found for your query: {query}"

        results = [f"Here are {len(books)} suggestions for books related to '{query}':"]

        for i, book in enumerate(books, start=1):
            info = book.get("volumeInfo", {})
            title = info.get("title", "Title not available")
            authors = self._format_authors(info.get("authors", ["Unknown author"]))
            summary = info.get("description", "No description available")
            source = info.get("infoLink", "No source available")

            desc = f'{i}. "{title}" by {authors}: {summary}\n'
            desc += f"You can read more at {source}"
            results.append(desc)

        return "\n\n".join(results)


google_books_tool = GoogleBooksQueryRun(api_wrapper=CustomGoogleBooksAPIWrapper(google_books_api_key=google_api_key))

# toolkit = O365Toolkit()
# o365_tools = toolkit.get_tools()

# ------------------------------------------------------------------------------
# Tools for learning support

youtube_search_tool = YouTubeSearchTool()

wolfram = WolframAlphaAPIWrapper()


def wolfram_tool(query: str):
    """Query Wolfram Alpha and return the response."""
    return wolfram.run(query)


wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


# class QwenTextAnalysis:
#     """Wrapper per l'analisi del testo tramite il modello di Hugging Face."""

#     def __init__(self, api_url: str = "Qwen/Qwen2.5-Turbo-1M-Demo"):
#         self.client = Client(api_url)

#     def analyze_text(self, text: str, files: list = []):
#         """Invia il testo e i file per l'analisi."""
#         try:
#             result = self.client.predict(
#                 _input={"files": files, "text": text},
#                 _chatbot=[],
#                 api_name="/add_text"
#             )
#             return result
#         except Exception as e:
#             return {"error": str(e)}


# text_analysis_tool = Tool(
#     name="text_analysis",
#     description="Analyze text or files (pdf/docx/pptx/txt/html) using Qwen model.",
#     func=QwenTextAnalysis().analyze_text
# )


class DocumentSummarizerWrapper:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.documents = self.load_documents()

    def load_documents(self):
        """Carica il documento dal percorso fornito (supporta PDF e TXT)."""
        if self.file_path.endswith(".pdf"):
            loader = PyPDFLoader(self.file_path)
        else:
            loader = TextLoader(self.file_path)
        return loader.load()

    def summarize_document(self) -> str:
        """Riassume il contenuto del documento utilizzando un LLM open-source via Together.ai."""

        llm = Together(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=0.3,
            max_tokens=1024,
            together_api_key=os.getenv("TOGETHER_API_KEY")
        )

        chain = load_summarize_chain(llm, chain_type="map_reduce")

        summary = chain.invoke(self.documents)
        return summary


doc_summary_tool = Tool(
    name="summarize_document",
    description="Riassume un documento PDF o TXT fornito tramite file path.",
    func=lambda path: DocumentSummarizerWrapper(path).summarize_document()
)


# ------------------------------------------------------------------------------
# Scientific research support tools
arxive_tool = ArxivQueryRun()


class CustomGoogleScholarAPIWrapper(GoogleScholarAPIWrapper):
    def run(self, query: str) -> str:
        """Run query through GoogleSearchScholar and parse result, including URL"""
        total_results = []
        page = 0
        while page < max((self.top_k_results - 20), 1):
            results = (
                self.google_scholar_engine(  # type: ignore
                    {
                        "q": query,
                        "start": page,
                        "hl": self.hl,
                        "num": min(self.top_k_results, 20),
                        "lr": self.lr,
                    }
                )
                .get_dict()
                .get("organic_results", [])
            )
            total_results.extend(results)
            if not results:
                break
            page += 20

        if self.top_k_results % 20 != 0 and page > 20 and total_results:
            results = (
                self.google_scholar_engine(  # type: ignore
                    {
                        "q": query,
                        "start": page,
                        "num": self.top_k_results % 20,
                        "hl": self.hl,
                        "lr": self.lr,
                    }
                )
                .get_dict()
                .get("organic_results", [])
            )
            total_results.extend(results)

        if not total_results:
            return "No good Google Scholar Result was found"

        docs = [
            f"Title: {result.get('title', '')}\n"
            f"Authors: {', '.join([author.get('name') for author in result.get('publication_info', {}).get('authors', [])])}\n"
            f"Summary: {result.get('publication_info', {}).get('summary', '')}\n"
            f"Total-Citations: {result.get('inline_links', {}).get('cited_by', {}).get('total', '')}\n"
            f"URL: {result.get('link', 'No URL available')}"
            for result in total_results
        ]

        return "\n\n".join(docs)


google_scholar_tool = GoogleScholarQueryRun(api_wrapper=CustomGoogleScholarAPIWrapper())


class CustomWikidataAPIWrapper:
    """Custom Wrapper for Wikidata API using SPARQL."""

    def run(self, query: str) -> str:
        """Search for an entity on Wikidata using SPARQL and return detailed info."""
        wikidata_sparql_url = "https://query.wikidata.org/sparql"
        headers = {"User-Agent": "MyWikidataBot/1.0 (myemail@example.com)"}  # Sostituisci con la tua email

        sparql_query = f"""
        SELECT ?item ?itemLabel ?description ?alias ?propertyLabel ?propertyValueLabel WHERE {{
          ?item ?label "{query}"@en.
          OPTIONAL {{ ?item skos:altLabel ?alias FILTER (LANG(?alias) = "en") }}
          OPTIONAL {{ ?item schema:description ?description FILTER (LANG(?description) = "en") }}
          OPTIONAL {{ ?item ?property ?propertyValue.
                     ?property wikibase:directClaim ?propClaim.
                     ?propClaim rdfs:label ?propertyLabel.
                     ?propertyValue rdfs:label ?propertyValueLabel.
                     FILTER(LANG(?propertyLabel) = "en" && LANG(?propertyValueLabel) = "en") }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """

        response = requests.get(wikidata_sparql_url, params={"query": sparql_query, "format": "json"}, headers=headers)

        if response.status_code != 200:
            return f"Error: Unable to fetch data from Wikidata (status code {response.status_code})"

        data = response.json()
        results = data.get("results", {}).get("bindings", [])

        if not results:
            return f"No Wikidata result found for '{query}'"

        entities = {}
        for result in results:
            entity_id = result["item"]["value"].split("/")[-1]  # Extract QID
            label = result.get("itemLabel", {}).get("value", "Unknown")
            description = result.get("description", {}).get("value", "No description available")
            alias = result.get("alias", {}).get("value", "")
            property_label = result.get("propertyLabel", {}).get("value", "")
            property_value = result.get("propertyValueLabel", {}).get("value", "")

            if entity_id not in entities:
                entities[entity_id] = {
                    "Label": label,
                    "Description": description,
                    "Aliases": set(),
                    "Properties": {}
                }

            if alias:
                entities[entity_id]["Aliases"].add(alias)

            if property_label and property_value:
                if property_label not in entities[entity_id]["Properties"]:
                    entities[entity_id]["Properties"][property_label] = set()
                entities[entity_id]["Properties"][property_label].add(property_value)

        # Format output
        formatted_results = []
        for entity_id, details in entities.items():
            result_str = f"Result {entity_id}:\nLabel: {details['Label']}\nDescription: {details['Description']}"
            if details["Aliases"]:
                result_str += f"\nAliases: {', '.join(details['Aliases'])}"
            for prop, values in details["Properties"].items():
                result_str += f"\n{prop.lower()}: {', '.join(values)}"
            formatted_results.append(result_str)

        return "\n\n".join(formatted_results)


wikidata_tool = Tool(
    name="Wikidata",
    description="Searches for an entity on Wikidata and returns detailed info.",
    func=CustomWikidataAPIWrapper().run
)


class CodeInterpreterInput(BaseModel):
    """Input schema for the Code Interpreter tool."""
    code: str = Field(description="Python code to execute.")


class CodeInterpreterWrapper:
    """Wrapper for running Python code inside an E2B sandbox."""

    def __init__(self):
        self._initialize_sandbox()

    def _initialize_sandbox(self):
        """Initialize or restart the E2B sandbox."""
        if "E2B_API_KEY" not in os.environ:
            raise EnvironmentError("E2B_API_KEY is not set in the environment variables.")
        self.code_interpreter = Sandbox()
        self.code_interpreter.set_timeout(60)

    def run(self, code: str) -> dict:
        """Execute Python code and return results, stdout, stderr, and any errors."""
        try:
            execution = self.code_interpreter.run_code(code)
            return {
                "results": execution.results,
                "stdout": execution.logs.stdout,
                "stderr": execution.logs.stderr,
                "error": execution.error,
            }
        except Exception as e:
            if "502 Bad Gateway" in str(e) or "timeout" in str(e).lower():
                print("Timeout or gateway error. Restarting sandbox...")
                self._initialize_sandbox()
                return self.run(code)  # Retry once
            else:
                raise RuntimeError(f"Error while executing code: {e}")

    def close(self):
        """Shut down the E2B sandbox."""
        self.code_interpreter.kill()


code_intertpreter = Tool(
    name="code_intertpreter",
    description="Executes Python code in a secure sandbox and returns stdout, stderr, and results.",
    func=CodeInterpreterWrapper().run,
)



# ------------------------------------------------------------------------------
# Tools for emotional support and mental health

class SpotifyAPIWrapper:
    """Wrapper for Spotify Web API to search and play music."""

    def __init__(self):
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.access_token = self.get_access_token()

    def get_access_token(self):
        """Retrieve an access token from Spotify API."""
        auth_url = "https://accounts.spotify.com/api/token"
        auth_response = requests.post(
            auth_url,
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
        )
        if auth_response.status_code != 200:
            raise Exception("Error getting Spotify access token")
        return auth_response.json()["access_token"]

    def search_music(self, query: str, search_type="track", limit=5):
        """Search for a song, artist, or playlist on Spotify."""
        url = f"https://api.spotify.com/v1/search?q={query}&type={search_type}&limit={limit}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return f"Error searching Spotify (status {response.status_code})"

        data = response.json()
        results = []
        for item in data[search_type + "s"]["items"]:
            name = item["name"]
            artist = item["artists"][0]["name"] if "artists" in item else "Unknown Artist"
            link = item["external_urls"]["spotify"]
            results.append(f"{name} by {artist} → [Listen]({link})")

        return "\n".join(results) if results else "No results found."


spotify_music_tool = Tool(
    name="Spotify_music",
    description="Searches for a song, artist, or playlist on Spotify and provides a listening link.",
    func=SpotifyAPIWrapper().search_music
)

# aggiungi music lofi generator


# ------------------------------------------------------------------------------
# Tools for accessibility and inclusivity

# Convert text into spoken voice
class ElevenLabsTTSWrapper:
    """Wrapper aggiornato per ElevenLabs Text-to-Speech API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.client = ElevenLabs(api_key=self.api_key)

    def text_to_speech(self, text: str, voice: str = "Rachel", model: str = "eleven_monolingual_v1") -> str:
        """Converte testo in voce e salva l'audio in un file MP3 temporaneo."""
        audio = self.client.generate(
            text=text,
            voice=voice,
            model=model
        )
        tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        save(audio, tmp_path)
        return tmp_path


text_to_speech_tool = Tool(
    name="text_to_speech",
    description="Converte testo in voce utilizzando l'API ElevenLabs.",
    func=ElevenLabsTTSWrapper().text_to_speech,
)


class AssemblyAISpeechToText:
    """Wrapper for AssemblyAI transcription API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        settings = aai.Settings(api_key=self.api_key)
        self.client = aai.Client(settings=settings)

    def _get_audio_path(self, audio_input: Union[str, bytes]) -> str:
        """Download or prepare local audio file path from various input types."""
        if isinstance(audio_input, str):
            if audio_input.startswith("http"):
                response = requests.get(audio_input, stream=True)
                if response.status_code != 200:
                    raise ValueError("Failed to download audio from URL.")
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        tmp_file.write(chunk)
                tmp_file.close()
                return tmp_file.name
            elif os.path.exists(audio_input):
                return audio_input
            else:
                raise ValueError("Audio path is invalid.")
        elif isinstance(audio_input, bytes):
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp_file.write(audio_input)
            tmp_file.close()
            return tmp_file.name
        else:
            raise TypeError("Unsupported audio input type.")

    def transcribe_audio(self, audio_input: Union[str, bytes]) -> str:
        """Transcribes audio input and returns text."""
        audio_path = self._get_audio_path(audio_input)
        transcript = self.client.transcribe(audio_path)
        os.remove(audio_path)
        return transcript.text


speech_to_text_tool = Tool(
    name="speech_to_text",
    description="Trascrive file audio utilizzando l'API AssemblyAI.",
    func=AssemblyAISpeechToText().transcribe_audio,
)

# ------------------------------------------------------------------------------
# Advanced tools (multimodal, ...)

google_lens_tool = GoogleLensQueryRun(api_wrapper=GoogleLensAPIWrapper())


# class ImageGenerationAPIWrapper:
#     def __init__(self, model_name: str):
#         self.client = Client(model_name)

#     def generate_image(self, prompt: str, seed: int = 0, randomize_seed: bool = True, width: int = 512, height: int = 512, num_inference_steps: int = 15):
#         """Generates an image from a prompt using the Hugging Face API."""
#         result = self.client.predict(
#             prompt=prompt,
#             seed=seed,
#             randomize_seed=randomize_seed,
#             width=width,
#             height=height,
#             num_inference_steps=num_inference_steps,
#             api_name="/infer"
#         )
#         return result  # This could be an image URL or file path depending on the API response


# image_generation_wrapper = ImageGenerationAPIWrapper("black-forest-labs/FLUX.1-schnell")

# image_generation_tool = Tool(
#     name="image_generator",
#     description="Generates an image following a given prompt and returns the result.",
#     func=image_generation_wrapper.generate_image
# )


# class CLIPInterrogatorAPIWrapper:
#     def __init__(self, api_url: str):
#         self.client = Client(api_url)

#     def interrogate_image(self, image_url: str, model: str = "ViT-L (best for Stable Diffusion 1.*)", mode: str = "classic"):
#         """Interrogate the image to get information using CLIP-Interrogator."""
#         result = self.client.predict(
#             image_url,         # Image URL
#             model,             # Model to use
#             mode,              # Mode ('best', 'fast', 'classic', 'negative')
#             fn_index=3         # Function index for image interrogation
#         )
#         return result


# image_interrogator_wrapper = CLIPInterrogatorAPIWrapper("https://pharmapsychotic-clip-interrogator.hf.space/")

# image_interrogator_tool = Tool(
#     name="image_interrogator",
#     description="Interrogate an image and return artistic information, movement, and more.",
#     func=image_interrogator_wrapper.interrogate_image
# )

# ------------------------------------------------------------------------------
# Other tools (sentiment analysis, ocr, ...)
# ------------------------------------------------------------------------------


class SentimentAnalyzerWrapper:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.text = self.read_file()

    def read_file(self) -> str:
        """
        Legge il contenuto di un file PDF o TXT e lo restituisce come stringa.
        """
        if self.file_path.endswith(".pdf"):
            loader = PyPDFLoader(self.file_path)
        elif self.file_path.endswith(".txt"):
            loader = TextLoader(self.file_path)
        else:
            raise ValueError("Formato file non supportato. Usa .txt o .pdf.")

        docs = loader.load()
        return " ".join([doc.page_content for doc in docs])

    def analyze_sentiment(self) -> str:
        """
        Analizza il sentimento del testo estratto dal file.
        Restituisce il sentimento (positivo, negativo, neutro), polarità e soggettività.
        """
        blob = TextBlob(self.text)

        # Polarità e soggettività
        polarity = blob.sentiment.polarity  # Va da -1 (negativo) a 1 (positivo)
        subjectivity = blob.sentiment.subjectivity  # Va da 0 (oggettivo) a 1 (soggettivo)

        if polarity > 0:
            sentiment = "Positivo"
        elif polarity < 0:
            sentiment = "Negativo"
        else:
            sentiment = "Neutrale"

        return f"Sentimento: {sentiment}\nPolarità: {polarity}\nSoggettività: {subjectivity}"


def analyze_sentiment_from_wrapper(file_path: str) -> str:
    sentiment_analyzer = SentimentAnalyzerWrapper(file_path)
    return sentiment_analyzer.analyze_sentiment()


sentiment_tool = Tool(
    name="SentimentAnalyzer",
    description="Analizza il sentimento di un file (PDF o TXT) e restituisce polarità e soggettività.",
    func=analyze_sentiment_from_wrapper
)


class TextExtractorWrapper:
    def __init__(self, file_path: str):
        """
        Inizializza il wrapper e legge il contenuto del file.
        """
        self.file_path = file_path
        self.text = self.extract_text()

    def extract_text(self) -> str:
        """
        Determina il tipo di file ed estrae il testo di conseguenza.
        """
        if self.file_path.lower().endswith(".pdf"):
            return self.extract_text_from_pdf()
        elif self.file_path.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
            return self.extract_text_from_image()
        else:
            raise ValueError("Formato file non supportato. Usa PDF o immagini.")

    def extract_text_from_pdf(self) -> str:
        """
        Estrae testo da un PDF con testo selezionabile o usa OCR se è scansionato.
        """
        try:
            doc = fitz.open(self.file_path)
            text = "\n".join([page.get_text("text") for page in doc])

            # Se il PDF è scansionato (senza testo), usa OCR
            if not text.strip():
                return self.extract_text_from_scanned_pdf()
            return text
        except Exception as e:
            return f"Errore nell'estrazione del testo dal PDF: {str(e)}"

    def extract_text_from_scanned_pdf(self) -> str:
        """
        Esegue OCR su un PDF scansionato convertendolo in immagini.
        """
        try:
            images = convert_from_path(self.file_path)
            return "\n".join([pytesseract.image_to_string(img) for img in images])
        except Exception as e:
            return f"Errore nell'esecuzione dell'OCR sul PDF: {str(e)}"

    def extract_text_from_image(self) -> str:
        """
        Estrae il testo da un'immagine usando OCR.
        """
        try:
            image = Image.open(self.file_path)
            return pytesseract.image_to_string(image)
        except Exception as e:
            return f"Errore nell'estrazione del testo dall'immagine: {str(e)}"


def extract_text_from_wrapper(file_path: str) -> str:
    """
    Funzione helper per usare la classe TextExtractorWrapper.
    """
    extractor = TextExtractorWrapper(file_path)
    return extractor.text


extract_text_tool = Tool(
    name="TextExtractor",
    description="Estrae il testo da file PDF o immagini usando OCR se necessario.",
    func=extract_text_from_wrapper
)


# Static analysis of CSV files
class CSVHybridAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = self.load_dataframe()
        self.documents = self.load_documents()

    def load_dataframe(self):
        """Carica il CSV in un DataFrame pandas."""
        return pd.read_csv(self.file_path)

    def load_documents(self):
        """Carica il CSV come Documenti LangChain per analisi testuale."""
        loader = CSVLoader(file_path=self.file_path)
        raw_docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        return splitter.split_documents(raw_docs)

    def analyze_dataset(self) -> str:
        """Crea una breve descrizione del dataset."""
        info = f"Colonne: {list(self.df.columns)}\n"
        info += f"Numero di righe: {len(self.df)}\n"
        info += f"Prime righe:\n{self.df.head(3).to_string(index=False)}\n"
        stats = self.df.describe(include='all').fillna("").to_string()
        return f"{info}\nStatistiche descrittive:\n{stats}"

    def summarize_with_llm(self) -> str:
        """Genera un riassunto ragionato del dataset via Together.ai."""
        # Inizializza modello Together
        llm = Together(
            model="mistralai/Mistral-7B-Instruct-v0.2",  # Sostituibile con deepseek-ai/deepseek-llm-7b-instruct
            temperature=0.3,
            max_tokens=1024,
            together_api_key=os.getenv("TOGETHER_API_KEY")
        )

        # Prendi i primi 10 chunk testuali dal CSV
        base_text = "\n\n".join([doc.page_content for doc in self.documents[:10]])

        prompt = f"""Hai ricevuto un dataset in formato CSV. Ecco una parte del contenuto:

{base_text}

Fornisci un riassunto del contenuto, identificando eventuali pattern, informazioni interessanti o anomalie."""
        
        response = llm.invoke(prompt)
        return response

    def full_report(self) -> str:
        """Combina analisi descrittiva e semantica."""
        analysis = self.analyze_dataset()
        llm_summary = self.summarize_with_llm()
        return f"""[ANALISI_DESCRITTIVA]
{analysis}

[ANALISI_SEMANTICA]
{llm_summary}
"""


def hybrid_csv_analysis(file_path: str) -> str:
    analyzer = CSVHybridAnalyzer(file_path)
    return analyzer.full_report()


csv_hybrid_tool = Tool(
    name="CSVHybridAnalyzer",
    description="Esegue una doppia analisi su un CSV: statistica (pandas) e testuale (LLM) per insight utili.",
    func=hybrid_csv_analysis
)







import os
import uuid
import base64
import pandas as pd
import requests
from typing import List
from pydantic import BaseModel, Field
from e2b_code_interpreter import Sandbox
from together import Together



class DataVizInput(BaseModel):
    csv_path: str = Field(description="Absolute path to the CSV file")
    query: str = Field(description="Natural language question about the dataset")


class DataVizTool:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.sandbox = Sandbox()
        self.llm = Together()

    def _upload_dataset(self, local_path: str) -> str:
        with open(local_path, "rb") as f:
            return self.sandbox.files.write("dataset.csv", f).path

    def _generate_prompt(self, query: str, dataset_path_in_sandbox: str) -> str:
    # Leggi il dataset dal percorso LOCALE, non quello nella sandbox
    # perché pandas non può leggere dalla sandbox
    # Quindi: leggi una preview locale, solo per ottenere info sulle colonne
        local_df = pd.read_csv(self.temp_local_csv, nrows=300)
        column_info = local_df.dtypes.astype(str).to_dict()

        return f"""
You are a Python data scientist working in a sandboxed environment.

The dataset is already uploaded and available at the following absolute path: `{dataset_path_in_sandbox}`

Here are the columns of the dataset and their data types:
{column_info}

User question:
\"{query}\"

Instructions:
- Load the dataset using `pd.read_csv("{dataset_path_in_sandbox}")`
- Do not hardcode any other path.
- Use matplotlib or seaborn to generate visualizations as needed.
- Save all plots using `plt.savefig("chart-<anything>.png")` (do NOT use `plt.show()`)
- Output only Python code — no text, no explanations.
"""


    def _call_llm(self, prompt: str) -> str:
        response = self.llm.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are a helpful Python data scientist."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.4,
            top_p=0.9
        )

        return response.choices[0].message.content.strip()
    
    def _clean_code(self, code: str) -> str:
    # Rimuove blocchi markdown (```python ... ```)
        if code.startswith("```"):
            code = code.strip("```")
            if code.startswith("python"):
                code = code[len("python"):].lstrip()
        code = code.strip("```").strip()
        return code

    def _run_code(self, code: str) -> List[str]:
        print("@@@@@@@@@@@@@@@@@@@@@ ----------------------> Generated code:\n", code)

        execution = self.sandbox.run_code(code)
        if execution.error:
            raise RuntimeError(f"Code execution error: {execution.error.value}")

        saved_paths = []
        for i, result in enumerate(execution.results):
            if result.png:
                filename = f"{uuid.uuid4().hex}.png"
                file_path = os.path.join(self.output_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(base64.b64decode(result.png))
                saved_paths.append(os.path.abspath(file_path))

        if not saved_paths:
            raise RuntimeError("No charts were produced by the generated code.")
        return saved_paths

    def run(self, input: DataVizInput) -> dict:
        try:
            print("[*] Uploading dataset...")
            self.temp_local_csv = input.csv_path  # Salviamo per il prompt
            dataset_path_in_sandbox = self._upload_dataset(input.csv_path)

            print("[*] Generating code via LLM...")
            prompt = self._generate_prompt(input.query, dataset_path_in_sandbox)
            code = self._call_llm(prompt)

            print("[*] Cleaning generated code...")
            code = self._clean_code(code)

            print("[*] Executing code in sandbox...")
            image_paths = self._run_code(code)

            markdown_images = "\n".join([f"![Chart]({path})" for path in image_paths])

            return {
                "content": f"Ecco il grafico richiesto:\n\n{markdown_images}",
                "image_paths": image_paths
            }

        except Exception as e:
            raise RuntimeError(f"Visualization error: {e}")


data_viz_tool = StructuredTool.from_function(
    func=DataVizTool().run,
    name="data_viz_tool",
    description="Generates data visualizations from a CSV file and a natural language query.",
)









# ------------------------------------------------------------------------------
tools = [
    retrieve_tool,
    web_search_tool,
    arxive_tool,
    code_intertpreter,
    google_books_tool,
    google_scholar_tool,
    wikidata_tool,
    wikipedia_tool,
    wolfram_tool,
    youtube_search_tool,
    spotify_music_tool,
    text_to_speech_tool,
    speech_to_text_tool,
    google_lens_tool,           # per analizzare immagini da url
    # image_generation_tool,
    # image_interrogator_tool,    # per analizzare immagini caricate in locale
    # text_analysis_tool,       # funziona ma non so come trattare la risposta
    doc_summary_tool,
    sentiment_tool,
    extract_text_tool,
    # csv_hybrid_tool,
    data_viz_tool,
]  # + base_tool   # + o365_tools
