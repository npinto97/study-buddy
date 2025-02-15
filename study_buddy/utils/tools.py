import os
import fitz  # PyMuPDF per estrazione testo da PDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import requests
import tempfile
from openai import OpenAI
from typing import Union
from textblob import TextBlob

from pydantic import BaseModel, Field

from langchain_core.tools import tool, Tool
# from langchain_community.agent_toolkits.load_tools import load_tools

# from langchain_community.agent_toolkits import O365Toolkit
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
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
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from e2b_code_interpreter import Sandbox

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


@tool(response_format="content_and_artifact")
def retrieve_tool(query: str):
    """Retrieve information related to a query."""
    vector_store = get_vector_store(FAISS_INDEX_DIR)
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


web_search_tool = TavilySearchResults(
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


class QwenTextAnalysis:
    """Wrapper per l'analisi del testo tramite il modello di Hugging Face."""

    def __init__(self, api_url: str = "Qwen/Qwen2.5-Turbo-1M-Demo"):
        self.client = Client(api_url)

    def analyze_text(self, text: str, files: list = []):
        """Invia il testo e i file per l'analisi."""
        try:
            result = self.client.predict(
                _input={"files": files, "text": text},
                _chatbot=[],
                api_name="/add_text"
            )
            return result
        except Exception as e:
            return {"error": str(e)}


text_analysis_tool = Tool(
    name="text_analysis",
    description="Analyze text or files (pdf/docx/pptx/txt/html) using Qwen model.",
    func=QwenTextAnalysis().analyze_text
)


class DocumentSummarizerWrapper:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.documents = self.load_documents()

    def load_documents(self):
        """
        Carica il documento dal percorso fornito (supporta PDF e TXT).
        """
        if self.file_path.endswith(".pdf"):
            loader = PyPDFLoader(self.file_path)
        else:
            loader = TextLoader(self.file_path)

        return loader.load()

    def summarize_document(self) -> str:
        """
        Riassume il contenuto del documento utilizzando il modello di linguaggio.
        """
        # Inizializza il modello di linguaggio
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)

        # Crea la catena di riassunto
        chain = load_summarize_chain(llm, chain_type="map_reduce")

        # Esegui il riassunto
        summary = chain.invoke(self.documents)
        return summary


# Funzione di supporto per l'integrazione con LangChain
def summarize_document_from_wrapper(file_path: str) -> str:
    summarizer = DocumentSummarizerWrapper(file_path)
    return summarizer.summarize_document()


# Creazione del Tool per LangChain
summarize_tool = Tool(
    name="DocumentSummarizer",
    description="Riassume un documento testuale dato un file path (PDF o TXT)",
    func=summarize_document_from_wrapper  # Funzione che invoca il riassunto dal wrapper
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
    """Input schema for the Code Interpreter."""
    code: str = Field(description="Python code to execute.")


class CodeInterpreterFunctionTool:
    """Tool to execute Python code in a Jupyter environment using E2B."""

    tool_name: str = "code_interpreter"

    def __init__(self):
        self._initialize_sandbox()

    def _initialize_sandbox(self):
        """Initialize or restart the E2B sandbox."""
        if "E2B_API_KEY" not in os.environ:
            raise Exception("E2B API key is missing. Set the E2B_API_KEY environment variable.")
        self.code_interpreter = Sandbox()
        self.code_interpreter.set_timeout(60)  # Extend timeout to 60 seconds

    def close(self):
        """Kill the sandbox."""
        self.code_interpreter.kill()

    def call(self, parameters: dict):
        code = parameters.get("code", "")
        try:
            execution = self.code_interpreter.run_code(code)
            return {
                "results": execution.results,
                "stdout": execution.logs.stdout,
                "stderr": execution.logs.stderr,
                "error": execution.error,
            }
        except Exception as e:
            if "502 Bad Gateway" in str(e) or "sandbox timeout" in str(e).lower():
                print("Sandbox timed out. Restarting...")
                self._initialize_sandbox()  # Restart the sandbox
                return self.call(parameters)  # Retry execution
            else:
                raise e

    def langchain_call(self, code: str):
        return self.call({"code": code})

    def to_langchain_tool(self) -> Tool:
        return Tool(
            name=self.tool_name,
            description="Executes Python code and returns stdout, stderr, and results.",
            func=self.langchain_call,
            args_schema=CodeInterpreterInput
        )


execute_python_tool = CodeInterpreterFunctionTool().to_langchain_tool()


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


# Convert audio (lectures, meetings) into text
class OpenAISpeechToText:
    """Wrapper for the OpenAI Whisper API to transcribe or translate audio from files, paths, or URLs."""

    def __init__(self):
        self.client = OpenAI()

    def _download_audio(self, audio_url: str) -> str:
        """Download a temporary audio file from a URL and return the local path."""
        response = requests.get(audio_url, stream=True)
        if response.status_code != 200:
            raise ValueError("Error downloading the audio file.")

        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                temp_audio.write(chunk)
        temp_audio.close()
        return temp_audio.name

    def _get_audio_path(self, audio_input: Union[str, bytes]) -> str:
        """Determine the audio file path based on the input type."""
        print(f"Ricevuto input audio: {audio_input}")
        if isinstance(audio_input, str):
            if audio_input.startswith("http"):  # Remote URL
                return self._download_audio(audio_input)
            elif os.path.exists(audio_input):  # Local path
                print(f"Path audio esistente: {audio_input}")
                return audio_input
            else:
                raise ValueError("The audio file path does not exist.")
        elif isinstance(audio_input, bytes):  # File uploaded as binary object
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_audio.write(audio_input)
            temp_audio.close()
            return temp_audio.name
        elif isinstance(audio_input, tempfile.NamedTemporaryFile):
            # Ritorna il percorso del file temporaneo esistente
            return audio_input.name
        else:
            raise TypeError("Invalid input. Provide a URL, file path, or binary file.")

    def transcribe_audio(self, audio_input: Union[str, bytes], task: str = "transcribe") -> str:
        """Transcribe or translate a local audio file, URL, or binary file."""
        audio_path = self._get_audio_path(audio_input)

        with open(audio_path, "rb") as audio_file:
            if task == "translate":
                response = self.client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file
                )
            else:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

        os.remove(audio_path)  # Delete the temporary file if created
        return response.text  # Return the transcribed/translated text


speech_to_text_tool = Tool(
    name="speech_to_text",
    description="Trascrive o traduce audio da file, percorsi locali o URL utilizzando OpenAI Whisper.",
    func=OpenAISpeechToText().transcribe_audio,
)


# Convert text into spoken voice
class OpenAITTSWrapper:
    """Wrapper for OpenAI's Text-to-Speech API."""

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def text_to_speech(self, text: str, model: str = "tts-1", voice: str = "sage", output_filename: str = "speech.mp3"):
        """Converts text to speech and saves the audio file in a temporary dictionary."""

        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_path = temp_audio_file.name

        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
        )
        with open(temp_audio_path, "wb") as audio_file:
            audio_file.write(response.content)

        return str(temp_audio_path)


# Instantiate the wrapper
tts_wrapper = OpenAITTSWrapper()

# Define the tool
tts_tool = Tool(
    name="text_to_speech",
    description="Converts text to speech using OpenAI's TTS API and returns an audio file.",
    func=tts_wrapper.text_to_speech,
)


# ------------------------------------------------------------------------------
# Advanced tools (multimodal, ...)

google_lens_tool = GoogleLensQueryRun(api_wrapper=GoogleLensAPIWrapper())


class ImageGenerationAPIWrapper:
    def __init__(self, model_name: str):
        self.client = Client(model_name)

    def generate_image(self, prompt: str, seed: int = 0, randomize_seed: bool = True, width: int = 100, height: int = 100, num_inference_steps: int = 2):
        """Generates an image from a prompt using the Hugging Face API."""
        result = self.client.predict(
            prompt=prompt,
            seed=seed,
            randomize_seed=randomize_seed,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            api_name="/infer"
        )
        return result  # This could be an image URL or file path depending on the API response


image_generation_wrapper = ImageGenerationAPIWrapper("black-forest-labs/FLUX.1-schnell")

image_generation_tool = Tool(
    name="image_generator",
    description="Generates an image following a given prompt and returns the result.",
    func=image_generation_wrapper.generate_image
)


class CLIPInterrogatorAPIWrapper:
    def __init__(self, api_url: str):
        self.client = Client(api_url)

    def interrogate_image(self, image_url: str, model: str = "ViT-L (best for Stable Diffusion 1.*)", mode: str = "classic"):
        """Interrogate the image to get information using CLIP-Interrogator."""
        result = self.client.predict(
            image_url,         # Image URL
            model,             # Model to use
            mode,              # Mode ('best', 'fast', 'classic', 'negative')
            fn_index=3         # Function index for image interrogation
        )
        return result


image_interrogator_wrapper = CLIPInterrogatorAPIWrapper("https://pharmapsychotic-clip-interrogator.hf.space/")

image_interrogator_tool = Tool(
    name="image_interrogator",
    description="Interrogate an image and return artistic information, movement, and more.",
    func=image_interrogator_wrapper.interrogate_image
)

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


tools = [
    retrieve_tool,
    web_search_tool,
    arxive_tool,
    execute_python_tool,
    google_books_tool,
    google_scholar_tool,
    wikidata_tool,
    wikipedia_tool,
    wolfram_tool,
    youtube_search_tool,
    spotify_music_tool,
    tts_tool,
    speech_to_text_tool,
    google_lens_tool,           # per analizzare immagini da url
    image_generation_tool,
    image_interrogator_tool,  # per analizzare immagini caricate in locale
    # text_analysis_tool,       # funziona ma non so come trattare la risposta
    summarize_tool,
    sentiment_tool,
    extract_text_tool
]  # + base_tool   # + o365_tools
