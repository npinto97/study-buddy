import os
import requests
from openai import OpenAI
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.tools import tool, Tool
from langchain_community.agent_toolkits.load_tools import load_tools

# from langchain_community.agent_toolkits import O365Toolkit

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

from study_buddy.utils.vector_store import get_vector_store
from study_buddy.config import FAISS_INDEX_DIR

google_api_key = os.getenv("GOOGLE_API_KEY")


# ------------------------------------------------------------------------------
# Basic tools
base_tool = load_tools(["human", "pubmed"])


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
            summary = info.get("description", "No description available")  # ✅ FIXED
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

# ------------------------------------------------------------------------------
# Scientific research support tools
arxive_tool = ArxivQueryRun()

google_scholar_tool = GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper())


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


class MoodBasedMusicRecommender:
    """Suggests music based on the user's mood."""

    MOOD_TO_GENRE = {
        "happy": "pop",
        "relaxed": "chill",
        "focused": "lo-fi",
        "motivated": "workout",
        "sad": "blues",
        "stressed": "ambient",
    }

    def __init__(self):
        self.spotify_api = SpotifyAPIWrapper()

    def recommend_music(self, mood: str):
        """Suggests music based on the mood."""
        genre = self.MOOD_TO_GENRE.get(mood.lower(), "pop")
        return self.spotify_api.search_music(query=genre, search_type="playlist", limit=3)


# TODO: non funziona
mood_music_tool = Tool(
    name="Mood-Based_music",
    description="Suggests music based on the user's mood (e.g., happy, relaxed, focused).",
    func=MoodBasedMusicRecommender().recommend_music
)


# ------------------------------------------------------------------------------
# Tools for accessibility and inclusivity


# Convert audio (lectures, meetings) into text
class WhisperAPIWrapper:
    """Wrapper for OpenAI Whisper API via Hugging Face for speech-to-text transcription and translation."""

    def __init__(self, api_url: str = "https://openai-whisper.hf.space/"):
        self.client = Client(api_url)

    def transcribe_audio(self, audio_url: str, task: Literal["transcribe", "translate"] = "transcribe"):
        """Transcribes or translates audio from a file, URL, or YouTube link."""
        result = self.client.predict(audio_url, task, api_name="/predict")
        return result  # Returns transcribed or translated text


whisper_wrapper = WhisperAPIWrapper()

whisper_tool = Tool(
    name="speech_to_text",
    description="Transcribes or translates audio from files, URLs, or YouTube videos using OpenAI Whisper.",
    func=whisper_wrapper.transcribe_audio,
)


# Convert text into spoken voice
class OpenAITTSWrapper:
    """Wrapper for OpenAI's Text-to-Speech API."""

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def text_to_speech(self, text: str, model: str = "tts-1", voice: str = "sage", output_filename: str = "speech.mp3"):
        """Converts text to speech and saves the audio file."""
        output_path = Path(__file__).parent / output_filename
        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
        )
        with open(output_path, "wb") as audio_file:
            audio_file.write(response.content)

        return str(output_path)


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


# image_generation_tool = StableDiffusionTool().langchain
# image_caption_tool = ImageCaptioningTool().langchain
# improve_prompt_tool = StableDiffusionPromptGeneratorTool().langchain
# video_generator_tool = TextToVideoTool().langchain


google_lens_tool = GoogleLensQueryRun(api_wrapper=GoogleLensAPIWrapper())


class ImageGenerationAPIWrapper:
    def __init__(self, model_name: str):
        self.client = Client(model_name)

    def generate_image(self, prompt: str, seed: int = 0, randomize_seed: bool = True, width: int = 1024, height: int = 1024, num_inference_steps: int = 4):
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

    def interrogate_image(self, image_url: str, model: str = "ViT-L (best for Stable Diffusion 1.*)", mode: str = "best"):
        """Interrogate the image to get information using CLIP-Interrogator."""
        result = self.client.predict(
            image_url,         # Image URL
            model,             # Model to use
            mode,              # Mode ('best', 'fast', 'classic', 'negative')
            fn_index=3         # Function index for image interrogation
        )
        return result


clip_interrogator_wrapper = CLIPInterrogatorAPIWrapper("https://pharmapsychotic-clip-interrogator.hf.space/")

clip_interrogator_tool = Tool(
    name="clip_interrogator",
    description="Interrogate an image and return artistic information, movement, and more.",
    func=clip_interrogator_wrapper.interrogate_image  # The function from the wrapper
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
    mood_music_tool,
    tts_tool,
    whisper_tool,
    google_lens_tool,
    image_generation_tool,
    clip_interrogator_tool
] + base_tool   # + o365_tools
