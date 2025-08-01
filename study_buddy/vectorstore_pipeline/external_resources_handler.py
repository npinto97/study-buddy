import requests
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
from langchain.schema import Document
from study_buddy.config import logger


def extract_text_from_url(url):
    """Scarica e restituisce il testo da una pagina web."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        logger.error(f"Errore nell'estrazione del testo da {url}: {e}")
        return None


def extract_readme_from_repo(repo_url):
    """Estrae il README.md da una repository GitHub."""
    try:
        if repo_url.endswith(".git"):
            repo_url = repo_url[:-4]
        readme_url = f"{repo_url}/raw/main/README.md"
        response = requests.get(readme_url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Errore nel recupero del README da {repo_url}: {e}")
        return None


def extract_transcript_from_youtube(youtube_url):
    """Ottiene la trascrizione automatica di un video YouTube."""
    try:
        video_id = youtube_url.split("v=")[-1].split("&")[0]
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        logger.error(f"Errore nella trascrizione del video {youtube_url}: {e}")
        return None


def extract_external_resources(lesson_data):
    """Estrae e restituisce documenti strutturati dalle risorse esterne di una lezione."""
    resources = lesson_data.get("resources", [])
    extracted_docs = []

    for resource in resources:
        url = resource.get("url")
        title = resource.get("title", "Risorsa Esterna")  # Nome predefinito se non specificato
        content = None

        if "github.com" in url:
            content = extract_readme_from_repo(url)
        elif "youtube.com" in url or "youtu.be" in url:
            content = extract_transcript_from_youtube(url)
        else:
            content = extract_text_from_url(url)

        if content:
            document = Document(
                page_content=content,
                metadata={"source": url, "title": title}
            )
            extracted_docs.append(document)

    return extracted_docs
