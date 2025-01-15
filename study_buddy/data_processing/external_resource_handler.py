import requests
from bs4 import BeautifulSoup


# Function for extracting metadata from a URL
def extract_metadata(url):
    """
    Estrae i metadati da una pagina web a partire da un URL.

    :param url: URL della risorsa da cui estrarre i metadati
    :return: Dizionario contenente i metadati
    """
    pass


# Function for extracting textual information in .txt format from content of type
# 'article', 'model', 'blog', 'paper' and 'announcement'
def extract_text_content(url):
    """
    Estrae il contenuto testuale da risorse come ARTICLE, MODEL, BLOG, PAPER, ANNOUNCEMENT.

    :param url: URL della risorsa da cui estrarre il contenuto
    :return: Stringa contenente il testo estratto
    """
    pass


# Function for extracting a .txt file from a repository's README.md
def extract_readme_from_repo(repo_url):
    """
    Estrae il contenuto del file README.md da una repository GitHub.

    :param repo_url: URL della repository GitHub
    :return: Stringa contenente il contenuto del README.md
    """
    pass


# Function for extracting information from metadata.json for content of type 'dataset' and 'website'
def extract_metadata_json(resource_url):
    """
    Estrae le informazioni da un file metadata.json associato alla risorsa.

    :param resource_url: URL della risorsa contenente il metadata.json
    :return: Dizionario contenente i dati estratti
    """
    pass
