# import os
# import requests
# from xml.etree import ElementTree

# APPID = "Q3XVG2-TVL3VE2ARX"
# query = "2x+5=-3x+7"
# url = f"http://api.wolframalpha.com/v2/query?input={query}&appid={APPID}"

# resp = requests.get(url)
# xml_root = ElementTree.fromstring(resp.text)

# # Prendi il primo pod con il risultato
# result_pod = xml_root.find(".//pod[@title='Result']/subpod/plaintext")
# if result_pod is not None:
#     print("Risultato:", result_pod.text)
# else:
#     print("Nessun risultato trovato")


import os
import requests
from xml.etree import ElementTree
from langchain.tools import Tool
from dotenv import load_dotenv

# Carica variabili dal .env
load_dotenv()
APPID = os.getenv("WOLFRAM_ALPHA_APPID")

from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.agent_toolkits.load_tools import load_tools


def wolfram_query(query: str) -> str:
    url = f"http://api.wolframalpha.com/v2/query?input={query}&appid={APPID}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return f"Errore nella richiesta: {resp.status_code}"
    
    root = ElementTree.fromstring(resp.text)
    result_pod = root.find(".//pod[@title='Result']/subpod/plaintext")
    
    if result_pod is not None and result_pod.text:
        return result_pod.text
    else:
        return "Nessun risultato trovato"

wolfram_tool = Tool(
    name="WolframAlpha",
    description="Esegue query su WolframAlpha",
    func=wolfram_query
)

print(wolfram_tool.func("2x+5=-3x+7"))
