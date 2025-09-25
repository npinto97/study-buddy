from langchain_community.tools.google_lens import GoogleLensQueryRun
from langchain_community.utilities.google_lens import GoogleLensAPIWrapper

tool = GoogleLensQueryRun(api_wrapper=GoogleLensAPIWrapper())

try:
    result = tool.run("https://i.imgur.com/HBrB8p0.png")
    print(f"Risultato grezzo dell'API: {result}")

    if isinstance(result, dict):
        print(f"Chiavi disponibili: {result.keys()}")

    if 'visual_matches' in result:
        print(f"Google Lens OK! Risultato: {result['visual_matches']}")
    else:
        print("Errore: la chiave 'visual_matches' non è presente nella risposta dell'API.")
except Exception as e:
    print(f"Errore: {e}")
