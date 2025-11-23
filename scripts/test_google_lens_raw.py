#!/usr/bin/env python3
"""
Test per Google Lens con supporto upload automatico a imgbb.

Modalità:
1. URL pubblico: testa direttamente con Google Lens
2. File locale + IMGBB_API_KEY: carica su imgbb, poi analizza con Google Lens
3. File locale senza IMGBB_API_KEY: mostra solo cosa farebbe OCR fallback

Esempi:
    # Test con URL pubblico
    python scripts/test_google_lens_raw.py "https://example.com/image.jpg" --url
    
    # Test upload + Google Lens
    python scripts/test_google_lens_raw.py uploaded_files/84.jpg
    
    # Test solo upload (senza Google Lens)
    python scripts/test_google_lens_raw.py uploaded_files/84.jpg --upload-only
"""
import os
import sys
import json
import base64
import requests
from datetime import datetime


def upload_to_imgbb(image_path, api_key):
    """Upload immagine a imgbb e restituisce URL pubblico."""
    if not os.path.exists(image_path):
        print(f"ERROR: File non trovato: {image_path}")
        return None
    
    try:
        # Leggi e codifica in base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Upload
        url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": api_key,
            "image": image_data,
            "expiration": 600  # 10 minuti
        }
        
        print(f"Uploading a imgbb: {os.path.basename(image_path)}...")
        response = requests.post(url, data=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if result.get("success"):
            public_url = result["data"]["url"]
            delete_url = result["data"]["delete_url"]
            print(f"✓ Upload completato!")
            print(f"  Public URL: {public_url}")
            print(f"  Delete URL: {delete_url}")
            print(f"  Scadenza: 10 minuti")
            return public_url
        else:
            print(f"✗ Upload fallito: {result}")
            return None
            
    except Exception as e:
        print(f"✗ Errore upload: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_google_lens_api(api_key, image_source, is_url=False):
    """
    Testa direttamente l'API SERP per Google Lens.
    Documentazione: https://serpapi.com/google-lens-api
    
    Args:
        api_key: Chiave API SERP
        image_source: Path locale o URL pubblico dell'immagine
        is_url: True se image_source è un URL pubblico, False se è un file locale
    """
    print(f"Testing Google Lens API...")
    print(f"API Key: {api_key[:10]}...")
    
    # Endpoint SERP API per Google Lens
    url = "https://serpapi.com/search"
    
    if is_url:
        # URL pubblico - usa direttamente
        image_url = image_source
        print(f"URL pubblico: {image_url}")
    else:
        # File locale - converti in path assoluto con file:///
        abs_path = os.path.abspath(image_source)
        image_url = f"file:///{abs_path}"
        print(f"File locale: {abs_path}")
        print(f"File URL: {image_url}")
    
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": api_key
    }
    
    print(f"\nFacendo richiesta a SERP API...")
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        print("\n✓ Risposta ricevuta!")
        print(f"Status code: {response.status_code}")
        print(f"Response keys: {list(result.keys())}")
        
        # Salva il risultato
        dump_dir = os.path.join(os.getcwd(), "logs", "tool_call_dumps")
        os.makedirs(dump_dir, exist_ok=True)
        
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        dump_path = os.path.join(dump_dir, f"google_lens_raw_{ts}.json")
        
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Risultato completo salvato in: {dump_path}")
        
        # Mostra un'anteprima
        print("\n--- Anteprima risultato ---")
        print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"\n ERROR: Errore nella richiesta HTTP: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text[:500]}")
        return None
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Google Lens API con upload automatico')
    parser.add_argument('image', nargs='?', default='uploaded_files/84.jpg', 
                       help='Path locale o URL pubblico dell\'immagine')
    parser.add_argument('--url', '-u', action='store_true',
                       help='Tratta il parametro come URL pubblico (salta upload)')
    parser.add_argument('--upload-only', action='store_true',
                       help='Testa solo l\'upload a imgbb, senza Google Lens')
    args = parser.parse_args()
    
    # Leggi le chiavi dall'ambiente
    serp_key = os.getenv("GOOGLE_LENS_API_KEY") or os.getenv("SERP_API_KEY")
    imgbb_key = os.getenv("IMGBB_API_KEY")
    
    if not serp_key and not args.upload_only:
        # Usa chiave hardcoded per test Google Lens
        serp_key = "fd860eb3de5e6daf87ba2ea16736d083a65d28cb5b24d1b121b09e98b2cadf29"
        print("WARNING: Usando chiave SERP hardcoded per test")
    
    # Determina se è URL o file locale
    is_url = args.url or args.image.startswith('http://') or args.image.startswith('https://')
    
    if is_url:
        # URL pubblico - usa direttamente
        print("Modalità: URL pubblico")
        image_url = args.image
    else:
        # File locale - verifica esistenza
        print("Modalità: File locale")
        if not os.path.exists(args.image):
            print(f"ERROR: File non trovato: {args.image}")
            sys.exit(1)
        
        if args.upload_only:
            # Test solo upload
            if not imgbb_key:
                print("ERROR: IMGBB_API_KEY non configurata")
                print("Imposta IMGBB_API_KEY nel file .env oppure:")
                print("  export IMGBB_API_KEY=your_key_here  # Linux/Mac")
                print("  $env:IMGBB_API_KEY='your_key_here'  # PowerShell")
                sys.exit(1)
            
            public_url = upload_to_imgbb(args.image, imgbb_key)
            if public_url:
                print("\n✓ SUCCESS: Upload completato!")
                sys.exit(0)
            else:
                print("\n✗ FAILED: Upload fallito")
                sys.exit(1)
        
        # Upload per Google Lens
        if imgbb_key:
            image_url = upload_to_imgbb(args.image, imgbb_key)
            if not image_url:
                print("\n✗ FAILED: Upload fallito, impossibile procedere con Google Lens")
                sys.exit(1)
        else:
            print("\nWARNING: IMGBB_API_KEY non configurata")
            print("Google Lens richiede URL pubblico. Per file locali serve upload automatico.")
            print("Configura IMGBB_API_KEY nel file .env per abilitare questa funzionalità.")
            print("\nPer ottenere la chiave:")
            print("  1. Vai su https://api.imgbb.com/")
            print("  2. Crea account gratuito")
            print("  3. Copia la API key")
            print("  4. Aggiungi IMGBB_API_KEY=your_key nel file .env")
            sys.exit(1)
    
    # Testa Google Lens
    if not serp_key:
        print("ERROR: GOOGLE_LENS_API_KEY o SERP_API_KEY non configurata")
        sys.exit(1)
    
    result = test_google_lens_api(serp_key, image_url, is_url=True)
    
    if result:
        print("\n✓ SUCCESS: Test completato!")
        sys.exit(0)
    else:
        print("\n✗ FAILED: Test fallito")
        sys.exit(1)
