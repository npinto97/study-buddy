from gradio_client import Client


class CLIPInterrogatorAPIWrapper:
    def __init__(self, api_url: str):
        self.client = Client(api_url)

    def interrogate_image(self, image_url: str, model: str = "ViT-L (best for Stable Diffusion 1.*)", mode: str = "best"):
        """Interroga l'immagine usando CLIP-Interrogator."""
        try:
            result = self.client.predict(
                image_url,  # URL o percorso dell'immagine
                model,      # Modello CLIP
                mode,       # Modalità ('best', 'fast', 'classic', 'negative')
                fn_index=3  # Index corretto per la funzione
            )
            return result
        except Exception as e:
            return f"Errore durante la chiamata API: {str(e)}"


# Creazione dell'istanza con il giusto endpoint
image_interrogator_wrapper = CLIPInterrogatorAPIWrapper("https://pharmapsychotic-clip-interrogator.hf.space/")

# Test
test_file_path = "C:/Users/Ningo/Desktop/bird.jpg"
try:
    result = image_interrogator_wrapper.interrogate_image(test_file_path)
    print(f"Immagine OK! Risultato: {result}")
except Exception as e:
    print(f"Errore: {e}")


# from gradio_client import Client

# client = Client("https://pharmapsychotic-clip-interrogator.hf.space/")
# result = client.predict(
#     "C:/Users/Ningo/Desktop/sample_image.jpg",	# str (filepath or URL to image) in 'Image' Image component
#     "ViT-L (best for Stable Diffusion 1.*)",	# str (Option from: ['ViT-L (best for Stable Diffusion 1.*)']) in 'CLIP Model' Dropdown component
#     fn_index=1
# )
# print(result)


# from gradio_client import Client

# client = Client("https://pharmapsychotic-clip-interrogator.hf.space/")
# result = client.predict(
#     "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",	# str (filepath or URL to image) in 'parameter_5' Image component
#     "ViT-L (best for Stable Diffusion 1.*)",	# str (Option from: ['ViT-L (best for Stable Diffusion 1.*)']) in 'CLIP Model' Dropdown component
#     "best",	 # str in 'Mode' Radio component
#     fn_index=3
# )
# print(result)
