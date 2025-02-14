from study_buddy.utils.tools import image_interrogator_tool

print("Test manuale di clip interrogator...")
test_file_path = "C:/Users/Ningo/Desktop/bird.jpg"
try:
    result = image_interrogator_tool.func(test_file_path)
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