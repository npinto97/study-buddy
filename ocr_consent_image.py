
try:
    from PIL import Image
    import pytesseract
    import os

    # Set tesseract path if needed, but usually it's in PATH or handled by the env
    # If this fails, I'll try to just read it conceptually or ask user, but let's try OCR first.
    
    img_path = r"C:/Users/Utente/.gemini/antigravity/brain/0be29558-667a-4829-988c-82db867315be/uploaded_media_1769784475435.png"
    
    if not os.path.exists(img_path):
        print(f"Error: File not found at {img_path}")
        exit(1)

    image = Image.open(img_path)
    text = pytesseract.image_to_string(image, lang='ita')
    print("--- EXTRACTED TEXT ---")
    print(text)
    print("--- END TEXT ---")

except Exception as e:
    print(f"Error: {e}")
