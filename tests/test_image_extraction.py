import fitz  # pymupdf
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import os

# Percorso di Tesseract (solo per Windows, altrimenti commenta)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_pdf(pdf_path):
    """Estrazione del testo da un PDF con testo selezionabile (pymupdf)"""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])

    if text.strip():
        print("Testo estratto dal PDF (pymupdf):")
        return text
    else:
        print("PDF scansionato, avvio OCR...")
        return extract_text_from_scanned_pdf(pdf_path)


def extract_text_from_scanned_pdf(pdf_path):
    """Estrazione del testo da un PDF scansionato (OCR pytesseract)"""
    images = convert_from_path(pdf_path)
    text = "\n".join([pytesseract.image_to_string(img) for img in images])
    return text


def extract_text_from_image(image_path):
    """Estrazione del testo da un'immagine (OCR pytesseract)"""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text


def extract_figures_from_image(image_path, output_folder="figures"):
    """Estrazione di figure da un'immagine, evitando frammenti di testo e immagini spezzate."""
    os.makedirs(output_folder, exist_ok=True)

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Rileva aree di testo con OCR
    h, w, _ = image.shape
    text_boxes = pytesseract.image_to_boxes(gray)
    text_regions = []

    for b in text_boxes.splitlines():
        b = b.split()
        x, y, x2, y2 = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
        text_regions.append((x, y, x2, y2))

    # Applica un filtro di contrasto per migliorare la separazione delle figure
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilatazione per unire contorni vicini (evita frammenti)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    figure_count = 0
    saved_regions = []  # Per evitare duplicati

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Esclude contorni troppo piccoli
        if w * h < 5000:  # Regola questa soglia se necessario
            continue

        # Esclude le aree di testo
        is_text = any(x >= tx and y >= ty and x + w <= tx2 and y + h <= y2 for (tx, ty, tx2, y2) in text_regions)
        if is_text:
            continue

        # Evita di salvare frammenti della stessa figura
        too_similar = any(abs(x - sx) < 30 and abs(y - sy) < 30 for (sx, sy, sw, sh) in saved_regions)
        if too_similar:
            continue

        # Salva la figura
        figure = image[y: y + h, x: x + w]
        figure_path = f"{output_folder}/figure_{figure_count}.png"
        cv2.imwrite(figure_path, figure)
        print(f"Figura salvata: {figure_path}")

        saved_regions.append((x, y, w, h))
        figure_count += 1


def extract_figures_from_pdf(pdf_path, output_folder="figures"):
    """Estrazione di figure da un PDF convertendo le pagine in immagini"""
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path)

    for i, img in enumerate(images):
        image_path = f"{output_folder}/page_{i + 1}.png"
        img.save(image_path, "PNG")
        extract_figures_from_image(image_path, output_folder)


def process_file(file_path):
    """Riconosce il tipo di file ed estrae testo e figure di conseguenza"""
    if file_path.lower().endswith(".pdf"):
        print(f"Elaborazione PDF: {file_path}")
        text = extract_text_from_pdf(file_path)
        extract_figures_from_pdf(file_path)
    elif file_path.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
        print(f"Elaborazione immagine: {file_path}")
        text = extract_text_from_image(file_path)
        extract_figures_from_image(file_path)
    else:
        print("Formato non supportato")
        return

    print("\nTesto Estratto:\n", text)


# **Esempio di utilizzo**
# file_path = "C:\\Users\\Ningo\\Desktop\\book_sample.pdf"
file_path = "C:\\Users\\Ningo\\Desktop\\sample_ocr.jpg"

process_file(file_path)
