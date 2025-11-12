import os
from pdf2image import convert_from_path
from PIL import Image

pdf_path = "C:\\Users\\dkz\\Downloads\\pltest2.pdf"
Image.MAX_IMAGE_PIXELS = None

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

pages = convert_from_path(pdf_path, dpi=200)

for i, page in enumerate(pages):
    page.save(f"W:\\competition\\model\\images\\train\\pltest2_{i}.jpg", "JPEG")
