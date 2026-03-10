"""OCR module: extracts text from scanned PDF pages."""

import io
import logging
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from config import OCR_DPI, TESSERACT_CONFIG

logger = logging.getLogger(__name__)


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """Extract text from each page of a PDF using OCR.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        List of dicts with keys 'page_number' and 'text'.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text().strip()

        # Fall back to OCR if no embedded text is found
        if not text:
            pix = page.get_pixmap(dpi=OCR_DPI)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)

        pages.append({
            "page_number": page_num + 1,
            "text": text.strip(),
        })
        logger.info("Page %d: extracted %d characters", page_num + 1, len(text))

    doc.close()
    return pages
