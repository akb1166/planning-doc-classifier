"""Configuration constants for the planning document classifier."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# OCR settings
OCR_DPI = 300
TESSERACT_CONFIG = "--oem 3 --psm 6"

# Classification candidate labels (determined from EDA of planning documents)
CANDIDATE_LABELS = [
    "Planning Permission Decision Notice",
    "Grant of Conditional Planning Permission",
    "Approval of Reserved Matters or Details",
    "Planning Charges Register Entry",
    "Refusal of Planning Permission",
    "Appeal Decision Notice",
    "Certificate of Lawful Use or Development",
    "Environmental Impact Assessment",
]

# Zero-shot classification model
CLASSIFICATION_MODEL = "typeform/distilbert-base-uncased-mnli"
CLASSIFICATION_THRESHOLD = 0.3

# NER model
NER_MODEL = "en_core_web_sm"

# Regex patterns for application number extraction (ordered most → least specific)
APPLICATION_NUMBER_PATTERNS = [
    r"(?:Application\s+(?:No\.?|Number)\s*[:\s])\s*([A-Z]?/?\d{2,4}/\d{3,5}(?:/\w+)?)",
    r"(?:Ref(?:erence)?\s*(?:No\.?|Number)?\s*[:\s])\s*([A-Z]?/?\d{2,4}/\d{3,5}(?:/\w+)?)",
    r"(?:Our\s+Ref\s*[:\s])\s*([A-Z]?/?\d{2,4}/\d{3,5}(?:/\w+)?)",
    r"\b([A-Z]/\d{2,4}/\d{3,5})\b",
    r"\b(\d{2}/\d{2}/\d{4,5})\b",
]
