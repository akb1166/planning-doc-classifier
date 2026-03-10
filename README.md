# Planning Document Classifier

Classifies pages of UK planning decision PDFs into document-type categories and extracts application numbers and applicant names using NLP.

## Approach

| Task | Method | Tool |
|---|---|---|
| Text extraction | OCR on scanned PDF pages | Tesseract via `pytesseract` + `PyMuPDF` |
| Page classification | Zero-shot classification with keyword fallback | `typeform/distilbert-base-uncased-mnli` (Hugging Face) |
| Application number extraction | Regex pattern cascade | `re` (stdlib) |
| Applicant name extraction | Named Entity Recognition + contextual regex | `spaCy` (`en_core_web_sm`) |

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd planning-doc-classifier

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the spaCy language model
python -m spacy download en_core_web_sm

# 5. Install Tesseract OCR (if not already present)
# Ubuntu/Debian:  sudo apt install tesseract-ocr
# macOS:          brew install tesseract
# Windows:        https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

Place the input PDF in the `data/` directory, then run:

```bash
python pipeline.py data/anonymised_1.pdf
```

Optional arguments:

```bash
python pipeline.py data/anonymised_1.pdf --output output/my_results.json
```

Results are saved to `output/` as both JSON and CSV.

## Project Structure

```
planning-doc-classifier/
├── pipeline.py            # Main entry point
├── config.py              # Configuration constants
├── requirements.txt       # Python dependencies
├── analysis_report.md     # Brief analysis report
├── src/
│   ├── ocr.py             # PDF → text via OCR
│   ├── classifier.py      # Zero-shot + keyword classification
│   └── extractor.py       # Application number + name extraction
├── data/                  # Place input PDFs here
└── output/                # Pipeline outputs (JSON, CSV)
```

## Output Format

Each page produces a result entry:

```json
{
  "page_number": 1,
  "classification": "Planning Charges Register Entry",
  "confidence": 0.87,
  "method": "zero-shot",
  "application_numbers": ["02/80/1609", "02/81/1237"],
  "applicant_names": ["Mr. & Mrs. J. M Doe"]
}
```

## Dependencies

- Python 3.10+
- Tesseract OCR 4.0+
- See `requirements.txt` for Python packages
