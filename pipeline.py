"""
Planning Document Classifier
=============================
Classifies pages of UK planning decision PDFs and extracts
application numbers and applicant names.

Usage:
    python pipeline.py data/anonymised_1.pdf
    python pipeline.py data/anonymised_1.pdf --no-gpu --output output/results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from src.ocr import extract_pages
from src.classifier import classify_page, load_zero_shot_pipeline
from src.extractor import extract_entities, load_ner_model
from config import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def run_pipeline(
    pdf_path: str,
    output_path: str | None = None,
    skip_ner: bool = False,
) -> list[dict]:
    """Run the full classification and extraction pipeline.

    Args:
        pdf_path: Path to the input PDF.
        output_path: Optional path for the JSON results file.
        skip_ner: If True, skip spaCy NER and use regex-only for names.

    Returns:
        List of result dicts, one per page.
    """
    pdf_path = Path(pdf_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = OUTPUT_DIR / f"{pdf_path.stem}_results.json"
    else:
        output_path = Path(output_path)

    # ------------------------------------------------------------------
    # Step 1: OCR
    # ------------------------------------------------------------------
    logger.info("Extracting text from %s", pdf_path.name)
    pages = extract_pages(pdf_path)
    logger.info("Extracted text from %d pages", len(pages))

    # ------------------------------------------------------------------
    # Step 2: Load models
    # ------------------------------------------------------------------
    logger.info("Loading classification model...")
    try:
        zs_pipeline = load_zero_shot_pipeline()
    except Exception as e:
        logger.error("Could not load zero-shot model: %s", e)
        raise

    logger.info("Loading NER model...")
    if skip_ner:
        logger.info("NER skipped (--no-ner flag). Using regex-only for name extraction.")
        nlp = None
    else:
        try:
            nlp = load_ner_model()
        except Exception as e:
            logger.warning("Could not load NER model (%s). Using regex only.", e)
            nlp = None

    # ------------------------------------------------------------------
    # Step 3: Classify and extract
    # ------------------------------------------------------------------
    results = []
    for page in pages:
        page_num = page["page_number"]
        text = page["text"]

        classification = classify_page(text, zs_pipeline=zs_pipeline)
        entities = extract_entities(text, nlp=nlp)

        result = {
            "page_number": page_num,
            "classification": classification["label"],
            "confidence": classification["score"],
            "method": classification["method"],
            "application_numbers": entities["application_numbers"],
            "applicant_names": entities["applicant_names"],
        }
        results.append(result)

        logger.info(
            "Page %d → %s (%.2f, %s) | Apps: %s | Names: %s",
            page_num,
            result["classification"],
            result["confidence"] if isinstance(result["confidence"], float) else result["confidence"],
            result["method"],
            result["application_numbers"],
            result["applicant_names"],
        )

    # ------------------------------------------------------------------
    # Step 4: Save outputs
    # ------------------------------------------------------------------
    # JSON output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)

    # CSV summary
    csv_path = output_path.with_suffix(".csv")
    df = pd.DataFrame([
        {
            "page": r["page_number"],
            "classification": r["classification"],
            "confidence": r["confidence"],
            "method": r["method"],
            "application_numbers": "; ".join(r["application_numbers"]),
            "applicant_names": "; ".join(r["applicant_names"]),
        }
        for r in results
    ])
    df.to_csv(csv_path, index=False)
    logger.info("CSV summary saved to %s", csv_path)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Classify planning decision PDF pages and extract entities.",
    )
    parser.add_argument("pdf", help="Path to the input PDF file.")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path for the output JSON file (default: output/<stem>_results.json).",
    )
    parser.add_argument(
        "--no-ner",
        action="store_true",
        help="Skip spaCy NER model loading (use regex-only for name extraction).",
    )
    args = parser.parse_args()

    results = run_pipeline(
        args.pdf, args.output,
        skip_ner=args.no_ner,
    )

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Page':<6} {'Classification':<45} {'App Numbers':<20} {'Names'}")
    print("-" * 80)
    for r in results:
        apps = "; ".join(r["application_numbers"]) or "—"
        names = "; ".join(r["applicant_names"]) or "—"
        print(f"{r['page_number']:<6} {r['classification']:<45} {apps:<20} {names}")
    print("=" * 80)


if __name__ == "__main__":
    main()
