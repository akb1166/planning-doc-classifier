"""Classifier module: categorises planning document pages."""

import logging
import re

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None

from config import CANDIDATE_LABELS, CLASSIFICATION_MODEL, CLASSIFICATION_THRESHOLD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule-based classifier (lightweight fallback)
# ---------------------------------------------------------------------------

KEYWORD_RULES: list[tuple[str, list[str]]] = [
    # More specific categories first to avoid false matches
    (
        "Approval of Reserved Matters or Details",
        ["approval of details", "notice of approval of details",
         "approval has been granted in respect of",
         "reserved matters", "discharge of condition"],
    ),
    (
        "Grant of Conditional Planning Permission",
        ["conditional planning permission", "grant of conditional",
         "conditional approval granted"],
    ),
    (
        "Planning Charges Register Entry",
        ["planning charges", "charges register", "conditions imposed",
         "nature of consent in brief"],
    ),
    (
        "Refusal of Planning Permission",
        ["permission has been refused", "refusal of planning",
         "hereby refused"],
    ),
    (
        "Appeal Decision Notice",
        ["appeal decision", "planning inspectorate", "appeal is allowed",
         "appeal is dismissed"],
    ),
    (
        "Certificate of Lawful Use or Development",
        ["certificate of lawful", "lawful development certificate",
         "certificate of lawfulness"],
    ),
    # General planning permission last (broadest match)
    (
        "Planning Permission Decision Notice",
        ["planning permission has been granted", "notice of approval",
         "application for planning permission"],
    ),
]


def classify_by_keywords(text: str) -> dict:
    """Simple keyword-matching classifier.

    Returns:
        Dict with 'label' and 'score' (count of matched keywords).
    """
    text_lower = text.lower()
    best_label = "Unknown"
    best_score = 0

    for label, keywords in KEYWORD_RULES:
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > best_score:
            best_score = matches
            best_label = label

    return {"label": best_label, "score": best_score}


# ---------------------------------------------------------------------------
# Zero-shot transformer classifier
# ---------------------------------------------------------------------------

def load_zero_shot_pipeline():
    """Load the zero-shot classification pipeline (downloads on first use)."""
    if hf_pipeline is None:
        logger.warning("transformers not installed. Using keyword fallback only.")
        return None
    logger.info("Loading zero-shot model: %s", CLASSIFICATION_MODEL)
    return hf_pipeline(
        "zero-shot-classification",
        model=CLASSIFICATION_MODEL,
        device=-1,  # CPU
    )


def classify_page(
    text: str,
    zs_pipeline=None,
    candidate_labels: list[str] | None = None,
) -> dict:
    """Classify a single page using zero-shot classification with keyword fallback.

    Args:
        text: OCR text of the page.
        zs_pipeline: Pre-loaded zero-shot pipeline (or None to skip).
        candidate_labels: Labels for zero-shot classification.

    Returns:
        Dict with 'label', 'score', and 'method'.
    """
    if candidate_labels is None:
        candidate_labels = CANDIDATE_LABELS

    # Attempt zero-shot classification first
    if zs_pipeline is not None:
        try:
            # Truncate to avoid token limits
            truncated = text[:1500]
            result = zs_pipeline(truncated, candidate_labels, multi_label=False)
            top_label = result["labels"][0]
            top_score = result["scores"][0]

            if top_score >= CLASSIFICATION_THRESHOLD:
                return {
                    "label": top_label,
                    "score": round(top_score, 4),
                    "method": "zero-shot",
                }
        except Exception as e:
            logger.warning("Zero-shot classification failed: %s", e)

    # Fallback to keyword matching
    kw_result = classify_by_keywords(text)
    return {
        "label": kw_result["label"],
        "score": kw_result["score"],
        "method": "keyword",
    }
