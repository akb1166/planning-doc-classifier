"""Extractor module: extracts application numbers and applicant names."""

import logging
import re

try:
    import spacy
except ImportError:
    spacy = None

from config import APPLICATION_NUMBER_PATTERNS, NER_MODEL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application number extraction (regex-based)
# ---------------------------------------------------------------------------

def extract_application_numbers(text: str) -> list[str]:
    """Extract planning application reference numbers from text.

    Uses a cascade of regex patterns, from most specific to broadest.

    Returns:
        Deduplicated list of matched application numbers.
    """
    matches = []
    for pattern in APPLICATION_NUMBER_PATTERNS:
        found = re.findall(pattern, text, re.IGNORECASE)
        matches.extend(found)

    def _is_date_like(s: str) -> bool:
        """Return True if the string looks like a date (DD/MM/YYYY)."""
        m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)
        if not m:
            return False
        month = int(m.group(2))
        return 1 <= month <= 12

    seen = set()
    unique = []
    for m in matches:
        m_clean = m.strip().rstrip(".")
        if (
            m_clean
            and m_clean not in seen
            and len(m_clean) >= 5
            and not _is_date_like(m_clean)
        ):
            seen.add(m_clean)
            unique.append(m_clean)

    return unique


# ---------------------------------------------------------------------------
# Applicant name extraction (spaCy NER + contextual regex)
# ---------------------------------------------------------------------------

def load_ner_model():
    """Load spaCy NER model (downloads if not present)."""
    if spacy is None:
        logger.warning("spaCy not installed. NER will be unavailable.")
        return None
    try:
        nlp = spacy.load(NER_MODEL)
    except OSError:
        logger.info("Downloading spaCy model: %s", NER_MODEL)
        spacy.cli.download(NER_MODEL)
        nlp = spacy.load(NER_MODEL)
    return nlp


# Title patterns including common OCR variants
_TITLE = r"(?:Mr|Mrs|Ms|Miss|Dr|Prof|Mra|Virs)"

_APPLICANT_CONTEXT_PATTERNS = [
    # "Conditional approval granted to Mr. & Mrs. J. M Doe"
    re.compile(
        r"(?:granted\s+to)\s+(" + _TITLE + r"\.?\s[A-Za-z\s.&,]{2,50}?)"
        r"(?:\n|,\s*\d|\bunder\b|\bdated\b|\bCouncil\b|\bcounty\b)",
        re.IGNORECASE,
    ),
    # "Applicant\n...\n<Title> <Name>" - name with title after Applicant header
    re.compile(
        r"Applicant.*?\n.*?\n\s*(" + _TITLE + r"\s*\.?\s*[A-Za-z\s.&,]{2,50}?)(?:\n|,)",
        re.IGNORECASE | re.DOTALL,
    ),
    # Standalone name line with title (e.g. "Mr M Dale,")
    re.compile(
        r"^(" + _TITLE + r"\.?\s+[A-Z][A-Za-z\s.]{2,40}?),?\s*$",
        re.MULTILINE,
    ),
    # Company names: "granted to <Company> Ltd"
    re.compile(
        r"(?:granted\s+to)\s+([A-Z][A-Za-z\s]+(?:Ltd|Limited|PLC|Inc)\.?)",
        re.IGNORECASE,
    ),
]


def extract_applicant_names(text: str, nlp=None) -> list[str]:
    """Extract applicant names using spaCy NER plus contextual regex.

    Args:
        text: OCR text of the page.
        nlp: Pre-loaded spaCy model (or None to skip NER).

    Returns:
        Deduplicated list of applicant names found.
    """
    names = set()

    # 1. Contextual regex extraction
    for pattern in _APPLICANT_CONTEXT_PATTERNS:
        for match in pattern.finditer(text):
            candidate = match.group(1).strip().rstrip(",.")
            if 3 < len(candidate) < 60:
                names.add(candidate)

    # 2. spaCy NER extraction
    if nlp is not None:
        header_text = text[:2000]
        doc = nlp(header_text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                if len(name) > 3 and not name.isupper():
                    names.add(name)

    return sorted(names)


def extract_entities(text: str, nlp=None) -> dict:
    """Run all extraction on a page's text.

    Returns:
        Dict with 'application_numbers' and 'applicant_names'.
    """
    return {
        "application_numbers": extract_application_numbers(text),
        "applicant_names": extract_applicant_names(text, nlp=nlp),
    }
