# Analysis Report

## Exploratory Data Analysis

The input PDF contains 4 scanned pages with no embedded text, requiring OCR. Tesseract produced usable output at 300 DPI, though OCR noise (e.g. character substitutions, broken words) was evident on older documents. Pages varied in layout: structured forms (page 1), typed decision letters (pages 2–4), and older typewritten documents (page 3). Four distinct document categories were identified:

1. **Planning Charges Register Entry** — tabular records listing conditions imposed on planning consents.
2. **Planning Permission Decision Notice** — formal notices granting planning permission under the Town and Country Planning Act.
3. **Grant of Conditional Planning Permission** — letters granting permission subject to specified conditions.
4. **Approval of Reserved Matters or Details** — notices approving details pursuant to earlier permissions.

Application numbers followed identifiable formats (e.g. `P/00/0759`, `02/80/1609`) and were reliably captured by regex. Applicant names appeared near the top of decision notices, sometimes preceded by titles (Mr, Mrs, Ltd).

## Method

**Classification** uses zero-shot inference via `facebook/bart-large-mnli`, which requires no training data and generalises well to unseen document types. A keyword-based fallback ensures robustness when the model is unavailable or confidence is low.

**Entity extraction** combines regex patterns for application numbers with spaCy's `en_core_web_sm` NER model for applicant names. Contextual regex targets phrases like "approval granted to" to improve recall in OCR-degraded text.

## Limitations

- **OCR quality**: older or poorly scanned pages introduce noise that can degrade both classification and NER accuracy. Pre-processing (deskewing, binarisation) could improve this.
- **Sample size**: with only 4 pages, classification categories and regex patterns may not generalise to the full range of planning document types (e.g. refusals, appeals, enforcement notices).
- **NER on OCR text**: spaCy's NER was trained on clean text; OCR artefacts reduce entity recognition recall. Fine-tuning on OCR-degraded planning documents would help.
- **Zero-shot ceiling**: the BART-MNLI model provides reasonable zero-shot accuracy but would be outperformed by a classifier fine-tuned on labelled planning documents.

## Alternate Methods

- **Vision approach**: a CNN-based image classifier (e.g. ResNet or EfficientNet) trained on page images could classify documents without OCR, sidestepping text quality issues entirely.
- **Layout-aware models**: LayoutLM or Donut, which jointly model text and spatial layout, would better capture the structural differences between document types.
- **Fine-tuned NER**: training a custom spaCy or transformer NER model on annotated planning documents would significantly improve applicant name extraction.
