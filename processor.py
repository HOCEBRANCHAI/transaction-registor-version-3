# processor.py
# Robust extraction, OCR, FX, validation, mapping, and posting.
# Used by app.py. No server here.
import io
import os
import re
import json
import logging
import imghdr
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, timedelta
# Money / math
from decimal import Decimal, ROUND_HALF_UP
# OCR & PDF
import PyPDF2
import boto3
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import numpy as np
import pytesseract

# LLM + HTTP
from openai import OpenAI
import requests

# Env
from dotenv import load_dotenv
import pathlib

# Initialize logging first
log = logging.getLogger("invoice-processor")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Load .env file - try multiple locations
env_paths = [
    pathlib.Path(__file__).parent / '.env',  # Same directory as processor.py
    pathlib.Path.cwd() / '.env',  # Current working directory
    pathlib.Path.home() / '.env',  # Home directory (fallback)
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        log.info(f"Loaded .env from: {env_path}")
        env_loaded = True
        break

if not env_loaded:
    # Fallback: try loading from current directory without explicit path
    load_dotenv()
    log.info("Attempted to load .env from current directory")

# Verify OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    log.warning("WARNING: OPENAI_API_KEY not found in environment variables!")
    log.warning("Please ensure your .env file contains: OPENAI_API_KEY=your_key_here")
else:
    log.info("OpenAI API key loaded successfully")
log = logging.getLogger("invoice-processor")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------- Money helpers --------------------
CENT = Decimal("0.01")
RATE_PREC = Decimal("0.0001")

def q_money(x) -> Decimal:
    return Decimal(str(x)).quantize(CENT, rounding=ROUND_HALF_UP)

def q_rate(x) -> Decimal:
    return Decimal(str(x)).quantize(RATE_PREC, rounding=ROUND_HALF_UP)

def nearly_equal_money(a: Decimal, b: Decimal, tol: Decimal = CENT) -> bool:
    return abs(q_money(a) - q_money(b)) <= tol

# -------------------- Dates & currency --------------------
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def ensure_iso_date(s: Optional[str], field: str, errors: List[str]) -> Optional[date]:
    if not s or not ISO_DATE.match(s):
        errors.append(f"{field} must be YYYY-MM-DD (got {s!r}).")
        return None
    try:
        y, m, d = map(int, s.split("-"))
        return date(y, m, d)
    except Exception:
        errors.append(f"{field} is not a valid calendar date (got {s!r}).")
        return None

KNOWN_CURRENCIES = {
    "EUR","USD","GBP","INR","EGP","AED","SAR","CAD","AUD","NZD",
    "JPY","CNY","DKK","SEK","NOK","CHF","PLN","CZK","HUF"
}

def normalize_currency(cur: Optional[str], errors: List[str]) -> Optional[str]:
    cur = (cur or "").strip().upper()
    if cur not in KNOWN_CURRENCIES:
        errors.append(f"Unknown or missing currency {cur!r}.")
        return None
    return cur

def _prev_business_day(d: date) -> date:
    while d.weekday() >= 5:  # 5=Sat,6=Sun
        d -= timedelta(days=1)
    return d

def get_eur_rate(invoice_date: date, ccy: str) -> Tuple[Decimal, str]:
    """
    Return (rate, rate_date_str) for 1 CCY -> EUR using exchangerate.host (ECB).
    Try direct (base=CCY&symbols=EUR) and fallback by inversion.
    Look back up to 7 business days.
    """
    ccy = (ccy or "").upper().strip()
    if ccy == "EUR":
        return Decimal("1"), invoice_date.isoformat()

    d = _prev_business_day(invoice_date)
    for _ in range(7):
        # direct
        url1 = f"https://api.exchangerate.host/{d.isoformat()}?base={ccy}&symbols=EUR"
        try:
            r1 = requests.get(url1, timeout=8)
            js1 = r1.json() if r1.content else {}
            rate = (js1.get("rates") or {}).get("EUR")
            if r1.status_code == 200 and rate:
                return q_rate(rate), d.isoformat()
            log.warning(f"FX miss (direct) {url1} status={r1.status_code}")
        except Exception as ex:
            log.warning(f"FX direct failed {url1}: {ex}")

        # invert
        url2 = f"https://api.exchangerate.host/{d.isoformat()}?base=EUR&symbols={ccy}"
        try:
            r2 = requests.get(url2, timeout=8)
            js2 = r2.json() if r2.content else {}
            base_rate = (js2.get("rates") or {}).get(ccy)
            if r2.status_code == 200 and base_rate and float(base_rate) != 0.0:
                inv = Decimal("1") / Decimal(str(base_rate))
                return q_rate(inv), d.isoformat()
            log.warning(f"FX miss (invert) {url2} status={r2.status_code}")
        except Exception as ex2:
            log.warning(f"FX invert failed {url2}: {ex2}")

        d = _prev_business_day(d - timedelta(days=1))
    raise ValueError(f"No EUR rate found for {ccy} near {invoice_date.isoformat()}.")

# -------------------- OCR (robust, supports PDFs & images) --------------------
def _aws_region() -> str:
    """
    Prefer AWS_REGION if set, otherwise AWS_DEFAULT_REGION, otherwise us-east-1.
    This keeps behaviour backwards compatible but makes region selection explicit.
    """
    return (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def _preprocess_for_tesseract(pil_img: Image.Image) -> Image.Image:
    """
    Aggressive but safe preprocessing to improve OCR quality:
      - convert to grayscale
      - bilateral filter to reduce noise but keep edges
      - adaptive threshold for better contrast
    """
    img = np.array(pil_img.convert("L"))
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return Image.fromarray(img)


def _textract_analyze_image(img_bytes: bytes) -> str:
    textract = boto3.client("textract", region_name=_aws_region())
    resp = textract.analyze_document(
        Document={'Bytes': img_bytes},
        FeatureTypes=['TABLES', 'FORMS']
    )
    blocks = resp.get("Blocks", [])
    text_lines = []
    block_map = {b["Id"]: b for b in blocks}

    for b in blocks:
        if b.get("BlockType") == "LINE" and b.get("Text"):
            text_lines.append(b["Text"])

    kv_pairs = []
    for b in blocks:
        if b.get("BlockType") == "KEY_VALUE_SET" and "KEY" in (b.get("EntityTypes") or []):
            key_words, val_words = [], []
            for rel in b.get("Relationships", []):
                if rel["Type"] == "CHILD":
                    for cid in rel.get("Ids", []):
                        w = block_map.get(cid)
                        if w and w.get("BlockType") == "WORD" and w.get("Text"):
                            key_words.append(w["Text"])
                if rel["Type"] == "VALUE":
                    for vid in rel.get("Ids", []):
                        v = block_map.get(vid)
                        if not v: continue
                        for rel2 in v.get("Relationships", []):
                            if rel2["Type"] == "CHILD":
                                for vcid in rel2.get("Ids", []):
                                    w = block_map.get(vcid)
                                    if w and w.get("BlockType") == "WORD" and w.get("Text"):
                                        val_words.append(w["Text"])
            k = " ".join(key_words).strip()
            v = " ".join(val_words).strip()
            if k or v:
                kv_pairs.append(f"{k}: {v}")

    combined = "\n".join(text_lines)
    if kv_pairs:
        combined += "\n--- Key-Value Pairs ---\n" + "\n".join(kv_pairs)
    return combined

def _tesseract_ocr(pil_img: Image.Image) -> str:
    pre = _preprocess_for_tesseract(pil_img)
    # Use English by default; --psm 6 handles block of text with uniform size.
    # This dramatically improves character accuracy compared to the default.
    return pytesseract.image_to_string(pre, config="--psm 6 -l eng")

def get_text_from_pdf(pdf_bytes: bytes, filename: str) -> str:
    """
    Robust, multi-strategy PDF text extraction with AWS Textract + Tesseract.

    Strategy (in order):
      1) PyPDF2 text (fast, cheap)
      2) Textract detect_document_text (whole PDF, if AWS available)
      3) PDF → images → Textract AnalyzeDocument (per page, for scanned PDFs)
      4) PDF → images → Tesseract OCR (last resort)

    We keep track of the *best* text seen so far (longest non-empty)
    and, if all strategies are \"minimal\", we still return the best
    attempt instead of failing. This guarantees that we always return
    some text for the LLM, even on very difficult PDFs.
    """
    best_text: str = ""
    best_source: str = ""

    def _update_best(candidate: str, source: str) -> None:
        nonlocal best_text, best_source
        if candidate and len(candidate.strip()) > len(best_text.strip()):
            best_text = candidate
            best_source = source

    # 1) PyPDF2
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = "".join([(p.extract_text() or "") for p in reader.pages])
        _update_best(text, "PyPDF2")
        if len(text.strip()) > 80:
            log.info(f"[PyPDF2] {filename}")
            return text
        log.warning(f"[PyPDF2] minimal for {filename}; trying Textract detect.")
    except Exception as e:
        log.warning(f"[PyPDF2] failed for {filename}: {e}")

    # 2) Textract detect (only if AWS creds/region likely configured)
    try:
        textract = boto3.client("textract", region_name=_aws_region())
        resp = textract.detect_document_text(Document={"Bytes": pdf_bytes})
        text = "\n".join(
            [
                b.get("Text", "")
                for b in (resp.get("Blocks") or [])
                if b.get("BlockType") == "LINE"
            ]
        )
        _update_best(text, "Textract.detect")
        if len(text.strip()) > 60:
            log.info(f"[Textract.detect] {filename}")
            return text
        log.warning(f"[Textract.detect] minimal for {filename}; trying Analyze per page.")
    except Exception as e:
        log.warning(f"[Textract.detect] failed for {filename}: {e}; Analyze per page.")

    # 3) Textract Analyze per-page (images)
    try:
        images: List[Image.Image] = convert_from_bytes(pdf_bytes, dpi=300)
        texts = []
        for im in images:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            page_text = _textract_analyze_image(buf.getvalue())
            if page_text:
                texts.append(page_text)
                _update_best(page_text, "Textract.analyze IMG")
        combined = "\n\n--- PAGE BREAK ---\n\n".join(texts)
        if len(combined.strip()) > 60:
            log.info(f"[Textract.analyze IMG] {filename}")
            return combined
        log.warning(f"[Textract.analyze IMG] minimal; trying Tesseract.")
    except Exception as e:
        log.warning(f"[Textract.analyze IMG] failed for {filename}: {e}; Tesseract fallback.")

    # 4) Tesseract fallback
    try:
        images = convert_from_bytes(pdf_bytes, dpi=300)
        ocr = []
        for im in images:
            page_text = _tesseract_ocr(im)
            if page_text:
                ocr.append(page_text)
                _update_best(page_text, "Tesseract")
        combined = "\n\n--- PAGE BREAK ---\n\n".join(ocr)
        if len(combined.strip()) > 20:
            log.info(f"[Tesseract] {filename}")
            return combined
    except Exception as e:
        log.error(f"[Tesseract] failed for {filename}: {e}")

    # If we reached this point, all strategies were "minimal".
    # For production we still prefer returning *something* over hard failure.
    if best_text.strip():
        log.error(
            f"PDF text extraction for {filename} only produced minimal text; "
            f"returning best attempt from {best_source} with length={len(best_text.strip())}."
        )
        return best_text

    # Absolute fallback – nothing at all could be read.
    raise ValueError(
        f"PDF text extraction failed for {filename}: "
        f"PyPDF2, Textract detect, Textract analyze (per image), and Tesseract all returned empty text."
    )


def get_text_from_image(image_bytes: bytes, filename: str) -> str:
    """
    Robust text extraction for image-based invoices (JPEG, PNG, TIFF, etc.)
    using AWS Textract + Tesseract.
    Strategy:
      1) Textract detect_document_text
      2) Textract analyze_document (FORMS+TABLES)
      3) Tesseract OCR (with preprocessing)
    """
    best_text: str = ""
    best_source: str = ""

    def _update_best(candidate: str, source: str) -> None:
        nonlocal best_text, best_source
        if candidate and len(candidate.strip()) > len(best_text.strip()):
            best_text = candidate
            best_source = source

    # 1) Textract detect
    try:
        textract = boto3.client("textract", region_name=_aws_region())
        resp = textract.detect_document_text(Document={"Bytes": image_bytes})
        text = "\n".join(
            [
                b.get("Text", "")
                for b in (resp.get("Blocks") or [])
                if b.get("BlockType") == "LINE"
            ]
        )
        _update_best(text, "Textract.detect IMG")
        if len(text.strip()) > 40:
            log.info(f"[Textract.detect IMG] {filename}")
            return text
        log.warning(f"[Textract.detect IMG] minimal for {filename}; trying AnalyzeDocument.")
    except Exception as e:
        log.warning(f"[Textract.detect IMG] failed for {filename}: {e}; trying AnalyzeDocument.")

    # 2) Textract AnalyzeDocument (FORMS + TABLES)
    try:
        analyzed = _textract_analyze_image(image_bytes)
        _update_best(analyzed, "Textract.analyze IMG")
        if len(analyzed.strip()) > 40:
            log.info(f"[Textract.analyze IMG] {filename}")
            return analyzed
        log.warning(f"[Textract.analyze IMG] minimal for {filename}; trying Tesseract.")
    except Exception as e:
        log.warning(f"[Textract.analyze IMG helper] failed for {filename}: {e}; trying Tesseract.")

    # 3) Tesseract fallback
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        text = _tesseract_ocr(pil_img)
        _update_best(text, "Tesseract IMG")
        if len(text.strip()) > 20:
            log.info(f"[Tesseract IMG] {filename}")
            return text
    except Exception as e:
        log.error(f"[Tesseract IMG] failed for {filename}: {e}")

    # If we reached this point, only minimal text.
    if best_text.strip():
        log.error(
            f"Image text extraction for {filename} only produced minimal text; "
            f"returning best attempt from {best_source} with length={len(best_text.strip())}."
        )
        return best_text

    raise ValueError(
        f"Image text extraction failed for {filename}: "
        f"Textract detect, Textract analyze, and Tesseract all returned empty text."
    )


def get_text_from_document(file_bytes: bytes, filename: str) -> str:
    """
    Entry point for *any* uploaded document.

    - Detects whether the content is PDF or image using:
        * PDF header (%PDF-) OR .pdf extension
        * Common image file extensions for JPEG/PNG/TIFF/BMP/GIF
    - Routes to the appropriate extraction function.
    - If detection is ambiguous, defaults to the PDF pipeline because
      most uploads in this app are invoice PDFs.
    """
    name = (filename or "").lower()
    ext = os.path.splitext(name)[1]

    # Quick PDF signature check
    is_pdf_header = file_bytes.startswith(b"%PDF-")
    if is_pdf_header or ext == ".pdf":
        return get_text_from_pdf(file_bytes, filename)

    # Image type detection via filename extension (imghdr was removed in Python 3.13)
    if ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}:
        return get_text_from_image(file_bytes, filename)

    # Ambiguous: try PDF pipeline first, then fall back to image logic if that fails
    try:
        return get_text_from_pdf(file_bytes, filename)
    except Exception as e_pdf:
        log.warning(f"PDF pipeline failed for {filename} ({e_pdf}); trying image pipeline.")
        return get_text_from_image(file_bytes, filename)

# -------------------- LLM extraction --------------------
SECTION_LABELS = [
    "invoice", "total", "subtotal", "tax", "vat", "btw", "reverse charge",
    "verlegd", "omgekeerde heffing", "bill to", "payer", "customer", "vendor",
    "supplier", "line items", "description", "due", "payment terms", "amount"
]

def reduce_invoice_text(raw_text: str, window: int = 300) -> str:
    text = raw_text or ""
    text_low = text.lower()
    spans: List[Tuple[int, int]] = []
    for label in SECTION_LABELS:
        for m in re.finditer(re.escape(label), text_low):
            start = max(0, m.start() - window)
            end = min(len(text), m.end() + window)
            spans.append((start, end))
    if not spans:
        return text[:8000]
    spans.sort()
    merged = []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s <= cur_e + 50:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    chunks = [text[s:e] for s, e in merged]
    reduced = "\n---\n".join(chunks)
    return reduced[:12000]

LLM_PROMPT = """
You are an expert, high-accuracy financial data extraction model. Your sole task is to extract structured data from the provided invoice text and respond with a single, minified JSON object. You must think like an accountant.

RULES:
1) JSON ONLY. Include all top-level keys even if null. Dates = YYYY-MM-DD. Numbers = floats (no symbols).
2) Use the invoice currency shown in the official TOTAL block; ignore "for reference" currencies.
3) subtotal = goods/services only; total_vat = all taxes; total_amount = subtotal + total_vat.
4) VAT category:
   - "Import-VAT" for import VAT.
   - "Reverse-Charge" if reverse-charge applies (e.g., verlegd, omgekeerde heffing).
   - "Standard" for normal % VAT charged.
   - "Zero-Rated" if 0% VAT and not reverse charge.
   - "Out-of-Scope" if outside tax scope (e.g., Article 44).
5) Line items:
   - Extract goods/services only; do NOT include taxes as a line item.
   - unit_price only if explicitly printed. Do not compute it.
6) VAT percentage:
   - If the invoice explicitly shows a single VAT rate (e.g., 21%), put that in vat_breakdown.rate.
   - If text states "VAT out of scope / Article 44 / reverse charge / 0%", use 0.0 in vat_breakdown.
   - If multiple rates exist, list them; total_vat must equal the sum of tax_amount.
7) ADDRESSES - CRITICAL FOR VAT CLASSIFICATION:
   - Extract COMPLETE addresses including street, city, postal code, and COUNTRY.
   - The country is ESSENTIAL for determining EU vs non-EU transactions.
   - Look for country names (e.g., "Netherlands", "Germany", "France") or country codes (e.g., "NL", "DE", "FR") at the end of addresses.
   - Include the full address as a single string with all address components.
   - This is critical for production-level VAT subcategory classification.
8) GOODS VS SERVICES:
   - Determine if the invoice is for "goods" or "services" based on line item descriptions.
   - Look for keywords like "product", "item", "goods", "merchandise" for goods.
   - Look for keywords like "service", "consulting", "support", "maintenance", "software license" for services.
   - If unclear, default to "services" for B2B transactions, "goods" for physical products.
   - Set "goods_services_indicator" to "goods" or "services" or null if truly unclear.
9) BANK ACCOUNT DETAILS (IBAN):
   - Extract IBAN numbers for both vendor and customer if present on the invoice.
   - IBAN is typically found in payment/banking sections of invoices.
   - Format: IBAN codes start with 2-letter country code followed by 2 digits and up to 30 alphanumeric characters.
   - Extract "vendor_iban" from vendor's bank account details.
   - Extract "customer_iban" from customer's bank account details (if shown).
   - IBAN can be a signal for local vs non-local transactions affecting VAT categorization.
10) NOTES AND COMMENTS:
   - Extract any general notes, comments, or additional information from the invoice.
   - Include payment instructions, special terms, or any other relevant notes.
   - Store in "notes" field.
SCHEMA:

{
  "invoice_number": "string | null",
  "invoice_date": "YYYY-MM-DD | null",
  "due_date": "YYYY-MM-DD | null",
  "vendor_name": "string | null",
  "vendor_vat_id": "string | null",
  "vendor_address": "string | null (MUST include full address with country)",
  "customer_name": "string | null",
  "customer_vat_id": "string | null",
  "customer_address": "string | null (MUST include full address with country)",
  "currency": "string | null",
  "vat_category": "string | null",
  "subtotal": "float | null",
  "total_amount": "float | null",
  "total_vat": "float | null",
  "vat_breakdown": [
    {"rate": "float | 'import'", "base_amount": "float | null", "tax_amount": "float"}
  ],
  "line_items": [
    {"description": "string", "quantity": "float | null", "unit_price": "float | null", "line_total": "float | null"}
  ],
  "payment_terms": "string | null",
  "goods_services_indicator": "goods | services | null",
  "vendor_iban": "string | null",
  "customer_iban": "string | null",
  "notes": "string | null"
}

"""
def structure_text_with_llm(invoice_text: str, filename: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")

    client = OpenAI(api_key=api_key)
    reduced = reduce_invoice_text(invoice_text)
    try:
        log.info(f"LLM extracting {filename}...")
        r = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            temperature=0.0,
            messages=[
                {"role": "system", "content": LLM_PROMPT},
                {"role": "user", "content": f"**INVOICE TEXT TO PARSE (reduced):**\n{reduced}"}
            ]
        )
        return json.loads(r.choices[0].message.content)
    except json.JSONDecodeError as e:
        log.error(f"LLM JSON error {filename}: {e}")
        raise ValueError(f"Extraction failed for {filename}: malformed JSON.")
    except Exception as e:
        log.error(f"LLM API error {filename}: {e}")
        raise ValueError("Extraction failed: LLM API error.")

def _translate_to_english_if_dutch(text: str) -> str:
    """
    Best‑effort translation helper:
      - If the text appears in Dutch, translate it to English.
      - If it is already English or another language, return it unchanged.
    Uses the same OpenAI client as the main extraction. On any failure, returns
    the original text so we never break the pipeline because of translation.
    """
    if not text or not text.strip():
        return text

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # No API key → cannot translate; keep original description.
        return text

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a translation helper. "
                        "If the user text is Dutch, respond with an accurate English translation. "
                        "If the text is already English or clearly not Dutch, return it exactly as-is. "
                        "Output plain text only, no explanations, no quotes."
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        return content or text
    except Exception as ex:
        log.error(f"Description translation failed: {ex}")
        return text


# -------------------- Validation & mapping --------------------
def _estimate_extraction_confidence(invoice_text: str, llm_data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Heuristic confidence score for how reliable the extraction likely is.
    This is *not* a guarantee – just a signal to help prioritise manual review.

    Factors:
      - Length of extracted text (very short text => low confidence).
      - Presence of key invoice keywords.
      - Presence of critical LLM fields (invoice_number, dates, totals, parties).

    Returns:
      (level, reason) where level ∈ {"high", "medium", "low"}.
    """
    text = (invoice_text or "").strip()
    text_len = len(text)

    # 1) Base on text length
    if text_len < 200:
        level = "low"
        reasons = [f"very short extracted text ({text_len} chars)"]
    elif text_len < 800:
        level = "medium"
        reasons = [f"moderate extracted text length ({text_len} chars)"]
    else:
        level = "high"
        reasons = [f"long extracted text ({text_len} chars)"]

    # 2) Check for presence of section labels
    text_low = text.lower()
    present_labels = [lbl for lbl in SECTION_LABELS if lbl in text_low]
    if len(present_labels) <= 2:
        # Very few typical invoice markers found
        reasons.append(f"few invoice keywords found ({len(present_labels)})")
        if level == "high":
            level = "medium"
    elif len(present_labels) <= 5 and level == "high":
        reasons.append(f"some invoice keywords found ({len(present_labels)})")

    # 3) Check for missing critical LLM fields
    critical_fields = [
        "invoice_number",
        "invoice_date",
        "vendor_name",
        "customer_name",
        "subtotal",
        "total_vat",
        "total_amount",
    ]
    missing = [f for f in critical_fields if not llm_data.get(f)]
    if missing:
        reasons.append(f"missing critical fields from LLM: {', '.join(missing)}")
        # Any missing core field should cap at medium; if already low, keep low
        if level == "high":
            level = "medium"

    reason_str = "; ".join(reasons)
    return level, reason_str
def validate_extraction(data: dict, filename: str) -> Tuple[date, str, Decimal, Decimal, Decimal]:
    errors: List[str] = []
    for f in ["invoice_number","invoice_date","vendor_name","customer_name",
              "vat_category","currency","subtotal","total_amount","total_vat"]:
        if data.get(f) in (None, ""):
            errors.append(f"Missing {f!r}.")

    inv_date = ensure_iso_date(data.get("invoice_date"), "invoice_date", errors)
    if data.get("due_date"):
        _ = ensure_iso_date(data.get("due_date"), "due_date", errors)

    currency = normalize_currency(data.get("currency"), errors)

    try:
        sub = q_money(data["subtotal"])
        vat = q_money(data["total_vat"])
        tot = q_money(data["total_amount"])
        if not nearly_equal_money(sub + vat, tot):
            errors.append(f"Subtotal({sub}) + VAT({vat}) != Total({tot}).")
    except Exception:
        errors.append("Invalid numeric values for subtotal/total_vat/total_amount.")

    if errors:
        msg = f"Validation failed for {filename}: {' | '.join(errors)}"
        log.error(msg)
        raise ValueError(msg)

    assert inv_date and currency
    return inv_date, currency, sub, vat, tot

def _normalize_company_name(name: str) -> str:
    """
    Normalizes company name for matching by:
    - Converting to lowercase
    - Removing all punctuation and special characters
    - Removing common legal suffixes (B.V., BV, B.V, Ltd, Limited, etc.)
    - Normalizing spaces
    - Removing common stop words
    """
    if not name:
        return ""
    
    # Convert to lowercase
    normalized = name.casefold()
    
    # Remove common legal entity suffixes (case-insensitive)
    legal_suffixes = [
        r'\bb\.v\.?\b', r'\bbv\b', r'\bb\.v\b',
        r'\bltd\.?\b', r'\blimited\b',
        r'\binc\.?\b', r'\bincorporated\b',
        r'\bllc\b', r'\bll\.?c\.?\b',
        r'\bcorp\.?\b', r'\bcorporation\b',
        r'\bs\.a\.?\b', r'\bsa\b',
        r'\bs\.a\.?r\.?l\.?\b', r'\bsarl\b',
        r'\bgmbh\b', r'\bag\b',
        r'\bn\.?v\.?\b', r'\bnv\b',
        r'\bspa\b', r'\bsp\.?z\.?o\.?o\.?\b',
        r'\bsl\b', r'\bs\.?l\.?\b'
    ]
    
    for suffix_pattern in legal_suffixes:
        normalized = re.sub(suffix_pattern, '', normalized)
    
    # Remove common stop words that don't help with matching
    stop_words = [r'\bthe\b', r'\ba\b', r'\ban\b', r'\band\b', r'\bor\b', r'\bof\b']
    for stop_word in stop_words:
        normalized = re.sub(stop_word, '', normalized)
    
    # Remove all punctuation, special characters, and normalize spaces
    normalized = re.sub(r"[\s\-_/.,()\[\]{}'\"&]+", " ", normalized)
    
    # Remove multiple spaces and trim
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def _calculate_name_similarity(name1: str, name2: str) -> float:
    """
    Calculates similarity between two normalized company names.
    Returns a score between 0.0 and 1.0.
    Uses word overlap and substring matching.
    """
    if not name1 or not name2:
        return 0.0
    
    n1 = _normalize_company_name(name1)
    n2 = _normalize_company_name(name2)
    
    if n1 == n2:
        return 1.0
    
    # Check if one is substring of another
    if n1 in n2 or n2 in n1:
        # Calculate ratio of shorter to longer
        shorter = min(len(n1), len(n2))
        longer = max(len(n1), len(n2))
        return shorter / longer if longer > 0 else 0.0
    
    # Word-based similarity
    words1 = set(n1.split())
    words2 = set(n2.split())
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity (intersection over union)
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def _split_company_list(raw: str) -> List[str]:
    if not raw: return []
    return [p.strip() for p in re.split(r"[,\n;]", raw) if p and p.strip()]

# ----------------------------------------------------------------------------
# EU Country Detection & Address Parsing
# ----------------------------------------------------------------------------

# Comprehensive list of EU member countries (as of 2024)
EU_COUNTRIES = {
    # Full country names
    "austria", "belgium", "bulgaria", "croatia", "cyprus", "czech republic", "czechia",
    "denmark", "estonia", "finland", "france", "germany", "greece", "hungary",
    "ireland", "italy", "latvia", "lithuania", "luxembourg", "malta", "netherlands",
    "poland", "portugal", "romania", "slovakia", "slovenia", "spain", "sweden",
    # Country codes
    "at", "be", "bg", "hr", "cy", "cz", "dk", "ee", "fi", "fr", "de", "gr", "hu",
    "ie", "it", "lv", "lt", "lu", "mt", "nl", "pl", "pt", "ro", "sk", "si", "es", "se",
    # Alternative names
    "nederland", "holland", "deutschland", "italia", "espana", "frankreich",
    "belgie", "belgique"
}

# Common country names (including non-EU) for extraction
# Maps country names/variants to normalized country names
COMMON_COUNTRIES = {
    # EU countries (from above) - map to themselves
    **{country: country for country in EU_COUNTRIES if len(country) > 2},  # Full names only
    # Non-EU countries
    "united kingdom": "united kingdom", "uk": "united kingdom", "great britain": "united kingdom", "britain": "united kingdom",
    "united states": "united states", "usa": "united states", "us": "united states", "america": "united states",
    "switzerland": "switzerland", "norway": "norway", "iceland": "iceland",
    "china": "china", "japan": "japan", "india": "india", "canada": "canada",
    "australia": "australia", "new zealand": "new zealand", "south africa": "south africa",
    "brazil": "brazil", "mexico": "mexico", "argentina": "argentina",
    "egypt": "egypt", "uae": "united arab emirates", "united arab emirates": "united arab emirates", 
    "saudi arabia": "saudi arabia", "singapore": "singapore", "hong kong": "hong kong", 
    "south korea": "south korea", "taiwan": "taiwan", "turkey": "turkey",
    "russia": "russia", "ukraine": "ukraine"
}
def _extract_country_from_address(address: str) -> Optional[str]:
    """
    Extracts country name or code from an address string.
    Returns the country name if found, None otherwise.
    Handles both EU and non-EU countries.
    """
    if not address:
        return None
    
    address_lower = address.lower()
    
    # First, check for common country names (prioritize longer/more specific names first)
    # Sort by length descending to match "United Kingdom" before "Kingdom"
    country_names = sorted(COMMON_COUNTRIES.keys(), key=len, reverse=True)
    
    for country_name in country_names:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(country_name) + r'\b'
        if re.search(pattern, address_lower):
            # Get normalized country name
            normalized = COMMON_COUNTRIES[country_name]
            # Return properly capitalized country name
            if normalized == "united kingdom":
                return "United Kingdom"
            elif normalized == "united states":
                return "United States"
            elif normalized == "united arab emirates":
                return "United Arab Emirates"
            else:
                return normalized.title()
    
    # Check for country codes (usually at the end of address, after postal code)
    # Pattern: postal code + country code (e.g., "1234 AB NL" or "75001 FR")
    country_code_pattern = r'\b([a-z]{2})\b'
    matches = re.findall(country_code_pattern, address_lower)
    
    # Check matches against known country codes (check from end, country usually at end)
    code_to_country = {
        "at": "Austria", "be": "Belgium", "bg": "Bulgaria", "hr": "Croatia",
        "cy": "Cyprus", "cz": "Czech Republic", "dk": "Denmark", "ee": "Estonia",
        "fi": "Finland", "fr": "France", "de": "Germany", "gr": "Greece",
        "hu": "Hungary", "ie": "Ireland", "it": "Italy", "lv": "Latvia",
        "lt": "Lithuania", "lu": "Luxembourg", "mt": "Malta", "nl": "Netherlands",
        "pl": "Poland", "pt": "Portugal", "ro": "Romania", "sk": "Slovakia",
        "si": "Slovenia", "es": "Spain", "se": "Sweden",
        "gb": "United Kingdom", "uk": "United Kingdom", "us": "United States", "ch": "Switzerland"
    }
    
    for match in reversed(matches):
        if match in code_to_country:
            return code_to_country[match]
    
    return None

def _is_eu_country(country: Optional[str]) -> bool:
    """
    Checks if a country is in the EU.
    """
    if not country:
        return False
    return country.lower() in EU_COUNTRIES

def _is_nl_country(country: Optional[str]) -> bool:
    """
    Checks if a country is Netherlands (NL).
    """
    if not country:
        return False
    country_lower = country.lower()
    return (country_lower == "netherlands" or 
            country_lower == "nl" or 
            country_lower == "nederland" or 
            country_lower == "holland")

def _determine_goods_services_indicator(llm_data: Dict[str, Any], invoice_text: str = "") -> Optional[str]:
    """
    Determines if the invoice is for goods or services.
    Returns "goods", "services", or None if unclear.
    """
    # First check if LLM extracted it
    indicator = llm_data.get("goods_services_indicator")
    if indicator and indicator.lower() in ["goods", "services"]:
        return indicator.lower()
    
    # Fallback: analyze line items and description
    text_lower = (invoice_text or "").lower()
    description = ""
    if llm_data.get("line_items"):
        descriptions = [item.get("description", "") for item in llm_data.get("line_items", [])]
        description = " ".join(descriptions).lower()
    
    combined_text = (text_lower + " " + description).lower()
    
    # Goods indicators
    goods_keywords = [
        "product", "products", "item", "items", "goods", "merchandise",
        "physical", "tangible", "shipment", "delivery", "warehouse",
        "stock", "inventory", "material", "materials", "equipment",
        "hardware", "component", "parts", "supplies"
    ]
    
    # Services indicators
    services_keywords = [
        "service", "services", "consulting", "consultancy", "support",
        "maintenance", "repair", "installation", "training", "advice",
        "advisory", "software license", "licensing", "subscription",
        "professional", "expertise", "expert", "assistance", "help",
        "management", "administration", "processing", "handling"
    ]
    
    goods_count = sum(1 for keyword in goods_keywords if keyword in combined_text)
    services_count = sum(1 for keyword in services_keywords if keyword in combined_text)
    
    if goods_count > services_count and goods_count > 0:
        return "goods"
    elif services_count > goods_count and services_count > 0:
        return "services"
    
    # Default to services for B2B if unclear
    return "services"

def _determine_invoice_subcategory(
    invoice_type: str,
    vendor_address: Optional[str],
    customer_address: Optional[str],
    vat_percentage: Optional[float],
    invoice_text: str = ""
) -> str:
    """
    Determines the invoice subcategory based on:
    - Invoice type (Sales/Purchase)
    - Vendor and customer addresses (EU vs non-EU)
    - VAT rate (21%, 9%, etc.)
    
    Returns subcategory string like:
    - For Sales: "Standard 21%", "Reduced Rate 9%", "Sales to EU Countries", "Sales to Non-EU Countries"
    - For Purchase: "Purchase from EU Countries", "Purchase from Non-EU Countries (Import VAT)"
    """
    # Extract countries from addresses
    vendor_country = _extract_country_from_address(vendor_address) if vendor_address else None
    customer_country = _extract_country_from_address(customer_address) if customer_address else None
    
    # Check for import VAT indicators in invoice text
    import_vat_keywords = [
        "import vat", "importvat", "import btw", "importbtw",
        "invoer btw", "invoerbtw", "import tax", "customs"
    ]
    is_import_vat = False
    if invoice_text:
        text_lower = invoice_text.lower()
        is_import_vat = any(keyword in text_lower for keyword in import_vat_keywords)
    
    # Classify based on invoice type
    if invoice_type == "Sales":
        # For sales invoices, check customer country
        if customer_country:
            is_customer_eu = _is_eu_country(customer_country)
            
            # Check VAT rate for subcategory
            if vat_percentage is not None:
                vat_rate = float(vat_percentage)
                # Standard 21% rate
                if abs(vat_rate - 21.0) < 0.1 or (20.0 <= vat_rate <= 22.0):
                    if is_customer_eu:
                        return "Standard 21% - Sales to EU Countries"
                    else:
                        return "Standard 21% - Sales to Non-EU Countries"
                # Reduced 9% rate
                elif abs(vat_rate - 9.0) < 0.1 or (8.0 <= vat_rate <= 10.0):
                    if is_customer_eu:
                        return "Reduced Rate 9% - Sales to EU Countries"
                    else:
                        return "Reduced Rate 9% - Sales to Non-EU Countries"
                # Other rates
                else:
                    if is_customer_eu:
                        return f"VAT {vat_rate}% - Sales to EU Countries"
                    else:
                        return f"VAT {vat_rate}% - Sales to Non-EU Countries"
            else:
                # No VAT rate, classify by country only
                if is_customer_eu:
                    return "Sales to EU Countries"
                else:
                    return "Sales to Non-EU Countries"
        else:
            # No customer country found, use VAT rate if available
            if vat_percentage is not None:
                vat_rate = float(vat_percentage)
                if abs(vat_rate - 21.0) < 0.1 or (20.0 <= vat_rate <= 22.0):
                    return "Standard 21%"
                elif abs(vat_rate - 9.0) < 0.1 or (8.0 <= vat_rate <= 10.0):
                    return "Reduced Rate 9%"
                else:
                    return f"VAT {vat_rate}%"
            return "Sales - Country Unknown"
    
    elif invoice_type == "Purchase":
        # For purchase invoices, check vendor country
        if vendor_country:
            is_vendor_eu = _is_eu_country(vendor_country)
            is_vendor_nl = _is_nl_country(vendor_country)
            
            log.info(f"Purchase invoice - Vendor country: {vendor_country}, EU: {is_vendor_eu}, NL: {is_vendor_nl}")
            
            if is_vendor_eu and not is_vendor_nl:
                return "Purchase from EU Countries"
            elif is_vendor_nl:
                return "Purchase from NL Countries"
            else:
                # Non-EU purchase - check if it's import VAT
                if is_import_vat or (vat_percentage is not None and vat_percentage > 0):
                    return "Purchase from Non-EU Countries (Import VAT)"
                else:
                    return "Purchase from Non-EU Countries"
        else:
            # No vendor country found, check for import VAT indicators
            log.warning(f"Purchase invoice - No vendor country found. Vendor address: {vendor_address}")
            if is_import_vat:
                return "Purchase from Non-EU Countries (Import VAT)"
            return "Purchase - Country Unknown"
    
    # Unclassified invoices
    return "Unclassified"

def _determine_dutch_vat_return_category(
    invoice_type: str,
    vendor_country: Optional[str],
    customer_country: Optional[str],
    vat_percentage: Optional[float],
    vat_amount: Optional[float],
    goods_services_indicator: Optional[str],
    reverse_charge_applied: bool,
    customer_vat_id: Optional[str],
    vendor_vat_id: Optional[str],
    vat_category: Optional[str],
) -> Optional[str]:
    """
    Determines the Dutch VAT return category according to the Dutch VAT return boxes:
      - 1a: Sales taxed at standard rate (21%)
      - 1b: Sales taxed at reduced rate (9%)
      - 1c: Sales taxed at other rates / zero‑rated domestic supplies
      - 1d: Private use of business assets  (typically manual adjustments; rarely auto‑detected)
      - 1e: Sales exempt from VAT / out‑of‑scope
      - 2a: Reverse‑charge supplies (domestic reverse‑charge)
      - 3a: Supplies of goods to EU countries (B2B)
      - 3b: Supplies of services to EU countries (B2B)
      - 3c: Distance / installation sales to EU private individuals (B2C, no VAT ID)
      - 4a: Purchases of goods from EU countries
      - 4b: Purchases of services from EU countries
      - 5a: Input VAT on domestic purchases with Dutch VAT

    Logic is based on:
      - Sales vs Purchase invoice
      - Supplier/Customer country (NL, EU, non‑EU)
      - VAT rate and VAT amount
      - Goods vs Services
      - Reverse charge signal
      - Availability of VAT numbers
      - High‑level VAT category (standard / zero‑rated / reverse‑charge / out‑of‑scope)

    Returns the category code or None if no match.
    """
    # Normalize VAT percentage
    vat_rate = None
    if vat_percentage is not None:
        vat_rate = float(vat_percentage)
    
    # Normalise VAT category from LLM ("standard", "zero-rated", "reverse-charge", "out-of-scope")
    vat_cat = (vat_category or "").strip().lower()
    is_zero_rated = vat_cat == "zero-rated"
    is_out_of_scope_or_exempt = vat_cat in {"out-of-scope", "exempt", "exempt-supplies"}

    # Check if customer/vendor is NL
    is_customer_nl = _is_nl_country(customer_country) if customer_country else False
    is_vendor_nl = _is_nl_country(vendor_country) if vendor_country else False
    
    # Check if customer/vendor is EU (but not NL) - for Dutch VAT categories, we need EU excluding NL
    is_customer_eu = (_is_eu_country(customer_country) and not _is_nl_country(customer_country)) if customer_country else False
    is_vendor_eu = (_is_eu_country(vendor_country) and not _is_nl_country(vendor_country)) if vendor_country else False
    
    # Check if customer/vendor is non-EU (must not be EU AND not be NL)
    is_customer_non_eu = customer_country and not is_customer_eu and not is_customer_nl
    is_vendor_non_eu = vendor_country and not is_vendor_eu and not is_vendor_nl
    
    # Check if VAT number is present
    has_customer_vat = bool(customer_vat_id and customer_vat_id.strip())
    has_vendor_vat = bool(vendor_vat_id and vendor_vat_id.strip())
    
    # SALES INVOICES
    if invoice_type == "Sales":
        # ---- BOX 1: Domestic sales (NL customer) ----
        if is_customer_nl:
            # 1a = NL customer, VAT 21%
            if vat_rate is not None and abs(vat_rate - 21.0) < 0.1:
                return "1a"

            # 1b = NL customer, VAT 9%
            if vat_rate is not None and abs(vat_rate - 9.0) < 0.1:
                return "1b"

            # 2a = NL customer, reverse charge (VAT 0%)
            if reverse_charge_applied and (vat_rate is None or abs(vat_rate - 0.0) < 0.01):
                return "2a"

            # 1e = Sales exempt from VAT / out-of-scope (no VAT charged, exempt category)
            if is_out_of_scope_or_exempt and (vat_rate is None or abs(vat_rate - 0.0) < 0.01):
                return "1e"

            # 1c = Sales taxed at other rates or zero‑rated domestic supplies
            # Any domestic sales with 0% VAT (not reverse charge, not exempt),
            # or with a positive VAT rate that is not 21% or 9%.
            if vat_rate is not None:
                if abs(vat_rate - 0.0) < 0.01 and not reverse_charge_applied and not is_out_of_scope_or_exempt:
                    return "1c"
                if vat_rate > 0.01 and abs(vat_rate - 21.0) >= 0.1 and abs(vat_rate - 9.0) >= 0.1:
                    return "1c"

        # ---- BOX 3: Intra‑EU supplies ----
        # 3a = EU customer (not NL), goods, VAT 0%, VAT number present
        if (
            is_customer_eu
            and goods_services_indicator == "goods"
            and vat_rate is not None
            and abs(vat_rate - 0.0) < 0.01
            and has_customer_vat
        ):
            return "3a"

        # 3b = EU customer (not NL), services, VAT 0%, VAT number present
        if (
            is_customer_eu
            and goods_services_indicator == "services"
            and vat_rate is not None
            and abs(vat_rate - 0.0) < 0.01
            and has_customer_vat
        ):
            return "3b"

        # 3c = Supplies of goods to EU private individuals (no VAT ID, distance / installation sales)
        # Heuristic: EU (not NL) customer, GOODS, no VAT ID → Box 3c.
        if (
            is_customer_eu
            and goods_services_indicator == "goods"
            and not has_customer_vat
        ):
            return "3c"

        # Exports to non‑EU with 0% VAT are typically zero‑rated supplies.
        # These are often reported in 1c or 1e depending on exact Dutch guidance;
        # we map them to 1c as zero‑rated non‑EU exports.
        if is_customer_non_eu and vat_rate is not None and abs(vat_rate - 0.0) < 0.01:
            return "1c"
    
    # PURCHASE INVOICES
    elif invoice_type == "Purchase":
        # 4a = EU supplier, goods
        # Note: VAT rate on invoice is supplier's country rate, but Dutch VAT is self-accounted
        # So we only check: EU supplier (not NL) + goods indicator
        if (is_vendor_eu and not is_vendor_nl and 
            goods_services_indicator == "goods"):
            return "4a"
        
        # 4b = EU supplier, services
        # Note: VAT rate on invoice is supplier's country rate, but Dutch VAT is self-accounted
        # So we only check: EU supplier (not NL) + services indicator
        if (is_vendor_eu and not is_vendor_nl and 
            goods_services_indicator == "services"):
            return "4b"
        
        # 5a = NL supplier, Dutch VAT charged
        # Dutch VAT is typically 21% or 9%, and there should be VAT amount > 0
        # Check for Dutch VAT rates (21%, 9%, or any VAT > 0 with VAT amount > 0)
        if is_vendor_nl:
            if vat_rate is not None and vat_rate > 0.01:
                # Has VAT rate > 0
                if vat_amount is None or vat_amount > 0:
                    return "5a"
            elif vat_amount is not None and vat_amount > 0:
                # Has VAT amount even if rate not explicitly stated
                return "5a"
    
    # No match - return None (will be set to empty string in mapping)
    return None


# Descriptions for Dutch VAT return categories (for transaction register)
DUTCH_VAT_CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "1a": "Sales taxed at standard rate (21%)",
    "1b": "Sales taxed at reduced rate (9%)",
    "1c": "Sales taxed at other rates / zero-rated supplies",
    "1d": "Private use of business assets",
    "1e": "Sales exempt from VAT / out-of-scope",
    "2a": "Reverse-charge supplies (domestic)",
    "3a": "Supplies of goods to EU countries (B2B)",
    "3b": "Supplies of services to EU countries (B2B)",
    "3c": "Distance/installation sales to EU private individuals (B2C)",
    "4a": "Purchases of goods from EU countries",
    "4b": "Purchases of services from EU countries",
    "5a": "Input VAT on domestic purchases (Dutch VAT)",
}


def _set_icp_fields_for_nl(register_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determine ICP reporting flags for a Dutch company (Netherlands) for a single transaction.

    Implements the following logic:
      Step 1 – Only NL company:
        - We assume the company is established in the Netherlands.
      Step 2 – ICP‑relevant transaction (Netherlands perspective):
        - Counterparty country is an EU country other than the Netherlands
        - B2B (we approximate B2B as: counterparty VAT number present)
        - Counterparty VAT number is not empty
        - Dutch VAT Return Category in {2a, 3a, 3b, 4a}
        - Reverse charge applies
        - Exclude purchases/acquisitions from other EU countries (only Sales are ICP‑relevant)
      Step 3 – Assign ICP reporting category label based on Dutch VAT category.

    Adds / updates on register_entry:
      - "ICP Return Required": "Yes" or "No"   (per transaction)
      - "ICP Reporting Category": human‑readable label for ICP‑relevant lines
    """
    # Defaults – assume no ICP unless all conditions are met
    register_entry["ICP Return Required"] = "No"
    register_entry["ICP Reporting Category"] = ""

    invoice_type = register_entry.get("Type")
    if invoice_type != "Sales":
        # ICP report from NL perspective only covers outbound intra‑EU B2B supplies,
        # not purchases/acquisitions from other EU countries.
        return register_entry

    # Determine counterparty details from our perspective
    counterparty_country = register_entry.get("Customer Country")
    counterparty_vat_number = register_entry.get("Customer VAT ID")

    if not counterparty_country:
        return register_entry

    # Counterparty must be in EU but not Netherlands
    if not _is_eu_country(counterparty_country) or _is_nl_country(counterparty_country):
        return register_entry

    # B2B only – require a non‑empty VAT number
    if not counterparty_vat_number or not str(counterparty_vat_number).strip():
        return register_entry

    # VAT category must be one of the ICP‑relevant Dutch VAT boxes
    dutch_vat_category = (register_entry.get("Dutch VAT Return Category") or "").strip().lower()
    icp_relevant_categories = {"2a", "3a", "3b", "4a"}
    if dutch_vat_category not in icp_relevant_categories:
        return register_entry

    # Reverse charge must apply
    if not register_entry.get("Reverse Charge Applied", False):
        return register_entry

    # Map Dutch VAT category to ICP reporting category label
    icp_reporting_category_map = {
        "2a": "Intra-EU supply of goods (B2B)",
        "3a": "Intra-EU supply of services (B2B, reverse charge)",
        "3b": "Adjustments/credit notes for ICP goods/services",
        "4a": "Other EU B2B reverse-charge transactions",
    }

    register_entry["ICP Return Required"] = "Yes"
    register_entry["ICP Reporting Category"] = icp_reporting_category_map.get(dutch_vat_category, "")
    return register_entry

def _derive_vat_rate_percent(llm_data: Dict[str, Any]) -> Optional[float]:
    # look in vat_breakdown
    for v in (llm_data.get("vat_breakdown") or []):
        r = v.get("rate")
        if isinstance(r, (int, float)):
            try:
                return float(r)
            except Exception:
                pass
    # fallback based on category
    vcat = (llm_data.get("vat_category") or "").strip().lower()
    if vcat in {"reverse-charge", "out-of-scope", "zero-rated"}:
        return 0.0
    return None

def _map_llm_output_to_register_entry(llm_data: Dict[str, Any]) -> Dict[str, Any]:
    description = ""
    if llm_data.get("line_items"):
        raw_desc = llm_data["line_items"][0].get("description", "") or ""
        # If the description is in Dutch, convert it to English. Otherwise keep as-is.
        description = _translate_to_english_if_dutch(raw_desc)

    vat_percentage = _derive_vat_rate_percent(llm_data)
    invoice_text = llm_data.get("_invoice_text", "")

    # Heuristic confidence score for extraction quality
    extraction_confidence_level, extraction_confidence_reason = _estimate_extraction_confidence(
        invoice_text, llm_data
    )
    
    # Determine goods/services indicator
    goods_services_indicator = _determine_goods_services_indicator(llm_data, invoice_text)
    
    # Check for reverse charge - check VAT category and invoice text for keywords
    vat_category = (llm_data.get("vat_category") or "").strip().lower()
    invoice_text_lower = (invoice_text or "").lower()
    reverse_charge_keywords = [
        "reverse charge", "reverse-charge", "reversecharge",
        "btw verlegd", "btwverlegd", "vat verlegd", "vatverlegd",
        "omgekeerde heffing", "omgekeerdeheffing"
    ]
    reverse_charge_applied = (
        vat_category == "reverse-charge" or 
        any(keyword in invoice_text_lower for keyword in reverse_charge_keywords)
    )

    # Extract countries from addresses
    vendor_address = llm_data.get("vendor_address")
    customer_address = llm_data.get("customer_address")
    vendor_country = _extract_country_from_address(vendor_address) if vendor_address else None
    customer_country = _extract_country_from_address(customer_address) if customer_address else None
    
    # Extract reverse charge note if present
    reverse_charge_note = None
    if reverse_charge_applied and invoice_text:
        # Try to find the sentence or phrase mentioning reverse charge
        reverse_charge_patterns = [
            r"(?i)(?:reverse\s+charge|btw\s+verlegd|vat\s+verlegd|omgekeerde\s+heffing)[^.]*",
            r"(?i)vat[^.]*(?:verlegd|reverse)[^.]*",
        ]
        for pattern in reverse_charge_patterns:
            match = re.search(pattern, invoice_text)
            if match:
                reverse_charge_note = match.group(0).strip()
                break

    return {
        "Date": llm_data.get("invoice_date"),
        "Invoice Number": llm_data.get("invoice_number"),
        "Type": "Unclassified",  # Will be set by _classify_type
        "Vendor Name": llm_data.get("vendor_name"),
        "Vendor VAT ID": llm_data.get("vendor_vat_id"),
        "Vendor Country": vendor_country,  # Extracted from address
        "Vendor Address": vendor_address,
        "Vendor IBAN": llm_data.get("vendor_iban"),  # From LLM extraction
        "Customer Name": llm_data.get("customer_name"),
        "Customer VAT ID": llm_data.get("customer_vat_id"),
        "Customer Country": customer_country,  # Extracted from address
        "Customer Address": customer_address,
        "Customer IBAN": llm_data.get("customer_iban"),  # From LLM extraction
        "Description": description,
        "Nett Amount": float(q_money(llm_data.get("subtotal") or 0.0)),
        "VAT %": vat_percentage,
        "VAT Amount": float(q_money(llm_data.get("total_vat") or 0.0)),
        "Gross Amount": float(q_money(llm_data.get("total_amount") or 0.0)),
        "Currency": (llm_data.get("currency") or "EUR"),
        "VAT Category": llm_data.get("vat_category"),
        "Reverse Charge Applied": reverse_charge_applied,
        "Reverse Charge Note": reverse_charge_note,  # Extracted note if reverse charge
        "Goods Services Indicator": goods_services_indicator,
        "Subcategory": "Unclassified",  # Will be set after type classification
        "Dutch VAT Return Category": None,  # Will be set after type classification
        # ICP fields (Netherlands – will be populated after VAT/category classification)
        "ICP Return Required": "No",            # "Yes"/"No" per transaction
        "ICP Reporting Category": "",           # Human‑readable ICP category label
        # Extraction quality signal (for manual review / dashboards)
        "Extraction Confidence": extraction_confidence_level,
        "Extraction Confidence Reason": extraction_confidence_reason,
        "Due Date": llm_data.get("due_date"),  # Payment due date
        "Payment Terms": llm_data.get("payment_terms"),  # Payment terms/notes
        "Notes": llm_data.get("notes"),  # General notes/comments
        "Full_Extraction_Data": llm_data
    }
def _classify_type(register_entry: Dict[str, Any], our_companies_list: List[str]) -> str:
    """
    Classifies invoice as Purchase or Sales based on whether our company appears
    as customer (Purchase) or vendor (Sales).
    
    Uses multiple matching strategies with similarity scoring for robust classification.
      """
    vendor_name = register_entry.get("Vendor Name") or ""
    customer_name = register_entry.get("Customer Name") or ""
    if not vendor_name and not customer_name:
        log.warning("Both vendor and customer names are empty - cannot classify")
        return "Unclassified"
    v = _normalize_company_name(vendor_name)
    c = _normalize_company_name(customer_name)
    ours = [_normalize_company_name(x) for x in our_companies_list]
    
    # Track best matches with similarity scores
    best_purchase_match = {"score": 0.0, "our_company": "", "customer": ""}
    best_sales_match = {"score": 0.0, "our_company": "", "vendor": ""}
    
    # Strategy 1: Exact substring match (normalized) - highest priority
    for our_company in ours:
        if not our_company:
            continue
        
        # Check if our company name appears in customer name (Purchase)
        if our_company in c:
            score = len(our_company) / len(c) if c else 1.0
            if score > best_purchase_match["score"]:
                best_purchase_match = {"score": score, "our_company": our_company, "customer": customer_name}
        
        # Check if our company name appears in vendor name (Sales)
        if our_company in v:
            score = len(our_company) / len(v) if v else 1.0
            if score > best_sales_match["score"]:
                best_sales_match = {"score": score, "our_company": our_company, "vendor": vendor_name}
    
    # Strategy 2: Reverse check - check if customer/vendor name appears in our company names
    for our_company in ours:
        if not our_company:
            continue
        
        # Check if customer name appears in our company name (Purchase)
        if c and c in our_company:
            score = len(c) / len(our_company) if our_company else 1.0
            if score > best_purchase_match["score"]:
                best_purchase_match = {"score": score, "our_company": our_company, "customer": customer_name}
        
        # Check if vendor name appears in our company name (Sales)
        if v and v in our_company:
            score = len(v) / len(our_company) if our_company else 1.0
            if score > best_sales_match["score"]:
                best_sales_match = {"score": score, "our_company": our_company, "vendor": vendor_name}
    
    # Strategy 3: Word-based matching with similarity scoring
    for our_company in ours:
        if not our_company:
            continue
        
        # Calculate similarity with customer
        customer_similarity = _calculate_name_similarity(our_company, customer_name)
        if customer_similarity > best_purchase_match["score"] and customer_similarity >= 0.6:  # 60% threshold
            best_purchase_match = {"score": customer_similarity, "our_company": our_company, "customer": customer_name}
        
        # Calculate similarity with vendor
        vendor_similarity = _calculate_name_similarity(our_company, vendor_name)
        if vendor_similarity > best_sales_match["score"] and vendor_similarity >= 0.6:  # 60% threshold
            best_sales_match = {"score": vendor_similarity, "our_company": our_company, "vendor": vendor_name}
    
    # Strategy 4: Key word matching (for partial matches like "Dutch Food" matching "Dutch Food Solutions")
    for our_company in ours:
        if not our_company:
            continue
        
        # Extract significant words (length > 3 to avoid common words)
        our_words = set([w for w in our_company.split() if len(w) > 3])
        if len(our_words) >= 2:  # Need at least 2 significant words
            # Check customer name
            c_words = set([w for w in c.split() if len(w) > 3])
            if our_words and c_words:
                overlap = our_words.intersection(c_words)
                if len(overlap) >= 2:  # At least 2 words match
                    score = len(overlap) / len(our_words)
                    if score > best_purchase_match["score"]:
                        best_purchase_match = {"score": score, "our_company": our_company, "customer": customer_name}
            
            # Check vendor name
            v_words = set([w for w in v.split() if len(w) > 3])
            if our_words and v_words:
                overlap = our_words.intersection(v_words)
                if len(overlap) >= 2:  # At least 2 words match
                    score = len(overlap) / len(our_words)
                    if score > best_sales_match["score"]:
                        best_sales_match = {"score": score, "our_company": our_company, "vendor": vendor_name}
    
    # Determine classification based on best matches
    # Purchase takes priority if both match (customer is more reliable indicator)
    if best_purchase_match["score"] > 0.5:  # 50% threshold for purchase
        log.info(f"Classified as Purchase: similarity {best_purchase_match['score']:.2f} - '{best_purchase_match['our_company']}' matches customer '{best_purchase_match['customer']}'")
        return "Purchase"
    
    if best_sales_match["score"] > 0.5:  # 50% threshold for sales
        log.info(f"Classified as Sales: similarity {best_sales_match['score']:.2f} - '{best_sales_match['our_company']}' matches vendor '{best_sales_match['vendor']}'")
        return "Sales"
    
    # If no match, log for debugging
    log.warning(f"Could not classify invoice. Vendor: '{vendor_name}', Customer: '{customer_name}', Our companies: {our_companies_list}")
    log.warning(f"Normalized - Vendor: '{v}', Customer: '{c}', Our normalized: {ours}")
    log.warning(f"Best matches - Purchase: {best_purchase_match['score']:.2f}, Sales: {best_sales_match['score']:.2f}")
    return "Unclassified"

def _classify_and_set_subcategory(register_entry: Dict[str, Any], our_companies_list: List[str]) -> Dict[str, Any]:
    """
    Classifies the invoice type and determines the subcategory and Dutch VAT return category.
    This should be called after _map_llm_output_to_register_entry.
    Updates the register_entry with Type, Subcategory, and Dutch VAT Return Category.
    
    Uses fallback logic to determine type if initial classification fails.
    """
    # First classify the type
    invoice_type = _classify_type(register_entry, our_companies_list)
    
    # Fallback: If unclassified, try to infer from addresses and VAT patterns
    if invoice_type == "Unclassified":
        vendor_address = register_entry.get("Vendor Address") or ""
        customer_address = register_entry.get("Customer Address") or ""
        vendor_country = _extract_country_from_address(vendor_address)
        customer_country = _extract_country_from_address(customer_address)
        vat_percentage = register_entry.get("VAT %")
        
        # Heuristic: If vendor is NL and customer is non-NL, likely Sales
        # If customer is NL and vendor is non-NL, likely Purchase
        if vendor_country and customer_country:
            is_vendor_nl = _is_nl_country(vendor_country)
            is_customer_nl = _is_nl_country(customer_country)
            
            if is_vendor_nl and not is_customer_nl:
                invoice_type = "Sales"
                log.info(f"Inferred type as Sales: NL vendor ({vendor_country}) to non-NL customer ({customer_country})")
            elif is_customer_nl and not is_vendor_nl:
                invoice_type = "Purchase"
                log.info(f"Inferred type as Purchase: NL customer ({customer_country}) from non-NL vendor ({vendor_country})")
            elif is_vendor_nl and is_customer_nl:
                # Both NL - check VAT pattern (NL sales usually have VAT, purchases from NL have VAT)
                if vat_percentage and vat_percentage > 0:
                    # Could be either, but default to Purchase if customer matches our companies better
                    # This is a conservative approach
                    invoice_type = "Purchase"  # Default assumption
                    log.info(f"Inferred type as Purchase: Both NL, defaulting to Purchase")
    
    register_entry["Type"] = invoice_type
    
    # Then determine subcategory based on type, addresses, and VAT
    vendor_address = register_entry.get("Vendor Address")
    customer_address = register_entry.get("Customer Address")
    vat_percentage = register_entry.get("VAT %")
    
    # Get invoice text from full extraction data if available
    invoice_text = ""
    full_data = register_entry.get("Full_Extraction_Data", {})
    if isinstance(full_data, dict):
        invoice_text = full_data.get("_invoice_text", "")
    
    subcategory = _determine_invoice_subcategory(
        invoice_type=invoice_type,
        vendor_address=vendor_address,
        customer_address=customer_address,
        vat_percentage=vat_percentage,
        invoice_text=invoice_text
    )
    register_entry["Subcategory"] = subcategory
    
    # Determine Dutch VAT return category
    # Use countries already extracted in register entry, or extract from addresses if not present
    vendor_country = register_entry.get("Vendor Country")
    customer_country = register_entry.get("Customer Country")
    
    # If countries not in register entry, extract from addresses
    if not vendor_country and vendor_address:
        vendor_country = _extract_country_from_address(vendor_address)
    if not customer_country and customer_address:
        customer_country = _extract_country_from_address(customer_address)
    
    # For VAT classification, prioritize VAT ID country over billing address country
    # VAT ID country is more reliable for VAT purposes (e.g., French VAT ID = France, even if billing address is NL)
    
    # Extract country from VAT ID if available (more reliable for VAT classification)
    customer_vat_id = register_entry.get("Customer VAT ID")
    vendor_vat_id = register_entry.get("Vendor VAT ID")
    
    # VAT ID country codes mapping
    vat_id_country_map = {
        "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "HR": "Croatia",
        "CY": "Cyprus", "CZ": "Czech Republic", "DK": "Denmark", "EE": "Estonia",
        "FI": "Finland", "FR": "France", "DE": "Germany", "GR": "Greece",
        "HU": "Hungary", "IE": "Ireland", "IT": "Italy", "LV": "Latvia",
        "LT": "Lithuania", "LU": "Luxembourg", "MT": "Malta", "NL": "Netherlands",
        "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SK": "Slovakia",
        "SI": "Slovenia", "ES": "Spain", "SE": "Sweden",
        "GB": "United Kingdom", "UK": "United Kingdom"
    }
    
    # If customer VAT ID exists, extract country code and use it for VAT classification
    if customer_vat_id:
        vat_id_upper = customer_vat_id.strip().upper()
        # VAT IDs typically start with 2-letter country code (e.g., "FR49794194852")
        for country_code, country_name in vat_id_country_map.items():
            if vat_id_upper.startswith(country_code):
                customer_country = country_name
                log.info(f"Using VAT ID country for customer: {country_name} (from VAT ID: {customer_vat_id})")
                break
    
    # If vendor VAT ID exists and differs from address, use it for vendor country
    if vendor_vat_id:
        vat_id_upper = vendor_vat_id.strip().upper()
        for country_code, country_name in vat_id_country_map.items():
            if vat_id_upper.startswith(country_code):
                # Only override if address country is different or missing
                if not vendor_country or vendor_country != country_name:
                    vendor_country = country_name
                    log.info(f"Using VAT ID country for vendor: {country_name} (from VAT ID: {vendor_vat_id})")
                break
    
    vat_amount = register_entry.get("VAT Amount")
    goods_services_indicator = register_entry.get("Goods Services Indicator")
    reverse_charge_applied = register_entry.get("Reverse Charge Applied", False)
    vat_category = register_entry.get("VAT Category")
    
    dutch_vat_category = _determine_dutch_vat_return_category(
        invoice_type=invoice_type,
        vendor_country=vendor_country,
        customer_country=customer_country,
        vat_percentage=vat_percentage,
        vat_amount=vat_amount,
        goods_services_indicator=goods_services_indicator,
        reverse_charge_applied=reverse_charge_applied,
        customer_vat_id=customer_vat_id,
        vendor_vat_id=vendor_vat_id,
        vat_category=vat_category,
    )
    # Set to empty string if None (as per requirements)
    register_entry["Dutch VAT Return Category"] = dutch_vat_category if dutch_vat_category else ""
    
    # Human-readable description for the Dutch VAT category
    if dutch_vat_category:
        register_entry["Dutch VAT Return Category Description"] = DUTCH_VAT_CATEGORY_DESCRIPTIONS.get(
            dutch_vat_category, ""
        )
    else:
        register_entry["Dutch VAT Return Category Description"] = ""

    # Set ICP flags/labels for this transaction according to Dutch ICP rules
    register_entry = _set_icp_fields_for_nl(register_entry)
    
    # Log classification results for debugging
    log.info(
        f"Final classification - Type: {invoice_type}, "
        f"Subcategory: {subcategory}, "
        f"Dutch VAT Category: {dutch_vat_category or 'None'}, "
        f"ICP Return Required: {register_entry.get('ICP Return Required')}, "
        f"ICP Reporting Category: {register_entry.get('ICP Reporting Category')}"
    )
    
    return register_entry

def _convert_to_eur_fields(entry: dict, conversion_enabled: bool = True) -> dict:
    """
    Adds EUR-converted fields based on invoice date & currency using exchangerate.host.
    Adds: FX Rate (ccy->EUR), Nett Amount (EUR), VAT Amount (EUR), Gross Amount (EUR)
    """
    if not conversion_enabled:
        entry["FX Rate (ccy->EUR)"] = None
        entry["Nett Amount (EUR)"] = None
        entry["VAT Amount (EUR)"] = None
        entry["Gross Amount (EUR)"] = None
        entry["FX Conversion Note"] = "Currency conversion disabled"
        return entry
    try:
        ccy = (entry.get("Currency") or "").upper().strip()
        inv_date_str = entry.get("Date")

        # Missing date or currency → cannot convert, but don't crash the pipeline
        if not inv_date_str or not ccy:
            entry["FX Rate (ccy->EUR)"] = None
            entry["Nett Amount (EUR)"] = None
            entry["VAT Amount (EUR)"] = None
            entry["Gross Amount (EUR)"] = None
            entry["FX Error"] = "Missing date or currency for conversion"
            return entry

        # Already in EUR: copy numbers directly and use rate 1.0
        if ccy == "EUR":
            entry["FX Rate (ccy->EUR)"] = "1.0000"
            entry["FX Rate Date"] = inv_date_str
            entry["Nett Amount (EUR)"] = round(float(entry.get("Nett Amount", 0) or 0), 2)
            entry["VAT Amount (EUR)"] = round(float(entry.get("VAT Amount", 0) or 0), 2)
            entry["Gross Amount (EUR)"] = round(float(entry.get("Gross Amount", 0) or 0), 2)
            return entry

        inv_dt = date.fromisoformat(inv_date_str)
        rate, used_date = get_eur_rate(inv_dt, ccy)
        entry["FX Rate (ccy->EUR)"] = str(rate)
        entry["FX Rate Date"] = used_date

        # Convert amounts using Decimal for precision
        for k_src, k_dst in [
            ("Nett Amount", "Nett Amount (EUR)"),
            ("VAT Amount",  "VAT Amount (EUR)"),
            ("Gross Amount","Gross Amount (EUR)")
        ]:
            amt = Decimal(str(entry.get(k_src, 0) or 0))
            converted = q_money(amt * rate)
            entry[k_dst] = round(float(converted), 2)

        return entry
    except Exception as ex:
        # Do not fail the whole invoice if FX fetch fails—attach note.
        log.error(f"EUR conversion failed for entry: {ex}")
        entry["FX Rate (ccy->EUR)"] = None
        entry["Nett Amount (EUR)"] = None
        entry["VAT Amount (EUR)"] = None
        entry["Gross Amount (EUR)"] = None
        entry["FX Error"] = f"EUR conversion failed: {str(ex)}"
        return entry

# -------------------- Main pipeline --------------------
def robust_invoice_processor(pdf_bytes: bytes, filename: str) -> dict:
    # Supports PDFs and common image formats (JPEG/PNG/TIFF/BMP/GIF).
    # Uses AWS Textract + Tesseract under the hood.
    invoice_text = get_text_from_document(pdf_bytes, filename)
    llm_data = structure_text_with_llm(invoice_text, filename)
    _ = validate_extraction(llm_data, filename)
    # attach raw text to Full_Extraction_Data for downstream heuristics if needed
    llm_data["_invoice_text"] = invoice_text[:20000]
    return llm_data

# -------------------- Posting rules (data-driven) --------------------
# Default COA codes (override per client in DB later)
DEFAULT_COA = {
    "AR": "1100",
    "AP": "2000",
    "SALES": "4000",
    "COGS": "5000",
    "FREIGHT": "5100",
    "SOFTWARE": "5200",
    "VAT_PAYABLE": "2100",
    "VAT_RECOVERABLE": "1400",
    "CASH": "1000",
}

# Rule format supports client overrides later (condition + posting)
DEFAULT_RULES = [
    # Sales with VAT
    {
        "condition": {"Type": "Sales"},
        "posting": {
            "dr": [{"account": "AR", "amount": "Gross Amount (EUR)"}],
            "cr": [
                {"account": "SALES", "amount": "Nett Amount (EUR)"},
                {"account": "VAT_PAYABLE", "amount": "VAT Amount (EUR)"},
            ],
        },
    },
    # Purchase zero-rated (e.g., freight)
    {
        "condition": {"Type": "Purchase", "VAT Category": "Zero Rated"},
        "posting": {
            "dr": [{"account": "FREIGHT", "amount": "Nett Amount (EUR)"}],
            "cr": [{"account": "AP", "amount": "Gross Amount (EUR)"}],
        },
    },
    # Example vendor-specific rule (foreign VAT as non-recoverable)
    {
        "condition": {"Type": "Unclassified", "Vendor Name_regex": ".*Google Cloud.*"},
        "posting": {
            "dr": [{"account": "SOFTWARE", "amount": "Gross Amount (EUR)"}],
            "cr": [{"account": "AP", "amount": "Gross Amount (EUR)"}],
        },
    },
]

def _match_rule(entry: dict, rules: list) -> Optional[dict]:
    import re as _re
    for r in rules:
        cond = r.get("condition", {})
        ok = True
        for k, v in cond.items():
            if k.endswith("_regex"):
                field = k[:-6]
                if not _re.match(str(v), str(entry.get(field, "") or ""), flags=_re.I):
                    ok = False; break
            else:
                if str(entry.get(k, "")).strip() != str(v).strip():
                    ok = False; break
        if ok:
            return r
    return None

def _amount_field_to_value(entry: dict, field: str) -> float:
    # Prefer EUR fields if present
    if "(EUR)" in field:
        return float(entry.get(field) or 0.0)
    mapping = {
        "Nett Amount": "Nett Amount (EUR)",
        "VAT Amount": "VAT Amount (EUR)",
        "Gross Amount": "Gross Amount (EUR)",
    }
    f = mapping.get(field, field)
    return float(entry.get(f) if entry.get(f) is not None else entry.get(field, 0.0) or 0.0)

def build_journal_from_entry(entry: dict, coa: dict = None, rules: list = None) -> dict:
    coa = {**DEFAULT_COA, **(coa or {})}
    rules = rules or DEFAULT_RULES

    # Guard rails
    if entry.get("FX Error"):
        return {"status": "blocked", "reason": "Needs FX", "entry": entry}
    if not entry.get("Date"):
        return {"status": "blocked", "reason": "Missing Date", "entry": entry}

    rule = _match_rule(entry, rules)
    if not rule:
        return {"status": "blocked", "reason": "Unmapped", "entry": entry}

    lines = []
    for side in ("dr", "cr"):
        for post in rule["posting"].get(side, []):
            acct_key = post["account"]
            acct = coa.get(acct_key, acct_key)  # allow literal code
            amt = _amount_field_to_value(entry, post["amount"])
            lines.append({
                "account_code": acct,
                "debit": round(amt, 2) if side == "dr" else 0.0,
                "credit": round(amt, 2) if side == "cr" else 0.0,
            })

    d = round(sum(x["debit"] for x in lines), 2)
    c = round(sum(x["credit"] for x in lines), 2)
    if d != c:
        return {"status": "blocked", "reason": f"Imbalance {d} != {c}", "entry": entry, "lines": lines}

    return {
        "status": "posted",
        "journal": {
            "entry_date": entry["Date"],
            "memo": entry.get("Description") or f"{entry.get('Vendor Name')} / {entry.get('Invoice Number')}",
            "currency": entry.get("Currency"),
            "fx_rate": entry.get("FX Rate (ccy->EUR)"),
            "client_id": entry.get("client_id"),
            "lines": lines,
        }
    }

# -------------------- Public API for app.py --------------------
__all__ = [
    "robust_invoice_processor",
    "_map_llm_output_to_register_entry",
    "_classify_type",
    "_classify_and_set_subcategory",
    "_determine_invoice_subcategory",
    "_extract_country_from_address",
    "_is_eu_country",
    "_split_company_list",
    "_convert_to_eur_fields",
    "build_journal_from_entry",
    "DEFAULT_COA",
    "DEFAULT_RULES",
    "q_money",
]
