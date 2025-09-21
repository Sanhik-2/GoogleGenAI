import streamlit as st
from io import BytesIO
import sys
from typing import List

st.set_page_config(page_title="Legal Document Analysis", layout="wide")

# Try to import PyPDF2 at the top
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

def extract_text_from_pdf_bytes(pdf_bytes: bytes, use_ocr: bool = False) -> str:
    if not PyPDF2:
        return "PyPDF2 not installed. Install with: pip install PyPDF2"

    text_pages: List[str] = []
    try:
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        for page in reader.pages:
            txt = page.extract_text() or ""
            text_pages.append(txt)
    except Exception as e:
        return f"PDF extraction error: {e}"

    text = "\n".join(text_pages).strip()
    if text:
        return text

    if use_ocr:
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
        except Exception:
            return "OCR dependencies missing. Install pdf2image and pytesseract to enable OCR."

        try:
            images = convert_from_bytes(pdf_bytes)
            ocr_text = [pytesseract.image_to_string(img) for img in images]
            return "\n".join(ocr_text)
        except Exception as e:
            return f"OCR extraction failed: {e}"
    return "No extractable text found. Try enabling OCR."

# Attempt to reuse repo analysis functions if present
summarize_fn = None
ner_fn = None
try:
    import importlib.util, pathlib
    repo_root = pathlib.Path(__file__).resolve().parent
    candidate = repo_root / "code" / "analysis.py"
    if candidate.exists():
        spec = importlib.util.spec_from_file_location("repo_analysis", str(candidate))
        repo_analysis = importlib.util.module_from_spec(spec)
        sys.modules["repo_analysis"] = repo_analysis
        spec.loader.exec_module(repo_analysis)
        summarize_fn = getattr(repo_analysis, "summarize_text", None)
        ner_fn = getattr(repo_analysis, "extract_entities", None)
except Exception as e:
    st.warning(f"Could not import analysis functions: {e}")

# Fallback to Hugging Face pipelines if repo functions not present
hf_summarizer = None
hf_ner = None
if not summarize_fn or not ner_fn:
    try:
        from transformers import pipeline
        if not summarize_fn:
            hf_summarizer = pipeline("summarization", model="google/flan-t5-small")
        if not ner_fn:
            hf_ner = pipeline("ner", grouped_entities=True)
    except Exception as e:
        st.warning(f"Could not load Hugging Face pipelines: {e}")

st.title("Legal Document Analysis")
uploaded = st.file_uploader("Upload PDF", type=["pdf"])
use_ocr = st.checkbox("Use OCR (for scanned PDFs)", value=False)
show_raw = st.checkbox("Show raw extracted text", value=False)

if uploaded:
    bytes_data = uploaded.read()
    with st.spinner("Extracting text..."):
        extracted = extract_text_from_pdf_bytes(bytes_data, use_ocr=use_ocr)

    if show_raw:
        st.subheader("Extracted Text")
        st.text_area("Raw text", extracted, height=300)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Summarize"):
            if summarize_fn:
                with st.spinner("Summarizing using repo function..."):
                    summary = summarize_fn(extracted)
            elif hf_summarizer:
                with st.spinner("Summarizing with Hugging Face model..."):
                    MAX_CHUNK = 1000
                    chunks = [extracted[i:i+MAX_CHUNK] for i in range(0, len(extracted), MAX_CHUNK)]
                    summaries = [hf_summarizer(c)[0]["summary_text"] for c in chunks]
                    summary = "\n\n".join(summaries)
            else:
                summary = "No summarization backend available. Install transformers or add summarize_text in code/analysis.py"
            st.subheader("Summary")
            st.write(summary)

    with col2:
        if st.button("Extract Entities"):
            if ner_fn:
                with st.spinner("Extracting entities using repo function..."):
                    entities = ner_fn(extracted)
            elif hf_ner:
                with st.spinner("Extracting entities with Hugging Face model..."):
                    entities = hf_ner(extracted)
            else:
                entities = "No NER backend available. Install transformers or add extract_entities in code/analysis.py"
            st.subheader("Named Entities")
            st.write(entities)

    st.markdown("---")
    st.caption("Notes: For OCR enable 'Use OCR' and ensure pdf2image + pytesseract are installed and configured. To reuse your repo logic, implement summarize_text(text)->str and extract_entities(text)->Any in code/analysis.py.")
else:
    st.info("Upload a PDF to start analysis.")