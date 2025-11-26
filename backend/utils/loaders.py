from pathlib import Path
from typing import Dict

from pypdf import PdfReader


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
PDF_FOLDER = DATA_ROOT / "pdfs"
TEXT_FOLDER = DATA_ROOT / "text"


def load_pdf(file_path: str | Path) -> str:
    """Load a single PDF and return its text"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def load_text(file_path: str | Path) -> str:
    """Load a single text file and return its content"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_all_pdfs() -> Dict[str, str]:
    """Load all PDFs from your PDF_FOLDER"""
    all_texts = {}
    for pdf_path in sorted(PDF_FOLDER.glob("*.pdf")):
        text = load_pdf(pdf_path)
        all_texts[pdf_path.name] = text
    return all_texts


def load_all_texts() -> Dict[str, str]:
    """Load all TXT files from your TEXT_FOLDER"""
    all_texts = {}
    for txt_path in sorted(TEXT_FOLDER.glob("*.txt")):
        text = load_text(txt_path)
        all_texts[txt_path.name] = text
    return all_texts


if __name__ == "__main__":
    pdfs = load_all_pdfs()
    texts = load_all_texts()

    print("PDFs loaded:")
    for name in pdfs:
        print(f"- {name} ({len(pdfs[name])} characters)")

    print("\nText files loaded:")
    for name in texts:
        print(f"- {name} ({len(texts[name])} characters)")
