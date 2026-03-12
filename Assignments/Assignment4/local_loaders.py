from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoadedDoc:
    source: str
    text: str


def load_txt(path: str | Path) -> LoadedDoc:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    return LoadedDoc(source=str(p), text=text)


def load_pdf(path: str | Path) -> LoadedDoc:
    from pypdf import PdfReader

    p = Path(path)
    reader = PdfReader(str(p))
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    text = "\n".join(pages_text)
    return LoadedDoc(source=str(p), text=text)


def load_docx(path: str | Path) -> LoadedDoc:
    import docx

    p = Path(path)
    d = docx.Document(str(p))
    text = "\n".join([para.text for para in d.paragraphs if para.text])
    return LoadedDoc(source=str(p), text=text)


def load_any(path: str | Path) -> LoadedDoc:
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".txt":
        return load_txt(p)
    if ext == ".pdf":
        return load_pdf(p)
    if ext == ".docx":
        return load_docx(p)

    raise ValueError(f"Unsupported file type: {ext}. Use .txt, .pdf, or .docx")
