from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


@dataclass(frozen=True)
class TextDocument:
    path: str
    text: str


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into overlapping character chunks."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    n = len(cleaned)

    while start < n:
        end = min(start + chunk_size, n)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start += step

    return chunks


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError("Reading .pdf requires pypdf. Install with: pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def _read_epub(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
        from ebooklib import ITEM_DOCUMENT, epub
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "Reading .epub requires ebooklib and beautifulsoup4. Install with: pip install ebooklib beautifulsoup4"
        ) from exc

    book = epub.read_epub(str(path))
    blocks: list[str] = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        html = item.get_content()
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        text = _normalize_whitespace(soup.get_text(separator=" ", strip=True))
        if text:
            blocks.append(text)

    return "\n".join(blocks).strip()


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def load_text_documents(input_path: Path) -> list[TextDocument]:
    """Load documents from a file or directory for ingestion."""
    allowed_ext = {".txt", ".md", ".rst", ".py", ".pdf", ".epub"}
    path = input_path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    files: list[Path]
    if path.is_file():
        files = [path]
    else:
        files = sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in allowed_ext])

    docs: list[TextDocument] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix not in allowed_ext:
            continue

        if suffix == ".pdf":
            text = _read_pdf(file_path)
        elif suffix == ".epub":
            text = _read_epub(file_path)
        else:
            text = _read_text_file(file_path)

        if not text:
            continue

        docs.append(TextDocument(path=str(file_path), text=text))

    return docs
