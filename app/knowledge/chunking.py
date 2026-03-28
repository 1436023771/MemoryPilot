from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_EPUB_TOC_LINE_RE = re.compile(r"^第\s*\d+\s*话$")
_EPUB_CHAPTER_BREAK = "\n\n[[EPUB_CHAPTER_BREAK]]\n\n"


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


def _split_long_unit(unit: str, chunk_size: int) -> list[str]:
    """Split an oversized paragraph by sentence first, then by character fallback."""
    cleaned = unit.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    sentences = [s.strip() for s in re.split(r"(?<=[。！？!?\.])\s+", cleaned) if s.strip()]
    if len(sentences) <= 1:
        return split_text(cleaned, chunk_size=chunk_size, overlap=0)

    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        sent_len = len(sentence)
        if sent_len > chunk_size:
            if current:
                parts.append(" ".join(current).strip())
                current = []
                current_len = 0
            parts.extend(split_text(sentence, chunk_size=chunk_size, overlap=0))
            continue

        extra = sent_len if not current else sent_len + 1
        if current and current_len + extra > chunk_size:
            parts.append(" ".join(current).strip())
            current = [sentence]
            current_len = sent_len
        else:
            current.append(sentence)
            current_len += extra

    if current:
        parts.append(" ".join(current).strip())

    return [p for p in parts if p]


def split_epub_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split EPUB text with paragraph-aware windows to reduce narrative breaks."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chapter_texts = [cleaned]
    if _EPUB_CHAPTER_BREAK in cleaned:
        chapter_texts = [c.strip() for c in cleaned.split(_EPUB_CHAPTER_BREAK) if c.strip()]

    all_chunks: list[str] = []
    for chapter_text in chapter_texts:
        chapter_chunks = _split_epub_body_text(chapter_text, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chapter_chunks)

    return all_chunks


def _split_epub_body_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a single EPUB chapter body into paragraph-aware windows."""
    raw_units = [u.strip() for u in re.split(r"\n\s*\n+", text) if u.strip()]
    if not raw_units:
        return split_text(text, chunk_size=chunk_size, overlap=overlap)

    units: list[str] = []
    for unit in raw_units:
        units.extend(_split_long_unit(unit, chunk_size=chunk_size))

    if not units:
        return split_text(text, chunk_size=chunk_size, overlap=overlap)

    overlap_units = max(1, overlap // 240)
    chunks: list[str] = []
    i = 0
    n = len(units)

    while i < n:
        piece_parts: list[str] = []
        piece_len = 0
        j = i

        while j < n:
            block = units[j]
            extra = len(block) if not piece_parts else len(block) + 2
            if piece_parts and piece_len + extra > chunk_size:
                break
            piece_parts.append(block)
            piece_len += extra
            j += 1

        if not piece_parts:
            piece_parts = [units[i]]
            j = i + 1

        merged = "\n\n".join(piece_parts).strip()
        if merged:
            chunks.append(merged)

        next_i = max(i + 1, j - overlap_units)
        i = next_i

    return chunks


def split_document_text(text: str, path: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Use format-aware chunking strategy by file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".epub":
        return split_epub_text(text, chunk_size=chunk_size, overlap=overlap)
    return split_text(text, chunk_size=chunk_size, overlap=overlap)


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

    def _is_toc_like_text(text: str) -> bool:
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        if len(lines) < 12:
            return False
        matched = sum(1 for line in lines if _EPUB_TOC_LINE_RE.match(line))
        return matched >= 10 and (matched / max(1, len(lines))) >= 0.6

    def _extract_item_text(soup: BeautifulSoup) -> str:
        local_parts: list[str] = []
        for node in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote"]):
            text = _normalize_whitespace(node.get_text(separator=" ", strip=True))
            if text:
                local_parts.append(text)

        structured_text = "\n\n".join(local_parts).strip()
        fallback_text = _normalize_whitespace(soup.get_text(separator=" ", strip=True))

        if not structured_text:
            return fallback_text

        # If only headings/TOC were captured while the page has much more text,
        # prefer full-page extraction to avoid dropping正文.
        if _is_toc_like_text(structured_text) and len(fallback_text) >= max(1200, len(structured_text) * 2):
            return fallback_text

        if len(structured_text) < 240 and len(fallback_text) >= 1200:
            return fallback_text

        return structured_text

    book = epub.read_epub(str(path))
    blocks: list[str] = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        html = item.get_content()
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        item_text = _extract_item_text(soup)
        if item_text:
            blocks.append(item_text)

    return _EPUB_CHAPTER_BREAK.join(blocks).strip()


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
