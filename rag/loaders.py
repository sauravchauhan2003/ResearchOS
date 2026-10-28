from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    source_filename: str
    text: str
    content_hash: str


def load_document(filename: str, content: bytes, *, max_bytes: int = MAX_DOCUMENT_BYTES) -> LoadedDocument:
    import hashlib

    safe_name = Path(filename).name
    if len(content) > max_bytes:
        raise ValueError(f"Document exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")
    suffix = Path(safe_name).suffix.casefold()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Supported document formats are PDF, TXT, and Markdown.")
    if suffix == ".pdf":
        try:
            reader = PdfReader(BytesIO(content), strict=True)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise ValueError(f"Could not read PDF: {type(exc).__name__}.") from exc
    else:
        text = content.decode("utf-8-sig", errors="replace")
    if not text.strip():
        raise ValueError("Document contains no extractable text.")
    return LoadedDocument(safe_name, text, hashlib.sha256(content).hexdigest())
