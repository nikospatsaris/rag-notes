"""Turn PDF files into overlapping text chunks that can be embedded.

The chunk is the unit the retriever works with, so two things matter here:
chunks must be small enough to be specific, and every chunk must remember
where it came from so the answer can cite a real page.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

# ~1000 characters is roughly 200-250 words: big enough to hold a whole
# idea, small enough that a hit is actually about the question. The overlap
# stops a definition that straddles a boundary from being lost by both chunks.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage, with enough metadata to cite it."""

    text: str
    source: str  # file name, e.g. "networks-lecture-3.pdf"
    page: int  # 1-indexed, matches what the reader sees in a PDF viewer
    chunk_id: str  # stable hash, so re-indexing replaces instead of duplicating


def _clean(text: str) -> str:
    """Collapse the whitespace noise that PDF extraction leaves behind."""
    text = text.replace("\xa0", " ")
    # PDF bullet glyphs and layout markers often extract as control characters.
    # They carry no meaning, waste context, and make the embedding noisier.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping windows, preferring to break at a boundary.

    A hard character cut mid-sentence produces chunks that embed badly, so
    each window is trimmed back to the last paragraph break, sentence end, or
    space it contains — whichever comes first when searching backwards.
    """
    if len(text) <= size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + size, length)
        window = text[start:end]

        if end < length:
            # Look for a clean break in the last quarter of the window only —
            # searching the whole window could shrink a chunk to almost nothing.
            tail_start = int(size * 0.75)
            tail = window[tail_start:]
            for marker in ("\n\n", ". ", "\n", " "):
                pos = tail.rfind(marker)
                if pos != -1:
                    window = window[: tail_start + pos + len(marker)]
                    break

        trimmed = window.strip()
        if trimmed:
            chunks.append(trimmed)

        # Stop once this window reached the end of the text. Without this the
        # overlap keeps pulling `start` forward by a shrinking amount and the
        # tail gets emitted once per character.
        if end >= length:
            break

        start += max(len(window) - overlap, 1)

    return chunks


def load_pdf(path: str | Path, source_name: str | None = None) -> list[Chunk]:
    """Read one PDF and return its chunks, page by page.

    Chunking per page (rather than over the whole document) keeps page
    numbers exact, which is what makes the citations trustworthy.

    `source_name` overrides the name recorded on each chunk. The web UI needs
    this: it writes uploads to a randomly-named temp file, and hashing that
    name would give the same document a different ID on every upload — so
    re-uploading would pile up duplicates instead of replacing them.
    """
    path = Path(path)
    name = source_name or path.name
    reader = PdfReader(str(path))
    chunks: list[Chunk] = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = _clean(raw)
        if not text:
            continue  # scanned image page with no text layer — nothing to embed

        for piece in _split(text):
            digest = hashlib.sha256(
                f"{name}|{page_number}|{piece}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(
                Chunk(text=piece, source=name, page=page_number, chunk_id=digest)
            )

    return chunks


def load_folder(folder: str | Path) -> list[Chunk]:
    """Read every PDF in a folder."""
    folder = Path(folder)
    chunks: list[Chunk] = []
    for pdf in sorted(folder.glob("*.pdf")):
        chunks.extend(load_pdf(pdf))
    return chunks
