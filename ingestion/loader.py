"""
ingestion/loader.py

Phase 1 — Step 1: Document loading.

Supported sources:
  - PDF          (.pdf)
  - Word         (.docx)
  - CSV          (.csv)
  - Plain text   (.txt, .md)
  - Web URL      (http/https — scrapes visible text)

Returns a unified list of RawDocument objects that the chunker consumes.
All source-specific logic is isolated here — chunker/embedder never import this directly.

Usage:
    from ingestion.loader import DocumentLoader
    loader = DocumentLoader()
    docs = loader.load("path/to/file.pdf")
    docs = loader.load("https://example.com/page")
    docs = loader.load_directory("path/to/folder")   # batch-loads all supported files
"""

from __future__ import annotations

import csv
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Unified document model
# ---------------------------------------------------------------------------

@dataclass
class RawDocument:
    """
    A single loaded document before chunking.

    Attributes:
        doc_id   : unique ID assigned at load time
        text     : full raw text content
        source   : original file path or URL
        doc_type : pdf | docx | csv | txt | web
        metadata : page count, title, URL, etc. — passed through to chunks
    """
    text: str
    source: str
    doc_type: Literal["pdf", "docx", "csv", "txt", "web"]
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class DocumentLoader:
    """
    Loads documents from files or URLs into RawDocument objects.

    Constraints:
      - Max file size: 50 MB (configurable via max_file_mb)
      - Web scraping: extracts <p>, <h1>-<h6>, <li> text only (no scripts/styles)
      - CSV: joins all cell values per row as a single text block per row
      - Encoding: UTF-8 with latin-1 fallback for text files
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".csv", ".txt", ".md"}

    def __init__(self, max_file_mb: int = 50):
        self.max_bytes = max_file_mb * 1024 * 1024

    # ── Public API ──────────────────────────────────────────────────────────

    def load(self, source: str) -> list[RawDocument]:
        """
        Load a single file path or URL.
        Returns a list because some sources (e.g. PDF) produce one doc per page.
        """
        if source.startswith("http://") or source.startswith("https://"):
            return [self._load_web(source)]

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        size = path.stat().st_size
        if size > self.max_bytes:
            raise ValueError(f"File too large ({size // (1024*1024)} MB > {self.max_bytes // (1024*1024)} MB limit): {source}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'. Supported: {self.SUPPORTED_EXTENSIONS}")

        loaders = {
            ".pdf":  self._load_pdf,
            ".docx": self._load_docx,
            ".csv":  self._load_csv,
            ".txt":  self._load_text,
            ".md":   self._load_text,
        }
        return loaders[ext](path)

    def load_directory(self, directory: str, recursive: bool = True) -> list[RawDocument]:
        """
        Batch-load all supported files in a directory.

        Args:
            directory : path to folder
            recursive : if True, walks subdirectories
        """
        docs: list[RawDocument] = []
        pattern = "**/*" if recursive else "*"
        for path in Path(directory).glob(pattern):
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    docs.extend(self.load(str(path)))
                    print(f"  [loader] loaded: {path.name}")
                except Exception as e:
                    print(f"  [loader] SKIP {path.name}: {e}")
        return docs

    # ── PDF ─────────────────────────────────────────────────────────────────

    def _load_pdf(self, path: Path) -> list[RawDocument]:
        """
        Extracts text page-by-page using pypdf.
        Each page becomes a separate RawDocument so chunking can respect page boundaries.

        Constraint: Scanned PDFs (image-only) will return empty text per page.
        For those, integrate an OCR step (pytesseract) before chunking.
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required for PDF loading: pip install pypdf")

        reader = PdfReader(str(path))
        docs = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                print(f"  [loader] WARNING: empty page {page_num} in {path.name} — may be scanned PDF")
                continue
            docs.append(RawDocument(
                text=text,
                source=str(path),
                doc_type="pdf",
                metadata={
                    "page": page_num,
                    "total_pages": len(reader.pages),
                    "filename": path.name,
                },
            ))
        return docs

    # ── DOCX ────────────────────────────────────────────────────────────────

    def _load_docx(self, path: Path) -> list[RawDocument]:
        """
        Extracts paragraph text from Word documents.
        Tables are extracted row-by-row and appended after paragraphs.
        """
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx is required: pip install python-docx")

        doc = Document(str(path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        table_rows = []
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    table_rows.append(" | ".join(cells))

        full_text = "\n".join(paragraphs)
        if table_rows:
            full_text += "\n\n[Tables]\n" + "\n".join(table_rows)

        return [RawDocument(
            text=full_text,
            source=str(path),
            doc_type="docx",
            metadata={"filename": path.name, "paragraph_count": len(paragraphs)},
        )]

    # ── CSV ─────────────────────────────────────────────────────────────────

    def _load_csv(self, path: Path) -> list[RawDocument]:
        """
        Converts CSV rows to text blocks.
        Each row becomes: "col1: val1 | col2: val2 | ..."
        All rows are combined into a single RawDocument.

        Constraint: For very wide CSVs (>50 columns), consider filtering columns
        before loading to avoid bloating chunk context.
        """
        encoding = self._detect_encoding(path)
        rows_text = []
        with open(path, encoding=encoding, newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for i, row in enumerate(reader):
                row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                if row_text:
                    rows_text.append(row_text)

        return [RawDocument(
            text="\n".join(rows_text),
            source=str(path),
            doc_type="csv",
            metadata={"filename": path.name, "columns": headers, "row_count": len(rows_text)},
        )]

    # ── Plain text / Markdown ───────────────────────────────────────────────

    def _load_text(self, path: Path) -> list[RawDocument]:
        encoding = self._detect_encoding(path)
        text = path.read_text(encoding=encoding)
        doc_type = "txt"
        return [RawDocument(
            text=text.strip(),
            source=str(path),
            doc_type=doc_type,
            metadata={"filename": path.name, "extension": path.suffix},
        )]

    # ── Web ──────────────────────────────────────────────────────────────────

    def _load_web(self, url: str) -> RawDocument:
        """
        Scrapes visible text from a web page.
        Extracts only: <p>, <h1>-<h6>, <li>, <td>, <th>
        Strips scripts, styles, nav, footer.

        Constraint: Does not handle JS-rendered pages (SPAs).
        For those, use Playwright: playwright.chromium.launch() → page.content()
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("requests and beautifulsoup4 are required: pip install requests beautifulsoup4")

        headers = {"User-Agent": "Mozilla/5.0 (RAG-demo-bot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Extract meaningful text tags
        text_tags = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th"])
        lines = [t.get_text(separator=" ", strip=True) for t in text_tags if t.get_text(strip=True)]
        text = "\n".join(lines)

        title = soup.title.string.strip() if soup.title and soup.title.string else url

        return RawDocument(
            text=text,
            source=url,
            doc_type="web",
            metadata={"url": url, "title": title},
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_encoding(path: Path) -> str:
        """Try UTF-8 first, fall back to latin-1 (never fails)."""
        try:
            path.read_text(encoding="utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            return "latin-1"