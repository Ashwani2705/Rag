"""

ingestion/chunker.py
 
Phase 1 — Step 2: Chunking.

Takes RawDocument Objects -> produces Chunk Objects ready for Embedding.

Strategy:

    -RecursiveCharacterTextSplitter(Langchain) as the Base Splitter
    -Splits on: paragraph breaks-> sentences -> words -> characters (in that order)
    - Chunk Size : 512 tokens (≈ 2048 chars), overlap: 15% (≈ 307 chars)
    -Every chunk gets a stable chunk_id(doc_id+index) so the graph builder 
    can reference chunks by ID without reloading the text


Constraints to be aware of :
     Token count is approximated as chars/4 (fast, no tokeniser needed)
  - PDF page-boundary chunks: chunker respects them — each RawDocument (page) 
    is chunked independently, so chunks never span page boundaries
  - Very short docs (< min_chunk_size) are returned as a single chunk unchanged

"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass,field
from typing import Literal

from ingestion.loader import RawDocument


# ---------------------------------------------------------------------------
# Unified chunk model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    A single text chunk ready for embedding and storage.
 
    Attributes:
        chunk_id    : stable ID — sha256(doc_id + index) — used as Neo4j node ref
        doc_id      : parent document ID from RawDocument
        text        : chunk text content
        index       : position of this chunk within the parent document
        source      : original file path or URL (from RawDocument)
        doc_type    : pdf | docx | csv | txt | web
        metadata    : page, filename, char_start, char_end — passed to vector store
    """
    chunk_id: str
    doc_id: str
    text: str
    index: int
    source: str
    doc_type: str
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self)->int:
        """Approximate token count (chars / 4). Good enough for size checks."""
        return len(self.text) // 4


class DocumentChunker:
    """
    Splits RawDocuments into fixed-size overlapping Chunk objects.
 
    Args:
        chunk_size      : target chunk size in characters (default 2048 ≈ 512 tokens)
        chunk_overlap   : overlap between consecutive chunks in characters (default 307 ≈ 15%)
        min_chunk_size  : chunks shorter than this are dropped (default 100 chars)
 
    The FAISS→Pinecone swap is transparent here — Chunk objects are the same
    regardless of which vector store receives them.
    """

    # Separators tried in order — paragraph → sentence → word → char
    _SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(self, chunk_size: int = 2048, chunk_overlap: int = 307, min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self._splitter=self._build_splitter()


    def _build_splitter(self):
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self._SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )


    def chunk_document(self, doc: RawDocument) -> list[Chunk]:
        """Chunk a single RawDocument. Returns [] if document text is empty."""
        if not doc.text or not doc.text.strip():
            print(f"  [chunker] SKIP empty doc: {doc.source}")
            return []
 
        # Short doc — return as single chunk
        if len(doc.text) <= self.min_chunk_size:
            return [self._make_chunk(doc, doc.text, 0, 0, len(doc.text))]
 
        raw_chunks = self._splitter.split_text(doc.text)
 
        chunks: list[Chunk] = []
        char_cursor = 0
        for i, text in enumerate(raw_chunks):
            if len(text.strip()) < self.min_chunk_size:
                continue   # skip noise fragments
 
            # Approximate char positions for metadata (non-overlapping window)
            char_start = doc.text.find(text, char_cursor)
            if char_start == -1:
                char_start = char_cursor
            char_end = char_start + len(text)
            char_cursor = max(char_cursor, char_end - self.chunk_overlap)
 
            chunk = self._make_chunk(doc, text, i, char_start, char_end)
            chunks.append(chunk)
 
        print(f"  [chunker] {doc.source} → {len(chunks)} chunks")
        return chunks

    def chunk_documents(self, docs: list[RawDocument]) -> list[Chunk]:
        """Chunk a list of RawDocuments. Preserves per-document chunk ordering."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


     # ── Internal ────────────────────────────────────────────────────────────
 
    def _make_chunk(
        self,
        doc: RawDocument,
        text: str,
        index: int,
        char_start: int,
        char_end: int,
    ) -> Chunk:
        chunk_id = self._stable_id(doc.doc_id, index)
        metadata = {
            **doc.metadata,             # page, filename, url, etc. from loader
            "chunk_id":  chunk_id,
            "doc_id":    doc.doc_id,
            "doc_type":  doc.doc_type,
            "source":    doc.source,
            "char_start": char_start,
            "char_end":   char_end,
            "chunk_index": index,
        }
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            text=text.strip(),
            index=index,
            source=doc.source,
            doc_type=doc.doc_type,
            metadata=metadata,
        )
 
    @staticmethod
    def _stable_id(doc_id: str, index: int) -> str:
        """
        SHA-256-based stable chunk ID.
        Same doc_id + index always produces the same chunk_id —
        safe to re-ingest without duplicating Neo4j nodes.
        """
        raw = f"{doc_id}::{index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
 
     


    

    


    

    
        



