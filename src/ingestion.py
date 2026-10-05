

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import E5_PASSAGE_PREFIX, E5_QUERY_PREFIX, Settings, get_settings

logger = logging.getLogger(__name__)


# Constants

DOC_TYPE_LAW = "egyptian_law"
DOC_TYPE_CONVENTION = "international_convention"

GENERAL_NUMBER = "general"
GENERAL_LABEL = "مقدمة / نص عام"

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt"})
_TEXT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "cp1256")

_CONVENTION_HINTS: tuple[str, ...] = (
    "convention", "treaty", "treaties", "international", "protocol",
    "اتفاقية", "اتفاقيات", "معاهدة", "معاهدات", "بروتوكول",
)

_ORDINALS: dict[str, str] = {
    "الأولى": "1", "الاولى": "1", "الثانية": "2", "الثالثة": "3",
    "الرابعة": "4", "الخامسة": "5", "السادسة": "6", "السابعة": "7",
    "الثامنة": "8", "التاسعة": "9", "العاشرة": "10",
}
_ORDINAL_ALT = "|".join(re.escape(word) for word in _ORDINALS)



_ARTICLE_HEADER_RE = re.compile(
    rf"""^[ \t]*
        (?P<word>المادة|مادة|Article)
        [ \t]*\(?[ \t]*
        (?P<num>\d{{1,4}}|{_ORDINAL_ALT})
        [ \t]*\)?
        (?P<suffix>[ \t]*مكرر(?:[ \t]*\([ \t]*[\u0621-\u064A][ \t]*\))?)?
        [ \t]*(?:[:\-–—.]|$)
    """,
    re.MULTILINE | re.VERBOSE | re.IGNORECASE,
)

# Text-cleaning primitives
_DIGIT_TABLE = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
_DIACRITICS_RE = re.compile("[\u064b-\u065f\u0670]")
_SPACES_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TATWEEL = "\u0640"

# Splitter separators 
_ARABIC_SEPARATORS: list[str] = [
    r"\n\n",
    r"\n",
    r"(?<=[.!?؟؛;])\s+",
    r"(?<=[،,:])\s+",
    r"\s+",
    "",
]


# Errors
class IngestionError(RuntimeError):
    """Raised when a document cannot be loaded or processed."""


class EmbeddingModelError(RuntimeError):
    """Raised when the embedding model cannot be initialised."""


# Embeddings
def _with_prefix(text: str, prefix: str) -> str:
    """Prepend ``prefix`` unless it is already present (idempotent)."""
    if not prefix or text.startswith(prefix):
        return text
    return f"{prefix}{text}"


class PrefixedEmbeddings(Embeddings):
    """Embeddings decorator that adds E5-style prefixes idempotently.

    Documents get ``passage: `` and queries get ``query: ``. Because prefixing is
    idempotent, callers that already prepend ``query: `` (e.g. ``SearchService``)
    never produce a double prefix, while plain LangChain retrievers stay correct.
    """

    def __init__(self, base: Embeddings, document_prefix: str = "", query_prefix: str = "") -> None:
        self._base = base
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._base.embed_documents([_with_prefix(t, self._document_prefix) for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(_with_prefix(text, self._query_prefix))


def resolve_device(configured: str) -> str:
    """Resolve ``auto`` into the best available torch device."""
    if configured.strip().lower() != "auto":
        return configured.strip()
    try:
        import torch  

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except ImportError:
        logger.debug("torch not importable; falling back to CPU.")
    return "cpu"


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    """Create the LangChain embeddings object for the configured model."""
    settings = settings or get_settings()
    device = resolve_device(settings.EMBEDDING_DEVICE)
    logger.info("Loading embedding model '%s' on device '%s'.", settings.EMBEDDING_MODEL_NAME, device)
    try:
        base = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc: 
        raise EmbeddingModelError(
            f"Failed to load embedding model '{settings.EMBEDDING_MODEL_NAME}': {exc}"
        ) from exc

    if settings.is_e5_model:
        return PrefixedEmbeddings(
            base,
            document_prefix=E5_PASSAGE_PREFIX,
            query_prefix=E5_QUERY_PREFIX,
        )
    return base


# Cleaning
def clean_text(text: str) -> str:
    """Normalise Arabic/legal text for indexing and querying.

    - Unicode NFKC (folds Arabic presentation forms produced by many PDFs)
    - Removes zero-width/bidi marks, control characters, tatweel and diacritics
    - Converts Arabic-Indic / Persian digits to ASCII digits
    - Collapses runs of spaces and blank lines while preserving paragraph breaks
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    text = _DIACRITICS_RE.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = text.translate(_DIGIT_TABLE)
    lines = [_SPACES_RE.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


# Structural + recursive chunking
@dataclass(frozen=True, slots=True)
class ArticleSection:
    """A structural unit of a legal document (one article or the preamble)."""

    number: str  
    label: str  
    text: str


def split_into_articles(text: str) -> list[ArticleSection]:
    """Split cleaned text into article sections using Arabic/English header patterns."""
    matches = list(_ARTICLE_HEADER_RE.finditer(text))
    if not matches:
        body = text.strip()
        return [ArticleSection(GENERAL_NUMBER, GENERAL_LABEL, body)] if body else []

    sections: list[ArticleSection] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(ArticleSection(GENERAL_NUMBER, GENERAL_LABEL, preamble))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.start():end].strip()
        raw = match.group("num")
        number = str(int(raw)) if raw.isdigit() else _ORDINALS[raw]
        suffix = " ".join((match.group("suffix") or "").split())
        number_label = f"{number} {suffix}".strip()
        sections.append(
            ArticleSection(
                number=number_label,
                label=f"{match.group('word')} {number_label}",
                text=body,
            )
        )
    return sections


class ArabicLegalChunker:
    """Splits articles; only articles exceeding ``max_chars`` are split recursively."""

    def __init__(self, max_chars: int, overlap_chars: int) -> None:
        self._max_chars = max_chars
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=overlap_chars,
            separators=_ARABIC_SEPARATORS,
            is_separator_regex=True,
            keep_separator=True,
            strip_whitespace=True,
            length_function=len,
        )

    def chunk_section(self, section: ArticleSection) -> list[str]:
        if len(section.text) <= self._max_chars:
            return [section.text]
        parts = (part.strip() for part in self._splitter.split_text(section.text))
        return [part for part in parts if part]


# Loading helpers
def infer_doc_type(source_doc: str) -> str:
    """Infer ``doc_type`` from the relative path / file name."""
    lowered = source_doc.lower()
    if any(hint in lowered for hint in _CONVENTION_HINTS):
        return DOC_TYPE_CONVENTION
    return DOC_TYPE_LAW


def _load_pdf(path: Path) -> str:
    pages = PyPDFLoader(str(path)).load()
    return "\n".join(page.page_content for page in pages)


def _load_txt(path: Path) -> str:
    last_error: Exception | None = None
    for encoding in _TEXT_ENCODINGS:
        try:
            docs = TextLoader(str(path), encoding=encoding).load()
            return "\n".join(doc.page_content for doc in docs)
        except (RuntimeError, UnicodeDecodeError) as exc:
            last_error = exc
    raise IngestionError(f"Unable to decode '{path}' with {list(_TEXT_ENCODINGS)}: {last_error}")


# Pipeline
class IngestionPipeline:
    """Turns raw files into metadata-rich LangChain ``Document`` chunks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._chunker = ArabicLegalChunker(
            self._settings.MAX_CHUNK_CHARS,
            self._settings.CHUNK_OVERLAP_CHARS,
        )
        self.failed_files: list[str] = []

    def discover_files(self, root: Path | str | None = None) -> list[Path]:
        base = Path(root or self._settings.RAW_DATA_DIR)
        if not base.is_dir():
            raise IngestionError(f"Data directory does not exist: {base}")
        return sorted(
            p for p in base.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    @staticmethod
    def source_name(path: Path, root: Path) -> str:
        """Stable source identifier: POSIX path relative to ``root``."""
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return path.name

    def process_file(
        self,
        path: Path | str,
        root: Path | str | None = None,
        doc_type: str | None = None,
    ) -> list[Document]:
        """Load, clean and chunk a single file."""
        file_path = Path(path)
        base = Path(root) if root else file_path.parent
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise IngestionError(f"Unsupported file type '{suffix}' for {file_path}")

        try:
            raw = _load_pdf(file_path) if suffix == ".pdf" else _load_txt(file_path)
        except IngestionError:
            raise
        except Exception as exc:  # loader boundary: pypdf/OS errors vary widely
            raise IngestionError(f"Failed to read '{file_path}': {exc}") from exc

        cleaned = clean_text(raw)
        if not cleaned:
            raise IngestionError(
                f"No extractable text in '{file_path}' (scanned PDF? run OCR first)."
            )

        source_doc = self.source_name(file_path, base)
        resolved_type = doc_type or infer_doc_type(source_doc)

        documents: list[Document] = []
        seq = 0
        for section_index, section in enumerate(split_into_articles(cleaned)):
            for part, chunk in enumerate(self._chunker.chunk_section(section)):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source_doc": source_doc,
                            "doc_type": resolved_type,
                            "article_number": section.number,
                            "article_label": section.label,
                            "section_index": section_index,
                            "part": part,
                            "seq": seq,
                            "chunk_id": f"{source_doc}::{seq:06d}",
                        },
                    )
                )
                seq += 1

        logger.info("Processed '%s': %d chunks (%s).", source_doc, len(documents), resolved_type)
        return documents

    def process_directory(
        self, root: Path | str | None = None
    ) -> Iterator[tuple[str, list[Document]]]:
        """Yield ``(source_doc, chunks)`` per file; failures are logged and recorded."""
        base = Path(root or self._settings.RAW_DATA_DIR)
        self.failed_files = []
        for path in self.discover_files(base):
            try:
                documents = self.process_file(path, root=base)
            except IngestionError as exc:
                logger.error("Skipping '%s': %s", path, exc)
                self.failed_files.append(str(path))
                continue
            yield self.source_name(path, base), documents


# CLI
def main(argv: Sequence[str] | None = None) -> int:
    """Index all supported files; existing chunks of each source are replaced."""
    parser = argparse.ArgumentParser(description="Index legal documents into ChromaDB.")
    parser.add_argument("--path", default=None, help="Directory to ingest (default: RAW_DATA_DIR).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from .vector_store import LegalVectorStore, VectorStoreError  # local: keeps module deps one-way

    settings = get_settings()
    pipeline = IngestionPipeline(settings)
    try:
        store = LegalVectorStore(
            embeddings=build_embeddings(settings),
            persist_directory=settings.CHROMA_PERSIST_DIR,
            collection_name=settings.COLLECTION_NAME,
            batch_size=settings.UPSERT_BATCH_SIZE,
        )
        store.initialize()

        total = 0
        for source_doc, documents in pipeline.process_directory(args.path):
            removed = store.delete_by_source(source_doc)
            written = store.upsert_documents(documents)
            total += written
            logger.info("Indexed '%s': removed %d old, wrote %d chunks.", source_doc, removed, written)
    except (IngestionError, EmbeddingModelError, VectorStoreError) as exc:
        logger.error("Ingestion aborted: %s", exc)
        return 1

    logger.info("Done. %d chunks indexed, %d files failed.", total, len(pipeline.failed_files))
    return 1 if pipeline.failed_files else 0


if __name__ == "__main__":
    sys.exit(main())
