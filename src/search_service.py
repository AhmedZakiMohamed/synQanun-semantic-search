from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from .config import E5_QUERY_PREFIX, Settings, get_settings
from .ingestion import GENERAL_LABEL, GENERAL_NUMBER, clean_text
from .vector_store import LegalVectorStore, VectorStoreError

logger = logging.getLogger(__name__)

_GAP_MARKER = "[...]"
_MIN_OVERLAP_CHARS = 5
_MAX_ARTICLES_PER_DOC = 3  

class SearchServiceError(RuntimeError):
    """Raised when the underlying retrieval fails."""


class InvalidQueryError(ValueError):
    """Raised when the caller supplies an invalid query or parameter."""


@dataclass(frozen=True, slots=True)
class AggregatedResult:
    """One legal article (or document section) assembled from its chunks."""

    source_doc: str
    doc_type: str
    article_number: str
    article_label: str
    section_index: int
    score: float
    text: str
    matched_chunks: int
    chunk_seqs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DocumentResult:
    """One document, ranked by its best-matching article."""

    source_doc: str
    doc_type: str
    score: float  
    matched_chunks: int  
    matched_articles: tuple[AggregatedResult, ...] 


def _meta_str(metadata: dict[str, Any], key: str, default: str = "") -> str:
    value = metadata.get(key, default)
    return value if isinstance(value, str) else str(value)


def _meta_int(metadata: dict[str, Any], key: str, default: int = -1) -> int:
    try:
        return int(metadata.get(key, default))
    except (TypeError, ValueError):
        return default


def _merge_overlap(previous: str, following: str, max_overlap: int) -> str:
    """Join two consecutive chunks, removing the splitter's overlapping text."""
    limit = min(len(previous), len(following), max_overlap)
    for size in range(limit, _MIN_OVERLAP_CHARS - 1, -1):
        if previous.endswith(following[:size]):
            return previous + following[size:]
    return f"{previous}\n{following}"


class SearchService:
    """Retrieves chunks via LangChain similarity search and aggregates them by document."""

    def __init__(
        self,
        vector_store: LegalVectorStore,
        settings: Settings | None = None,
        expand_articles: bool = True,
    ) -> None:
        self._vector_store = vector_store
        self._settings = settings or get_settings()
        self._expand_articles = expand_articles

    # Public API
    def search(
        self,
        query: str,
        top_k: int | None = None,
        doc_type: str | None = None,
    ) -> list[DocumentResult]:  # [CHANGED] returns documents, not articles
        """Return up to ``top_k`` documents ordered by their best-matching article."""
        k = self._settings.TOP_K if top_k is None else top_k
        if not 1 <= k <= self._settings.MAX_TOP_K:
            raise InvalidQueryError(f"top_k must be between 1 and {self._settings.MAX_TOP_K}.")

        prepared = self._prepare_query(query)
        fetch_k = k * self._settings.CANDIDATE_MULTIPLIER
        search_kwargs: dict[str, Any] = {"k": fetch_k}
        if doc_type:
            search_kwargs["filter"] = {"doc_type": doc_type}

        try:
            hits = self._vector_store.store.similarity_search_with_relevance_scores(
                prepared, **search_kwargs
            )
        except VectorStoreError as exc:
            raise SearchServiceError(str(exc)) from exc
        except Exception as exc:
            raise SearchServiceError(f"Similarity search failed: {exc}") from exc

        return self._aggregate_documents(hits, k)  

    async def asearch(
        self,
        query: str,
        top_k: int | None = None,
        doc_type: str | None = None,
    ) -> list[DocumentResult]: 
        """Async wrapper that keeps the event loop free during embedding and I/O."""
        return await asyncio.to_thread(self.search, query, top_k, doc_type)

    # Internals
    def _prepare_query(self, query: str) -> str:
        cleaned = " ".join(clean_text(query).split())
        if not cleaned:
            raise InvalidQueryError("Query must not be empty.")
        if len(cleaned) > self._settings.MAX_QUERY_CHARS:
            raise InvalidQueryError(
                f"Query exceeds {self._settings.MAX_QUERY_CHARS} characters."
            )
        return f"{E5_QUERY_PREFIX}{cleaned}" if self._settings.is_e5_model else cleaned

    def _aggregate_documents(
        self, hits: list[tuple[Document, float]], top_k: int
    ) -> list[DocumentResult]:
        article_groups: dict[tuple[str, int], list[tuple[Document, float]]] = {}
        for document, score in hits:
            key = (
                _meta_str(document.metadata, "source_doc"),
                _meta_int(document.metadata, "section_index"),
            )
            article_groups.setdefault(key, []).append((document, score))

        doc_groups: dict[str, list[tuple[int, list[tuple[Document, float]]]]] = {}
        for (source_doc, section_index), group in article_groups.items():
            doc_groups.setdefault(source_doc, []).append((section_index, group))

        def best(group: list[tuple[Document, float]]) -> float:
            return max(score for _, score in group)

       
        ranked_docs = sorted(
            doc_groups.items(),
            key=lambda item: (-max(best(group) for _, group in item[1]), item[0]),
        )[:top_k]

        results: list[DocumentResult] = []
        for source_doc, sections in ranked_docs:
            sections.sort(key=lambda item: (-best(item[1]), item[0]))
            articles = tuple(
                self._build_result(source_doc, section_index, group)
                for section_index, group in sections[:_MAX_ARTICLES_PER_DOC]
            )
            results.append(
                DocumentResult(
                    source_doc=source_doc,
                    doc_type=articles[0].doc_type,
                    score=articles[0].score,
                    matched_chunks=sum(len(group) for _, group in sections),
                    matched_articles=articles,
                )
            )
        return results

    def _build_result(
        self,
        source_doc: str,
        section_index: int,
        group: list[tuple[Document, float]],
    ) -> AggregatedResult:
        best_score = max(score for _, score in group)
        reference = max(group, key=lambda item: item[1])[0].metadata

        chunks = self._collect_chunks(source_doc, section_index, group)
        text = self._join_chunks(chunks)

        return AggregatedResult(
            source_doc=source_doc,
            doc_type=_meta_str(reference, "doc_type", "unknown"),
            article_number=_meta_str(reference, "article_number", GENERAL_NUMBER),
            article_label=_meta_str(reference, "article_label", GENERAL_LABEL),
            section_index=section_index,
            score=round(min(1.0, max(0.0, float(best_score))), 4),
            text=text,
            matched_chunks=len(group),
            chunk_seqs=tuple(_meta_int(doc.metadata, "seq") for doc in chunks),
        )

    def _collect_chunks(
        self,
        source_doc: str,
        section_index: int,
        group: list[tuple[Document, float]],
    ) -> list[Document]:
        """Prefer the complete article; fall back to the retrieved chunks."""
        if self._expand_articles and section_index >= 0:
            try:
                full = self._vector_store.get_section_documents(source_doc, section_index)
            except VectorStoreError as exc:
                logger.warning("Article expansion failed for '%s': %s", source_doc, exc)
                full = []
            if full:
                return full

        by_seq: dict[int, Document] = {}
        for document, _ in group:
            by_seq.setdefault(_meta_int(document.metadata, "seq"), document)
        return [by_seq[seq] for seq in sorted(by_seq)]

    def _join_chunks(self, chunks: list[Document]) -> str:
        text = ""
        previous_seq: int | None = None
        for chunk in chunks:
            content = chunk.page_content.strip()
            seq = _meta_int(chunk.metadata, "seq")
            if not content:
                continue
            if not text:
                text = content
            elif previous_seq is not None and seq == previous_seq + 1:
                text = _merge_overlap(text, content, self._settings.CHUNK_OVERLAP_CHARS)
            else:
                text = f"{text}\n{_GAP_MARKER}\n{content}"
            previous_seq = seq
        return text