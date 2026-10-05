
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)

COSINE_SPACE = "cosine"
_DISTANCE_KEY = "hnsw:space"
_DELETE_BATCH_SIZE = 500

MetadataValue = str | int | float | bool


class VectorStoreError(RuntimeError):
    """Raised for any vector store initialisation or I/O failure."""


class LegalVectorStore:
    """Thin, typed wrapper around LangChain's ``Chroma`` vector store."""

    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: str,
        collection_name: str,
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._embeddings = embeddings
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._batch_size = batch_size
        self._store: Chroma | None = None

    # Lifecycle
    def initialize(self) -> None:
        """Open (or create) the persistent cosine-distance collection. Idempotent."""
        if self._store is not None:
            return
        try:
            Path(self._persist_directory).mkdir(parents=True, exist_ok=True)
            store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embeddings,
                persist_directory=self._persist_directory,
                collection_metadata={_DISTANCE_KEY: COSINE_SPACE},
            )
        except Exception as exc:  
            raise VectorStoreError(
                f"Failed to initialise Chroma collection '{self._collection_name}' "
                f"at '{self._persist_directory}': {exc}"
            ) from exc

        self._verify_distance_metric(store)
        self._store = store
        logger.info(
            "Vector store ready (collection='%s', dir='%s').",
            self._collection_name,
            self._persist_directory,
        )

    @property
    def store(self) -> Chroma:
        """The underlying LangChain ``Chroma`` instance."""
        if self._store is None:
            raise VectorStoreError("Vector store is not initialised; call initialize() first.")
        return self._store

    def _verify_distance_metric(self, store: Chroma) -> None:
        """Fail fast if an existing collection was created with a non-cosine metric."""
        collection = getattr(store, "_collection", None)
        metadata: Mapping[str, Any] = getattr(collection, "metadata", None) or {}
        space = metadata.get(_DISTANCE_KEY)
        if space is not None and space != COSINE_SPACE:
            raise VectorStoreError(
                f"Collection '{self._collection_name}' uses distance '{space}', expected "
                f"'{COSINE_SPACE}'. Delete '{self._persist_directory}' and re-ingest."
            )

    # Writes
    @staticmethod
    def _sanitize_metadata(metadata: Mapping[str, Any]) -> dict[str, MetadataValue]:
        """Chroma only accepts scalar metadata values; drop ``None`` and stringify the rest."""
        clean: dict[str, MetadataValue] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            clean[str(key)] = value if isinstance(value, (str, int, float, bool)) else str(value)
        return clean

    @staticmethod
    def _document_id(document: Document) -> str:
        chunk_id = document.metadata.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            return chunk_id
        source = document.metadata.get("source_doc", "")
        seq = document.metadata.get("seq", "")
        payload = f"{source}|{seq}|{document.page_content}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    def upsert_documents(self, documents: Sequence[Document]) -> int:
        """Upsert documents in batches (deterministic IDs make this idempotent).

        Returns the number of unique chunks written.
        """
        unique: dict[str, Document] = {}
        for document in documents:
            if not document.page_content or not document.page_content.strip():
                continue
            prepared = Document(
                page_content=document.page_content,
                metadata=self._sanitize_metadata(document.metadata),
            )
            unique[self._document_id(prepared)] = prepared

        if not unique:
            return 0

        store = self.store
        ids = list(unique)
        docs = [unique[doc_id] for doc_id in ids]
        for start in range(0, len(docs), self._batch_size):
            end = start + self._batch_size
            try:
                store.add_documents(docs[start:end], ids=ids[start:end])
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to upsert batch [{start}:{min(end, len(docs))}]: {exc}"
                ) from exc
        return len(docs)

    def delete_by_source(self, source_doc: str) -> int:
        """Delete every chunk whose ``source_doc`` metadata equals ``source_doc``."""
        store = self.store
        try:
            found = store.get(where={"source_doc": source_doc}, include=["metadatas"])
            ids: list[str] = list(found.get("ids") or [])
            for start in range(0, len(ids), _DELETE_BATCH_SIZE):
                store.delete(ids=ids[start:start + _DELETE_BATCH_SIZE])
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete source '{source_doc}': {exc}") from exc
        return len(ids)

    # Reads
    def get_section_documents(self, source_doc: str, section_index: int) -> list[Document]:
        """Return all chunks of one article/section, ordered by ``seq``."""
        where = {
            "$and": [
                {"source_doc": {"$eq": source_doc}},
                {"section_index": {"$eq": section_index}},
            ]
        }
        try:
            payload = self.store.get(where=where, include=["documents", "metadatas"])
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to load section {section_index} of '{source_doc}': {exc}"
            ) from exc

        texts = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        documents = [
            Document(page_content=text, metadata=dict(meta or {}))
            for text, meta in zip(texts, metadatas)
        ]
        documents.sort(key=lambda doc: int(doc.metadata.get("seq", 0)))
        return documents

    def count(self) -> int:
        """Number of chunks stored in the collection."""
        try:
            return int(self.store._collection.count())  
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to count documents: {exc}") from exc

    def as_retriever(self, k: int, filter: dict[str, Any] | None = None) -> VectorStoreRetriever:
        """LangChain retriever over the collection (cosine similarity)."""
        search_kwargs: dict[str, Any] = {"k": k}
        if filter:
            search_kwargs["filter"] = filter
        return self.store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
