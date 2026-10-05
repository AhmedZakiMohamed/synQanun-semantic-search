from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .config import Settings, get_settings
from .ingestion import EmbeddingModelError, build_embeddings
from .search_service import (
    InvalidQueryError,
    SearchService,
    SearchServiceError,
)
from .vector_store import LegalVectorStore, VectorStoreError

logger = logging.getLogger(__name__)


# Schemas
class SearchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(
        ...,
        validation_alias=AliasChoices("query", "q"),
        min_length=1,
        max_length=1000,
        description="Natural-language legal query (JSON keys: `query` or `q`).",
    )
    top_k: int | None = Field(
        default=None,
        validation_alias=AliasChoices("top_k", "topK"),
        ge=1,
     
        description="Maximum documents to return (JSON keys: `top_k` or `topK`; defaults to TOP_K).",
    )
    doc_type: Literal["egyptian_law", "international_convention"] | None = Field(
        default=None, description="Optional filter on document type."
    )



class ArticleHit(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    article_number: str
    article_label: str
    section_index: int
    score: float
    text: str
    matched_chunks: int
    chunk_seqs: list[int]


class DocumentHit(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    source_doc: str
    doc_type: str
    score: float
    matched_chunks: int
    matched_articles: list[ArticleHit]


class SearchResponse(BaseModel):
    query: str
    top_k: int
    total: int
    results: list[DocumentHit]  


class HealthResponse(BaseModel):
    status: Literal["ok"]
    embedding_model: str
    collection: str
    indexed_chunks: int



# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the embedding model and open the vector store exactly once."""
    settings = get_settings()
    logger.info("Starting up: loading embedding model and vector store.")

    try:
        embeddings = await asyncio.to_thread(build_embeddings, settings)
        vector_store = LegalVectorStore(
            embeddings=embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR,
            collection_name=settings.COLLECTION_NAME,
            batch_size=settings.UPSERT_BATCH_SIZE,
        )
        await asyncio.to_thread(vector_store.initialize)
        # Warm-up so the first real request does not pay model initialisation cost.
        await asyncio.to_thread(embeddings.embed_query, "query: warm-up")
    except (EmbeddingModelError, VectorStoreError) as exc:
        logger.error("Startup failed: %s", exc)
        raise

    app.state.settings = settings
    app.state.vector_store = vector_store
    app.state.search_service = SearchService(vector_store, settings)
    logger.info("Startup complete.")

    try:
        yield
    finally:
        app.state.search_service = None
        app.state.vector_store = None
        logger.info("Shutdown complete.")


# Dependencies
def get_search_service(request: Request) -> SearchService:
    service: SearchService | None = getattr(request.app.state, "search_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service is not ready.",
        )
    return service


def get_app_settings(request: Request) -> Settings:
    settings: Settings | None = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready.",
        )
    return settings


# App factory
def create_app() -> FastAPI:
    app = FastAPI(
        title="Legal Semantic Search API",
        description="Semantic search over Egyptian laws and international conventions.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health(
        request: Request,
        settings: Settings = Depends(get_app_settings),
    ) -> HealthResponse:
        vector_store: LegalVectorStore | None = getattr(request.app.state, "vector_store", None)
        if vector_store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector store is not ready.",
            )
        try:
            indexed = await asyncio.to_thread(vector_store.count)
        except VectorStoreError as exc:
            logger.error("Health check failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector store is unavailable.",
            ) from exc
        return HealthResponse(
            status="ok",
            embedding_model=settings.EMBEDDING_MODEL_NAME,
            collection=settings.COLLECTION_NAME,
            indexed_chunks=indexed,
        )

    @app.post("/search", response_model=SearchResponse, tags=["search"])
    async def search(
        payload: SearchRequest,
        service: SearchService = Depends(get_search_service),
        settings: Settings = Depends(get_app_settings),
    ) -> SearchResponse:
        effective_top_k = payload.top_k if payload.top_k is not None else settings.TOP_K
        try:
            results = await service.asearch(payload.query, effective_top_k, payload.doc_type)
        except InvalidQueryError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except SearchServiceError as exc:
            logger.error("Search failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Search failed due to an internal error.",
            ) from exc

        hits = [DocumentHit.model_validate(result) for result in results]  
        return SearchResponse(
            query=payload.query,
            top_k=effective_top_k,
            total=len(hits),
            results=hits,
        )

    return app


app = create_app()