"""
api/dependencies.py

FastAPI dependency injection for shared module instances.
All heavy objects (stores, engines) are initialized once at startup
and injected into route handlers via FastAPI's dependency system.

Usage in routers:
    from api.dependencies import get_milvus, get_retrieval_engine

    @router.get("/example")
    async def example(milvus: MilvusStore = Depends(get_milvus)):
        ...
"""

from fastapi import Request

from modules.indexing.milvus_store import MilvusStore
from modules.indexing.tantivy_index import TantivyIndex
from modules.indexing.sqlite_store import SQLiteStore
from modules.retrieval.retrieval_engine import RetrievalEngine
from modules.llm.llm_engine import LLMEngine
from modules.citation.citation_engine import CitationEngine
from modules.export.export_engine import ExportEngine


def get_milvus(request: Request) -> MilvusStore:
    return request.app.state.milvus


def get_tantivy(request: Request) -> TantivyIndex:
    return request.app.state.tantivy


def get_sqlite(request: Request) -> SQLiteStore:
    return request.app.state.sqlite


def get_retrieval_engine(request: Request) -> RetrievalEngine:
    return request.app.state.retrieval_engine


def get_llm_engine(request: Request) -> LLMEngine:
    return request.app.state.llm_engine


def get_citation_engine(request: Request) -> CitationEngine:
    return request.app.state.citation_engine


def get_export_engine(request: Request) -> ExportEngine:
    return request.app.state.export_engine
