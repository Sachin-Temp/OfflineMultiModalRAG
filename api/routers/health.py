"""
api/routers/health.py

Health and system monitoring endpoints.

GET /health/        — Basic liveness check
GET /health/ready   — Readiness check (all systems initialized)
GET /health/vram    — VRAM usage and model loading status
GET /health/stats   — Full system statistics
"""

import time

from fastapi import APIRouter, Depends
from loguru import logger

from api.dependencies import (
    get_milvus, get_sqlite, get_tantivy,
)
from core.vram_manager import vram_manager
from modules.indexing.milvus_store import MilvusStore
from modules.indexing.sqlite_store import SQLiteStore
from modules.indexing.tantivy_index import TantivyIndex

router = APIRouter()


@router.get(
    "/",
    summary="Liveness check",
    description="Returns 200 if the server is alive.",
)
async def health_live():
    """Simple liveness probe — used by Docker/K8s health checks."""
    return {"status": "alive", "timestamp": time.time()}


@router.get(
    "/ready",
    summary="Readiness check",
    description=(
        "Returns 200 if all backing stores are initialized and ready. "
        "Returns 503 if any store is not ready."
    ),
)
async def health_ready(
    milvus:  MilvusStore  = Depends(get_milvus),
    tantivy: TantivyIndex = Depends(get_tantivy),
    sqlite:  SQLiteStore  = Depends(get_sqlite),
):
    """
    Readiness probe — checks that all stores respond correctly.
    Use this to determine when the server is ready to accept requests.
    """
    issues = []

    # Check Milvus
    try:
        stats = milvus.get_collection_stats()
        milvus_ok = all(
            info.get("status") == "ok"
            for info in stats.values()
        )
        if not milvus_ok:
            issues.append("Milvus collections not all healthy")
    except Exception as e:
        issues.append(f"Milvus unreachable: {e}")

    # Check Tantivy
    try:
        t_stats  = tantivy.get_stats()
        if t_stats.get("status") != "ok":
            issues.append(f"Tantivy issue: {t_stats.get('status')}")
    except Exception as e:
        issues.append(f"Tantivy unreachable: {e}")

    # Check SQLite
    try:
        s_stats = sqlite.get_stats()
        if s_stats.get("status") != "ok":
            issues.append(f"SQLite issue: {s_stats.get('status')}")
    except Exception as e:
        issues.append(f"SQLite unreachable: {e}")

    if issues:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "issues": issues,
            },
        )

    return {"status": "ready", "timestamp": time.time()}


@router.get(
    "/vram",
    summary="VRAM usage and loaded models",
    description="Returns current VRAM allocation from VRAMManager.",
)
async def health_vram():
    """
    VRAM status from VRAMManager.
    Shows used/ceiling GB and which models are currently loaded.
    """
    try:
        status = vram_manager.status()
        return {
            "vram":      status,
            "timestamp": time.time(),
        }
    except Exception as e:
        return {
            "vram":      {"error": str(e)},
            "timestamp": time.time(),
        }


@router.get(
    "/stats",
    summary="Full system statistics",
    description=(
        "Returns combined stats from all backing stores: "
        "Milvus collection sizes, Tantivy doc count, SQLite table counts, "
        "VRAM usage."
    ),
)
async def health_stats(
    milvus:  MilvusStore  = Depends(get_milvus),
    tantivy: TantivyIndex = Depends(get_tantivy),
    sqlite:  SQLiteStore  = Depends(get_sqlite),
):
    """
    Combined statistics from all system components.
    Used by the monitoring dashboard in Phase 10.
    """
    stats = {
        "timestamp": time.time(),
        "milvus":    {},
        "tantivy":   {},
        "sqlite":    {},
        "vram":      {},
    }

    try:
        stats["milvus"] = milvus.get_collection_stats()
    except Exception as e:
        stats["milvus"] = {"error": str(e)}

    try:
        stats["tantivy"] = tantivy.get_stats()
    except Exception as e:
        stats["tantivy"] = {"error": str(e)}

    try:
        stats["sqlite"] = sqlite.get_stats()
    except Exception as e:
        stats["sqlite"] = {"error": str(e)}

    try:
        stats["vram"] = vram_manager.status()
    except Exception as e:
        stats["vram"] = {"error": str(e)}

    return stats
