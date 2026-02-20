"""
scripts/benchmark.py

Performance benchmarks for key system components.
Measures latency and throughput for SQLite operations,
citation processing, export generation, and VRAM management.

Usage:
    python scripts/benchmark.py
"""

import sys
import os
import time
import tempfile
import statistics
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def fmt_ms(seconds: float) -> str:
    """Format seconds as milliseconds string."""
    return f"{seconds * 1000:.2f}ms"


def fmt_ops(count: int, seconds: float) -> str:
    """Format operations per second."""
    if seconds == 0:
        return "∞ ops/s"
    return f"{count / seconds:.0f} ops/s"


def run_benchmark(name: str, func, iterations: int = 100):
    """Run a benchmark function N times and report statistics."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg = statistics.mean(times)
    p50 = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    p99 = sorted(times)[int(len(times) * 0.99)]
    total = sum(times)

    return {
        "name": name,
        "iterations": iterations,
        "avg": avg,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "total": total,
        "throughput": fmt_ops(iterations, total),
    }


# ── SQLite Benchmarks ──────────────────────────────────────────────────────

def bench_sqlite_insert():
    """Benchmark SQLite message insertion."""
    from modules.indexing.sqlite_store import SQLiteStore

    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteStore(db_path=Path(tmp) / "bench.db")
        store.initialize()

        def insert():
            store.add_message("bench-sess", "user", "What is machine learning?")

        return run_benchmark("SQLite INSERT (message)", insert, iterations=500)


def bench_sqlite_query():
    """Benchmark SQLite history retrieval."""
    from modules.indexing.sqlite_store import SQLiteStore

    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteStore(db_path=Path(tmp) / "bench.db")
        store.initialize()

        # Pre-populate
        for i in range(50):
            store.add_message("bench-sess", "user" if i % 2 == 0 else "assistant", f"Message {i}")

        def query():
            store.get_history("bench-sess", last_n=10)

        return run_benchmark("SQLite SELECT (history)", query, iterations=500)


# ── Citation Benchmarks ────────────────────────────────────────────────────

def bench_citation_processing():
    """Benchmark CitationEngine.process()."""
    from modules.citation.citation_engine import CitationEngine
    from modules.retrieval.retrieval_engine import GoldChunk

    engine = CitationEngine()
    response = "Machine learning [1] uses data [2] for training models [3]. Neural networks [1] are powerful [2]."
    index_map = {
        1: GoldChunk(chunk_id="c1", modality="text", text="ML text", source_file="a.pdf", page_number=1, reranker_score=0.9, rrf_score=0.8),
        2: GoldChunk(chunk_id="c2", modality="image", text="Image desc", source_file="a.pdf", page_number=2, reranker_score=0.85, rrf_score=0.7),
        3: GoldChunk(chunk_id="c3", modality="audio", text="Audio text", source_file="b.mp3", page_number=0, reranker_score=0.8, rrf_score=0.6),
    }

    def process():
        engine.process(response, [], index_map)

    return run_benchmark("Citation process()", process, iterations=200)


# ── Export Benchmarks ──────────────────────────────────────────────────────

def bench_export(fmt: str):
    """Benchmark export generation for a given format."""
    from modules.export.export_engine import ExportEngine

    export_data = {
        "title": "Benchmark Report",
        "summary": "Testing export performance.",
        "format": fmt,
        "sections": [
            {"heading": f"Section {i}", "body": f"Content for section {i}. " * 20}
            for i in range(5)
        ],
        "sources": [
            {"file": f"source_{i}.pdf", "pages": f"{i}-{i+5}"}
            for i in range(3)
        ],
    }

    if fmt == "csv":
        export_data["sheets"] = [
            {
                "name": "Data",
                "headers": ["Topic", "Description", "Score"],
                "rows": [[f"Topic {i}", f"Description {i}", f"{0.9 - i * 0.1:.1f}"] for i in range(10)],
            }
        ]

    with tempfile.TemporaryDirectory() as tmp:
        with patch("modules.export.export_engine.OUTPUT_DIR", Path(tmp)):
            engine = ExportEngine()

            def export():
                engine.export(export_data)

            return run_benchmark(f"Export {fmt.upper()}", export, iterations=20)


# ── VRAM Manager Benchmarks ───────────────────────────────────────────────

def bench_vram_acquire_release():
    """Benchmark VRAM acquire/release cycles."""
    from core.vram_manager import VRAMManager

    def cycle():
        mgr = object.__new__(VRAMManager)
        mgr._initialized = False
        mgr.__init__()
        mgr.acquire("clip")
        mgr.acquire("bge_m3")
        mgr.release("clip")
        mgr.release("bge_m3")

    return run_benchmark("VRAM acquire/release", cycle, iterations=1000)


# ── Main ───────────────────────────────────────────────────────────────────

def print_results(results):
    """Print benchmark results as a formatted table."""
    print()
    print("┌" + "─" * 32 + "┬" + "─" * 10 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┬" + "─" * 14 + "┐")
    print(f"│ {'Benchmark':<30} │ {'Iters':>8} │ {'Avg':>10} │ {'P50':>10} │ {'P95':>10} │ {'Throughput':>12} │")
    print("├" + "─" * 32 + "┼" + "─" * 10 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 14 + "┤")

    for r in results:
        print(
            f"│ {r['name']:<30} │ {r['iterations']:>8} │ "
            f"{fmt_ms(r['avg']):>10} │ {fmt_ms(r['p50']):>10} │ "
            f"{fmt_ms(r['p95']):>10} │ {r['throughput']:>12} │"
        )

    print("└" + "─" * 32 + "┴" + "─" * 10 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┴" + "─" * 14 + "┘")


def main():
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    print("=" * 60)
    print("  Multimodal RAG — Performance Benchmarks")
    print("=" * 60)

    results = []

    print("\n  Running SQLite benchmarks...")
    results.append(bench_sqlite_insert())
    results.append(bench_sqlite_query())

    print("  Running Citation benchmarks...")
    results.append(bench_citation_processing())

    print("  Running Export benchmarks...")
    for fmt in ["xlsx", "docx", "pptx", "csv"]:
        try:
            results.append(bench_export(fmt))
        except Exception as e:
            print(f"    ⚠ {fmt.upper()} benchmark failed: {e}")

    print("  Running VRAM benchmarks...")
    results.append(bench_vram_acquire_release())

    print_results(results)

    print(f"\n  Total benchmark time: {sum(r['total'] for r in results):.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
