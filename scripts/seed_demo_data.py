"""
scripts/seed_demo_data.py

Seeds the SQLiteStore with sample demo data for immediate system validation.
Creates sample chat history, cross-modal links, and ingested file records
across multiple sessions.

Usage:
    python scripts/seed_demo_data.py
"""

import sys
import os
import uuid
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.indexing.sqlite_store import SQLiteStore
from config.settings import SQLITE_DB_PATH


def seed_chat_history(store: SQLiteStore) -> int:
    """Seed chat history across 3 demo sessions."""
    sessions = {
        "demo-sess-001": [
            ("user", "What is machine learning?"),
            ("assistant", "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience [1]. It involves algorithms that can identify patterns in data [2]."),
            ("user", "Show me the diagram from the report"),
            ("assistant", "Here is the ML pipeline diagram from report.pdf, page 3 [1]. It shows the data preprocessing, model training, and evaluation stages."),
        ],
        "demo-sess-002": [
            ("user", "Summarize the lecture recording"),
            ("assistant", "The lecture covers three main topics [1]: First, the fundamentals of neural networks [2]. Second, backpropagation and gradient descent [3]. Third, practical applications in NLP."),
            ("user", "What was said about transformers?"),
            ("assistant", "In the lecture at 15:30, the speaker discusses transformer architecture [1], noting that attention mechanisms are key to their success."),
        ],
        "demo-sess-003": [
            ("user", "Compare the quarterly reports"),
            ("assistant", "Q1 revenue was $2.3M [1] while Q2 rose to $3.1M [2]. The growth chart on page 7 [3] shows a clear upward trend."),
            ("user", "Export this as an Excel file"),
            ("assistant", "I've generated the Excel report with quarterly comparisons. You can download it from the export panel."),
        ],
    }

    total = 0
    for session_id, messages in sessions.items():
        for role, message in messages:
            cited = []
            # Extract citation markers
            import re
            refs = re.findall(r"\[(\d+)\]", message)
            cited = [f"chunk-{session_id}-{r}" for r in refs]
            store.add_message(session_id, role, message, cited_chunks=cited)
            total += 1

    return total


def seed_cross_modal_links(store: SQLiteStore) -> int:
    """Seed cross-modal links between demo chunks."""
    links = [
        # Same-page links (text ↔ image on same PDF page)
        ("chunk-txt-001", "chunk-img-001", "text", "image", "same_page", 1.0, "report.pdf"),
        ("chunk-txt-002", "chunk-img-002", "text", "image", "same_page", 1.0, "report.pdf"),
        # Temporal proximity (audio ↔ text in same session)
        ("chunk-aud-001", "chunk-txt-001", "audio", "text", "temporal_proximity", 0.7, None),
        ("chunk-aud-002", "chunk-txt-002", "audio", "text", "temporal_proximity", 0.7, None),
        # Semantic links
        ("chunk-txt-001", "chunk-aud-001", "text", "audio", "semantic", 0.85, None),
        ("chunk-img-001", "chunk-aud-001", "image", "audio", "semantic", 0.72, None),
    ]

    count = 0
    for chunk_a, chunk_b, mod_a, mod_b, link_type, strength, source in links:
        try:
            store._upsert_link(chunk_a, chunk_b, mod_a, mod_b, link_type, strength, source)
            count += 1
        except Exception as e:
            print(f"  ⚠ Failed to create link {chunk_a} ↔ {chunk_b}: {e}")

    return count


def seed_ingested_files(store: SQLiteStore) -> int:
    """Seed ingested file records."""
    import hashlib

    files = [
        ("report.pdf", "text", "demo-sess-001", "/data/uploads/report.pdf"),
        ("slides.pptx", "text", "demo-sess-001", "/data/uploads/slides.pptx"),
        ("lecture.mp3", "audio", "demo-sess-002", "/data/uploads/lecture.mp3"),
        ("chart.png", "image", "demo-sess-003", "/data/uploads/chart.png"),
        ("q1_report.pdf", "text", "demo-sess-003", "/data/uploads/q1_report.pdf"),
    ]

    count = 0
    conn = store._get_connection()
    try:
        for filename, modality, session_id, file_path in files:
            file_hash = hashlib.sha256(filename.encode()).hexdigest()
            conn.execute(
                """INSERT OR REPLACE INTO ingested_files
                   (file_hash, filename, file_path, modality, session_id, chunk_count, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (file_hash, filename, file_path, modality, session_id, 10, "completed"),
            )
            count += 1
        conn.commit()
    finally:
        conn.close()

    return count


def main():
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    print("=" * 60)
    print("  Multimodal RAG — Demo Data Seeder")
    print("=" * 60)
    print()

    db_path = SQLITE_DB_PATH
    print(f"  Database: {db_path}")
    print()

    store = SQLiteStore(db_path=db_path)
    store.initialize()

    # Seed chat history
    msg_count = seed_chat_history(store)
    print(f"  ✓ Seeded {msg_count} chat messages across 3 sessions")

    # Seed cross-modal links
    link_count = seed_cross_modal_links(store)
    print(f"  ✓ Seeded {link_count} cross-modal links")

    # Seed ingested files
    file_count = seed_ingested_files(store)
    print(f"  ✓ Seeded {file_count} ingested file records")

    print()

    # Summary
    sessions = store.get_session_ids()
    print(f"  Sessions in database: {len(sessions)}")
    for sid in sessions:
        history = store.get_history(sid, last_n=100)
        print(f"    • {sid}: {len(history)} messages")

    print()
    print("  Demo data seeding complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
