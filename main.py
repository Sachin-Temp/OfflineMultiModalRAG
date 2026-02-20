"""
main.py — Entry point for the Multimodal RAG System.
Launches the FastAPI backend and Gradio frontend.
In later phases, this will wire all modules together.
"""

from loguru import logger
from config.settings import LOG_DIR, LOG_LEVEL

# Configure logging
logger.add(
    LOG_DIR / "rag_system.log",
    rotation="50MB",
    retention="7 days",
    level=LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}"
)

logger.info("Multimodal RAG System starting...")
logger.info("Phase 0 scaffold loaded. Run subsequent phases to build modules.")

if __name__ == "__main__":
    print("="*60)
    print("Multimodal RAG System")
    print("="*60)
    print("To run the backend API:")
    print("  python api/main.py")
    print()
    print("To run the frontend UI:")
    print("  python frontend/app.py")
    print()
    print("To check system health:")
    print("  python scripts/health_check.py")
    print("="*60)
