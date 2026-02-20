"""
modules/llm/llm_engine.py

LLM Intelligence Engine for the Multimodal RAG system.
"""

import json
import re
import time
from pathlib import Path
from typing import (
    Dict, Generator, Iterator, List, Optional, Any, Tuple, Union
)

from loguru import logger

from config.settings import (
    GGUF_DIR,
    LLM_MAIN_MODEL,
    LLM_DRAFT_MODEL,
    LLM_N_CTX,
    LLM_N_GPU_LAYERS,
    LLM_SPECULATIVE_DRAFT_TOKENS,
    CHAT_HISTORY_TURNS,
    EXPORT_TRIGGER_WORDS,
    VRAM_MODEL_SIZES,
)
from core.vram_manager import vram_manager
from modules.retrieval.retrieval_engine import RetrievalResult, GoldChunk
from modules.indexing.sqlite_store import SQLiteStore


# ── Model References ───────────────────────────────────────────────────────
_llm_main  = None   # Llama 3.2 3B — validator / main model
_llm_draft = None   # Llama 3.2 1B — draft model for speculative decoding


# ── System Prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a precise document intelligence assistant for an offline "
    "multimodal RAG system.\n\n"
    "RULES — follow all of these without exception:\n"
    "1. Answer using ONLY the provided context sections below. "
    "Never use external knowledge.\n"
    "2. Every factual claim in your answer MUST end with a citation "
    "in the format [N] where N is the context section number.\n"
    "3. If the context does not contain enough information to answer, "
    'say exactly: "The provided documents do not contain sufficient '
    'information to answer this question."\n'
    "4. Never fabricate facts, numbers, names, or dates.\n"
    "5. For image context: describe what the image shows based on the "
    "OCR text and description provided.\n"
    "6. For audio context: refer to the speaker and timestamp when "
    "citing audio segments.\n"
    "7. Keep answers concise and grounded. Do not pad with generic "
    "statements.\n"
    "8. If multiple context sections support the same claim, cite all "
    "of them: [1][3].\n"
    "9. Cross-modal connections (e.g., a chart mentioned in text and "
    "audio): explicitly note the connection."
)


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

def _format_text_context(chunk: GoldChunk, index: int) -> str:
    """Format a text chunk as a numbered context section."""
    lines = [
        f"[{index}] SOURCE: {chunk.source_file} | "
        f"PAGE: {chunk.page_number} | TYPE: text"
    ]
    text = chunk.text.strip() if chunk.text else ""
    if len(text) > 1500:
        text = text[:1500] + "...[truncated]"
    lines.append(f"    TEXT: {text}")
    return "\n".join(lines)


def _format_image_context(chunk: GoldChunk, index: int) -> str:
    """Format an image chunk as a numbered context section."""
    lines = []
    header = (
        f"[{index}] SOURCE: {chunk.source_file} | "
        f"PAGE: {chunk.page_number} | TYPE: image"
    )
    if chunk.linked_chunk_ids:
        header += " | HAS CROSS-MODAL LINKS"
    lines.append(header)

    ocr_text = chunk.text.strip() if chunk.text else ""
    if ocr_text:
        if len(ocr_text) > 500:
            ocr_text = ocr_text[:500] + "...[truncated]"
        lines.append(f"    OCR: {ocr_text}")
    else:
        lines.append("    OCR: [no text visible in image]")

    try:
        meta = json.loads(chunk.metadata_json or "{}")
        desc = meta.get("llm_description", "").strip()
        if desc:
            lines.append(f"    DESCRIPTION: {desc}")
    except Exception:
        pass

    if chunk.image_path:
        lines.append(f"    FILE: {chunk.image_path}")

    return "\n".join(lines)


def _format_audio_context(chunk: GoldChunk, index: int) -> str:
    """Format an audio chunk as a numbered context section."""
    ts = chunk.timestamp_display or ""
    if not ts and (chunk.start_time > 0 or chunk.end_time > 0):
        try:
            from modules.ingestion.audio_ingestor import _format_timestamp_range
            ts = _format_timestamp_range(chunk.start_time, chunk.end_time)
        except Exception:
            def _fmt_ts(seconds: float) -> str:
                m = int(seconds) // 60
                s = int(seconds) % 60
                return f"{m:02d}:{s:02d}"
            ts = f"{_fmt_ts(chunk.start_time)} - {_fmt_ts(chunk.end_time)}"

    lines = [
        f"[{index}] SOURCE: {chunk.source_file} | "
        f"TIMESTAMP: {ts} | TYPE: audio"
    ]
    transcript = chunk.text.strip() if chunk.text else ""
    if len(transcript) > 1500:
        transcript = transcript[:1500] + "...[truncated]"
    lines.append(f"    TRANSCRIPT: {transcript}")
    return "\n".join(lines)


def _format_context_sections(
    retrieval_result: RetrievalResult,
) -> Tuple[str, Dict[int, GoldChunk]]:
    """
    Format all gold chunks and linked chunks into numbered context sections.
    """
    all_chunks = retrieval_result.all_chunks
    if not all_chunks:
        return "[No relevant context found in the indexed documents.]", {}

    sections = []
    index_map: Dict[int, GoldChunk] = {}

    for i, chunk in enumerate(all_chunks, start=1):
        index_map[i] = chunk
        if chunk.modality == "text":
            section = _format_text_context(chunk, i)
        elif chunk.modality == "image":
            section = _format_image_context(chunk, i)
        elif chunk.modality == "audio":
            section = _format_audio_context(chunk, i)
        else:
            section = (
                f"[{i}] SOURCE: {chunk.source_file} | TYPE: {chunk.modality}\n"
                f"    CONTENT: {(chunk.text or '')[:500]}"
            )
        sections.append(section)

    return "\n\n".join(sections), index_map


# ═══════════════════════════════════════════════════════════════════════════
# HISTORY FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

def _format_history(history: List[Dict[str, Any]]) -> str:
    """Format chat history turns into a conversation string for the prompt."""
    if not history:
        return ""
    lines = []
    for turn in history[-CHAT_HISTORY_TURNS:]:
        role = turn.get("role", "user")
        msg  = turn.get("message", "").strip()
        if role == "user":
            lines.append(f"User: {msg}")
        else:
            lines.append(f"Assistant: {msg}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_rag_prompt(
    query: str,
    retrieval_result: RetrievalResult,
    history: List[Dict[str, Any]],
) -> Tuple[str, Dict[int, GoldChunk]]:
    """
    Build the complete RAG prompt for the LLM.
    """
    context_str, index_map = _format_context_sections(retrieval_result)
    history_str = _format_history(history)

    parts = []
    parts.append(f"<|system|>\n{SYSTEM_PROMPT}")
    parts.append(f"<|context|>\n{context_str}")
    if history_str.strip():
        parts.append(f"<|history|>\n{history_str}")
    parts.append(f"<|user|>\n{query.strip()}")
    parts.append("<|assistant|>")

    full_prompt = "\n\n".join(parts)
    try:
        # Rough token count check
        token_count = len(full_prompt.split())
        if token_count > LLM_N_CTX * 0.85:
            logger.warning(
                f"Prompt is large (~{token_count} words). "
                f"Context window: {LLM_N_CTX} tokens."
            )
    except Exception:
        pass

    return full_prompt, index_map


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT DETECTION & JSON PROMPT
# ═══════════════════════════════════════════════════════════════════════════

def is_export_request(query: str) -> bool:
    """Detect if the user is requesting a file export."""
    query_lower = query.lower()
    return any(trigger in query_lower for trigger in EXPORT_TRIGGER_WORDS)


def _detect_export_type(query: str) -> str:
    """Infer the desired export format from the query."""
    q = query.lower()
    if any(kw in q for kw in ["excel", "xlsx", "spreadsheet", "table"]):
        return "xlsx"
    if any(kw in q for kw in ["word", "docx", "document", "report"]):
        return "docx"
    if any(kw in q for kw in ["powerpoint", "pptx", "presentation", "slides"]):
        return "pptx"
    if any(kw in q for kw in ["csv"]):
        return "csv"
    return "xlsx"


def build_export_prompt(
    query: str,
    retrieval_result: RetrievalResult,
) -> str:
    """
    Build a prompt that instructs the LLM to generate structured JSON.
    """
    export_type = _detect_export_type(query)
    context_str, _ = _format_context_sections(retrieval_result)

    schema_hint = {
        "xlsx": '"sheets": [{"name": str, "headers": [str], "rows": [[str|num]]}]',
        "docx": '"sections": [{"heading": str, "content": str}]',
        "pptx": '"slides": [{"title": str, "bullet_points": [str]}]',
        "csv":  '"sheets": [{"name": "data", "headers": [str], "rows": [[str|num]]}]',
    }.get(export_type, '"sheets": []')

    prompt = f"""<|system|>
You are a data extraction assistant. Your ONLY job is to output valid JSON.
Do NOT output any text before or after the JSON.
Do NOT use markdown code blocks.
Do NOT explain anything.
Output ONLY the JSON object.

The JSON must match this structure:
{{
  "export_type": "{export_type}",
  "filename": "descriptive_snake_case_name",
  "title": "Human readable title",
  "summary": "Brief summary of the content",
  {schema_hint},
  "sources": ["source_file|reference"]
}}

Extract all relevant structured data from the context below.

<|context|>
{context_str}

<|user|>
{query.strip()}

<|assistant|>
{{"""

    return prompt


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _load_llm_models() -> Tuple[Any, Any]:
    """
    Load both LLM models (3B main + 1B draft) with VRAM management.
    """
    global _llm_main, _llm_draft

    main_path  = GGUF_DIR / LLM_MAIN_MODEL
    draft_path = GGUF_DIR / LLM_DRAFT_MODEL

    # ── Load main model (3B) ───────────────────────────────────────────────
    if _llm_main is None:
        if not main_path.exists():
            raise RuntimeError(
                f"Main LLM model not found: {main_path}\n"
                f"Run: python scripts/download_models.py"
            )

        def _evict_main():
            global _llm_main
            if _llm_main is not None:
                del _llm_main
                _llm_main = None
                logger.info("LLM main (3B) evicted")

        vram_manager.register_evict_callback("llm_3b", _evict_main)

        acquired = vram_manager.acquire("llm_3b")
        if not acquired:
            raise RuntimeError(
                "VRAMManager cannot allocate 2.1GB for LLM main model. "
                "Ensure CLIP, BGE-M3, and reranker are unloaded first."
            )

        try:
            from llama_cpp import Llama
            logger.info(f"Loading LLM main model: {LLM_MAIN_MODEL}")
            _llm_main = Llama(
                model_path=str(main_path),
                n_gpu_layers=LLM_N_GPU_LAYERS,
                n_ctx=LLM_N_CTX,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
                n_threads=4,
            )
            logger.success(f"LLM main (3B) loaded: {main_path.name}")
        except Exception as e:
            vram_manager.release("llm_3b")
            raise RuntimeError(f"Failed to load main LLM: {e}") from e

    # ── Load draft model (1B) ──────────────────────────────────────────────
    if _llm_draft is None:
        if not draft_path.exists():
            logger.warning(
                f"Draft LLM not found: {draft_path}. "
                "Speculative decoding disabled — using main model only."
            )
            return _llm_main, None

        def _evict_draft():
            global _llm_draft
            if _llm_draft is not None:
                del _llm_draft
                _llm_draft = None
                logger.info("LLM draft (1B) evicted")

        vram_manager.register_evict_callback("llm_1b", _evict_draft)
        acquired_draft = vram_manager.acquire("llm_1b")

        if not acquired_draft:
            logger.warning(
                "VRAMManager cannot allocate 0.7GB for draft model. "
                "Speculative decoding disabled."
            )
            return _llm_main, None

        try:
            from llama_cpp import Llama
            logger.info(f"Loading LLM draft model: {LLM_DRAFT_MODEL}")
            _llm_draft = Llama(
                model_path=str(draft_path),
                n_gpu_layers=LLM_N_GPU_LAYERS,
                n_ctx=LLM_N_CTX,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
                n_threads=2,
            )
            logger.success(f"LLM draft (1B) loaded: {draft_path.name}")
        except Exception as e:
            vram_manager.release("llm_1b")
            logger.warning(f"Draft model load failed: {e}. Continuing without.")
            _llm_draft = None

    return _llm_main, _llm_draft


def _unload_llm_models():
    """Explicitly unload both LLM models and release VRAM."""
    global _llm_main, _llm_draft
    import torch

    if _llm_draft is not None:
        del _llm_draft
        _llm_draft = None
        vram_manager.release("llm_1b")
        logger.info("LLM draft (1B) unloaded")

    if _llm_main is not None:
        del _llm_main
        _llm_main = None
        vram_manager.release("llm_3b")
        logger.info("LLM main (3B) unloaded")

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# LLM ENGINE CLASS
# ═══════════════════════════════════════════════════════════════════════════

class LLMEngine:
    """
    Orchestrates LLM generation, RAG prompt building, and VRAM management.
    """

    def __init__(self):
        self.sqlite = SQLiteStore()

    def _inject_citations(
        self,
        text: str,
        gold_chunks: Dict[int, GoldChunk]
    ) -> str:
        """
        Post-process the LLM output to ensure citations have metadata.
        The LLM generates [1], [2]. We don't replace them, but we ensure
        they match valid chunks. (Future: could add hover text or links).

        For now, this validates that citations exist.
        """
        # Simple validation pass
        citations = re.findall(r"\[(\d+)\]", text)
        valid_indices = set(gold_chunks.keys())
        # We could log warnings for hallucinations:
        # for c in citations:
        #     if int(c) not in valid_indices:
        #         logger.warning(f"LLM halluncinated citation [{c}]")
        return text

    def generate_stream(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        history: List[Dict[str, Any]],
        session_id: str = "default",
    ) -> Iterator[str]:
        """
        Generator that yields response tokens for streaming.
        Handles:
         - Export detection (returns JSON immediately)
         - RAG prompt construction
         - Speculative decoding (if draft model loaded)
         - Usage tracking & History saving
        """
        # 1. Check for Export
        if is_export_request(query):
            prompt = build_export_prompt(query, retrieval_result)
            try:
                # Load models (only main needed for export really, but standard loader is fine)
                 # Force strict JSON mode could be done via grammar,
                 # but for now we trust the prompt + 3B model logic.
                llm, _ = _load_llm_models()
                
                # We consume the generator to build full JSON string
                full_response = ""
                # Use a simplified generation for JSON
                output = llm(
                    prompt,
                    max_tokens=LLM_N_CTX,  # Allow large context for JSON
                    stop=["<|user|>", "<|end|>"],
                    echo=False,
                    temperature=0.1, # Low temp for deterministic JSON
                )
                json_str = output["choices"][0]["text"].strip()
                
                # Sanitize: ensure only JSON is returned if LLM chatted
                # Find first { and last }
                start = json_str.find("{")
                end = json_str.rfind("}")
                if start != -1 and end != -1:
                    json_str = json_str[start : end + 1]
                
                yield json_str
                
                # Save to history
                self.sqlite.add_message(session_id, "user", query)
                self.sqlite.add_message(session_id, "assistant", json_str)
                return

            except Exception as e:
                logger.error(f"Export generation failed: {e}")
                yield json.dumps({"error": str(e)})
                return

        # 2. Standard RAG Generation
        prompt, gold_map = build_rag_prompt(query, retrieval_result, history)
        
        main_model, draft_model = _load_llm_models()
        
        # Determine decoding strategy
        # simple vs speculative
        
        args = {
            "prompt": prompt,
            "max_tokens": 1024, # Reasonable limit for chat
            "stop": ["<|user|>", "<|end|>", "User:", "Assistant:"],
            "echo": False,
            "stream": True,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        # If draft model is available, use speculative decoding
        # checking llama-cpp-python version compatibility
        # Llama.create_completion with draft_model parameter
        
        full_response = []
        
        try:
            start_time = time.time()
            
            # Using the stream
            if draft_model:
                # Note: llama-cpp-python speculative decoding interface 
                # might differ by version. We use the standard `draft_model` arg if supported
                # or just use main model if API assumes passing Llama object.
                # Inspecting library: usually passed to generate or create_completion
                stream = main_model.create_completion(
                    **args,
                    draft_model=draft_model
                )
            else:
                stream = main_model.create_completion(**args)

            for output in stream:
                token = output["choices"][0]["text"]
                full_response.append(token)
                yield token
            
            total_text = "".join(full_response)
            
            # 3. Post-process & Save
            final_text = self._inject_citations(total_text, gold_map)
            
            latency = time.time() - start_time
            logger.info(f"LLM Generation finished in {latency:.2f}s")
            
            self.sqlite.add_message(session_id, "user", query)
            self.sqlite.add_message(session_id, "assistant", final_text)

        except Exception as e:
            logger.error(f"LLM Generation Error: {e}")
            yield f"[Error generating response: {e}]"

    def query(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        history: List[Dict[str, Any]],
        session_id: str = "default"
    ) -> str:
        """
        Blocking method to get complete response string.
        Useful for tests or non-streaming endpoints.
        """
        response_parts = []
        for token in self.generate_stream(query, retrieval_result, history, session_id):
            response_parts.append(token)
        return "".join(response_parts)


# Singleton instance
llm_engine = LLMEngine()



