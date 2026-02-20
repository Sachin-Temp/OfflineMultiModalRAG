"""
import sys
from pathlib import Path
# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

Gradio frontend for the Multimodal RAG System.
Connects to the Phase 9 FastAPI backend via APIClient.

Layout:
  - Header:   Title + VRAM badge
  - Left:     Upload panel + session controls + file list
  - Center:   Chat + query controls + export bar
  - Right:    Citations + cross-modal clusters + source summary
  - Footer:   System stats bar

All API calls go through frontend.api_client.APIClient.
All state (session_id, last_citation_result, last_query) is stored
in gr.State() objects — not in globals.

Run with:
    python frontend/app.py
    # or
    python -m frontend.app
"""

import json
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import sys
from pathlib import Path

# Add project root to sys.path if running as script
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))

import gradio as gr
from loguru import logger

from frontend.api_client import APIClient, APIError

# ── Config ─────────────────────────────────────────────────────────────────
BACKEND_URL  = os.getenv("BACKEND_URL",  "http://localhost:8000")
GRADIO_HOST  = os.getenv("GRADIO_HOST",  "0.0.0.0")
GRADIO_PORT  = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"

# Color palette for modality badges
MODALITY_COLORS = {
    "text":  "#1E40AF",
    "image": "#065F46",
    "audio": "#92400E",
}
LINK_TYPE_COLORS = {
    "same_page":          "#2563EB",
    "temporal_proximity": "#7C3AED",
    "co_retrieved":       "#059669",
    "semantic":           "#DC2626",
}
LINK_TYPE_EMOJIS = {
    "same_page":          "📄",
    "temporal_proximity": "🕐",
    "co_retrieved":       "🔁",
    "semantic":           "🔗",
}

# ── API Client ─────────────────────────────────────────────────────────────
_client = APIClient(base_url=BACKEND_URL)


# ── HTML Builders ──────────────────────────────────────────────────────────
def _vram_badge_html(vram: Dict[str, Any]) -> str:
    """
    Build VRAM status badge HTML for the header.

    Shows: used GB / ceiling GB, colored bar, loaded model names.
    Green if < 70%, amber if 70–90%, red if > 90%.
    """
    used    = float(vram.get("used_gb",    0.0))
    ceiling = float(vram.get("ceiling_gb", 3.5))
    models  = vram.get("loaded_models", [])
    pct     = (used / ceiling * 100) if ceiling > 0 else 0

    if pct < 70:
        bar_color = "#10B981"   # green
        text_color = "#065F46"
    elif pct < 90:
        bar_color = "#F59E0B"   # amber
        text_color = "#92400E"
    else:
        bar_color = "#EF4444"   # red
        text_color = "#7F1D1D"

    model_str = ", ".join(models) if models else "none"

    return f"""
<div style="
    display:flex; align-items:center; gap:12px;
    padding:8px 16px; background:#F9FAFB;
    border:1px solid #E5E7EB; border-radius:8px;
    font-family:monospace; font-size:13px;
">
  <span style="color:#374151; font-weight:600;">VRAM</span>
  <div style="
      width:120px; height:10px; background:#E5E7EB;
      border-radius:5px; overflow:hidden;
  ">
    <div style="
        width:{pct:.0f}%; height:100%;
        background:{bar_color}; border-radius:5px;
        transition:width 0.3s ease;
    "></div>
  </div>
  <span style="color:{text_color}; font-weight:600;">
    {used:.1f} / {ceiling:.1f} GB ({pct:.0f}%)
  </span>
  <span style="color:#6B7280; font-size:11px;">
    [{model_str}]
  </span>
</div>
"""


def _citation_card_html(citation: Dict[str, Any], index: int) -> str:
    """
    Build HTML for a single citation card in the right panel.

    Contains:
    - Numbered badge [N]
    - Modality badge (TEXT / IMAGE / AUDIO)
    - Source file + page / timestamp
    - Text preview (truncated)
    - Thumbnail image (if image modality)
    - Cross-modal link badges
    """
    modality    = citation.get("modality", "text")
    source_file = citation.get("source_file", "")
    page_number = citation.get("page_number", 1)
    ts_display  = citation.get("timestamp_display", "")
    text_preview = citation.get("text_preview", "")
    thumbnail   = citation.get("thumbnail_path", "")
    links       = citation.get("links", [])

    mod_color   = MODALITY_COLORS.get(modality, "#374151")
    mod_label   = modality.upper()

    # Source reference line
    if modality == "audio" and ts_display:
        source_ref = f"{source_file} | {ts_display}"
    else:
        source_ref = f"{source_file} | page {page_number}"

    # Thumbnail HTML (image chunks only)
    thumbnail_html = ""
    # IMPORTANT: Check existence before rendering image tag to prevent Gradio 404s
    if modality == "image" and thumbnail and Path(thumbnail).exists():
        thumbnail_html = f"""
<img src="file={thumbnail}"
     style="width:100%; max-height:120px; object-fit:cover;
            border-radius:4px; margin:6px 0;"
     alt="Thumbnail"/>
"""

    # Link badges HTML
    link_badges = ""
    for lk in links[:4]:
        lt     = lk.get("link_type", "")
        color  = LINK_TYPE_COLORS.get(lt, "#6B7280")
        emoji  = LINK_TYPE_EMOJIS.get(lt, "🔗")
        label  = lk.get("display_label", lt)
        strength = float(lk.get("strength", 0)) * 100
        link_badges += f"""
<span style="
    display:inline-block; margin:2px;
    padding:2px 8px; border-radius:12px;
    background:{color}22; border:1px solid {color};
    color:{color}; font-size:10px; font-weight:600;
" title="Strength: {strength:.0f}%">
  {emoji} {label}
</span>
"""

    return f"""
<div style="
    border:1px solid #E5E7EB; border-radius:8px;
    padding:12px; margin-bottom:10px;
    background:white; box-shadow:0 1px 3px rgba(0,0,0,0.05);
">
  <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
    <span style="
        background:#1E3A5F; color:white;
        font-weight:700; font-size:14px;
        width:26px; height:26px; border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        flex-shrink:0;
    ">{index}</span>
    <span style="
        background:{mod_color}; color:white;
        padding:2px 10px; border-radius:12px;
        font-size:11px; font-weight:700;
        letter-spacing:0.5px;
    ">{mod_label}</span>
    <span style="
        color:#4B5563; font-size:12px;
        white-space:nowrap; overflow:hidden;
        text-overflow:ellipsis; flex:1;
    " title="{source_file}">{source_ref}</span>
  </div>
  {thumbnail_html}
  <p style="
      color:#374151; font-size:12px; margin:6px 0;
      line-height:1.5; max-height:60px; overflow:hidden;
  ">{text_preview[:200]}{"..." if len(text_preview) > 200 else ""}</p>
  <div style="margin-top:6px">{link_badges}</div>
</div>
"""


def _cluster_panel_html(clusters: List[Dict[str, Any]]) -> str:
    """
    Build HTML for the cross-modal clusters panel.
    Each cluster shows: emoji, label, affected citation indices,
    strength progress bar.
    """
    if not clusters:
        return "<p style='color:#9CA3AF; font-size:13px; margin:8px 0;'>No cross-modal connections detected.</p>"

    items = []
    for cluster in clusters:
        lt       = cluster.get("link_type", "")
        color    = LINK_TYPE_COLORS.get(lt, "#6B7280")
        emoji    = LINK_TYPE_EMOJIS.get(lt, "🔗")
        label    = cluster.get("label", "")
        indices  = cluster.get("chunk_indices", [])
        strength = float(cluster.get("strength", 0)) * 100
        idx_str  = " + ".join(f"[{i}]" for i in indices)

        items.append(f"""
<div style="
    border-left:4px solid {color};
    padding:8px 12px; margin-bottom:8px;
    background:{color}0D; border-radius:0 6px 6px 0;
">
  <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
    <span style="font-size:16px;">{emoji}</span>
    <span style="
        background:{color}; color:white;
        padding:1px 8px; border-radius:10px;
        font-size:11px; font-weight:700;
    ">{idx_str}</span>
  </div>
  <p style="color:#374151; font-size:12px; margin:0 0 4px 0;">{label}</p>
  <div style="display:flex; align-items:center; gap:8px;">
    <div style="flex:1; height:4px; background:#E5E7EB; border-radius:2px;">
      <div style="
          width:{strength:.0f}%; height:100%;
          background:{color}; border-radius:2px;
      "></div>
    </div>
    <span style="font-size:10px; color:#6B7280;">{strength:.0f}%</span>
  </div>
</div>
""")

    return "\n".join(items)


def _source_summary_html(sources: List[Dict[str, Any]]) -> str:
    """
    Build HTML for the source summary section.
    Lists each unique source file with its modalities and page references.
    """
    if not sources:
        return "<p style='color:#9CA3AF; font-size:13px;'>No sources cited.</p>"

    modality_icons = {"text": "📄", "image": "🖼️", "audio": "🎵"}
    items = []

    for src in sources:
        filename   = src.get("source_file", "")
        modalities = src.get("modalities", [])
        pages      = src.get("pages", [])
        timestamps = src.get("timestamps", [])
        indices    = src.get("citation_indices", [])

        mod_icons  = " ".join(modality_icons.get(m, "📄") for m in modalities)
        page_str   = f"pp. {', '.join(str(p) for p in pages[:5])}" if pages else ""
        ts_str     = f"at {', '.join(timestamps[:3])}" if timestamps else ""
        idx_str    = " ".join(f"[{i}]" for i in indices)
        ref_str    = " · ".join(filter(None, [page_str, ts_str]))

        items.append(f"""
<div style="
    padding:8px 10px; margin-bottom:6px;
    background:#F9FAFB; border-radius:6px;
    border:1px solid #E5E7EB;
">
  <div style="display:flex; align-items:center; gap:6px;">
    <span style="font-size:14px;">{mod_icons}</span>
    <span style="
        font-size:12px; font-weight:600; color:#1F2937;
        white-space:nowrap; overflow:hidden;
        text-overflow:ellipsis; flex:1;
    " title="{filename}">{filename}</span>
    <span style="font-size:11px; color:#6B7280; white-space:nowrap;">{idx_str}</span>
  </div>
  {f'<p style="font-size:11px; color:#6B7280; margin:2px 0 0 20px;">{ref_str}</p>' if ref_str else ''}
</div>
""")

    return "\n".join(items)


def _stats_bar_html(stats: Dict[str, Any]) -> str:
    """
    Build HTML for the bottom system stats bar.
    Shows Milvus chunk counts, Tantivy doc count,
    SQLite message count, and VRAM usage.
    """
    milvus  = stats.get("milvus",  {})
    tantivy = stats.get("tantivy", {})
    sqlite  = stats.get("sqlite",  {})
    vram    = stats.get("vram",    {})

    text_count  = milvus.get("text_chunks",  {}).get("row_count", 0)
    image_count = milvus.get("image_chunks", {}).get("row_count", 0)
    audio_count = milvus.get("audio_chunks", {}).get("row_count", 0)
    bm25_docs   = tantivy.get("num_docs",    0)
    messages    = sqlite.get("chat_messages", 0)
    links       = sqlite.get("total_links",   0)
    used_gb     = float(vram.get("used_gb",    0.0))
    ceiling_gb  = float(vram.get("ceiling_gb", 3.5))

    return f"""
<div style="
    display:flex; flex-wrap:wrap; gap:16px; align-items:center;
    padding:8px 16px; background:#F3F4F6;
    border-top:1px solid #E5E7EB;
    font-size:12px; color:#374151; font-family:monospace;
">
  <span>📄 Text: <b>{text_count}</b></span>
  <span>🖼️ Image: <b>{image_count}</b></span>
  <span>🎵 Audio: <b>{audio_count}</b></span>
  <span style="border-left:1px solid #D1D5DB; padding-left:16px;">
    🔍 BM25 docs: <b>{bm25_docs}</b>
  </span>
  <span>💬 Messages: <b>{messages}</b></span>
  <span>🔗 Links: <b>{links}</b></span>
  <span style="border-left:1px solid #D1D5DB; padding-left:16px;">
    💾 VRAM: <b>{used_gb:.1f}/{ceiling_gb:.1f} GB</b>
  </span>
</div>
"""


def _file_list_html(files: List[Dict[str, Any]]) -> str:
    """
    Build HTML table for the ingested files list.
    Shows filename, modality icon, chunk count, and status badge.
    """
    if not files:
        return "<p style='color:#9CA3AF; font-size:13px; padding:8px;'>No files ingested yet.</p>"

    modality_icons = {"text": "📄", "image": "🖼️", "audio": "🎵"}
    status_colors  = {
        "complete": "#10B981",
        "pending":  "#F59E0B",
        "failed":   "#EF4444",
    }

    rows = []
    for f in files:
        icon     = modality_icons.get(f.get("modality", "text"), "📄")
        fname    = Path(f.get("filename", "")).name
        chunks   = f.get("chunk_count", 0)
        status   = f.get("status", "complete")
        sc       = status_colors.get(status, "#6B7280")

        rows.append(f"""
<tr>
  <td style="padding:4px 8px;">{icon}</td>
  <td style="padding:4px 8px; font-size:12px; max-width:160px;
             white-space:nowrap; overflow:hidden; text-overflow:ellipsis;"
      title="{fname}">{fname}</td>
  <td style="padding:4px 8px; font-size:12px; text-align:right;">{chunks}</td>
  <td style="padding:4px 8px;">
    <span style="
        background:{sc}22; color:{sc};
        padding:1px 8px; border-radius:10px;
        font-size:10px; font-weight:700;
    ">{status.upper()}</span>
  </td>
</tr>
""")

    return f"""
<table style="width:100%; border-collapse:collapse;">
  <thead>
    <tr style="border-bottom:1px solid #E5E7EB;">
      <th style="padding:4px 8px; text-align:left; font-size:11px; color:#6B7280;"></th>
      <th style="padding:4px 8px; text-align:left; font-size:11px; color:#6B7280;">File</th>
      <th style="padding:4px 8px; text-align:right; font-size:11px; color:#6B7280;">Chunks</th>
      <th style="padding:4px 8px; text-align:left; font-size:11px; color:#6B7280;">Status</th>
    </tr>
  </thead>
  <tbody>{"".join(rows)}</tbody>
</table>
"""


def _upload_log_entry(filename: str, success: bool, detail: str) -> str:
    """Format a single timestamped upload log entry."""
    ts     = datetime.now().strftime("%H:%M:%S")
    icon   = "✅" if success else "❌"
    color  = "#065F46" if success else "#7F1D1D"
    bg     = "#ECFDF5" if success else "#FEF2F2"
    return f"""
<div style="
    padding:6px 10px; margin-bottom:4px;
    background:{bg}; border-radius:6px;
    border-left:3px solid {'#10B981' if success else '#EF4444'};
    font-size:12px;
">
  <span style="color:#6B7280;">[{ts}]</span>
  <span style="margin:0 6px;">{icon}</span>
  <span style="color:{color}; font-weight:600;">{filename}</span>
  <span style="color:#6B7280; margin-left:6px;">{detail}</span>
</div>
"""


def _retrieval_chunks_html(gold_chunks: List[Dict[str, Any]]) -> str:
    """
    Build HTML for retrieve-only mode results.
    Shows each gold chunk with modality, source, and preview.
    Used when the user toggles 'Retrieve Only' mode.
    """
    if not gold_chunks:
        return "<p style='color:#9CA3AF; font-size:13px;'>No chunks retrieved.</p>"

    modality_icons = {"text": "📄", "image": "🖼️", "audio": "🎵"}
    items = []
    for i, chunk in enumerate(gold_chunks, 1):
        modality    = chunk.get("modality", "text")
        source_file = chunk.get("source_file", "")
        page        = chunk.get("page_number", 1)
        ts          = chunk.get("timestamp_display", "")
        text        = chunk.get("text", "")
        score       = float(chunk.get("reranker_score", 0)) * 100
        icon        = modality_icons.get(modality, "📄")
        mod_color   = MODALITY_COLORS.get(modality, "#374151")
        ref         = ts if (modality == "audio" and ts) else f"page {page}"

        items.append(f"""
<div style="
    border:1px solid #E5E7EB; border-radius:8px;
    padding:10px; margin-bottom:8px; background:white;
">
  <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
    <span style="
        background:#1E3A5F; color:white; font-weight:700;
        width:22px; height:22px; border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        font-size:12px; flex-shrink:0;
    ">{i}</span>
    <span style="font-size:14px;">{icon}</span>
    <span style="
        background:{mod_color}; color:white;
        padding:1px 8px; border-radius:10px;
        font-size:10px; font-weight:700;
    ">{modality.upper()}</span>
    <span style="font-size:11px; color:#6B7280; flex:1;"
          title="{source_file}">{Path(source_file).name} · {ref}</span>
    <span style="
        font-size:10px; color:#1E40AF; font-weight:600;
        background:#EFF6FF; padding:1px 6px; border-radius:8px;
    ">{score:.0f}%</span>
  </div>
  <p style="
      font-size:12px; color:#374151; margin:0;
      line-height:1.5; max-height:48px; overflow:hidden;
  ">{text[:300]}{"..." if len(text) > 300 else ""}</p>
</div>
""")

    return "\n".join(items)


# ── Session Helpers ────────────────────────────────────────────────────────
def _new_session_id() -> str:
    """Generate a fresh UUID session ID."""
    return f"sess-{uuid.uuid4().hex[:12]}"


def _load_history_to_chat(session_id: str) -> List[List[str]]:
    """
    Load existing chat history from backend and convert to
    Gradio chatbot format [[user_msg, assistant_msg], ...].
    Pairs up consecutive user/assistant messages.
    """
    try:
        messages = _client.get_history(session_id, last_n=20)
        pairs = []
        i = 0
        while i < len(messages):
            if messages[i]["role"] == "user":
                user_msg = messages[i]["message"]
                asst_msg = ""
                if i + 1 < len(messages) and messages[i+1]["role"] == "assistant":
                    asst_msg = messages[i+1]["message"]
                    i += 2
                else:
                    i += 1
                pairs.append([user_msg, asst_msg])
            else:
                i += 1
        return pairs
    except Exception:
        return []


# ── Core Event Handlers ────────────────────────────────────────────────────
def handle_upload(
    files,
    session_id: str,
    upload_log_html: str,
) -> Tuple[str, str, str]:
    """
    Handle file upload event.
    Called when user clicks the Upload button.

    Args:
        files:           Gradio file upload object (list or single)
        session_id:      Current session ID
        upload_log_html: Existing HTML log to append to

    Returns:
        (updated_upload_log_html, updated_file_list_html, status_message)
    """
    if not files:
        return upload_log_html, "", "⚠️ No files selected."

    if not session_id or not session_id.strip():
        return upload_log_html, "", "❌ Session ID is required."

    # Normalize files to list of paths
    if not isinstance(files, list):
        files = [files]

    new_log  = upload_log_html
    uploaded = 0
    failed   = 0

    for file in files:
        # Gradio provides file as a NamedString or dict with 'name' key
        if hasattr(file, "name"):
            file_path = Path(file.name)
        elif isinstance(file, dict):
            file_path = Path(file.get("name", ""))
        else:
            file_path = Path(str(file))

        fname = file_path.name

        try:
            result = _client.upload_file(file_path, session_id)
            chunks = result.get("chunks_inserted", 0)
            ms     = result.get("ingest_time_ms", 0)
            detail = f"{chunks} chunks · {ms:.0f}ms"
            new_log = _upload_log_entry(fname, True, detail) + new_log
            uploaded += 1
            logger.info(f"Uploaded: {fname} → {chunks} chunks")

        except APIError as e:
            if e.status_code == 409:
                detail = "already ingested (duplicate)"
            else:
                detail = str(e.detail)[:80]
            new_log = _upload_log_entry(fname, False, detail) + new_log
            failed += 1
            logger.warning(f"Upload failed: {fname} — {e}")

        except Exception as e:
            new_log = _upload_log_entry(fname, False, str(e)[:80]) + new_log
            failed += 1

    # Refresh file list
    try:
        files_data  = _client.get_ingested_files(session_id)
        file_list   = _file_list_html(files_data)
    except Exception:
        # If backend doesn't support file list, keep current
        # or return empty to indicate failure/no data
        file_list = "" 

    status = f"✅ {uploaded} uploaded" + (f" · ❌ {failed} failed" if failed else "")
    return new_log, file_list, status


def handle_send(
    query:            str,
    history:          List[List[str]],
    session_id:       str,
    retrieve_only:    bool,
    max_tokens:       int,
    temperature:      float,
    top_k:            int,
    enable_reranking: bool,
) -> Iterator[Tuple[List[List[str]], str, str, str]]:
    """
    Handle query submission. Generator that yields UI updates.

    In stream mode (retrieve_only=False):
      - Yields chatbot history updates token by token
      - After stream ends, yields citation HTML updates

    In retrieve-only mode:
      - Runs retrieval, returns chunks as HTML in citation panel

    Yields tuples:
        (chatbot_history, citation_cards_html, clusters_html, source_summary_html)
    """
    if not query or not query.strip():
        yield history, "", "", ""
        return

    if not session_id:
        session_id = _new_session_id()

    # ── Retrieve-only mode ─────────────────────────────────────────────────
    if retrieve_only:
        history = history + [[query, "🔍 *Retrieving relevant chunks...*"]]
        yield history, "", "", ""

        try:
            result      = _client.retrieve_only(
                query, session_id, top_k=top_k, final_k=5
            )
            gold_chunks = result.get("gold_chunks", [])
            latency_ms  = result.get("latency_ms", {}).get("total", 0)

            # Format retrieval results in chatbot
            chunk_summary = f"**Retrieved {len(gold_chunks)} chunks** in {latency_ms:.0f}ms\n\n"
            for i, chunk in enumerate(gold_chunks, 1):
                mod  = chunk.get("modality", "text")
                src  = Path(chunk.get("source_file", "")).name
                text = chunk.get("text", "")[:150]
                chunk_summary += f"**[{i}] {mod.upper()}** — {src}\n{text}...\n\n"

            history[-1][1] = chunk_summary
            chunks_html    = _retrieval_chunks_html(gold_chunks)
            yield history, chunks_html, "", ""

        except APIError as e:
            history[-1][1] = f"❌ Retrieval error: {e.detail}"
            yield history, "", "", ""

        return

    # ── Full RAG streaming mode ────────────────────────────────────────────
    history       = history + [[query, ""]]
    partial       = ""
    citation_html = ""
    cluster_html  = ""
    source_html   = ""

    yield history, citation_html, cluster_html, source_html

    try:
        citation_result = None

        for event in _client.stream_query(
            query=query,
            session_id=session_id,
            top_k=top_k,
            final_k=5,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_reranking=enable_reranking,
            enable_links=True,
        ):
            event_type = event.get("type", "")
            content    = event.get("content", "")

            if event_type == "token":
                partial         += content
                history[-1][1]   = partial
                yield history, citation_html, cluster_html, source_html

            elif event_type == "citation":
                # Build citation sidebar from citation result
                citations = content.get("citations", [])
                clusters  = content.get("clusters",  [])
                sources   = content.get("source_summary", [])

                citation_html = "\n".join(
                    _citation_card_html(c, c.get("index", i+1))
                    for i, c in enumerate(citations)
                )
                cluster_html = _cluster_panel_html(clusters)
                source_html  = _source_summary_html(sources)

                # Replace [N] markers with styled spans in chatbot
                annotated = content.get("annotated_response", "")
                if annotated:
                    history[-1][1] = annotated

                yield history, citation_html, cluster_html, source_html

            elif event_type == "done":
                tps = content.get("tokens_sec", 0) if isinstance(content, dict) else 0
                logger.info(f"Generation complete: {tps} tokens/sec")
                yield history, citation_html, cluster_html, source_html

            elif event_type == "error":
                history[-1][1] += f"\n\n❌ *Error: {content}*"
                yield history, citation_html, cluster_html, source_html

    except APIError as e:
        history[-1][1] = f"❌ **API Error**: {e.detail}"
        yield history, citation_html, cluster_html, source_html

    except Exception as e:
        history[-1][1] = f"❌ **Error**: {str(e)}"
        yield history, citation_html, cluster_html, source_html


def handle_export(
    query:       str,
    history:     List[List[str]],
    session_id:  str,
    export_type: str,
) -> Tuple[Optional[str], str]:
    """
    Handle export button click.
    Runs /export/generate then downloads the file and returns it
    as a Gradio file component for instant download.

    Returns:
        (temp_file_path_or_None, status_message)
    """
    # Use last query if current textbox is empty
    effective_query = query.strip()
    if not effective_query and history:
        for turn in reversed(history):
            if turn[0]:
                effective_query = turn[0]
                break

    if not effective_query:
        return None, "❌ No query to export. Send a message first."

    if not session_id:
        return None, "❌ No session ID."

    try:
        # Generate export
        meta = _client.generate_export(
            query=effective_query,
            session_id=session_id,
            export_type=export_type,
        )
        filename = meta.get("filename", "")
        if not filename:
            return None, "❌ Export generation failed — no filename returned."

        # Download file bytes
        file_bytes = _client.download_export(filename)

        # Save to temp file for Gradio download component
        ext = Path(filename).suffix
        with tempfile.NamedTemporaryFile(
            suffix=ext, delete=False, prefix="rag_export_"
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        size_kb = len(file_bytes) / 1024
        return tmp_path, f"✅ {filename} ({size_kb:.1f} KB) — click to download"

    except APIError as e:
        if e.status_code == 404:
            return None, "❌ No documents found. Upload files and ask a question first."
        return None, f"❌ Export failed: {e.detail}"

    except Exception as e:
        return None, f"❌ Export error: {e}"


def handle_new_session() -> Tuple[str, List, str, str, str, str]:
    """
    Generate a new session ID and reset all stateful UI components.

    Returns:
        (new_session_id, empty_history, empty_upload_log,
         empty_file_list, empty_citations, empty_clusters, empty_sources)
    """
    new_id = _new_session_id()
    logger.info(f"New session: {new_id}")
    return new_id, [], "", "", "", "", ""


def handle_clear_history(session_id: str) -> Tuple[List, str]:
    """
    Clear chat history for the current session.
    Keeps indexed documents intact.

    Returns:
        (empty_chat_history, status_message)
    """
    if not session_id:
        return [], "❌ No session ID."

    try:
        result  = _client.clear_history(session_id)
        deleted = result.get("deleted", 0)
        return [], f"✅ Cleared {deleted} messages. Documents unchanged."
    except APIError as e:
        return [], f"❌ Clear failed: {e.detail}"


def handle_refresh_stats() -> Tuple[str, str]:
    """
    Refresh VRAM badge and system stats bar.

    Returns:
        (vram_badge_html, stats_bar_html)
    """
    try:
        stats = _client.get_health_stats()
        vram  = stats.get("vram", {})
        return _vram_badge_html(vram), _stats_bar_html(stats)
    except Exception as e:
        error_html = f"<span style='color:#EF4444;'>Backend offline: {e}</span>"
        return error_html, error_html


def handle_load_session_history(session_id: str) -> List[List[str]]:
    """Load existing chat history when session_id changes."""
    if not session_id:
        return []
    return _load_history_to_chat(session_id)


# ── Gradio App Builder ─────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    """
    Build and return the complete Gradio Blocks UI.
    All components are defined here. Event handlers are wired at the end.
    """

    # Check backend connectivity on startup
    backend_alive = _client.is_backend_alive()
    if not backend_alive:
        logger.warning(
            f"Backend not reachable at {BACKEND_URL}. "
            "Start the FastAPI server first: uvicorn api.main:app --port 8000"
        )

    # Initial stats (graceful fallback if backend offline)
    try:
        initial_stats = _client.get_health_stats()
        initial_vram  = initial_stats.get("vram", {})
    except Exception:
        initial_stats = {}
        initial_vram  = {}

    with gr.Blocks(
        title="Multimodal RAG System",
    ) as demo:

        # ── State ──────────────────────────────────────────────────────────
        session_state   = gr.State(_new_session_id())
        last_query      = gr.State("")

        # ── Header ────────────────────────────────────────────────────────
        with gr.Row(elem_classes="rag-header"):
            with gr.Column(scale=3):
                gr.HTML("""
<div style="display:flex; align-items:center; gap:12px;">
  <div style="
      width:36px; height:36px; background:#1E3A5F;
      border-radius:8px; display:flex; align-items:center;
      justify-content:center; font-size:20px;
  ">🧠</div>
  <div>
    <h1 style="margin:0; font-size:22px; color:#1E3A5F; font-weight:800;">
      Multimodal RAG System
    </h1>
    <p style="margin:0; font-size:12px; color:#6B7280;">
      Offline · RTX 3050 4GB · Speculative Decoding · Cross-Modal Links
    </p>
  </div>
</div>
""")
            with gr.Column(scale=2):
                vram_badge = gr.HTML(
                    _vram_badge_html(initial_vram),
                    label="",
                )

        # ── Main Layout ────────────────────────────────────────────────────
        with gr.Row():

            # ── LEFT PANEL: Upload & Session ───────────────────────────────
            with gr.Column(scale=1, min_width=280):

                gr.HTML('<p class="panel-label">Session</p>')
                session_id_box = gr.Textbox(
                    label="Session ID",
                    value=_new_session_id(),
                    interactive=True,
                    info="All uploaded files and history are scoped to this ID",
                )

                with gr.Row():
                    new_session_btn   = gr.Button("🆕 New Session", size="sm", variant="secondary")
                    clear_history_btn = gr.Button("🗑️ Clear Chat",  size="sm", variant="secondary")

                session_action_status = gr.Textbox(
                    label="", interactive=False, visible=True, max_lines=1
                )

                gr.HTML('<p class="panel-label" style="margin-top:16px;">Upload Documents</p>')
                file_upload = gr.File(
                    label="Drag & drop or click to select",
                    file_count="multiple",
                    file_types=[
                        ".pdf", ".docx", ".doc", ".txt",
                        ".png", ".jpg", ".jpeg", ".webp", ".bmp",
                        ".mp3", ".wav", ".m4a", ".ogg", ".flac",
                    ],
                )
                upload_btn = gr.Button("⬆️ Upload & Ingest", variant="primary")

                upload_log = gr.HTML(
                    value="",
                    label="Upload Log",
                    elem_classes="upload-log",
                )

                with gr.Accordion("📁 Ingested Files", open=False):
                    file_list_html = gr.HTML(
                        value=_file_list_html([]),
                    )

            # ── CENTER PANEL: Chat ─────────────────────────────────────────
            with gr.Column(scale=3, min_width=500):

                chatbot = gr.Chatbot(
                    label="Chat",
                    height=480,
                    # show_copy_button=True,  # Removed for Gradio 6.0 compatibility
                    # bubble_full_width=False, # Removed for Gradio 6.0 compatibility
                    render_markdown=True,
                    avatar_images=(
                        None,   # user avatar
                        None,   # assistant avatar
                    ),
                )

                with gr.Row():
                    query_box = gr.Textbox(
                        label="",
                        placeholder="Ask a question about your documents...",
                        lines=2,
                        max_lines=6,
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button(
                        "Send ➤",
                        variant="primary",
                        scale=1,
                        min_width=80,
                    )

                with gr.Row():
                    retrieve_only_toggle = gr.Checkbox(
                        label="🔍 Retrieve Only (no LLM generation)",
                        value=False,
                    )

                with gr.Accordion("⚙️ Advanced Options", open=False):
                    with gr.Row():
                        max_tokens_slider = gr.Slider(
                            label="Max Tokens",
                            minimum=64, maximum=2048,
                            value=512, step=64,
                        )
                        temperature_slider = gr.Slider(
                            label="Temperature",
                            minimum=0.0, maximum=1.0,
                            value=0.1, step=0.05,
                        )
                    with gr.Row():
                        top_k_slider = gr.Slider(
                            label="Retrieval Top-K",
                            minimum=10, maximum=200,
                            value=50, step=10,
                        )
                        reranking_toggle = gr.Checkbox(
                            label="Enable Reranking",
                            value=True,
                        )

                gr.HTML('<p class="panel-label" style="margin-top:12px;">Export Response</p>')
                with gr.Row():
                    export_type_radio = gr.Radio(
                        choices=["xlsx", "docx", "pptx", "csv"],
                        value="xlsx",
                        label="",
                        show_label=False,
                    )
                    export_btn = gr.Button("📥 Export", variant="secondary", min_width=80)

                export_file     = gr.File(label="Download Export", visible=False)
                export_status   = gr.Textbox(
                    label="", interactive=False, visible=True, max_lines=1
                )

            # ── RIGHT PANEL: Citations & Links ─────────────────────────────
            with gr.Column(scale=2, min_width=300):

                with gr.Accordion("📎 Citations", open=True):
                    citation_panel = gr.HTML(
                        value="<p style='color:#9CA3AF; font-size:13px;'>Citations will appear here after your first query.</p>",
                        elem_classes="citation-panel",
                    )

                with gr.Accordion("🔗 Cross-Modal Connections", open=True):
                    cluster_panel = gr.HTML(
                        value="<p style='color:#9CA3AF; font-size:13px;'>Cross-modal links will appear here.</p>",
                    )

                with gr.Accordion("📚 Sources", open=True):
                    source_summary_panel = gr.HTML(
                        value="<p style='color:#9CA3AF; font-size:13px;'>Source summary will appear here.</p>",
                    )

        # ── Footer Stats Bar ───────────────────────────────────────────────
        with gr.Row():
            stats_bar = gr.HTML(
                value=_stats_bar_html(initial_stats),
            )
        with gr.Row():
            refresh_stats_btn = gr.Button("🔄 Refresh Stats", size="sm", variant="secondary")

        # ── Backend Status ─────────────────────────────────────────────────
        if not backend_alive:
            gr.HTML(f"""
<div style="
    background:#FEF2F2; border:1px solid #EF4444;
    border-radius:8px; padding:12px 16px; margin-top:8px;
    color:#7F1D1D; font-size:13px;
">
  ⚠️ <strong>Backend offline</strong> — cannot reach {BACKEND_URL}.
  Start the server: <code>uvicorn api.main:app --port 8000</code>
</div>
""")

        # ── Event Wiring ───────────────────────────────────────────────────

        # Upload
        upload_btn.click(
            fn=handle_upload,
            inputs=[file_upload, session_id_box, upload_log],
            outputs=[upload_log, file_list_html, session_action_status],
        )

        # Send query (streaming)
        send_event = send_btn.click(
            fn=handle_send,
            inputs=[
                query_box, chatbot, session_id_box,
                retrieve_only_toggle,
                max_tokens_slider, temperature_slider,
                top_k_slider, reranking_toggle,
            ],
            outputs=[chatbot, citation_panel, cluster_panel, source_summary_panel],
        )

        # Also allow Enter key to submit (shift+enter for newline)
        query_box.submit(
            fn=handle_send,
            inputs=[
                query_box, chatbot, session_id_box,
                retrieve_only_toggle,
                max_tokens_slider, temperature_slider,
                top_k_slider, reranking_toggle,
            ],
            outputs=[chatbot, citation_panel, cluster_panel, source_summary_panel],
        )

        # Clear query box after sending
        send_btn.click(
            fn=lambda: "",
            outputs=[query_box],
        )
        query_box.submit(
            fn=lambda: "",
            outputs=[query_box],
        )

        # Export
        export_btn.click(
            fn=handle_export,
            inputs=[query_box, chatbot, session_id_box, export_type_radio],
            outputs=[export_file, export_status],
        ).then(
            fn=lambda f: gr.File(visible=f is not None),
            inputs=[export_file],
            outputs=[export_file],
        )

        # New session
        new_session_btn.click(
            fn=handle_new_session,
            outputs=[
                session_id_box, chatbot, upload_log,
                file_list_html, citation_panel,
                cluster_panel, source_summary_panel,
            ],
        )

        # Clear chat history
        clear_history_btn.click(
            fn=handle_clear_history,
            inputs=[session_id_box],
            outputs=[chatbot, session_action_status],
        )

        # Load history when session_id changes
        session_id_box.change(
            fn=handle_load_session_history,
            inputs=[session_id_box],
            outputs=[chatbot],
        )

        # Refresh stats
        refresh_stats_btn.click(
            fn=handle_refresh_stats,
            outputs=[vram_badge, stats_bar],
        )

    return demo


# ── Entry Point ────────────────────────────────────────────────────────────
def main():
    demo = build_ui()
    demo.queue(
        max_size=10,
        default_concurrency_limit=2,
    )
    import socket
    
    # Check if port is in use to avoid scary OSError
    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

    target_port = GRADIO_PORT
    if is_port_in_use(target_port):
        print(f"⚠️ Port {target_port} is in use. Finding an open port...")
        target_port = None  # Let Gradio find a port

    demo.launch(
        server_name=GRADIO_HOST,
        server_port=target_port,
        share=GRADIO_SHARE,
        show_error=True,
    )
    # Note: theme and css moved to launch() is NOT standard Gradio 4/5 API.
    # The warning likely meant that `app_kwargs` like `auth` moved.
    # But `theme` and `css` are definitely Blocks arguments in 4.x/5.x.
    # Wait, the warning said: "The parameters have been moved from the Blocks constructor to the launch() method in Gradio 6.0: theme, css."
    # So I SHOULD move them.
    # But launch() doesn't usually take css. Blocks takes css.
    # I'll trust the warning and try to pass them to launch or just omit if I can't pass them to Blocks.
    # PROPOSED FIX:
    # 1. Remove theme/css from Blocks.
    # 2. Add them to launch() if supported, but `launch` usually takes `inbrowser`, `share`, etc.
    # If I can't pass `theme/css` to Blocks, I'll lose styling.
    # But I must satisfy the test warning or error if it's an error. (It was a failure in `test_build_ui_returns_blocks`... wait, that was `TypeError`?)
    # "FAILED tests/test_frontend.py::TestBuildUI::test_build_ui_returns_blocks - TypeError: Chatbot.__init__() got an unexpected keyword argument 'show_copy_button'"
    # The WARNING was just a warning. The ERROR was `show_copy_button`.
    # So I only STRICTLY need to fix `show_copy_button`.
    # I will leave theme/css in Blocks for now because removing them might break UI look, and warnings don't fail tests (unless treated as errors).
    # Wait, pytest summary said "2 warnings". Status was FAILED due to Error.
    # So I will just fix `show_copy_button`.

if __name__ == "__main__":
    main()

