# 🏆 Multimodal RAG System

### Offline · Cross-Modal · Hardware-Aware · Makeathon 2026

An **offline-capable Multimodal Retrieval-Augmented Generation (RAG)** system that ingests **PDF, DOCX, PPTX, Images, and Audio**, performs **hybrid search (Milvus + Tantivy)**, and generates **grounded answers with rich, cross-modal citations**.

Designed to run on a **4GB VRAM GPU (RTX 3050)**.
No cloud APIs. No external inference services.

---

## 👥 Team — Makeathon 2026

**Team Leader:** Praveen Ram<br>
**Architecture Designed By:** Praveen Ram

**Team Members:**

* Praveen Ram
* Sachin Aadithya. V
* Abishek Roshan KMS
* Murugan
* Sahana N
* Deepa L

💻 Developed and tested on Sachin’s laptop (RTX 3050 – 4GB VRAM)

---

# ✨ What Makes This Different?

Most “multimodal” systems convert everything to text and call it RAG.

This system uses **native embedding spaces per modality**:

* **Text & Audio** → BGE-M3 (1024-dim)
* **Images** → CLIP ViT-B/32 (512-dim)
* **Audio Transcription** → Whisper (word-level timestamps)
* **Hybrid Retrieval** → Milvus (Vector) + Tantivy (BM25F)
* **Cross-Modal Link Graph** → SQLite
* **Speculative Decoding** → Llama 3.2 (3B + 1B)

True multimodal retrieval — not OCR-only search.

---

# 🧠 Core Capabilities

* 📄 PDF / DOCX / PPTX ingestion
* 🖼 Image extraction + visual embeddings
* 🎧 Audio transcription with timestamps
* 🔎 Hybrid search (semantic + keyword)
* 🔗 Cross-modal linking (text ↔ image ↔ audio)
* 📌 Grounded answers with structured citations
* 📤 Export (DOCX / XLSX / PPTX / CSV)
* 🧠 Session memory
* 🔒 Fully offline execution
* ⚙️ 4GB VRAM-aware architecture

---

# 🏗 Architecture Overview

### 1️⃣ Ingestion Engine

* Structured chunking (500 tokens, 50 overlap)
* Rich metadata per chunk
* Page-level and timestamp-level traceability

### 2️⃣ Dual Index System

**Milvus (Vector Search)**

* `text_chunks` → 1024-dim
* `image_chunks` → 512-dim
* `audio_chunks` → 1024-dim

**Tantivy (BM25F Sparse Search)**

* Incremental indexing
* No full rebuild on new uploads

### 3️⃣ Hybrid Retrieval

* Parallel ANN + BM25
* Reciprocal Rank Fusion (RRF)
* Cross-encoder reranking
* Modality diversification
* Cross-modal enrichment

### 4️⃣ LLM Engine

* Llama 3.2 3B (main model)
* Llama 3.2 1B (draft model)
* Speculative decoding (80–120 tokens/sec)
* Citation-aware prompting

### 5️⃣ Citation Engine

* Page-level text citations
* Image bounding-box references
* Timestamped audio citations
* Cross-modal link display

---

# 🚀 Quick Start (Windows)

### 1️⃣ Activate Environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

### 2️⃣ Start Milvus (Docker Required)

```powershell
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.4.5
```

Wait ~20 seconds.

---

### 3️⃣ Install Dependencies

```powershell
pip install -r requirements.txt
```

Verify:

```powershell
python scripts/health_check.py
```

---

# ▶ Running the System

### Backend (FastAPI)

```powershell
python api/main.py
```

Runs on:
`http://localhost:8000`

---

### Frontend (Gradio UI)

```powershell
python frontend/app.py
```

Runs on:
`http://localhost:7860`

---

# 📂 Project Structure

```
api/         → FastAPI backend
frontend/    → Gradio UI
modules/     → Core logic (Ingestion, Retrieval, LLM, Citation, Export)
scripts/     → Utilities (Health, Benchmark, Seeding)
tests/       → Unit + integration tests
data/        → Uploads, DB, Vector Index
models/      → GGUF model files
```

---

# 🧪 Testing

Run all tests:

```powershell
pytest
```

Run end-to-end test:

```powershell
pytest tests/test_e2e_pipeline.py
```

---

# 🎬 Demo Highlights (Makeathon)

### 🔹 Text → Image Retrieval

> “Show me the Q3 revenue chart”

Retrieves the correct image via CLIP semantic similarity — even without matching OCR text.

---

### 🔹 Timestamp Navigation

> “What was discussed at 14 minutes?”

Returns:

* Audio segment
* Transcript
* Linked documents

---

### 🔹 Cross-Format Evidence

> “Find all evidence about budget approval”

Returns:

* PDF paragraph
* Signed image
* Audio confirmation
  All cross-linked.

---

# 🛠 Utilities

Health Check:

```powershell
python scripts/health_check.py
```

Benchmark:

```powershell
python scripts/benchmark.py
```

Seed Demo Data:

```powershell
python scripts/seed_demo_data.py
```

---

# 🔒 Fully Offline

* No OpenAI API
* No cloud inference
* No external embedding services
* No internet required during runtime

Designed for constrained GPU environments.

---

# 🏁 Conclusion

A production-ready, hardware-aware, fully offline multimodal RAG system with:

* Native modality embeddings
* Hybrid retrieval
* Cross-modal linking
* Transparent citations
* Export-ready structured outputs

Built for Makeathon 2026.
Engineered for real-world constraints.




