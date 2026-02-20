# Multimodal RAG System

An offline-capable Multimodal RAG system that ingests PDF, DOCX, PPTX, Images, and Audio, retrieves relevant content using hybrid search (Milvus + Tantivy), and provides grounded answers with citations.

## 👥 Team — Makeathon 2026

## Team Leader: Praveen Ram
Architecture Designed By: Praveen Ram

## Team Members:

Praveen Ram

Sachin Aadithya. V

Abishek Roshan KMS

Murugan

Sahana N

Deepa L

💻 Developed and tested on: Sachin’s laptop (RTX 3050 4GB VRAM)

## 🚀 Quick Start (Windows)

### 1. Prerequisite: Activate Environment
In PowerShell, run this to allow scripts and activate the `.venv`:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
*(Your prompt should show `(.venv)`)*

### 2. Prerequisite: Start Database
Since you are on Windows with Python 3.13, you **must use Docker**:
```powershell
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.4.5
```
*Wait 20-30 seconds for it to initialize.*

### 3. Setup
Install dependencies:
```powershell
pip install -r requirements.txt
```

Verify environment:
```powershell
python scripts/health_check.py
```

### 3. Running the System

**Backend (API)**
Start the FastAPI server:
```bash
python api/main.py
```
*Runs on http://localhost:8000*

**Frontend (UI)**
Start the Gradio interface:
```bash
python frontend/app.py
```
*Runs on http://localhost:7860*

## 📂 Project Structure

- `api/`: FastAPI backend (routers, dependencies).
- `frontend/`: Gradio UI and API client.
- `modules/`: Core logic (Ingestion, Indexing, Retrieval, LLM, Citation, Export).
- `scripts/`: Utility scripts (Health check, Seeding, Benchmarks).
- `data/`: Stored data (Uploads, Vector DB, SQLite, Index).
- `models/`: GGUF models and weights.

## 🛠 Utilities

- **Health Check**: `python scripts/health_check.py`
- **Benchmark**: `python scripts/benchmark.py`
- **Seed Data**: `python scripts/seed_demo_data.py`

## 🧪 Testing

Run all tests:
```bash
pytest
```

Run specific tests:
```bash
pytest tests/test_e2e_pipeline.py
```

## ❓ Troubleshooting

**Windows PowerShell: "running scripts is disabled on this system"**
If you see this error when activating `.venv`, run this command to temporarily allow scripts in the current terminal session:
```powershell
Set-ExecutionPolicy -ExecutionPolicy Process -Scope Process
```
*(Select 'Y' if prompted)*

Alternatively, you can skip activation and run Python directly using the full path:
```powershell
.venv\Scripts\python.exe api/main.py
```

**Frontend: "Backend not reachable"**
If the frontend says "Backend not reachable", ensure you have started the backend API in a separate terminal and it is fully running (you should see "Application startup complete").

**Frontend: "Port 7860 is in use"**
The frontend will now automatically try to find an open port if 7860 is busy. Check the terminal output for the actual URL (e.g., `http://127.0.0.1:7861`).

