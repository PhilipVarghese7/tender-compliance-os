# ⚖️ Tender Compliance OS — AI-Powered RFP & Vendor Intelligence Platform

> **Cymonic Technologies Hackathon Submission**  
> Turning complex procurement documents into decisive, evidence-backed decisions.

---

## The Problem

Public and enterprise procurement processes involve hundreds of pages of RFP (Request for Proposal) documents, each containing intricate compliance requirements spread across technical, legal, financial, and operational sections. Procurement teams must manually cross-reference every vendor proposal against these requirements — a process that is slow, error-prone, and highly susceptible to missed clauses. A single overlooked non-compliance or hidden contractual risk can cost organisations millions and expose them to significant legal liability.

---

## The Solution

**Tender Compliance OS** is a RAG-powered (Retrieval-Augmented Generation) AI platform that automates the entire RFP compliance lifecycle — from requirement extraction to vendor scoring, risk detection, and board-level recommendation — in a fraction of the time it would take a human team.

The system operates in two modes:
- **⚡ Fast Screening** — Extracts the 30–50 most critical requirements and scores all vendors in seconds, enabling rapid elimination of weak candidates.
- **🔬 Deep Analysis** — Performs full extraction of 800+ requirements, runs parallel RAG auditing against every vendor proposal, detects hidden risk clauses, and generates C-suite-ready analysis.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **UI Framework** | Streamlit |
| **LLM** | OpenAI GPT-4o-mini |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Vector Database** | ChromaDB (persistent, on-disk) |
| **RAG Framework** | LangChain (text splitters, prompt templates, chain composition) |
| **PDF Processing** | PyMuPDF (fitz) — fast, accurate text + page rendering |
| **Data Layer** | Pandas, JSON file-based hash cache |
| **Visualisation** | Plotly (bar, stacked bar, radar, heatmap charts) |
| **Concurrency** | Python `ThreadPoolExecutor` — parallel LLM batching |
| **Environment** | `python-dotenv` — secure API key management |

---

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- An OpenAI API key

### 1. Clone the Repository
```bash
git clone https://github.com/PhilipVarghese7/tender-compliance-os.git
cd tender-compliance-os
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-openai-key-here
```
> ⚠️ The API key is loaded securely via `python-dotenv` and is never exposed in the UI or logs.

### 5. Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## Key Features at a Glance

- **Dual-mode analysis** — Fast Screening (seconds) or Deep Analysis (minutes)
- **AI requirement extraction** — Identifies mandatory, implied, and eligibility obligations
- **Parallel RAG auditing** — Scores each vendor against every extracted requirement
- **PDF evidence viewer** — Click any finding to view the exact page in the vendor PDF
- **Risk clause detection** — Identifies liability caps, hidden fees, unfavourable IP terms
- **Multi-vendor comparison** — Score cards, radar charts, stacked bar, and risk heatmaps
- **Board-level recommendation** — Winner selection, negotiation tips, and key risk summary
- **Hash-cached processing** — Re-opening the same PDF costs zero API calls
- **Requirement management** — Add, edit, delete, and filter requirements before locking
- **CSV export** — Download extracted requirements for offline review

---

## Motivation & Alignment

This project was built to demonstrate that production-grade AI engineering is not just about connecting an LLM to a prompt — it is about designing for real-world conditions: large documents, inconsistent data, API latency, concurrency, and the need for verifiable, evidence-linked outputs.

Every architectural decision — from hash-cached embeddings to parallel batch auditing, from PyMuPDF's page rendering to ChromaDB's persistent vector store — was made to solve a genuine problem that procurement professionals face daily. The platform was tested against a real-world banking RFP scenario (State Bank of India) with four distinct vendor archetypes: a strong compliant vendor, a partially compliant vendor, a weak vendor, and one containing deliberately hidden risk clauses.

The skills developed and demonstrated here — RAG system design, LLM orchestration, vector search, structured AI output, and production UI engineering — are directly transferable to the kinds of intelligent document processing and enterprise AI challenges that define the current frontier of applied AI development.

---

## Project Structure

```
tender-compliance-os/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # API key (not committed)
├── .tcos_cache/            # Auto-generated hash cache + ChromaDB
│   └── chroma/             # Persistent vector store
└── README.md
```


*Built with LangChain · ChromaDB · GPT-4o-mini · PyMuPDF · Streamlit*