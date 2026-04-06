import os, json, hashlib, re, base64, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

import streamlit as st
import pandas as pd
import fitz
import plotly.graph_objects as go

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="Tender Compliance OS",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# GLOBAL STYLES  — cached so they don't re-inject every rerun
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _get_styles() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Syne:wght@400;600;700;800&display=swap');

:root {
  --bg:      #0a0a0a; --s1: #111111; --s2: #181818; --s3: #222222; --s4: #2a2a2a;
  --border:  #2a2a2a; --border2: #383838;
  --amber:   #f59e0b; --amber2: #fbbf24; --amber3: #fde68a;
  --green:   #10b981; --green2: #34d399;
  --red:     #ef4444; --red2: #f87171;
  --blue:    #3b82f6; --purple: #8b5cf6; --cyan: #06b6d4;
  --text:    #e8e6e1; --text2: #b8b5ae; --muted: #6b7280;
  --fh: 'Syne', sans-serif; --fc: 'IBM Plex Mono', monospace;
  --radius-sm: 4px; --radius: 8px; --radius-lg: 12px;
  --shadow-sm: 0 1px 3px rgba(0,0,0,.4); --shadow: 0 4px 16px rgba(0,0,0,.5);
  --shadow-lg: 0 8px 32px rgba(0,0,0,.6);
  --trans: all .18s cubic-bezier(.4,0,.2,1);
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important; color: var(--text) !important;
  font-family: var(--fh) !important;
}
[data-testid="stSidebar"] { background: var(--s1) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] hr { border-color: var(--border) !important; margin: 1rem 0 !important; }
.stButton > button {
  background: linear-gradient(135deg, var(--amber), #e08d00) !important;
  color: #000 !important; border: none !important; border-radius: var(--radius-sm) !important;
  font-family: var(--fc) !important; font-weight: 700 !important; font-size: .78rem !important;
  letter-spacing: .05em !important; text-transform: uppercase !important;
  padding: .55rem 1.5rem !important; transition: var(--trans) !important;
  box-shadow: var(--shadow-sm) !important; white-space: nowrap !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, var(--amber2), var(--amber)) !important;
  transform: translateY(-2px) !important; box-shadow: 0 4px 14px rgba(245,158,11,.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; box-shadow: var(--shadow-sm) !important; }
.stTextInput input, .stTextArea textarea {
  background: var(--s2) !important; border: 1px solid var(--border2) !important;
  color: var(--text) !important; border-radius: var(--radius-sm) !important;
  font-family: var(--fc) !important; font-size: .84rem !important; transition: border-color .15s !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--amber) !important; box-shadow: 0 0 0 2px rgba(245,158,11,.15) !important;
}
.stTabs [data-baseweb="tab-list"] {
  background: var(--s1) !important; border-bottom: 1px solid var(--border) !important;
  gap: 2px !important; padding: 0 .25rem !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; color: var(--muted) !important;
  font-family: var(--fc) !important; font-size: .75rem !important;
  letter-spacing: .07em !important; text-transform: uppercase !important;
  padding: .55rem 1.1rem !important; border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
  transition: color .15s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text2) !important; }
.stTabs [aria-selected="true"] {
  color: var(--amber) !important; border-bottom: 2px solid var(--amber) !important;
  background: rgba(245,158,11,.05) !important;
}
.stProgress > div > div {
  background: linear-gradient(90deg, var(--amber), var(--amber2)) !important; border-radius: 999px !important;
}
.stProgress > div { background: var(--s3) !important; border-radius: 999px !important; }
[data-testid="stExpander"] {
  background: var(--s2) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important; transition: border-color .15s !important; margin-bottom: .5rem !important;
}
[data-testid="stExpander"]:hover { border-color: var(--border2) !important; }
[data-testid="stExpander"] summary { padding: .7rem 1rem !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--s1); }
::-webkit-scrollbar-thumb { background: var(--amber); border-radius: 999px; }
div[data-testid="stAlert"] { border-radius: var(--radius) !important; }
.hdr {
  font-family: var(--fc); font-size: .58rem; text-transform: uppercase;
  letter-spacing: .2em; color: var(--amber); margin-bottom: .65rem;
  padding-bottom: .4rem; border-bottom: 1px solid var(--border);
}
.metric {
  background: var(--s2); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 1rem 1.1rem .9rem; text-align: center; transition: border-color .15s, transform .15s;
}
.metric:hover { border-color: var(--border2); transform: translateY(-1px); }
.metric .v { font-family: var(--fc); font-size: 1.8rem; font-weight: 700; line-height: 1.1; }
.metric .l { font-family: var(--fc); font-size: .58rem; text-transform: uppercase; letter-spacing: .14em; color: var(--muted); margin-top: .3rem; }
.pill { display: inline-block; padding: 2px 10px; border-radius: var(--radius-sm); font-family: var(--fc); font-size: .62rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; line-height: 1.6; }
.p-green  { background: rgba(16,185,129,.12); color: var(--green);  border: 1px solid rgba(16,185,129,.2);  }
.p-amber  { background: rgba(245,158,11,.12); color: var(--amber);  border: 1px solid rgba(245,158,11,.2);  }
.p-red    { background: rgba(239,68,68,.12);  color: var(--red);    border: 1px solid rgba(239,68,68,.2);   }
.p-blue   { background: rgba(59,130,246,.12); color: var(--blue);   border: 1px solid rgba(59,130,246,.2);  }
.p-gray   { background: rgba(107,114,128,.1); color: var(--muted);  border: 1px solid rgba(107,114,128,.2); }
.p-purple { background: rgba(139,92,246,.12); color: var(--purple); border: 1px solid rgba(139,92,246,.2);  }
.p-cyan   { background: rgba(6,182,212,.12);  color: var(--cyan);   border: 1px solid rgba(6,182,212,.2);   }
.mode-badge { display: inline-block; padding: 4px 14px; border-radius: 999px; font-family: var(--fc); font-size: .6rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
.mode-fast { background: rgba(6,182,212,.12);  color: var(--cyan);   border: 1px solid rgba(6,182,212,.25);  }
.mode-deep { background: rgba(139,92,246,.12); color: var(--purple); border: 1px solid rgba(139,92,246,.25); }
.risk-card { background: rgba(239,68,68,.04); border: 1px solid rgba(239,68,68,.18); border-left: 3px solid var(--red); border-radius: var(--radius); padding: .8rem 1rem; margin-bottom: .45rem; transition: border-color .15s; }
.risk-card:hover { border-color: rgba(239,68,68,.35); }
.winner-banner { background: linear-gradient(135deg,rgba(245,158,11,.07),rgba(16,185,129,.04)); border: 1px solid rgba(245,158,11,.4); border-radius: var(--radius-lg); padding: 1.6rem 2rem; margin-bottom: 1.6rem; box-shadow: 0 2px 24px rgba(245,158,11,.08); }
.screen-card { background: var(--s2); border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem 1.3rem; margin-bottom: .55rem; transition: var(--trans); }
.screen-card:hover { border-color: var(--border2); transform: translateY(-1px); box-shadow: var(--shadow-sm); }
.screen-pass { border-left: 4px solid var(--green); }
.screen-fail { border-left: 4px solid var(--red); }
.vendor-card { background: var(--s2); border: 1px solid var(--border); border-radius: var(--radius); padding: .9rem 1.2rem; margin-bottom: .45rem; display: flex; align-items: center; gap: 1rem; transition: var(--trans); }
.vendor-card:hover { border-color: var(--border2); }
.vendor-card-name { font-weight: 700; font-size: .92rem; font-family: var(--fh); color: var(--text); }
.vendor-card-status-done { font-family: var(--fc); font-size: .63rem; color: var(--green);  margin-top: .15rem; }
.vendor-card-status-proc { font-family: var(--fc); font-size: .63rem; color: var(--amber); margin-top: .15rem; }
.pdf-viewer-frame { text-align: center; background: var(--s2); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.2rem; margin-top: .6rem; }
.pdf-viewer-frame img { max-width: 100%; border-radius: var(--radius-sm); box-shadow: var(--shadow-lg); }
.pdf-viewer-caption { font-family: var(--fc); font-size: .62rem; color: var(--muted); margin-top: .6rem; letter-spacing: .06em; }
.rfp-quote-box { background: rgba(245,158,11,.06); border-left: 3px solid var(--amber); border-radius: 0 var(--radius-sm) var(--radius-sm) 0; padding: .6rem .9rem; font-family: var(--fc); font-size: .74rem; line-height: 1.6; color: var(--text2); }
.evidence-box { background: rgba(16,185,129,.05); border-left: 3px solid var(--green); border-radius: 0 var(--radius-sm) var(--radius-sm) 0; padding: .6rem .9rem; font-family: var(--fc); font-size: .74rem; line-height: 1.6; color: var(--text2); }
button[data-testid="baseButton-secondary"][kind="secondary"], .btn-evidence > button, .btn-source > button {
  background: rgba(6,182,212,.12) !important; color: var(--cyan) !important;
  border: 1px solid rgba(6,182,212,.35) !important; border-radius: var(--radius-sm) !important;
  font-family: var(--fc) !important; font-size: .72rem !important; font-weight: 600 !important;
  letter-spacing: .04em !important; padding: .4rem 1rem !important; transition: var(--trans) !important;
  box-shadow: none !important; text-transform: none !important;
}
button[data-testid="baseButton-secondary"][kind="secondary"]:hover, .btn-evidence > button:hover, .btn-source > button:hover {
  background: rgba(6,182,212,.22) !important; border-color: rgba(6,182,212,.6) !important;
  transform: translateY(-1px) !important; box-shadow: 0 2px 10px rgba(6,182,212,.2) !important;
}
.gap-xs { margin-bottom: .4rem; } .gap-sm { margin-bottom: .75rem; }
.gap-md { margin-bottom: 1.2rem; } .gap-lg { margin-bottom: 2rem; }
.page-title { font-size: 1.55rem; font-weight: 800; font-family: var(--fh); margin-bottom: .2rem; }
.page-sub   { font-family: var(--fc); font-size: .72rem; color: var(--muted); margin-bottom: 1.6rem; }
.note-muted { font-family: var(--fc); font-size: .68rem; color: var(--muted); margin-top: .35rem; display: flex; align-items: center; gap: .35rem; }
.reasoning  { color: var(--muted); font-family: var(--fc); font-size: .7rem; margin-top: .45rem; line-height: 1.55; }
.stSelectbox > div > div { background: var(--s2) !important; border: 1px solid var(--border2) !important; color: var(--text) !important; border-radius: var(--radius-sm) !important; }
.stNumberInput input { background: var(--s2) !important; border: 1px solid var(--border2) !important; color: var(--text) !important; font-family: var(--fc) !important; border-radius: var(--radius-sm) !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--amber) !important; border-color: var(--amber) !important; }
.stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] { color: var(--amber) !important; font-family: var(--fc) !important; font-size: .7rem !important; }
.stRadio > div { gap: .2rem !important; }
.stRadio label { font-family: var(--fc) !important; font-size: .74rem !important; padding: .4rem .6rem !important; border-radius: var(--radius-sm) !important; transition: background .12s !important; }
.stRadio label:hover { background: rgba(245,158,11,.06) !important; }
.stCheckbox label span { font-family: var(--fc) !important; font-size: .75rem !important; }
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
.js-plotly-plot .plotly { border-radius: var(--radius) !important; }
</style>
"""

st.markdown(_get_styles(), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════
CACHE_DIR        = Path(".tcos_cache")
CACHE_DIR.mkdir(exist_ok=True)
CHROMA_DIR       = str(CACHE_DIR / "chroma")
CHUNK_SIZE       = 1200
CHUNK_OVERLAP    = 150
BATCH_SIZE       = 12
EXTRACTION_BATCH = 4000
RETRIEVAL_K      = 6
PARALLEL_WORKERS = 4
EMBED_MODEL      = "text-embedding-3-small"
LLM_MODEL        = "gpt-4o-mini"
FAST_REQ_TARGET  = 50
FAST_THRESHOLD   = 60
CATEGORIES       = ["Technical","Legal","Financial","Compliance","Scope","Deadlines","Evaluation","Documents"]

# ═══════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════
def _init():
    defs = dict(
        view="RFP Setup",
        rfp_hash=None, rfp_text="", rfp_pages=[], rfp_bytes=None, rfp_chunks=None,
        rfp_vs_built=False,
        requirements=[], requirements_locked=False,
        critical_requirements=[],
        vendors={},
        vendor_screening={},
        vendors_for_deep=None,
        analysis_mode=None,
        board_cache=None,
    )
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

if not OPENAI_API_KEY:
    st.error("**OPENAI_API_KEY not found.**\n\nCreate a `.env` file:\n```\nOPENAI_API_KEY=sk-your-key-here\n```\nThen restart.")
    st.stop()

# ═══════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════
def sha(data):
    if isinstance(data, str): data = data.encode()
    return hashlib.sha256(data).hexdigest()[:16]

def cache_get(key):
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try: return json.loads(p.read_text())
        except: return None
    return None

def cache_set(key, val):
    try: (CACHE_DIR / f"{key}.json").write_text(json.dumps(val, default=str))
    except: pass

def parse_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try: return json.loads(text)
    except: pass
    for pat in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        m = re.search(pat, text)
        if m:
            try: return json.loads(m.group())
            except: pass
    return None

def pill(label, cls):
    return f"<span class='pill {cls}'>{label}</span>"

STATUS_CLS = {"Met":"p-green","Partial":"p-amber","Missing":"p-red","Non-compliant":"p-red"}
RISK_CLS   = {"High":"p-red","Medium":"p-amber","Low":"p-green","Critical":"p-red"}

# ═══════════════════════════════════════════════════════════
# LLM / EMBEDDING
# — st.cache_resource: these are heavy singleton objects.
#   They are built once per server session and reused across
#   all reruns, eliminating repeated initialisation overhead.
# ═══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY, chunk_size=500)

@st.cache_resource(show_spinner=False)
def get_llm(max_tokens=2048):
    return ChatOpenAI(model=LLM_MODEL, temperature=0.0, openai_api_key=OPENAI_API_KEY, max_tokens=max_tokens)

def build_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n","\n",". ",", "," ",""], length_function=len)

# ═══════════════════════════════════════════════════════════
# PDF PROCESSING
# — load_pdf_bytes_cached: st.cache_data keyed on raw bytes.
#   Same file uploaded again → zero re-parsing, instant return.
# — render_page_b64_cached: THE biggest performance fix.
#   Rendering a PDF page is the most expensive per-click op.
#   Cached by (bytes_hash, page_num) so every page is rendered
#   exactly once per session. Subsequent clicks are <50ms.
#   Resolution lowered from 1.6× to 1.3× — visually identical
#   in a browser but ~35% faster to render and encode.
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_pdf_bytes_cached(file_bytes: bytes, source_name: str = "document"):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    lc_docs = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            lc_docs.append(Document(
                page_content=text,
                metadata={"source": source_name, "page": i + 1}
            ))
    doc.close()
    return lc_docs

# Keep original name as thin wrapper so all call-sites stay unchanged
def load_pdf_bytes(file_bytes, source_name="document"):
    return load_pdf_bytes_cached(file_bytes, source_name)

@st.cache_data(show_spinner=False)
def render_page_b64_cached(file_bytes: bytes, page_num: int) -> str:
    """Render a single PDF page to base64 PNG. Cached by (bytes, page)."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    idx = max(0, min(page_num - 1, len(doc) - 1))
    pix = doc[idx].get_pixmap(matrix=fitz.Matrix(1.3, 1.3))   # 1.3× = fast + crisp
    b64 = base64.b64encode(pix.tobytes("png")).decode()
    doc.close()
    return b64

# Backwards-compatible wrapper — replaces render_page_b64 everywhere
def render_page_b64(file_bytes, page_num):
    return render_page_b64_cached(file_bytes, page_num)

def split_docs(docs):
    return build_splitter().split_documents(docs)

# ═══════════════════════════════════════════════════════════
# RFP SOURCE VIEWER
# — Spinner added so the user immediately sees "Fetching…"
#   instead of a frozen UI during the first uncached render.
# ═══════════════════════════════════════════════════════════
def render_rfp_source_viewer(req, viewer_key):
    pg = req.get("page_reference")
    rfp_bytes = st.session_state.get("rfp_bytes")
    if pg and rfp_bytes:
        toggle_key = f"show_{viewer_key}"
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = False
        st.markdown("<div class='btn-source'>", unsafe_allow_html=True)
        if st.button(f"📄 View Source — Page {pg}", key=viewer_key):
            st.session_state[toggle_key] = not st.session_state[toggle_key]
        st.markdown("</div>", unsafe_allow_html=True)
        if st.session_state[toggle_key]:
            st.markdown(
                f"<div class='hdr' style='margin-top:.6rem'>📄 RFP Source — Page {pg}</div>",
                unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-family:var(--fc);font-size:.7rem;color:var(--amber);margin-bottom:.5rem'>"
                f"{req.get('requirement_summary','')[:100]}</div>",
                unsafe_allow_html=True)
            if req.get("exact_quote"):
                st.markdown(
                    f"<div class='rfp-quote-box gap-sm'>{req['exact_quote'][:240]}</div>",
                    unsafe_allow_html=True)
            try:
                # Spinner gives instant visual feedback; cached call returns in <50ms after first load
                with st.spinner(f"Loading RFP page {pg}…"):
                    b64 = render_page_b64_cached(rfp_bytes, pg)
                st.markdown(
                    f"<div class='pdf-viewer-frame'>"
                    f"<img src='data:image/png;base64,{b64}'/>"
                    f"<div class='pdf-viewer-caption'>RFP — Page {pg}</div>"
                    f"</div>",
                    unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"Could not render page {pg}: {exc}")
    else:
        st.markdown("<div class='note-muted'>⚠ Source page not available</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# VECTORSTORE
# — Vectorstore objects are pinned to session_state after
#   first construction so Chroma is never rebuilt mid-session.
# ═══════════════════════════════════════════════════════════
def get_or_build_vectorstore(chunks, collection_name):
    # Check session cache first — avoids re-opening ChromaDB on every rerun
    vs_key = f"_vs_{collection_name}"
    if vs_key in st.session_state and st.session_state[vs_key] is not None:
        return st.session_state[vs_key], False

    emb = get_embeddings()
    try:
        vs = Chroma(collection_name=collection_name, embedding_function=emb, persist_directory=CHROMA_DIR)
        if vs._collection.count() > 0:
            st.session_state[vs_key] = vs
            return vs, False
    except:
        pass
    vs = Chroma.from_documents(chunks, emb, collection_name=collection_name, persist_directory=CHROMA_DIR)
    st.session_state[vs_key] = vs
    return vs, True

# ═══════════════════════════════════════════════════════════
# DEEP EXTRACTION
# ═══════════════════════════════════════════════════════════
EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", f"""You are a senior procurement analyst.
Extract ALL compliance requirements from the RFP text below.
Include: mandatory language (shall/must/required/mandatory) AND intent-based obligations.
Return ONLY a valid JSON array — no prose, no markdown fences.
Each object MUST have:
{{{{
  "requirement_summary": "concise 1-sentence summary",
  "exact_quote": "verbatim text ≤180 chars",
  "category": "one of: {" | ".join(CATEGORIES)}",
  "page_reference": integer or null,
  "risk_level": "High" or "Medium" or "Low"
}}}}"""),
    ("human", "RFP SECTIONS:\n\n{context}"),
])

def _call_extract_batch(args):
    i, batch, llm = args
    context_text = "\n\n---\n\n".join(
        f"[Page {c.metadata.get('page','?')}]\n{c.page_content}" for c in batch)
    try:
        raw = (EXTRACT_PROMPT | llm).invoke({"context": context_text})
        if not raw.content or not raw.content.strip():
            return i, [], f"Batch {i+1}: empty response"
        reqs = parse_json(raw.content)
        if isinstance(reqs, list): return i, reqs, None
        return i, [], f"Batch {i+1}: unexpected format"
    except Exception as exc:
        return i, [], f"Batch {i+1}: {exc}"

def extract_requirements_from_rfp(rfp_hash, chunks, progress_cb=None):
    cache_key = f"extract_deep_{rfp_hash}"
    cached = cache_get(cache_key)
    if cached:
        if progress_cb: progress_cb(1.0)
        return cached
    llm = get_llm()
    batches, current, cur_len = [], [], 0
    for chunk in chunks:
        cl = len(chunk.page_content)
        if cur_len + cl > EXTRACTION_BATCH and current:
            batches.append(current); current, cur_len = [chunk], cl
        else:
            current.append(chunk); cur_len += cl
    if current: batches.append(current)
    all_reqs, seen, req_id = [], set(), 0
    batch_errors, completed = [], 0
    args_list = [(i, batch, llm) for i, batch in enumerate(batches)]
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        future_map = {ex.submit(_call_extract_batch, args): args[0] for args in args_list}
        ordered_results = [None] * len(batches)
        for future in as_completed(future_map):
            i, reqs, err = future.result()
            ordered_results[i] = (reqs, err)
            completed += 1
            if progress_cb: progress_cb(completed / len(batches))
    for reqs, err in ordered_results:
        if err: batch_errors.append(err); continue
        for r in (reqs or []):
            key = r.get("exact_quote","")[:60]
            if key and key not in seen:
                seen.add(key); r["id"] = req_id; req_id += 1; all_reqs.append(r)
    if batch_errors:
        st.warning("⚠ Some batches had issues:\n" + "\n".join(f"• {e}" for e in batch_errors[:5]))
    if not all_reqs:
        detail = "\n".join(batch_errors[:3]) if batch_errors else "No data from LLM."
        raise RuntimeError(f"No requirements extracted.\nDetails: {detail}")
    cache_set(cache_key, all_reqs)
    return all_reqs

# ═══════════════════════════════════════════════════════════
# FAST SCREENING EXTRACTION
# ═══════════════════════════════════════════════════════════
CRITICAL_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", f"""You are a procurement analyst performing a rapid screening pass.

Extract only the most critical requirements (maximum {FAST_REQ_TARGET}).
Prioritize eligibility criteria, qualification requirements, disqualification conditions,
and mandatory clauses. Eligibility criteria are CRITICAL and must NEVER be skipped even
if not written using 'shall' or 'must'. Extract each eligibility rule as a separate requirement.

Priority order:
1. Eligibility criteria — VERY IMPORTANT, NEVER MISS ANY
2. Qualification requirements (minimum turnover, years of experience, certifications, licences)
3. Disqualification conditions (blacklisting, legal proceedings, conflict of interest)
4. Mandatory clauses (shall/must/required/mandatory)
5. High-risk financial or IP obligations

Return ONLY a valid JSON array — no prose, no markdown fences.
Each object:
{{{{
  "requirement_summary": "concise 1-sentence summary",
  "exact_quote": "verbatim text ≤150 chars",
  "page_reference": integer or null,
  "risk_level": "High" or "Medium" or "Low"
}}}}
Return AT MOST {FAST_REQ_TARGET} items. Quality over quantity."""),
    ("human", "RFP TEXT (representative sample):\n\n{context}"),
])

def extract_critical_requirements(rfp_hash, chunks, progress_cb=None):
    cache_key = f"extract_fast_{rfp_hash}"
    cached = cache_get(cache_key)
    if cached:
        if progress_cb: progress_cb(1.0)
        return cached
    llm = get_llm(max_tokens=3000)
    ELIGIBILITY_KW = re.compile(
        r'\b(eligib|qualif|turnover|experience|certif|licen|blacklist|disqualif|'
        r'conflict.of.interest|minimum.requirement|criteria|mandatory.requirement)\b', re.I)
    MANDATORY_KW = re.compile(r'\b(shall|must|required|mandatory|obligat|comply)\b', re.I)
    scored = []
    for c in chunks:
        score = (len(ELIGIBILITY_KW.findall(c.page_content)) * 3 +
                 len(MANDATORY_KW.findall(c.page_content)))
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    selected, total_len = [], 0
    for score, chunk in scored:
        if total_len + len(chunk.page_content) > 10000: break
        selected.append(chunk); total_len += len(chunk.page_content)
    if not selected: selected = chunks[:8]
    context_text = "\n\n---\n\n".join(
        f"[Page {c.metadata.get('page','?')}]\n{c.page_content}" for c in selected)
    if progress_cb: progress_cb(0.3)
    try:
        raw = (CRITICAL_EXTRACT_PROMPT | llm).invoke({"context": context_text})
        reqs = parse_json(raw.content)
    except Exception as exc:
        raise RuntimeError(f"Fast extraction failed: {exc}")
    if progress_cb: progress_cb(0.8)
    if not isinstance(reqs, list) or not reqs:
        raise RuntimeError("Fast extraction returned no requirements.")
    for i, r in enumerate(reqs[:FAST_REQ_TARGET]):
        r["id"] = i; r.setdefault("category", "Compliance")
    result = reqs[:FAST_REQ_TARGET]
    cache_set(cache_key, result)
    if progress_cb: progress_cb(1.0)
    return result

def display_and_export_requirements(requirements, table_title="Extracted Requirements"):
    if not requirements: st.warning("No requirements to display."); return
    df = pd.DataFrame(requirements)
    st.subheader(table_title)
    st.dataframe(df, use_container_width=True)
    st.download_button("📥 Export as CSV", df.to_csv(index=False).encode(), "requirements.csv", "text/csv")

# ═══════════════════════════════════════════════════════════
# REQUIREMENT REVIEW PANEL
# ═══════════════════════════════════════════════════════════
def render_requirement_review(reqs, on_lock_callback, reqs_key="requirements"):
    rc = {"High":0,"Medium":0,"Low":0}
    for r in reqs: rc[r.get("risk_level","Low")] = rc.get(r.get("risk_level","Low"),0)+1

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"<div class='metric'><div class='v'>{len(reqs)}</div><div class='l'>Total</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric'><div class='v' style='color:var(--red)'>{rc['High']}</div><div class='l'>High Risk</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric'><div class='v' style='color:var(--amber)'>{rc['Medium']}</div><div class='l'>Medium</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric'><div class='v' style='color:var(--green)'>{rc['Low']}</div><div class='l'>Low Risk</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='gap-md'></div>", unsafe_allow_html=True)

    display_and_export_requirements(reqs)

    with st.expander("➕ Add Custom Requirement"):
        nc1,nc2 = st.columns(2)
        with nc1:
            ns = st.text_input("Summary", key=f"{reqs_key}_ns")
            nq = st.text_area("Exact Quote / Clause", key=f"{reqs_key}_nq", height=80)
        with nc2:
            ncat  = st.selectbox("Category", CATEGORIES, key=f"{reqs_key}_ncat")
            nrisk = st.selectbox("Risk Level", ["High","Medium","Low"], key=f"{reqs_key}_nrisk")
            npg   = st.number_input("Page", min_value=1, value=1, key=f"{reqs_key}_npg")
        if st.button("Add Requirement", key=f"{reqs_key}_add"):
            if ns:
                new_id = max((r["id"] for r in reqs), default=-1)+1
                reqs.append({"id":new_id,"requirement_summary":ns,"exact_quote":nq,
                             "category":ncat,"page_reference":npg,"risk_level":nrisk})
                st.session_state[reqs_key] = reqs
                st.rerun()

    cats_available = sorted(set(r.get("category","?") for r in reqs))
    fcat = st.selectbox("Filter Category", ["All"]+cats_available, key=f"{reqs_key}_fcat")
    to_delete = []

    for req in reqs:
        if fcat != "All" and req.get("category") != fcat: continue
        rl_cls = RISK_CLS.get(req.get("risk_level","Low"),"p-gray")
        with st.expander(f"[{req.get('category','?')}] {req.get('requirement_summary','')[:75]}"):
            cc1,cc2 = st.columns([3,1])
            with cc1:
                new_s = st.text_input("Summary", value=req["requirement_summary"], key=f"{reqs_key}_s_{req['id']}")
                new_q = st.text_area("Exact Quote", value=req.get("exact_quote",""), key=f"{reqs_key}_q_{req['id']}", height=65)
            with cc2:
                new_c = st.selectbox("Category", CATEGORIES,
                    index=CATEGORIES.index(req.get("category","Technical")) if req.get("category") in CATEGORIES else 0,
                    key=f"{reqs_key}_c_{req['id']}")
                new_r = st.selectbox("Risk", ["High","Medium","Low"],
                    index=["High","Medium","Low"].index(req.get("risk_level","Low")),
                    key=f"{reqs_key}_r_{req['id']}")
                new_p = st.number_input("Page", value=int(req.get("page_reference") or 1), min_value=1,
                    key=f"{reqs_key}_p_{req['id']}")
            st.markdown("<div class='gap-xs'></div>", unsafe_allow_html=True)
            b1,b2,b3 = st.columns([1,1,2])
            with b1:
                if st.button("💾 Save", key=f"{reqs_key}_sv_{req['id']}"):
                    req.update({"requirement_summary":new_s,"exact_quote":new_q,
                                "category":new_c,"risk_level":new_r,"page_reference":new_p})
                    st.rerun()
            with b2:
                if st.button("🗑 Delete", key=f"{reqs_key}_dl_{req['id']}"):
                    to_delete.append(req["id"])
            with b3:
                render_rfp_source_viewer(req, viewer_key=f"{reqs_key}_rfpsrc_{req['id']}")

    if to_delete:
        st.session_state[reqs_key] = [r for r in reqs if r["id"] not in to_delete]
        st.rerun()

    st.markdown("<div class='gap-md'></div>", unsafe_allow_html=True)
    if st.button("🔒 Lock Requirements → Proceed to Vendors", use_container_width=True, key=f"{reqs_key}_lock"):
        on_lock_callback()

# ═══════════════════════════════════════════════════════════
# VENDOR AUDIT
# ═══════════════════════════════════════════════════════════
AUDIT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a compliance auditor.
For EACH requirement, judge compliance based ONLY on the vendor evidence.
"Met" = clear explicit evidence; "Partial" = vague/implied; "Missing" = not addressed; "Non-compliant" = contradicts.
Return ONLY a JSON array:
{{
  "requirement_id": <integer>,
  "status": "Met"|"Partial"|"Missing"|"Non-compliant",
  "confidence": 0-100,
  "matched_claim": "verbatim evidence ≤150 chars or null",
  "reasoning": "1-sentence justification",
  "page_reference": integer or null
}}"""),
    ("human", "REQUIREMENTS:\n{requirements}\n\nVENDOR EVIDENCE:\n{evidence}"),
])

def _call_audit_batch(args):
    bi, batch, vendor_vs, llm = args
    combined_query = " ".join(
        r.get("requirement_summary","") + " " + r.get("exact_quote","")[:60] for r in batch)[:800]
    try:
        docs = vendor_vs.similarity_search(combined_query, k=RETRIEVAL_K)
        evidence = "\n\n---\n\n".join(
            f"[Page {d.metadata.get('page','?')}]\n{d.page_content}" for d in docs) or "No evidence found."
    except:
        evidence = "No evidence found."
    reqs_payload = json.dumps([
        {"id":r["id"],"summary":r["requirement_summary"],"quote":r.get("exact_quote","")[:100],"category":r.get("category","?")}
        for r in batch])
    try:
        raw = (AUDIT_PROMPT | llm).invoke({"requirements": reqs_payload, "evidence": evidence})
        res_list = parse_json(raw.content)
        if isinstance(res_list, list): return bi, res_list, None
        return bi, [], f"Batch {bi+1}: unexpected format"
    except Exception as exc:
        return bi, [], f"Batch {bi+1}: {exc}"

def audit_vendor_rag(requirements, vendor_vs, vendor_hash, req_hash, progress_cb=None):
    cache_key = f"audit_{vendor_hash}_{req_hash}"
    cached = cache_get(cache_key)
    if cached:
        if progress_cb: progress_cb(1.0)
        return {int(k):v for k,v in cached.items()}
    llm = get_llm()
    results, batches, completed = {}, [requirements[i:i+BATCH_SIZE] for i in range(0, len(requirements), BATCH_SIZE)], 0
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        future_map = {ex.submit(_call_audit_batch, (bi, batch, vendor_vs, llm)): bi for bi, batch in enumerate(batches)}
        for future in as_completed(future_map):
            bi, res_list, err = future.result()
            if err: st.warning(f"Audit {err}")
            for item in res_list:
                rid = item.get("requirement_id")
                if rid is not None: results[rid] = item
            completed += 1
            if progress_cb: progress_cb(completed / len(batches))
    for req in requirements:
        if req["id"] not in results:
            results[req["id"]] = {"requirement_id":req["id"],"status":"Missing","confidence":0,
                                   "matched_claim":None,"reasoning":"No evidence found.","page_reference":None}
    cache_set(cache_key, {str(k):v for k,v in results.items()})
    return results

def compute_score(requirements, audit):
    total = len(requirements)
    met  = sum(1 for r in requirements if audit.get(r["id"],{}).get("status")=="Met")
    part = sum(1 for r in requirements if audit.get(r["id"],{}).get("status")=="Partial")
    return round((met + 0.5*part) / max(total,1) * 100)

# ═══════════════════════════════════════════════════════════
# RISK DETECTION
# ═══════════════════════════════════════════════════════════
RISK_QUERIES = [
    "subject to change price modification unilateral","limited liability cap damages",
    "additional fees surcharge cost increase","termination convenience penalty",
    "intellectual property ownership rights","warranty disclaimer limitation",
    "force majeure delay excuse","indemnification hold harmless",
]

RISK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a legal risk analyst scanning vendor proposals for risky clauses.
Detect: liability caps, unilateral change rights, hidden fees, vague obligations,
unfavourable IP terms, weak warranty, force majeure overreach, indemnification gaps.
Return ONLY a JSON array (empty [] if no risks):
{{
  "risk_type": "short category",
  "risk_level": "Critical"|"High"|"Medium"|"Low",
  "exact_clause": "verbatim excerpt ≤200 chars",
  "explanation": "1-sentence risk impact"
}}"""),
    ("human", "VENDOR DOCUMENT SECTIONS:\n\n{context}"),
])

def detect_risks_rag(vendor_vs, vendor_hash):
    cache_key = f"risks_{vendor_hash}"
    cached = cache_get(cache_key)
    if cached is not None: return cached
    llm = get_llm()
    all_docs, seen_content = [], set()
    for q in RISK_QUERIES:
        try:
            for d in vendor_vs.similarity_search(q, k=3):
                c = d.page_content[:80]
                if c not in seen_content: seen_content.add(c); all_docs.append(d)
        except: pass
    if not all_docs: cache_set(cache_key, []); return []
    context = "\n\n---\n\n".join(
        f"[Page {d.metadata.get('page','?')}]\n{d.page_content}" for d in all_docs[:8])
    all_risks, seen = [], set()
    try:
        raw = (RISK_PROMPT | llm).invoke({"context": context})
        risks = parse_json(raw.content)
        if isinstance(risks, list):
            for r in risks:
                key = r.get("exact_clause","")[:50]
                if key and key not in seen: seen.add(key); all_risks.append(r)
    except Exception as exc:
        st.warning(f"Risk detection error: {exc}")
    cache_set(cache_key, all_risks)
    return all_risks

# ═══════════════════════════════════════════════════════════
# BOARD ANALYSIS
# ═══════════════════════════════════════════════════════════
BOARD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a chief procurement officer writing a board-level brief. Be decisive and professional.
Return ONLY valid JSON:
{{
  "winner": "vendor name",
  "winner_score": integer,
  "winner_rationale": "2-3 sentences",
  "losers": [{{"name":"...","score":int,"failure_reason":"1-2 sentences"}}],
  "executive_summary": "2-sentence C-suite summary",
  "negotiation_tips": ["tip1","tip2","tip3"],
  "key_risks": ["risk1","risk2","risk3"]
}}"""),
    ("human", "VENDOR COMPLIANCE DATA:\n{data}"),
])

def run_board_analysis(summaries):
    cache_key = f"board_{sha(json.dumps(summaries, sort_keys=True))}"
    cached = cache_get(cache_key)
    if cached: return cached
    llm = get_llm()
    try:
        raw = (BOARD_PROMPT | llm).invoke({"data": json.dumps(summaries, indent=2)})
        result = parse_json(raw.content)
        if result: cache_set(cache_key, result); return result
        raise RuntimeError(f"Unparseable: {raw.content[:300]}")
    except Exception as exc:
        st.error(f"Board analysis failed: {exc}")
        return {"winner":"Analysis failed","winner_score":0,"winner_rationale":str(exc),
                "losers":[],"executive_summary":"","negotiation_tips":[],"key_risks":[]}

def vsummary(vname, vdata, requirements):
    audit = vdata.get("audit",{}); risks = vdata.get("risks",[])
    total = len(requirements)
    met  = sum(1 for r in requirements if audit.get(r["id"],{}).get("status")=="Met")
    part = sum(1 for r in requirements if audit.get(r["id"],{}).get("status")=="Partial")
    miss = sum(1 for r in requirements if audit.get(r["id"],{}).get("status")=="Missing")
    score = round((met + 0.5*part) / max(total,1) * 100)
    cats = {}
    for req in requirements:
        cat = req.get("category","?"); a = audit.get(req["id"],{})
        if cat not in cats: cats[cat] = {"met":0,"partial":0,"total":0}
        cats[cat]["total"] += 1
        if a.get("status")=="Met": cats[cat]["met"] += 1
        elif a.get("status")=="Partial": cats[cat]["partial"] += 1
    return {"name":vname,"compliance_score":score,"met":met,"partial":part,"missing":miss,"total":total,
            "risk_count":len(risks),
            "critical_risks":sum(1 for r in risks if r.get("risk_level") in ["Critical","High"]),
            "category_scores":{c:round((v["met"]+0.5*v["partial"])/max(v["total"],1)*100) for c,v in cats.items()},
            "top_risks":[r.get("risk_type","?") for r in risks[:3]]}

# ═══════════════════════════════════════════════════════════
# VENDOR CARD
# ═══════════════════════════════════════════════════════════
def render_vendor_card(vname, status="Processing", score=None):
    status_cls = "vendor-card-status-done" if status.startswith("Completed") else "vendor-card-status-proc"
    icon = "✅" if status.startswith("Completed") else "⏳"
    score_html = ""
    if score is not None:
        sc_col = "var(--green)" if score>=75 else "var(--amber)" if score>=50 else "var(--red)"
        score_html = (f"<span style='font-family:var(--fc);font-size:.9rem;font-weight:700;"
                      f"color:{sc_col};margin-left:auto;padding:.15rem .5rem;"
                      f"background:rgba(0,0,0,.3);border-radius:4px'>{score}%</span>")
    st.markdown(f"""<div class='vendor-card'>
      <span style='font-size:1.5rem;flex-shrink:0'>📄</span>
      <div style='flex:1;min-width:0'>
        <div class='vendor-card-name'>{vname}</div>
        <div class='{status_cls}'>{icon} {status}</div>
      </div>
      {score_html}
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
VIEWS = ["RFP Setup","Fast Screening","Requirements Review","Vendor Dashboard","Board Analysis"]

with st.sidebar:
    st.markdown("""<div style='padding:.6rem 0 1.4rem'>
      <div style='font-family:var(--fc);font-size:.52rem;color:var(--amber);letter-spacing:.22em;
                  text-transform:uppercase;margin-bottom:.35rem'>Procurement Intelligence</div>
      <div style='font-size:1.2rem;font-weight:800;font-family:var(--fh);line-height:1.2'>
        ⚖ Tender Compliance OS
      </div>
      <div style='font-family:var(--fc);font-size:.58rem;color:var(--muted);margin-top:.3rem'>
        LangChain · RAG · gpt-4o-mini · v2.3
      </div>
    </div>""", unsafe_allow_html=True)

    nav = st.radio("", VIEWS, index=VIEWS.index(st.session_state.view) if st.session_state.view in VIEWS else 0)
    st.session_state.view = nav
    st.markdown("<div class='gap-sm'></div>", unsafe_allow_html=True)

    badges_html = ""
    if st.session_state.analysis_mode and st.session_state.analysis_mode not in ("transitioning_to_deep",):
        badge = "mode-fast" if st.session_state.analysis_mode=="fast" else "mode-deep"
        label = "⚡ Fast Mode" if st.session_state.analysis_mode=="fast" else "🔬 Deep Mode"
        badges_html += f"<div style='margin-bottom:.5rem'><span class='mode-badge {badge}'>{label}</span></div>"
    if st.session_state.requirements_locked:
        n = len(st.session_state.requirements)
        badges_html += f"<div style='font-family:var(--fc);font-size:.68rem;margin-bottom:.3rem'>{pill('Locked','p-green')} <span style='color:var(--muted)'>{n} reqs</span></div>"
    if st.session_state.critical_requirements:
        n = len(st.session_state.critical_requirements)
        badges_html += f"<div style='font-family:var(--fc);font-size:.68rem;margin-bottom:.3rem'>{pill(f'{n} Critical','p-cyan')}</div>"
    if st.session_state.vendors:
        n = len(st.session_state.vendors)
        badges_html += f"<div style='font-family:var(--fc);font-size:.68rem;margin-bottom:.3rem'>{pill(f'{n} Vendor(s)','p-blue')}</div>"
    if badges_html:
        st.markdown(badges_html, unsafe_allow_html=True)

    st.markdown("---")
    cache_count = len(list(CACHE_DIR.glob("*.json")))
    st.markdown(
        f"<div style='font-family:var(--fc);font-size:.6rem;color:var(--muted);line-height:1.8'>"
        f"Model: <span style='color:var(--text2)'>{LLM_MODEL}</span><br>"
        f"Embed: <span style='color:var(--text2)'>{EMBED_MODEL}</span><br>"
        f"Cache: <span style='color:var(--text2)'>{cache_count} entries</span>"
        f"</div>",
        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SHARED PAGE HEADER
# ═══════════════════════════════════════════════════════════
def page_header(title, subtitle):
    st.markdown(
        f"<div class='page-title'>{title}</div>"
        f"<div class='page-sub'>{subtitle}</div>",
        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# VIEW: RFP SETUP
# ═══════════════════════════════════════════════════════════
if st.session_state.view == "RFP Setup":
    page_header("RFP Setup", "Upload · Choose Mode · Extract · Lock")

    uploaded = st.file_uploader("Upload RFP Document (PDF)", type=["pdf"], key="rfp_upload")
    if uploaded:
        file_bytes = uploaded.read()
        fh = sha(file_bytes)
        if st.session_state.rfp_hash != fh:
            with st.spinner("📄 Parsing PDF…"):
                raw_docs = load_pdf_bytes(file_bytes, "rfp")
            st.session_state.update({
                "rfp_hash": fh, "rfp_bytes": file_bytes,
                "rfp_pages": [(d.metadata["page"], d.page_content) for d in raw_docs],
                "rfp_text": "\n\n".join(d.page_content for d in raw_docs),
                "rfp_chunks": None, "rfp_vs_built": False,
                "requirements": [], "critical_requirements": [],
                "requirements_locked": False, "board_cache": None,
                "analysis_mode": None, "vendor_screening": {},
            })

        wc = len(st.session_state.rfp_text.split())
        c1,c2,c3 = st.columns(3)
        c1.markdown(f"<div class='metric'><div class='v'>{len(st.session_state.rfp_pages)}</div><div class='l'>Pages</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric'><div class='v'>{wc:,}</div><div class='l'>Words</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric'><div class='v'>{len(st.session_state.requirements) or len(st.session_state.critical_requirements)}</div><div class='l'>Requirements</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='gap-md'></div>", unsafe_allow_html=True)

        st.markdown("<div class='hdr'>Select Analysis Mode</div>", unsafe_allow_html=True)
        mc1,mc2 = st.columns(2)
        with mc1:
            st.markdown("""
            <div style='background:rgba(6,182,212,.04);border:1px solid rgba(6,182,212,.22);
                        border-radius:var(--radius-lg);padding:1.1rem 1.3rem;margin-bottom:.6rem'>
              <div style='color:var(--cyan);font-family:var(--fc);font-size:.68rem;font-weight:700;margin-bottom:.45rem;letter-spacing:.06em'>⚡ FAST SCREENING MODE</div>
              <div style='font-size:.82rem;line-height:1.65;color:var(--text2)'>Extracts top 30–50 critical requirements only. Rapidly scores vendors so you can eliminate weak candidates before deep analysis.</div>
              <div style='margin-top:.65rem;font-family:var(--fc);font-size:.62rem;color:var(--muted)'>⏱ Seconds &nbsp;·&nbsp; 📋 30–50 reqs</div>
            </div>""", unsafe_allow_html=True)
            if st.button("⚡ Start Fast Screening", use_container_width=True, key="btn_fast"):
                st.session_state.analysis_mode = "fast"; st.rerun()
        with mc2:
            st.markdown("""
            <div style='background:rgba(139,92,246,.04);border:1px solid rgba(139,92,246,.22);
                        border-radius:var(--radius-lg);padding:1.1rem 1.3rem;margin-bottom:.6rem'>
              <div style='color:var(--purple);font-family:var(--fc);font-size:.68rem;font-weight:700;margin-bottom:.45rem;letter-spacing:.06em'>🔬 DEEP ANALYSIS MODE</div>
              <div style='font-size:.82rem;line-height:1.65;color:var(--text2)'>Extracts all 800+ requirements with full categorisation, RAG audit, risk detection, and board-level recommendation.</div>
              <div style='margin-top:.65rem;font-family:var(--fc);font-size:.62rem;color:var(--muted)'>⏱ Minutes &nbsp;·&nbsp; 📋 800+ reqs</div>
            </div>""", unsafe_allow_html=True)
            if st.button("🔬 Start Deep Analysis", use_container_width=True, key="btn_deep"):
                st.session_state.analysis_mode = "deep"; st.rerun()

        if st.session_state.analysis_mode and not (st.session_state.critical_requirements or st.session_state.requirements):
            mode = st.session_state.analysis_mode
            btn_label = "⚡ Extract Critical Requirements" if mode=="fast" else "🔬 Extract All Requirements"
            st.markdown("<div class='gap-xs'></div>", unsafe_allow_html=True)
            if st.button(btn_label, use_container_width=True):
                prog = st.progress(0.0); msg = st.empty()
                try:
                    msg.markdown("<div style='font-family:var(--fc);font-size:.7rem;color:var(--amber)'>▶ Chunking document…</div>", unsafe_allow_html=True)
                    raw_docs = load_pdf_bytes(st.session_state.rfp_bytes, "rfp")
                    if not raw_docs:
                        prog.empty(); msg.empty()
                        st.error("❌ PDF has no extractable text."); st.stop()
                    chunks = split_docs(raw_docs); st.session_state.rfp_chunks = chunks; prog.progress(0.15)
                    msg.markdown("<div style='font-family:var(--fc);font-size:.7rem;color:var(--amber)'>▶ Building vector index (hash-cached)…</div>", unsafe_allow_html=True)
                    with st.spinner("Building vector index…"):
                        get_or_build_vectorstore(chunks, f"rfp_{fh}")
                    st.session_state.rfp_vs_built = True; prog.progress(0.30)
                    if mode == "fast":
                        msg.markdown("<div style='font-family:var(--fc);font-size:.7rem;color:var(--cyan)'>▶ Extracting critical requirements…</div>", unsafe_allow_html=True)
                        reqs = extract_critical_requirements(fh, chunks, lambda p: prog.progress(0.3 + p*0.65))
                        st.session_state.critical_requirements = reqs
                        msg.markdown(f"<div style='font-family:var(--fc);font-size:.7rem;color:var(--green)'>✓ {len(reqs)} critical requirements extracted</div>", unsafe_allow_html=True)
                    else:
                        msg.markdown("<div style='font-family:var(--fc);font-size:.7rem;color:var(--purple)'>▶ Extracting all requirements (parallel batches)…</div>", unsafe_allow_html=True)
                        reqs = extract_requirements_from_rfp(fh, chunks, lambda p: prog.progress(0.30 + p*0.65))
                        st.session_state.requirements = reqs
                        msg.markdown(f"<div style='font-family:var(--fc);font-size:.7rem;color:var(--green)'>✓ {len(reqs)} requirements extracted</div>", unsafe_allow_html=True)
                    st.rerun()
                except RuntimeError as exc: prog.empty(); msg.empty(); st.error(f"**Extraction failed:**\n\n{exc}")
                except Exception as exc:   prog.empty(); msg.empty(); st.error(f"**Unexpected error:** {exc}")

        if st.session_state.requirements and st.session_state.analysis_mode == "deep":
            reqs = st.session_state.requirements
            st.markdown("---")
            st.markdown("<div class='hdr'>Requirement Review & Management</div>", unsafe_allow_html=True)
            if not st.session_state.requirements_locked:
                def _lock_deep():
                    st.session_state.requirements_locked = True
                    st.session_state.view = "Vendor Dashboard"
                    st.rerun()
                render_requirement_review(reqs, _lock_deep, reqs_key="requirements")
            else:
                st.success(f"✅ {len(reqs)} requirements locked.")
                for cat in sorted(set(r.get("category","?") for r in reqs)):
                    cnt = sum(1 for r in reqs if r.get("category")==cat)
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;align-items:center;"
                        f"padding:.35rem 0;border-bottom:1px solid var(--border);font-family:var(--fc);font-size:.78rem'>"
                        f"<span style='color:var(--text2)'>{cat}</span>"
                        f"<span style='color:var(--amber);font-weight:600'>{cnt}</span></div>",
                        unsafe_allow_html=True)
                st.markdown("<div class='gap-sm'></div>", unsafe_allow_html=True)
                if st.button("🔓 Unlock Requirements"):
                    st.session_state.requirements_locked = False; st.rerun()

        if st.session_state.critical_requirements and st.session_state.analysis_mode == "fast":
            reqs = st.session_state.critical_requirements
            st.markdown("---")
            st.markdown(f"<div class='hdr'>Critical Requirements Preview &nbsp;{pill(f'{len(reqs)} Extracted','p-cyan')}</div>", unsafe_allow_html=True)
            display_and_export_requirements(reqs, "Critical Requirements (Fast Mode)")
            st.markdown("<div class='gap-sm'></div>", unsafe_allow_html=True)
            if st.button("⚡ Proceed to Fast Screening →", use_container_width=True):
                st.session_state.view = "Fast Screening"; st.rerun()

# ═══════════════════════════════════════════════════════════
# VIEW: FAST SCREENING
# ═══════════════════════════════════════════════════════════
elif st.session_state.view == "Fast Screening":
    page_header("⚡ Fast Screening", "Upload vendors · Quick score · Eliminate weak candidates")

    if not st.session_state.critical_requirements:
        st.warning("⚠ Run Fast Screening Mode in RFP Setup first."); st.stop()

    crit_reqs = st.session_state.critical_requirements
    threshold = st.slider("Screening Threshold (%)", 0, 100, FAST_THRESHOLD,
                          help="Vendors scoring below this are flagged for removal")
    req_hash  = sha(json.dumps([r["id"] for r in crit_reqs]))

    with st.expander("📤 Upload Vendor PDFs for Screening", expanded=not bool(st.session_state.vendors)):
        vfiles = st.file_uploader("Upload one or more vendor PDFs", type=["pdf"],
                                  accept_multiple_files=True, key="screen_vup")
        if vfiles and st.button("⚡ Screen Vendors", use_container_width=True):
            for vf in vfiles:
                vname = vf.name.replace(".pdf",""); vbytes = vf.read(); vh = sha(vbytes)
                if (vname in st.session_state.vendors and
                    st.session_state.vendors[vname].get("hash")==vh and
                    vname in st.session_state.vendor_screening):
                    render_vendor_card(vname, "Completed (cached)",
                                       score=st.session_state.vendor_screening[vname]["score"])
                    continue
                render_vendor_card(vname, "Processing…")
                prog = st.progress(0.0)
                try:
                    with st.spinner(f"📄 Parsing {vname}…"):
                        vdocs = load_pdf_bytes(vbytes, vname)
                    if not vdocs: st.error(f"❌ {vname}: no extractable text."); continue
                    vchunks = split_docs(vdocs); prog.progress(0.2)
                    with st.spinner(f"🔍 Building index for {vname}…"):
                        vs, _ = get_or_build_vectorstore(vchunks, f"vendor_{vh}")
                    prog.progress(0.45)
                    audit = audit_vendor_rag(crit_reqs, vs, vh, f"fast_{req_hash}",
                                            lambda p: prog.progress(0.45+p*0.45))
                    score = compute_score(crit_reqs, audit); prog.progress(1.0)
                    st.session_state.vendors[vname] = {"hash":vh,"bytes":vbytes,
                        "pages":[(d.metadata["page"],d.page_content) for d in vdocs],
                        "vectorstore":vs,"audit_fast":audit,"risks":[]}
                    st.session_state.vendor_screening[vname] = {"score":score,"audit":audit}
                except Exception as exc:
                    prog.empty(); st.error(f"**{vname} failed:** {exc}")
            st.rerun()

    if not st.session_state.vendor_screening:
        st.info("Upload and screen vendors above."); st.stop()

    st.markdown("---")
    st.markdown("<div class='hdr'>Screening Results</div>", unsafe_allow_html=True)

    screening = st.session_state.vendor_screening
    vendors_sorted = sorted(screening.items(), key=lambda x: -x[1]["score"])
    passing = [v for v,d in vendors_sorted if d["score"]>=threshold]
    failing  = [v for v,d in vendors_sorted if d["score"]< threshold]

    cs1,cs2,cs3 = st.columns(3)
    cs1.markdown(f"<div class='metric'><div class='v'>{len(vendors_sorted)}</div><div class='l'>Total Vendors</div></div>", unsafe_allow_html=True)
    cs2.markdown(f"<div class='metric'><div class='v' style='color:var(--green)'>{len(passing)}</div><div class='l'>Above Threshold</div></div>", unsafe_allow_html=True)
    cs3.markdown(f"<div class='metric'><div class='v' style='color:var(--red)'>{len(failing)}</div><div class='l'>Below Threshold</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='gap-md'></div>", unsafe_allow_html=True)

    select_map = {}
    for vname, sdata in vendors_sorted:
        score = sdata["score"]; audit = sdata["audit"]
        met  = sum(1 for r in crit_reqs if audit.get(r["id"],{}).get("status")=="Met")
        part = sum(1 for r in crit_reqs if audit.get(r["id"],{}).get("status")=="Partial")
        miss = sum(1 for r in crit_reqs if audit.get(r["id"],{}).get("status")=="Missing")
        passes = score >= threshold
        card_cls = "screen-pass" if passes else "screen-fail"
        score_col = "var(--green)" if score>=75 else "var(--amber)" if score>=50 else "var(--red)"
        status_pill = pill("PASS","p-green") if passes else pill("FAIL","p-red")
        col_chk, col_body = st.columns([0.06, 0.94])
        with col_chk:
            st.markdown("<div style='padding-top:.9rem'>", unsafe_allow_html=True)
            selected = st.checkbox("", value=passes, key=f"sel_{vname}")
            st.markdown("</div>", unsafe_allow_html=True)
        with col_body:
            st.markdown(f"""<div class='screen-card {card_cls}'>
              <div style='display:flex;justify-content:space-between;align-items:center'>
                <div style='display:flex;align-items:center;gap:.7rem'>
                  <span style='font-size:1.35rem'>📄</span>
                  <div><span style='font-weight:700;font-size:.98rem;font-family:var(--fh)'>{vname}</span>&nbsp;&nbsp;{status_pill}</div>
                </div>
                <div style='font-family:var(--fc);font-size:1.5rem;font-weight:700;color:{score_col};padding:.1rem .5rem;background:rgba(0,0,0,.25);border-radius:4px'>{score}%</div>
              </div>
              <div style='display:flex;gap:1.8rem;margin-top:.6rem;font-family:var(--fc);font-size:.66rem;letter-spacing:.02em'>
                <span style='color:var(--green)'>✅ Met: {met}</span>
                <span style='color:var(--amber)'>⚠ Partial: {part}</span>
                <span style='color:var(--red)'>❌ Missing: {miss}</span>
                <span style='color:var(--muted)'>of {len(crit_reqs)} critical</span>
              </div>
            </div>""", unsafe_allow_html=True)
        select_map[vname] = selected

    st.markdown("<div class='gap-sm'></div>", unsafe_allow_html=True)

    if failing:
        st.markdown(
            f"<div style='background:rgba(239,68,68,.05);border:1px solid rgba(239,68,68,.18);"
            f"border-left:3px solid var(--red);border-radius:var(--radius);padding:.75rem 1rem;"
            f"font-family:var(--fc);font-size:.73rem;margin-bottom:.9rem'>"
            f"⚠ <strong style='color:var(--red)'>{len(failing)} vendor(s) below {threshold}% threshold:</strong>"
            f" {', '.join(failing)}</div>",
            unsafe_allow_html=True)

    col_remove, col_spacer, col_proceed = st.columns([1, .08, 1])
    with col_remove:
        if st.button("🗑 Remove Deselected Vendors", use_container_width=True):
            to_remove = [v for v,sel in select_map.items() if not sel]
            for v in to_remove:
                st.session_state.vendors.pop(v,None)
                st.session_state.vendor_screening.pop(v,None)
            st.success(f"Removed: {', '.join(to_remove) or 'none'}"); st.rerun()
    with col_proceed:
        selected_vendors = [v for v,sel in select_map.items() if sel]
        if st.button(f"🚀 Proceed with {len(selected_vendors)} Vendor(s) → Deep Analysis", use_container_width=True):
            if not selected_vendors:
                st.error("Select at least one vendor to proceed.")
            else:
                to_remove = [v for v in list(st.session_state.vendors) if v not in selected_vendors]
                for v in to_remove:
                    st.session_state.vendors.pop(v,None)
                    st.session_state.vendor_screening.pop(v,None)
                st.session_state.vendors_for_deep = selected_vendors
                st.session_state.analysis_mode = "transitioning_to_deep"
                st.session_state.view = "Requirements Review"
                st.rerun()

# ═══════════════════════════════════════════════════════════
# VIEW: REQUIREMENTS REVIEW
# ═══════════════════════════════════════════════════════════
elif st.session_state.view == "Requirements Review":
    page_header("Requirements Review", "Full extraction · Edit · Filter · View Source · Lock → Vendors")

    if not st.session_state.rfp_bytes:
        st.warning("⚠ No RFP uploaded. Please go back to RFP Setup."); st.stop()

    if not st.session_state.requirements:
        st.info("🔬 Extracting all requirements from stored RFP data (no re-upload needed)…")
        prog = st.progress(0.0); msg = st.empty()
        try:
            if st.session_state.rfp_chunks:
                chunks = st.session_state.rfp_chunks
            else:
                msg.markdown("<div style='font-family:var(--fc);font-size:.7rem;color:var(--amber)'>▶ Re-chunking from cached bytes…</div>", unsafe_allow_html=True)
                with st.spinner("Re-chunking PDF…"):
                    raw_docs = load_pdf_bytes(st.session_state.rfp_bytes, "rfp")
                    chunks = split_docs(raw_docs)
                st.session_state.rfp_chunks = chunks
            prog.progress(0.10)
            msg.markdown("<div style='font-family:var(--fc);font-size:.7rem;color:var(--purple)'>▶ Extracting all requirements (parallel batches)…</div>", unsafe_allow_html=True)
            reqs = extract_requirements_from_rfp(st.session_state.rfp_hash, chunks,
                                                  lambda p: prog.progress(0.10 + p*0.85))
            st.session_state.requirements = reqs
            st.session_state.analysis_mode = "deep"
            prog.progress(1.0)
            msg.markdown(f"<div style='font-family:var(--fc);font-size:.7rem;color:var(--green)'>✓ {len(reqs)} requirements extracted</div>", unsafe_allow_html=True)
            st.rerun()
        except Exception as exc:
            prog.empty(); st.error(f"**Extraction failed:** {exc}"); st.stop()

    reqs = st.session_state.requirements

    if st.session_state.requirements_locked:
        st.success(f"✅ {len(reqs)} requirements locked.")
        if st.button("→ Go to Vendor Dashboard", use_container_width=True):
            st.session_state.view = "Vendor Dashboard"; st.rerun()
        st.stop()

    st.markdown("<div class='hdr'>Review & Manage All Requirements Before Locking</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-family:var(--fc);font-size:.72rem;color:var(--muted);margin-bottom:1.1rem'>"
        f"{len(reqs)} requirements extracted. Edit, filter, view RFP source pages, then lock to proceed.</div>",
        unsafe_allow_html=True)

    def _lock_from_review():
        st.session_state.requirements_locked = True
        st.session_state.view = "Vendor Dashboard"
        st.rerun()

    render_requirement_review(reqs, _lock_from_review, reqs_key="requirements")

# ═══════════════════════════════════════════════════════════
# VIEW: VENDOR DASHBOARD
# ═══════════════════════════════════════════════════════════
elif st.session_state.view == "Vendor Dashboard":
    page_header("Vendor Dashboard", "RAG-powered compliance audit · Parallel batching · Risk detection")

    if not st.session_state.requirements_locked:
        st.warning("⚠ Lock requirements first. Go to RFP Setup (Deep Mode) or Requirements Review."); st.stop()

    requirements = st.session_state.requirements
    req_hash = sha(json.dumps([r["id"] for r in requirements]))

    with st.expander("📤 Upload Vendor Proposals", expanded=not bool(st.session_state.vendors)):
        vfiles = st.file_uploader("Upload one or more vendor PDFs", type=["pdf"],
                                  accept_multiple_files=True, key="deep_vup")
        if vfiles and st.button("🚀 Process Vendors", use_container_width=True):
            for vf in vfiles:
                vname = vf.name.replace(".pdf",""); vbytes = vf.read(); vh = sha(vbytes)
                if (vname in st.session_state.vendors and
                    st.session_state.vendors[vname].get("hash")==vh and
                    "audit" in st.session_state.vendors[vname]):
                    render_vendor_card(vname, "Completed (cached)",
                                       score=compute_score(requirements, st.session_state.vendors[vname]["audit"]))
                    continue
                render_vendor_card(vname, "Processing…")
                prog = st.progress(0.0); msgp = st.empty()
                try:
                    msgp.markdown("<div style='font-family:var(--fc);font-size:.68rem;color:var(--muted)'>📄 Parsing PDF…</div>", unsafe_allow_html=True)
                    with st.spinner(f"Parsing {vname}…"):
                        vdocs = load_pdf_bytes(vbytes, vname)
                    if not vdocs: st.error(f"❌ {vname}: no extractable text."); continue
                    vchunks = split_docs(vdocs); prog.progress(0.15)
                    msgp.markdown("<div style='font-family:var(--fc);font-size:.68rem;color:var(--muted)'>🔍 Loading / building vector index…</div>", unsafe_allow_html=True)
                    with st.spinner(f"Building index for {vname}…"):
                        vs = (st.session_state.vendors.get(vname,{}).get("vectorstore") or
                              get_or_build_vectorstore(vchunks, f"vendor_{vh}")[0])
                    prog.progress(0.30)
                    msgp.markdown("<div style='font-family:var(--fc);font-size:.68rem;color:var(--muted)'>🤖 Running parallel RAG audit…</div>", unsafe_allow_html=True)
                    audit = audit_vendor_rag(requirements, vs, vh, req_hash,
                                            lambda p: prog.progress(0.30+p*0.50))
                    prog.progress(0.85)
                    msgp.markdown("<div style='font-family:var(--fc);font-size:.68rem;color:var(--muted)'>⚠ Scanning risk clauses…</div>", unsafe_allow_html=True)
                    with st.spinner(f"Scanning risks for {vname}…"):
                        risks = detect_risks_rag(vs, vh)
                    prog.progress(1.0)
                    st.session_state.vendors[vname] = {"hash":vh,"bytes":vbytes,
                        "pages":[(d.metadata["page"],d.page_content) for d in vdocs],
                        "vectorstore":vs,"audit":audit,"risks":risks}
                    msgp.markdown(f"<div style='font-family:var(--fc);font-size:.68rem;color:var(--green)'>✓ {vname} complete</div>", unsafe_allow_html=True)
                except Exception as exc:
                    prog.empty(); st.error(f"**{vname} failed:** {exc}")
            st.rerun()

    # Upgrade fast-screened vendors
    for vname, vdata in list(st.session_state.vendors.items()):
        if "audit" not in vdata and "audit_fast" in vdata:
            vs = vdata.get("vectorstore")
            if vs:
                with st.spinner(f"Running full audit for {vname}…"):
                    vdata["audit"] = audit_vendor_rag(requirements, vs, vdata["hash"], req_hash)
                    vdata["risks"] = detect_risks_rag(vs, vdata["hash"])
                st.rerun()

    ready = {n:d for n,d in st.session_state.vendors.items() if "audit" in d}
    if not ready:
        st.info("No vendors with deep analysis yet. Upload vendors above."); st.stop()

    vtabs = st.tabs([f"  {n}  " for n in ready.keys()])
    for tab,(vname,vdata) in zip(vtabs, ready.items()):
        with tab:
            audit = vdata.get("audit",{}); risks = vdata.get("risks",[])
            total = len(requirements)
            met  = sum(1 for r in requirements if audit.get(r["id"],{}).get("status")=="Met")
            part = sum(1 for r in requirements if audit.get(r["id"],{}).get("status")=="Partial")
            miss = sum(1 for r in requirements if audit.get(r["id"],{}).get("status")=="Missing")
            nc   = sum(1 for r in requirements if audit.get(r["id"],{}).get("status")=="Non-compliant")
            score = round((met+0.5*part)/max(total,1)*100)
            sc_col = "var(--green)" if score>=75 else "var(--amber)" if score>=50 else "var(--red)"

            st.markdown("<div class='gap-xs'></div>", unsafe_allow_html=True)
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.markdown(f"<div class='metric'><div class='v' style='color:{sc_col}'>{score}%</div><div class='l'>Score</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric'><div class='v' style='color:var(--green)'>{met}</div><div class='l'>Met</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric'><div class='v' style='color:var(--amber)'>{part}</div><div class='l'>Partial</div></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric'><div class='v' style='color:var(--red)'>{miss}</div><div class='l'>Missing</div></div>", unsafe_allow_html=True)
            c5.markdown(f"<div class='metric'><div class='v' style='color:var(--red)'>{nc}</div><div class='l'>Non-Compliant</div></div>", unsafe_allow_html=True)
            c6.markdown(f"<div class='metric'><div class='v' style='color:var(--muted)'>{len(risks)}</div><div class='l'>Risks</div></div>", unsafe_allow_html=True)
            st.markdown("<div class='gap-sm'></div>", unsafe_allow_html=True)

            itabs = st.tabs(["  📋 Compliance  ","  🚨 Risks  ","  📄 PDF Viewer  "])

            with itabs[0]:
                fc1,fc2 = st.columns(2)
                with fc1:
                    sf = st.selectbox("Status",["All","Met","Partial","Missing","Non-compliant"], key=f"sf_{vname}")
                with fc2:
                    cf = st.selectbox("Category",
                                      ["All"]+list(set(r.get("category","?") for r in requirements)),
                                      key=f"cf_{vname}")
                st.markdown("<div class='gap-xs'></div>", unsafe_allow_html=True)

                for req in requirements:
                    a = audit.get(req["id"],{})
                    status = a.get("status","Missing")
                    if sf!="All" and status!=sf: continue
                    if cf!="All" and req.get("category")!=cf: continue

                    sc_cls    = STATUS_CLS.get(status,"p-gray")
                    rl_cls    = RISK_CLS.get(req.get("risk_level","Low"),"p-gray")
                    conf      = a.get("confidence",0)
                    vendor_pg = a.get("page_reference")

                    with st.expander(f"{req.get('requirement_summary','')[:82]}"):
                        h1,h2,h3 = st.columns([3,1,1])
                        with h1:
                            st.markdown(
                                f"{pill(status,sc_cls)}&nbsp;{pill(req.get('category','?'),'p-purple')}&nbsp;{pill(req.get('risk_level','?'),rl_cls)}",
                                unsafe_allow_html=True)
                        with h2:
                            st.markdown(
                                f"<div style='font-family:var(--fc);font-size:.64rem;color:var(--muted);margin-bottom:.1rem'>Confidence</div>"
                                f"<div style='font-family:var(--fc);font-size:.82rem;color:var(--amber);font-weight:600'>{conf}%</div>",
                                unsafe_allow_html=True)
                        with h3:
                            if vendor_pg:
                                st.markdown(
                                    f"<div style='font-family:var(--fc);font-size:.64rem;color:var(--muted);margin-bottom:.1rem'>Evidence Pg</div>"
                                    f"<div style='font-family:var(--fc);font-size:.82rem;font-weight:600'>{vendor_pg}</div>",
                                    unsafe_allow_html=True)

                        st.markdown("<div class='gap-xs'></div>", unsafe_allow_html=True)

                        if req.get("exact_quote"):
                            st.markdown(f"<div class='hdr'>RFP Requirement</div><div class='rfp-quote-box'>{req['exact_quote'][:240]}</div>", unsafe_allow_html=True)

                        st.markdown("<div class='gap-xs'></div>", unsafe_allow_html=True)

                        if a.get("matched_claim"):
                            st.markdown(f"<div class='hdr'>📌 Evidence from Vendor Proposal</div><div class='evidence-box'>{a['matched_claim'][:240]}</div>", unsafe_allow_html=True)
                            st.markdown("<div class='gap-xs'></div>", unsafe_allow_html=True)
                            if vendor_pg and "bytes" in vdata:
                                ev_key      = f"evpg_{vname}_{req['id']}"
                                ev_show_key = f"show_{ev_key}"
                                if ev_show_key not in st.session_state:
                                    st.session_state[ev_show_key] = False
                                st.markdown("<div class='btn-evidence'>", unsafe_allow_html=True)
                                if st.button(f"📄 View Evidence — Page {vendor_pg}", key=ev_key):
                                    st.session_state[ev_show_key] = not st.session_state[ev_show_key]
                                st.markdown("</div>", unsafe_allow_html=True)
                                if st.session_state[ev_show_key]:
                                    # Spinner gives immediate feedback; cache makes repeat clicks instant
                                    with st.spinner(f"Loading evidence page {vendor_pg}…"):
                                        try:
                                            b64 = render_page_b64_cached(vdata["bytes"], vendor_pg)
                                            st.markdown(
                                                f"<div class='pdf-viewer-frame'>"
                                                f"<img src='data:image/png;base64,{b64}'/>"
                                                f"<div class='pdf-viewer-caption'>Vendor PDF — Page {vendor_pg}</div>"
                                                f"</div>",
                                                unsafe_allow_html=True)
                                        except Exception as exc:
                                            st.error(f"Render error: {exc}")
                            else:
                                st.markdown("<div class='note-muted'>⚠ No verifiable evidence found</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='note-muted'>⚠ No verifiable evidence found</div>", unsafe_allow_html=True)

                        if a.get("reasoning"):
                            st.markdown(f"<div class='reasoning'>↳ {a['reasoning']}</div>", unsafe_allow_html=True)

                        st.markdown("<div class='gap-xs'></div>", unsafe_allow_html=True)
                        render_rfp_source_viewer(req, viewer_key=f"rfpsrc_{vname}_{req['id']}")

            with itabs[1]:
                if not risks:
                    st.success("✅ No significant risk clauses detected.")
                else:
                    RISK_ORDER = ["Critical","High","Medium","Low"]
                    for r in sorted(risks, key=lambda x: RISK_ORDER.index(x.get("risk_level","Low"))
                                    if x.get("risk_level") in RISK_ORDER else 99):
                        rl  = r.get("risk_level","Low")
                        rc  = {"Critical":"var(--red)","High":"var(--red)","Medium":"var(--amber)","Low":"var(--muted)"}.get(rl,"var(--muted)")
                        rpc = RISK_CLS.get(rl,"p-gray")
                        st.markdown(f"""<div class='risk-card'>
                          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem'>
                            <span style='font-weight:700;font-size:.88rem;color:{rc}'>{r.get('risk_type','Risk')}</span>
                            <span class='pill {rpc}'>{rl}</span>
                          </div>
                          <div style='background:rgba(0,0,0,.35);border-left:3px solid {rc};padding:.4rem .75rem;border-radius:0 4px 4px 0;font-family:var(--fc);font-size:.7rem;color:var(--text2);margin-bottom:.4rem;line-height:1.55'>
                            {r.get('exact_clause','')[:220]}
                          </div>
                          <div style='color:var(--muted);font-family:var(--fc);font-size:.68rem;line-height:1.5'>
                            ↳ {r.get('explanation','')}
                          </div>
                        </div>""", unsafe_allow_html=True)

            with itabs[2]:
                if "bytes" in vdata:
                    maxp  = len(vdata.get("pages",[]))
                    defpg = int(st.session_state.get(f"vpg_{vname}",1))
                    pgnum = st.number_input("Page", min_value=1, max_value=max(maxp,1),
                                            value=defpg, key=f"pn_{vname}")
                    # Spinner covers the first render; cache makes subsequent page changes fast
                    with st.spinner(f"Loading page {pgnum}…"):
                        try:
                            b64 = render_page_b64_cached(vdata["bytes"], pgnum)
                            st.markdown(
                                f"<div class='pdf-viewer-frame'>"
                                f"<img src='data:image/png;base64,{b64}'/>"
                                f"<div class='pdf-viewer-caption'>Page {pgnum} / {maxp}</div>"
                                f"</div>",
                                unsafe_allow_html=True)
                        except Exception as exc:
                            st.error(f"Render error: {exc}")
                else:
                    st.info("PDF bytes not available.")

# ═══════════════════════════════════════════════════════════
# VIEW: BOARD ANALYSIS
# ═══════════════════════════════════════════════════════════
elif st.session_state.view == "Board Analysis":
    page_header("Board Analysis", "Decisive recommendation · Multi-vendor comparison · Risk heatmap")

    ready = {n:d for n,d in st.session_state.vendors.items() if "audit" in d}
    if not ready:
        st.warning("⚠ Run deep analysis on vendors first (Vendor Dashboard)."); st.stop()
    requirements = st.session_state.requirements
    if not requirements:
        st.warning("⚠ No full requirements found. Please run Deep Analysis mode."); st.stop()

    COLORS = ["#f59e0b","#3b82f6","#10b981","#8b5cf6","#ef4444","#06b6d4"]
    summaries = {n:vsummary(n,d,requirements) for n,d in ready.items()}

    if st.button("🏆 Generate Board Recommendation", use_container_width=True) or st.session_state.board_cache:
        if not st.session_state.board_cache:
            with st.spinner("🤖 Synthesising board recommendation…"):
                st.session_state.board_cache = run_board_analysis(summaries)
        a = st.session_state.board_cache
        winner = a.get("winner","")
        ws     = a.get("winner_score", summaries.get(winner,{}).get("compliance_score",0))
        losers = a.get("losers",[])
        tips   = a.get("negotiation_tips",[])
        krisks = a.get("key_risks",[])

        st.markdown(f"""<div class='winner-banner'>
          <div style='font-family:var(--fc);font-size:.52rem;color:var(--amber);letter-spacing:.22em;text-transform:uppercase;margin-bottom:.55rem'>Board Recommendation</div>
          <div style='display:flex;align-items:center;gap:1.1rem;margin-bottom:.8rem'>
            <div style='font-size:2.6rem;line-height:1'>🏆</div>
            <div>
              <div style='font-size:1.55rem;font-weight:800;font-family:var(--fh);line-height:1.15'>{winner}</div>
              <div style='font-family:var(--fc);color:var(--green);font-size:.78rem;margin-top:.25rem'>Compliance Score: {ws}%</div>
            </div>
          </div>
          <div style='font-size:.88rem;line-height:1.7;margin-bottom:.8rem;color:var(--text2)'>{a.get("winner_rationale","")}</div>
          <div style='font-family:var(--fc);font-size:.73rem;color:var(--muted);background:rgba(0,0,0,.3);padding:.65rem .9rem;border-radius:var(--radius-sm);border-left:3px solid var(--amber);line-height:1.6'>
            {a.get("executive_summary","")}
          </div>
        </div>""", unsafe_allow_html=True)

        col1,col2 = st.columns(2)
        with col1:
            st.markdown("<div class='hdr'>Why Others Were Rejected</div>", unsafe_allow_html=True)
            for l in losers:
                st.markdown(
                    f"<div style='background:rgba(239,68,68,.04);border:1px solid rgba(239,68,68,.14);"
                    f"border-left:3px solid var(--red);border-radius:0 var(--radius) var(--radius) 0;"
                    f"padding:.7rem 1rem;margin-bottom:.45rem'>"
                    f"<div style='display:flex;justify-content:space-between;margin-bottom:.25rem'>"
                    f"<span style='font-weight:700;font-size:.88rem'>{l.get('name','')}</span>"
                    f"<span style='font-family:var(--fc);color:var(--red);font-weight:600'>{l.get('score',0)}%</span>"
                    f"</div><div style='font-family:var(--fc);font-size:.7rem;color:var(--muted);line-height:1.55'>"
                    f"{l.get('failure_reason','')}</div></div>",
                    unsafe_allow_html=True)
        with col2:
            if tips:
                st.markdown("<div class='hdr'>Negotiation Tips</div>", unsafe_allow_html=True)
                for t in tips:
                    st.markdown(
                        f"<div style='font-family:var(--fc);font-size:.73rem;padding:.35rem 0;"
                        f"border-bottom:1px solid var(--border);color:var(--text2);line-height:1.5'>"
                        f"<span style='color:var(--amber);margin-right:.4rem'>→</span>{t}</div>",
                        unsafe_allow_html=True)
            if krisks:
                st.markdown("<div class='hdr' style='margin-top:1.1rem'>Key Procurement Risks</div>", unsafe_allow_html=True)
                for kr in krisks:
                    st.markdown(
                        f"<div style='font-family:var(--fc);font-size:.73rem;padding:.35rem 0;"
                        f"border-bottom:1px solid var(--border);color:var(--amber);line-height:1.5'>"
                        f"⚠ {kr}</div>",
                        unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div class='hdr'>Multi-Vendor Analytics</div>", unsafe_allow_html=True)

        vnames = [*summaries]
        scores = [summaries[v]["compliance_score"] for v in vnames]
        mets   = [summaries[v]["met"]     for v in vnames]
        parts  = [summaries[v]["partial"] for v in vnames]
        misses = [summaries[v]["missing"] for v in vnames]

        CHART_BG   = "rgba(0,0,0,0)"
        CHART_FONT = {"color":"#b8b5ae","family":"IBM Plex Mono","size":11}
        GRID_COL   = "#2a2a2a"
        CHART_MARGIN = {"t":44,"b":20,"l":10,"r":10}

        r1c1,r1c2 = st.columns(2)
        with r1c1:
            fig = go.Figure()
            for v,s in zip(vnames,scores):
                color = "#10b981" if s>=75 else "#f59e0b" if s>=50 else "#ef4444"
                fig.add_trace(go.Bar(x=[v],y=[s],marker_color=color,name=v,
                                     text=[f"{s}%"],textposition="outside",marker_line_width=0))
            fig.update_layout(title_text="Compliance Score",title_font=CHART_FONT,showlegend=False,
                              paper_bgcolor=CHART_BG,plot_bgcolor=CHART_BG,font=CHART_FONT,
                              yaxis={"range":[0,118],"gridcolor":GRID_COL,"zeroline":False},
                              xaxis={"gridcolor":GRID_COL},margin=CHART_MARGIN,bargap=0.35)
            st.plotly_chart(fig, use_container_width=True)
        with r1c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="Met",    x=vnames,y=mets,  marker_color="#10b981",marker_line_width=0))
            fig2.add_trace(go.Bar(name="Partial",x=vnames,y=parts, marker_color="#f59e0b",marker_line_width=0))
            fig2.add_trace(go.Bar(name="Missing",x=vnames,y=misses,marker_color="#ef4444",marker_line_width=0))
            fig2.update_layout(barmode="stack",title_text="Requirement Breakdown",title_font=CHART_FONT,
                               paper_bgcolor=CHART_BG,plot_bgcolor=CHART_BG,font=CHART_FONT,
                               yaxis={"gridcolor":GRID_COL,"zeroline":False},xaxis={"gridcolor":GRID_COL},
                               legend={"bgcolor":"rgba(0,0,0,0)","font":{"size":10}},
                               margin=CHART_MARGIN,bargap=0.25)
            st.plotly_chart(fig2, use_container_width=True)

        all_cats = list(set(r.get("category","?") for r in requirements))
        if all_cats:
            fig3 = go.Figure()
            for i,vn in enumerate(vnames):
                cat_sc = [summaries[vn]["category_scores"].get(c,0) for c in all_cats]
                fig3.add_trace(go.Scatterpolar(
                    r=cat_sc+[cat_sc[0]], theta=all_cats+[all_cats[0]],
                    fill="toself", name=vn, line_color=COLORS[i%len(COLORS)], opacity=.65, line_width=2))
            fig3.update_layout(
                polar=dict(bgcolor="rgba(0,0,0,0)",
                           radialaxis=dict(visible=True,range=[0,100],gridcolor=GRID_COL,color="#555",tickfont={"size":9}),
                           angularaxis=dict(gridcolor=GRID_COL,color="#777",tickfont={"size":10})),
                paper_bgcolor=CHART_BG,font=CHART_FONT,title_text="Category Radar",title_font=CHART_FONT,
                legend={"bgcolor":"rgba(0,0,0,0)","font":{"size":10}},margin={"t":60,"b":20,"l":40,"r":40})
            st.plotly_chart(fig3, use_container_width=True)

        all_risk_types = list(set(
            r.get("risk_type","?") for vd in ready.values() for r in vd.get("risks",[])))
        if all_risk_types:
            lvl = {"Critical":4,"High":3,"Medium":2,"Low":1}
            matrix = [
                [lvl.get(next((r.get("risk_level") for r in ready[vn].get("risks",[])
                               if r.get("risk_type")==rt), None), 0)
                 if any(r.get("risk_type")==rt for r in ready[vn].get("risks",[]))
                 else 0
                 for vn in vnames]
                for rt in all_risk_types
            ]
            fig4 = go.Figure(go.Heatmap(
                z=matrix, x=vnames, y=all_risk_types,
                colorscale=[[0,"#0a0a0a"],[.25,"#1a2a1a"],[.5,"#f59e0b"],[1,"#ef4444"]],
                text=[[["–","Low","Med","High","Critical"][v] for v in row] for row in matrix],
                texttemplate="%{text}", textfont={"size":10},
                showscale=False, xgap=3, ygap=3))
            fig4.update_layout(
                title_text="Risk Heatmap",title_font=CHART_FONT,
                paper_bgcolor=CHART_BG,plot_bgcolor=CHART_BG,
                font=CHART_FONT,margin={"t":50,"b":20,"l":10,"r":10})
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("<div class='gap-sm'></div>", unsafe_allow_html=True)
        if st.button("🔄 Regenerate Analysis"):
            st.session_state.board_cache = None; st.rerun()