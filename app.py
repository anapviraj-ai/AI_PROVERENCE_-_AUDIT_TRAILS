"""
app.py
─────────────────────────────────────────────────────────────────────────────
AI Model Provenance & Cryptographic Audit Trail — Interactive Streamlit Dashboard

Tabs
────
  1. Ingest Existing AI        — Upload or Quick-Load a pre-trained model → Genesis block
  2. Log Model Update          — Record fine-tuning / quantization → child block
  3. Audit Ledger & Lineage    — Searchable blockchain viewer + interactive tree
  4. Independent Verifier      — Interactive Tamper & Bit-Flip Attack Simulator

Run:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path

import requests
import streamlit as st

# ─── Config ───────────────────────────────────────────────────────────────────
# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE = os.environ.get("PROVENANCE_API_URL", "http://localhost:8000")
SAMPLE_DIR = Path("sample_models")

st.set_page_config(
    page_title="AI Provenance & Cryptographic Audit Trail",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global Design System (High-Contrast & Crystal-Clear Readability) ─────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

  /* ── Base & Universal Typography (Safe: Does NOT override icon fonts) ─────── */
  html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: radial-gradient(circle at 15% 15%, #0f172a 0%, #090e1a 60%, #050811 100%) !important;
    min-height: 100vh;
    color: #f8fafc !important;
  }

  /* ── Prevent Streamlit / Material Symbols Icon Corruption & Text Overlap ── */
  [data-testid="stIconMaterial"], 
  .material-symbols-rounded,
  .material-symbols-outlined,
  span[data-testid="stExpanderIcon"],
  div[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    font-style: normal !important;
    font-weight: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    color: #38bdf8 !important;
  }

  /* ── All Headings & Paragraphs ────────────────────────────────────────────── */
  h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-weight: 800 !important;
    letter-spacing: -0.3px !important;
  }
  p, li {
    color: #e2e8f0 !important;
    line-height: 1.6 !important;
  }

  /* ── Streamlit Form Labels (High Visibility) ──────────────────────────────── */
  label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label, label p {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.94rem !important;
    letter-spacing: 0.2px !important;
  }

  /* ── Code & Monospace Badges (Fixes White Box Issue) ─────────────────────── */
  code, span[data-testid="stMarkdownContainer"] code, .glass-card code, p code, li code {
    background: #090e1a !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    padding: 2px 7px !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.86em !important;
    font-weight: 600 !important;
  }

  /* ── Streamlit Header Bar (Fixes White Top Header) ────────────────────────── */
  header[data-testid="stHeader"] {
    background: rgba(8, 14, 26, 0.92) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.15) !important;
  }
  .block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1350px !important;
  }

  /* ── Hero Banner ─────────────────────────────────────────────────────────── */
  .hero-banner {
    background: linear-gradient(135deg, rgba(24, 34, 58, 0.96) 0%, rgba(15, 23, 42, 0.98) 100%);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1.5px solid rgba(56, 189, 248, 0.45);
    border-radius: 20px;
    padding: 30px 38px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.65), 0 0 35px rgba(56, 189, 248, 0.12);
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -90px; right: -90px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.25) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
  }
  .hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 45%, #ffffff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
  }
  .hero-sub {
    color: #e2e8f0 !important;
    font-size: 1.02rem;
    font-weight: 400;
    margin: 0;
    line-height: 1.7;
    max-width: 980px;
  }
  .hero-badges {
    display: flex;
    gap: 12px;
    margin-top: 16px;
    flex-wrap: wrap;
  }
  .badge {
    background: rgba(14, 116, 144, 0.3) !important;
    border: 1.5px solid #38bdf8 !important;
    color: #f0fdf4 !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    padding: 5px 14px !important;
    border-radius: 20px !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase !important;
    box-shadow: 0 2px 10px rgba(56, 189, 248, 0.2) !important;
  }

  /* ── Interactive Stepper Flow Bar ────────────────────────────────────────── */
  .workflow-bar {
    background: rgba(24, 34, 58, 0.88);
    border: 1.5px solid rgba(56, 189, 248, 0.35);
    border-radius: 14px;
    padding: 12px 20px;
    margin-bottom: 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
  }
  .step-pill {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.92rem;
    color: #ffffff;
    font-weight: 700;
  }
  .step-num {
    background: #0284c7;
    color: #ffffff;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.82rem;
    font-weight: 800;
    box-shadow: 0 2px 8px rgba(2, 132, 199, 0.4);
  }
  .step-arrow {
    color: #38bdf8;
    font-size: 1.1rem;
    font-weight: 800;
  }

  /* ── High-Contrast Glassmorphism Cards ───────────────────────────────────── */
  .glass-card {
    background: rgba(24, 34, 58, 0.88) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: 1.5px solid rgba(148, 163, 184, 0.32) !important;
    border-radius: 16px !important;
    padding: 22px 24px !important;
    margin-bottom: 18px !important;
    box-shadow: 0 10px 36px 0 rgba(0, 0, 0, 0.5) !important;
    transition: all 0.3s ease !important;
  }
  .glass-card:hover {
    border-color: rgba(56, 189, 248, 0.5) !important;
    box-shadow: 0 12px 40px 0 rgba(56, 189, 248, 0.16) !important;
  }
  .glass-card-title {
    color: #ffffff !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    margin-bottom: 12px !important;
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: -0.2px;
  }
  .glass-card p {
    color: #e2e8f0 !important;
    font-size: 0.94rem !important;
    line-height: 1.7 !important;
  }

  /* ── Framework Lineage Showcase Cards (Tab 3) ────────────────────────────── */
  .framework-card {
    background: rgba(20, 30, 52, 0.92);
    border: 1.5px solid rgba(56, 189, 248, 0.35);
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
    transition: all 0.25s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .framework-card:hover {
    transform: translateY(-2px);
    border-color: #38bdf8;
    box-shadow: 0 8px 26px rgba(56, 189, 248, 0.25);
  }
  .fw-badge {
    display: inline-block;
    font-size: 0.74rem;
    font-weight: 800;
    padding: 3px 10px;
    border-radius: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
  }
  .fw-hf { background: rgba(251, 191, 36, 0.2) !important; color: #fbbf24 !important; border: 1px solid #fbbf24 !important; }
  .fw-pt { background: rgba(239, 68, 68, 0.2) !important; color: #f87171 !important; border: 1px solid #f87171 !important; }
  .fw-on { background: rgba(56, 189, 248, 0.2) !important; color: #38bdf8 !important; border: 1px solid #38bdf8 !important; }
  .fw-sk { background: rgba(52, 211, 153, 0.2) !important; color: #34d399 !important; border: 1px solid #34d399 !important; }

  .model-name-title {
    color: #ffffff;
    font-size: 1.12rem;
    font-weight: 800;
    letter-spacing: -0.2px;
  }
  .model-meta-tag {
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 600;
  }
  .lineage-step {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 8px;
    padding: 8px 12px;
    margin: 6px 0;
  }

  /* ── Hash Display Box (High Contrast) ────────────────────────────────────── */
  .hash-box {
    background: #060911 !important;
    border: 1.5px solid #0284c7 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #38bdf8 !important;
    word-break: break-all !important;
    line-height: 1.6 !important;
    letter-spacing: 0.4px !important;
    font-weight: 600 !important;
  }
  .hash-label {
    color: #e2e8f0 !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    margin-bottom: 5px !important;
  }

  /* ── High-Contrast Threat Detection Cards (Tab 4) ────────────────────────── */
  .pass-threat-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(5, 150, 105, 0.18) 100%) !important;
    backdrop-filter: blur(16px) !important;
    border: 2.5px solid #10b981 !important;
    border-radius: 20px !important;
    padding: 30px !important;
    text-align: center !important;
    box-shadow: 0 0 45px rgba(16, 185, 129, 0.45), inset 0 0 20px rgba(16, 185, 129, 0.2) !important;
    animation: glow-green 2.5s infinite alternate ease-in-out !important;
    margin-bottom: 22px !important;
  }
  @keyframes glow-green {
    0%   { box-shadow: 0 0 30px rgba(16, 185, 129, 0.35); }
    100% { box-shadow: 0 0 55px rgba(16, 185, 129, 0.65); }
  }

  .fail-threat-card {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.3) 0%, rgba(185, 28, 28, 0.22) 100%) !important;
    backdrop-filter: blur(16px) !important;
    border: 2.5px solid #ef4444 !important;
    border-radius: 20px !important;
    padding: 30px !important;
    text-align: center !important;
    box-shadow: 0 0 50px rgba(239, 68, 68, 0.6), inset 0 0 25px rgba(239, 68, 68, 0.3) !important;
    animation: alarm-pulse 1.3s infinite alternate ease-in-out !important;
    margin-bottom: 22px !important;
  }
  @keyframes alarm-pulse {
    0%   { box-shadow: 0 0 30px rgba(239, 68, 68, 0.45); transform: scale(1); }
    100% { box-shadow: 0 0 65px rgba(239, 68, 68, 0.85); transform: scale(1.01); }
  }

  .threat-icon {
    font-size: 3.5rem;
    margin-bottom: 10px;
  }
  .threat-title-pass {
    font-size: 1.85rem;
    font-weight: 900;
    color: #34d399 !important;
    letter-spacing: -0.5px;
  }
  .threat-title-fail {
    font-size: 1.85rem;
    font-weight: 900;
    color: #f87171 !important;
    letter-spacing: -0.5px;
  }
  .threat-desc {
    color: #f8fafc !important;
    font-size: 1.02rem;
    margin-top: 10px;
    line-height: 1.7;
    max-width: 740px;
    margin-left: auto;
    margin-right: auto;
    font-weight: 500;
  }

  /* ── Event Badges & Connectors ───────────────────────────────────────────── */
  .block-event {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.6px;
    text-transform: uppercase;
  }
  .event-genesis    { background: rgba(56,189,248,0.22); color: #38bdf8; border: 1.5px solid #38bdf8; }
  .event-finetuning { background: rgba(52,211,153,0.22); color: #34d399; border: 1.5px solid #34d399; }
  .event-pruning    { background: rgba(251,191,36,0.22); color: #fbbf24; border: 1.5px solid #fbbf24; }
  .connector {
    text-align: center;
    color: #38bdf8;
    font-size: 1.5rem;
    margin: 4px 0;
    font-weight: 800;
  }

  /* ── Streamlit UI Elements Customization (High Legibility & No Clipping) ─── */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.95) !important;
    border-radius: 14px !important;
    gap: 10px !important;
    padding: 8px 12px !important;
    border: 1.5px solid rgba(148, 163, 184, 0.3) !important;
    margin-bottom: 22px !important;
    display: flex !important;
    flex-wrap: wrap !important;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 10px 22px !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
    border: 1px solid transparent !important;
  }
  .stTabs [data-baseweb="tab"]:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.06) !important;
    border-color: rgba(56, 189, 248, 0.3) !important;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(2, 132, 199, 0.35) !important;
    color: #ffffff !important;
    border: 1.5px solid #38bdf8 !important;
    box-shadow: 0 4px 18px rgba(56, 189, 248, 0.35) !important;
  }
  .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {
    display: none !important;
  }

  /* File Uploader */
  div[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.85) !important;
    border: 2px dashed #38bdf8 !important;
    border-radius: 14px !important;
    padding: 16px !important;
  }
  div[data-testid="stFileUploader"] section {
    background: transparent !important;
  }
  div[data-testid="stFileUploader"] span, div[data-testid="stFileUploader"] small, div[data-testid="stFileUploader"] p {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
  }
  div[data-testid="stFileUploader"] button {
    background: #1e293b !important;
    border: 1px solid #38bdf8 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 11px 22px !important;
    font-size: 0.95rem !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
    box-shadow: 0 4px 18px rgba(2, 132, 199, 0.4) !important;
    letter-spacing: 0.3px !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(56, 189, 248, 0.6) !important;
  }

  /* Inputs & Textareas */
  .stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background-color: #0f172a !important;
    border: 1.5px solid #334155 !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 0.94rem !important;
    font-weight: 500 !important;
  }
  .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.3) !important;
  }

  /* Selectbox / Dropdown menus */
  div[data-baseweb="select"] > div {
    background-color: #0f172a !important;
    border: 1.5px solid #334155 !important;
    border-radius: 10px !important;
    color: #ffffff !important;
  }
  div[data-baseweb="select"]:hover > div {
    border-color: #38bdf8 !important;
  }
  div[data-baseweb="select"] * {
    color: #ffffff !important;
    font-weight: 600 !important;
  }
  div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
    background: #0f172a !important;
    border: 1.5px solid #38bdf8 !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.7) !important;
  }
  ul[role="listbox"] li {
    color: #ffffff !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
  }
  ul[role="listbox"] li:hover {
    background: #1e293b !important;
    color: #38bdf8 !important;
  }

  /* Radio Buttons & Checkboxes */
  div[data-testid="stRadio"] label, div[data-testid="stCheckbox"] label,
  div[data-testid="stRadio"] span, div[data-testid="stCheckbox"] span,
  div[data-testid="stRadio"] p, div[data-testid="stCheckbox"] p {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.94rem !important;
  }

  /* Metrics */
  .stMetric {
    background: rgba(24, 34, 58, 0.92) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    border: 1.5px solid rgba(56, 189, 248, 0.35) !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
  }
  .stMetric label, .stMetric label p {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
  }
  .stMetric [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
  }

  /* ── Clean Expanders (No Text Overlap, Clean Chevron Alignment) ──────────── */
  div[data-testid="stExpander"] {
    background: rgba(20, 30, 52, 0.88) !important;
    border: 1.5px solid rgba(56, 189, 248, 0.28) !important;
    border-radius: 12px !important;
    margin-bottom: 14px !important;
    overflow: hidden !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
  }
  div[data-testid="stExpander"]:hover {
    border-color: rgba(56, 189, 248, 0.5) !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35) !important;
  }
  div[data-testid="stExpander"] summary {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    padding: 12px 18px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.96rem !important;
    cursor: pointer !important;
  }
  div[data-testid="stExpander"] summary p {
    margin: 0 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.96rem !important;
    display: inline-block !important;
  }
  div[data-testid="stExpander"] summary:hover {
    color: #38bdf8 !important;
  }
  div[data-testid="stExpanderDetails"] {
    padding: 16px 20px !important;
    background: rgba(15, 23, 42, 0.85) !important;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #080d1a !important;
    border-right: 1.5px solid rgba(148, 163, 184, 0.25) !important;
  }
  .sidebar-box {
    background: rgba(24, 34, 58, 0.9);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 14px;
    border: 1.5px solid rgba(148, 163, 184, 0.28);
    font-size: 0.88rem;
    color: #e2e8f0;
    line-height: 1.6;
  }
</style>
""", unsafe_allow_html=True)


# ─── API & Helper Utilities ───────────────────────────────────────────────────

def api_get(path: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot reach the API server. Start it with: `python -m uvicorn main:app --port 8000`")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post_multipart(path: str, data: dict, files: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", data=data, files=files, timeout=60)
        if r.status_code in (200, 201):
            return r.json()
        st.error(f"API {r.status_code}: {r.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot reach the API server. Start it with: `python -m uvicorn main:app --port 8000`")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def event_badge_html(event_type: str) -> str:
    cls = {
        "GENESIS_INGESTION":    "event-genesis",
        "FINE_TUNING":          "event-finetuning",
        "PRUNING_QUANTIZATION": "event-pruning",
    }.get(event_type, "event-genesis")
    labels = {
        "GENESIS_INGESTION":    "🧬 Genesis",
        "FINE_TUNING":          "🔧 Fine-Tuning",
        "PRUNING_QUANTIZATION": "✂️ Pruning / Quant",
    }
    return f'<span class="block-event {cls}">{labels.get(event_type, event_type)}</span>'


def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_sample_file(filename: str) -> tuple[bytes, str] | None:
    path = SAMPLE_DIR / filename
    if path.exists():
        return path.read_bytes(), filename
    return None


def tamper_model_bytes(data: bytes, injection: bytes = b"\xde\xad\xbe\xef\xca\xfe\xba\xbe") -> bytes:
    """Subtly perturbs bytes in memory to simulate a malicious backdoor or bit corruption."""
    if len(data) < 100:
        return data + injection
    mid = len(data) // 2
    return data[:mid] + injection + data[mid + len(injection):]


def detect_framework_and_icon(model_name: str) -> tuple[str, str, str]:
    """Returns (framework_name, badge_css_class, icon_emoji) based on model name."""
    m = model_name.lower()
    if "bert" in m or "transformer" in m:
        return "Hugging Face", "fw-hf", "🤗"
    elif "resnet" in m or "torch" in m:
        return "PyTorch", "fw-pt", "🔥"
    elif "yolo" in m or "onnx" in m:
        return "ONNX Runtime", "fw-on", "⚡"
    elif "forest" in m or "sklearn" in m or "scikit" in m:
        return "Scikit-Learn", "fw-sk", "🌲"
    return "AI Model", "fw-hf", "🤖"


def get_sample_file_for_model(model_name: str) -> tuple[bytes, str] | None:
    """Finds the matching sample file from sample_models directory for a given AI model name."""
    m = model_name.lower()
    if "bert-finance" in m or "adapter" in m:
        return get_sample_file("bert_finance_adapter.safetensors")
    elif "bert" in m:
        return get_sample_file("bert_base.safetensors")
    elif "resnet-50-pruned" in m or "pruned" in m:
        return get_sample_file("resnet50_pruned.pt")
    elif "resnet" in m:
        return get_sample_file("resnet50.pt")
    elif "yolov8-int8" in m or "quantized" in m:
        return get_sample_file("yolov8_int8_quantized.onnx")
    elif "yolo" in m:
        return get_sample_file("yolov8.onnx")
    elif "forest" in m or "fraud" in m:
        return get_sample_file("random_forest.pkl")
    return get_sample_file("bert_base.safetensors")


# ─── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🔐 AI Provenance & Cryptographic Audit Trail</div>
  <div class="hero-sub">
    Cryptographic verification & tamper-evident lifecycle governance for AI models —
    powered by chunked SHA-256 digests, RSA-PSS 2048-bit digital signatures, and an immutable blockchain ledger.
  </div>
  <div class="hero-badges">
    <span class="badge">🔒 SHA-256 Chunked</span>
    <span class="badge">✍️ RSA-PSS 2048-Bit</span>
    <span class="badge">⛓️ Hash-Chained Blocks</span>
    <span class="badge">🧠 Hugging Face · PyTorch · ONNX · Scikit-Learn</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Interactive Stepper Workflow Guide ───────────────────────────────────────
st.markdown("""
<div class="workflow-bar">
  <div class="step-pill"><span class="step-num">1</span> Ingest Genesis Model</div>
  <span class="step-arrow">➔</span>
  <div class="step-pill"><span class="step-num">2</span> Record Evolutionary Update</div>
  <span class="step-arrow">➔</span>
  <div class="step-pill"><span class="step-num">3</span> Audit Chain & Lineage</div>
  <span class="step-arrow">➔</span>
  <div class="step-pill"><span class="step-num">4</span> Test Tamper Defense</div>
</div>
""", unsafe_allow_html=True)

# Backend status indicator
api_ok = check_api_health()
if api_ok:
    st.markdown('<div style="text-align:right;margin-top:-20px;margin-bottom:16px;font-size:0.82rem;color:#68d391;font-weight:600;">● Backend Active (http://localhost:8000)</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="text-align:right;margin-top:-20px;margin-bottom:16px;font-size:0.82rem;color:#fc8181;font-weight:600;">● Backend Offline — run: <code>python -m uvicorn main:app --port 8000</code></div>', unsafe_allow_html=True)


# ─── Navigation Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🧬  Ingest Existing AI",
    "🔧  Log Model Update",
    "⛓️  Audit Ledger & Lineage",
    "🔍  Independent Verifier",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ingest Existing AI Model
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_info, col_action = st.columns([1, 1], gap="large")

    with col_info:
        st.markdown("""
        <div class="glass-card">
          <div class="glass-card-title">📖 Genesis Ingestion Protocol</div>
          <p style="color:#e2e8f0;font-size:0.95rem;line-height:1.7;">
            Register an existing pre-trained model artifact as the foundational <b>Genesis Block (#1)</b>
            in its verifiable provenance chain.
          </p>
          <div style="background:rgba(2,132,199,0.18);border-left:4px solid #38bdf8;padding:14px 16px;border-radius:6px;margin:14px 0;font-size:0.92rem;color:#f8fafc;">
            <b>Supported Formats:</b> PyTorch (<code>.pt</code>, <code>.bin</code>), SafeTensors (<code>.safetensors</code>), ONNX (<code>.onnx</code>), Scikit-Learn (<code>.pkl</code>), Keras (<code>.h5</code>).
          </div>
          <div style="color:#cbd5e1;font-size:0.88rem;line-height:1.7;">
            • Memory-safe chunked hashing (64 KB buffers)<br>
            • Deterministic canonical JSON hash for hyperparameter configurations<br>
            • RSA-2048 PSS asymmetric digital signature applied at registration
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("⚙️ Architecture & Hyperparameter JSON Schema", expanded=False):
            st.markdown('<p style="color:#e2e8f0;font-size:0.9rem;">Enter model configuration or hyper-parameters below. These will be hashed into the block payload:</p>', unsafe_allow_html=True)
            config_str = st.text_area(
                "Architecture Config (JSON)",
                value='{\n  "n_estimators": 50,\n  "max_depth": 10,\n  "criterion": "gini"\n}',
                height=110,
                key="t1_cfg",
            )

        with st.expander("🔐 Cryptographic Pre-Image Details", expanded=False):
            st.markdown("""
            <div style="font-size:0.88rem;color:#e2e8f0;line-height:1.7;">
              The signed payload combines:<br>
              <code style="color:#38bdf8;font-weight:700;">sha256(artifact_bytes) : sha256(canonical_config) : "0"</code><br><br>
              This binds the binary model weights directly to its architecture specifications under a digital signature.
            </div>
            """, unsafe_allow_html=True)

    with col_action:
        st.markdown('<div class="glass-card"><div class="glass-card-title">📁 Model Weights File & Details</div>', unsafe_allow_html=True)

        # Interactive One-Click Sample Presets
        ingest_mode = st.radio(
            "Select Artifact Source:",
            ["⚡ Use Pre-Loaded Demo Sample", "📤 Upload Custom File"],
            horizontal=True,
            key="t1_mode",
        )

        selected_bytes = None
        selected_filename = None

        if ingest_mode == "⚡ Use Pre-Loaded Demo Sample":
            sample_map = {
                "🤗 Hugging Face: bert_base.safetensors": ("bert_base.safetensors", "BERT-Base-v2", "HuggingFace / SafeTensors"),
                "🔥 PyTorch: resnet50.pt": ("resnet50.pt", "ResNet-50-v2", "PyTorch"),
                "⚡ ONNX: yolov8.onnx": ("yolov8.onnx", "YOLOv8-v2", "ONNX"),
                "🌲 Scikit-Learn: random_forest.pkl": ("random_forest.pkl", "RandomForest-v2", "Scikit-learn"),
            }
            sample_choice = st.selectbox(
                "Choose Demo Model:",
                list(sample_map.keys()),
                key="t1_sample_pick",
            )
            fname, default_name, default_fw = sample_map[sample_choice]
            loaded = get_sample_file(fname)
            if loaded:
                selected_bytes, selected_filename = loaded
                st.info(f"Loaded demo artifact `{selected_filename}` ({len(selected_bytes) / 1024:.1f} KB)")
        else:
            default_name = ""
            default_fw = "Scikit-learn"
            uploaded_file = st.file_uploader(
                "Upload model artifact",
                type=["pt", "pth", "bin", "safetensors", "pkl", "onnx", "h5", "pb", "tflite", "model", "zip"],
                key="tab1_file",
            )
            if uploaded_file:
                selected_bytes = uploaded_file.getvalue()
                selected_filename = uploaded_file.name
                default_name = Path(uploaded_file.name).stem

        # Real-time SHA-256 preview
        if selected_bytes:
            hasher = hashlib.sha256()
            hasher.update(selected_bytes)
            preview_hash = hasher.hexdigest()

            st.markdown(f"""
            <div style="margin:10px 0 14px 0;">
              <div class="hash-label">⚡ Real-time SHA-256 Digest Preview</div>
              <div class="hash-box">{preview_hash}</div>
            </div>
            """, unsafe_allow_html=True)

        model_name = st.text_input(
            "Model Name *",
            value=default_name,
            placeholder="e.g. BERT-Base-Uncased, Llama-3-8B",
            key="t1_name",
        )
        c_a, c_b = st.columns(2)
        with c_a:
            author = st.text_input("Author / Org", value="AI Security Team", key="t1_author")
        with c_b:
            fw_list = ["Scikit-learn", "PyTorch", "HuggingFace / SafeTensors", "ONNX", "Keras", "Other"]
            fw_idx = fw_list.index(default_fw) if default_fw in fw_list else 0
            framework = st.selectbox("Framework", fw_list, index=fw_idx, key="t1_fw")
        source_url = st.text_input("Source URL", value="https://huggingface.co/models", key="t1_url")
        description = st.text_area("Training Lineage Description", value="Baseline pre-trained model checkpoint.", height=60, key="t1_desc")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 Commit Genesis Provenance Record", key="t1_submit"):
            if not selected_bytes or not selected_filename:
                st.warning("⚠️ Please provide a model file or select a demo sample.")
            elif not model_name.strip():
                st.warning("⚠️ Model Name is required.")
            else:
                try:
                    config_parsed = json.loads(config_str)
                except json.JSONDecodeError:
                    st.error("Invalid configuration JSON syntax.")
                    st.stop()

                # Tangible multi-step progress bar
                progress_container = st.empty()
                steps = [
                    (20, "🔍 Streaming 64KB artifact chunks & computing SHA-256 digest..."),
                    (50, "🧬 Formulating canonical JSON configuration digest..."),
                    (75, "✍️ Applying RSA-PSS 2048-bit asymmetric digital signature..."),
                    (95, "⛓️ Sealing Genesis block into immutable SQLite ledger..."),
                ]
                for pct, text in steps:
                    progress_container.progress(pct, text=text)
                    time.sleep(0.3)

                metadata = {
                    "author":      author,
                    "source_url":  source_url,
                    "framework":   framework,
                    "description": description,
                }

                result = api_post_multipart(
                    "/api/ingest",
                    data={
                        "model_name":    model_name.strip(),
                        "config_json":   json.dumps(config_parsed),
                        "metadata_json": json.dumps(metadata),
                    },
                    files={"file": (selected_filename, selected_bytes, "application/octet-stream")},
                )

                progress_container.progress(100, text="✅ Genesis block successfully sealed!")
                time.sleep(0.4)
                progress_container.empty()

                if result:
                    st.toast(f"Genesis block #{result['block_id']} sealed to ledger", icon="🔐")
                    st.success("🎉 Genesis block registered with tamper-proof cryptographic proofs!")
                    st.balloons()

                    st.markdown(f"""
                    <div class="glass-card" style="border-color:rgba(72,187,120,0.45);background:rgba(16,185,129,0.04);">
                      <div class="glass-card-title">✅ Registered Genesis Block</div>
                      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                        <div>
                          <div class="hash-label">Block ID</div>
                          <div style="color:#63b3ed;font-family:'JetBrains Mono',monospace;font-size:1.4rem;font-weight:700;">#{result["block_id"]}</div>
                        </div>
                        <div>
                          <div class="hash-label">Event Type</div>
                          {event_badge_html(result["event_type"])}
                        </div>
                      </div>
                      <div style="margin-top:14px;">
                        <div class="hash-label">Artifact SHA-256 Digest</div>
                        <div class="hash-box">{result["artifact_hash"]}</div>
                      </div>
                      <div style="margin-top:10px;">
                        <div class="hash-label">Current Block Hash</div>
                        <div class="hash-box">{result["current_block_hash"]}</div>
                      </div>
                      <div style="margin-top:10px;color:#94a3b8;font-size:0.78rem;">Timestamp: {result["timestamp"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("🔍 View Raw Cryptographic Ledger Block Payload", expanded=False):
                        st.json(result)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Log Model Update / Fine-Tuning
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    ledger_data = api_get("/api/ledger")
    records = ledger_data.get("records", []) if ledger_data else []

    if not records:
        st.info("ℹ️ No models ingested yet. Please navigate to the **Ingest Existing AI** tab to create a Genesis block first.")
    else:
        col_u_info, col_u_action = st.columns([1, 1], gap="large")

        with col_u_info:
            st.markdown("""
            <div class="glass-card">
              <div class="glass-card-title">🔗 Evolution & Lineage Protocol</div>
              <p style="color:#e2e8f0;font-size:0.95rem;line-height:1.7;">
                Cryptographically link fine-tuned, pruned, or quantized models to their predecessor block.
                The new block seals the parent's hash, forming an unforgeable evolutionary chain.
              </p>
            </div>
            """, unsafe_allow_html=True)

            parent_options = {
                f"#{r['id']} — {r['model_name']} [{r['event_type']}]": r["id"]
                for r in records
            }
            parent_label = st.selectbox("Select Predecessor / Parent Block", list(parent_options.keys()), key="t2_parent")
            parent_id = parent_options[parent_label]
            parent_rec = next((r for r in records if r["id"] == parent_id), None)

            if parent_rec:
                st.markdown(f"""
                <div class="glass-card" style="border-color:rgba(56,189,248,0.4);">
                  <div class="glass-card-title">Parent Block #{parent_rec["id"]} Verification</div>
                  <div style="color:#ffffff;font-size:1.02rem;font-weight:700;">{parent_rec["model_name"]}</div>
                  <div style="margin:8px 0;">{event_badge_html(parent_rec["event_type"])}</div>
                  <div class="hash-label" style="margin-top:12px;">Parent Block Hash (Pre-image Anchor)</div>
                  <div class="hash-box" style="font-size:0.82rem;">{parent_rec["current_block_hash"]}</div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("⚙️ Fine-Tuning / Quantization Config (JSON)", expanded=False):
                config_t2 = st.text_area(
                    "Hyperparameters / Optimization Config",
                    value='{\n  "epochs": 3,\n  "learning_rate": "2e-5",\n  "batch_size": 16,\n  "optimizer": "AdamW"\n}',
                    height=120,
                    key="t2_cfg",
                )

        with col_u_action:
            st.markdown('<div class="glass-card"><div class="glass-card-title">📁 Updated Weights & Training Metadata</div>', unsafe_allow_html=True)

            u_mode = st.radio(
                "Source for Updated Artifact:",
                ["⚡ Use Pre-Loaded Fine-Tuned Sample", "📤 Upload Custom File"],
                horizontal=True,
                key="t2_mode",
            )

            u_bytes = None
            u_filename = None

            if u_mode == "⚡ Use Pre-Loaded Fine-Tuned Sample":
                u_sample_map = {
                    "🤗 Hugging Face: bert_finance_adapter.safetensors": ("bert_finance_adapter.safetensors", "BERT-Finance-Adapter", "FINE_TUNING", "Financial-PhraseBank-v2"),
                    "🔥 PyTorch: resnet50_pruned.pt": ("resnet50_pruned.pt", "ResNet-50-Pruned", "PRUNING_QUANTIZATION", "ImageNet-Val-50k"),
                    "⚡ ONNX: yolov8_int8_quantized.onnx": ("yolov8_int8_quantized.onnx", "YOLOv8-INT8-Quantized", "PRUNING_QUANTIZATION", "COCO-val2017"),
                }
                u_pick = st.selectbox("Choose Demo Evolution Artifact:", list(u_sample_map.keys()), key="t2_sample_pick")
                u_fname, u_def_name, u_def_ev, u_def_ds = u_sample_map[u_pick]
                loaded2 = get_sample_file(u_fname)
                if loaded2:
                    u_bytes, u_filename = loaded2
                    st.info(f"Loaded demo evolution artifact `{u_filename}` ({len(u_bytes) / 1024:.1f} KB)")
            else:
                u_def_name = ""
                u_def_ev = "FINE_TUNING"
                u_def_ds = "Custom-Dataset-v1"
                updated_file = st.file_uploader(
                    "Upload updated / fine-tuned model artifact",
                    type=["pt", "pth", "bin", "safetensors", "pkl", "onnx", "h5", "model", "zip"],
                    key="tab2_file",
                )
                if updated_file:
                    u_bytes = updated_file.getvalue()
                    u_filename = updated_file.name

            if u_bytes:
                hasher2 = hashlib.sha256()
                hasher2.update(u_bytes)
                preview_hash2 = hasher2.hexdigest()
                st.markdown(f"""
                <div style="margin:10px 0;">
                  <div class="hash-label">⚡ Updated Artifact SHA-256 Preview</div>
                  <div class="hash-box">{preview_hash2}</div>
                </div>
                """, unsafe_allow_html=True)

            ev_list = ["FINE_TUNING", "PRUNING_QUANTIZATION"]
            ev_idx = ev_list.index(u_def_ev) if u_def_ev in ev_list else 0
            event_type = st.selectbox("Evolution Event Type", ev_list, index=ev_idx, key="t2_event")
            model_name_t2 = st.text_input(
                "New Model Name (blank to inherit)",
                value=u_def_name,
                placeholder="e.g. BERT-FineTuned-SQuAD",
                key="t2_name",
            )
            dataset = st.text_input("Dataset Lineage", value=u_def_ds, key="t2_ds")
            notes = st.text_area("Audit Notes", value="Simulated evolution run with updated hyperparameters.", height=60, key="t2_notes")
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("🔗 Append Evolutionary Block to Chain", key="t2_submit"):
                if not u_bytes or not u_filename:
                    st.warning("⚠️ Please provide an updated model file or select a sample.")
                else:
                    try:
                        config_parsed_t2 = json.loads(config_t2)
                    except json.JSONDecodeError:
                        st.error("Invalid config JSON syntax.")
                        st.stop()

                    prog2 = st.empty()
                    for pct, msg in [
                        (25, "🔍 Computing SHA-256 on updated weights..."),
                        (55, "🔗 Anchoring previous block cryptographic hash..."),
                        (80, "✍️ Generating RSA-PSS 2048-bit digital signature..."),
                        (95, "⛓️ Committing evolutionary block to ledger..."),
                    ]:
                        prog2.progress(pct, text=msg)
                        time.sleep(0.3)

                    metadata_t2 = {
                        "dataset": dataset,
                        "notes":   notes,
                    }

                    result2 = api_post_multipart(
                        "/api/log-update",
                        data={
                            "parent_block_id": parent_id,
                            "event_type":      event_type,
                            "model_name":      model_name_t2.strip(),
                            "config_json":     json.dumps(config_parsed_t2),
                            "metadata_json":   json.dumps(metadata_t2),
                        },
                        files={"file": (u_filename, u_bytes, "application/octet-stream")},
                    )

                    prog2.progress(100, text="✅ Block chained successfully!")
                    time.sleep(0.4)
                    prog2.empty()

                    if result2:
                        st.toast(f"Evolution block #{result2['block_id']} chained to #{parent_id}", icon="🔗")
                        st.success(f"✅ Block #{result2['block_id']} successfully linked to Parent #{parent_id}!")

                        st.markdown(f"""
                        <div class="glass-card" style="border-color:rgba(154,230,180,0.4);background:rgba(154,230,180,0.03);">
                          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                            <div>
                              <div class="hash-label">New Block ID</div>
                              <div style="color:#68d391;font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;">#{result2["block_id"]}</div>
                            </div>
                            <div>
                              <div class="hash-label">Parent Block ID</div>
                              <div style="color:#63b3ed;font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;">#{parent_id}</div>
                            </div>
                          </div>
                          <div style="margin-top:12px;">
                            <div class="hash-label">New Artifact Hash</div>
                            <div class="hash-box">{result2["artifact_hash"]}</div>
                          </div>
                          <div style="margin-top:8px;">
                            <div class="hash-label">Previous Block Hash (Parent Seal)</div>
                            <div class="hash-box">{result2["previous_block_hash"]}</div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Audit Ledger & Lineage
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_t3_header, col_t3_actions = st.columns([2, 1])
    with col_t3_header:
        st.markdown("### ⛓️ Cryptographic Audit Ledger & Lineage Explorer")
        st.markdown('<p style="color:#e2e8f0;font-size:0.95rem;">Searchable blockchain explorer verifying block integrity, hash linkage, and evolutionary lineages for all AI Models.</p>', unsafe_allow_html=True)
    with col_t3_actions:
        b1, b2 = st.columns(2)
        with b1:
            refresh_clicked = st.button("🔄 Refresh Ledger", key="t3_refresh")
        with b2:
            verify_chain_clicked = st.button("🛡️ Audit Entire Chain", key="t3_verify")

    ledger3 = api_get("/api/ledger")

    if ledger3:
        records3   = ledger3.get("records", [])
        integrity3 = ledger3.get("integrity", {})
        is_valid   = ledger3.get("is_chain_valid", False)
        total      = ledger3.get("total_blocks", 0)

        # Top Executive Metrics Bar
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Ledger Blocks", total)
        with m2:
            genesis_count = sum(1 for r in records3 if r.get("event_type") == "GENESIS_INGESTION")
            st.metric("Genesis Lineages", genesis_count)
        with m3:
            update_count = sum(1 for r in records3 if r.get("event_type") != "GENESIS_INGESTION")
            st.metric("Evolution Events", update_count)
        with m4:
            chain_status = "✅ INTACT (100% VALID)" if is_valid else "❌ TAMPER DETECTED"
            st.metric("Blockchain Status", chain_status)

        if verify_chain_clicked and integrity3:
            st.markdown("<br>", unsafe_allow_html=True)
            if is_valid:
                st.markdown("""
                <div style="background:rgba(16,185,129,0.12);border:1.5px solid rgba(16,185,129,0.5);border-radius:12px;padding:16px 20px;color:#34d399;font-weight:700;">
                  🔒 Full Chain Verification Passed — Every block hash, parent linkage, and RSA digital signature is mathematically valid.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:rgba(239,68,68,0.16);border:1.5px solid rgba(239,68,68,0.6);border-radius:12px;padding:16px 20px;color:#f87171;font-weight:700;">
                  🚨 Chain Integrity Breach Detected — One or more blocks have been compromised or modified post-registration!
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ─── Framework AI Models Showcase (4 Independent Trees) ───────────────
        st.markdown("#### 🧠 AI Model Framework Lineages (4 Independent Trees)")
        fw_c1, fw_c2, fw_c3, fw_c4 = st.columns(4)

        with fw_c1:
            st.markdown("""
            <div class="framework-card">
              <div>
                <span class="fw-badge fw-hf">🤗 Hugging Face</span>
                <div class="model-name-title">BERT-Base</div>
                <div class="model-meta-tag">Genesis Block #1 · NLP Transformer</div>
                <div class="lineage-step">
                  <div style="color:#94a3b8;font-size:0.75rem;font-weight:700;">⬇ FINE-TUNING CHECKPOINT</div>
                  <div style="color:#ffffff;font-size:0.92rem;font-weight:700;">BERT-Finance-Adapter</div>
                  <div style="color:#38bdf8;font-size:0.75rem;font-weight:700;">Block #2 (Chained to #1)</div>
                </div>
              </div>
              <div style="margin-top:10px;font-size:0.78rem;color:#cbd5e1;">Artifact: <code>bert_base.safetensors</code></div>
            </div>
            """, unsafe_allow_html=True)

        with fw_c2:
            st.markdown("""
            <div class="framework-card">
              <div>
                <span class="fw-badge fw-pt">🔥 PyTorch</span>
                <div class="model-name-title">ResNet-50</div>
                <div class="model-meta-tag">Genesis Block #3 · Deep ConvNet</div>
                <div class="lineage-step">
                  <div style="color:#94a3b8;font-size:0.75rem;font-weight:700;">⬇ PRUNING & OPTIMIZATION</div>
                  <div style="color:#ffffff;font-size:0.92rem;font-weight:700;">ResNet-50-Pruned</div>
                  <div style="color:#38bdf8;font-size:0.75rem;font-weight:700;">Block #4 (Chained to #3)</div>
                </div>
              </div>
              <div style="margin-top:10px;font-size:0.78rem;color:#cbd5e1;">Artifact: <code>resnet50.pt</code></div>
            </div>
            """, unsafe_allow_html=True)

        with fw_c3:
            st.markdown("""
            <div class="framework-card">
              <div>
                <span class="fw-badge fw-on">⚡ ONNX Runtime</span>
                <div class="model-name-title">YOLOv8</div>
                <div class="model-meta-tag">Genesis Block #5 · Object Detector</div>
                <div class="lineage-step">
                  <div style="color:#94a3b8;font-size:0.75rem;font-weight:700;">⬇ INT8 QUANTIZATION</div>
                  <div style="color:#ffffff;font-size:0.92rem;font-weight:700;">YOLOv8-INT8-Quantized</div>
                  <div style="color:#38bdf8;font-size:0.75rem;font-weight:700;">Block #6 (Chained to #5)</div>
                </div>
              </div>
              <div style="margin-top:10px;font-size:0.78rem;color:#cbd5e1;">Artifact: <code>yolov8.onnx</code></div>
            </div>
            """, unsafe_allow_html=True)

        with fw_c4:
            st.markdown("""
            <div class="framework-card">
              <div>
                <span class="fw-badge fw-sk">🌲 Scikit-Learn</span>
                <div class="model-name-title">RandomForest-FraudDetection</div>
                <div class="model-meta-tag">Genesis Block #7 · Ensemble Trees</div>
                <div class="lineage-step">
                  <div style="color:#94a3b8;font-size:0.75rem;font-weight:700;">STANDALONE GENESIS</div>
                  <div style="color:#ffffff;font-size:0.92rem;font-weight:700;">Production Baseline</div>
                  <div style="color:#34d399;font-size:0.75rem;font-weight:700;">Root Ingestion</div>
                </div>
              </div>
              <div style="margin-top:10px;font-size:0.78rem;color:#cbd5e1;">Artifact: <code>random_forest.pkl</code></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Interactive Filter & Search Box
        c_filter, c_search = st.columns([1, 2])
        with c_filter:
            event_filter = st.selectbox("Filter by Event:", ["All Events", "GENESIS_INGESTION", "FINE_TUNING", "PRUNING_QUANTIZATION"], key="t3_filter")
        with c_search:
            search_query = st.text_input("🔍 Search by AI Model Name or Block ID:", "", placeholder="Type model name e.g. BERT, ResNet, YOLO...", key="t3_search")

        filtered_records = records3
        if event_filter != "All Events":
            filtered_records = [r for r in filtered_records if r["event_type"] == event_filter]
        if search_query.strip():
            q = search_query.strip().lower()
            filtered_records = [r for r in filtered_records if q in str(r["id"]) or q in r["model_name"].lower()]

        if filtered_records:
            st.markdown(f"#### 📦 Blockchain Visual Trail ({len(filtered_records)} Blocks Audited)")
            for i, rec in enumerate(filtered_records):
                res_info = next((r for r in integrity3.get("results", []) if r["block_id"] == rec["id"]), None)
                block_ok = res_info["ok"] if res_info else True
                tamper_icon = "🟢" if block_ok else "🔴"
                fw_name, fw_cls, fw_icon = detect_framework_and_icon(rec["model_name"])

                with st.expander(f"{tamper_icon} Block #{rec['id']} — 🤖 AI Model: {rec['model_name']} [{fw_icon} {fw_name}] · {rec['timestamp'][:19]}Z", expanded=(i == 0)):
                    # Prominent AI Model Name Banner inside the block
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg, rgba(2,132,199,0.2) 0%, rgba(15,23,42,0.7) 100%);border:1.5px solid #0284c7;border-radius:12px;padding:12px 18px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
                      <div>
                        <span style="color:#94a3b8;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;">Audited AI Model:</span>
                        <div style="color:#ffffff;font-size:1.3rem;font-weight:800;letter-spacing:-0.2px;">🤖 AI Model: {rec['model_name']}</div>
                      </div>
                      <div style="text-align:right;">
                        <span class="fw-badge {fw_cls}" style="margin-right:6px;">{fw_icon} {fw_name}</span>
                        <span style="background:#0284c7;color:#fff;padding:4px 12px;border-radius:14px;font-size:0.8rem;font-weight:700;">Block #{rec['id']}</span>
                        <span style="margin-left:6px;">{event_badge_html(rec['event_type'])}</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f'<div class="hash-label">Artifact SHA-256 Digest</div><div class="hash-box">{rec["artifact_hash"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="hash-label" style="margin-top:10px;">Config JSON Hash</div><div class="hash-box">{rec["config_hash"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="hash-label" style="margin-top:10px;">Previous Block Hash (Parent Anchor)</div><div class="hash-box">{rec["previous_block_hash"]}</div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="hash-label">Current Block Hash</div><div class="hash-box">{rec["current_block_hash"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="hash-label" style="margin-top:10px;">Digital Signature (RSA-PSS 2048)</div><div class="hash-box" style="font-size:0.62rem;">{rec["digital_signature"][:120]}…</div>', unsafe_allow_html=True)
                        if rec.get("parent_block_id"):
                            parent_rec_find = next((r for r in records3 if r["id"] == rec["parent_block_id"]), None)
                            parent_model_disp = f" (AI Model: {parent_rec_find['model_name']})" if parent_rec_find else ""
                            st.markdown(f'<div style="margin-top:10px;color:#68d391;font-size:0.82rem;font-weight:700;">🔗 Chained to Predecessor: <b>Block #{rec["parent_block_id"]}</b>{parent_model_disp}</div>', unsafe_allow_html=True)

                    try:
                        meta = json.loads(rec.get("metadata_json", "{}"))
                        if meta:
                            st.markdown('<div class="hash-label" style="margin-top:12px;">Metadata Payload</div>', unsafe_allow_html=True)
                            st.json(meta)
                    except Exception:
                        pass

                if i < len(filtered_records) - 1:
                    st.markdown('<div class="connector">⬇</div>', unsafe_allow_html=True)

            # Interactive Model Lineage Hierarchy
            st.markdown("<br>#### 🌳 Cryptographic Model Lineage Hierarchy", unsafe_allow_html=True)
            genesis_blocks = [r for r in records3 if r.get("parent_block_id") is None]
            for genesis in genesis_blocks:
                fw_name_g, fw_cls_g, fw_icon_g = detect_framework_and_icon(genesis["model_name"])
                children = [r for r in records3 if r.get("parent_block_id") == genesis["id"]]

                st.markdown(f"""
                <div class="glass-card" style="padding:16px 20px;margin-bottom:12px;">
                  <div style="font-size:1.02rem;color:#ffffff;font-weight:700;">
                    🧬 <b>Genesis AI Model:</b> <span style="color:#38bdf8;">{genesis['model_name']}</span>
                    <span class="fw-badge {fw_cls_g}" style="margin-left:8px;">{fw_icon_g} {fw_name_g}</span>
                    <span style="color:#94a3b8;font-size:0.82rem;">(Block #{genesis['id']} · {genesis['timestamp'][:10]})</span>
                  </div>
                """, unsafe_allow_html=True)

                for child in children:
                    fw_name_c, fw_cls_c, fw_icon_c = detect_framework_and_icon(child["model_name"])
                    st.markdown(f"""
                    <div style="margin-left:26px;padding:6px 0 6px 14px;border-left:2px dashed #0284c7;font-size:0.95rem;color:#f8fafc;">
                      └── 🔧 <b>Evolved AI Model:</b> <span style="color:#34d399;font-weight:700;">{child['model_name']}</span>
                      <span class="fw-badge {fw_cls_c}" style="margin-left:6px;">{fw_icon_c} {fw_name_c}</span>
                      <span style="color:#cbd5e1;font-size:0.82rem;font-weight:600;">[{child['event_type']}] · Block #{child['id']}</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — High-Contrast Threat Detection & Independent Verifier
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    col_v_info, col_v_action = st.columns([1, 1], gap="large")

    ledger4 = api_get("/api/ledger")
    records4 = ledger4.get("records", []) if ledger4 else []

    with col_v_info:
        st.markdown("""
        <div class="glass-card">
          <div class="glass-card-title">🛡️ Zero-Trust Verification Protocol</div>
          <p style="color:#e2e8f0;font-size:0.95rem;line-height:1.7;">
            Anyone can independently verify a candidate AI model file against any block in the immutable ledger.
          </p>
          <div style="background:rgba(2,132,199,0.18);border-left:4px solid #38bdf8;padding:12px 14px;border-radius:6px;margin:12px 0;font-size:0.92rem;color:#f8fafc;">
            <b>Cryptographic Checks Performed:</b><br>
            1. <b>Byte Integrity:</b> Computes SHA-256 in 64 KB buffers and matches against the ledger.<br>
            2. <b>Signature Authenticity:</b> Mathematically validates RSA-2048 PSS signature with public key.
          </div>
        </div>
        """, unsafe_allow_html=True)

        if not records4:
            st.info("ℹ️ No models found in ledger. Please ingest a model in Tab 1 first.")
            target_rec = None
            target_block_id = 1
            target_model_name = "BERT-Base"
        else:
            # Intuitive Dropdown with AI Model Name front and center
            options_map = {}
            for r in records4:
                fw_name, _, fw_icon = detect_framework_and_icon(r["model_name"])
                label = f"Block #{r['id']} — 🤖 AI Model: {r['model_name']} ({fw_icon} {fw_name} · {r['event_type']})"
                options_map[label] = r

            selected_label = st.selectbox(
                "🎯 Select AI Model to Audit & Verify:",
                list(options_map.keys()),
                key="t4_model_select"
            )
            target_rec = options_map[selected_label]
            target_block_id = target_rec["id"]
            target_model_name = target_rec["model_name"]
            fw_name, fw_cls, fw_icon = detect_framework_and_icon(target_model_name)

            # Prominent Profile Card of the selected AI Model
            st.markdown(f"""
            <div class="glass-card" style="border-color:rgba(56,189,248,0.5);">
              <div class="glass-card-title">🎯 Audited AI Model Profile</div>
              <div style="font-size:1.35rem;font-weight:800;color:#ffffff;margin-bottom:8px;">
                🤖 AI Model: {target_model_name}
              </div>
              <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;align-items:center;">
                <span class="fw-badge {fw_cls}">{fw_icon} {fw_name}</span>
                <span style="background:#0284c7;color:#fff;padding:3px 10px;border-radius:12px;font-size:0.75rem;font-weight:700;">Block #{target_block_id}</span>
                {event_badge_html(target_rec["event_type"])}
              </div>
              <div class="hash-label">Registered Artifact SHA-256 Digest</div>
              <div class="hash-box" style="font-size:0.82rem;">{target_rec["artifact_hash"]}</div>
              <div class="hash-label" style="margin-top:10px;">Sealed Block Hash</div>
              <div class="hash-box" style="font-size:0.82rem;">{target_rec["current_block_hash"]}</div>
              <div style="color:#cbd5e1;font-size:0.82rem;margin-top:10px;">
                Sealed Timestamp: {target_rec["timestamp"][:19]}Z
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_v_action:
        st.markdown('<div class="glass-card"><div class="glass-card-title">🔬 Candidate Model Artifact & Tamper Playground</div>', unsafe_allow_html=True)

        # Interactive Test Mode Selector
        test_mode = st.radio(
            f"Select Test Scenario for AI Model '{target_model_name}':",
            [
                f"🟢 Authentic Model Test (Load Genuine Weights for {target_model_name})",
                f"🔴 Simulated Weight Tampering / Poisoning Attack (Corrupt Bytes in {target_model_name})",
                "📤 Upload Custom Candidate Model File",
            ],
            key="t4_mode",
        )

        candidate_bytes = None
        candidate_filename = None

        if "Authentic Model Test" in test_mode:
            sample_candidate = get_sample_file_for_model(target_model_name)
            if sample_candidate:
                candidate_bytes, candidate_filename = sample_candidate
                st.success(f"🟢 Loaded Authentic Weights for AI Model: **{target_model_name}** (`{candidate_filename}` · {len(candidate_bytes)/1024:.1f} KB).")
        elif "Simulated Weight Tampering" in test_mode:
            sample_candidate = get_sample_file_for_model(target_model_name)
            if sample_candidate:
                orig_bytes, candidate_filename = sample_candidate
                candidate_bytes = tamper_model_bytes(orig_bytes)
                st.error(f"🚨 ATTACK SIMULATOR: Injected 8 altered bytes into AI Model: **{target_model_name}** (`{candidate_filename}`). Simulating malicious backdoor injection!")
        else:
            verify_file = st.file_uploader(
                f"Upload candidate file to verify against AI Model '{target_model_name}'",
                type=["pt", "pth", "bin", "safetensors", "pkl", "onnx", "h5", "model", "zip"],
                key="tab4_file",
            )
            if verify_file:
                candidate_bytes = verify_file.getvalue()
                candidate_filename = verify_file.name

        if candidate_bytes:
            hasher_v = hashlib.sha256()
            hasher_v.update(candidate_bytes)
            preview_hash_v = hasher_v.hexdigest()
            st.markdown(f"""
            <div style="margin:10px 0;">
              <div class="hash-label">⚡ Candidate Artifact SHA-256 Computed Preview</div>
              <div class="hash-box">{preview_hash_v}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button(f"🔬 Execute Cryptographic Verification for '{target_model_name}'", key="t4_verify"):
            if not candidate_bytes or not candidate_filename:
                st.warning("⚠️ Please provide a candidate model file or select a test scenario.")
            else:
                with st.spinner(f"Computing SHA-256 for AI Model '{target_model_name}' and validating RSA-PSS signature..."):
                    time.sleep(0.5)
                    vresult = api_post_multipart(
                        "/api/verify",
                        data={"block_id": int(target_block_id)},
                        files={"file": (candidate_filename, candidate_bytes, "application/octet-stream")},
                    )

                if vresult:
                    is_authentic = vresult.get("is_authentic", False)
                    hash_matches = vresult.get("hash_matches", False)
                    sig_valid    = vresult.get("signature_valid", False)

                    # DRAMATIC HIGH-CONTRAST THREAT DETECTION WITH AI MODEL NAME
                    if is_authentic:
                        st.toast(f"Verification Passed: {target_model_name} is 100% Authentic", icon="✅")
                        st.markdown(f"""
                        <div class="pass-threat-card">
                          <div class="threat-icon">🛡️</div>
                          <div class="threat-title-pass">INTEGRITY INTACT: 🤖 {target_model_name} VERIFIED</div>
                          <div class="threat-desc">
                            Every single byte of AI Model <b>{target_model_name}</b> matches the cryptographic SHA-256 fingerprint registered in Ledger Block #{target_block_id}.
                            The RSA-PSS 2048-bit digital signature is authentic with ZERO modifications. Safe for production deployment!
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.toast(f"SECURITY ALERT: Tampering Detected in {target_model_name}!", icon="🚨")
                        st.markdown(f"""
                        <div class="fail-threat-card">
                          <div class="threat-icon">🚨</div>
                          <div class="threat-title-fail">TAMPER DETECTED: ⚠️ {target_model_name} COMPROMISED</div>
                          <div class="threat-desc">
                            CRITICAL SECURITY BREACH: The candidate weights for AI Model <b>{target_model_name}</b> do NOT match the cryptographic fingerprint
                            recorded in Ledger Block #{target_block_id}! The computed SHA-256 digest diverged from the immutable ledger seal. High risk of backdoor or weight poisoning.
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Comparison metric cards
                    k1, k2, k3 = st.columns(3)
                    with k1:
                        hash_badge = "✅ MATCH" if hash_matches else "❌ MISMATCH"
                        h_col = "#34d399" if hash_matches else "#f87171"
                        st.markdown(f"""
                        <div class="glass-card" style="text-align:center;padding:16px;">
                          <div class="hash-label">Byte Hash Check</div>
                          <div style="color:{h_col};font-size:1.15rem;font-weight:800;margin-top:6px;">{hash_badge}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with k2:
                        sig_badge = "✅ VALID" if sig_valid else "❌ INVALID"
                        s_col = "#34d399" if sig_valid else "#f87171"
                        st.markdown(f"""
                        <div class="glass-card" style="text-align:center;padding:16px;">
                          <div class="hash-label">RSA-PSS Signature</div>
                          <div style="color:{s_col};font-size:1.15rem;font-weight:800;margin-top:6px;">{sig_badge}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with k3:
                        auth_badge = "🟢 AUTHENTIC" if is_authentic else "🔴 COMPROMISED"
                        a_col = "#34d399" if is_authentic else "#f87171"
                        st.markdown(f"""
                        <div class="glass-card" style="text-align:center;padding:16px;">
                          <div class="hash-label">Final Audit Verdict</div>
                          <div style="color:{a_col};font-size:1.15rem;font-weight:800;margin-top:6px;">{auth_badge}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Detailed Hash Comparison Box featuring the Model Name
                    st.markdown(f"""
                    <div class="glass-card">
                      <div class="glass-card-title">🔍 Byte-by-Byte Digest Audit for AI Model: {target_model_name}</div>
                      <div class="hash-label">Computed Digest (Candidate Artifact)</div>
                      <div class="hash-box" style="color:#f1f5f9;">{vresult.get("computed_hash","—")}</div>
                      <div class="hash-label" style="margin-top:12px;">Recorded Digest (Ledger Block #{vresult.get("block_id")})</div>
                      <div class="hash-box" style="{'color:#34d399;border-color:rgba(16,185,129,0.5);' if hash_matches else 'color:#f87171;border-color:rgba(239,68,68,0.6);'}">
                        {vresult.get("stored_hash","—")}
                      </div>
                      <div style="margin-top:12px;font-size:0.92rem;color:#e2e8f0;">
                        Audited AI Model: <b style="color:#ffffff;font-size:1.05rem;">🤖 {target_model_name}</b> (Block #{vresult.get("block_id")})
                      </div>
                    </div>
                    """, unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 20px;">
      <div style="font-size:2.8rem;">🔐</div>
      <div style="font-size:1.2rem;font-weight:800;color:#ffffff;margin-top:6px;letter-spacing:0.5px;">AI PROVENANCE</div>
      <div style="font-size:0.82rem;color:#cbd5e1;font-weight:500;">Cryptographic Audit Ledger</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-box">
      <b style="color:#ffffff;">API Endpoint:</b><br>
      <code style="color:#38bdf8;font-size:0.82rem;font-weight:700;">{API_BASE}</code>
    </div>
    """, unsafe_allow_html=True)

    ledger_s = api_get("/api/ledger")
    if ledger_s:
        recs_s = ledger_s.get("records", [])
        valid_s = ledger_s.get("is_chain_valid", False)
        st.markdown(f"""
        <div class="sidebar-box">
          <b style="color:#ffffff;">Ledger Blocks:</b> <span style="color:#38bdf8;font-weight:800;font-size:1.05rem;">{len(recs_s)}</span><br>
          <b style="color:#ffffff;">Chain Health:</b> <span style="color:{'#34d399' if valid_s else '#f87171'};font-weight:800;font-size:1.02rem;">{'✅ Valid' if valid_s else '❌ Compromised'}</span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔑 Active Public Key (RSA 2048)", expanded=False):
        try:
            pub_path = Path("public_key.pem")
            if pub_path.exists():
                st.code(pub_path.read_text(), language="text")
            else:
                st.caption("Key will be auto-generated at startup.")
        except Exception as e:
            st.caption(f"Error reading key: {e}")

    st.markdown("""
    <div class="sidebar-box" style="margin-top:14px;">
      <b style="color:#ffffff;font-size:0.92rem;">Interactive Quick Guide:</b><br><br>
      1. Tab 1: Click <b>⚡ Use Pre-Loaded Demo Sample</b> & commit Genesis.<br>
      2. Tab 2: Choose Parent Block & append Fine-Tuned checkpoint.<br>
      3. Tab 3: Inspect chain & run Audit.<br>
      4. Tab 4: Try <b>🔴 Simulated Attack</b> to witness tamper alarms!
    </div>
    """, unsafe_allow_html=True)
