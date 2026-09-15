"""
Streamlit frontend for the Autonomous AI Data Analyst.

Talks only to the FastAPI backend over HTTP (no direct imports of
the agent/data-processing layers) so the UI and API can be deployed
and scaled independently.
"""
from __future__ import annotations

import os
import sys

import plotly.io as pio
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.ui.insights import (  # noqa: E402
    SUGGESTED_QUESTIONS,
    anomaly_narrative,
    column_flags,
    correlation_narrative,
    data_quality_score,
    overview_summary,
)

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
CHART_COLORWAY = ["#5549AC", "#00B894", "#FDCB6E", "#74B9FF", "#E17055", "#A29BFE", "#00CEC9", "#FD79A8"]

st.set_page_config(
    page_title="Autonomous AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #FAFAFF 0%, #F3F1FD 100%); }

    .hero-banner {
        background: linear-gradient(120deg, #6C5CE7 0%, #A29BFE 45%, #74B9FF 100%);
        padding: 2rem 2.2rem; border-radius: 18px; margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(108, 92, 231, 0.25);
    }
    .hero-banner h1 { color: white; font-size: 2rem; margin: 0; font-weight: 800; }
    .hero-banner p { color: rgba(255,255,255,0.92); margin: 0.4rem 0 0 0; font-size: 1.02rem; }

    div[data-testid="stMetric"] {
        background: white; border-radius: 14px; padding: 1rem 1.1rem;
        box-shadow: 0 2px 10px rgba(108, 92, 231, 0.10); border: 1px solid #EDEBFB;
    }
    div[data-testid="stMetricValue"] { color: #6C5CE7; font-weight: 800; }

    button[data-baseweb="tab"] { font-weight: 600; font-size: 0.98rem; }
    div[data-baseweb="tab-list"] { gap: 4px; }

    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #F3F1FD 0%, #EAE6FB 100%); }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #4834D4; }

    .stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; border: none; }
    .stButton > button[kind="primary"] { background: linear-gradient(120deg, #6C5CE7, #A29BFE); }

    div[data-testid="stChatMessage"] { border-radius: 14px; padding: 0.25rem; }

    .section-badge {
        display: inline-block; background: #EDEBFB; color: #4834D4;
        padding: 0.15rem 0.7rem; border-radius: 999px; font-size: 0.78rem;
        font-weight: 700; letter-spacing: 0.02em; margin-bottom: 0.5rem;
    }

    .insight-card {
        background: white; border-left: 4px solid #6C5CE7; border-radius: 10px;
        padding: 0.9rem 1.1rem; margin-bottom: 0.7rem; box-shadow: 0 2px 8px rgba(108,92,231,0.08);
        font-size: 0.95rem; line-height: 1.5;
    }
    .flag-card {
        background: #FFF9EC; border-left: 4px solid #FDCB6E; border-radius: 10px;
        padding: 0.7rem 1rem; margin-bottom: 0.5rem; font-size: 0.92rem;
    }
    .quality-ring {
        text-align: center; padding: 1.2rem; background: white; border-radius: 16px;
        box-shadow: 0 2px 10px rgba(108,92,231,0.10);
    }
</style>
""", unsafe_allow_html=True)


def _themed(fig):
    fig.update_layout(
        colorway=CHART_COLORWAY,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1E1B2E"),
        margin=dict(t=50, b=30, l=20, r=20),
    )
    return fig


if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "upload_result" not in st.session_state:
    st.session_state.upload_result = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

st.markdown("""
<div class="hero-banner">
    <h1>🤖 Autonomous AI Data Analyst</h1>
    <p>Upload a dataset, get instant automated insights, and ask it questions in plain English.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📁 1. Upload dataset")
    uploaded_file = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls", "tsv"])

    if uploaded_file is not None and st.button("✨ Analyze dataset", type="primary", use_container_width=True):
        with st.spinner("Uploading and running automatic EDA..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                resp = requests.post(f"{API_BASE}/api/upload", files=files, timeout=120)
                resp.raise_for_status()
                st.session_state.upload_result = resp.json()
                st.session_state.session_id = st.session_state.upload_result["session_id"]
                st.session_state.chat_history = []
                st.success("Dataset loaded and analyzed.")
            except requests.exceptions.RequestException as exc:
                st.error(f"Upload failed: {exc}")

    if st.session_state.session_id:
        st.divider()
        st.markdown("### 📄 2. Download report")
        if st.button("Generate report", use_container_width=True):
            with st.spinner("Building report..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/api/report",
                        json={"session_id": st.session_state.session_id},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    st.download_button(
                        "⬇️ Download analysis_report.md",
                        data=resp.content,
                        file_name="analysis_report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                except requests.exceptions.RequestException as exc:
                    st.error(f"Report generation failed: {exc}")

        st.divider()
        st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
        st.caption(f"File: **{st.session_state.upload_result.get('filename', '—')}**")

if not st.session_state.upload_result:
    st.info("👈 Upload a CSV or Excel file from the sidebar to get started.")
    st.stop()

result = st.session_state.upload_result
profile = result["profile"]
anomalies = result.get("anomalies", {})

tab_overview, tab_charts, tab_anomalies, tab_chat = st.tabs(
    ["📊  Overview", "📈  Charts", "🚨  Anomalies", "💬  Ask the Analyst"]
)

# ---------------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------------
with tab_overview:
    score, verdict = data_quality_score(profile)
    col_score, col_metrics = st.columns([1, 3])

    with col_score:
        st.markdown('<div class="quality-ring">', unsafe_allow_html=True)
        st.metric("Data Quality Score", f"{score}/100")
        st.caption(verdict)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{profile['n_rows']:,}")
        c2.metric("Columns", profile["n_cols"])
        c3.metric("Duplicate rows", profile["duplicate_rows"])
        c4.metric("Missing cells", profile["total_missing_cells"])

    st.markdown('<span class="section-badge">WHAT THIS DATA LOOKS LIKE</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-card">{overview_summary(profile)}</div>', unsafe_allow_html=True)

    corr_text = correlation_narrative(profile)
    if corr_text:
        st.markdown(f'<div class="insight-card">🔗 {corr_text}</div>', unsafe_allow_html=True)

    flags = column_flags(profile)
    if flags:
        st.markdown('<span class="section-badge">WORTH A CLOSER LOOK</span>', unsafe_allow_html=True)
        for flag in flags:
            st.markdown(f'<div class="flag-card">{flag}</div>', unsafe_allow_html=True)

    st.markdown('<span class="section-badge">COLUMN DETAILS</span>', unsafe_allow_html=True)
    st.dataframe(profile["columns"], use_container_width=True)

    strong_pairs = profile.get("correlations", {}).get("strong_pairs", [])
    if strong_pairs:
        st.markdown('<span class="section-badge">ALL NOTABLE CORRELATIONS</span>', unsafe_allow_html=True)
        st.dataframe(strong_pairs, use_container_width=True)

# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------
with tab_charts:
    charts = result.get("charts", [])
    if not charts:
        st.info("No automatic charts were generated for this dataset.")
    else:
        st.markdown(
            '<div class="insight-card">These charts were picked automatically: distributions for the '
            'first few numeric columns, a correlation heatmap if there are enough numeric columns, '
            'the top categories of the leading categorical column, and a scatter plot of the two most '
            'correlated numeric columns.</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, chart in enumerate(charts):
            fig = _themed(pio.from_json(chart["figure_json"]))
            with cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# ANOMALIES
# ---------------------------------------------------------------------------
with tab_anomalies:
    if "error" in anomalies:
        st.info(anomalies["error"])
    else:
        st.markdown(f'<div class="insight-card">🔍 {anomaly_narrative(anomalies)}</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("🚨 Flagged rows", anomalies["n_anomalies"])
        c2.metric("Anomaly %", f"{anomalies['anomaly_pct']}%")
        c3.metric("Columns analyzed", len(anomalies["columns_used"]))

        import plotly.graph_objects as go
        normal_n = profile["n_rows"] - anomalies["n_anomalies"]
        pie = go.Figure(data=[go.Pie(
            labels=["Normal", "Anomalous"],
            values=[normal_n, anomalies["n_anomalies"]],
            hole=0.55,
            marker=dict(colors=["#6C5CE7", "#E17055"]),
        )])
        pie.update_layout(title="Normal vs. Flagged Rows")
        st.plotly_chart(_themed(pie), use_container_width=True)

        if anomalies.get("anomaly_sample"):
            st.markdown('<span class="section-badge">SAMPLE FLAGGED ROWS</span>', unsafe_allow_html=True)
            st.dataframe(anomalies["anomaly_sample"], use_container_width=True)

# ---------------------------------------------------------------------------
# CHAT
# ---------------------------------------------------------------------------
with tab_chat:
    if not st.session_state.chat_history:
        st.markdown('<span class="section-badge">TRY ASKING</span>', unsafe_allow_html=True)
        q_cols = st.columns(len(SUGGESTED_QUESTIONS[:3]))
        for i, q in enumerate(SUGGESTED_QUESTIONS[:3]):
            if q_cols[i].button(q, use_container_width=True, key=f"suggest_{i}"):
                st.session_state.pending_question = q

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            for chart in turn.get("charts", []):
                fig = _themed(pio.from_json(chart["figure_json"]))
                st.plotly_chart(fig, use_container_width=True)
            if turn.get("sources"):
                with st.expander("📚 Sources"):
                    for src in turn["sources"]:
                        st.markdown(f"**{src['metadata'].get('source', 'doc')}**: {src['text'][:300]}...")

    user_msg = st.chat_input("Ask about your data, e.g. 'What's the average revenue by region?'")
    if st.session_state.pending_question:
        user_msg = st.session_state.pending_question
        st.session_state.pending_question = None

    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/chat",
                    json={"session_id": st.session_state.session_id, "message": user_msg},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "charts": data.get("charts", []),
                    "sources": data.get("sources", []),
                })
            except requests.exceptions.RequestException as exc:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Sorry, something went wrong: {exc}",
                })
        st.rerun()