"""
app.py  –  Battery Health Intelligence Dashboard
=================================================
Physics + AI-Based Early Degradation Detection

Run:
    streamlit run app.py
"""

import time
import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_all, get_eis_at_cycle, STAGE_META, ANOMALY_THRESHOLD
from utils import (
    PALETTE,
    chart_anomaly,
    chart_capacity,
    chart_ml_vs_classical,
    chart_nyquist,
    chart_physics_trends,
    get_row_at_cycle,
    warning_status,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Battery Health Intelligence Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS  –  dark industrial / monospace aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@300;400;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    background-color: #0a0e1a !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif;
}
.block-container { padding: 1.2rem 2rem 2rem 2rem !important; max-width: 1600px; }

/* ── Header ── */
.dash-header {
    border-bottom: 1px solid #1e2d3d;
    padding-bottom: 1.1rem;
    margin-bottom: 1.4rem;
}
.dash-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #38bdf8 0%, #a78bfa 60%, #00e5a0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.dash-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #64748b;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}

/* ── KPI cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.8rem;
    margin-bottom: 1.4rem;
}
.kpi-card {
    background: #111827;
    border: 1px solid #1e2d3d;
    border-radius: 8px;
    padding: 1rem 1.1rem 0.8rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-card.kpi-blue::before   { background: #38bdf8; }
.kpi-card.kpi-violet::before { background: #a78bfa; }
.kpi-card.kpi-orange::before { background: #ff8c42; }
.kpi-card.kpi-red::before    { background: #ff4757; }
.kpi-card.kpi-green::before  { background: #00e5a0; }
.kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.35rem;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.65rem;
    font-weight: 600;
    line-height: 1.1;
    color: #e2e8f0;
}
.kpi-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 0.25rem;
}

/* ── Section headers ── */
.section-head {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin: 1.4rem 0 0.5rem;
    border-left: 2px solid #38bdf8;
    padding-left: 0.6rem;
}

/* ── Physics table ── */
.phys-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
}
.phys-row {
    background: #111827;
    border: 1px solid #1e2d3d;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.phys-key {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #64748b;
}
.phys-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    color: #e2e8f0;
}

/* ── Insight pills ── */
.insight-box {
    background: #0f172a;
    border: 1px solid #1e2d3d;
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
.insight-item {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #a78bfa;
    border-left: 2px solid #a78bfa;
    padding-left: 0.75rem;
    margin-bottom: 0.6rem;
    line-height: 1.5;
}
.insight-item:last-child { margin-bottom: 0; }

/* ── Missing-file banner ── */
.warn-banner {
    background: #1c1208;
    border: 1px solid #ff8c42;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #ff8c42;
    margin-bottom: 1rem;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSlider"] > div { padding: 0 !important; }
div[data-testid="stSlider"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
button[kind="primary"] {
    background: #1e2d3d !important;
    border: 1px solid #38bdf8 !important;
    color: #38bdf8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
}
div[data-testid="column"] { gap: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "playing"   not in st.session_state: st.session_state.playing   = False
if "speed"     not in st.session_state: st.session_state.speed     = 5
if "data"      not in st.session_state: st.session_state.data      = None
if "auto_cycle" not in st.session_state: st.session_state.auto_cycle = 0


# ─────────────────────────────────────────────────────────────────────────────
# Load data (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading pipeline data…")
def cached_load(data_dir: str):
    return load_all(data_dir)


data = cached_load(".")
df:         pd.DataFrame = data["df"]
eis_spectra: list        = data["eis_spectra"]
meta:        dict        = data["meta"]
missing:     list        = data["missing"]

MAX_CYCLE = int(df["cycle"].max())
MIN_CYCLE = int(df["cycle"].min())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A – HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <div class="dash-title">Battery Health Intelligence Dashboard</div>
  <div class="dash-subtitle">Physics + AI-Based Early Degradation Detection &nbsp;|&nbsp; PyBaMM · EIS · Isolation Forest</div>
</div>
""", unsafe_allow_html=True)

# Missing files warning
if missing:
    st.markdown(
        f'<div class="warn-banner">⚠ Data files not found (using synthetic fallback): '
        f'{", ".join(missing)}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B – CONTROL PANEL
# ─────────────────────────────────────────────────────────────────────────────
ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([6, 1, 1, 1])

with ctrl_col1:
    if st.session_state.playing:
        selected_cycle = st.session_state.auto_cycle
        st.slider(
            "SELECT CYCLE", MIN_CYCLE, MAX_CYCLE,
            value=selected_cycle, disabled=True, key="cycle_slider_dis",
        )
    else:
        selected_cycle = st.slider(
            "SELECT CYCLE", MIN_CYCLE, MAX_CYCLE,
            value=st.session_state.auto_cycle, key="cycle_slider",
        )
        st.session_state.auto_cycle = selected_cycle

with ctrl_col2:
    if st.button("▶ Play" if not st.session_state.playing else "⏸ Pause"):
        st.session_state.playing = not st.session_state.playing
        if st.session_state.playing and st.session_state.auto_cycle >= MAX_CYCLE:
            st.session_state.auto_cycle = MIN_CYCLE
        st.rerun()

with ctrl_col3:
    if st.button("↺ Reset"):
        st.session_state.playing = False
        st.session_state.auto_cycle = MIN_CYCLE
        st.rerun()

with ctrl_col4:
    speed = st.selectbox("Speed", [1, 2, 5, 10, 20], index=2, label_visibility="collapsed")
    st.session_state.speed = speed

# Auto-play logic
if st.session_state.playing:
    if st.session_state.auto_cycle < MAX_CYCLE:
        st.session_state.auto_cycle = min(
            st.session_state.auto_cycle + st.session_state.speed,
            MAX_CYCLE,
        )
        time.sleep(0.05)
        st.rerun()
    else:
        st.session_state.playing = False


# ─────────────────────────────────────────────────────────────────────────────
# Derive per-cycle values
# ─────────────────────────────────────────────────────────────────────────────
row = get_row_at_cycle(df, selected_cycle)
stage        = int(row["stage"])
s_meta       = STAGE_META.get(stage, {"label": f"Stage {stage}", "color": "#888"})
cap_fade     = float(row["capacity_fade"])
anom_score   = float(row["anomaly_score"])
warn_label, warn_color = warning_status(anom_score)
eis_spectrum = get_eis_at_cycle(eis_spectra, selected_cycle)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C – KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-grid">

  <div class="kpi-card kpi-blue">
    <div class="kpi-label">Current Cycle</div>
    <div class="kpi-value">{selected_cycle}</div>
    <div class="kpi-sub">of {MAX_CYCLE} total</div>
  </div>

  <div class="kpi-card kpi-violet">
    <div class="kpi-label">Degradation Stage</div>
    <div class="kpi-value" style="color:{s_meta['color']}">{stage}</div>
    <div class="kpi-sub">{s_meta['label']}</div>
  </div>

  <div class="kpi-card kpi-orange">
    <div class="kpi-label">Capacity Fade</div>
    <div class="kpi-value">{cap_fade:.1f}<span style="font-size:1rem">%</span></div>
    <div class="kpi-sub">{'⚠ Below BMS threshold' if cap_fade < 20 else '🔴 BMS alarm triggered'}</div>
  </div>

  <div class="kpi-card kpi-red">
    <div class="kpi-label">Anomaly Score</div>
    <div class="kpi-value">{anom_score:.3f}</div>
    <div class="kpi-sub">threshold = {ANOMALY_THRESHOLD}</div>
  </div>

  <div class="kpi-card {'kpi-red' if anom_score > ANOMALY_THRESHOLD else 'kpi-green'}">
    <div class="kpi-label">Early Warning Status</div>
    <div class="kpi-value" style="font-size:1.15rem;color:{warn_color}">{warn_label}</div>
    <div class="kpi-sub">{'ML anomaly active' if anom_score > ANOMALY_THRESHOLD else 'All systems nominal'}</div>
  </div>

</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTIONS D & E  (two columns)
# ─────────────────────────────────────────────────────────────────────────────
col_d, col_e = st.columns(2)

with col_d:
    st.markdown('<div class="section-head">Anomaly Detection</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_anomaly(df, selected_cycle),
                    use_container_width=True, config={"displayModeBar": False})

with col_e:
    st.markdown('<div class="section-head">Capacity vs Time</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_capacity(df, selected_cycle),
                    use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# SECTION F  – ML vs Classical comparison (full width)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">ML vs Classical Detection Comparison</div>',
            unsafe_allow_html=True)
st.plotly_chart(chart_ml_vs_classical(df),
                use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# SECTIONS G & H  (two columns)
# ─────────────────────────────────────────────────────────────────────────────
col_g, col_h = st.columns([1, 1])

with col_g:
    st.markdown('<div class="section-head">Nyquist (EIS) Plot</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_nyquist(eis_spectrum, selected_cycle),
                    use_container_width=True, config={"displayModeBar": False})

with col_h:
    st.markdown('<div class="section-head">Internal Physics Panel</div>', unsafe_allow_html=True)

    def _fmt(val, decimals=3, unit=""):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{val:.{decimals}f}{' ' + unit if unit else ''}"

    physics_rows = [
        ("SEI Thickness",       _fmt(row.get("sei_thickness_um"),      3, "μm")),
        ("Li Plating",          _fmt(row.get("li_plating_um"),         3, "μm")),
        ("Ohmic Resistance",    _fmt(row.get("ohmic_resistance_mohm"), 2, "mΩ")),
        ("CT Resistance",       _fmt(row.get("ct_resistance_mohm"),    2, "mΩ")),
        ("Rs (EIS)",            _fmt(row.get("Rs_Ohm"),                4, "Ω")),
        ("Rct (EIS)",           _fmt(row.get("Rct_Ohm"),               4, "Ω")),
    ]

    rows_html = "".join(
        f'<div class="phys-row"><span class="phys-key">{k}</span>'
        f'<span class="phys-val">{v}</span></div>'
        for k, v in physics_rows
    )
    st.markdown(f'<div class="phys-grid">{rows_html}</div>', unsafe_allow_html=True)

    # Physics trend chart underneath
    st.plotly_chart(chart_physics_trends(df, selected_cycle),
                    use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# SECTION I  – Auto-generated insights
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Auto-Generated Insights</div>', unsafe_allow_html=True)

insights = meta.get("insights", [])
if not insights:
    insights = ["No insights available. Run the full pipeline to generate data."]

items_html = "".join(
    f'<div class="insight-item">▸ {ins}</div>' for ins in insights
)
st.markdown(f'<div class="insight-box">{items_html}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid #1e2d3d;
            text-align:center;font-family:'IBM Plex Mono',monospace;
            font-size:0.65rem;color:#334155;letter-spacing:0.1em;">
  BATTERY HEALTH INTELLIGENCE DASHBOARD &nbsp;·&nbsp;
  PyBaMM · EIS · Isolation Forest &nbsp;·&nbsp;
  Research Prototype
</div>
""", unsafe_allow_html=True)
