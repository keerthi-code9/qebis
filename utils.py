"""
utils.py
Shared helper functions for chart building and UI formatting.
"""

import numpy as np
import plotly.graph_objects as go
from data_loader import STAGE_META, ANOMALY_THRESHOLD, CAPACITY_THRESHOLD

# ─── Colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    "bg":       "#0a0e1a",
    "panel":    "#111827",
    "border":   "#1e2d3d",
    "text":     "#e2e8f0",
    "muted":    "#64748b",
    "accent":   "#38bdf8",       # sky blue
    "accent2":  "#a78bfa",       # violet
    "danger":   "#ff4757",
    "success":  "#00e5a0",
    "warn":     "#f0c040",
    "grid":     "rgba(100,116,139,0.15)",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'IBM Plex Mono', monospace", color=PALETTE["text"], size=11),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(
        gridcolor=PALETTE["grid"],
        showgrid=True,
        zeroline=False,
        color=PALETTE["muted"],
        title_font=dict(size=11),
    ),
    yaxis=dict(
        gridcolor=PALETTE["grid"],
        showgrid=True,
        zeroline=False,
        color=PALETTE["muted"],
        title_font=dict(size=11),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0.4)",
        bordercolor=PALETTE["border"],
        borderwidth=1,
        font=dict(size=10),
    ),
    hoverlabel=dict(
        bgcolor=PALETTE["panel"],
        bordercolor=PALETTE["accent"],
        font=dict(family="'IBM Plex Mono', monospace", size=11),
    ),
)


def apply_layout(fig: go.Figure, title: str = "", height: int = 320) -> go.Figure:
    """Apply consistent dark-theme layout to a Plotly figure."""
    layout = dict(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=13), x=0.01))
    layout["height"] = height
    fig.update_layout(**layout)
    return fig


# ─── Chart builders ──────────────────────────────────────────────────────────

def chart_anomaly(df, selected_cycle: int) -> go.Figure:
    """Section D – Anomaly score over cycles with threshold band."""
    fig = go.Figure()

    # Shaded anomaly region
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["anomaly_score"],
        fill="tozeroy", fillcolor="rgba(255,71,87,0.08)",
        line=dict(color=PALETTE["danger"], width=0),
        showlegend=False, hoverinfo="skip",
    ))

    # Main anomaly score line
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["anomaly_score"],
        mode="lines",
        line=dict(color=PALETTE["danger"], width=2.5),
        name="Anomaly Score",
        hovertemplate="Cycle %{x}<br>Score: %{y:.3f}<extra></extra>",
    ))

    # Threshold line
    fig.add_hline(
        y=ANOMALY_THRESHOLD,
        line_dash="dash", line_color=PALETTE["warn"], line_width=1.5,
        annotation_text="Threshold 0.5",
        annotation_font_color=PALETTE["warn"],
        annotation_position="right",
    )

    # Current cycle marker
    _add_cycle_vline(fig, df, selected_cycle)

    fig.update_yaxes(range=[0, 1.05], title_text="Anomaly Score")
    fig.update_xaxes(title_text="Cycle")
    return apply_layout(fig, "⚡ ML Anomaly Detection  (Isolation Forest)")


def chart_capacity(df, selected_cycle: int) -> go.Figure:
    """Section E – Capacity fade over cycles."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["capacity_fade"],
        fill="tozeroy", fillcolor="rgba(56,189,248,0.07)",
        line=dict(color=PALETTE["accent"], width=2.5),
        name="Capacity Fade %",
        hovertemplate="Cycle %{x}<br>Fade: %{y:.2f}%<extra></extra>",
    ))

    fig.add_hline(
        y=CAPACITY_THRESHOLD,
        line_dash="dash", line_color=PALETTE["warn"], line_width=1.5,
        annotation_text="Classical BMS alarm (20%)",
        annotation_font_color=PALETTE["warn"],
        annotation_position="right",
    )

    _add_cycle_vline(fig, df, selected_cycle)
    fig.update_yaxes(title_text="Capacity Fade (%)")
    fig.update_xaxes(title_text="Cycle")
    return apply_layout(fig, "🔋 Capacity Fade Over Cycles")


def chart_ml_vs_classical(df) -> go.Figure:
    """Section F – Overlay ML detection vs classical BMS."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["anomaly_score"],
        mode="lines", name="ML Score",
        line=dict(color=PALETTE["danger"], width=2),
        hovertemplate="Cycle %{x}<br>ML: %{y:.3f}<extra></extra>",
    ))

    # Normalise capacity fade to 0-1 for overlay
    cap_norm = df["capacity_fade"] / 100.0
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=cap_norm,
        mode="lines", name="Capacity Fade (norm.)",
        line=dict(color=PALETTE["accent"], width=2, dash="dot"),
        hovertemplate="Cycle %{x}<br>Fade (norm): %{y:.3f}<extra></extra>",
    ))

    # ML threshold
    fig.add_hline(y=ANOMALY_THRESHOLD, line_dash="dash",
                  line_color=PALETTE["danger"], line_width=1,
                  annotation_text="ML threshold", annotation_font_color=PALETTE["danger"],
                  annotation_position="left")

    # Classical threshold (20% / 100 normalised)
    fig.add_hline(y=0.20, line_dash="dash",
                  line_color=PALETTE["accent"], line_width=1,
                  annotation_text="Classical threshold", annotation_font_color=PALETTE["accent"],
                  annotation_position="right")

    # Shade the early-warning window
    ml_cyc  = df[df["anomaly_score"] > ANOMALY_THRESHOLD]["cycle"].min()
    cls_cyc = df[df["capacity_fade"] >= CAPACITY_THRESHOLD]["cycle"].min()
    if not (np.isnan(ml_cyc) or np.isnan(cls_cyc)) and ml_cyc < cls_cyc:
        fig.add_vrect(
            x0=ml_cyc, x1=cls_cyc,
            fillcolor="rgba(0,229,160,0.08)",
            line_width=0,
            annotation_text=f"Early warning: {int(cls_cyc - ml_cyc)} cycles",
            annotation_position="top left",
            annotation_font_color=PALETTE["success"],
        )

    fig.update_yaxes(title_text="Score / Normalised Fade", range=[0, 1.05])
    fig.update_xaxes(title_text="Cycle")
    return apply_layout(fig, "📊 ML vs Classical Detection Comparison", height=300)


def chart_nyquist(eis_spectrum: dict | None, cycle: int) -> go.Figure:
    """Section G – Nyquist plot for a single EIS spectrum."""
    fig = go.Figure()

    if eis_spectrum and "Z_real" in eis_spectrum and "Z_imag" in eis_spectrum:
        z_real = np.array(eis_spectrum["Z_real"])
        z_imag = -np.array(eis_spectrum["Z_imag"])  # Convention: -Z_imag on Y

        fig.add_trace(go.Scatter(
            x=z_real, y=z_imag,
            mode="lines+markers",
            line=dict(color=PALETTE["accent2"], width=2),
            marker=dict(size=4, color=PALETTE["accent2"]),
            name=f"Cycle {cycle}",
            hovertemplate="Z_real: %{x:.4f} Ω<br>-Z_imag: %{y:.4f} Ω<extra></extra>",
        ))
    else:
        # Placeholder arc
        theta = np.linspace(0, np.pi, 60)
        r = 0.05
        cx = 0.015
        z_real = cx + r * np.cos(theta)
        z_imag = r * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=z_real, y=z_imag,
            mode="lines",
            line=dict(color=PALETTE["muted"], width=2, dash="dot"),
            name="No EIS data",
        ))

    fig.update_yaxes(title_text="-Z_imag (Ω)", scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="Z_real (Ω)")
    return apply_layout(fig, f"〰 Nyquist Plot  (Cycle {cycle})", height=320)


def chart_physics_trends(df, selected_cycle: int) -> go.Figure:
    """Section H helper – small multiples of SEI / Li plating / resistances."""
    fig = go.Figure()

    col_cfg = [
        ("sei_thickness_um",       PALETTE["accent"],   "SEI Thickness (μm)"),
        ("li_plating_um",          PALETTE["danger"],   "Li Plating (μm)"),
        ("ohmic_resistance_mohm",  PALETTE["warn"],     "Ohmic R (mΩ)"),
        ("ct_resistance_mohm",     PALETTE["accent2"],  "CT Resistance (mΩ)"),
    ]

    for col, color, label in col_cfg:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["cycle"], y=df[col],
                mode="lines", name=label,
                line=dict(color=color, width=1.8),
                hovertemplate=f"Cycle %{{x}}<br>{label}: %{{y:.3f}}<extra></extra>",
            ))

    _add_cycle_vline(fig, df, selected_cycle)
    fig.update_xaxes(title_text="Cycle")
    fig.update_yaxes(title_text="Value")
    return apply_layout(fig, "🔬 Internal Physics Variables", height=300)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _add_cycle_vline(fig: go.Figure, df, cycle: int):
    """Add a vertical dashed line at the selected cycle."""
    if cycle in df["cycle"].values or (df["cycle"].min() <= cycle <= df["cycle"].max()):
        fig.add_vline(
            x=cycle,
            line_dash="solid", line_color="rgba(255,255,255,0.45)", line_width=1.5,
            annotation_text=f"c{cycle}",
            annotation_font_color="rgba(255,255,255,0.6)",
            annotation_position="top",
        )


def get_row_at_cycle(df, cycle: int):
    """Return the DataFrame row nearest to the given cycle."""
    idx = (df["cycle"] - cycle).abs().idxmin()
    return df.loc[idx]


def warning_status(score: float) -> tuple[str, str]:
    """Return (label, colour) for the anomaly warning status card."""
    if score > 0.75:
        return "🔴 CRITICAL", PALETTE["danger"]
    elif score > ANOMALY_THRESHOLD:
        return "🟡 WARNING", PALETTE["warn"]
    else:
        return "🟢 NORMAL", PALETTE["success"]
