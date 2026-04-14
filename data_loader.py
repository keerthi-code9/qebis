"""
data_loader.py
Loads and synchronizes all pipeline JSON outputs into a unified dataframe.
Handles missing files gracefully and aligns data by cycle index.
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path


# ─── Stage metadata ───────────────────────────────────────────────────────────
STAGE_META = {
    1: {"label": "Pristine",          "color": "#00e5a0", "hex": "#00e5a0"},
    2: {"label": "Early Degradation", "color": "#f0c040", "hex": "#f0c040"},
    3: {"label": "Moderate",          "color": "#ff8c42", "hex": "#ff8c42"},
    4: {"label": "Advanced",          "color": "#ff4757", "hex": "#ff4757"},
    5: {"label": "Critical",          "color": "#c0392b", "hex": "#c0392b"},
}

ANOMALY_THRESHOLD = 0.5
CAPACITY_THRESHOLD = 20.0  # % fade = classical BMS alarm


def _safe_load(path: str) -> dict | list | None:
    """Load JSON file; return None if missing or corrupt."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def load_all(data_dir: str = ".") -> dict:
    """
    Load and align all pipeline outputs.

    Returns a dict with keys:
        df          – merged per-cycle DataFrame
        eis_spectra – list of per-cycle EIS dicts (Z_real / Z_imag arrays)
        meta        – pipeline statistics / insights
        missing     – list of files that could not be loaded
    """
    p = Path(data_dir)
    missing = []

    # ── 1. Load individual files ───────────────────────────────────────────
    ground_truth  = _safe_load(p / "battery_ground_truth.json")
    eis_spectra   = _safe_load(p / "eis_spectra.json")
    deg_labels    = _safe_load(p / "degradation_labels.json")
    anomaly_data  = _safe_load(p / "anomaly_scores.json")

    for name, obj in [
        ("battery_ground_truth.json", ground_truth),
        ("eis_spectra.json",          eis_spectra),
        ("degradation_labels.json",   deg_labels),
        ("anomaly_scores.json",       anomaly_data),
    ]:
        if obj is None:
            missing.append(name)

    # ── 2. Build base DataFrame from anomaly_scores (always present if pipeline ran) ──
    if anomaly_data:
        cycles     = anomaly_data["cycles"]
        df = pd.DataFrame({
            "cycle":          cycles,
            "anomaly_score":  anomaly_data["anomaly_scores"],
            "stage":          anomaly_data["stages"],
            "capacity_fade":  anomaly_data["capacity_fade_pct"],
        })
    elif deg_labels:
        df = pd.DataFrame(deg_labels)
        df = df.rename(columns={"capacity_fade_pct": "capacity_fade"})
        df["anomaly_score"] = 0.0
    else:
        # Minimal synthetic fallback so the UI still loads
        n = 500
        df = _synthetic_fallback(n)

    df = df.sort_values("cycle").reset_index(drop=True)

    # ── 3. Merge ground-truth physics variables ────────────────────────────
    if ground_truth:
        gt_df = pd.DataFrame(ground_truth)
        # Normalise column names
        rename_map = {}
        for col in gt_df.columns:
            low = col.lower()
            if "sei" in low and "thickness" in low:
                rename_map[col] = "sei_thickness_um"
            elif "plating" in low or "li_plating" in low:
                rename_map[col] = "li_plating_um"
            elif "ohmic" in low or "rs_" in low:
                rename_map[col] = "ohmic_resistance_mohm"
            elif "charge" in low and "transfer" in low:
                rename_map[col] = "ct_resistance_mohm"
            elif "capacity" in low and "fade" not in low:
                rename_map[col] = "capacity_ah"
        gt_df = gt_df.rename(columns=rename_map)
        if "cycle" in gt_df.columns:
            df = df.merge(gt_df, on="cycle", how="left")

    # ── 4. Merge EIS scalar features ──────────────────────────────────────
    if eis_spectra and isinstance(eis_spectra, list) and len(eis_spectra) > 0:
        eis_scalar_cols = ["cycle", "Rs_Ohm", "Rct_Ohm",
                           "sigma_warburg_Ohm_Hz_neg05", "arc_diameter_Ohm",
                           "peak_freq_Hz"]
        eis_rows = []
        for sp in eis_spectra:
            row = {k: sp.get(k) for k in eis_scalar_cols if k in sp}
            eis_rows.append(row)
        eis_df = pd.DataFrame(eis_rows)
        if "cycle" in eis_df.columns:
            df = df.merge(eis_df, on="cycle", how="left")

    # ── 5. Computed columns ────────────────────────────────────────────────
    df["anomaly_flag"]      = df["anomaly_score"] > ANOMALY_THRESHOLD
    df["classical_flag"]    = df["capacity_fade"] >= CAPACITY_THRESHOLD
    df["stage_label"]       = df["stage"].map(lambda s: STAGE_META.get(s, {}).get("label", f"Stage {s}"))
    df["stage_color"]       = df["stage"].map(lambda s: STAGE_META.get(s, {}).get("color", "#888"))

    # ── 6. Compute pipeline meta / insights ───────────────────────────────
    meta = _compute_meta(df)

    return {
        "df":          df,
        "eis_spectra": eis_spectra or [],
        "meta":        meta,
        "missing":     missing,
    }


def _compute_meta(df: pd.DataFrame) -> dict:
    """Derive pipeline-level statistics and auto-generated insights."""
    meta = {}

    # Detection cycles
    ml_cycle = df[df["anomaly_flag"]]["cycle"].min() if df["anomaly_flag"].any() else None
    cl_cycle = df[df["classical_flag"]]["cycle"].min() if df["classical_flag"].any() else None
    meta["ml_detection_cycle"]        = int(ml_cycle) if ml_cycle is not None and not np.isnan(ml_cycle) else None
    meta["classical_detection_cycle"] = int(cl_cycle) if cl_cycle is not None and not np.isnan(cl_cycle) else None

    if meta["ml_detection_cycle"] and meta["classical_detection_cycle"]:
        adv = meta["classical_detection_cycle"] - meta["ml_detection_cycle"]
        meta["early_warning_cycles"] = adv
        meta["early_warning_pct"]    = round(100 * adv / meta["classical_detection_cycle"], 1)
    else:
        meta["early_warning_cycles"] = None
        meta["early_warning_pct"]    = None

    # Stage transitions
    stage_transitions = {}
    for _, row in df.iterrows():
        s = int(row["stage"])
        if s not in stage_transitions:
            stage_transitions[s] = int(row["cycle"])
    meta["stage_transitions"] = stage_transitions

    # Impedance rise
    if "Rs_Ohm" in df.columns:
        rs_first = df["Rs_Ohm"].iloc[0]
        rs_last  = df["Rs_Ohm"].iloc[-1]
        if rs_first and rs_first > 0:
            meta["rs_rise_pct"] = round(100 * (rs_last - rs_first) / rs_first, 1)
        else:
            meta["rs_rise_pct"] = None
    else:
        meta["rs_rise_pct"] = None

    # Auto insights
    insights = []
    for stage in range(2, 6):
        cyc = stage_transitions.get(stage)
        if cyc is not None:
            label = STAGE_META[stage]["label"]
            insights.append(f"Battery entered Stage {stage} ({label}) at cycle {cyc}.")

    if meta["early_warning_cycles"]:
        insights.append(
            f"ML detected anomaly {meta['early_warning_cycles']} cycles "
            f"before classical BMS ({meta['early_warning_pct']}% earlier)."
        )

    if meta.get("rs_rise_pct") and meta["rs_rise_pct"] > 100:
        insights.append(
            f"Rapid ohmic resistance rise detected: +{meta['rs_rise_pct']}% over full lifecycle."
        )

    if "li_plating_um" in df.columns:
        plate_start = df[df["li_plating_um"] > 0.05]["cycle"].min()
        if not np.isnan(plate_start):
            insights.append(f"Lithium plating onset detected at cycle {int(plate_start)}.")

    meta["insights"] = insights
    return meta


def get_eis_at_cycle(eis_spectra: list, cycle: int) -> dict | None:
    """Return the EIS spectrum closest to the requested cycle."""
    if not eis_spectra:
        return None
    closest = min(eis_spectra, key=lambda x: abs(x.get("cycle", 0) - cycle))
    return closest


def _synthetic_fallback(n: int = 500) -> pd.DataFrame:
    """Generate minimal synthetic data so the dashboard still renders."""
    cycles = np.arange(n)
    capacity_fade = np.clip(cycles * 0.15, 0, 80)
    sei = 1.0 + cycles * 0.005
    li_plate = np.maximum(0, (cycles - 100) * 0.01)

    stages = []
    for i in range(n):
        s = sei[i]
        lp = li_plate[i]
        if s < 0.5:
            stages.append(1)
        elif s < 1.5:
            stages.append(2)
        elif s < 3.0 and lp < 0.5:
            stages.append(3)
        elif s < 5.0 and lp < 1.0:
            stages.append(4)
        else:
            stages.append(5)

    raw_scores = np.clip((cycles - 20) / 130, 0, 1)
    scores = raw_scores + np.random.normal(0, 0.02, n)
    scores = np.clip(scores, 0, 1)

    return pd.DataFrame({
        "cycle":               cycles,
        "anomaly_score":       scores,
        "stage":               stages,
        "capacity_fade":       capacity_fade,
        "sei_thickness_um":    sei,
        "li_plating_um":       li_plate,
        "ohmic_resistance_mohm":  10 + cycles * 0.1,
        "ct_resistance_mohm":     50 + cycles * 0.3,
        "capacity_ah":            5.0 * (1 - capacity_fade / 100),
        "Rs_Ohm":                 0.01 + cycles * 0.0001,
        "Rct_Ohm":                0.05 + cycles * 0.0003,
    })
