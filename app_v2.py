"""
FRAIXL - FRAM Digital Twin Landing Phase Demonstrator — v2
============================================================

Implements a FRAM-informed Landing Phase Twin built around five
functions, with a binary stable-approach check at the 1000 ft AGL gate
(flap 30 + gear down + CAS in [Vref, Vref+20]).

Run:  streamlit run app_v2.py
"""

from __future__ import annotations
import json
import os
from dataclasses import asdict

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from parameter_dictionary import PARAMETER_DICTIONARY
from fraixl_extract import (
    load_fdr_csv,
    build_canonical,
    detect_landing_events,
    compute_distance_from_touchdown,
    summarize_flight,
)
from fram_metadata_v2 import CalibrationProfile, compute_states, stabilization_gate_assessment


# ============================================================================
# Page setup
# ============================================================================

st.set_page_config(
    page_title="FRAIXL - FRAM Digital Twin Landing Phase Demonstrator",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ FRAIXL — FRAM Digital Twin Landing Phase Demonstrator")
st.caption(
    "A FRAM-informed Epistemic Digital Twin demonstrator for approach and landing. "
    "Five functions tracked through the approach; binary stable-approach check at the 1000 ft AGL gate."
)


# ============================================================================
# Sidebar — file source & calibration
# ============================================================================

with st.sidebar:
    st.header("📂 Flight data")

    samples_dir = "samples"
    sample_files = []
    if os.path.isdir(samples_dir):
        sample_files = sorted([f for f in os.listdir(samples_dir) if f.lower().endswith(".csv")])

    sample_choice = None
    if sample_files:
        options = ["— Upload my own —"] + sample_files
        sample_choice = st.selectbox(
            "Sample flight", options,
            index=1 if len(sample_files) >= 1 else 0,
        )

    uploaded = None
    if not sample_files or sample_choice == "— Upload my own —":
        uploaded = st.file_uploader(
            "Upload an FDR/FOQA CSV", type=["csv"],
        )

    st.divider()
    st.header("⚙️ Calibration profile")
    st.caption(
        "**Provisional thresholds.** Per FRAIXL-Air WP4, these must be "
        "replaced by calibration from reviewed normal landings before "
        "being treated as a divergence detector."
    )

    with st.expander("Stable-approach criteria", expanded=True):
        landing_flap = st.selectbox("Required landing flap (deg)", [15, 25, 30, 40], index=2)
        speed_min = st.slider("Speed window min (CAS − Vref)", -10.0, 10.0, 0.0, 1.0)
        speed_max = st.slider("Speed window max (CAS − Vref)", 5.0, 30.0, 20.0, 1.0)
        n1_min = st.slider("Minimum N1 for energy (%)", 30.0, 60.0, 40.0, 1.0)
        gate_alt = st.slider("Stabilization gate (ft AGL)", 500, 1000, 1000, 50,
                             help="Some operators use 500 ft (VMC); 1000 ft is the typical IMC gate.")

    with st.expander("Glidepath / lateral path", expanded=False):
        gs_normal = st.slider("Glideslope normal (dots)", 0.5, 2.0, 1.0, 0.1)
        gs_caution = st.slider("Glideslope caution (dots)", 1.0, 3.0, 2.0, 0.1)
        loc_normal = st.slider("Localizer normal (dots)", 0.5, 2.0, 1.0, 0.1)
        loc_caution = st.slider("Localizer caution (dots)", 1.0, 3.0, 2.0, 0.1)

    cal = CalibrationProfile(
        required_landing_flap_deg=landing_flap,
        speed_window_min_kt=speed_min,
        speed_window_max_kt=speed_max,
        n1_min_pct=n1_min,
        stabilization_gate_radalt_ft=float(gate_alt),
        gs_normal_dots=gs_normal,
        gs_caution_dots=gs_caution,
        loc_normal_dots=loc_normal,
        loc_caution_dots=loc_caution,
    )

    st.divider()
    st.header("🤖 Bounded LLM layer")
    st.caption(
        "Optional. Claude as the bounded interpretive layer on top of "
        "the formal model."
    )
    api_key_from_secrets = None
    try:
        api_key_from_secrets = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        pass
    if api_key_from_secrets:
        api_key = api_key_from_secrets
        st.success("✅ LLM layer enabled (server-side key)")
    else:
        env_key = os.environ.get("ANTHROPIC_API_KEY", "")
        api_key = st.text_input("Anthropic API key", type="password", value=env_key)


# ============================================================================
# Load & process
# ============================================================================

@st.cache_data(show_spinner="Loading FDR CSV...")
def _load(path_or_bytes, is_path: bool):
    if is_path:
        return load_fdr_csv(path_or_bytes)
    with open("/tmp/_uploaded_fdr.csv", "wb") as f:
        f.write(path_or_bytes)
    return load_fdr_csv("/tmp/_uploaded_fdr.csv")


@st.cache_data(show_spinner="Building canonical metadata...")
def _process(_raw):
    canonical = build_canonical(_raw)
    events = detect_landing_events(canonical)
    canonical["dist_from_td_nm"] = compute_distance_from_touchdown(canonical, events)
    return canonical, events


@st.cache_data(show_spinner="Computing FRAM function states...")
def _states(_canonical, cal_dict):
    return compute_states(_canonical, CalibrationProfile(**cal_dict))


raw = None
source_label = None
if uploaded is not None:
    raw = _load(uploaded.getvalue(), is_path=False)
    source_label = uploaded.name
elif sample_choice and sample_choice != "— Upload my own —":
    raw = _load(os.path.join(samples_dir, sample_choice), is_path=True)
    source_label = f"Sample: {sample_choice}"

if raw is None:
    st.info("👈 Pick a sample flight or upload a CSV in the sidebar to begin.")
    st.stop()

canonical, events = _process(raw)
states = _states(canonical, asdict(cal))

st.caption(f"📊 Source: **{source_label}** — {raw.shape[0]:,} rows × {raw.shape[1]:,} cols → "
           f"{canonical.shape[1]} canonical metadata fields")


def _render_state_counts(states_df, col_name):
    """Show the state distribution for a function over the approach phase."""
    approach = states_df[(states_df["dist_from_td_nm"] >= -4) & (states_df["dist_from_td_nm"] <= 0)]
    if len(approach) == 0:
        return
    counts = approach[col_name].value_counts()
    pct = (counts / counts.sum() * 100).round(1)
    disp = pd.DataFrame({"State": counts.index, "Samples": counts.values, "%": pct.values})
    with st.expander(f"State distribution (last 4 nm before touchdown)"):
        st.dataframe(disp, width="stretch", hide_index=True)


# ============================================================================
# Tabs
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Raw & Parameter Dictionary",
    "2️⃣ Flight Profile",
    "3️⃣ FRAM Twin (5 functions)",
    "4️⃣ Calibration & Discrepancy",
])


# ----------------------------------------------------------------------------
# TAB 1 — Raw + parameter dictionary
# ----------------------------------------------------------------------------

with tab1:
    st.header("Raw FDR data and data-to-model dictionary")
    st.write(
        "The data-to-model dictionary called for in WP2 of the FRAIXL-Air "
        "proposal — explicit mapping from raw FDR columns to canonical "
        "metadata keys. Downloadable as JSON for reuse in the .xfmv environment."
    )

    summary = summarize_flight(canonical, events)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segment duration", f"{summary['duration_s']:.0f} s")
    c2.metric("Sample rate", f"{summary['sample_rate_hz']:.1f} Hz")
    c3.metric("Samples", f"{summary['n_samples']:,}")
    c4.metric("Touched down?", "Yes" if summary["touched_down"] else "No")

    st.subheader("Parameter dictionary — mappings found in this file")
    rows = []
    for canonical_key, meta in PARAMETER_DICTIONARY.items():
        matched = next((c for c in meta["fdr_columns"] if c in raw.columns), None)
        rows.append({
            "Canonical key": canonical_key,
            "FRAM function": meta["function"],
            "Unit": meta["unit"],
            "Matched FDR column": matched if matched else "— not found —",
            "Status": "✅" if matched else "❌",
            "Description": meta["description"],
        })
    dict_df = pd.DataFrame(rows)
    found = (dict_df["Status"] == "✅").sum()
    st.caption(f"**{found} of {len(dict_df)}** canonical keys mapped from this file.")
    st.dataframe(dict_df, width="stretch", hide_index=True)

    dict_json = json.dumps(
        {k: {**v, "matched_in_file": next(
            (c for c in v["fdr_columns"] if c in raw.columns), None
        )} for k, v in PARAMETER_DICTIONARY.items()},
        indent=2,
    )
    st.download_button(
        "📥 Download parameter dictionary (JSON)",
        data=dict_json,
        file_name="fraixl_parameter_dictionary.json",
        mime="application/json",
    )

    with st.expander("Show raw column inventory"):
        st.dataframe(pd.DataFrame({"Column": raw.columns}), width="stretch", hide_index=True, height=400)

    with st.expander("Show canonical metadata (first 200 rows)"):
        st.dataframe(canonical.head(200), width="stretch")


# ----------------------------------------------------------------------------
# TAB 2 — Flight Profile
# ----------------------------------------------------------------------------

with tab2:
    st.header("Flight profile")

    # ----- Vertical profile -----
    st.subheader("Vertical profile (altitude vs distance from touchdown)")
    fig_prof = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=("Altitude profile", "Vertical speed"),
    )
    x = canonical["dist_from_td_nm"]

    if "observed_radalt_ft" in canonical.columns:
        fig_prof.add_trace(go.Scatter(x=x, y=canonical["observed_radalt_ft"],
                                       mode="lines", name="Radio altitude",
                                       line=dict(color="#1f77b4", width=2)), row=1, col=1)
    if "observed_alt_baro_ft" in canonical.columns:
        fig_prof.add_trace(go.Scatter(x=x, y=canonical["observed_alt_baro_ft"],
                                       mode="lines", name="Baro altitude",
                                       line=dict(color="#aec7e8", width=1, dash="dot")), row=1, col=1)
    gs_x = np.array([-6, 0])
    gs_y = -np.tan(np.radians(3.0)) * gs_x * 6076.12
    fig_prof.add_trace(go.Scatter(x=gs_x, y=gs_y, mode="lines", name="3° reference",
                                   line=dict(color="gray", dash="dash", width=1)), row=1, col=1)

    if "observed_vrate_fpm" in canonical.columns:
        fig_prof.add_trace(go.Scatter(x=x, y=canonical["observed_vrate_fpm"],
                                       mode="lines", name="Vertical speed",
                                       line=dict(color="#ff7f0e")), row=2, col=1)
        fig_prof.add_hline(y=0, line_color="gray", line_dash="dot", row=2, col=1)

    event_color = {
        "gear_down_selected": "#2ca02c",
        "flap_5_reached": "#9467bd", "flap_10_reached": "#9467bd",
        "flap_15_reached": "#9467bd", "flap_25_reached": "#9467bd", "flap_30_reached": "#9467bd",
        "loc_captured": "#17becf", "gs_captured": "#17becf",
        "flare_engaged": "#d62728", "touchdown": "#000000",
        "ap_disengaged": "#8c564b", "tr_deployed": "#e377c2",
        "auto_speedbrake_deploy": "#7f7f7f",
    }
    for name, ev in events.items():
        td_idx = (canonical["time_s"] - ev["time_s"]).abs().idxmin()
        ev_x = canonical["dist_from_td_nm"].iloc[td_idx]
        ev_y = ev["radalt_ft"] if pd.notna(ev["radalt_ft"]) else ev["alt_baro_ft"]
        col = event_color.get(name, "#666666")
        fig_prof.add_trace(go.Scatter(
            x=[ev_x], y=[ev_y], mode="markers+text",
            marker=dict(size=10, color=col, symbol="diamond"),
            text=[name.replace("_", " ")], textposition="top center", textfont=dict(size=9),
            name=name, showlegend=False,
            hovertemplate=f"<b>{name}</b><br>t={ev['time_s']:.1f}s<br>"
                          f"radalt={ev['radalt_ft']:.0f}ft<br>CAS={ev['cas_kt']:.0f}kt<extra></extra>",
        ), row=1, col=1)

    fig_prof.update_xaxes(title_text="Distance from touchdown (nm)", row=2, col=1)
    fig_prof.update_yaxes(title_text="Altitude (ft)", row=1, col=1)
    fig_prof.update_yaxes(title_text="Vertical speed (fpm)", row=2, col=1)
    fig_prof.update_layout(height=620, hovermode="x unified", showlegend=True)
    st.plotly_chart(fig_prof, width="stretch")

    # ----- Energy plot -----
    st.subheader("Energy state — CAS vs Vref")
    fig_spd = go.Figure()
    if "observed_speed_kt" in canonical.columns:
        fig_spd.add_trace(go.Scatter(x=x, y=canonical["observed_speed_kt"],
                                     mode="lines", name="CAS",
                                     line=dict(color="#1f77b4", width=2)))
    if "vref_kt" in canonical.columns and canonical["vref_kt"].notna().any():
        fig_spd.add_trace(go.Scatter(x=x, y=canonical["vref_kt"],
                                     mode="lines", name="Vref",
                                     line=dict(color="red", dash="dot")))
        # Vref+20 band
        fig_spd.add_trace(go.Scatter(x=x, y=canonical["vref_kt"] + cal.speed_window_max_kt,
                                     mode="lines", name=f"Vref+{cal.speed_window_max_kt:.0f}",
                                     line=dict(color="orange", dash="dash")))
    if "target_speed_kt" in canonical.columns:
        fig_spd.add_trace(go.Scatter(x=x, y=canonical["target_speed_kt"],
                                     mode="lines", name="Target",
                                     line=dict(color="green", dash="dash")))
    fig_spd.update_xaxes(title_text="Distance from touchdown (nm)")
    fig_spd.update_yaxes(title_text="Speed (kt)")
    fig_spd.update_layout(height=320, hovermode="x unified")
    st.plotly_chart(fig_spd, width="stretch")

    # ----- Configuration timeline -----
    st.subheader("Configuration timeline (flap & gear)")
    fig_cfg = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.55, 0.45],
                            subplot_titles=("Flap handle (deg)", "Gear / WoW discretes"))
    if "flap_handle_deg" in canonical.columns:
        fig_cfg.add_trace(go.Scatter(x=x, y=canonical["flap_handle_deg"],
                                     mode="lines", name="Flap handle",
                                     line=dict(color="#9467bd", width=2)), row=1, col=1)
    if "gear_lever_up" in canonical.columns:
        # Decoded gear-down for this 737 mapping (see parameter dictionary)
        gear_down_decoded = (canonical["gear_lever_up"] == 1).astype(float)
        fig_cfg.add_trace(go.Scatter(x=x, y=gear_down_decoded,
                                     mode="lines", name="Gear DOWN (decoded)",
                                     line=dict(color="#2ca02c")), row=2, col=1)
    if "air_ground" in canonical.columns:
        fig_cfg.add_trace(go.Scatter(x=x, y=canonical["air_ground"],
                                     mode="lines", name="AIR/GROUND (1=air)",
                                     line=dict(color="#d62728", dash="dot")), row=2, col=1)
    fig_cfg.update_xaxes(title_text="Distance from touchdown (nm)", row=2, col=1)
    fig_cfg.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig_cfg, width="stretch")

    # ----- Ground track -----
    if "lat_deg" in canonical.columns and "lon_deg" in canonical.columns:
        st.subheader("Ground track")
        use_new_map = hasattr(go, "Scattermap")
        if use_new_map:
            MapTrace, map_key = go.Scattermap, "map"
        else:
            MapTrace, map_key = go.Scattermapbox, "mapbox"

        fig_map = go.Figure()
        fig_map.add_trace(MapTrace(
            lat=canonical["lat_deg"], lon=canonical["lon_deg"],
            mode="lines+markers",
            marker=dict(size=4, color=canonical["observed_radalt_ft"], colorscale="Viridis",
                        showscale=True, colorbar=dict(title="Radalt (ft)")),
            line=dict(width=2), name="Track",
        ))
        if "touchdown" in events:
            td_idx = (canonical["time_s"] - events["touchdown"]["time_s"]).abs().idxmin()
            fig_map.add_trace(MapTrace(
                lat=[canonical["lat_deg"].iloc[td_idx]],
                lon=[canonical["lon_deg"].iloc[td_idx]],
                mode="markers", marker=dict(size=14, color="red"),
                name="Touchdown",
            ))
        fig_map.update_layout(
            **{map_key: dict(style="open-street-map",
                             center=dict(lat=canonical["lat_deg"].median(),
                                         lon=canonical["lon_deg"].median()),
                             zoom=10)},
            height=500, margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_map, width="stretch")

    # ----- Event table -----
    st.subheader("Detected events")
    if events:
        ev_df = pd.DataFrame([
            {"Event": k, "Time (s)": v["time_s"], "Radalt (ft)": v["radalt_ft"],
             "CAS (kt)": v["cas_kt"], "Note": v["note"]}
            for k, v in sorted(events.items(), key=lambda x: x[1]["time_s"])
        ])
        st.dataframe(ev_df, width="stretch", hide_index=True)


# ----------------------------------------------------------------------------
# TAB 3 — FRAM Twin (5 functions)
# ----------------------------------------------------------------------------

with tab3:
    st.header("FRAM Twin — five functions tracked through approach")
    st.write(
        "Each function below is derived from observed metadata. State "
        "labels are *categorical*, not scores. The underlying parameter "
        "trace is shown alongside so you can see why each state is what it is."
    )

    x = canonical["dist_from_td_nm"]

    # ====== Function 1: Configure the aircraft ======
    st.subheader("Function 1 — Configure the aircraft")
    st.caption("**Inputs:** flap handle position, gear lever discrete. "
               "**States:** CONFIG_COMPLETE / CONFIG_INCOMPLETE.")
    fig_fn1 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                            vertical_spacing=0.08,
                            subplot_titles=("Flap & gear", "State"))
    if "flap_handle_deg" in canonical.columns:
        fig_fn1.add_trace(go.Scatter(x=x, y=canonical["flap_handle_deg"],
                                     mode="lines", name="Flap handle (deg)",
                                     line=dict(color="#9467bd")), row=1, col=1)
        fig_fn1.add_hline(y=cal.required_landing_flap_deg, line_color="green",
                          line_dash="dash", row=1, col=1,
                          annotation_text=f"Required ≥ {cal.required_landing_flap_deg:.0f}°")
    if "gear_lever_up" in canonical.columns:
        # Decoded: gear is down when GEAR LEVER UP == 1 (lever NOT in UP detent)
        gear_down_decoded = (canonical["gear_lever_up"] == 1).astype(float)
        fig_fn1.add_trace(go.Scatter(x=x, y=gear_down_decoded * 40,
                                     mode="lines", name="Gear DOWN (decoded) ×40",
                                     line=dict(color="#2ca02c", dash="dot")), row=1, col=1)
    state_num = (states["fn1_configure_aircraft"] == "CONFIG_COMPLETE").astype(int)
    fig_fn1.add_trace(go.Scatter(x=x, y=state_num, mode="lines",
                                 name="CONFIG_COMPLETE (1=yes)",
                                 line=dict(color="#1f77b4", width=2),
                                 fill="tozeroy"), row=2, col=1)
    fig_fn1.update_xaxes(title_text="Distance from touchdown (nm)", row=2, col=1)
    fig_fn1.update_yaxes(title_text="deg", row=1, col=1)
    fig_fn1.update_yaxes(title_text="State", range=[-0.1, 1.2], row=2, col=1)
    fig_fn1.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig_fn1, width="stretch")
    _render_state_counts(states, "fn1_configure_aircraft")

    # ====== Function 2: Maintain glidepath ======
    st.subheader("Function 2 — Maintain glidepath")
    st.caption("**Input:** glideslope deviation (DDM, L). "
               "**States:** ON_GLIDEPATH / GP_DEVIATION / GP_EXCURSION.")
    fig_fn2 = go.Figure()
    if "gs_deviation_ddm" in canonical.columns:
        fig_fn2.add_trace(go.Scatter(
            x=x, y=canonical["gs_deviation_ddm"] / cal.gs_one_dot_ddm,
            mode="lines", name="GS deviation (dots)",
            line=dict(color="#17becf", width=2),
        ))
        fig_fn2.add_hline(y=cal.gs_normal_dots, line_color="green", line_dash="dash",
                          annotation_text=f"±{cal.gs_normal_dots:.1f} dot")
        fig_fn2.add_hline(y=-cal.gs_normal_dots, line_color="green", line_dash="dash")
        fig_fn2.add_hline(y=cal.gs_caution_dots, line_color="orange", line_dash="dot")
        fig_fn2.add_hline(y=-cal.gs_caution_dots, line_color="orange", line_dash="dot")
        fig_fn2.add_hline(y=0, line_color="gray")
        fig_fn2.update_xaxes(title_text="Distance from touchdown (nm)")
        fig_fn2.update_yaxes(title_text="GS deviation (dots, ±)")
        fig_fn2.update_layout(height=320, hovermode="x unified")
    st.plotly_chart(fig_fn2, width="stretch")
    _render_state_counts(states, "fn2_maintain_glidepath")

    # ====== Function 3: Manage energy state ======
    st.subheader("Function 3 — Manage energy state")
    st.caption(f"**Inputs:** CAS, Vref, average N1. "
               f"**States:** ON_TARGET (in window AND N1 ≥ {cal.n1_min_pct:.0f}%) / "
               f"HIGH_ENERGY / LOW_ENERGY / LOW_N1. "
               f"Window = Vref{cal.speed_window_min_kt:+.0f} to Vref+{cal.speed_window_max_kt:.0f}.")
    fig_fn3 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
                            vertical_spacing=0.08,
                            subplot_titles=("CAS − Vref", "Average N1"))
    if "cas_minus_vref_kt" in states.columns:
        fig_fn3.add_trace(go.Scatter(
            x=x, y=states["cas_minus_vref_kt"],
            mode="lines", name="CAS − Vref (kt)",
            line=dict(color="#1f77b4", width=2),
        ), row=1, col=1)
        fig_fn3.add_hline(y=cal.speed_window_min_kt, line_color="green", line_dash="dash",
                          row=1, col=1,
                          annotation_text=f"Vref{cal.speed_window_min_kt:+.0f}")
        fig_fn3.add_hline(y=cal.speed_window_max_kt, line_color="green", line_dash="dash",
                          row=1, col=1,
                          annotation_text=f"Vref+{cal.speed_window_max_kt:.0f}")
        fig_fn3.add_hline(y=0, line_color="red", line_dash="dot", row=1, col=1,
                          annotation_text="Vref")
    if "n1_avg_pct" in states.columns:
        fig_fn3.add_trace(go.Scatter(
            x=x, y=states["n1_avg_pct"],
            mode="lines", name="Avg N1 (%)",
            line=dict(color="#ff7f0e", width=2),
        ), row=2, col=1)
        fig_fn3.add_hline(y=cal.n1_min_pct, line_color="green", line_dash="dash",
                          row=2, col=1,
                          annotation_text=f"N1 ≥ {cal.n1_min_pct:.0f}%")
    fig_fn3.update_xaxes(title_text="Distance from touchdown (nm)", row=2, col=1)
    fig_fn3.update_yaxes(title_text="kt", row=1, col=1)
    fig_fn3.update_yaxes(title_text="% RPM", row=2, col=1)
    fig_fn3.update_layout(height=440, hovermode="x unified")
    st.plotly_chart(fig_fn3, width="stretch")
    _render_state_counts(states, "fn3_manage_energy")

    # ====== Function 4: Maintain stability (continuous) ======
    st.subheader("Function 4 — Maintain stability")
    st.caption("**Inputs:** lateral path (LOC) + vertical path (GS) + speed in window. "
               "Configuration is **not** part of stability (separate function). "
               "**States:** STABLE / UNSTABLE.")
    state_num4 = (states["fn4_maintain_stability"] == "STABLE").astype(int)
    fig_fn4 = go.Figure()
    fig_fn4.add_trace(go.Scatter(x=x, y=state_num4, mode="lines",
                                 name="STABLE (1=yes)",
                                 line=dict(color="#2ca02c", width=2),
                                 fill="tozeroy"))
    fig_fn4.update_xaxes(title_text="Distance from touchdown (nm)")
    fig_fn4.update_yaxes(title_text="State", range=[-0.1, 1.2])
    fig_fn4.update_layout(height=260, hovermode="x unified")
    st.plotly_chart(fig_fn4, width="stretch")
    _render_state_counts(states, "fn4_maintain_stability")

    # ====== Function 5: Maintain lateral path ======
    st.subheader("Function 5 — Maintain lateral path")
    st.caption("**Input:** localizer deviation (DDM, L). "
               "**States:** ON_LATERAL / LAT_DEVIATION / LAT_EXCURSION.")
    fig_fn5 = go.Figure()
    if "loc_deviation_ddm" in canonical.columns:
        fig_fn5.add_trace(go.Scatter(
            x=x, y=canonical["loc_deviation_ddm"] / cal.loc_one_dot_ddm,
            mode="lines", name="LOC deviation (dots)",
            line=dict(color="#e377c2", width=2),
        ))
        fig_fn5.add_hline(y=cal.loc_normal_dots, line_color="green", line_dash="dash",
                          annotation_text=f"±{cal.loc_normal_dots:.1f} dot")
        fig_fn5.add_hline(y=-cal.loc_normal_dots, line_color="green", line_dash="dash")
        fig_fn5.add_hline(y=cal.loc_caution_dots, line_color="orange", line_dash="dot")
        fig_fn5.add_hline(y=-cal.loc_caution_dots, line_color="orange", line_dash="dot")
        fig_fn5.add_hline(y=0, line_color="gray")
        fig_fn5.update_xaxes(title_text="Distance from touchdown (nm)")
        fig_fn5.update_yaxes(title_text="LOC deviation (dots, ±)")
        fig_fn5.update_layout(height=320, hovermode="x unified")
    st.plotly_chart(fig_fn5, width="stretch")
    _render_state_counts(states, "fn5_maintain_lateral_path")

    # ====== Stabilization gate panel ======
    st.divider()
    st.subheader(f"📏 Stabilization gate assessment ({cal.stabilization_gate_radalt_ft:.0f} ft AGL)")
    st.caption(f"Binary: STABLE iff flap ≥ {cal.required_landing_flap_deg:.0f}, gear DOWN, "
               f"and CAS in [Vref{cal.speed_window_min_kt:+.0f}, Vref+{cal.speed_window_max_kt:.0f}].")
    gate = stabilization_gate_assessment(states, cal)
    if gate.get("available"):
        verdict = gate["stable_approach"]
        if verdict == "STABLE":
            st.success(f"### ✅ {verdict} approach")
        else:
            st.error(f"### ❌ {verdict} approach")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Configuration", gate["configuration_state"],
                      f"flap {gate['flap_deg']:.0f}°, gear {'DN' if gate['gear_is_down'] else 'UP'}")
            st.metric("Energy", gate["energy_state"], f"{gate['cas_minus_vref_kt']:+.1f} kt")
        with c2:
            st.metric("CAS at gate", gate["cas_label"],
                      f"CAS {gate['cas_kt']:.0f} / Vref {gate['vref_kt']:.0f}")
            st.metric("Vertical", gate["vertical_state"], f"{gate['vrate_fpm']:.0f} fpm")
        with c3:
            st.metric("Control mode", gate["control_mode_state"])
            n1_val = gate.get("n1_avg_pct")
            if n1_val is not None and not (isinstance(n1_val, float) and np.isnan(n1_val)):
                st.metric("Avg N1 at gate", f"{n1_val:.0f}%",
                          "OK" if n1_val >= cal.n1_min_pct else f"below {cal.n1_min_pct:.0f}%")
            else:
                st.metric("Time at gate", f"{gate['time_s']:.1f} s")
    else:
        st.warning(f"Gate assessment unavailable: {gate.get('reason')}")


# ----------------------------------------------------------------------------
# TAB 4 — Calibration & Discrepancy
# ----------------------------------------------------------------------------

with tab4:
    st.header("Calibration & discrepancy")
    st.write(
        "The active calibration profile and a delta view showing where "
        "each function crossed its bound during the approach. Per the "
        "FRAIXL-Air method, divergences are only meaningful *after* "
        "calibration against reviewed normal landings."
    )

    st.subheader("Active calibration profile")
    cal_df = pd.DataFrame([{"Parameter": k, "Value": v} for k, v in cal.as_dict().items()])
    st.dataframe(cal_df, width="stretch", hide_index=True)
    st.download_button(
        "📥 Download calibration profile (JSON)",
        data=json.dumps(cal.as_dict(), indent=2),
        file_name="fraixl_calibration_profile.json",
        mime="application/json",
    )

    # Approach-phase summary (last 4 nm before TD)
    st.subheader("Function states — approach phase (last 4 nm before touchdown)")
    approach = states[(states["dist_from_td_nm"] >= -4) & (states["dist_from_td_nm"] <= 0)]
    if len(approach) > 0:
        cols = st.columns(5)
        for i, (col, label) in enumerate([
            ("fn1_configure_aircraft", "F1 Configure"),
            ("fn2_maintain_glidepath", "F2 Glidepath"),
            ("fn3_manage_energy", "F3 Energy"),
            ("fn4_maintain_stability", "F4 Stability"),
            ("fn5_maintain_lateral_path", "F5 Lateral"),
        ]):
            with cols[i]:
                st.caption(f"**{label}**")
                counts = approach[col].value_counts()
                pct = (counts / counts.sum() * 100).round(1)
                disp = pd.DataFrame({"State": counts.index, "%": pct.values})
                st.dataframe(disp, width="stretch", hide_index=True)

    # Bounded LLM layer
    st.divider()
    st.subheader("🤖 Bounded LLM interpretive layer")
    st.write(
        "Per the FRAIXL-Air proposal, the LLM is a bounded analyst-support "
        "layer that *explains* what the formal model has detected. "
        "It is not the monitoring engine."
    )

    if not api_key:
        st.info("Provide your Anthropic API key in the sidebar to enable this panel.")
    else:
        if st.button("Generate analyst summary with Claude", type="primary"):
            gate_info = stabilization_gate_assessment(states, cal)
            approach_window = states[(states["dist_from_td_nm"] >= -4) & (states["dist_from_td_nm"] <= 0)]
            payload = {
                "calibration_profile": cal.as_dict(),
                "flight_summary": summarize_flight(canonical, events),
                "events": {k: {kk: vv for kk, vv in v.items() if kk != "note"}
                           for k, v in events.items()},
                "stabilization_gate": gate_info,
                "approach_phase_state_distribution": {
                    "fn1_configure_aircraft": approach_window["fn1_configure_aircraft"].value_counts().to_dict(),
                    "fn2_maintain_glidepath": approach_window["fn2_maintain_glidepath"].value_counts().to_dict(),
                    "fn3_manage_energy": approach_window["fn3_manage_energy"].value_counts().to_dict(),
                    "fn4_maintain_stability": approach_window["fn4_maintain_stability"].value_counts().to_dict(),
                    "fn5_maintain_lateral_path": approach_window["fn5_maintain_lateral_path"].value_counts().to_dict(),
                },
            }

            system_prompt = """You are an analyst-support layer within the FRAIXL \
FRAM Digital Twin Landing Phase Demonstrator. The formal FRAM-based model has \
already computed five function states (Configure aircraft, Maintain glidepath, \
Manage energy state, Maintain stability, Maintain lateral path) and a binary \
stabilization-gate assessment from FDR data.

Your job is NOT to monitor or judge. Your job is to explain what the model has \
detected in operationally meaningful terms.

Write a concise analyst-facing interpretation that:
1. States the binary stable-approach verdict at the 1000 ft AGL gate
2. Describes the landing in functional terms across the five FRAM functions
3. Notes any function states that warrant analyst attention, framed neutrally \
as "the model classifies X as Y because Z" rather than as judgements about the crew
4. Acknowledges that thresholds are provisional pending calibration against a \
corpus of reviewed normal landings (WP4 of the FRAIXL-Air proposal)
5. Stays grounded in the structured data provided; do not invent details

Keep it to 3-4 short paragraphs. Use the non-blaming, systems-oriented framing \
appropriate to FRAM."""

            user_msg = (
                "Here is the structured Landing Phase Twin output for one flight:\n\n"
                + json.dumps(payload, indent=2, default=str)
            )

            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                with st.spinner("Claude is interpreting..."):
                    msg = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=1500,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_msg}],
                    )
                text = "".join(b.text for b in msg.content if hasattr(b, "text"))
                st.markdown("### Analyst summary")
                st.markdown(text)
                with st.expander("View structured payload sent to Claude"):
                    st.json(payload)
            except ImportError:
                st.error("`anthropic` package not installed. Run: `pip install anthropic`")
            except Exception as e:
                st.error(f"API call failed: {e}")
