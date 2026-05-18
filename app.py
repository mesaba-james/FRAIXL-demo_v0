"""
FRAIXL-Air Landing Phase Twin — Streamlit Demonstrator
=======================================================

A proof-of-concept Streamlit app that turns a raw FDR/FOQA CSV into a
FRAIXL-Air Landing Phase Twin v0.3 view.

Architecture (matching the FRAIXL-Air proposal):
  Tab 1 — Raw & Parameter Dictionary  (WP2: data-to-model dictionary)
  Tab 2 — Flight Profile              (visualization of observed metadata)
  Tab 3 — FRAM Twin                   (derived function states, v0.3)
  Tab 4 — Calibration & Discrepancy   (predictor-corrector layer + LLM)

Run:  streamlit run app.py
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

from parameter_dictionary import PARAMETER_DICTIONARY, list_by_function
from fraixl_extract import (
    load_fdr_csv,
    build_canonical,
    detect_landing_events,
    compute_distance_from_touchdown,
    summarize_flight,
)
from fram_metadata import CalibrationProfile, compute_states, stabilization_gate_assessment


# ============================================================================
# Page setup
# ============================================================================

st.set_page_config(
    page_title="FRAIXL-Air Landing Phase Twin",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ FRAIXL-Air Landing Phase Twin — Demonstrator")
st.caption(
    "A FRAM-informed Epistemic Digital Twin demonstrator for the approach and landing phase. "
    "Implements the Landing Phase Twin v0.3 metadata equations against real FDR data."
)


# ============================================================================
# Sidebar — file upload & calibration profile
# ============================================================================

with st.sidebar:
    st.header("📂 Flight data")

    # Discover bundled sample flights in the ./samples folder
    samples_dir = "samples"
    sample_files = []
    if os.path.isdir(samples_dir):
        sample_files = sorted([f for f in os.listdir(samples_dir) if f.lower().endswith(".csv")])

    sample_choice = None
    if sample_files:
        options = ["— Upload my own —"] + sample_files
        sample_choice = st.selectbox(
            "Sample flight",
            options,
            index=1 if len(sample_files) >= 1 else 0,
            help="Pre-loaded sample flights you can try without uploading anything.",
        )

    uploaded = None
    if not sample_files or sample_choice == "— Upload my own —":
        uploaded = st.file_uploader(
            "Upload an FDR/FOQA CSV", type=["csv"],
            help="ARINC 717-style export with one row per sub-frame sample."
        )

    st.divider()
    st.header("⚙️ Calibration profile")
    st.caption(
        "**Provisional thresholds.** Per FRAIXL-Air WP4, these must be "
        "replaced by calibration from reviewed normal landings before "
        "being treated as a divergence detector."
    )

    with st.expander("Energy", expanded=False):
        e_normal = st.slider("Normal band (±kt)", 1.0, 15.0, 5.0, 0.5)
        e_caution = st.slider("Caution band (±kt)", 5.0, 25.0, 10.0, 0.5)

    with st.expander("Vertical profile", expanded=False):
        v_normal = st.slider("Normal max |vrate| (fpm)", 400, 1500, 900, 50)
        v_caution = st.slider("Caution max |vrate| (fpm)", 600, 2000, 1200, 50)
        fpa_target = st.slider("Target FPA (deg)", -5.0, -2.0, -3.0, 0.1)
        fpa_tol = st.slider("FPA tolerance (deg)", 0.1, 1.5, 0.5, 0.1)

    with st.expander("Configuration", expanded=False):
        landing_flap = st.selectbox("Required landing flap (deg)", [15, 25, 30, 40], index=2)
        stab_gate = st.slider("Stabilization gate (ft AGL)", 500, 1500, 1000, 50)

    with st.expander("Stability weights", expanded=False):
        w_energy = st.slider("Energy weight", 0.0, 1.0, 0.30, 0.05)
        w_vert = st.slider("Vertical weight", 0.0, 1.0, 0.30, 0.05)
        w_conf = st.slider("Configuration weight", 0.0, 1.0, 0.25, 0.05)
        w_ctrl = st.slider("Control mode weight", 0.0, 1.0, 0.15, 0.05)
        total_w = w_energy + w_vert + w_conf + w_ctrl
        if abs(total_w - 1.0) > 0.001:
            st.warning(f"Weights sum to {total_w:.2f} (not 1.0). Scores will be unnormalized.")

    cal = CalibrationProfile(
        energy_normal_band_kt=e_normal,
        energy_caution_band_kt=e_caution,
        vrate_normal_max_fpm=v_normal,
        vrate_caution_max_fpm=v_caution,
        fpa_target_deg=fpa_target,
        fpa_tolerance_deg=fpa_tol,
        required_landing_flap_deg=landing_flap,
        stabilization_gate_radalt_ft=stab_gate,
        energy_weight=w_energy,
        vertical_weight=w_vert,
        configuration_weight=w_conf,
        control_mode_weight=w_ctrl,
    )

    st.divider()
    st.header("🤖 Bounded LLM layer")
    st.caption(
        "Optional. Uses Claude (Anthropic API) as the bounded interpretive "
        "layer on top of the formal model — exactly as the FRAIXL-Air "
        "proposal architects it."
    )

    # Priority: Streamlit secrets > env var > user input
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
        api_key = st.text_input(
            "Anthropic API key",
            type="password",
            value=env_key,
            help="Stored only for this session. Required for the analyst-summary tab.",
        )


# ============================================================================
# Load data
# ============================================================================

@st.cache_data(show_spinner="Loading FDR CSV...")
def _load(path_or_bytes, is_path: bool):
    if is_path:
        return load_fdr_csv(path_or_bytes)
    # uploaded file
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
    cal_obj = CalibrationProfile(**cal_dict)
    return compute_states(_canonical, cal_obj)


# Decide what to load
raw = None
source_label = None

if uploaded is not None:
    raw = _load(uploaded.getvalue(), is_path=False)
    source_label = uploaded.name
elif sample_choice and sample_choice != "— Upload my own —":
    sample_path = os.path.join(samples_dir, sample_choice)
    raw = _load(sample_path, is_path=True)
    source_label = f"Sample: {sample_choice}"
elif not sample_files:
    # No samples and no upload yet
    pass

if raw is None:
    st.info("👈 Upload a CSV in the sidebar to begin, or pick a sample flight.")
    st.stop()

canonical, events = _process(raw)
states = _states(canonical, asdict(cal))

st.caption(f"📊 Source: **{source_label}** — {raw.shape[0]:,} rows × {raw.shape[1]:,} cols → "
           f"{canonical.shape[1]} canonical metadata fields")


# ============================================================================
# Tabs
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Raw & Parameter Dictionary",
    "2️⃣ Flight Profile",
    "3️⃣ FRAM Twin (v0.3)",
    "4️⃣ Calibration & Discrepancy",
])


# ----------------------------------------------------------------------------
# TAB 1 — Raw data + parameter dictionary
# ----------------------------------------------------------------------------

with tab1:
    st.header("Raw FDR data and data-to-model dictionary")
    st.write(
        "This tab implements the *data-to-model dictionary* called for in "
        "WP2 of the FRAIXL-Air proposal: the explicit mapping from raw FDR "
        "column names to canonical metadata keys used by the Twin. The "
        "dictionary is the artefact you can take and reuse in the .xfmv "
        "environment."
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
        matched = None
        for fdr_col in meta["fdr_columns"]:
            if fdr_col in raw.columns:
                matched = fdr_col
                break
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

    # Download the dictionary as JSON
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
        help="The artefact David can reuse in the .xfmv metadata environment.",
    )

    with st.expander("Show raw column inventory (all 1,182+ columns)"):
        st.write(f"Total raw columns: **{len(raw.columns)}**")
        st.dataframe(
            pd.DataFrame({"Column": raw.columns}),
            width="stretch", hide_index=True, height=400,
        )

    with st.expander("Show canonical metadata (first 200 rows)"):
        st.dataframe(canonical.head(200), width="stretch")


# ----------------------------------------------------------------------------
# TAB 2 — Flight profile visualizations
# ----------------------------------------------------------------------------

with tab2:
    st.header("Flight profile")
    st.write(
        "Vertical profile, energy state, configuration timeline, and "
        "ground track. Event markers reflect the discrete-based event "
        "detection in `fraixl_extract.detect_landing_events`."
    )

    # ----- Vertical profile -----
    st.subheader("Vertical profile (altitude vs distance from touchdown)")

    fig_prof = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=("Altitude profile", "Vertical speed"),
    )

    x = canonical["dist_from_td_nm"]
    if "observed_radalt_ft" in canonical.columns:
        fig_prof.add_trace(
            go.Scatter(x=x, y=canonical["observed_radalt_ft"],
                       mode="lines", name="Radio altitude",
                       line=dict(color="#1f77b4", width=2)),
            row=1, col=1,
        )
    if "observed_alt_baro_ft" in canonical.columns:
        fig_prof.add_trace(
            go.Scatter(x=x, y=canonical["observed_alt_baro_ft"],
                       mode="lines", name="Baro altitude",
                       line=dict(color="#aec7e8", width=1, dash="dot")),
            row=1, col=1,
        )

    # Reference 3° glideslope from touchdown
    # 3 deg ≈ 318 ft per nm; line goes from x=-6, y=318*6
    gs_x = np.array([-6, 0])
    gs_y = -np.tan(np.radians(3.0)) * gs_x * 6076.12  # nm to ft
    fig_prof.add_trace(
        go.Scatter(x=gs_x, y=gs_y, mode="lines", name="3° reference",
                   line=dict(color="gray", dash="dash", width=1)),
        row=1, col=1,
    )

    # Vertical speed
    if "observed_vrate_fpm" in canonical.columns:
        fig_prof.add_trace(
            go.Scatter(x=x, y=canonical["observed_vrate_fpm"],
                       mode="lines", name="Vertical speed",
                       line=dict(color="#ff7f0e")),
            row=2, col=1,
        )
        fig_prof.add_hline(y=0, line_color="gray", line_dash="dot", row=2, col=1)
        fig_prof.add_hline(y=-cal.vrate_normal_max_fpm, line_color="orange",
                           line_dash="dash", row=2, col=1,
                           annotation_text=f"-{cal.vrate_normal_max_fpm:.0f} fpm")

    # Event markers
    event_color = {
        "gear_down_selected": "#2ca02c",
        "flap_5_reached": "#9467bd",
        "flap_10_reached": "#9467bd",
        "flap_15_reached": "#9467bd",
        "flap_25_reached": "#9467bd",
        "flap_30_reached": "#9467bd",
        "loc_captured": "#17becf",
        "gs_captured": "#17becf",
        "flare_engaged": "#d62728",
        "touchdown": "#000000",
        "ap_disengaged": "#8c564b",
        "tr_deployed": "#e377c2",
        "auto_speedbrake_deploy": "#7f7f7f",
    }
    for name, ev in events.items():
        # Need x position — interpolate via time
        td_idx = (canonical["time_s"] - ev["time_s"]).abs().idxmin()
        ev_x = canonical["dist_from_td_nm"].iloc[td_idx]
        ev_y = ev["radalt_ft"] if pd.notna(ev["radalt_ft"]) else ev["alt_baro_ft"]
        col = event_color.get(name, "#666666")
        fig_prof.add_trace(
            go.Scatter(
                x=[ev_x], y=[ev_y],
                mode="markers+text",
                marker=dict(size=10, color=col, symbol="diamond"),
                text=[name.replace("_", " ")],
                textposition="top center",
                textfont=dict(size=9),
                name=name, showlegend=False,
                hovertemplate=f"<b>{name}</b><br>t={ev['time_s']:.1f}s<br>"
                              f"radalt={ev['radalt_ft']:.0f}ft<br>"
                              f"CAS={ev['cas_kt']:.0f}kt<extra></extra>",
            ),
            row=1, col=1,
        )

    fig_prof.update_xaxes(title_text="Distance from touchdown (nm)", row=2, col=1)
    fig_prof.update_yaxes(title_text="Altitude (ft)", row=1, col=1)
    fig_prof.update_yaxes(title_text="Vertical speed (fpm)", row=2, col=1)
    fig_prof.update_layout(height=620, hovermode="x unified", showlegend=True)
    st.plotly_chart(fig_prof, width="stretch")

    # ----- Speed/energy plot -----
    st.subheader("Energy state — CAS vs target")
    fig_spd = go.Figure()
    if "observed_speed_kt" in canonical.columns:
        fig_spd.add_trace(go.Scatter(
            x=x, y=canonical["observed_speed_kt"],
            mode="lines", name="CAS", line=dict(color="#1f77b4", width=2),
        ))
    if "target_speed_kt" in canonical.columns:
        fig_spd.add_trace(go.Scatter(
            x=x, y=canonical["target_speed_kt"],
            mode="lines", name="Target", line=dict(color="green", dash="dash"),
        ))
    if "vref_kt" in canonical.columns and canonical["vref_kt"].notna().any():
        fig_spd.add_trace(go.Scatter(
            x=x, y=canonical["vref_kt"],
            mode="lines", name="Vref", line=dict(color="red", dash="dot"),
        ))
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
        fig_cfg.add_trace(go.Scatter(
            x=x, y=canonical["flap_handle_deg"],
            mode="lines", name="Flap handle", line=dict(color="#9467bd", width=2),
        ), row=1, col=1)
    if "gear_lever_down" in canonical.columns:
        fig_cfg.add_trace(go.Scatter(
            x=x, y=canonical["gear_lever_down"],
            mode="lines", name="Gear lever DOWN", line=dict(color="#2ca02c"),
        ), row=2, col=1)
    if "air_ground" in canonical.columns:
        fig_cfg.add_trace(go.Scatter(
            x=x, y=canonical["air_ground"],
            mode="lines", name="AIR/GROUND (1=air)", line=dict(color="#d62728", dash="dot"),
        ), row=2, col=1)
    fig_cfg.update_xaxes(title_text="Distance from touchdown (nm)", row=2, col=1)
    fig_cfg.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig_cfg, width="stretch")

    # ----- Ground track -----
    if "lat_deg" in canonical.columns and "lon_deg" in canonical.columns:
        st.subheader("Ground track")

        # Pick the right Plotly trace and layout keys based on installed version.
        # Plotly >= 5.24 introduced go.Scattermap (MapLibre);
        # earlier versions use go.Scattermapbox.
        use_new_map = hasattr(go, "Scattermap")
        if use_new_map:
            MapTrace = go.Scattermap
            map_layout_key = "map"
        else:
            MapTrace = go.Scattermapbox
            map_layout_key = "mapbox"

        fig_map = go.Figure()
        fig_map.add_trace(MapTrace(
            lat=canonical["lat_deg"], lon=canonical["lon_deg"],
            mode="lines+markers",
            marker=dict(size=4, color=canonical["observed_radalt_ft"], colorscale="Viridis",
                        showscale=True, colorbar=dict(title="Radalt (ft)")),
            line=dict(width=2),
            name="Track",
        ))
        if "touchdown" in events:
            td_idx = (canonical["time_s"] - events["touchdown"]["time_s"]).abs().idxmin()
            fig_map.add_trace(MapTrace(
                lat=[canonical["lat_deg"].iloc[td_idx]],
                lon=[canonical["lon_deg"].iloc[td_idx]],
                mode="markers",
                marker=dict(size=14, color="red"),
                name="Touchdown",
            ))
        fig_map.update_layout(
            **{map_layout_key: dict(
                style="open-street-map",
                center=dict(lat=canonical["lat_deg"].median(),
                            lon=canonical["lon_deg"].median()),
                zoom=10,
            )},
            height=500, margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_map, width="stretch")

    # Event table
    st.subheader("Detected events")
    if events:
        ev_df = pd.DataFrame([
            {"Event": k, "Time (s)": v["time_s"], "Radalt (ft)": v["radalt_ft"],
             "CAS (kt)": v["cas_kt"], "Note": v["note"]}
            for k, v in sorted(events.items(), key=lambda x: x[1]["time_s"])
        ])
        st.dataframe(ev_df, width="stretch", hide_index=True)


# ----------------------------------------------------------------------------
# TAB 3 — FRAM Twin v0.3 derived states
# ----------------------------------------------------------------------------

with tab3:
    st.header("FRAM Twin — derived function states")
    st.write(
        "Implementation of the Landing Phase Twin v0.3 metadata equation "
        "set. Each function output below is **derived** from observed "
        "metadata using the calibration profile in the sidebar; the "
        "underlying parameters are shown alongside so you can see *why* "
        "each state is what it is."
    )

    # Composite stability over time
    st.subheader("Composite landing stability")
    fig_stab = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
        vertical_spacing=0.07,
        subplot_titles=("Stability score (0–1)", "Component scores"),
    )
    x = canonical["dist_from_td_nm"]

    fig_stab.add_trace(go.Scatter(
        x=x, y=states["landing_stability_score"],
        mode="lines", name="Composite",
        line=dict(color="#1f77b4", width=3),
        fill="tozeroy",
    ), row=1, col=1)
    fig_stab.add_hline(y=cal.stable_min_score, line_color="green",
                       line_dash="dash", row=1, col=1,
                       annotation_text=f"STABLE ≥ {cal.stable_min_score}")
    fig_stab.add_hline(y=cal.cautionary_min_score, line_color="orange",
                       line_dash="dash", row=1, col=1,
                       annotation_text=f"CAUTIONARY ≥ {cal.cautionary_min_score}")

    for col, label, color in [
        ("energy_score", "Energy", "#ff7f0e"),
        ("vertical_score", "Vertical", "#2ca02c"),
        ("configuration_score", "Config", "#9467bd"),
        ("control_mode_score", "Control mode", "#8c564b"),
    ]:
        fig_stab.add_trace(go.Scatter(
            x=x, y=states[col], mode="lines", name=label,
            line=dict(color=color, width=1.5),
        ), row=2, col=1)

    fig_stab.update_xaxes(title_text="Distance from touchdown (nm)", row=2, col=1)
    fig_stab.update_yaxes(title_text="Score", range=[0, 1.05], row=1, col=1)
    fig_stab.update_yaxes(title_text="Component score", range=[0, 1.05], row=2, col=1)
    fig_stab.update_layout(height=560, hovermode="x unified")
    st.plotly_chart(fig_stab, width="stretch")

    # State distribution
    st.subheader("Function state distribution over the segment")
    cols = st.columns(4)
    for i, (col, label) in enumerate([
        ("landing_energy_state", "Energy"),
        ("landing_vertical_profile_state", "Vertical"),
        ("landing_configuration_state", "Configuration"),
        ("landing_control_mode_state", "Control mode"),
    ]):
        with cols[i]:
            st.caption(f"**{label}**")
            counts = states[col].value_counts()
            pct = (counts / counts.sum() * 100).round(1)
            disp = pd.DataFrame({"State": counts.index, "%": pct.values})
            st.dataframe(disp, width="stretch", hide_index=True)

    # Stabilization gate assessment
    st.subheader(f"Stabilization gate assessment ({cal.stabilization_gate_radalt_ft:.0f} ft AGL)")
    gate = stabilization_gate_assessment(states, cal)
    if gate.get("available"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Energy", gate["energy_state"], f"{gate['speed_error_kt']:+.1f} kt error")
            st.metric("Vertical", gate["vertical_state"], f"{gate['vrate_fpm']:.0f} fpm")
        with c2:
            st.metric("Configuration", gate["configuration_state"],
                      f"flap {gate['flap_deg']:.0f}°")
            st.metric("Control mode", gate["control_mode_state"])
        with c3:
            st.metric("Stability score", f"{gate['stability_score']:.3f}",
                      gate["stability_assessment"])
            st.metric("CAS at gate", f"{gate['cas_kt']:.0f} kt")
    else:
        st.warning(f"Gate assessment unavailable: {gate.get('reason')}")

    # State table around touchdown
    if "touchdown" in events:
        st.subheader("Function states near touchdown")
        td_time = events["touchdown"]["time_s"]
        window = states[(states["time_s"] >= td_time - 15) & (states["time_s"] <= td_time + 5)]
        show_cols = [
            "time_s", "observed_radalt_ft", "observed_speed_kt",
            "speed_error_kt", "observed_vrate_fpm",
            "landing_energy_state", "landing_vertical_profile_state",
            "landing_configuration_state", "landing_control_mode_state",
            "landing_stability_score", "landing_stability_assessment",
        ]
        show_cols = [c for c in show_cols if c in window.columns]
        # Subsample to every 4th row for readability
        st.dataframe(window[show_cols].iloc[::4], width="stretch", hide_index=True)


# ----------------------------------------------------------------------------
# TAB 4 — Calibration & discrepancy
# ----------------------------------------------------------------------------

with tab4:
    st.header("Calibration & discrepancy")
    st.write(
        "This tab makes the predictor-corrector logic explicit. The "
        "Expected Twin is generated from the calibration profile; the "
        "Actual Twin is generated from the FDR data; the delta is what "
        "the system would interpret as functional divergence — but only "
        "*after* calibration against reviewed normal landings is complete."
    )

    st.subheader("Active calibration profile")
    cal_df = pd.DataFrame(
        [{"Parameter": k, "Value": v} for k, v in cal.as_dict().items()]
    )
    st.dataframe(cal_df, width="stretch", hide_index=True)

    cal_json = json.dumps(cal.as_dict(), indent=2)
    st.download_button(
        "📥 Download calibration profile (JSON)",
        data=cal_json,
        file_name="fraixl_calibration_profile.json",
        mime="application/json",
    )

    # Delta view
    st.subheader("Predicted vs observed — energy axis")
    st.caption("For this nominal flight, deltas should remain small. Large "
               "deltas would indicate either real divergence OR poor calibration; "
               "the FRAIXL-Air method requires calibration *before* discrepancies "
               "are treated as meaningful.")

    if "speed_error_kt" in states.columns:
        x = canonical["dist_from_td_nm"]
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(
            x=x, y=states["speed_error_kt"],
            mode="lines", name="CAS - Target (kt)",
            line=dict(color="#1f77b4", width=2),
        ))
        fig_d.add_hline(y=cal.energy_normal_band_kt, line_color="green",
                        line_dash="dash", annotation_text=f"+{cal.energy_normal_band_kt}")
        fig_d.add_hline(y=-cal.energy_normal_band_kt, line_color="green",
                        line_dash="dash", annotation_text=f"-{cal.energy_normal_band_kt}")
        fig_d.add_hline(y=cal.energy_caution_band_kt, line_color="orange", line_dash="dot")
        fig_d.add_hline(y=-cal.energy_caution_band_kt, line_color="orange", line_dash="dot")
        fig_d.add_hline(y=0, line_color="gray")
        fig_d.update_xaxes(title_text="Distance from touchdown (nm)")
        fig_d.update_yaxes(title_text="Speed error (kt)")
        fig_d.update_layout(height=350)
        st.plotly_chart(fig_d, width="stretch")

    # ----- Bounded LLM layer -----
    st.subheader("🤖 Bounded LLM interpretive layer")
    st.write(
        "Per the FRAIXL-Air proposal, the LLM is a bounded analyst-support "
        "layer that *explains* what the formal model has detected. It is "
        "not the monitoring engine. The button below sends the structured "
        "state summary (not raw flight data) to Claude and asks for an "
        "analyst-facing narrative interpretation."
    )

    if not api_key:
        st.info("Provide your Anthropic API key in the sidebar to enable this panel.")
    else:
        if st.button("Generate analyst summary with Claude", type="primary"):
            # Build a compact structured summary for the LLM
            gate_info = stabilization_gate_assessment(states, cal)
            payload = {
                "calibration_profile": cal.as_dict(),
                "flight_summary": summarize_flight(canonical, events),
                "events": {k: {kk: vv for kk, vv in v.items() if kk != "note"}
                           for k, v in events.items()},
                "stabilization_gate": gate_info,
                "state_distribution": {
                    "energy": states["landing_energy_state"].value_counts().to_dict(),
                    "vertical": states["landing_vertical_profile_state"].value_counts().to_dict(),
                    "configuration": states["landing_configuration_state"].value_counts().to_dict(),
                    "control_mode": states["landing_control_mode_state"].value_counts().to_dict(),
                    "stability": states["landing_stability_assessment"].value_counts().to_dict(),
                },
                "approach_phase_avg_scores": {
                    "energy": float(states["energy_score"].mean()),
                    "vertical": float(states["vertical_score"].mean()),
                    "configuration": float(states["configuration_score"].mean()),
                    "control_mode": float(states["control_mode_score"].mean()),
                    "composite": float(states["landing_stability_score"].mean()),
                },
            }

            system_prompt = """You are an analyst-support layer within the FRAIXL-Air \
Landing Phase Twin (v0.3). The formal FRAM-based model has already computed function \
states and identified events from FDR data. Your job is NOT to monitor or judge; your \
job is to explain what the model has detected in operationally meaningful terms.

Write a concise analyst-facing interpretation that:
1. Describes the landing in functional terms (energy, vertical profile, configuration, control mode)
2. Notes the stabilization gate assessment
3. Highlights any function states that warrant analyst attention, framed neutrally as \
"the model classifies X as Y because Z" rather than judgements about the crew
4. Acknowledges where calibration limitations apply (this is provisional thresholds, \
not calibrated against a normal-flight corpus)
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

    st.divider()
    st.caption(
        "**Note on architecture.** The Streamlit app talks to Claude via the "
        "Anthropic Python SDK. For a production version, this same extraction "
        "pipeline (`fraixl_extract.py` and `fram_metadata.py`) could be wrapped "
        "as an MCP server, so multiple consumers — analyst notebooks, the "
        ".xfmv runtime, or other LLMs — could hit the same logic. The module "
        "structure here is already framework-free to keep that option open."
    )
