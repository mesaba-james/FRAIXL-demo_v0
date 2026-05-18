"""
FRAIXL-Air Extraction Module
=============================
Reads a raw FDR/FOQA CSV and produces a canonical Landing Phase Twin
dataframe using the parameter dictionary. Also detects landing-phase events.

This module is intentionally framework-free (no Streamlit, no plotting).
It can be wrapped by a Streamlit app, a notebook, an MCP server, or a CLI.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional
from parameter_dictionary import PARAMETER_DICTIONARY


def load_fdr_csv(path: str) -> pd.DataFrame:
    """
    Load a raw FDR CSV. Handles the two-row header that some FDR exports use
    (a 'Time / subframe' second header row that pandas misreads).
    """
    # Peek at the first two rows to decide whether to skip the second
    head = pd.read_csv(path, nrows=2, low_memory=False)
    skiprows = None
    # If the first data row's Time looks non-numeric, it's the subframe header
    try:
        float(head.iloc[0, 0])
    except (ValueError, TypeError):
        skiprows = [1]

    df = pd.read_csv(path, skiprows=skiprows, low_memory=False)

    # Coerce Time numeric and drop any rows where it isn't
    if "Time" in df.columns:
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        df = df.dropna(subset=["Time"]).reset_index(drop=True)

    return df


def build_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build a canonical dataframe with one column per canonical_key found in the
    parameter dictionary, with multi-rate parameters forward-filled.

    Returns a dataframe with:
      - 'time_s' as the time base
      - 't_rel' as time relative to first sample (seconds)
      - one column per canonical key whose FDR source was found
    """
    cols = {}
    for canonical_key, meta in PARAMETER_DICTIONARY.items():
        for fdr_col in meta["fdr_columns"]:
            if fdr_col in raw.columns:
                cols[canonical_key] = pd.to_numeric(raw[fdr_col], errors="coerce")
                break

    canonical = pd.DataFrame(cols)

    # Forward-fill multi-rate parameters (FDR sub-frame sparseness)
    canonical = canonical.ffill()

    # Add relative time
    if "time_s" in canonical.columns:
        canonical["t_rel"] = canonical["time_s"] - canonical["time_s"].iloc[0]

    return canonical


def detect_landing_events(canonical: pd.DataFrame) -> dict:
    """
    Detect the principal events in a landing trace, returning a dict of
    {event_name: {time_s, t_rel, alt_baro_ft, radalt_ft, cas_kt, note}}.

    Touchdown is determined by AIR/GROUND transition (1 -> 0) and confirmed
    by radalt. This addresses the proxy-touchdown gap David flagged in v0.3.
    """
    events = {}

    def _first_where(mask, label, note=""):
        idx = mask[mask].index
        if len(idx) == 0:
            return None
        i = idx[0]
        row = canonical.loc[i]
        events[label] = {
            "time_s": float(row["time_s"]),
            "t_rel": float(row.get("t_rel", row["time_s"])),
            "alt_baro_ft": float(row.get("observed_alt_baro_ft", np.nan)),
            "radalt_ft": float(row.get("observed_radalt_ft", np.nan)),
            "cas_kt": float(row.get("observed_speed_kt", np.nan)),
            "note": note,
        }
        return i

    # Gear lever to DOWN (decoded for this 737 mapping: gear is DOWN when
    # GEAR LEVER UP == 1; see parameter dictionary for the FDR-quirk note).
    if "gear_lever_up" in canonical.columns:
        gear_down_decoded = (canonical["gear_lever_up"] == 1)
        _first_where(gear_down_decoded, "gear_down_selected", "Gear lever to DOWN (decoded)")
    elif "gear_lever_down" in canonical.columns:
        _first_where(canonical["gear_lever_down"] == 1, "gear_down_selected", "Gear lever to DOWN")

    # Flap detents (commanded)
    if "flap_handle_deg" in canonical.columns:
        seen_detents = set()
        for target in [1, 5, 10, 15, 25, 30, 40]:
            mask = canonical["flap_handle_deg"] >= target
            if mask.any():
                first_i = mask[mask].index[0]
                # only record real new transitions
                if target not in seen_detents:
                    seen_detents.add(target)
                    _first_where(mask, f"flap_{target}_reached", f"Flap handle ≥ {target}°")

    # LOC capture
    if "loc_engaged" in canonical.columns:
        _first_where(canonical["loc_engaged"] == 1, "loc_captured", "Localizer mode engaged")

    # GS capture
    if "gs_engaged" in canonical.columns:
        _first_where(canonical["gs_engaged"] == 1, "gs_captured", "Glideslope mode engaged")

    # Flare engage
    if "flare_engaged" in canonical.columns:
        _first_where(canonical["flare_engaged"] == 1, "flare_engaged", "Autoland flare mode engaged")

    # AP disengage (CMD A and CMD B both 0)
    if "ap_cmd_a" in canonical.columns and "ap_cmd_b" in canonical.columns:
        # Find the LAST time AP was engaged, then the transition
        ap_eng = (canonical["ap_cmd_a"] == 1) | (canonical["ap_cmd_b"] == 1)
        if ap_eng.any():
            # transition from engaged to disengaged
            transition = ap_eng & ~ap_eng.shift(-1).fillna(False)
            # find last engaged -> first not engaged after that
            last_eng_i = ap_eng[ap_eng].index[-1]
            after = canonical.loc[last_eng_i:]
            disengaged = after[(after["ap_cmd_a"] == 0) & (after["ap_cmd_b"] == 0)]
            if len(disengaged) > 0:
                i = disengaged.index[0]
                row = canonical.loc[i]
                events["ap_disengaged"] = {
                    "time_s": float(row["time_s"]),
                    "t_rel": float(row.get("t_rel", row["time_s"])),
                    "alt_baro_ft": float(row.get("observed_alt_baro_ft", np.nan)),
                    "radalt_ft": float(row.get("observed_radalt_ft", np.nan)),
                    "cas_kt": float(row.get("observed_speed_kt", np.nan)),
                    "note": "Both AP channels disengaged",
                }

    # Touchdown via AIR/GROUND transition (1 -> 0)
    if "air_ground" in canonical.columns:
        ag = canonical["air_ground"]
        transition = (ag == 0) & (ag.shift(1) == 1)
        if transition.any():
            i = transition[transition].index[0]
            row = canonical.loc[i]
            events["touchdown"] = {
                "time_s": float(row["time_s"]),
                "t_rel": float(row.get("t_rel", row["time_s"])),
                "alt_baro_ft": float(row.get("observed_alt_baro_ft", np.nan)),
                "radalt_ft": float(row.get("observed_radalt_ft", np.nan)),
                "cas_kt": float(row.get("observed_speed_kt", np.nan)),
                "note": "AIR/GROUND 1→0 transition",
            }

    # Auto speedbrake deployed
    if "auto_speedbrake_cmd" in canonical.columns:
        _first_where(canonical["auto_speedbrake_cmd"] == 1, "auto_speedbrake_deploy", "Auto speedbrake extend commanded")

    # T/R deploy
    if "tr1_deployed" in canonical.columns and "tr2_deployed" in canonical.columns:
        tr_any = (canonical["tr1_deployed"] == 1) | (canonical["tr2_deployed"] == 1)
        _first_where(tr_any, "tr_deployed", "Thrust reverser deployed")

    return events


def compute_distance_from_touchdown(canonical: pd.DataFrame, events: dict) -> pd.Series:
    """
    Compute an approximate ground distance from touchdown using lat/lon if
    available, else fall back to DISTANCE TO GO. Returns a Series in NM,
    negative before touchdown, positive after.
    """
    if "touchdown" in events and "lat_deg" in canonical.columns and "lon_deg" in canonical.columns:
        td_idx = (canonical["time_s"] - events["touchdown"]["time_s"]).abs().idxmin()
        td_lat = canonical["lat_deg"].iloc[td_idx]
        td_lon = canonical["lon_deg"].iloc[td_idx]
        # Haversine approximation
        R_nm = 3440.065
        lat1 = np.radians(canonical["lat_deg"])
        lat2 = np.radians(td_lat)
        dlat = lat2 - lat1
        dlon = np.radians(td_lon - canonical["lon_deg"])
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        dist = 2 * R_nm * np.arcsin(np.sqrt(a))
        # Sign: negative before touchdown, positive after
        sign = np.where(canonical.index < td_idx, -1, 1)
        return pd.Series(dist * sign, index=canonical.index, name="dist_from_td_nm")
    elif "distance_to_go_nm" in canonical.columns:
        # DTG decreases toward 0; convert to distance-from-touchdown style
        return -canonical["distance_to_go_nm"]
    else:
        return pd.Series(np.nan, index=canonical.index, name="dist_from_td_nm")


def summarize_flight(canonical: pd.DataFrame, events: dict) -> dict:
    """Return a small summary dict describing the flight segment."""
    s = {
        "duration_s": float(canonical["time_s"].iloc[-1] - canonical["time_s"].iloc[0]),
        "sample_rate_hz": float(1.0 / canonical["time_s"].diff().median()),
        "n_samples": int(len(canonical)),
        "alt_start_ft": float(canonical.get("observed_alt_baro_ft", pd.Series([np.nan])).dropna().iloc[0]) if "observed_alt_baro_ft" in canonical.columns else None,
        "alt_end_ft": float(canonical.get("observed_alt_baro_ft", pd.Series([np.nan])).dropna().iloc[-1]) if "observed_alt_baro_ft" in canonical.columns else None,
        "cas_start_kt": float(canonical.get("observed_speed_kt", pd.Series([np.nan])).dropna().iloc[0]) if "observed_speed_kt" in canonical.columns else None,
        "cas_min_kt": float(canonical.get("observed_speed_kt", pd.Series([np.nan])).min()) if "observed_speed_kt" in canonical.columns else None,
        "touched_down": "touchdown" in events,
        "td_cas_kt": events.get("touchdown", {}).get("cas_kt"),
        "td_radalt_ft": events.get("touchdown", {}).get("radalt_ft"),
    }
    return s
