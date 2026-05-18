"""
FRAIXL-Air Metadata Equations — v2
====================================
Simplified per pilot/FOQA-analyst review:

  - No composite stability score (binary thinking, not weighted scoring)
  - Five FRAM functions tracking the landing phase:
      1. Configure the aircraft   (flap + gear)
      2. Maintain glidepath       (GS deviation)
      3. Manage energy state      (CAS vs Vref window)
      4. Maintain stability       (continuous flap+gear+speed-window check)
      5. Maintain lateral path    (LOC deviation)

  - Stabilization gate at 1000 ft AGL: stable iff flap >= 30, gear DOWN,
    and CAS within [Vref, Vref+20]. Binary STABLE / UNSTABLE.

All thresholds live in the calibration profile. Per WP4 of the FRAIXL-Air
proposal these are PROVISIONAL until calibrated against a corpus of
reviewed normal landings.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass


# ----------------------------------------------------------------------------
# Calibration profile
# ----------------------------------------------------------------------------

@dataclass
class CalibrationProfile:
    """Provisional thresholds. Replace with calibrated values per WP4."""
    # --- Stable-approach speed window (Vref to Vref+max) ---
    speed_window_min_kt: float = 0.0     # CAS - Vref >= 0
    speed_window_max_kt: float = 20.0    # CAS - Vref <= 20

    # --- N1 threshold for energy state ---
    n1_min_pct: float = 40.0             # average N1 should be >= 40%

    # --- Configuration ---
    required_landing_flap_deg: float = 30.0

    # --- Stabilization gate (adjustable 500–1000 ft AGL) ---
    stabilization_gate_radalt_ft: float = 1000.0

    # --- Glidepath (function 2) ---
    # 1 dot = 0.0875 DDM full-scale glideslope deflection
    gs_one_dot_ddm: float = 0.0875
    gs_normal_dots: float = 1.0          # |dev| <= 1 dot = ON_GLIDEPATH
    gs_caution_dots: float = 2.0         # 1 < |dev| <= 2 = GP_DEVIATION

    # --- Lateral path (function 5) ---
    # 1 dot = 0.155 DDM full-scale localizer deflection
    loc_one_dot_ddm: float = 0.155
    loc_normal_dots: float = 1.0
    loc_caution_dots: float = 2.0

    # --- Control mode handover bounds (informational, not stability gating) ---
    ap_handover_min_radalt_ft: float = 50.0

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


# ----------------------------------------------------------------------------
# Function classifiers — each returns a state series
# ----------------------------------------------------------------------------

def _fn1_configure_aircraft(canonical: pd.DataFrame, cal: CalibrationProfile) -> pd.Series:
    """
    Function 1: Configure the aircraft.
    State labels: CONFIG_COMPLETE, CONFIG_INCOMPLETE.

    Gear-down rule (operator-specific, this 737 FDR mapping):
        gear_is_down = (GEAR LEVER UP == 1)
    See parameter dictionary entries for gear_lever_up / gear_lever_down
    for the FDR-quirk documentation. Other operators may decode differently;
    re-verify when reusing.
    """
    flap = canonical.get("flap_handle_deg", pd.Series(0.0, index=canonical.index)).fillna(0)
    gear_up = canonical.get("gear_lever_up", pd.Series(0.0, index=canonical.index)).fillna(0)

    flap_ok = flap >= cal.required_landing_flap_deg
    gear_ok = gear_up == 1   # decoded: lever NOT in UP detent => gear DOWN
    config_complete = flap_ok & gear_ok

    state = pd.Series("CONFIG_INCOMPLETE", index=canonical.index)
    state[config_complete] = "CONFIG_COMPLETE"
    return state


def _fn2_maintain_glidepath(canonical: pd.DataFrame, cal: CalibrationProfile) -> pd.Series:
    """
    Function 2: Maintain glidepath.
    State labels: ON_GLIDEPATH, GP_DEVIATION, GP_EXCURSION, UNKNOWN.
    """
    state = pd.Series("UNKNOWN", index=canonical.index)
    if "gs_deviation_ddm" not in canonical.columns:
        return state

    dev = canonical["gs_deviation_ddm"].abs()
    dots = dev / cal.gs_one_dot_ddm

    state[dots.notna() & (dots <= cal.gs_normal_dots)] = "ON_GLIDEPATH"
    state[(dots > cal.gs_normal_dots) & (dots <= cal.gs_caution_dots)] = "GP_DEVIATION"
    state[dots > cal.gs_caution_dots] = "GP_EXCURSION"
    return state


def _fn3_manage_energy(canonical: pd.DataFrame, cal: CalibrationProfile) -> pd.Series:
    """
    Function 3: Manage energy state.
    State labels: ON_TARGET, LOW_ENERGY, HIGH_ENERGY, LOW_N1, UNKNOWN.

    Energy is "ON_TARGET" when BOTH:
      - CAS - Vref is within [speed_window_min, speed_window_max], AND
      - average N1 >= n1_min_pct (thrust support adequate)

    Speed below the window -> LOW_ENERGY.
    Speed above the window -> HIGH_ENERGY.
    Speed in-window but N1 below threshold -> LOW_N1 (potentially low energy
    in spite of acceptable speed, e.g. spool-up time concerns).
    """
    state = pd.Series("UNKNOWN", index=canonical.index)
    if "observed_speed_kt" not in canonical.columns or "vref_kt" not in canonical.columns:
        return state

    cas = canonical["observed_speed_kt"]
    vref = canonical["vref_kt"]
    delta = cas - vref

    # Average N1 (graceful fallback if only one engine present)
    n1_cols = [c for c in ("n1_left_pct", "n1_right_pct") if c in canonical.columns]
    if n1_cols:
        n1_avg = canonical[n1_cols].mean(axis=1)
        n1_ok = n1_avg >= cal.n1_min_pct
    else:
        # No N1 data — treat as satisfied so the function still produces a state
        n1_ok = pd.Series(True, index=canonical.index)

    mask = delta.notna()
    speed_in_window = (delta >= cal.speed_window_min_kt) & (delta <= cal.speed_window_max_kt)

    state[mask & speed_in_window & n1_ok] = "ON_TARGET"
    state[mask & speed_in_window & ~n1_ok] = "LOW_N1"
    state[mask & (delta > cal.speed_window_max_kt)] = "HIGH_ENERGY"
    state[mask & (delta < cal.speed_window_min_kt)] = "LOW_ENERGY"
    return state


def _fn4_maintain_stability(canonical: pd.DataFrame, cal: CalibrationProfile) -> pd.Series:
    """
    Function 4: Maintain stability — continuous combination of the three
    flight-path stability axes:
      - lateral: on the localizer (function 5 state == ON_LATERAL)
      - vertical: on the glidepath (function 2 state == ON_GLIDEPATH)
      - speed: within [Vref, Vref+20]  (one of function 3's conditions)

    Configuration is NOT part of stability (handled by function 1 and by the
    1000 ft gate check separately).

    State labels: STABLE, UNSTABLE, UNKNOWN.
    """
    state = pd.Series("UNKNOWN", index=canonical.index)
    lat = _fn5_maintain_lateral(canonical, cal)
    vert = _fn2_maintain_glidepath(canonical, cal)

    # Speed in window (independent of N1 — N1 affects energy, not stability)
    if "observed_speed_kt" in canonical.columns and "vref_kt" in canonical.columns:
        delta = canonical["observed_speed_kt"] - canonical["vref_kt"]
        speed_ok = (delta >= cal.speed_window_min_kt) & (delta <= cal.speed_window_max_kt)
        speed_known = delta.notna()
    else:
        speed_ok = pd.Series(False, index=canonical.index)
        speed_known = pd.Series(False, index=canonical.index)

    lat_ok = lat == "ON_LATERAL"
    vert_ok = vert == "ON_GLIDEPATH"
    lat_known = lat != "UNKNOWN"
    vert_known = vert != "UNKNOWN"

    have_all = lat_known & vert_known & speed_known
    state[have_all & lat_ok & vert_ok & speed_ok] = "STABLE"
    state[have_all & ~(lat_ok & vert_ok & speed_ok)] = "UNSTABLE"
    return state


def _fn5_maintain_lateral(canonical: pd.DataFrame, cal: CalibrationProfile) -> pd.Series:
    """
    Function 5: Maintain lateral path.
    State labels: ON_LATERAL, LAT_DEVIATION, LAT_EXCURSION, UNKNOWN.
    """
    state = pd.Series("UNKNOWN", index=canonical.index)
    if "loc_deviation_ddm" not in canonical.columns:
        return state

    dev = canonical["loc_deviation_ddm"].abs()
    dots = dev / cal.loc_one_dot_ddm

    state[dots.notna() & (dots <= cal.loc_normal_dots)] = "ON_LATERAL"
    state[(dots > cal.loc_normal_dots) & (dots <= cal.loc_caution_dots)] = "LAT_DEVIATION"
    state[dots > cal.loc_caution_dots] = "LAT_EXCURSION"
    return state


def _control_mode(canonical: pd.DataFrame, cal: CalibrationProfile) -> pd.Series:
    """Supplementary: control mode (AP_COUPLED / AUTOLAND / MANUAL)."""
    cmd_a = canonical.get("ap_cmd_a", pd.Series(0, index=canonical.index)).fillna(0)
    cmd_b = canonical.get("ap_cmd_b", pd.Series(0, index=canonical.index)).fillna(0)
    flare_eng = canonical.get("flare_engaged", pd.Series(0, index=canonical.index)).fillna(0)
    ap_on = (cmd_a == 1) | (cmd_b == 1)
    state = pd.Series("MANUAL", index=canonical.index)
    state[ap_on] = "AP_COUPLED"
    state[ap_on & (flare_eng == 1)] = "AUTOLAND"
    return state


def _vertical_profile(canonical: pd.DataFrame, cal: CalibrationProfile) -> pd.Series:
    """
    Supplementary: vertical profile from GS deviation, used only to label
    'Vertical: ON_PROFILE / OFF_PROFILE' in the gate panel.
    """
    return _fn2_maintain_glidepath(canonical, cal).replace({
        "ON_GLIDEPATH": "ON_PROFILE",
        "GP_DEVIATION": "OFF_PROFILE",
        "GP_EXCURSION": "OFF_PROFILE",
    })


# ----------------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------------

def compute_states(canonical: pd.DataFrame, cal: CalibrationProfile | None = None) -> pd.DataFrame:
    """
    Compute the five FRAM function states (plus supplementary control-mode
    and vertical-profile labels) for the canonical flight data.
    """
    if cal is None:
        cal = CalibrationProfile()

    out = canonical.copy()

    # Auxiliary numeric (useful for plotting / panels)
    if "observed_speed_kt" in out.columns and "vref_kt" in out.columns:
        out["cas_minus_vref_kt"] = out["observed_speed_kt"] - out["vref_kt"]

    n1_cols = [c for c in ("n1_left_pct", "n1_right_pct") if c in out.columns]
    if n1_cols:
        out["n1_avg_pct"] = out[n1_cols].mean(axis=1)

    out["fn1_configure_aircraft"]    = _fn1_configure_aircraft(out, cal)
    out["fn2_maintain_glidepath"]    = _fn2_maintain_glidepath(out, cal)
    out["fn3_manage_energy"]         = _fn3_manage_energy(out, cal)
    out["fn4_maintain_stability"]    = _fn4_maintain_stability(out, cal)
    out["fn5_maintain_lateral_path"] = _fn5_maintain_lateral(out, cal)

    out["control_mode_state"]   = _control_mode(out, cal)
    out["vertical_profile_state"] = _vertical_profile(out, cal)

    return out


def stabilization_gate_assessment(state_df: pd.DataFrame, cal: CalibrationProfile) -> dict:
    """
    Binary stable-approach check at 1000 ft AGL:
      STABLE  iff  flap >= 30  AND  gear DOWN  AND  Vref <= CAS <= Vref+20.
    Returns a dict with the three sub-criteria and the overall verdict.
    """
    if "observed_radalt_ft" not in state_df.columns:
        return {"available": False, "reason": "no radio altitude"}

    radalt = state_df["observed_radalt_ft"]
    # First crossing of the gate from above
    crossings = (radalt < cal.stabilization_gate_radalt_ft) & \
                (radalt.shift(1) >= cal.stabilization_gate_radalt_ft)
    if not crossings.any():
        return {"available": False,
                "reason": f"flight did not cross {cal.stabilization_gate_radalt_ft:.0f} ft AGL"}

    i = crossings[crossings].index[0]
    row = state_df.loc[i]

    flap = float(row.get("flap_handle_deg", np.nan))
    gear_dn_raw = float(row.get("gear_lever_down", np.nan))
    gear_up_raw = float(row.get("gear_lever_up", np.nan))
    # Decoded operational gear-down for this 737 FDR mapping
    gear_is_down = (gear_up_raw == 1)
    cas = float(row.get("observed_speed_kt", np.nan))
    vref = float(row.get("vref_kt", np.nan))
    target = float(row.get("target_speed_kt", np.nan))
    vrate = float(row.get("observed_vrate_fpm", np.nan))
    n1_avg = float(row.get("n1_avg_pct", np.nan))

    # Three sub-checks
    config_ok = (flap >= cal.required_landing_flap_deg) and gear_is_down
    config_label = "OK" if config_ok else "LATE"

    delta = cas - vref if (not np.isnan(cas) and not np.isnan(vref)) else np.nan
    if not np.isnan(delta):
        energy_ok = (delta >= cal.speed_window_min_kt) and (delta <= cal.speed_window_max_kt)
        energy_label = "ON_TARGET" if energy_ok else ("HIGH_ENERGY" if delta > cal.speed_window_max_kt else "LOW_ENERGY")
    else:
        energy_ok = False
        energy_label = "UNKNOWN"

    overall = "STABLE" if (config_ok and energy_ok) else "UNSTABLE"

    return {
        "available": True,
        "gate_radalt_ft": float(cal.stabilization_gate_radalt_ft),
        "time_s": float(row["time_s"]),

        # Sub-criteria
        "configuration_state": config_label,
        "energy_state": energy_label,
        "vertical_state": row.get("vertical_profile_state", "UNKNOWN"),
        "control_mode_state": row.get("control_mode_state"),

        # Numeric evidence
        "flap_deg": flap,
        "gear_is_down": bool(gear_is_down),
        "gear_lever_up_raw": gear_up_raw,
        "gear_lever_down_raw": gear_dn_raw,
        "cas_kt": cas,
        "vref_kt": vref,
        "target_speed_kt": target,
        "cas_minus_vref_kt": delta,
        "cas_label": f"Vref {'+' if delta >= 0 else ''}{delta:.0f}" if not np.isnan(delta) else "—",
        "vrate_fpm": vrate,
        "n1_avg_pct": n1_avg,

        # Overall verdict
        "stable_approach": overall,
    }
