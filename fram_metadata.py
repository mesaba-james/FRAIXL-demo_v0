"""
FRAIXL-Air Metadata Equation Set
=================================
Implements the v0.3 metadata equations from David's development note,
producing derived function states from observed metadata.

Outputs:
  landing_energy_state         (Manage landing energy)
  landing_vertical_profile_state (Maintain landing vertical profile)
  landing_configuration_state  (Confirm landing configuration)
  landing_control_mode_state   (Manage automation/manual transition)
  landing_stability_score      (Monitor landing stability)
  landing_stability_assessment (Monitor landing stability - classified)
  touchdown_transition_state   (Execute flare / touchdown transition)
  rollout_state                (Manage rollout and deceleration)

All thresholds live in the calibration profile, never inside the topology.
Thresholds are provisional; per WP4 they must be replaced by calibration
from reviewed normal landings before being used as a divergence detector.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# Calibration profile
# ----------------------------------------------------------------------------

@dataclass
class CalibrationProfile:
    """
    Calibration constants for the Landing Phase Twin v0.3.

    PROVISIONAL — these are placeholder values for demonstration. They MUST be
    replaced by values derived from reviewed normal landings before being
    used to interpret divergence (per the FRAIXL-Air proposal WP4).
    """
    # --- Energy thresholds (kt above/below target) ---
    energy_normal_band_kt: float = 5.0       # |error| <= 5 kt => ON_TARGET
    energy_caution_band_kt: float = 10.0     # 5 < |error| <= 10 => CAUTION
    # beyond caution => HIGH_ENERGY or LOW_ENERGY

    # --- Vertical profile thresholds ---
    vrate_normal_max_fpm: float = 900.0      # |vrate| <= 900 fpm normal in approach
    vrate_caution_max_fpm: float = 1200.0    # 900 < |vrate| <= 1200 caution
    # beyond caution => HIGH_DESCENT
    fpa_target_deg: float = -3.0             # standard ILS glideslope
    fpa_tolerance_deg: float = 0.5           # ±0.5° normal
    fpa_caution_deg: float = 1.0             # ±1° caution

    # --- Configuration ---
    required_landing_flap_deg: float = 30.0  # 737 typically flaps 30 or 40 for landing
    config_stable_gate_radalt_ft: float = 1000.0  # config should be complete by 1000 AGL
    config_caution_gate_radalt_ft: float = 500.0  # marginal between 1000 and 500

    # --- Control mode ---
    ap_handover_min_radalt_ft: float = 50.0   # AP off below 50 ft = late handover (autoland case different)
    ap_handover_normal_radalt_ft: float = 200.0  # AP off between 50-200 normal manual landing

    # --- Stability composite weights (must sum to 1.0) ---
    energy_weight: float = 0.30
    vertical_weight: float = 0.30
    configuration_weight: float = 0.25
    control_mode_weight: float = 0.15

    # --- Stability classification ---
    stable_min_score: float = 0.80           # >= 0.80 STABLE
    cautionary_min_score: float = 0.60       # 0.60 - 0.80 CAUTIONARY
    # below 0.60 UNSTABLE

    # --- Stabilization gate ---
    stabilization_gate_radalt_ft: float = 1000.0  # IMC stabilization gate

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


# ----------------------------------------------------------------------------
# Component scoring helpers
# ----------------------------------------------------------------------------

def _score_energy(speed_error_kt: pd.Series, cal: CalibrationProfile) -> tuple[pd.Series, pd.Series]:
    """
    Returns (state_label_series, score_0_to_1_series).
    State labels: ON_TARGET, ENERGY_CAUTION_HIGH, ENERGY_CAUTION_LOW,
                  HIGH_ENERGY, LOW_ENERGY, UNKNOWN.
    Score: 1.0 = on-target, decays linearly through caution to 0 at 2× caution band.
    """
    abs_err = speed_error_kt.abs()
    state = pd.Series("UNKNOWN", index=speed_error_kt.index)
    state[(abs_err <= cal.energy_normal_band_kt) & speed_error_kt.notna()] = "ON_TARGET"
    state[(speed_error_kt > cal.energy_normal_band_kt) & (speed_error_kt <= cal.energy_caution_band_kt)] = "ENERGY_CAUTION_HIGH"
    state[(speed_error_kt < -cal.energy_normal_band_kt) & (speed_error_kt >= -cal.energy_caution_band_kt)] = "ENERGY_CAUTION_LOW"
    state[speed_error_kt > cal.energy_caution_band_kt] = "HIGH_ENERGY"
    state[speed_error_kt < -cal.energy_caution_band_kt] = "LOW_ENERGY"

    # Score
    score = pd.Series(np.nan, index=speed_error_kt.index)
    score[state == "ON_TARGET"] = 1.0
    # Linear decay in caution band: 1.0 -> 0.5
    mask_caution = state.isin(["ENERGY_CAUTION_HIGH", "ENERGY_CAUTION_LOW"])
    over = (abs_err - cal.energy_normal_band_kt) / (cal.energy_caution_band_kt - cal.energy_normal_band_kt)
    score[mask_caution] = (1.0 - 0.5 * over.clip(0, 1))[mask_caution]
    # Beyond caution: 0.5 -> 0 linearly over another caution-width
    mask_beyond = state.isin(["HIGH_ENERGY", "LOW_ENERGY"])
    over_b = (abs_err - cal.energy_caution_band_kt) / (cal.energy_caution_band_kt - cal.energy_normal_band_kt)
    score[mask_beyond] = (0.5 - 0.5 * over_b.clip(0, 1))[mask_beyond]

    return state, score.fillna(0.0)


def _score_vertical(vrate_fpm: pd.Series, fpa_deg: pd.Series | None, cal: CalibrationProfile) -> tuple[pd.Series, pd.Series]:
    """
    Combines vertical rate and (if available) flight path angle into a vertical
    profile state and score.
    """
    abs_v = vrate_fpm.abs()
    state = pd.Series("UNKNOWN", index=vrate_fpm.index)
    state[(abs_v <= cal.vrate_normal_max_fpm)] = "ON_PROFILE"
    state[(abs_v > cal.vrate_normal_max_fpm) & (abs_v <= cal.vrate_caution_max_fpm)] = "PROFILE_CAUTION"
    state[abs_v > cal.vrate_caution_max_fpm] = "HIGH_DESCENT"

    # Score from vrate
    score_v = pd.Series(0.0, index=vrate_fpm.index)
    score_v[state == "ON_PROFILE"] = 1.0
    mask_c = state == "PROFILE_CAUTION"
    over = (abs_v - cal.vrate_normal_max_fpm) / (cal.vrate_caution_max_fpm - cal.vrate_normal_max_fpm)
    score_v[mask_c] = (1.0 - 0.5 * over.clip(0, 1))[mask_c]
    mask_h = state == "HIGH_DESCENT"
    over_h = (abs_v - cal.vrate_caution_max_fpm) / cal.vrate_caution_max_fpm
    score_v[mask_h] = (0.5 - 0.5 * over_h.clip(0, 1))[mask_h]

    # If FPA available, blend (50/50)
    if fpa_deg is not None and fpa_deg.notna().any():
        fpa_err = (fpa_deg - cal.fpa_target_deg).abs()
        score_fpa = pd.Series(0.5, index=fpa_deg.index)
        score_fpa[fpa_err <= cal.fpa_tolerance_deg] = 1.0
        mask_fc = (fpa_err > cal.fpa_tolerance_deg) & (fpa_err <= cal.fpa_caution_deg)
        over_f = (fpa_err - cal.fpa_tolerance_deg) / (cal.fpa_caution_deg - cal.fpa_tolerance_deg)
        score_fpa[mask_fc] = (1.0 - 0.5 * over_f.clip(0, 1))[mask_fc]
        score_fpa[fpa_err > cal.fpa_caution_deg] = 0.3
        # blend
        score_v = 0.5 * score_v + 0.5 * score_fpa.fillna(score_v)

    return state, score_v.fillna(0.0)


def _score_configuration(canonical: pd.DataFrame, cal: CalibrationProfile) -> tuple[pd.Series, pd.Series]:
    """
    Configuration state and score: gear DOWN and flap >= required landing flap.
    Considers radalt gate timing.

    Gear-down evidence: we prefer the GEAR LEVER DOWN discrete when its logic
    is clear, but on this 737 dataset the lever discretes are inverted in a way
    that doesn't match the air/ground sensors, so we treat gear as DOWN when
    BOTH the air/ground sensors read 1 (= on ground / WoW-style logic) at low
    altitude is not the right test in flight. For now, in flight we use the
    lever-up==0 AND lever-down==1 combination, falling back to lever-down alone.
    This is exactly the kind of mapping decision that needs an operator-specific
    parameter dictionary review.
    """
    flap = canonical.get("flap_handle_deg", pd.Series(0, index=canonical.index)).fillna(0)
    gear_dn = canonical.get("gear_lever_down", pd.Series(0, index=canonical.index)).fillna(0)
    gear_up = canonical.get("gear_lever_up", pd.Series(1, index=canonical.index)).fillna(1)
    radalt = canonical.get("observed_radalt_ft", pd.Series(np.nan, index=canonical.index))

    flap_ok = flap >= cal.required_landing_flap_deg
    # Gear is DOWN when GEAR LEVER DOWN=1 OR (GEAR LEVER UP=0 in approach phase)
    # For demonstration, prefer either signal showing gear is extended
    gear_ok = (gear_dn == 1) | (gear_up == 0)
    config_complete = flap_ok & gear_ok

    state = pd.Series("CONFIG_INCOMPLETE", index=canonical.index)
    state[config_complete] = "CONFIG_COMPLETE"
    # Late if not complete below gate
    below_gate = radalt < cal.config_stable_gate_radalt_ft
    state[below_gate & ~config_complete] = "CONFIG_LATE"
    below_caution = radalt < cal.config_caution_gate_radalt_ft
    state[below_caution & ~config_complete] = "CONFIG_VERY_LATE"

    score = pd.Series(0.0, index=canonical.index)
    score[state == "CONFIG_COMPLETE"] = 1.0
    score[state == "CONFIG_INCOMPLETE"] = 0.5  # still configuring, above gate
    score[state == "CONFIG_LATE"] = 0.3
    score[state == "CONFIG_VERY_LATE"] = 0.0

    return state, score


def _score_control_mode(canonical: pd.DataFrame, cal: CalibrationProfile) -> tuple[pd.Series, pd.Series]:
    """
    Control mode state. Distinguishes AP-coupled, manual, and the handover
    region. An autoland keeps AP engaged through touchdown (flare engaged).
    """
    cmd_a = canonical.get("ap_cmd_a", pd.Series(0, index=canonical.index)).fillna(0)
    cmd_b = canonical.get("ap_cmd_b", pd.Series(0, index=canonical.index)).fillna(0)
    flare_eng = canonical.get("flare_engaged", pd.Series(0, index=canonical.index)).fillna(0)
    radalt = canonical.get("observed_radalt_ft", pd.Series(np.nan, index=canonical.index))

    ap_on = (cmd_a == 1) | (cmd_b == 1)
    autoland = flare_eng == 1

    state = pd.Series("MANUAL", index=canonical.index)
    state[ap_on] = "AP_COUPLED"
    state[ap_on & autoland] = "AUTOLAND"

    # Score: AP_COUPLED above gate = 1.0, MANUAL above gate = 1.0,
    # but a transition between gate and a sensible handover height is fine.
    # The questionable case is AP-on right to flare-without-autoland-arming.
    score = pd.Series(1.0, index=canonical.index)
    # Very late manual handover (AP off below 50 ft and not autoland) is a flag
    very_late_handover = (~ap_on) & (radalt < cal.ap_handover_min_radalt_ft) & (~autoland)
    score[very_late_handover] = 0.6

    return state, score


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

def compute_states(canonical: pd.DataFrame, cal: CalibrationProfile | None = None) -> pd.DataFrame:
    """
    Compute all v0.3 derived function states for the canonical flight data.
    Returns a new dataframe (same index as canonical) with state and score
    columns added.
    """
    if cal is None:
        cal = CalibrationProfile()

    out = canonical.copy()

    # --- Energy ---
    if "observed_speed_kt" in out.columns and "target_speed_kt" in out.columns:
        out["speed_error_kt"] = out["observed_speed_kt"] - out["target_speed_kt"]
        state_e, score_e = _score_energy(out["speed_error_kt"], cal)
        out["landing_energy_state"] = state_e
        out["energy_score"] = score_e
    else:
        out["speed_error_kt"] = np.nan
        out["landing_energy_state"] = "UNKNOWN"
        out["energy_score"] = 0.0

    # --- Vertical profile ---
    if "observed_vrate_fpm" in out.columns:
        fpa = out.get("observed_fpa_deg")
        state_v, score_v = _score_vertical(out["observed_vrate_fpm"], fpa, cal)
        out["landing_vertical_profile_state"] = state_v
        out["vertical_score"] = score_v
    else:
        out["landing_vertical_profile_state"] = "UNKNOWN"
        out["vertical_score"] = 0.0

    # --- Configuration ---
    state_c, score_c = _score_configuration(out, cal)
    out["landing_configuration_state"] = state_c
    out["configuration_score"] = score_c

    # --- Control mode ---
    state_m, score_m = _score_control_mode(out, cal)
    out["landing_control_mode_state"] = state_m
    out["control_mode_score"] = score_m

    # --- Composite stability ---
    out["landing_stability_score"] = (
        cal.energy_weight * out["energy_score"]
        + cal.vertical_weight * out["vertical_score"]
        + cal.configuration_weight * out["configuration_score"]
        + cal.control_mode_weight * out["control_mode_score"]
    )

    assess = pd.Series("UNSTABLE", index=out.index)
    assess[out["landing_stability_score"] >= cal.cautionary_min_score] = "CAUTIONARY"
    assess[out["landing_stability_score"] >= cal.stable_min_score] = "STABLE"
    out["landing_stability_assessment"] = assess

    # --- Touchdown transition state ---
    if "air_ground" in out.columns:
        ag = out["air_ground"]
        out["touchdown_transition_state"] = "AIRBORNE"
        out.loc[ag == 0, "touchdown_transition_state"] = "ON_GROUND"

    # --- Rollout state ---
    if "auto_speedbrake_cmd" in out.columns and "air_ground" in out.columns:
        on_ground = out["air_ground"] == 0
        sb = out["auto_speedbrake_cmd"] == 1
        tr = ((out.get("tr1_deployed", 0) == 1) | (out.get("tr2_deployed", 0) == 1))
        out["rollout_state"] = "N/A"
        out.loc[on_ground, "rollout_state"] = "ROLLOUT_PRE_DECEL"
        out.loc[on_ground & sb, "rollout_state"] = "ROLLOUT_SPOILERS"
        out.loc[on_ground & sb & tr, "rollout_state"] = "ROLLOUT_FULL_DECEL"

    return out


def stabilization_gate_assessment(state_df: pd.DataFrame, cal: CalibrationProfile) -> dict:
    """
    Standard FOQA-style: was the aircraft stable at the stabilization gate
    (default 1000 ft AGL)? Returns a small assessment dict.
    """
    if "observed_radalt_ft" not in state_df.columns:
        return {"available": False, "reason": "no radio altitude"}

    radalt = state_df["observed_radalt_ft"]
    # Find the first crossing of the gate from above
    above = radalt >= cal.stabilization_gate_radalt_ft
    below = radalt < cal.stabilization_gate_radalt_ft
    crossings = below & above.shift(1).fillna(False)
    if not crossings.any():
        return {"available": False, "reason": f"flight did not cross {cal.stabilization_gate_radalt_ft} ft AGL"}

    i = crossings[crossings].index[0]
    row = state_df.loc[i]
    return {
        "available": True,
        "gate_radalt_ft": float(cal.stabilization_gate_radalt_ft),
        "time_s": float(row["time_s"]),
        "energy_state": row.get("landing_energy_state"),
        "vertical_state": row.get("landing_vertical_profile_state"),
        "configuration_state": row.get("landing_configuration_state"),
        "control_mode_state": row.get("landing_control_mode_state"),
        "stability_score": float(row.get("landing_stability_score", np.nan)),
        "stability_assessment": row.get("landing_stability_assessment"),
        "speed_error_kt": float(row.get("speed_error_kt", np.nan)),
        "cas_kt": float(row.get("observed_speed_kt", np.nan)),
        "vrate_fpm": float(row.get("observed_vrate_fpm", np.nan)),
        "flap_deg": float(row.get("flap_handle_deg", np.nan)),
        "gear_down": float(row.get("gear_lever_down", np.nan)),
    }
