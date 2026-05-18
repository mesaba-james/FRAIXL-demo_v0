"""
FRAIXL-Air Parameter Dictionary
================================
Maps raw FDR/FOQA column names to canonical metadata keys used by the
Landing Phase Twin. This is the data-to-model dictionary called for in
WP2 of the FRAIXL-Air proposal.

Each entry:
  canonical_key : the stable identifier used inside the Twin
  fdr_columns   : ordered list of possible source columns (first match wins)
  unit          : engineering unit of the canonical value
  function      : FRAM function this parameter serves as observed metadata for
  description   : short, analyst-readable description

This file is the artifact David can take and reuse in the .xfmv environment.
"""

PARAMETER_DICTIONARY = {
    # --- Time base ---
    "time_s": {
        "fdr_columns": ["Time"],
        "unit": "s",
        "function": "Import landing-phase flight-data record",
        "description": "FDR frame time (seconds since recorder start)",
    },

    # --- Air data / energy ---
    "observed_alt_baro_ft": {
        "fdr_columns": [
            "ALTITUDE. BAROMETRIC (CORRECTED) (FT)",
            "ALTITUDE (1013.25mB) (FT)",
        ],
        "unit": "ft",
        "function": "Provide landing air-data observation",
        "description": "Barometric altitude (QNH-corrected preferred, 1013.25 fallback)",
    },
    "observed_alt_std_ft": {
        "fdr_columns": ["ALTITUDE (1013.25mB) (FT)"],
        "unit": "ft",
        "function": "Provide landing air-data observation",
        "description": "Standard pressure altitude (1013.25 mB)",
    },
    "observed_radalt_ft": {
        "fdr_columns": [
            "RADIO HEIGHT - DISPLAYED (FT)",
            "RADIO HEIGHT - LT LRRA (FT)",
            "RADIO HEIGHT - RT LRRA (FT)",
        ],
        "unit": "ft",
        "function": "Provide landing air-data observation",
        "description": "Radio altitude above terrain (fills a key gap in v0.3)",
    },
    "observed_speed_kt": {
        "fdr_columns": ["COMPUTED AIRSPEED (KT)"],
        "unit": "kt",
        "function": "Provide landing air-data observation",
        "description": "Computed (calibrated) airspeed",
    },
    "observed_gs_kt": {
        "fdr_columns": ["GROUNDSPEED (KT)"],
        "unit": "kt",
        "function": "Provide landing air-data observation",
        "description": "Groundspeed (used for rollout deceleration)",
    },
    "observed_vrate_fpm": {
        "fdr_columns": ["INERTIAL VERT SPD (FT/MIN)"],
        "unit": "ft/min",
        "function": "Provide landing air-data observation",
        "description": "Inertial vertical speed (descent rate)",
    },
    "observed_pitch_deg": {
        "fdr_columns": ["CAPT DISPLAY PITCH ATT (DEG)"],
        "unit": "deg",
        "function": "Provide landing air-data observation",
        "description": "Pitch attitude (flare quality evidence)",
    },
    "observed_roll_deg": {
        "fdr_columns": ["CAPT DISPLAY ROLL ATT (DEG)"],
        "unit": "deg",
        "function": "Provide landing air-data observation",
        "description": "Roll attitude (lateral control evidence)",
    },
    "observed_fpa_deg": {
        "fdr_columns": ["FLIGHT PATH ANGLE (DEG)"],
        "unit": "deg",
        "function": "Provide landing air-data observation",
        "description": "Flight path angle (descent profile evidence)",
    },

    # --- Reference / target speeds ---
    "target_speed_kt": {
        "fdr_columns": ["TARGET AIRSPEED (KT)", "SELECTED AIRSPEED (KT)"],
        "unit": "kt",
        "function": "Provide landing reference observation",
        "description": "Selected/target approach speed",
    },
    "vref_kt": {
        "fdr_columns": ["VREF SPEED (KT)"],
        "unit": "kt",
        "function": "Provide landing reference observation",
        "description": "Reference landing speed (Vref)",
    },

    # --- Configuration ---
    "flap_handle_deg": {
        "fdr_columns": ["FLAP HANDLE POSITION (DEG)"],
        "unit": "deg",
        "function": "Provide landing configuration observation",
        "description": "Flap handle position (commanded detent)",
    },
    "flap_position_l_deg": {
        "fdr_columns": ["TE FLAP POSN-LT (DEG)", "FLAP POSITION (T.E.. LEFT) SYNCHRO (DEG)"],
        "unit": "deg",
        "function": "Provide landing configuration observation",
        "description": "Actual trailing-edge flap position, left",
    },
    "gear_lever_down": {
        "fdr_columns": ["GEAR LEVER DOWN", "GEAR LEVER SELECTED_DOWN"],
        "unit": "discrete",
        "function": "Provide landing configuration observation",
        "description": (
            "GEAR LEVER DOWN discrete. NOTE: in this 737 FDR mapping the "
            "discrete is INVERTED — 1 = 'lever NOT in DOWN detent', "
            "0 = 'lever IN DOWN detent'. The canonical 'gear is down' "
            "indicator in this dictionary is derived from gear_lever_up == 1 "
            "(see that entry). Verify before reusing on other operator data."
        ),
    },
    "gear_lever_up": {
        "fdr_columns": ["GEAR LEVER UP"],
        "unit": "discrete",
        "function": "Provide landing configuration observation",
        "description": (
            "GEAR LEVER UP discrete. In this 737 FDR mapping the discrete "
            "is INVERTED — 0 = 'lever IN UP detent' (gear UP), "
            "1 = 'lever NOT in UP detent' (gear DOWN). The canonical "
            "'gear is down' indicator for this dataset is therefore "
            "(gear_lever_up == 1). Confirmed against the operational "
            "timeline: lever moves out of UP at ~2250 ft AGL, then aircraft "
            "touches down normally."
        ),
    },

    # --- Automation / control mode ---
    "ap_cmd_a": {
        "fdr_columns": ["CMD A"],
        "unit": "discrete",
        "function": "Provide automation/control-mode observation",
        "description": "Autopilot channel A engaged (1=engaged)",
    },
    "ap_cmd_b": {
        "fdr_columns": ["CMD B"],
        "unit": "discrete",
        "function": "Provide automation/control-mode observation",
        "description": "Autopilot channel B engaged (1=engaged)",
    },
    "ap_off": {
        "fdr_columns": ["A/P OFF"],
        "unit": "discrete",
        "function": "Provide automation/control-mode observation",
        "description": "Autopilot OFF discrete (1=off)",
    },
    "at_engaged": {
        "fdr_columns": ["A/T ENGAGED"],
        "unit": "discrete",
        "function": "Provide automation/control-mode observation",
        "description": "Autothrottle engaged (1=engaged)",
    },
    "gs_engaged": {
        "fdr_columns": ["A/P G/S ENGAGE"],
        "unit": "discrete",
        "function": "Provide automation/control-mode observation",
        "description": "Glideslope capture engaged",
    },
    "loc_engaged": {
        "fdr_columns": ["A/P VOR/LOC ENGAGE"],
        "unit": "discrete",
        "function": "Provide automation/control-mode observation",
        "description": "Localizer capture engaged",
    },
    "flare_engaged": {
        "fdr_columns": ["A/P FLARE ENGAGE"],
        "unit": "discrete",
        "function": "Execute flare / touchdown transition",
        "description": "Autoland flare mode engaged",
    },
    "toga_pressed": {
        "fdr_columns": ["TOGA SW PRESSED"],
        "unit": "discrete",
        "function": "Manage automation/manual transition",
        "description": "TOGA (go-around) switch pressed",
    },

    # --- Engine N1 (for energy state) ---
    "n1_left_pct": {
        "fdr_columns": ["LT ENG N1 TACHOMETER (%RPM)", "SELECTED N1 INDICATED #1 (%RPM)"],
        "unit": "%RPM",
        "function": "Manage landing energy",
        "description": "Left engine N1 (% rpm) — thrust evidence for energy state",
    },
    "n1_right_pct": {
        "fdr_columns": ["RT ENG N1 TACHOMETER (%RPM)", "SELECTED N1 INDICATED #2 (%RPM)"],
        "unit": "%RPM",
        "function": "Manage landing energy",
        "description": "Right engine N1 (% rpm) — thrust evidence for energy state",
    },

    # --- ILS deviations ---
    "gs_deviation_ddm": {
        "fdr_columns": ["GLIDESLOPE DEV/ELEVATION-L (DDM)", "GLIDESLOPE DEV/ELEVATION-R (DDM)"],
        "unit": "DDM",
        "function": "Maintain landing vertical profile",
        "description": "Glideslope deviation (DDM, fly-up = +ve in conv.)",
    },
    "loc_deviation_ddm": {
        "fdr_columns": ["LOCALIZER DEV/AZIMUTH-L (DDM)", "LOCALIZER DEV/AZIMUTH-R (DDM)"],
        "unit": "DDM",
        "function": "Maintain landing vertical profile",
        "description": "Localizer deviation (DDM)",
    },

    # --- Touchdown / rollout ---
    "air_ground": {
        "fdr_columns": ["AIR/GROUND"],
        "unit": "discrete",
        "function": "Execute flare / touchdown transition",
        "description": "Air/ground logic (1=air, 0=ground per FDR convention)",
    },
    "nose_gear_air_ground": {
        "fdr_columns": ["NOSE GEAR AIR/GND"],
        "unit": "discrete",
        "function": "Execute flare / touchdown transition",
        "description": "Nose gear weight-on-wheels (1=air, 0=ground)",
    },
    "auto_speedbrake_cmd": {
        "fdr_columns": ["AUTO SPD BRAKE EXTEND CMD"],
        "unit": "discrete",
        "function": "Manage rollout and deceleration",
        "description": "Auto speedbrake extend command",
    },
    "speedbrake_armed": {
        "fdr_columns": ["SPEED BRAKE ARMED LIGHT"],
        "unit": "discrete",
        "function": "Manage rollout and deceleration",
        "description": "Speedbrake armed annunciation",
    },
    "ground_spoiler_spdbrk": {
        "fdr_columns": ["GROUND SPOILER/SPD BRK"],
        "unit": "discrete",
        "function": "Manage rollout and deceleration",
        "description": "Ground spoiler / speed brake discrete",
    },
    "spoiler_pos_03_deg": {
        "fdr_columns": ["SPOILER POSITION NO. 03 (DEG)"],
        "unit": "deg",
        "function": "Manage rollout and deceleration",
        "description": "Spoiler panel 3 deflection",
    },
    "spoiler_pos_09_deg": {
        "fdr_columns": ["SPOILER POSITION NO. 09 (DEG)"],
        "unit": "deg",
        "function": "Manage rollout and deceleration",
        "description": "Spoiler panel 9 deflection",
    },
    "tr1_deployed": {
        "fdr_columns": ["T/R FULL DEPLOY #1"],
        "unit": "discrete",
        "function": "Manage rollout and deceleration",
        "description": "Engine 1 thrust reverser fully deployed",
    },
    "tr2_deployed": {
        "fdr_columns": ["T/R FULL DEPLOY #2"],
        "unit": "discrete",
        "function": "Manage rollout and deceleration",
        "description": "Engine 2 thrust reverser fully deployed",
    },
    "brake_press_l_psi": {
        "fdr_columns": ["BRAKE PRESS MAIN-LT (PSI)"],
        "unit": "psi",
        "function": "Manage rollout and deceleration",
        "description": "Left main brake pressure",
    },
    "brake_press_r_psi": {
        "fdr_columns": ["BRAKE PRESS MAIN-RT (PSI)"],
        "unit": "psi",
        "function": "Manage rollout and deceleration",
        "description": "Right main brake pressure",
    },

    # --- Navigation / geometry ---
    "lat_deg": {
        "fdr_columns": ["PRES POSN LAT (DEG)", "EMS Trajectory Latitude"],
        "unit": "deg",
        "function": "Provide landing air-data observation",
        "description": "Present position latitude",
    },
    "lon_deg": {
        "fdr_columns": ["PRES POSN LONG (DEG)", "EMS Trajectory Longitude"],
        "unit": "deg",
        "function": "Provide landing air-data observation",
        "description": "Present position longitude",
    },
    "distance_to_go_nm": {
        "fdr_columns": ["DISTANCE TO GO (NM)"],
        "unit": "nm",
        "function": "Establish landing-phase context",
        "description": "FMC distance to destination",
    },
    "mag_heading_deg": {
        "fdr_columns": ["MAGNETIC HEADING (IRU) (DEG)"],
        "unit": "deg",
        "function": "Provide landing air-data observation",
        "description": "Magnetic heading",
    },
    "wind_speed_kt": {
        "fdr_columns": ["WIND SPEED (KT)"],
        "unit": "kt",
        "function": "Establish landing-phase context",
        "description": "Wind speed",
    },
    "wind_dir_deg": {
        "fdr_columns": ["WIND DIRECTION TRUE (DEG)"],
        "unit": "deg",
        "function": "Establish landing-phase context",
        "description": "Wind direction (true)",
    },

    # --- Warnings ---
    "stick_shaker_l": {
        "fdr_columns": ["STICK SHAKER - LT"],
        "unit": "discrete",
        "function": "Monitor landing stability",
        "description": "Left stick shaker activation",
    },
    "gpws_sink_rate": {
        "fdr_columns": ["GPWS SINK RATE"],
        "unit": "discrete",
        "function": "Monitor landing stability",
        "description": "GPWS sink rate warning",
    },
    "gpws_glideslope": {
        "fdr_columns": ["GPWS GLIDESLOPE"],
        "unit": "discrete",
        "function": "Monitor landing stability",
        "description": "GPWS glideslope alert",
    },
}


def get_dictionary():
    """Return the parameter dictionary."""
    return PARAMETER_DICTIONARY


def list_by_function():
    """Group canonical keys by the FRAM function they serve."""
    by_fn = {}
    for key, meta in PARAMETER_DICTIONARY.items():
        fn = meta["function"]
        by_fn.setdefault(fn, []).append(key)
    return by_fn
