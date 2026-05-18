# FRAIXL-Air Landing Phase Twin — Demonstrator

A Streamlit app that turns a raw 737 FDR/FOQA CSV into a working FRAIXL-Air
Landing Phase Twin v0.3 view. Built to give David Slater a concrete
proof-of-concept implementation of the architecture in the Cambrensis
proposal, exercised against real flight data.

## What it does

Implements the four-layer architecture from the FRAIXL-Air proposal:

1. **Data-to-model dictionary (WP2)** — explicit mapping from raw FDR
   columns (1,182 of them in the demo CSV) to canonical metadata keys
   used by the Twin. Exports as JSON for reuse in the `.xfmv` environment.

2. **Observed metadata extraction** — reads the multi-rate FDR CSV,
   forward-fills sub-frame sparseness, and produces a canonical
   dataframe with the parameters the Twin needs.

3. **Derived function states (v0.3)** — implements David's metadata
   equation set: `landing_energy_state`, `landing_vertical_profile_state`,
   `landing_configuration_state`, `landing_control_mode_state`,
   composite `landing_stability_score`, plus `touchdown_transition_state`
   and `rollout_state` driven by AIR/GROUND and weight-on-wheels.

4. **Bounded LLM interpretive layer** — Claude is plugged in as the
   analyst-support layer that *explains* what the formal model has
   detected, exactly as the proposal architects it. Never runs the
   monitoring; only interprets the structured outputs.

## What's in the four tabs

| Tab | Purpose |
|-----|---------|
| Raw & Parameter Dictionary | Show what was loaded, what mapped to canonical keys, and download the dictionary |
| Flight Profile | Vertical profile, energy plot, configuration timeline, ground track, event markers |
| FRAM Twin (v0.3) | Composite stability over time, component scores, state distributions, 1000ft gate assessment |
| Calibration & Discrepancy | Active calibration profile, delta view, optional Claude-generated analyst summary |

## Run it

```bash
cd fraixl_app
pip install -r requirements.txt
streamlit run app.py
```

Then either upload an FDR CSV in the sidebar, or use the bundled
`737_nominal_v0.csv` demo file (the ATL approach we analyzed).

## File structure

```
fraixl_app/
├── app.py                      # Streamlit UI (4 tabs)
├── parameter_dictionary.py     # Data-to-model dictionary (the WP2 artefact)
├── fraixl_extract.py           # CSV load, canonical build, event detection
├── fram_metadata.py            # v0.3 metadata equations + calibration profile
├── requirements.txt
└── README.md
```

The three Python modules are intentionally framework-free — no
Streamlit dependencies. They could be:
- imported into a Jupyter notebook
- wrapped as a CLI tool
- wrapped as an MCP server so the .xfmv runtime or other LLMs could
  hit the same extraction logic

For the v1 demonstrator, we use the Anthropic SDK directly from
Streamlit; MCP is the right move if you want multiple consumers.

## Notes for David

A few things worth highlighting:

- **All thresholds live in the calibration profile** (`fram_metadata.CalibrationProfile`), not the topology. Adjustable from the sidebar.
- **Thresholds are PROVISIONAL.** WP4 of your proposal calls for calibration against reviewed normal landings before treating discrepancies as meaningful. The app prominently flags this.
- **Touchdown detection** uses AIR/GROUND transition + radalt cross-check — addressing the proxy gap you flagged in v0.3.
- **Gear-down logic** revealed an interesting FDR quirk on this dataset (GEAR LEVER UP=1 with the aircraft clearly extended below 1000ft AGL). The data-to-model dictionary is exactly the right place to record operator/airframe-specific interpretation decisions like this.
- The parameter dictionary covers radalt, pitch/roll, glideslope/loc deviation, WoW, brakes, spoilers, T/R — the gaps explicitly listed in the v0.3 development note.
