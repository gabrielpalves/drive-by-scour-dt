"""
drive_by_DT.py
==============
Drive-by digital twin for scour damage monitoring.

This script is the entry point for the online simulation phase.
It contains only configuration and execution — no class or function
definitions.  All implementation lives in the digital_twin/ and core/
packages built from the ablation study outputs.

What this script does
---------------------
1. Load the offline package produced by the ablation
   (model weights, scaler, architecture metadata).
2. Configure the DBN from the champion model's confusion matrix.
3. Instantiate the four framework objects: PhysicalAsset, DigitalAsset,
   Planner, and Graph.
4. Run the filtering simulation for sim_years steps.
5. Run a forward prediction from the final belief.
6. Save figures and history CSVs.

What to edit between runs
-------------------------
Only the block labelled CONFIGURATION below.  Everything else derives
from the metadata JSON or carries sensible defaults.
"""

import os
import sys

import numpy as np
import torch

# ── Digital twin framework ────────────────────────────────────────────────────
from digital_twin.config  import DTConfig
from digital_twin.assets  import PhysicalAsset, DigitalAsset
from digital_twin.dbn     import GetSetup, Graph, normalize_cpd
from digital_twin.planner import Planner
from digital_twin.dt_plots import Plot

# ── pgmpy — hard dependency for the DBN ──────────────────────────────────────
try:
    from pgmpy.models import DynamicBayesianNetwork   # noqa: F401 (import check)
except ImportError:
    print("ERROR: pgmpy is not installed.  Run:  pip install pgmpy")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit only this block between runs
# ─────────────────────────────────────────────────────────────────────────────

# Paths to the DT package written by export_digital_twin_package.
# Point these at whichever champion output folder the ablation produced.
CHAMPION_DIR   = "results_architectures_ntv_DOF_0_1_2_3_4_5_6_7_disc1/7_PAA_S2V_NHiTS"
METADATA_PATH  = os.path.join(CHAMPION_DIR, "DT_metadata.json")
WEIGHTS_PATH   = os.path.join(CHAMPION_DIR, "DT_champion_weights.pth")
SCALER_PATH    = os.path.join(CHAMPION_DIR, "DT_scaler.pkl")
CONF_MAT_PATH  = os.path.join(CHAMPION_DIR, "DT_conf_matrix.npy")

# Simulation parameters — these do NOT live in the metadata.
SIM_YEARS             = 60       # total monitoring horizon (years)
MONITORING_INTERVAL   = 1.0      # years between drive-by passages
REPAIR_THRESHOLD_LABEL = 6       # labels ≥ this → repair  (label 6 = 30% damage when disc=5)
P_DETERIORATION       = 0.05     # annual probability of advancing one damage class

# DBN inference samples — higher = more accurate but slower per step.
N_DBN_SAMPLES  = 100

# Forward prediction horizon.
PREDICT_STEPS  = 20
PREDICT_SAMPLES = 1_000

# Output directory for figures and CSVs.
PLOT_DIR = "./dt_results"

# Reproducibility.
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Validate paths before doing anything expensive
# ─────────────────────────────────────────────────────────────────────────────

missing = [p for p in (METADATA_PATH, WEIGHTS_PATH, SCALER_PATH, CONF_MAT_PATH)
           if not os.path.exists(p)]
if missing:
    print("ERROR: The following offline package files are missing:")
    for p in missing:
        print(f"  {p}")
    print("\nRun the ablation notebook through to completion (including "
          "export_digital_twin_package) before launching this script.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────────

config = DTConfig.from_metadata(
    metadata_path=METADATA_PATH,
    repair_threshold_label=REPAIR_THRESHOLD_LABEL,
    sim_years=SIM_YEARS,
    monitoring_interval=MONITORING_INTERVAL,
)
print(f"\n{config}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DBN setup
# ─────────────────────────────────────────────────────────────────────────────

possible_states  = [str(i) for i in range(config.n_classes)]
possible_actions = ["do_nothing", "perfect_repair"]

conf_mat = np.load(CONF_MAT_PATH)
if conf_mat.shape != (config.n_classes, config.n_classes):
    print(
        f"ERROR: Confusion matrix shape {conf_mat.shape} does not match "
        f"n_classes={config.n_classes}.  "
        f"Ensure CONF_MAT_PATH points to the correct champion's matrix."
    )
    sys.exit(1)

setup       = GetSetup(
    states_list=possible_states,
    actions_list=possible_actions,
    conf_mat_dt=conf_mat,
)
transitions = setup.get_transitions(p_nothing=P_DETERIORATION)
planner     = Planner(
    states_list=possible_states,
    actions_list=possible_actions,
    threshold_label=REPAIR_THRESHOLD_LABEL,
)
graph_structure, digital_subgraph, list_cpd_graph, list_cpd_subgraph = \
    setup.get_graph(cpd_d_to_u=planner.policy.T, transitions=transitions)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Asset instantiation
# ─────────────────────────────────────────────────────────────────────────────

physical = PhysicalAsset(
    config=config,
    degradation_law="scour_gradual",
    monitoring_interval=MONITORING_INTERVAL,
)

digital = DigitalAsset(
    model_path=WEIGHTS_PATH,
    metadata_path=METADATA_PATH,
    scaler_path=SCALER_PATH,
    config=config,
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Graph assembly
# ─────────────────────────────────────────────────────────────────────────────

graph = Graph(
    physical_asset=physical,
    digital_asset=digital,
    states_list=possible_states,
    actions_list=possible_actions,
    planner=planner,
)
graph.assemble_graph(
    graph_structure=graph_structure,
    cpd_list=list_cpd_graph,
    digital_subgraph=digital_subgraph,
    list_cpd_subgraph=list_cpd_subgraph,
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Simulation
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n--- Running simulation: {SIM_YEARS} steps ---\n")
graph.simulate(n_steps=SIM_YEARS, n_samples=N_DBN_SAMPLES)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Results
# ─────────────────────────────────────────────────────────────────────────────

plotter = Plot(graph=graph, output_dir=PLOT_DIR)
plotter.plot_history_all_together()
plotter.plot_prediction_all_together(n_steps=PREDICT_STEPS, n_samples=PREDICT_SAMPLES)
plotter.save_history_csv()

print(f"\nDone.  Results saved to '{PLOT_DIR}'.")