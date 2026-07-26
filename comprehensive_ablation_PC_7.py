"""
comprehensive_ablation_PC_7.py
==============================

Drives the long-running comprehensive ablation on PC_7. Splits the 4-arch
grid with PC_11 (this PC owns *PAA + N-HiTS* and *PAA + S2V + N-HiTS*).

Phases on this PC, in execution order:
    A. Single-DOF sweep      (8 DOFs)        — 2 archs × 8 DOFs × 3 seeds
    B. 8-DOF architecture     (1 set)         — 2 archs × 1 set × 3 seeds
    C. Leave-one-out          (8 drops)       — 2 archs × 8 drops × 3 seeds
    D. Pair sweep             (28 pairs)      — 2 archs × 28 pairs × 3 seeds
                                              = 90 (arch, DOF) combos × 3 seeds
                                              = 270 Optuna studies × 200 trials
                                              = 54 000 trials total

Resumability
------------
Both `execute_ablation_pipeline` and the per-study Optuna SQLite files use
`load_if_exists=True`, so re-running the script after a power blip / crash
picks up exactly where it stopped without duplicate work.

Naming convention
-----------------
Every Optuna study and every output sub-folder embeds the sampler seed in
its name, e.g.

    PAA_S2V_NHiTS_DOFs_4_6_seed1337

so seeds never collide on disk and post-hoc analysis can compute a
median-of-seeds UCB-95 ranking trivially.
"""

import copy
import itertools

import optuna

from core.utils       import (set_global_seed, define_save_locations,
                              IDX_TO_DOF_NAME)
from training.pipeline import execute_ablation_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
#  Configuration — keep this block in sync with comprehensive_ablation_PC_11.py
# =============================================================================
DATASET        = "data_all_variabilities"
DISCRETIZATION = 1
N_TRIALS       = 200
EPOCHS         = 50
OPTUNA_SEEDS   = [42, 1337, 2026]      # 3 sampler seeds per (arch, DOFs)
ALL_DOFS       = list(range(8))

# This PC owns 2 of the 4 reduced architectures.
ARCHITECTURES = [
    {
        "name_short":    "PAA_NHiTS",
        "method":        "PAA",
        "use_space2vec": False,
        "use_lstm":      False,
        "use_nhits":     True,
        "model_type":    "1D_MODULAR",
    },
    {
        "name_short":    "PAA_S2V_NHiTS",
        "method":        "PAA",
        "use_space2vec": True,
        "use_lstm":      False,
        "use_nhits":     True,
        "model_type":    "1D_MODULAR",
    },
]


# =============================================================================
#  Helpers
# =============================================================================
def make_config(arch: dict, dofs: list[int], seed: int) -> dict:
    """Build one experiment_path entry with seed-tagged name."""
    dof_str = "_".join(str(d) for d in dofs)
    return {
        "name":          f"{arch['name_short']}_DOFs_{dof_str}_seed{seed}",
        "seed":          seed,
        "method":        arch["method"],
        "dofs":          list(dofs),
        "discretization": DISCRETIZATION,
        "use_space2vec": arch["use_space2vec"],
        "use_lstm":      arch["use_lstm"],
        "use_nhits":     arch["use_nhits"],
        "model_type":    arch["model_type"],
    }


def run_phase(phase_label: str, dofs: list[int]) -> None:
    """
    Run all (architecture × seed) combinations on a single DOF subset.
    Each (arch, seed) becomes one entry inside one shared experiment_path,
    so Optuna shares the SQLite DB across the seeds for that DOF subset.
    The seed is in the study name, so studies don't overwrite each other.

    A separate seed-tagged save location is also created so the output
    folder structure mirrors the study names.
    """
    for seed in OPTUNA_SEEDS:
        configs = [make_config(arch, dofs, seed) for arch in ARCHITECTURES]
        db, out, cache = define_save_locations(
            f"{phase_label}_seed{seed}",
            dataset       = DATASET,
            DOFs          = dofs,
            discretization= DISCRETIZATION,
        )
        print(f"\n{'#' * 64}")
        print(f"#  {phase_label}   DOFs={dofs}   sampler-seed={seed}")
        print(f"#  archs: {[a['name_short'] for a in ARCHITECTURES]}")
        print(f"{'#' * 64}")
        try:
            execute_ablation_pipeline(
                experiment_path  = configs,
                database_name    = db,
                output_dir_name  = out,
                cache_dir_name   = cache,
                dataset          = DATASET,
                n_trials         = N_TRIALS,
                epochs           = EPOCHS,
                skip_robustness  = False,    # always run 30-seed MC
                optuna_seed      = seed,     # thread seed into TPESampler
                use_pruner       = True,     # SuccessiveHalvingPruner: kills bad trials early
            )
        except KeyboardInterrupt:
            print(f"\n  Interrupted in {phase_label} (seed={seed}). Resume by re-running the script.")
            raise


# =============================================================================
#  Phase A — Single-DOF sweep   (8 DOFs)
# =============================================================================
def phase_A_single_dof() -> None:
    print("\n" + "=" * 64)
    print("  PHASE A — single-DOF sweep")
    print("=" * 64)
    for dof in ALL_DOFS:
        sensor = IDX_TO_DOF_NAME[dof]
        run_phase(f"CompA_SingleDOF_{sensor}", [dof])


# =============================================================================
#  Phase B — full 8-DOF architecture ablation
# =============================================================================
def phase_B_full_eight() -> None:
    print("\n" + "=" * 64)
    print("  PHASE B — 8-DOF architecture ablation")
    print("=" * 64)
    run_phase("CompB_Full8DOF", ALL_DOFS)


# =============================================================================
#  Phase C — Leave-one-out
# =============================================================================
def phase_C_leave_one_out() -> None:
    print("\n" + "=" * 64)
    print("  PHASE C — leave-one-out (all 4 architectures)")
    print("=" * 64)
    for drop in ALL_DOFS:
        remaining = [d for d in ALL_DOFS if d != drop]
        sensor = IDX_TO_DOF_NAME[drop]
        run_phase(f"CompC_LOO_drop_{sensor}", remaining)


# =============================================================================
#  Phase D — Pair sweep   (all 28 pairs)
# =============================================================================
def phase_D_pairs() -> None:
    print("\n" + "=" * 64)
    print("  PHASE D — pair sweep (all 28)")
    print("=" * 64)
    for pair in itertools.combinations(ALL_DOFS, 2):
        a, b = (IDX_TO_DOF_NAME[i] for i in pair)
        run_phase(f"CompD_Pair_{a}__{b}", list(pair))


# =============================================================================
#  Driver
# =============================================================================
if __name__ == "__main__":
    set_global_seed(42)

    # Order chosen so that the cheapest, most diagnostic phases finish first.
    # If the script is ever stopped mid-run, A and B already give a usable
    # picture; C and D extend it.
    phase_A_single_dof()
    phase_B_full_eight()
    phase_C_leave_one_out()
    phase_D_pairs()

    print("\n" + "=" * 64)
    print("  PC_7 comprehensive ablation finished.")
    print("=" * 64)
