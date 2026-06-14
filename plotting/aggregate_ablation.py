"""
aggregate_ablation.py
=====================

Post-hoc analysis of the comprehensive drive-by scour ablation.

Walks the PC7 and PC11 result trees, parses every champion run's
``robustness_stochastic.json`` (30 stochastic evaluations of accuracy / MAE /
MSE, where MSE/MAE are measured in *scour-class units* on the 0-60 ordinal
label), and builds tidy tables plus the four headline figures for the
ablation paper.

Why median, not mean
--------------------
A model fed an uninformative channel (e.g. a single wheel/pitch sensor)
occasionally fails to train and *collapses* to near-random predictions. On the
0-60 ordinal label this produces per-eval MSE in the hundreds (predicting the
constant mean class already gives MSE ~310), so a handful of collapses dominate
the *mean*. We therefore report the **median MSE** (robust central tendency)
with the inter-quartile range, plus an explicit **collapse rate**
(fraction of stochastic evals with accuracy < ``COLLAPSE_ACC``). The collapse
rate is itself a result: it measures how reliably a given (architecture, sensor
set) can be trained at all.

Experiment grid recovered from disk
-----------------------------------
    4 architectures  x  45 DOF-sets  x  3 seeds  = 540 champion runs
        PC7  : PAA_NHiTS , PAA_S2V_NHiTS
        PC11 : PAA_LSTM_NHiTS , PAA_S2V_LSTM_NHiTS
        DOF-sets : 8 single (CompA) + 1 full-8 (CompB)
                 + 8 leave-one-out (CompC) + 28 pairs (CompD)
        seeds    : 42, 1337, 2026

Outputs (under ``ablation_analysis/``)
--------------------------------------
    master_long.csv        one row per (run, stochastic eval)
    summary_per_config.csv one row per (arch, DOF-set) pooled over seeds
    figures/fig1_architecture_full8dof.png
    figures/fig2_single_dof_importance.png
    figures/fig3_leave_one_out.png
    figures/fig4_best_pairs.png

Usage
-----
    python plotting/aggregate_ablation.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOTS = {
    "PC7": REPO_ROOT / "PC7" / "results",
    "PC11": REPO_ROOT / "PC11" / "results",
}
OUT_DIR = REPO_ROOT / "ablation_analysis"
FIG_DIR = OUT_DIR / "figures"

# A stochastic eval is "collapsed" if the champion degenerated to ~random
# predictions. Healthy models sit at 0.6-0.7 accuracy; chance is 1/61 ~ 0.016.
COLLAPSE_ACC = 0.10

# DOF index -> name (mirrors core/utils.py DOF_NAMES; kept inline so this
# script has no heavy project imports).
DOF_NAMES = [
    "CarBody_Vert",      # 0
    "FrontBogie_Vert",   # 1
    "RearBogie_Vert",    # 2
    "Wheel1_Vert",       # 3
    "Wheel2_Vert",       # 4
    "CarBody_Pitch",     # 5
    "FrontBogie_Pitch",  # 6
    "RearBogie_Pitch",   # 7
]
ALL_DOFS = set(range(len(DOF_NAMES)))

# Logical display order + friendly labels for the four architectures.
ARCH_ORDER = ["PAA_NHiTS", "PAA_S2V_NHiTS", "PAA_LSTM_NHiTS", "PAA_S2V_LSTM_NHiTS"]
ARCH_LABEL = {
    "PAA_NHiTS": "PAA + N-HiTS",
    "PAA_S2V_NHiTS": "PAA + S2V + N-HiTS",
    "PAA_LSTM_NHiTS": "PAA + LSTM + N-HiTS",
    "PAA_S2V_LSTM_NHiTS": "PAA + S2V + LSTM + N-HiTS",
}

PHASE_LABEL = {
    "CompA": "Single-DOF",
    "CompB": "Full 8-DOF",
    "CompC": "Leave-one-out",
    "CompD": "DOF pair",
}

# leaf folder, e.g. PAA_S2V_NHiTS_DOFs_0_1_2_3_4_5_6_7_seed1337
LEAF_RE = re.compile(r"^(?P<arch>.+)_DOFs_(?P<dofs>[\d_]+)_seed(?P<seed>\d+)$")

PALETTE = {
    "PAA_NHiTS": "#4C72B0",
    "PAA_S2V_NHiTS": "#55A868",
    "PAA_LSTM_NHiTS": "#C44E52",
    "PAA_S2V_LSTM_NHiTS": "#8172B3",
}


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def dof_list_label(dofs: list[int]) -> str:
    """Compact human label for a DOF subset."""
    return " + ".join(DOF_NAMES[d] for d in dofs)


def load_runs() -> pd.DataFrame:
    """Walk both result trees and return a tidy long DataFrame (one row per
    stochastic evaluation)."""
    rows: list[dict] = []
    n_runs = 0
    n_missing = 0

    for pc, root in RESULT_ROOTS.items():
        if not root.exists():
            print(f"  [warn] result root not found: {root}")
            continue
        for comp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            phase_key = comp_dir.name.split("_", 1)[0]  # CompA..CompD
            for leaf in sorted(p for p in comp_dir.iterdir() if p.is_dir()):
                m = LEAF_RE.match(leaf.name)
                if not m:
                    continue
                arch = m.group("arch")
                dofs = [int(x) for x in m.group("dofs").split("_")]
                seed = int(m.group("seed"))

                rj = leaf / "robustness_stochastic.json"
                if not rj.exists():
                    n_missing += 1
                    continue
                with rj.open() as fh:
                    data = json.load(fh)
                accs = data.get("accuracies", [])
                maes = data.get("maes", [])
                mses = data.get("mses", [])
                n_runs += 1

                dropped = sorted(ALL_DOFS - set(dofs)) if phase_key == "CompC" else []
                for i, (a, ma, ms) in enumerate(zip(accs, maes, mses)):
                    rows.append(
                        dict(
                            pc=pc,
                            phase=phase_key,
                            phase_label=PHASE_LABEL.get(phase_key, phase_key),
                            arch=arch,
                            dof_set=m.group("dofs"),
                            n_dofs=len(dofs),
                            dof_label=dof_list_label(dofs),
                            dropped_dof=DOF_NAMES[dropped[0]] if dropped else "",
                            seed=seed,
                            eval_idx=i,
                            accuracy=a,
                            mae=ma,
                            mse=ms,
                        )
                    )

    print(f"  parsed {n_runs} champion runs "
          f"({len(rows)} stochastic evals); {n_missing} missing robustness files")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Aggregation  (robust: median + IQR + collapse rate)
# --------------------------------------------------------------------------- #
def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (arch, phase, dof_set) pooled over 3 seeds x 30 evals."""
    g = df.groupby(["arch", "phase", "phase_label", "dof_set", "dof_label",
                    "n_dofs", "dropped_dof"], dropna=False)
    out = g.agg(
        mse_median=("mse", "median"),
        mse_q25=("mse", lambda s: s.quantile(0.25)),
        mse_q75=("mse", lambda s: s.quantile(0.75)),
        mse_mean=("mse", "mean"),               # kept for reference only
        mae_median=("mae", "median"),
        acc_median=("accuracy", "median"),
        acc_q25=("accuracy", lambda s: s.quantile(0.25)),
        acc_q75=("accuracy", lambda s: s.quantile(0.75)),
        collapse_rate=("accuracy", lambda s: float((s < COLLAPSE_ACC).mean())),
        n=("mse", "size"),
    ).reset_index()
    return out


def _iqr_err(sub: pd.DataFrame, med: str, lo: str, hi: str) -> np.ndarray:
    """Asymmetric error-bar array [[lower...],[upper...]] from median+quartiles."""
    return np.vstack([(sub[med] - sub[lo]).to_numpy(),
                      (sub[hi] - sub[med]).to_numpy()])


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def fig1_architecture(summary: pd.DataFrame) -> str:
    """Headline: 4 architectures on the Full-8DOF set (CompB)."""
    sub = summary[summary.phase == "CompB"].set_index("arch").reindex(ARCH_ORDER)
    fig, (axm, axa) = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(len(ARCH_ORDER))
    colors = [PALETTE[a] for a in ARCH_ORDER]
    best = sub.mse_median.idxmin()

    yerr = np.vstack([(sub.mse_median - sub.mse_q25).to_numpy(),
                      (sub.mse_q75 - sub.mse_median).to_numpy()])
    bars = axm.bar(x, sub.mse_median, yerr=yerr, capsize=4, color=colors)
    bi = ARCH_ORDER.index(best)
    bars[bi].set_edgecolor("black")
    bars[bi].set_linewidth(2)
    axm.set_ylabel("median MSE  (scour-class units)  —  lower is better")
    axm.set_title("Architecture comparison — Full 8-DOF input")
    for xi, v, q, cr in zip(x, sub.mse_median, sub.mse_q75, sub.collapse_rate):
        tag = f"{v:.3f}" + (f"\ncollapse {cr:.1%}" if cr > 0 else "")
        axm.text(xi, q, tag, ha="center", va="bottom", fontsize=8)

    yerr_a = np.vstack([(sub.acc_median - sub.acc_q25).to_numpy(),
                        (sub.acc_q75 - sub.acc_median).to_numpy()])
    axa.bar(x, sub.acc_median, yerr=yerr_a, capsize=4, color=colors)
    axa.set_ylabel("median accuracy  —  higher is better")
    axa.set_title("(secondary) Classification accuracy")
    for xi, v, q in zip(x, sub.acc_median, sub.acc_q75):
        axa.text(xi, q, f"{v:.1%}", ha="center", va="bottom", fontsize=8)

    for ax in (axm, axa):
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_LABEL[a] for a in ARCH_ORDER],
                           rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Median ± IQR pooled over 3 seeds × 30 stochastic evals",
                 y=1.02, fontsize=9, color="#555")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_architecture_full8dof.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    return best


def fig2_single_dof(summary: pd.DataFrame, best_arch: str) -> None:
    """Single-DOF informativeness for the winning architecture."""
    sub = summary[(summary.phase == "CompA") & (summary.arch == best_arch)].copy()
    sub = sub.sort_values("mse_median")  # best (lowest MSE) first
    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(sub))
    ax.barh(y, sub.mse_median, xerr=_iqr_err(sub, "mse_median", "mse_q25", "mse_q75"),
            capsize=3, color=PALETTE.get(best_arch, "#4C72B0"))
    ax.set_yticks(y)
    ax.set_yticklabels(sub.dof_label)
    ax.invert_yaxis()  # best on top
    ax.set_xlabel("median MSE  (scour-class units)  —  lower is better")
    ax.set_title(f"Single-sensor informativeness — {ARCH_LABEL[best_arch]}")
    for yi, v, q, cr in zip(y, sub.mse_median, sub.mse_q75, sub.collapse_rate):
        tag = f" {v:.2f}" + (f"  (collapse {cr:.0%})" if cr > 0.01 else "")
        ax.text(q, yi, tag, va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_single_dof_importance.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def fig3_leave_one_out(summary: pd.DataFrame, best_arch: str) -> None:
    """Delta-(median MSE) vs the full-8 baseline when each DOF is removed."""
    base = summary[(summary.phase == "CompB") & (summary.arch == best_arch)]
    if base.empty:
        return
    base_mse = float(base.mse_median.iloc[0])
    sub = summary[(summary.phase == "CompC") & (summary.arch == best_arch)].copy()
    sub["delta"] = sub.mse_median - base_mse  # >0 means dropping it hurts
    sub = sub.sort_values("delta", ascending=False)  # most important on top

    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(sub))
    colors = ["#C44E52" if d > 0 else "#55A868" for d in sub.delta]
    ax.barh(y, sub.delta, color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(sub.dropped_dof)
    ax.invert_yaxis()
    ax.set_xlabel(f"Δ median MSE vs full-8 baseline (={base_mse:.3f})  "
                  f"—  positive = sensor is important")
    ax.set_title(f"Leave-one-out sensor importance — {ARCH_LABEL[best_arch]}")
    for yi, d in zip(y, sub.delta):
        ax.text(d, yi, f" {d:+.3f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_leave_one_out.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig4_best_pairs(summary: pd.DataFrame, best_arch: str, top_n: int = 14) -> None:
    """Top DOF pairs for the winning architecture, with single/full refs."""
    sub = summary[(summary.phase == "CompD") & (summary.arch == best_arch)].copy()
    sub = sub.sort_values("mse_median").head(top_n)

    best_single = summary[(summary.phase == "CompA") &
                          (summary.arch == best_arch)].mse_median.min()
    full8 = summary[(summary.phase == "CompB") &
                    (summary.arch == best_arch)].mse_median
    full8 = float(full8.iloc[0]) if not full8.empty else None

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    y = np.arange(len(sub))
    ax.barh(y, sub.mse_median, xerr=_iqr_err(sub, "mse_median", "mse_q25", "mse_q75"),
            capsize=3, color=PALETTE.get(best_arch, "#4C72B0"))
    ax.set_yticks(y)
    ax.set_yticklabels(sub.dof_label, fontsize=8)
    ax.invert_yaxis()
    if np.isfinite(best_single):
        ax.axvline(best_single, color="#888", ls="--", lw=1.2,
                   label=f"best single-DOF ({best_single:.3f})")
    if full8 is not None:
        ax.axvline(full8, color="black", ls=":", lw=1.2,
                   label=f"full 8-DOF ({full8:.3f})")
    ax.set_xlabel("median MSE  (scour-class units)  —  lower is better")
    ax.set_title(f"Best 2-sensor combinations — {ARCH_LABEL[best_arch]}")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_best_pairs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Ranking table
# --------------------------------------------------------------------------- #
def print_ranking(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("ARCHITECTURE RANKING  —  Full 8-DOF (CompB), pooled over seeds (robust)")
    print("=" * 78)
    sub = (summary[summary.phase == "CompB"]
           .sort_values("mse_median"))
    print(f"{'architecture':<26}{'MSE(med)':>9}{'IQR':>14}{'acc':>8}"
          f"{'collapse':>10}{'n':>5}")
    for _, r in sub.iterrows():
        iqr = f"[{r.mse_q25:.2f},{r.mse_q75:.2f}]"
        print(f"{ARCH_LABEL.get(r.arch, r.arch):<26}"
              f"{r.mse_median:>9.3f}{iqr:>14}{r.acc_median:>8.1%}"
              f"{r.collapse_rate:>9.1%}{int(r.n):>6}")
    print("-" * 78)
    print(f"collapse = fraction of evals with accuracy < {COLLAPSE_ACC:.0%} "
          "(model degenerated to ~random). Mean MSE is reported in the CSV but "
          "is NOT robust here (collapses inflate it by 10-100x).")


# --------------------------------------------------------------------------- #
def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading runs from PC7 + PC11 ...")
    df = load_runs()
    if df.empty:
        print("  [error] no runs parsed — nothing to do.")
        return

    df.to_csv(OUT_DIR / "master_long.csv", index=False)
    summary = summarise(df)
    summary.sort_values(["phase", "arch", "mse_median"]).to_csv(
        OUT_DIR / "summary_per_config.csv", index=False)
    print(f"  wrote master_long.csv ({len(df)} rows) and summary_per_config.csv "
          f"({len(summary)} configs)")

    print_ranking(summary)

    print("\nRendering figures ...")
    best_arch = fig1_architecture(summary)
    print(f"  winning architecture (lowest Full-8DOF median MSE): "
          f"{ARCH_LABEL.get(best_arch, best_arch)}")
    fig2_single_dof(summary, best_arch)
    fig3_leave_one_out(summary, best_arch)
    fig4_best_pairs(summary, best_arch)
    print(f"  figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
