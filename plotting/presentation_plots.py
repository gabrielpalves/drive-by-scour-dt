"""
presentation_plots.py
=====================

Presentation-oriented plots for the drive-by scour ablation (PC7 + PC11).
Companion to ``aggregate_ablation.py`` (which produces the winner-only figures
1-4 and the master CSVs). This script focuses on:

  1. Architecture x seed grouped BOXPLOTS (3 seeds side-by-side per
     architecture) with two overlays per box:
        * UCB-95 of the mean MSE   (mean + 1.96 * SE)
        * the best Optuna validation MSE found in the 200-trial HPO study
     Rendered for the Full-8DOF set and for the best 2-sensor pair.
  2. Confusion matrix of the champion (PAA + N-HiTS) and a 2x2 panel of all
     four architectures, on Full-8DOF (seeds summed, row-normalised).
  3. All-FOUR-architecture versions of the single-DOF, leave-one-out, and
     pair sweeps (the aggregate_ablation figures show only the winner).

Everything reads the aggregated PC7/PC11 results; tweak the CONFIG block to
change scales / outlier handling / which sets are plotted, then re-run.

Usage
-----
    python plotting/presentation_plots.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse constants + loaders from the sibling analyzer.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aggregate_ablation import (  # noqa: E402
    ARCH_LABEL, ARCH_ORDER, COLLAPSE_ACC, DOF_NAMES, LEAF_RE, OUT_DIR,
    PALETTE, REPO_ROOT, RESULT_ROOTS, load_runs, summarise,
)

# =========================================================================== #
#  CONFIG  - tweak these and re-run
# =========================================================================== #
SEEDS = [42, 1337, 2026]
SEED_COLOR = {42: "#4C72B0", 1337: "#DD8452", 2026: "#55A868"}

LOG_SCALE_BOX = False    # log y-axis on the boxplots (helps if collapses kept)
SHOW_FLIERS = False      # show catastrophic-collapse outliers as boxplot fliers
LOG_X_BARS = True        # log x-axis on the single-DOF all-arch bar chart
PAIR_HEATMAP_VMAX = 1.5  # cap the pair-heatmap colour scale (median MSE units)
TOP_N_PAIRS_PER_ARCH = 28  # how many pairs in the per-arch ranking (max 28)
DPI = 200

FULL8 = "0_1_2_3_4_5_6_7"
DB_ROOTS = {"PC7": REPO_ROOT / "PC7" / "database",
            "PC11": REPO_ROOT / "PC11" / "database"}
PRES_DIR = OUT_DIR / "presentation"


# =========================================================================== #
#  Data access
# =========================================================================== #
def get_master() -> pd.DataFrame:
    csv = OUT_DIR / "master_long.csv"
    if csv.exists():
        return pd.read_csv(csv, dtype={"dof_set": str})
    print("  master_long.csv not found - rebuilding from result trees ...")
    df = load_runs()
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(csv, index=False)
    return df


def load_optuna_best() -> pd.DataFrame:
    """Best (minimum) completed validation MSE per Optuna study."""
    csv = OUT_DIR / "optuna_best.csv"
    if csv.exists():
        return pd.read_csv(csv, dtype={"dof_set": str})
    rows: list[dict] = []
    for _pc, dbroot in DB_ROOTS.items():
        if not dbroot.exists():
            continue
        for db in sorted(dbroot.glob("*.db")):
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                studies = con.execute(
                    "select study_id, study_name from studies").fetchall()
                for sid, name in studies:
                    r = con.execute(
                        "select min(tv.value) from trials t "
                        "join trial_values tv on tv.trial_id=t.trial_id "
                        "where t.study_id=? and t.state='COMPLETE'", (sid,)
                    ).fetchone()
                    best = r[0] if r else None
                    m = LEAF_RE.match(name)
                    if m and best is not None:
                        rows.append(dict(arch=m["arch"], dof_set=m["dofs"],
                                         seed=int(m["seed"]), optuna_best=best))
            finally:
                con.close()
    out = pd.DataFrame(rows)
    out.to_csv(csv, index=False)
    print(f"  extracted {len(out)} Optuna best values -> optuna_best.csv")
    return out


def find_leaf_dirs(arch: str, dof_set: str, phase: str = "CompB") -> list[Path]:
    dirs: list[Path] = []
    for root in RESULT_ROOTS.values():
        for comp in root.glob(phase + "*"):
            dirs.extend(comp.glob(f"{arch}_DOFs_{dof_set}_seed*"))
    return dirs


def champion_confusion(arch: str, dof_set: str = FULL8) -> np.ndarray | None:
    """Sum the confusion matrices over all seeds for one (arch, DOF-set)."""
    cm = None
    for leaf in find_leaf_dirs(arch, dof_set):
        f = leaf / "DT_conf_matrix.npy"
        if f.exists():
            m = np.load(f)
            cm = m if cm is None else cm + m
    return cm


# =========================================================================== #
#  1. Architecture x seed boxplots
# =========================================================================== #
def boxplot_arch_by_seed(df: pd.DataFrame, opt: pd.DataFrame,
                         dof_set: str, title: str, fname: str) -> None:
    sub = df[df.dof_set == dof_set]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    nb = len(SEEDS)
    off = np.linspace(-0.26, 0.26, nb)
    bw = 0.46 / nb

    ucb_handle = opt_handle = None
    for ai, arch in enumerate(ARCH_ORDER):
        for si, seed in enumerate(SEEDS):
            cell = sub[(sub.arch == arch) & (sub.seed == seed)]
            if cell.empty:
                continue
            # Drop degenerate collapses from the box/UCB stats; report their
            # count instead so the error-distribution axis stays meaningful.
            ok = cell[cell.accuracy >= COLLAPSE_ACC]
            n_collapse = len(cell) - len(ok)
            vals = ok.mse.to_numpy()
            if vals.size == 0:
                continue
            x = ai + off[si]
            bp = ax.boxplot(vals, positions=[x], widths=bw,
                            patch_artist=True, showfliers=SHOW_FLIERS,
                            medianprops=dict(color="black", lw=1.4))
            for box in bp["boxes"]:
                box.set(facecolor=SEED_COLOR[seed], alpha=0.85,
                        edgecolor="black", lw=0.8)
            # UCB-95 of the mean (collapses excluded)
            se = vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
            ucb = vals.mean() + 1.96 * se
            (ucb_handle,) = ax.plot([x - bw / 2, x + bw / 2], [ucb, ucb],
                                    color="crimson", lw=1.8, zorder=5)
            if n_collapse:
                ax.annotate(f"{n_collapse}X", (x, ucb), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7,
                            color="crimson", fontweight="bold")
            # Optuna best validation MSE
            ob = opt[(opt.arch == arch) & (opt.dof_set == dof_set)
                     & (opt.seed == seed)].optuna_best
            if not ob.empty:
                (opt_handle,) = ax.plot([x], [float(ob.iloc[0])], marker="*",
                                        ms=11, color="gold",
                                        markeredgecolor="black", zorder=6)

    ax.set_xticks(range(len(ARCH_ORDER)))
    ax.set_xticklabels([ARCH_LABEL[a] for a in ARCH_ORDER], rotation=15,
                       ha="right")
    ax.set_ylabel("MSE  (scour-class units)  -  lower is better")
    if LOG_SCALE_BOX:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    handles = [mpatches.Patch(facecolor=SEED_COLOR[s], edgecolor="black",
                              label=f"seed {s}") for s in SEEDS]
    if ucb_handle is not None:
        handles.append(ucb_handle); ucb_handle.set_label("UCB-95 (mean MSE)")
    if opt_handle is not None:
        handles.append(opt_handle); opt_handle.set_label("Optuna best (val MSE)")
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="upper left")
    fig.text(0.99, 0.01,
             "box = stochastic evals (collapses excluded) - X = # collapsed evals",
             ha="right", fontsize=8, color="#666")
    fig.tight_layout()
    fig.savefig(PRES_DIR / fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
#  2. Confusion matrices
# =========================================================================== #
def _plot_cm(ax, cm: np.ndarray, title: str) -> None:
    total = cm.sum()
    acc = np.trace(cm) / total
    idx = np.arange(cm.shape[0])
    di, dj = np.meshgrid(idx, idx, indexing="ij")
    within1 = cm[np.abs(di - dj) <= 1].sum() / total
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)  # row-normalise
    im = ax.imshow(cmn, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"{title}\nacc={acc:.1%}  -  within ±1 class={within1:.1%}",
                 fontsize=10)
    ax.set_xlabel("predicted scour class")
    ax.set_ylabel("true scour class")
    return im


def confusion_champion(arch: str) -> None:
    cm = champion_confusion(arch)
    if cm is None:
        print(f"  [warn] no confusion matrix for {arch}")
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = _plot_cm(ax, cm, f"Champion confusion - {ARCH_LABEL[arch]} (Full 8-DOF)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised frequency")
    fig.tight_layout()
    fig.savefig(PRES_DIR / "confusion_champion.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def confusion_all_archs() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    im = None
    for ax, arch in zip(axes.ravel(), ARCH_ORDER):
        cm = champion_confusion(arch)
        if cm is None:
            ax.axis("off"); continue
        im = _plot_cm(ax, cm, ARCH_LABEL[arch])
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025,
                     label="row-normalised frequency")
    fig.suptitle("Confusion matrices on Full 8-DOF (seeds summed)")
    fig.savefig(PRES_DIR / "confusion_all_archs.png", dpi=DPI)
    plt.close(fig)


# =========================================================================== #
#  3. All-architecture sweeps
# =========================================================================== #
def _grouped_barh(ax, pivot: pd.DataFrame, archs: list[str]) -> None:
    """pivot: index = category, columns = arch, values = metric."""
    n = len(archs)
    yb = np.arange(len(pivot))
    h = 0.8 / n
    for k, arch in enumerate(archs):
        ax.barh(yb + (k - (n - 1) / 2) * h, pivot[arch], height=h,
                color=PALETTE[arch], label=ARCH_LABEL[arch])
    ax.set_yticks(yb)
    ax.set_yticklabels(pivot.index)
    ax.invert_yaxis()


def all_archs_single_dof(summary: pd.DataFrame) -> None:
    s = summary[summary.phase == "CompA"].copy()
    s["dof"] = s.dof_set.astype(int).map(lambda i: DOF_NAMES[i])
    piv = s.pivot_table(index="dof", columns="arch", values="mse_median")
    piv = piv.reindex(columns=ARCH_ORDER)
    piv = piv.loc[piv.mean(axis=1).sort_values().index]  # best sensors on top
    fig, ax = plt.subplots(figsize=(9.5, 6))
    _grouped_barh(ax, piv, ARCH_ORDER)
    if LOG_X_BARS:
        ax.set_xscale("log")
    ax.set_xlabel("median MSE  (scour-class units, log scale)  -  lower is better")
    ax.set_title("Single-sensor informativeness - all architectures")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="x", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(PRES_DIR / "all_archs_single_dof.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)


def all_archs_leave_one_out(summary: pd.DataFrame) -> None:
    base = (summary[summary.phase == "CompB"]
            .set_index("arch").mse_median.to_dict())
    s = summary[summary.phase == "CompC"].copy()
    s["delta"] = s.apply(lambda r: r.mse_median - base.get(r.arch, np.nan), axis=1)
    piv = s.pivot_table(index="dropped_dof", columns="arch", values="delta")
    piv = piv.reindex(columns=ARCH_ORDER)
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    _grouped_barh(ax, piv, ARCH_ORDER)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Δ median MSE vs each architecture's full-8 baseline  "
                  "-  positive = sensor is important")
    ax.set_title("Leave-one-out sensor importance - all architectures")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PRES_DIR / "all_archs_leave_one_out.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)


def all_archs_pairs_heatmap(summary: pd.DataFrame) -> None:
    s = summary[summary.phase == "CompD"].copy()
    piv = s.pivot_table(index="dof_label", columns="arch", values="mse_median")
    piv = piv.reindex(columns=ARCH_ORDER)
    piv = piv.loc[piv.mean(axis=1).sort_values().index]  # best pairs on top
    piv = piv.head(TOP_N_PAIRS_PER_ARCH)
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(piv) + 1.8))
    data = piv.to_numpy()
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=np.nanmin(data),
                   vmax=PAIR_HEATMAP_VMAX)
    ax.set_xticks(range(len(ARCH_ORDER)))
    ax.set_xticklabels([ARCH_LABEL[a] for a in ARCH_ORDER], rotation=20,
                       ha="right", fontsize=8)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=7)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="black" if v < PAIR_HEATMAP_VMAX * 0.6 else "white")
    fig.colorbar(im, ax=ax, fraction=0.04, label="median MSE (capped)")
    ax.set_title("2-sensor pairs - median MSE by architecture")
    fig.tight_layout()
    fig.savefig(PRES_DIR / "all_archs_pairs_heatmap.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
def main() -> None:
    PRES_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading aggregated data ...")
    df = get_master()
    summary = summarise(df)
    opt = load_optuna_best()

    # winner + best pair (by the winning architecture)
    winner = (summary[summary.phase == "CompB"]
              .sort_values("mse_median").arch.iloc[0])
    best_pair = (summary[(summary.phase == "CompD") & (summary.arch == winner)]
                 .sort_values("mse_median").iloc[0])
    print(f"  winner: {ARCH_LABEL[winner]} | best pair: {best_pair.dof_label} "
          f"(dof_set={best_pair.dof_set})")

    print("Rendering boxplots ...")
    boxplot_arch_by_seed(df, opt, FULL8,
                         "Architecture x seed - Full 8-DOF input",
                         "box_architecture_full8dof.png")
    boxplot_arch_by_seed(df, opt, best_pair.dof_set,
                         f"Architecture x seed - best 2-sensor input "
                         f"({best_pair.dof_label})",
                         "box_architecture_bestpair.png")

    print("Rendering confusion matrices ...")
    confusion_champion(winner)
    confusion_all_archs()

    print("Rendering all-architecture sweeps ...")
    all_archs_single_dof(summary)
    all_archs_leave_one_out(summary)
    all_archs_pairs_heatmap(summary)

    print(f"Done. Figures in {PRES_DIR}")


if __name__ == "__main__":
    main()
