"""Draw vector architecture schematics (no simulated results).

f25_architecture goes to the F25 paper (paper1/), regression_architecture to the
four-block regression draft (paper_regression/). Sources: core/f25_models.py,
paper1/sections/methodology.tex, core/models.py,
paper_regression/sections/data_processing.tex. Run from any directory with
Python/Matplotlib.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
NAVY = "#233e54"
TEAL = "#167d8d"
ORANGE = "#b46b2d"
INK = "#253846"
GRAY = "#657480"
PALE = "#f0f5f7"
LIGHT_TEAL = "#e9f5f4"
LIGHT_ORANGE = "#fcf2e7"


def canvas(height):
    fig, ax = plt.subplots(figsize=(6.4, height))
    fig.subplots_adjust(left=.025, right=.975, bottom=.025, top=.975)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, sub="", *, color=NAVY, fill=PALE, fs=8.4, dashed=False):
    patch = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle="round,pad=0.1,rounding_size=1.1",
                          facecolor=fill, edgecolor=color, linewidth=.9,
                          linestyle=(0, (3, 2)) if dashed else "solid", zorder=3)
    ax.add_patch(patch)
    if sub:
        ax.text(x, y+h*.17, title, ha="center", va="center", color=color,
                weight="bold", fontsize=fs, zorder=4)
        ax.text(x, y-h*.22, sub, ha="center", va="center", color=INK,
                fontsize=fs-.5, linespacing=1.2, zorder=4)
    else:
        ax.text(x, y, title, ha="center", va="center", color=color,
                weight="bold", fontsize=fs, linespacing=1.2, zorder=4)


def arrow(ax, start, end, *, color=GRAY):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                linewidth=.9, color=color, zorder=2))


def line(ax, xy, *, color=GRAY):
    ax.plot([p[0] for p in xy], [p[1] for p in xy], color=color, lw=.9, zorder=1)


def save(fig, stem, paper):
    out = ROOT / paper / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=240, facecolor="white")
    plt.close(fig)


def regression():
    # Title and caveats belong to the LaTeX caption, not to the drawing.
    fig, ax = canvas(5.9)
    ax.set_ylim(3, 91)
    box(ax, 15, 84, 29, 11, "Time histories", "Eligible vehicle responses")
    box(ax, 50, 84, 30, 11, "Spatial crop", "100 samples/m")
    box(ax, 85, 84, 29, 11, "Shared noise", r"Gaussian, $\sigma_i=0.05|x_i|$")
    arrow(ax, (29.5, 84), (35, 84))
    arrow(ax, (65, 84), (70.5, 84))
    # Input representation is a binary experimental factor.
    line(ax, [(85, 78.5), (85, 73), (17, 73)])
    arrow(ax, (17, 73), (17, 69.5))
    arrow(ax, (50, 73), (50, 69.5))
    box(ax, 17, 65, 29, 9, "RAW", "Cropped sequence")
    box(ax, 50, 65, 29, 9, "PAA", "512 segment means", color=TEAL, fill=LIGHT_TEAL)
    line(ax, [(17, 60.5), (17, 57), (75, 57)])
    line(ax, [(50, 60.5), (50, 57)])
    arrow(ax, (75, 57), (75, 59.5))
    box(ax, 83, 65, 31, 11, "Standardization", "Training states only")
    # Optional position encoding precedes CNN; optional recurrence follows it.
    line(ax, [(83, 59.5), (83, 53), (17, 53)])
    arrow(ax, (17, 53), (17, 50))
    box(ax, 17, 43.5, 31, 13, "Position encoding", "Time2Vec-style; 8 features\nLegacy name: Space2Vec", color=TEAL,
        fill=LIGHT_TEAL, dashed=True)
    box(ax, 50, 43.5, 27, 11, "1-D CNN", "Convolution + ReLU\nOptional max pooling")
    box(ax, 83, 43.5, 29, 13, "LSTM", "Unidirectional\nOn / bypass", color=TEAL, fill=LIGHT_TEAL, dashed=True)
    ax.text(17, 34.8, "On / bypass", ha="center", fontsize=7.6, color=TEAL)
    arrow(ax, (32.5, 43.5), (36.5, 43.5))
    arrow(ax, (63.5, 43.5), (68.5, 43.5))
    # Pooling alternatives are mutually exclusive, not serial modules.
    line(ax, [(83, 37), (83, 31), (26, 31)])
    arrow(ax, (26, 31), (26, 28.5))
    arrow(ax, (76, 31), (76, 28.5))
    box(ax, 26, 23, 43, 11, "Pooling off: GAP", "One mean per feature channel")
    box(ax, 76, 23, 43, 11, "Pooling on: adaptive max pyramid", "N-HiTS-inspired; concatenate bins", color=ORANGE, fill=LIGHT_ORANGE,
        fs=8.0)
    line(ax, [(26, 17.5), (26, 14.4), (50, 14.4)])
    line(ax, [(76, 17.5), (76, 14.4), (50, 14.4)])
    arrow(ax, (50, 14.4), (50, 12.4))
    box(ax, 50, 8.5, 97, 7.5, "Dense regression heads", "Scour: 1 (F40) or 3 (L99)  |  + 2 bearing-fixity heads in M blocks", fs=8.4)
    save(fig, "regression_architecture", "paper_regression")


def f25():
    # Title and caveats belong to the LaTeX caption, not to the drawing.
    fig, ax = canvas(5.0)
    ax.set_xlim(-4, 100)
    ax.set_ylim(2.5, 90)
    box(ax, 16, 82.5, 31, 12, "Spatial crop + trim", "5,831 → 5,830 samples")
    box(ax, 50, 82.5, 29, 12, "Additive noise", r"$\sigma=0.05\,\mathrm{SD}(x)$")
    box(ax, 84, 82.5, 30, 12, "Min–max scaling", "Training set only")
    arrow(ax, (31.5, 82.5), (35.5, 82.5))
    arrow(ax, (64.5, 82.5), (69, 82.5))
    ax.text(0, 70, "Input / arm", color=GRAY, fontsize=7.8)
    ax.text(36, 70, "Feature extraction", color=GRAY, fontsize=7.8)
    ax.text(70, 70, "Aggregation", color=GRAY, fontsize=7.8)
    # A compact arm table keeps the distinction from the regression factorial clear.
    for y, title, sub, pool, color, fill in [
        (62, "RAW-CNN", "5,830 samples", "Flatten", NAVY, PALE),
        (46, "PAA-CNN", "583 block means", "Flatten", TEAL, LIGHT_TEAL),
        (30, "PAA-MR", "583 block means", "Adaptive max pyramid\n1 + 2 + 4 bins", ORANGE, LIGHT_ORANGE),
    ]:
        box(ax, 16, y, 31, 12, title, sub, color=color, fill=fill, fs=8.4)
        box(ax, 49, y, 27, 12, "1-D CNN", "Valid convolutions\nOptional max pooling")
        box(ax, 83, y, 31, 12, pool, color=color, fill=fill, fs=8.3)
        arrow(ax, (31.5,y), (35.5,y))
        arrow(ax, (62.5,y), (67.5,y))
    # A single trunk denotes common preprocessing for each of the three arms.
    line(ax, [(84, 76.5), (84, 73), (-3,73), (-3,30)])
    for y in (62,46,30):
        arrow(ax, (-3,y), (.4,y))
    # All arms use their dense layer + ten logits; rows are separate trained models.
    ax.text(50, 20.5, "Each arm: aggregation → dense + ReLU → 10 class logits", ha="center",
            color=NAVY, fontsize=8.7, weight="bold")
    # Mechanistic illustration of adaptive bins; this is a schematic, not measured data.
    ax.text(0, 14.8, "N-HiTS-inspired pooling", color=ORANGE, fontsize=8.5, weight="bold")
    for y, count in [(12.3, 1), (8.3,2), (4.3,4)]:
        ax.text(39, y+1.1, str(count), ha="right", fontsize=8, color=GRAY)
        for i in range(count):
            ax.add_patch(Rectangle((42+i*31/count, y), 31/count-.65, 2.2,
                                   facecolor=LIGHT_ORANGE, edgecolor=ORANGE, lw=.8))
    ax.text(77, 8.5, "7 maxima per\nfeature map", ha="left", va="center", color=ORANGE, fontsize=8.3)
    save(fig, "f25_architecture", "paper1")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    regression()
    f25()
    print("Generated regression_architecture and f25_architecture (PDF, PNG, SVG).")
