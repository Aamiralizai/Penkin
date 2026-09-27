#!/usr/bin/env python3
"""Generate manuscript Figure 1: biological architecture, mathematical model, and workflow.

This figure is intentionally generated from code so the model architecture shown in the
manuscript remains synchronized with the implemented equations and the 8,000-candidate
reference ensemble.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from penkin.paths import RESULTS_DIR, FIGURES_DIR
FIGDIR = FIGURES_DIR
os.makedirs(FIGDIR, exist_ok=True)

COL = {
    "ACVS": "#c0392b",
    "IPNS": "#1f77b4",
    "PCL": "#2e8b6f",
    "IAT": "#e07b39",
    "IAH": "#7b4f8f",
    "neutral": "#4a4a4a",
    "bluebox": "#e8eef7",
    "greenbox": "#eaf3ee",
    "redbox": "#fff2ef",
    "purplebox": "#f4edf7",
    "goldbox": "#fff8e6",
}


def _box(ax, x, y, w, h, text, fc, ec="#46516a", fs=9.5, lw=1.3, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                       fc=fc, ec=ec, lw=lw, linestyle=ls)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs)
    return p


def _arrow(ax, x1, y1, x2, y2, text=None, color="#566273", lw=1.6, ls="-", text_offset=0.018,
           mutation=12):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=mutation,
                        color=color, lw=lw, linestyle=ls, connectionstyle="arc3")
    ax.add_patch(a)
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + text_offset, text, ha="center", va="center",
                fontsize=8.5, color=color, fontstyle="italic")
    return a


def panel_a(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("A  Biological architecture", loc="left", fontweight="bold", fontsize=15, pad=8)

    # Compartments
    ax.add_patch(Rectangle((0.015, 0.56), 0.97, 0.40, fc="#fbfbfd", ec="#c4cad6", lw=1.1))
    ax.add_patch(Rectangle((0.015, 0.04), 0.97, 0.46, fc="#f3f6fa", ec="#9fb0c9", lw=1.1, ls="--"))
    ax.text(0.04, 0.59, "Cytosol", fontsize=11, color="#657083", fontstyle="italic")
    ax.text(0.04, 0.065, "Peroxisome", fontsize=11, color="#657083", fontstyle="italic")

    _box(ax, 0.035, 0.74, 0.20, 0.11, "AAA + Cys + Val\n(buffered pools)", COL["redbox"], ec="#b75b43", fs=9)
    _box(ax, 0.33, 0.74, 0.16, 0.11, "ACV", COL["bluebox"], fs=11)
    _box(ax, 0.65, 0.74, 0.15, 0.11, "IPN$_{cyt}$", COL["bluebox"], fs=11)
    _box(ax, 0.65, 0.36, 0.15, 0.10, "IPN$_{perox}$", COL["bluebox"], fs=10.5)
    _box(ax, 0.07, 0.25, 0.15, 0.09, "PAA", COL["redbox"], ec="#b75b43", fs=11)
    _box(ax, 0.07, 0.09, 0.18, 0.10, "PA-CoA", COL["bluebox"], fs=11)
    _box(ax, 0.34, 0.31, 0.15, 0.10, "6-APA", COL["purplebox"], ec="#7b4f8f", fs=11)
    _box(ax, 0.65, 0.14, 0.17, 0.10, "PenG$_{perox}$", COL["greenbox"], ec="#2f6b45", fs=10.5)
    _box(ax, 0.85, 0.14, 0.12, 0.10, "PenG$_{cyt}$", COL["greenbox"], ec="#2f6b45", fs=9.8)

    _arrow(ax, 0.235, 0.795, 0.33, 0.795, "ACVS", COL["ACVS"], lw=2.0)
    ax.text(0.272, 0.745, "pcbAB", fontsize=8, color=COL["ACVS"], fontstyle="italic")
    _arrow(ax, 0.49, 0.795, 0.65, 0.795, "IPNS", COL["IPNS"], lw=2.0)
    ax.text(0.555, 0.745, "pcbC", fontsize=8, color=COL["IPNS"], fontstyle="italic")
    _box(ax, 0.53, 0.62, 0.10, 0.075, "O$_2$", COL["redbox"], ec="#b75b43", fs=10)
    _arrow(ax, 0.58, 0.695, 0.58, 0.755, None, "#8a8f9c", lw=1.2)
    _arrow(ax, 0.725, 0.74, 0.725, 0.46, "IPN transport", COL["neutral"], lw=1.6, text_offset=0.026)

    _arrow(ax, 0.145, 0.25, 0.145, 0.19, "PCL", COL["PCL"], lw=2.0, text_offset=0.025)
    _arrow(ax, 0.65, 0.405, 0.49, 0.36, "IAH", COL["IAH"], lw=1.9, text_offset=0.025)
    _arrow(ax, 0.72, 0.36, 0.72, 0.24, "IAT", COL["IAT"], lw=2.0, text_offset=0.022)
    _arrow(ax, 0.25, 0.14, 0.65, 0.19, "PA-CoA co-substrate", COL["PCL"], lw=1.6, text_offset=0.020)
    _arrow(ax, 0.82, 0.19, 0.85, 0.19, "transport", COL["neutral"], lw=1.4, text_offset=0.025)
    _arrow(ax, 0.91, 0.14, 0.91, 0.055, "secretion", COL["neutral"], lw=1.4, text_offset=0.022)

    # Feedbacks
    fb = FancyArrowPatch((0.41, 0.855), (0.285, 0.835), connectionstyle="arc3,rad=-0.40",
                         arrowstyle="-[", mutation_scale=11, color=COL["ACVS"], lw=1.5, ls="--")
    ax.add_patch(fb)
    ax.text(0.285, 0.905, r"ACV feedback  $K_i=0.44$ mM", fontsize=8.5, color=COL["ACVS"])
    fb2 = FancyArrowPatch((0.17, 0.10), (0.16, 0.245), connectionstyle="arc3,rad=0.35",
                          arrowstyle="-[", mutation_scale=11, color=COL["ACVS"], lw=1.4, ls="--")
    ax.add_patch(fb2)
    ax.text(0.17, 0.215, "PA-CoA feedback", fontsize=8.1, color=COL["ACVS"])

    _box(ax, 0.31, 0.012, 0.31, 0.075,
         "penDE (Pc21g21370): IAT + IAH\nactivities are scaled jointly",
         COL["goldbox"], ec="#c59a18", fs=7.7, ls="--")


def panel_b(ax):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.set_title("B  Mathematical architecture", loc="left", fontweight="bold", fontsize=15, pad=8)

    _box(ax, 0.04, 0.83, 0.92, 0.11, r"$\frac{d\mathbf{x}}{dt}=\mathbf{S}\,\mathbf{v}(\mathbf{x};\theta)$",
         "#eef2f8", ec="#7d8fb0", fs=15)
    ax.text(0.50, 0.815, r"$\mathbf{S}$ from iAL1006  •  6 states  •  10 rate capacities",
            ha="center", va="top", fontsize=8.5, color="#4d5c79")

    ax.text(0.04, 0.75, "State vector x", fontsize=10, fontweight="bold")
    _box(ax, 0.04, 0.635, 0.92, 0.095,
         "$x_1$ ACV   $x_2$ IPN$_{cyt}$   $x_3$ IPN$_{perox}$\n"
         "$x_4$ PA-CoA$_{perox}$   $x_5$ PenG$_{perox}$   $x_6$ PenG$_{cyt}$",
         "#f8f9fb", ec="#c7ceda", fs=9.7)

    ax.text(0.04, 0.575, r"Rate laws $\mathbf{v}(\mathbf{x};\theta)$", fontsize=10, fontweight="bold")
    fs = 8.6
    y = 0.515
    lines = [
        (COL["ACVS"], r"$v_{ACVS}=V_{ACVS}\,\Pi(AAA,Cys,Val)\,\frac{1}{1+[ACV]/K_{i,ACV}}$"),
        (COL["IPNS"], r"$v_{IPNS}=V_{IPNS}\,\frac{[ACV]}{K_{ACV}+[ACV]}\,\frac{[O_2]}{K_{O_2}+[O_2]}$"),
        (COL["PCL"], r"$v_{PCL}=V_{PCL}[PAA]\,\frac{1}{1+[PA\!\!-\!CoA]/K_{i,PA-CoA}}$"),
        (COL["IAT"], r"$v_{IAT}=V_{IAT}\,\frac{[IPN_p]}{K_{IPN_p}+[IPN_p]}\,\frac{[PA\!\!-\!CoA]}{K_{PA-CoA}+[PA\!\!-\!CoA]}$"),
        (COL["IAH"], r"$v_{IAH}=V_{IAH}\,\frac{[IPN_p]}{K_{IAH}+[IPN_p]}$"),
    ]
    for color, txt in lines:
        ax.add_patch(Rectangle((0.055, y-0.018), 0.012, 0.042, fc=color, ec=color))
        ax.text(0.085, y, txt, fontsize=fs, va="center")
        y -= 0.075

    ax.text(0.055, 0.105,
            "Transport, secretion and PA-CoA hydrolysis are linear capacity terms.\n"
            "Buffered precursor and O$_2$ pools enter as fixed reference-state inputs.",
            fontsize=8.4, color="#505969")
    _box(ax, 0.04, 0.015, 0.92, 0.070,
         "GSE9825 transcript fold changes are applied only after ensemble admission:\n"
         "ACVS→Vmax_ACVS, IPNS→Vmax_IPNS, penDE→Vmax_IAT and Vmax_IAH jointly.",
         COL["goldbox"], ec="#c59a18", fs=8.3)


def panel_c(ax):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.set_title("C  Reference-ensemble → prediction/validation workflow", loc="left", fontweight="bold", fontsize=15, pad=8)

    # Left inputs
    _box(ax, 0.02, 0.69, 0.18, 0.12, "iAL1006\ntopology", "#e3ecf7", ec="#41618c", fs=9.5)
    _box(ax, 0.02, 0.48, 0.18, 0.15, "Literature-audited\nrate laws, constants\nand uncertainty ranges", "#e3ecf7", ec="#41618c", fs=8.8)
    _box(ax, 0.02, 0.15, 0.18, 0.14, "GSE9825\nhigh/low +PAA\ntranscript contrast", COL["redbox"], ec="#b5563c", fs=8.8)

    # Ensemble definition
    _box(ax, 0.29, 0.56, 0.22, 0.23,
         "Bounded parameter sampling\n\n8,000 candidates\n\nReference-state\nacceptance audit",
         "#eef1f8", ec="#5c6a8a", fs=9.0)
    _arrow(ax, 0.20, 0.75, 0.29, 0.69, color="#6b7488", lw=1.4)
    _arrow(ax, 0.20, 0.555, 0.29, 0.64, color="#6b7488", lw=1.4)

    # GSE applied after admission
    _box(ax, 0.29, 0.13, 0.22, 0.16,
         "Producer-state scaling\nAFTER ensemble admission\n\nNot an acceptance constraint",
         "#fff4f2", ec="#b5563c", fs=8.6, ls="--")
    _arrow(ax, 0.20, 0.22, 0.29, 0.21, color="#b5563c", lw=1.5)
    _arrow(ax, 0.40, 0.56, 0.40, 0.29, color="#b5563c", lw=1.1, ls="--")

    # Analyses
    ys = [0.77, 0.59, 0.41, 0.23]
    texts = [
        "MCA in low- and\nhigh-producer states",
        "Perturbation and\npairwise design scans",
        "Structural variants +\nbroad-prior robustness",
        "Independent external comparison\n(Weber, Theilgaard, Janoska, Nijland)",
    ]
    for yy, txt in zip(ys, texts):
        _box(ax, 0.61, yy-0.07, 0.24, 0.12, txt, "#eaf3ee", ec="#2f6b45", fs=8.6)
        _arrow(ax, 0.51, 0.675, 0.61, yy-0.01, color="#6b7488", lw=1.1)

    _box(ax, 0.61, 0.035, 0.24, 0.115,
         "Producer-ordering\nconsistency check\n(weak, non-independent)",
         "#fff4f2", ec="#b5563c", fs=8.5)
    _arrow(ax, 0.51, 0.21, 0.61, 0.095, color="#b5563c", lw=1.4)

    # Outcome
    _box(ax, 0.89, 0.30, 0.10, 0.38,
         "Ranked,\nuncertainty-\nqualified\nengineering\nhypotheses\n\n+\n\nexplicit\narchitecture-\ndependence",
         "#f3e8f5", ec="#6b3f73", fs=8.5)
    for yy in [0.76, 0.58, 0.40, 0.22, 0.09]:
        _arrow(ax, 0.85, yy, 0.89, 0.49, color="#6b7488", lw=1.0)


def render_figure1():
    fig = plt.figure(figsize=(12.4, 12.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 0.88], hspace=0.18, wspace=0.12,
                          left=0.035, right=0.985, top=0.975, bottom=0.035)
    axa = fig.add_subplot(gs[0,0]); axb = fig.add_subplot(gs[0,1]); axc = fig.add_subplot(gs[1,:])
    panel_a(axa); panel_b(axb); panel_c(axc)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGDIR, f"Figure1.{ext}"), dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(os.path.join(FIGDIR, "Figure1.png"))


def main():
    render_figure1()


if __name__ == "__main__":
    main()
