#!/usr/bin/env python3
"""Figure 1: biological architecture, mathematical architecture and the
constraint-then-predict workflow.

The rate laws and mass balances shown in panel b are checked numerically
against penkin.model.fluxes and penkin.model.rhs at 200 random states before
the figure is drawn (verify_displayed_equations), so the figure cannot drift
from the implemented model.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from penkin.paths import RESULTS_DIR, FIGURES_DIR
from penkin import plotstyle
plotstyle.apply()
from penkin.model import DEFAULT_PARAMS, POOLS, fluxes, rhs

COL = {
    "ACVS": "#c0392b", "IPNS": "#1f77b4", "PCL": "#2e8b6f", "IAT": "#e07b39",
    "IAH": "#7b4f8f", "grey": "#5a6270",
    "build_fc": "#e6edf7", "build_ec": "#41618c",
    "held_fc": "#fff1ec", "held_ec": "#b5563c",
    "ana_fc": "#eaf3ee", "ana_ec": "#2f6b45",
    "out_fc": "#f3e8f5", "out_ec": "#6b3f73",
    "met_fc": "#e9eef6", "met_ec": "#46516a",
    "pool_fc": "#fff1ec", "pool_ec": "#b75b43",
    "peng_fc": "#eaf3ee", "peng_ec": "#2f6b45",
    "gold_fc": "#fff8e6", "gold_ec": "#c59a18",
}


# --------------------------------------------------------------------------
# Numerical check of the displayed equations
# --------------------------------------------------------------------------
def displayed_fluxes(y, PAA, p, pools=POOLS):
    """The rate laws exactly as typeset in panel b."""
    ACV, IPNc, IPNp, PACoA, PENGp, PENGc = y
    Pi = (pools["AAA"] / (p["Km_AAA"] + pools["AAA"])
          * pools["CYS"] / (p["Km_CYS"] + pools["CYS"])
          * pools["VAL"] / (p["Km_VAL"] + pools["VAL"]))
    return {
        "ACVS": p["Vmax_ACVS"] * Pi / (1 + ACV / p["Ki_ACV_ACVS"]),
        "IPNS": p["Vmax_IPNS"] * ACV / (p["Km_ACV"] + ACV) * pools["O2"] / (p["Km_O2"] + pools["O2"]),
        "PCL": p["Vmax_PCL"] * PAA / (1 + PACoA / p["Ki_PAACoA_PCL"]),
        "IAT": p["Vmax_IAT"] * IPNp / (p["Km_IPNp"] + IPNp) * PACoA / (p["Km_PAACoA"] + PACoA),
        "IAH": p["Vmax_IAH"] * IPNp / (p["Km_IAH"] + IPNp),
        "tIPN": p["kt_IPN"] * (IPNc - IPNp),
        "IPNsecr": p["k_IPNsecr"] * IPNc,
        "tPENG": p["kt_PENG"] * (PENGp - PENGc),
        "secr": p["k_secr"] * PENGc,
        "PAAChyd": p["k_PAACoA_hyd"] * PACoA,
    }


def displayed_rhs(y, PAA, p):
    """The mass balances exactly as typeset in panel b."""
    v = displayed_fluxes(y, PAA, p)
    return [
        v["ACVS"] - v["IPNS"],
        v["IPNS"] - v["tIPN"] - v["IPNsecr"],
        v["tIPN"] - v["IAT"] - v["IAH"],
        v["PCL"] - v["IAT"] - v["PAAChyd"],
        v["IAT"] - v["tPENG"],
        v["tPENG"] - v["secr"],
    ]


def verify_displayed_equations(n=200, seed=0, rtol=1e-9):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        y = rng.uniform(0.0, 5.0, 6)
        PAA = rng.choice([0.0, 5.0, rng.uniform(0.0, 10.0)])
        p = {k: v * np.exp(rng.uniform(-1.0, 1.0)) for k, v in DEFAULT_PARAMS.items()}
        ref_v, shown_v = fluxes(y, PAA, p), displayed_fluxes(y, PAA, p)
        for k in shown_v:
            worst = max(worst, abs(ref_v[k] - shown_v[k]) / max(abs(ref_v[k]), 1e-300))
        ref_r, shown_r = np.asarray(rhs(0.0, y, PAA, p)), np.asarray(displayed_rhs(y, PAA, p))
        worst = max(worst, float(np.max(np.abs(ref_r - shown_r) / np.maximum(np.abs(ref_r), 1e-12))))
    if worst > rtol:
        raise AssertionError(f"Figure 1 equations differ from penkin.model (max rel. error {worst:.2e})")
    return worst


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc, ec, fs=7.5, ls="-", lw=1.1, weight="normal", color="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, color=color, linespacing=1.25)


def arrow(ax, p0, p1, color=COL["grey"], lw=1.2, ls="-", style="-|>", rad=0.0, ms=9):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def label(ax, x, y, text, color, fs=7, style="italic", weight="normal", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, fontstyle=style,
            fontweight=weight)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------
def panel_a(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("a  Biological architecture")

    ax.add_patch(Rectangle((0.01, 0.11), 0.80, 0.80, fc="#fafbfd", ec="#b9c1ce", lw=1.0))
    ax.text(0.03, 0.875, "Cytosol", fontsize=8.5, fontstyle="italic", color="#5d6778")
    ax.add_patch(Rectangle((0.04, 0.26), 0.72, 0.36, fc="#f1f5fa", ec="#8fa2bf", lw=1.0, ls="--"))
    ax.text(0.36, 0.285, "Peroxisome", fontsize=8.5, fontstyle="italic", color="#5d6778")
    ax.text(0.84, 0.93, "extracellular", fontsize=7.5, color="#5d6778")

    # metabolites
    box(ax, 0.03, 0.70, 0.17, 0.09, "AAA + Cys + Val\n(buffered pools)", COL["pool_fc"], COL["pool_ec"], fs=6.8)
    box(ax, 0.30, 0.71, 0.11, 0.07, "ACV", COL["met_fc"], COL["met_ec"], fs=8.5)
    box(ax, 0.55, 0.71, 0.13, 0.07, "IPN$_{cyt}$", COL["met_fc"], COL["met_ec"], fs=8.5)
    box(ax, 0.44, 0.645, 0.07, 0.045, "O$_2$", COL["pool_fc"], COL["pool_ec"], fs=7.5)
    box(ax, 0.55, 0.49, 0.13, 0.07, "IPN$_{perox}$", COL["met_fc"], COL["met_ec"], fs=8.5)
    box(ax, 0.30, 0.49, 0.11, 0.07, "6-APA", "#f4edf7", COL["IAH"], fs=8.5)
    box(ax, 0.07, 0.49, 0.10, 0.06, "PAA", COL["pool_fc"], COL["pool_ec"], fs=8.5)
    box(ax, 0.07, 0.33, 0.13, 0.07, "PA-CoA", COL["met_fc"], COL["met_ec"], fs=8.5)
    box(ax, 0.55, 0.31, 0.13, 0.07, "PenG$_{perox}$", COL["peng_fc"], COL["peng_ec"], fs=8.5)
    box(ax, 0.55, 0.15, 0.13, 0.07, "PenG$_{cyt}$", COL["peng_fc"], COL["peng_ec"], fs=8.5)

    # reactions
    arrow(ax, (0.20, 0.745), (0.30, 0.745), COL["ACVS"], lw=1.6)
    label(ax, 0.25, 0.722, "ACVS", COL["ACVS"], weight="bold"); label(ax, 0.25, 0.695, "pcbAB", COL["ACVS"], fs=6)
    arrow(ax, (0.41, 0.745), (0.55, 0.745), COL["IPNS"], lw=1.6)
    label(ax, 0.48, 0.775, "IPNS", COL["IPNS"], weight="bold"); label(ax, 0.50, 0.715, "pcbC", COL["IPNS"], fs=6, ha="left")
    arrow(ax, (0.475, 0.69), (0.475, 0.742), "#8a8f9c", lw=0.9)
    arrow(ax, (0.615, 0.71), (0.615, 0.56), COL["grey"], lw=1.2)
    label(ax, 0.625, 0.635, "IPN transport", COL["grey"], fs=6.5, ha="left")
    arrow(ax, (0.68, 0.76), (0.86, 0.90), COL["grey"], lw=1.0)
    label(ax, 0.80, 0.80, "IPN export", COL["grey"], fs=6.5)
    arrow(ax, (0.55, 0.525), (0.41, 0.525), COL["IAH"], lw=1.4)
    label(ax, 0.48, 0.55, "IAH", COL["IAH"], weight="bold")
    arrow(ax, (0.615, 0.49), (0.615, 0.38), COL["IAT"], lw=1.6)
    label(ax, 0.64, 0.435, "IAT", COL["IAT"], weight="bold", ha="left")
    arrow(ax, (0.12, 0.49), (0.12, 0.40), COL["PCL"], lw=1.6)
    label(ax, 0.10, 0.445, "PCL", COL["PCL"], weight="bold", ha="right")
    arrow(ax, (0.20, 0.37), (0.60, 0.43), COL["PCL"], lw=1.1)
    label(ax, 0.39, 0.425, "PA-CoA co-substrate", COL["PCL"], fs=6.3)
    arrow(ax, (0.135, 0.33), (0.135, 0.285), COL["grey"], lw=0.9)
    label(ax, 0.15, 0.30, "hydrolysis", COL["grey"], fs=6, ha="left")
    arrow(ax, (0.615, 0.31), (0.615, 0.22), COL["grey"], lw=1.2)
    label(ax, 0.625, 0.25, "PenG transport", COL["grey"], fs=6.5, ha="left")
    arrow(ax, (0.68, 0.185), (0.90, 0.185), COL["grey"], lw=1.2)
    label(ax, 0.86, 0.215, "PenG\nsecretion", COL["grey"], fs=6.5)

    # feedback (dashed, flat-headed)
    arrow(ax, (0.355, 0.78), (0.255, 0.752), COL["ACVS"], lw=1.0, ls="--", style="-[", rad=0.7, ms=6)
    label(ax, 0.30, 0.83, "ACV feedback ($K_i$ = 0.44 mM)", COL["ACVS"], fs=6.5, style="normal")
    arrow(ax, (0.20, 0.395), (0.128, 0.455), COL["ACVS"], lw=1.0, ls="--", style="-[", rad=0.45, ms=6)
    label(ax, 0.205, 0.465, "PA-CoA feedback", COL["ACVS"], fs=6.3, style="normal", ha="left")

    box(ax, 0.21, 0.02, 0.44, 0.065, "penDE (Pc21g21370) encodes IAT and IAH;\nthe two capacities are scaled jointly",
        COL["gold_fc"], COL["gold_ec"], fs=6.5, ls="--")


def panel_b(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("b  Mathematical architecture"); ax.set_ylim(-0.012, 1)

    box(ax, 0.01, 0.885, 0.98, 0.095, "", "#eef2f8", "#7d8fb0")
    ax.text(0.50, 0.945, r"$d\mathbf{x}/dt=\mathbf{S}\cdot\mathbf{v}(\mathbf{x};\theta)$", ha="center", va="center", fontsize=10)
    ax.text(0.50, 0.905, "S: reduced six-state stoichiometry mapped onto iAL1006 enzymes r0814, r0812,\n"
            "r0747, r0813, r0803 and transporters r1271, r1281", ha="center", va="center", fontsize=5.6,
            color="#3f4a5e")

    ax.text(0.01, 0.855, r"State vector $\mathbf{x}$", fontsize=7.5, fontweight="bold", va="center")
    box(ax, 0.01, 0.785, 0.98, 0.05,
        "$x_1$ ACV   $x_2$ IPN$_{cyt}$   $x_3$ IPN$_{perox}$   $x_4$ PA-CoA$_{perox}$   $x_5$ PenG$_{perox}$   $x_6$ PenG$_{cyt}$",
        "#f8f9fb", "#c7ceda", fs=6.3)

    ax.text(0.01, 0.755, r"Rate laws $\mathbf{v}(\mathbf{x};\theta)$", fontsize=7.5, fontweight="bold", va="center")
    fs = 6.6
    rows = [
        (COL["ACVS"], r"$v_{ACVS}=V_{max,ACVS}\,\Pi\cdot\dfrac{1}{1+[ACV]/K_{i,ACV}}$"),
        (COL["IPNS"], r"$v_{IPNS}=V_{max,IPNS}\,\dfrac{[ACV]}{K_{m,ACV}+[ACV]}\cdot\dfrac{[O_2]}{K_{m,O_2}+[O_2]}$"),
        (COL["PCL"], r"$v_{PCL}=V_{max,PCL}\,[PAA]\cdot\dfrac{1}{1+[PACoA_p]/K_{i,PACoA}}$"),
        (COL["IAT"], r"$v_{IAT}=V_{max,IAT}\,\dfrac{[IPN_p]}{K_{m,IPN}+[IPN_p]}\cdot\dfrac{[PACoA_p]}{K_{m,PACoA}+[PACoA_p]}$"),
        (COL["IAH"], r"$v_{IAH}=V_{max,IAH}\,\dfrac{[IPN_p]}{K_{m,IAH}+[IPN_p]}$"),
    ]
    y = 0.705
    for color, txt in rows:
        ax.add_patch(Rectangle((0.015, y - 0.022), 0.012, 0.044, fc=color, ec=color))
        ax.text(0.04, y, txt, fontsize=fs, va="center")
        y -= 0.063
    ax.text(0.04, y + 0.01,
            r"where $\Pi=\dfrac{[AAA]}{K_{m,AAA}+[AAA]}\cdot\dfrac{[Cys]}{K_{m,Cys}+[Cys]}\cdot\dfrac{[Val]}{K_{m,Val}+[Val]}$",
            fontsize=fs, va="center")

    ax.text(0.01, 0.335, "Transport and first-order terms", fontsize=7.5, fontweight="bold", va="center")
    t = 6.2
    ax.text(0.04, 0.30, r"$v_{tIPN}=k_{tIPN}([IPN_c]-[IPN_p])$", fontsize=t, va="center")
    ax.text(0.53, 0.30, r"$v_{IPNexp}=k_{IPNexp}[IPN_c]$", fontsize=t, va="center")
    ax.text(0.04, 0.27, r"$v_{tPenG}=k_{tPenG}([PenG_p]-[PenG_c])$", fontsize=t, va="center")
    ax.text(0.53, 0.27, r"$v_{secr}=k_{secr}[PenG_c]$", fontsize=t, va="center")
    ax.text(0.04, 0.24, r"$v_{hyd}=k_{hyd}[PACoA_p]$", fontsize=t, va="center")
    ax.text(0.53, 0.24, "buffered: AAA, Cys, Val, O$_2$; PAA = 0 or 5 mM", fontsize=t, va="center")

    ax.text(0.01, 0.205, "Mass balances (rows of S)", fontsize=7.5, fontweight="bold", va="center")
    mb = [
        r"$d[ACV]/dt=v_{ACVS}-v_{IPNS}$",
        r"$d[IPN_c]/dt=v_{IPNS}-v_{tIPN}-v_{IPNexp}$",
        r"$d[IPN_p]/dt=v_{tIPN}-v_{IAT}-v_{IAH}$",
        r"$d[PACoA_p]/dt=v_{PCL}-v_{IAT}-v_{hyd}$",
        r"$d[PenG_p]/dt=v_{IAT}-v_{tPenG}$",
        r"$d[PenG_c]/dt=v_{tPenG}-v_{secr}$",
    ]
    for i, txt in enumerate(mb):
        ax.text(0.04 + 0.49 * (i % 2), 0.172 - 0.031 * (i // 2), txt, fontsize=t, va="center")

    box(ax, 0.01, 0.0, 0.98, 0.065,
        "GSE9825 fold changes scale ACVS, IPNS and (jointly) IAT/IAH capacities;\n"
        "applied only after ensemble admission, never as an acceptance constraint",
        COL["gold_fc"], COL["gold_ec"], fs=6.3, ls="--")


def panel_c(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("c  Constraint-then-predict workflow")

    # column headers
    for x, txt, c in [(0.105, "Constrains the ensemble", COL["build_ec"]),
                      (0.40, "Ensemble", COL["met_ec"]),
                      (0.66, "Analyses", COL["ana_ec"]),
                      (0.915, "Output", COL["out_ec"])]:
        ax.text(x, 0.955, txt, ha="center", fontsize=7, fontweight="bold", color=c)

    # inputs that build/constrain the ensemble
    box(ax, 0.01, 0.72, 0.19, 0.17, "iAL1006\npathway topology\nand stoichiometry", COL["build_fc"], COL["build_ec"], fs=6.8)
    box(ax, 0.01, 0.49, 0.19, 0.19, "Literature-audited rate\nlaws, kinetic constants\nand uncertainty ranges",
        COL["build_fc"], COL["build_ec"], fs=6.8)

    # ensemble
    box(ax, 0.28, 0.52, 0.24, 0.34,
        "Bounded log-uniform sampling\n\n8,000 candidates;\nall admissible under the\nreference-state criteria\n(steady state, secretion,\nconcentration bounds)",
        "#eef1f8", COL["met_ec"], fs=6.6)
    arrow(ax, (0.20, 0.805), (0.28, 0.75))
    arrow(ax, (0.20, 0.585), (0.28, 0.63))

    # withheld data
    ax.text(0.105, 0.405, "Withheld from admission", ha="center", fontsize=7, fontweight="bold", color=COL["held_ec"])
    box(ax, 0.01, 0.22, 0.19, 0.16, "GSE9825 high/low\n+PAA transcript\ncontrast", COL["held_fc"], COL["held_ec"], fs=6.8, ls="--")
    box(ax, 0.01, 0.02, 0.19, 0.16, "External perturbation data\n(Weber, Theilgaard,\nJanoska, Nijland)",
        COL["held_fc"], COL["held_ec"], fs=6.4, ls="--")

    box(ax, 0.28, 0.21, 0.24, 0.18,
        "Producer-state scaling\napplied after admission;\nnot an acceptance constraint", COL["held_fc"], COL["held_ec"], fs=6.6, ls="--")
    arrow(ax, (0.20, 0.30), (0.28, 0.30), COL["held_ec"])
    arrow(ax, (0.40, 0.52), (0.40, 0.39), COL["held_ec"], ls="--")

    # analyses
    ys = [0.79, 0.62, 0.45]
    texts = ["Flux-control analysis at\nlow- and high-producer states",
             "Perturbation and pairwise\ndesign scans",
             "Structural variants and\nbroad-prior stress test"]
    for yy, txt in zip(ys, texts):
        box(ax, 0.57, yy - 0.065, 0.19, 0.13, txt, COL["ana_fc"], COL["ana_ec"], fs=6.4)
        arrow(ax, (0.52, 0.69), (0.57, yy))
    box(ax, 0.57, 0.215, 0.19, 0.13, "Producer-ordering\nconsistency check\n(weak; not independent)",
        COL["held_fc"], COL["held_ec"], fs=6.4)
    arrow(ax, (0.52, 0.30), (0.57, 0.28), COL["held_ec"])
    box(ax, 0.57, 0.025, 0.19, 0.13, "External comparison\n(data not used in\nmodel construction)",
        COL["ana_fc"], COL["ana_ec"], fs=6.4)
    arrow(ax, (0.20, 0.10), (0.57, 0.09), COL["held_ec"])
    arrow(ax, (0.52, 0.60), (0.57, 0.10))

    # output
    box(ax, 0.83, 0.20, 0.16, 0.62,
        "Ranked,\nuncertainty-\nqualified\nengineering\nhypotheses\n\n+\n\nexplicit\narchitecture-\ndependence",
        COL["out_fc"], COL["out_ec"], fs=6.6)
    for yy in [0.79, 0.62, 0.45, 0.28, 0.09]:
        arrow(ax, (0.76, yy), (0.83, 0.51 + (yy - 0.45) * 0.35), lw=0.9)


def render_figure1():
    worst = verify_displayed_equations()
    fig = plt.figure(figsize=(plotstyle.WIDTH, 10.6), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.22, 0.78], hspace=0.10, wspace=0.05,
                          left=0.01, right=0.99, top=0.975, bottom=0.01)
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))
    panel_c(fig.add_subplot(gs[1, :]))
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGURES_DIR, f"Figure1.{ext}"), dpi=600 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(os.path.join(FIGURES_DIR, "Figure1.png"), f"(equation check: max relative error {worst:.1e})")


def main():
    render_figure1()


if __name__ == "__main__":
    main()
