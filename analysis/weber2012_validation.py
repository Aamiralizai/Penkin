#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from penkin.paths import RESULTS_DIR, FIGURES_DIR

from penkin.data_io import load_expression
from penkin.ensemble import fit_ensemble
from penkin.model import scale_expression, steady_state

OUT_JSON = os.path.join(RESULTS_DIR, "weber2012_validation.json")
OUT_CSV = os.path.join(RESULTS_DIR, "weber2012_validation.csv")
OUT_PNG = os.path.join(FIGURES_DIR, "FigureS_Weber2012_validation.png")
OUT_SVG = os.path.join(FIGURES_DIR, "FigureS_Weber2012_validation.svg")
DIGITIZED_CSV = os.path.join(ROOT, "penkin", "data", "weber2012_digitized.csv")
DIGITIZED_REL_CSV = os.path.join(RESULTS_DIR, "weber2012_digitized_relative.csv")
OUT_EXT_PNG = os.path.join(FIGURES_DIR, "FigureS_Weber2012_external_comparison.png")
OUT_EXT_SVG = os.path.join(FIGURES_DIR, "FigureS_Weber2012_external_comparison.svg")
DOI = "10.1128/AEM.01529-12"

PENDE_FOLDS = [0.5, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0]
PCL_FOLDS = [1.0, 3.0, 10.0, 40.0]


def summ(x):
    a = np.asarray(x, dtype=float)
    return {
        "n": int(np.sum(np.isfinite(a))),
        "median": float(np.nanmedian(a)),
        "q05": float(np.nanpercentile(a, 5)),
        "q25": float(np.nanpercentile(a, 25)),
        "q75": float(np.nanpercentile(a, 75)),
        "q95": float(np.nanpercentile(a, 95)),
    }


def _ratio(value, baseline):
    if baseline is None or not np.isfinite(baseline) or baseline <= 0:
        return np.nan
    return value / baseline


def run_scan(n_try=800, n_sub=200, seed=23):
    accepted = fit_ensemble(n_try=n_try, verbose=False)
    expr, _ = load_expression()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(accepted), min(n_sub, len(accepted)), replace=False)

    pen = {
        str(x): {"penicillin": [], "IAT_flux": [], "IAH_flux": [], "IPNp": [], "PAACoAp": []}
        for x in PENDE_FOLDS
    }
    pcl = {str(x): {"penicillin": [], "PAACoAp": []} for x in PCL_FOLDS}

    baseline_models = 0
    for i in idx:
        r = accepted[i]
        ph = scale_expression(r["p"], expr, couple_pende=True)
        yh, fh = steady_state(5.0, ph, r["y_lo"])
        if yh is None or fh["secr"] <= 0:
            continue

        baseline_models += 1
        J0 = fh["secr"]
        IAT0 = fh["IAT"]
        IAH0 = fh["IAH"]
        IPNp0 = yh[2]
        PAACoAp0 = yh[3]

        for x in PENDE_FOLDS:
            q = copy.deepcopy(ph)
            q["Vmax_IAT"] *= x
            q["Vmax_IAH"] *= x
            y, f = steady_state(5.0, q, yh)
            if y is None or f["secr"] <= 0:
                continue
            d = pen[str(x)]
            d["penicillin"].append(f["secr"] / J0)
            d["IAT_flux"].append(_ratio(f["IAT"], IAT0))
            d["IAH_flux"].append(_ratio(f["IAH"], IAH0))
            d["IPNp"].append(_ratio(y[2], IPNp0))
            d["PAACoAp"].append(_ratio(y[3], PAACoAp0))

        for x in PCL_FOLDS:
            q = copy.deepcopy(ph)
            q["Vmax_PCL"] *= x
            y, f = steady_state(5.0, q, yh)
            if y is None or f["secr"] <= 0:
                continue
            d = pcl[str(x)]
            d["penicillin"].append(f["secr"] / J0)
            d["PAACoAp"].append(_ratio(y[3], PAACoAp0))

    pen_summary = {k: {m: summ(v) for m, v in d.items() if v} for k, d in pen.items()}
    pcl_summary = {k: {m: summ(v) for m, v in d.items() if v} for k, d in pcl.items()}

    moderate = [pen_summary[str(x)]["penicillin"]["median"] for x in [1.25, 1.5, 2.0, 3.0, 4.0]]
    high = [pen_summary[str(x)]["penicillin"]["median"] for x in [20.0, 30.0, 40.0]]
    pre_high = [pen_summary[str(x)]["penicillin"]["median"] for x in [3.0, 4.0, 5.0, 10.0]]
    high_iah = [pen_summary[str(x)]["IAH_flux"]["median"] for x in [20.0, 30.0, 40.0]]

    checks = {
        "penDE_direction_at_moderate_scaling": bool(max(moderate) > 1.0),
        "penDE_median_reaches_reported_1p6_to_2p0_range": bool(any(1.6 <= x <= 2.0 for x in moderate)),
        "PCL_response_is_marginal_at_3x_to_10x": bool(
            pcl_summary["3.0"]["penicillin"]["median"] <= 1.20
            and pcl_summary["10.0"]["penicillin"]["median"] <= 1.20
        ),
        "high_penDE_decline_reproduced": bool(min(high) < 0.95 * max(pre_high)),
        "high_penDE_IAH_flux_reaches_4x": bool(max(high_iah) >= 4.0),
    }

    out = {
        "source": {"doi": DOI, "study": "Weber et al. 2012"},
        "reported_experimental_results": {
            "penDE_high_producer_moderate_overexpression": "penicillin V 160-200% of parental level",
            "penDE_very_high_copy": "penicillin V response decreased at >20 additional copies",
            "PCL_overexpression": "slight/marginal increase",
            "high_penDE_6APA": "4-5-fold increase",
        },
        "model_state": "GSE9825 high-producer expression scaling",
        "penDE_coupling": "Vmax_IAT and Vmax_IAH scaled together",
        "activity_scaling_note": "Capacity-fold scans are not treated as direct gene-copy-number equivalents.",
        "six_state_model_note": "6-APA is not an explicit state; IAH branch flux is reported as the model-side proxy for 6-APA formation.",
        "digitized_experimental_data": {
            "file": "penkin/data/weber2012_digitized.csv",
            "figures": ["3A", "3B", "6"],
            "note": "Approximate point estimates digitized from the published figures; not raw experimental measurements."
        },
        "penDE_activity_scan": pen_summary,
        "PCL_activity_scan": pcl_summary,
        "comparison_checks": checks,
        "n_models_requested": int(min(n_sub, len(accepted))),
        "n_models_with_valid_high_state": int(baseline_models),
        "n_candidates": int(n_try),
        "n_accepted": int(len(accepted)),
        "seed": int(seed),
    }
    return out


def write_csv(out):
    rows = []
    for fold, metrics in out["penDE_activity_scan"].items():
        for metric, stats in metrics.items():
            rows.append(["penDE", fold, metric, stats["n"], stats["median"], stats["q05"], stats["q25"], stats["q75"], stats["q95"]])
    for fold, metrics in out["PCL_activity_scan"].items():
        for metric, stats in metrics.items():
            rows.append(["PCL", fold, metric, stats["n"], stats["median"], stats["q05"], stats["q25"], stats["q75"], stats["q95"]])
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "activity_fold", "metric", "n", "median", "q05", "q25", "q75", "q95"])
        w.writerows(rows)


def write_figure(out):
    folds = np.asarray(PENDE_FOLDS, dtype=float)
    pen_med = np.asarray([out["penDE_activity_scan"][str(x)]["penicillin"]["median"] for x in PENDE_FOLDS])
    pen_lo = np.asarray([out["penDE_activity_scan"][str(x)]["penicillin"]["q25"] for x in PENDE_FOLDS])
    pen_hi = np.asarray([out["penDE_activity_scan"][str(x)]["penicillin"]["q75"] for x in PENDE_FOLDS])
    iah_med = np.asarray([out["penDE_activity_scan"][str(x)]["IAH_flux"]["median"] for x in PENDE_FOLDS])
    iah_lo = np.asarray([out["penDE_activity_scan"][str(x)]["IAH_flux"]["q25"] for x in PENDE_FOLDS])
    iah_hi = np.asarray([out["penDE_activity_scan"][str(x)]["IAH_flux"]["q75"] for x in PENDE_FOLDS])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    ax = axes[0]
    ax.plot(folds, pen_med, marker="o", linewidth=1.5)
    ax.fill_between(folds, pen_lo, pen_hi, alpha=0.2)
    ax.axhline(1.0, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Coupled penDE activity scaling")
    ax.set_ylabel("Relative penicillin secretion")
    ax.set_title("A")

    ax = axes[1]
    ax.plot(folds, iah_med, marker="o", linewidth=1.5)
    ax.fill_between(folds, iah_lo, iah_hi, alpha=0.2)
    ax.axhline(1.0, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Coupled penDE activity scaling")
    ax.set_ylabel("Relative IAH branch flux")
    ax.set_title("B")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)



def load_digitized():
    rows = []
    with open(DIGITIZED_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            row["figure"] = int(row["figure"])
            row["gene_copy_min"] = float(row["gene_copy_min"])
            row["gene_copy_max"] = float(row["gene_copy_max"])
            row["gene_copy_mid"] = float(row["gene_copy_mid"])
            row["is_parental"] = int(row["is_parental"])
            row["value"] = float(row["value"])
            rows.append(row)
    return rows


def digitized_relative(rows):
    base = {}
    for row in rows:
        if row["is_parental"]:
            key = (row["figure"], row["panel"], row["target"], row["strain"], row["metric"])
            base[key] = row["value"]
    out = []
    for row in rows:
        key = (row["figure"], row["panel"], row["target"], row["strain"], row["metric"])
        q = dict(row)
        q["relative_to_parent"] = row["value"] / base[key] if base.get(key, 0) > 0 else np.nan
        out.append(q)
    return out



def summarize_digitized(rows):
    def subset(figure, panel, target, strain, metric):
        return [r for r in rows if r["figure"] == figure and r["panel"] == panel and r["target"] == target and r["strain"] == strain and r["metric"] == metric]

    out = {}
    for strain in ["DS47273", "DS17690"]:
        z = subset(3, "A", "penDE", strain, "penicillin_V")
        nonparent = [r for r in z if not r["is_parental"]]
        out[f"{strain}_penDE_peak_relative_to_parent"] = float(max(r["relative_to_parent"] for r in nonparent))
        high = max(nonparent, key=lambda r: r["gene_copy_mid"])
        out[f"{strain}_penDE_high_copy_relative_to_parent"] = float(high["relative_to_parent"])

        z = subset(6, "", "penDE", strain, "6_APA")
        high = max([r for r in z if not r["is_parental"]], key=lambda r: r["gene_copy_mid"])
        out[f"{strain}_6APA_high_copy_relative_to_parent"] = float(high["relative_to_parent"])

        z = subset(3, "B", "phl", strain, "penicillin_V")
        nonparent = [r for r in z if not r["is_parental"]]
        out[f"{strain}_phl_max_relative_to_parent"] = float(max(r["relative_to_parent"] for r in nonparent))
    return out

def write_digitized_relative(rows):
    fields = [
        "figure", "panel", "target", "strain", "cluster_copies",
        "gene_copy_min", "gene_copy_max", "gene_copy_mid", "is_parental",
        "metric", "value", "unit", "relative_to_parent", "source_note",
    ]
    with open(DIGITIZED_REL_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_external_figure(out, rows):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))

    ax = axes[0, 0]
    exp_pen = [r for r in rows if r["figure"] == 3 and r["panel"] == "A"]
    for strain in ["DS47274", "DS47273", "DS17690"]:
        z = sorted([r for r in exp_pen if r["strain"] == strain], key=lambda r: r["gene_copy_mid"])
        ax.plot([r["gene_copy_mid"] for r in z], [r["value"] for r in z], marker="o", label=strain)
    ax.set_xlabel("penDE copy number")
    ax.set_ylabel("Penicillin V (%)")
    ax.set_title("A  Weber et al. Fig. 3A")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    folds = np.asarray(PENDE_FOLDS, dtype=float)
    med = np.asarray([out["penDE_activity_scan"][str(x)]["penicillin"]["median"] for x in PENDE_FOLDS])
    lo = np.asarray([out["penDE_activity_scan"][str(x)]["penicillin"]["q25"] for x in PENDE_FOLDS])
    hi = np.asarray([out["penDE_activity_scan"][str(x)]["penicillin"]["q75"] for x in PENDE_FOLDS])
    ax.plot(folds, med, marker="o")
    ax.fill_between(folds, lo, hi, alpha=0.2)
    ax.axhline(1.0, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Coupled penDE activity scaling")
    ax.set_ylabel("Relative penicillin secretion")
    ax.set_title("B  Penkin")

    ax = axes[1, 0]
    exp_apa = [r for r in rows if r["figure"] == 6]
    for strain in ["DS47274", "DS47273", "DS17690"]:
        z = sorted([r for r in exp_apa if r["strain"] == strain], key=lambda r: r["gene_copy_mid"])
        ax.plot([r["gene_copy_mid"] for r in z], [r["value"] for r in z], marker="o", label=strain)
    ax.set_xlabel("penDE copy number")
    ax.set_ylabel("6-APA (µM)")
    ax.set_title("C  Weber et al. Fig. 6")

    ax = axes[1, 1]
    med = np.asarray([out["penDE_activity_scan"][str(x)]["IAH_flux"]["median"] for x in PENDE_FOLDS])
    lo = np.asarray([out["penDE_activity_scan"][str(x)]["IAH_flux"]["q25"] for x in PENDE_FOLDS])
    hi = np.asarray([out["penDE_activity_scan"][str(x)]["IAH_flux"]["q75"] for x in PENDE_FOLDS])
    ax.plot(folds, med, marker="o")
    ax.fill_between(folds, lo, hi, alpha=0.2)
    ax.axhline(1.0, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Coupled penDE activity scaling")
    ax.set_ylabel("Relative IAH branch flux")
    ax.set_title("D  Penkin")

    fig.tight_layout()
    fig.savefig(OUT_EXT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_EXT_SVG, bbox_inches="tight")
    plt.close(fig)

def run_validation(n_try=800, n_sub=200, seed=23):
    out = run_scan(n_try=n_try, n_sub=n_sub, seed=seed)
    digitized = digitized_relative(load_digitized())
    out["digitized_experimental_summary"] = summarize_digitized(digitized)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    write_csv(out)
    write_figure(out)
    write_digitized_relative(digitized)
    write_external_figure(out, digitized)
    print(OUT_JSON)
    print(OUT_CSV)
    print(OUT_PNG)
    print(OUT_SVG)
    print(DIGITIZED_REL_CSV)
    print(OUT_EXT_PNG)
    print(OUT_EXT_SVG)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-try", type=int, default=800)
    parser.add_argument("--n-sub", type=int, default=200)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    run_validation(args.n_try, args.n_sub, args.seed)


if __name__ == "__main__":
    main()
