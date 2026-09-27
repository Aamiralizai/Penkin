#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from penkin.paths import RESULTS_DIR, FIGURES_DIR

from penkin.data_io import load_expression
from penkin.ensemble import fit_ensemble, predict_producer_ratio
from penkin.model import POOLS, scale_expression
from penkin.variants import scale_cluster, steady_variant

DATA = os.path.join(ROOT, "penkin", "data")
RESULTS = RESULTS_DIR
FIGURES = FIGURES_DIR
OUT_JSON = os.path.join(RESULTS, "external_validation_suite.json")
OUT_CSV = os.path.join(RESULTS, "external_validation_summary.csv")
OUT_PNG = os.path.join(FIGURES, "FigureS_external_validation_suite.png")
OUT_SVG = os.path.join(FIGURES, "FigureS_external_validation_suite.svg")


def read_csv(name):
    with open(os.path.join(DATA, name), newline="") as fh:
        return list(csv.DictReader(fh))


def summary(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None, "q05": None, "q95": None}
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
        "q05": float(np.percentile(a, 5)),
        "q95": float(np.percentile(a, 95)),
    }


def sample_models(accepted, n_sub, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(accepted), min(n_sub, len(accepted)), replace=False)
    return [accepted[i] for i in idx]


def validate_gse9825(accepted, n_sub, seed):
    expr, reps = load_expression()
    pred, _ = predict_producer_ratio(accepted, expr, n_sub=n_sub, seed=seed, couple_pende=True)
    return {
        "source": "GSE9825",
        "expression_fold": {k: float(v) for k, v in expr.items()},
        "predicted_high_low_ratio": summary(pred["ratios"]),
        "fraction_high_gt_low": float(pred["frac_high_gt_low"]),
        "role": "expression input and producer-ordering consistency check",
    }


def validate_nijland(sub):
    rows = read_csv("nijland2010_digitized.csv")
    rows = [r for r in rows if int(float(r["copy_number"])) >= 1]
    copies = [int(float(r["copy_number"])) for r in rows]
    exp_pen = np.array([float(r["penicillin_V_mmol_gDW"]) for r in rows])
    exp_ipn = np.array([float(r["IPN_mg_gDW"]) for r in rows])
    exp_pen_rel = exp_pen / exp_pen[0]
    exp_ipn_rel = exp_ipn / exp_ipn[0]

    pred_copy = {c: [] for c in copies}
    pred_protein = {c: [] for c in copies}
    ipn_copy = {c: [] for c in copies}
    ipn_protein = {c: [] for c in copies}

    for r in sub:
        p = r["p"]
        y1, f1 = steady_variant(5.0, p, r["y_lo"], kind="baseline")
        if y1 is None or f1["secr"] <= 0 or y1[2] <= 0:
            continue
        J1, I1 = f1["secr"], y1[2]
        for row, c in zip(rows, copies):
            q = scale_cluster(p, acvs=c, ipns=c, pende=c, pcl=1.0)
            y, f = steady_variant(5.0, q, y1, kind="baseline")
            if y is not None and f["secr"] > 0:
                pred_copy[c].append(f["secr"] / J1)
                ipn_copy[c].append(y[2] / I1)

            acvs = float(row["ACVS_protein_pct"]) / float(rows[0]["ACVS_protein_pct"])
            ipns = float(row["IPNS_protein_pct"]) / float(rows[0]["IPNS_protein_pct"])
            pende = float(row["IAT_protein_pct"]) / float(rows[0]["IAT_protein_pct"])
            pcl = float(row["PCL_protein_pct"]) / float(rows[0]["PCL_protein_pct"])
            qp = scale_cluster(p, acvs=acvs, ipns=ipns, pende=pende, pcl=pcl)
            yp, fp = steady_variant(5.0, qp, y1, kind="baseline")
            if yp is not None and fp["secr"] > 0:
                pred_protein[c].append(fp["secr"] / J1)
                ipn_protein[c].append(yp[2] / I1)

    med_copy = np.array([summary(pred_copy[c])["median"] for c in copies], dtype=float)
    med_protein = np.array([summary(pred_protein[c])["median"] for c in copies], dtype=float)
    med_ipn_copy = np.array([summary(ipn_copy[c])["median"] for c in copies], dtype=float)
    med_ipn_protein = np.array([summary(ipn_protein[c])["median"] for c in copies], dtype=float)

    def sp(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            return None
        return float(spearmanr(a[ok], b[ok]).statistic)

    return {
        "source": "Nijland et al. 2010, DOI 10.1128/AEM.01702-10",
        "data_note": "Published-figure values are approximate digitizations, not raw measurements.",
        "copy_numbers": copies,
        "experimental_penicillin_relative_to_1copy": exp_pen_rel.tolist(),
        "experimental_IPN_relative_to_1copy": exp_ipn_rel.tolist(),
        "copy_proportional_prediction": {str(c): summary(pred_copy[c]) for c in copies},
        "protein_informed_prediction": {str(c): summary(pred_protein[c]) for c in copies},
        "copy_proportional_IPN": {str(c): summary(ipn_copy[c]) for c in copies},
        "protein_informed_IPN": {str(c): summary(ipn_protein[c]) for c in copies},
        "spearman_penicillin_copy_proportional": sp(exp_pen_rel, med_copy),
        "spearman_penicillin_protein_informed": sp(exp_pen_rel, med_protein),
        "spearman_IPN_copy_proportional": sp(exp_ipn_rel, med_ipn_copy),
        "spearman_IPN_protein_informed": sp(exp_ipn_rel, med_ipn_protein),
    }


def validate_janoska(sub, expr):
    rows = read_csv("janoska2022_oxygen_summary.csv")
    do_vals = [float(r["DO_mM"]) for r in rows]
    exp = np.array([float(r["relative_penicillin_rate"]) for r in rows])
    out = {}

    for kind in ["baseline", "legacy_feedforward", "reversible_IPNS"]:
        pred = {x: [] for x in do_vals}
        acv = {x: [] for x in do_vals}
        ipn = {x: [] for x in do_vals}
        valid = {x: 0 for x in do_vals}
        for r in sub:
            ph = scale_expression(r["p"], expr, couple_pende=True)
            pools_ref = dict(POOLS)
            pools_ref["O2"] = 0.136
            yref, fref = steady_variant(5.0, ph, r["y_lo"], pools=pools_ref, kind=kind)
            if yref is None or fref["secr"] <= 0 or yref[0] <= 0 or yref[2] <= 0:
                continue
            for x in do_vals:
                pools = dict(POOLS)
                pools["O2"] = x
                y, f = steady_variant(5.0, ph, yref, pools=pools, kind=kind)
                if y is None or f["secr"] < 0:
                    continue
                valid[x] += 1
                pred[x].append(f["secr"] / fref["secr"])
                acv[x].append(y[0] / yref[0])
                ipn[x].append(y[2] / yref[2])

        med = np.array([summary(pred[x])["median"] if summary(pred[x])["median"] is not None else np.nan for x in do_vals])
        ok = np.isfinite(med)
        rmse = float(np.sqrt(np.mean((med[ok] - exp[ok]) ** 2))) if ok.any() else None
        out[kind] = {
            "penicillin_relative": {str(x): summary(pred[x]) for x in do_vals},
            "ACV_relative": {str(x): summary(acv[x]) for x in do_vals},
            "IPNp_relative": {str(x): summary(ipn[x]) for x in do_vals},
            "valid_models": {str(x): int(valid[x]) for x in do_vals},
            "rmse_vs_approx_published_endpoints": rmse,
            "ACV_increase_fraction_at_0.013": float(np.mean(np.asarray(acv[0.013]) > 1)) if acv[0.013] else None,
            "IPN_decrease_fraction_at_0.013": float(np.mean(np.asarray(ipn[0.013]) < 1)) if ipn[0.013] else None,
        }

    return {
        "source": "Janoska et al. 2022/2023, DOI 10.1002/elsc.202100139",
        "data_note": "Penicillin-rate endpoints are approximate readings from the published trajectory; metabolite directions are text-supported.",
        "experimental": {str(x): float(y) for x, y in zip(do_vals, exp)},
        "architectures": out,
    }


def validate_theilgaard(sub):
    scenarios = {
        "ACVS_2x": (2, 1, 1),
        "ACVS_3x": (3, 1, 1),
        "IPNS_2x": (1, 2, 1),
        "IPNS_3x": (1, 3, 1),
        "pcbC_penDE_2x": (1, 2, 2),
        "pcbC_penDE_3x": (1, 3, 3),
        "whole_cluster_2x": (2, 2, 2),
        "whole_cluster_3x": (3, 3, 3),
    }
    vals = {k: [] for k in scenarios}
    for r in sub:
        y0, f0 = steady_variant(5.0, r["p"], r["y_lo"], kind="baseline")
        if y0 is None or f0["secr"] <= 0:
            continue
        for name, (a, i, d) in scenarios.items():
            q = scale_cluster(r["p"], acvs=a, ipns=i, pende=d)
            y, f = steady_variant(5.0, q, y0, kind="baseline")
            if y is not None and f["secr"] > 0:
                vals[name].append(f["secr"] / f0["secr"])
    s = {k: summary(v) for k, v in vals.items()}
    whole_obs = [2.24, 2.76]
    return {
        "source": "Theilgaard et al. 2001, DOI 10.1002/1097-0290(20000220)72:4<379::AID-BIT1000>3.0.CO;2-5",
        "experimental": {
            "whole_cluster_transformants": whole_obs,
            "pcbC_penDE_transformant": 0.91,
        },
        "model_scenarios": s,
        "mapping_note": "The paper abstract does not provide a direct enzyme-capacity fold for each transformant; 2x and 3x capacity scenarios are brackets, not point-matched reconstructions.",
        "whole_cluster_2x_in_experimental_range": bool(2.24 <= s["whole_cluster_2x"]["median"] <= 2.76),
        "whole_cluster_3x_in_experimental_range": bool(2.24 <= s["whole_cluster_3x"]["median"] <= 2.76),
        "pcbC_penDE_direction_matches_reported_decrease_2x": bool(s["pcbC_penDE_2x"]["median"] < 1.0),
        "pcbC_penDE_direction_matches_reported_decrease_3x": bool(s["pcbC_penDE_3x"]["median"] < 1.0),
    }


def validate_degeneration(sub, expr):
    scenarios = {
        "transcript_proxy": {"ACVS": 0.5, "IPNS": 0.85, "IAT": 0.85, "kt_PENG": 1.0},
        "protein_moderate": {"ACVS": 1/3, "IPNS": 1/5, "IAT": 0.8, "kt_PENG": 1.0},
        "protein_severe": {"ACVS": 1/3, "IPNS": 1/20, "IAT": 0.8, "kt_PENG": 1.0},
        "protein_severe_plus_transport": {"ACVS": 1/3, "IPNS": 1/20, "IAT": 0.8, "kt_PENG": 0.2},
    }
    out = {}
    for kind in ["baseline", "legacy_feedforward", "reversible_IPNS"]:
        vals = {k: [] for k in scenarios}
        invalid = {k: 0 for k in scenarios}
        for r in sub:
            ph = scale_expression(r["p"], expr, couple_pende=True)
            y0, f0 = steady_variant(5.0, ph, r["y_lo"], kind=kind)
            if y0 is None or f0["secr"] <= 0:
                continue
            for name, sc in scenarios.items():
                q = copy.deepcopy(ph)
                q["Vmax_ACVS"] *= sc["ACVS"]
                q["Vmax_IPNS"] *= sc["IPNS"]
                q["Vmax_IAT"] *= sc["IAT"]
                q["Vmax_IAH"] *= sc["IAT"]
                q["kt_PENG"] *= sc["kt_PENG"]
                y, f = steady_variant(5.0, q, y0, kind=kind)
                if y is None or f["secr"] <= 0:
                    invalid[name] += 1
                    continue
                vals[name].append(f["secr"] / f0["secr"])
        out[kind] = {
            "scenarios": {k: summary(v) for k, v in vals.items()},
            "invalid_models": invalid,
            "severe_reproduces_gt10fold_loss": bool(summary(vals["protein_severe"])["median"] is not None and summary(vals["protein_severe"])["median"] < 0.1),
        }
    return {
        "source": "Douma et al. 2011 / GSE24212, DOI 10.1186/1752-0509-5-132",
        "experimental": {
            "penicillin_productivity": "more than 10-fold decrease",
            "ACVS_protein": "approximately 3-fold decrease",
            "IPNS_protein": "5- to 20-fold decrease",
            "AT_protein": "smaller decrease",
        },
        "architectures": out,
        "interpretation_note": "This is a protein-level stress test. The transcript-only scenario is included to test the transcript-to-capacity approximation.",
    }


def load_weber_result():
    path = os.path.join(RESULTS, "weber2012_validation.json")
    if not os.path.exists(path):
        return {"status": "not_run", "instruction": "Run python analysis/weber2012_validation.py --n-try 800 --n-sub 200 --seed 23"}
    with open(path) as fh:
        d = json.load(fh)
    return {
        "status": "available",
        "source": d.get("source"),
        "comparison_checks": d.get("comparison_checks"),
        "n_models_with_valid_high_state": d.get("n_models_with_valid_high_state"),
        "n_candidates": d.get("n_candidates"),
        "n_accepted": d.get("n_accepted"),
        "penDE_2x_median": d.get("penDE_activity_scan", {}).get("2.0", {}).get("penicillin", {}).get("median"),
        "penDE_4x_median": d.get("penDE_activity_scan", {}).get("4.0", {}).get("penicillin", {}).get("median"),
        "PCL_3x_median": d.get("PCL_activity_scan", {}).get("3.0", {}).get("penicillin", {}).get("median"),
        "PCL_10x_median": d.get("PCL_activity_scan", {}).get("10.0", {}).get("penicillin", {}).get("median"),
    }


def write_summary_csv(result):
    rows = []
    rows.append(["GSE9825", "high/low producer ratio", result["GSE9825"]["predicted_high_low_ratio"]["median"], "all high>low", result["GSE9825"]["fraction_high_gt_low"]])

    n = result["Nijland2010"]
    rows.append(["Nijland2010", "Spearman PenV copy-proportional", n["spearman_penicillin_copy_proportional"], "higher is better", ""])
    rows.append(["Nijland2010", "Spearman PenV protein-informed", n["spearman_penicillin_protein_informed"], "higher is better", ""])
    rows.append(["Nijland2010", "Spearman IPN protein-informed", n["spearman_IPN_protein_informed"], "higher is better", ""])

    j = result["Janoska2023"]
    for kind in j["architectures"]:
        rows.append(["Janoska2023", f"oxygen RMSE {kind}", j["architectures"][kind]["rmse_vs_approx_published_endpoints"], "lower is better", ""])

    t = result["Theilgaard2001"]
    rows.append(["Theilgaard2001", "whole cluster 2x median", t["model_scenarios"]["whole_cluster_2x"]["median"], "experimental 2.24-2.76", t["whole_cluster_2x_in_experimental_range"]])
    rows.append(["Theilgaard2001", "pcbC+penDE 2x median", t["model_scenarios"]["pcbC_penDE_2x"]["median"], "experimental 0.91", t["pcbC_penDE_direction_matches_reported_decrease_2x"]])

    d = result["Douma2011_GSE24212"]
    for kind in d["architectures"]:
        rows.append(["Douma2011", f"severe protein loss {kind}", d["architectures"][kind]["scenarios"]["protein_severe"]["median"], "experimental <0.1", d["architectures"][kind]["severe_reproduces_gt10fold_loss"]])

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "metric", "value", "reference", "match"])
        w.writerows(rows)


def write_figure(result):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    n = result["Nijland2010"]
    x = np.array(n["copy_numbers"], dtype=float)
    axes[0, 0].plot(x, n["experimental_penicillin_relative_to_1copy"], marker="o", label="experiment")
    axes[0, 0].plot(x, [n["copy_proportional_prediction"][str(int(c))]["median"] for c in x], marker="o", label="copy-scaled")
    axes[0, 0].plot(x, [n["protein_informed_prediction"][str(int(c))]["median"] for c in x], marker="o", label="protein-informed")
    axes[0, 0].set_xlabel("Penicillin cluster copies")
    axes[0, 0].set_ylabel("Relative penicillin output")
    axes[0, 0].set_title("a  Nijland 2010")
    axes[0, 0].legend(frameon=False, fontsize=8)

    j = result["Janoska2023"]
    ox = np.array(sorted(float(k) for k in j["experimental"]), dtype=float)
    axes[0, 1].plot(ox, [j["experimental"][str(v)] for v in ox], marker="o", label="experiment")
    for kind in ["baseline", "legacy_feedforward", "reversible_IPNS"]:
        axes[0, 1].plot(ox, [j["architectures"][kind]["penicillin_relative"][str(v)]["median"] for v in ox], marker="o", label=kind)
    axes[0, 1].set_xlabel("Dissolved O2 (mM)")
    axes[0, 1].set_ylabel("Relative penicillin rate")
    axes[0, 1].set_title("b  Janoska oxygen response")
    axes[0, 1].legend(frameon=False, fontsize=7)

    t = result["Theilgaard2001"]
    labels = ["ACVS 2x", "IPNS 2x", "pcbC+penDE 2x", "cluster 2x", "cluster 3x"]
    vals = [
        t["model_scenarios"]["ACVS_2x"]["median"],
        t["model_scenarios"]["IPNS_2x"]["median"],
        t["model_scenarios"]["pcbC_penDE_2x"]["median"],
        t["model_scenarios"]["whole_cluster_2x"]["median"],
        t["model_scenarios"]["whole_cluster_3x"]["median"],
    ]
    axes[1, 0].bar(np.arange(len(labels)), vals)
    axes[1, 0].axhspan(2.24, 2.76, alpha=0.12)
    axes[1, 0].axhline(0.91, linewidth=0.8)
    axes[1, 0].set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    axes[1, 0].set_ylabel("Relative penicillin output")
    axes[1, 0].set_title("c  Theilgaard intervention classes")

    d = result["Douma2011_GSE24212"]
    labels = ["transcript", "protein moderate", "protein severe"]
    for kind in ["baseline", "legacy_feedforward", "reversible_IPNS"]:
        vals = [
            d["architectures"][kind]["scenarios"]["transcript_proxy"]["median"],
            d["architectures"][kind]["scenarios"]["protein_moderate"]["median"],
            d["architectures"][kind]["scenarios"]["protein_severe"]["median"],
        ]
        axes[1, 1].plot(np.arange(3), vals, marker="o", label=kind)
    axes[1, 1].axhline(0.1, linewidth=0.8)
    axes[1, 1].set_xticks(np.arange(3), labels, rotation=25, ha="right")
    axes[1, 1].set_ylabel("Relative penicillin output")
    axes[1, 1].set_title("d  GSE24212 degeneration")
    axes[1, 1].legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-try", type=int, default=800)
    parser.add_argument("--n-sub", type=int, default=200)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    accepted = fit_ensemble(n_try=args.n_try, seed=7, verbose=True)
    sub = sample_models(accepted, args.n_sub, args.seed)
    expr, _ = load_expression()

    result = {
        "run": {"n_candidates": args.n_try, "n_accepted": len(accepted), "n_sub": len(sub), "seed": args.seed},
        "GSE9825": validate_gse9825(accepted, args.n_sub, args.seed),
        "Weber2012": load_weber_result(),
        "Nijland2010": validate_nijland(sub),
        "Janoska2023": validate_janoska(sub, expr),
        "Theilgaard2001": validate_theilgaard(sub),
        "Douma2011_GSE24212": validate_degeneration(sub, expr),
        "construction_based_checks": {
            "Deshmukh2015": "PAA dependence and 6-APA routing informed model construction/acceptance and are not counted as independent validation.",
            "Douma2012_transport": "Reversible PenG transport is supported by the transport study but is a structural consistency check, not independent validation of a fitted parameter.",
        },
    }

    with open(OUT_JSON, "w") as fh:
        json.dump(result, fh, indent=2)
    write_summary_csv(result)
    write_figure(result)
    print(OUT_JSON)
    print(OUT_CSV)
    print(OUT_PNG)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
