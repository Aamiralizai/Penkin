#!/usr/bin/env python3
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from penkin.paths import RESULTS_DIR, FIGURES_DIR

from penkin import data_io, ensemble
from penkin.design import analyze, analyze_high_state

OUT = os.path.join(RESULTS_DIR, "eval_results.json")


def summary(a):
    a = np.asarray(a, dtype=float)
    return {
        "median": float(np.nanmedian(a)),
        "q25": float(np.nanpercentile(a, 25)),
        "q75": float(np.nanpercentile(a, 75)),
        "q05": float(np.nanpercentile(a, 5)),
        "q95": float(np.nanpercentile(a, 95)),
    }


def top_control_fractions(fcc):
    """Fraction of ensemble members in which each enzyme carries the largest
    flux-control coefficient. Reported in the abstract and Results, so it is
    written to the output file rather than only printed."""
    names = list(fcc)
    mat = np.vstack([np.asarray(fcc[e], dtype=float) for e in names])
    top = np.nanargmax(mat, axis=0)
    return {e: float(np.mean(top == i)) for i, e in enumerate(names)}


# Manuscript configuration. These are the settings used for every value
# reported in the manuscript and Supplementary Information. n_sub=None means
# the control analysis is evaluated on the full admissible ensemble rather
# than on a subsample, so the reported medians do not depend on a subsample
# size.
MANUSCRIPT_N_TRY = 8000
MANUSCRIPT_N_SUB = None          # None -> use the full admissible ensemble


def main(n_try=MANUSCRIPT_N_TRY, n_sub=MANUSCRIPT_N_SUB):
    fold, reps = data_io.load_expression()
    boot = data_io.bootstrap_expression(reps)
    accepted = ensemble.fit_ensemble(n_try=n_try, verbose=False)
    n_eval = len(accepted) if n_sub is None else n_sub
    pred, _ = ensemble.predict_producer_ratio(accepted, fold, boot, n_sub=n_eval, couple_pende=True)
    low = analyze(accepted, n_sub=n_eval, verbose=False)
    high = analyze_high_state(accepted, fold, n_sub=n_eval)

    low_sum = np.sum(np.vstack([low["ALL_FCC"][k] for k in low["ALL_FCC"]]), axis=0)
    high_sum = np.sum(np.vstack([high["ALL_FCC"][k] for k in high["ALL_FCC"]]), axis=0)

    out = {
        "n_candidates": n_try,
        "n_accepted": len(accepted),
        "n_models_evaluated": n_eval,
        "control_difference_scheme": "central",
        "expression_fold": fold,
        "producer_ratio": summary(pred["ratios"]),
        "producer_fraction_high_gt_low": pred["frac_high_gt_low"],
        "control_low": {e: summary(low["FCC"][e]) for e in low["FCC"]},
        "control_high": {e: summary(high["FCC"][e]) for e in high["FCC"]},
        "top_control_fraction_low": top_control_fractions(low["FCC"]),
        "top_control_fraction_high": top_control_fractions(high["FCC"]),
        "control_sum_all_rates_low": summary(low_sum),
        "control_sum_all_rates_high": summary(high_sum),
        "penDE_3x_coupled_low": summary(low["penDE_gain3"]),
        "penDE_3x_coupled_high": summary(high["penDE_gain3"]),
    }

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(OUT)


if __name__ == "__main__":
    main()
