import copy
import numpy as np
from .model import DEFAULT_PARAMS, scale_expression, steady_state

# Baseline uncertainty is centered on literature-anchored affinities where available,
# while reaction capacities remain deliberately broad model-scale factors.
FREE = {
    "Vmax_ACVS": (0.4, 2.5),
    "Vmax_IPNS": (0.4, 2.5),
    "Vmax_PCL": (0.4, 2.5),
    "Vmax_IAT": (0.4, 2.5),
    "Vmax_IAH": (0.2, 5.0),
    "Ki_ACV_ACVS": (0.36, 1.64),  # spans ~0.16-0.72 mM around 0.44 mM
    "Km_ACV": (0.70, 1.30),       # spans ~0.091-0.169 mM around 0.13 mM
    "Km_O2": (0.25, 4.0),         # poorly constrained; propagated explicitly
    "Km_IPNp": (0.5, 2.0),
    "Km_PAACoA": (0.5, 2.0),
    "Ki_PAACoA_PCL": (0.25, 4.0),
    "Km_IAH": (0.25, 4.0),
}

Y0 = [0.05, 0.02, 0.02, 0.05, 0.02, 0.01]
PAA_REFERENCE = 5.0
MIN_REFERENCE_SECRETION = 1e-4
MAX_NO_PAA_FRACTION = 0.05
MAX_INTERMEDIATE_MM = 10.0


def sample_candidate(rng):
    p = copy.deepcopy(DEFAULT_PARAMS)
    for k, (lo, hi) in FREE.items():
        p[k] = DEFAULT_PARAMS[k] * np.exp(rng.uniform(np.log(lo), np.log(hi)))
    return p


def accept_candidate(p):
    y5, f5 = steady_state(PAA_REFERENCE, p, Y0)
    if y5 is None or f5["secr"] < MIN_REFERENCE_SECRETION:
        return None
    # At PAA=0 the reduced PCL rate is identically zero, so PenG secretion is
    # structurally zero. We therefore do not use a redundant no-PAA integration
    # as an acceptance filter.
    if np.any(y5[:4] > MAX_INTERMEDIATE_MM):
        return None
    return {"p": p, "y_lo": list(y5), "J_lo": f5["secr"]}



def fit_ensemble(n_try=8000, seed=7, verbose=True):
    rng = np.random.default_rng(seed)
    accepted = [r for r in (accept_candidate(sample_candidate(rng)) for _ in range(n_try)) if r]
    if verbose:
        print(f"[ensemble] accepted={len(accepted)} candidates={n_try} rate={len(accepted)/n_try:.3f}")
    return accepted


def predict_producer_ratio(accepted, expr_fold, expr_boot=None, n_sub=700, seed=3, couple_pende=True):
    rng = np.random.default_rng(seed)
    sub = [accepted[i] for i in rng.choice(len(accepted), min(n_sub, len(accepted)), replace=False)]
    ratios = []
    for r in sub:
        ph = scale_expression(r["p"], expr_fold, couple_pende=couple_pende)
        yh, fh = steady_state(PAA_REFERENCE, ph, r["y_lo"])
        if yh is not None:
            ratios.append(fh["secr"] / r["J_lo"])
    ratios = np.asarray(ratios)
    out = {"ratios": ratios, "frac_high_gt_low": float(np.mean(ratios > 1))}

    if expr_boot is not None:
        boot = []
        n_boot = len(next(iter(expr_boot.values())))
        for _ in range(300):
            r = sub[rng.integers(len(sub))]
            b = rng.integers(n_boot)
            fb = {e: expr_boot[e][b] for e in expr_fold}
            ph = scale_expression(r["p"], fb, couple_pende=couple_pende)
            yh, fh = steady_state(PAA_REFERENCE, ph, r["y_lo"])
            if yh is not None:
                boot.append(fh["secr"] / r["J_lo"])
        out["boot_ratios"] = np.asarray(boot)
    return out, sub
