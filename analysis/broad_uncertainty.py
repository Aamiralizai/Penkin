#!/usr/bin/env python3
import copy
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from penkin.paths import RESULTS_DIR, FIGURES_DIR

from penkin.ensemble import MAX_INTERMEDIATE_MM, MAX_NO_PAA_FRACTION, MIN_REFERENCE_SECRETION
from penkin.model import DEFAULT_PARAMS, ENZYMES, POOLS, steady_state

OUT = os.path.join(RESULTS_DIR, "broad_uncertainty.json")
Y0 = [0.05, 0.02, 0.02, 0.05, 0.02, 0.01]

CAPACITY_KEYS = [k for k in DEFAULT_PARAMS if k.startswith("Vmax_") or k.startswith("kt_") or k.startswith("k_")]
KINETIC_KEYS = [k for k in DEFAULT_PARAMS if k.startswith("Km_") or k.startswith("Ki_")]
POOL_KEYS = list(POOLS)


def draw(rng):
    p = copy.deepcopy(DEFAULT_PARAMS)
    pools = copy.deepcopy(POOLS)
    for k in CAPACITY_KEYS:
        p[k] *= np.exp(rng.uniform(np.log(0.2), np.log(5.0)))
    for k in KINETIC_KEYS:
        p[k] *= np.exp(rng.uniform(np.log(0.25), np.log(4.0)))
    for k in POOL_KEYS:
        pools[k] *= np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
    return p, pools


def accept(p, pools):
    y5, f5 = steady_state(5.0, p, Y0, pools=pools)
    if y5 is None or f5["secr"] < MIN_REFERENCE_SECRETION:
        return None
    y0, f0 = steady_state(0.0, p, Y0, pools=pools)
    if y0 is None or f0["secr"] >= MAX_NO_PAA_FRACTION * f5["secr"]:
        return None
    if np.any(y5[:4] > MAX_INTERMEDIATE_MM):
        return None
    return y5, f5


def control(p, pools, y0, enzyme, d=1e-3):
    par = ENZYMES[enzyme]
    pp = copy.deepcopy(p)
    pm = copy.deepcopy(p)
    pp[par] *= 1.0 + d
    pm[par] *= 1.0 - d
    yp, fp = steady_state(5.0, pp, y0, pools=pools)
    ym, fm = steady_state(5.0, pm, y0, pools=pools)
    if yp is None or ym is None or fp["secr"] <= 0 or fm["secr"] <= 0:
        return None
    return (np.log(fp["secr"]) - np.log(fm["secr"])) / (np.log(1.0 + d) - np.log(1.0 - d))


def summ(x):
    a = np.asarray(x, dtype=float)
    return {
        "median": float(np.nanmedian(a)),
        "q05": float(np.nanpercentile(a, 5)),
        "q25": float(np.nanpercentile(a, 25)),
        "q75": float(np.nanpercentile(a, 75)),
        "q95": float(np.nanpercentile(a, 95)),
    }


def main(n=1200, seed=19):
    rng = np.random.default_rng(seed)
    prior = {k: [] for k in CAPACITY_KEYS + KINETIC_KEYS + [f"pool_{k}" for k in POOL_KEYS]}
    accepted = {k: [] for k in prior}
    coeff = {e: [] for e in ENZYMES}
    top = []
    n_acc = 0

    for _ in range(n):
        p, pools = draw(rng)
        for k in CAPACITY_KEYS + KINETIC_KEYS:
            prior[k].append(p[k] / DEFAULT_PARAMS[k])
        for k in POOL_KEYS:
            prior[f"pool_{k}"].append(pools[k] / POOLS[k])

        state = accept(p, pools)
        if state is None:
            continue
        y, _ = state
        c = {e: control(p, pools, y, e) for e in ENZYMES}
        if any(v is None for v in c.values()):
            continue

        n_acc += 1
        for k in CAPACITY_KEYS + KINETIC_KEYS:
            accepted[k].append(p[k] / DEFAULT_PARAMS[k])
        for k in POOL_KEYS:
            accepted[f"pool_{k}"].append(pools[k] / POOLS[k])
        for e, v in c.items():
            coeff[e].append(v)
        top.append(max(c, key=c.get))

    out = {
        "sampling": {
            "n": n,
            "accepted": n_acc,
            "capacity_factor_range": [0.2, 5.0],
            "kinetic_factor_range": [0.25, 4.0],
            "pool_factor_range": [0.5, 2.0],
            "interpretation": "broad uncertainty stress test; not literature-derived priors",
        },
        "prior_summary": {k: summ(v) for k, v in prior.items()},
        "accepted_summary": {k: summ(v) for k, v in accepted.items() if len(v)},
        "control_summary": {e: summ(v) for e, v in coeff.items()},
        "top_fraction": {e: float(np.mean(np.asarray(top) == e)) for e in ENZYMES},
        "control_values": coeff,
    }

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(OUT)


if __name__ == "__main__":
    main()
