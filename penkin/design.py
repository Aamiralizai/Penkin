import copy
import numpy as np
from .model import ENZYMES, RATE_CAPACITIES, scale_expression, steady_state


def control_coefficient(p, y0, par, PAA=5.0, d=1e-3):
    p0 = copy.deepcopy(p)
    yb, fb = steady_state(PAA, p0, y0)
    if yb is None or fb["secr"] <= 0:
        return None

    pp = copy.deepcopy(p0)
    pm = copy.deepcopy(p0)
    pp[par] *= 1.0 + d
    pm[par] *= 1.0 - d
    yp, fp = steady_state(PAA, pp, yb)
    ym, fm = steady_state(PAA, pm, yb)
    if yp is None or ym is None or fp["secr"] <= 0 or fm["secr"] <= 0:
        return None

    return (np.log(fp["secr"]) - np.log(fm["secr"])) / (np.log(1.0 + d) - np.log(1.0 - d))


def _gain(p, y0, J0, par, fold, PAA=5.0):
    p2 = copy.deepcopy(p)
    p2[par] *= fold
    y, f = steady_state(PAA, p2, y0)
    return None if y is None else f["secr"] / J0


def analyze(accepted, n_sub=800, seed=1, verbose=True, include_all_rates=True):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(accepted), size=min(n_sub, len(accepted)), replace=False)
    sub = [accepted[i] for i in idx]

    FCC = {e: [] for e in ENZYMES}
    ALL_FCC = {e: [] for e in RATE_CAPACITIES} if include_all_rates else {}
    gain3 = {e: [] for e in ENZYMES}
    gain_del = {e: [] for e in ENZYMES}
    pende3 = []

    for r in sub:
        p, y0, J0 = r["p"], r["y_lo"], r["J_lo"]

        control_map = RATE_CAPACITIES if include_all_rates else ENZYMES
        for e, par in control_map.items():
            c = control_coefficient(p, y0, par)
            if c is not None:
                if include_all_rates:
                    ALL_FCC[e].append(c)
                if e in FCC:
                    FCC[e].append(c)

        for e, par in ENZYMES.items():
            g = _gain(p, y0, J0, par, 3.0)
            gd = _gain(p, y0, J0, par, 0.1)
            if g is not None:
                gain3[e].append(g)
            if gd is not None:
                gain_del[e].append(gd)

        pp = scale_expression(p, {"IAT": 3.0}, couple_pende=True)
        y, f = steady_state(5.0, pp, y0)
        if y is not None:
            pende3.append(f["secr"] / J0)

    out = {
        "FCC": {e: np.asarray(v) for e, v in FCC.items()},
        "ALL_FCC": {e: np.asarray(v) for e, v in ALL_FCC.items()},
        "gain3": {e: np.asarray(v) for e, v in gain3.items()},
        "gain_del": {e: np.asarray(v) for e, v in gain_del.items()},
        "penDE_gain3": np.asarray(pende3),
    }
    if verbose:
        print(f"[design] models={len(sub)}")
    return out


def analyze_high_state(accepted, expr_fold, n_sub=800, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(accepted), size=min(n_sub, len(accepted)), replace=False)
    states = []
    for i in idx:
        r = accepted[i]
        p = scale_expression(r["p"], expr_fold, couple_pende=True)
        y, f = steady_state(5.0, p, r["y_lo"])
        if y is not None:
            states.append({"p": p, "y_lo": list(y), "J_lo": f["secr"]})
    return analyze(states, n_sub=len(states), seed=seed, verbose=False)


def summarize(res):
    lines = ["enzyme | C^J median [IQR] | 3x response median [IQR] | P(response>1.02)"]
    for e in ENZYMES:
        c = res["FCC"][e]
        g = res["gain3"][e]
        lines.append(
            f"{e:6s} | {np.median(c):+.3f} [{np.percentile(c,25):+.3f},{np.percentile(c,75):+.3f}] | "
            f"{np.median(g):.3f} [{np.percentile(g,25):.3f},{np.percentile(g,75):.3f}] | {np.mean(g>1.02):.3f}"
        )
    return "\n".join(lines)
