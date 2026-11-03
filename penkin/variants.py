import copy
import numpy as np
from scipy.optimize import least_squares

from .model import POOLS, fluxes as base_fluxes, rhs as base_rhs

Y0 = [0.05, 0.02, 0.02, 0.05, 0.02, 0.01]


def variant_fluxes(y, PAA, p, pools=POOLS, kind="baseline"):
    allowed = {"baseline", "legacy_feedforward", "reversible_IPNS", "PCL_no_feedback", "ACVS_classical_Ki", "IAH_classical_4mM", "combined"}
    if kind not in allowed:
        raise ValueError(f"Unknown structural variant: {kind}")
    f = base_fluxes(y, PAA, p, pools)
    ACV, IPNc, IPNp, PAACoAp = y[0], y[1], y[2], y[3]

    if kind == "legacy_feedforward":
        fb = 1.0 / (1.0 + ACV / p["Ki_ACV_ACVS"])
        if fb > 0:
            f["ACVS"] /= fb
    elif kind == "reversible_IPNS":
        reverse = 0.3 * p["Vmax_IPNS"] * IPNc / (p["Km_ACV"] + IPNc)
        f["IPNS"] -= reverse
    elif kind == "PCL_no_feedback":
        fb = 1.0 / (1.0 + PAACoAp / p["Ki_PAACoA_PCL"])
        if fb > 0:
            f["PCL"] /= fb
    elif kind == "ACVS_classical_Ki":
        old_fb = 1.0 / (1.0 + ACV / p["Ki_ACV_ACVS"])
        new_fb = 1.0 / (1.0 + ACV / 12.5)
        if old_fb > 0:
            f["ACVS"] *= new_fb / old_fb
    elif kind == "IAH_classical_4mM":
        old_sat = IPNp / (p["Km_IAH"] + IPNp) if IPNp > 0 else 0.0
        new_sat = IPNp / (4.0 + IPNp) if IPNp > 0 else 0.0
        if old_sat > 0:
            f["IAH"] *= new_sat / old_sat
    elif kind == "combined":
        reverse = 0.3 * p["Vmax_IPNS"] * IPNc / (p["Km_ACV"] + IPNc)
        f["IPNS"] -= reverse
        fb = 1.0 / (1.0 + PAACoAp / p["Ki_PAACoA_PCL"])
        if fb > 0:
            f["PCL"] /= fb
    return f


def variant_rhs(t, y, PAA, p, pools, kind):
    f = variant_fluxes(y, PAA, p, pools, kind)
    return [
        f["ACVS"] - f["IPNS"],
        f["IPNS"] - f["tIPN"] - f["IPNsecr"],
        f["tIPN"] - f["IAT"] - f["IAH"],
        f["PCL"] - f["IAT"] - f["PAAChyd"],
        f["IAT"] - f["tPENG"],
        f["tPENG"] - f["secr"],
    ]


def steady_variant(PAA, p, y0=None, pools=None, kind="baseline", tmax=300):
    if pools is None:
        pools = POOLS
    if y0 is None:
        y0 = Y0
    y0 = np.clip(np.asarray(y0, dtype=float), 0.0, 60.0)
    rhs_fun = base_rhs if kind == "baseline" else variant_rhs
    args = (PAA, p, pools) if kind == "baseline" else (PAA, p, pools, kind)
    flux_fun = base_fluxes if kind == "baseline" else (lambda yy, ppaa, pp, ppool: variant_fluxes(yy, ppaa, pp, ppool, kind))
    fun = lambda yy: np.asarray(rhs_fun(0.0, yy, *args), dtype=float)
    try:
        opt = least_squares(fun, y0, bounds=(np.zeros(6), np.full(6, 60.0)),
                            xtol=1e-9, ftol=1e-9, gtol=1e-9, max_nfev=120)
        yy = opt.x
        if opt.success and np.linalg.norm(fun(yy), ord=np.inf) < 1e-6:
            return yy, flux_fun(yy, PAA, p, pools)
    except Exception:
        pass
    return None, None


def scale_cluster(p, acvs=1.0, ipns=1.0, pende=1.0, pcl=1.0):
    q = copy.deepcopy(p)
    q["Vmax_ACVS"] *= acvs
    q["Vmax_IPNS"] *= ipns
    q["Vmax_IAT"] *= pende
    q["Vmax_IAH"] *= pende
    q["Vmax_PCL"] *= pcl
    return q
