import copy
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# Buffered pools are reference-state model inputs, not fitted kinetic constants.
# PAA=5 mM is handled as the external producing condition in ensemble.py.
POOLS = {"AAA": 0.50, "CYS": 0.20, "VAL": 2.00, "O2": 0.18}

# Affinity/inhibition constants that have direct literature counterparts are
# anchored to published values. Capacity and transport terms remain model-scale
# reference values because the available enzyme-activity literature uses
# strain-, assay-, or cell-volume-specific units that are not directly
# interchangeable with this reduced six-state formulation.
DEFAULT_PARAMS = {
    "Vmax_ACVS": 0.60,
    "Km_AAA": 0.045,        # mM; Theilgaard et al. 1997, cited by Nasution et al. 2008
    "Km_CYS": 0.080,        # mM; purified P. chrysogenum ACVS, Theilgaard et al. 1997
    "Km_VAL": 0.080,        # mM; Theilgaard et al. 1997, cited by Nasution et al. 2008
    "Ki_ACV_ACVS": 0.44,    # mM; Deshmukh et al. 2015 (0.44 +/- 0.28); 0.54 mM in Theilgaard & Nielsen 1999
    "Vmax_IPNS": 0.90,
    "Km_ACV": 0.13,         # mM; Ramos et al. 1985; used by Nielsen 1995 / Deshmukh 2015
    "Km_O2": 0.10,          # mM; phenomenological reference, not a literature point estimate
    "kt_IPN": 5.0,
    "Vmax_PCL": 1.20,
    "Ki_PAACoA_PCL": 0.0039, # mM; 3.9 mmol/m^3 from Deshmukh et al. 2015
    "Vmax_IAT": 0.80,
    "Km_IPNp": 0.023,       # mM; Alvarez et al. 1987, fixed in Deshmukh et al. 2015
    "Km_PAACoA": 0.006,     # mM; Alvarez et al. 1987, fixed in Deshmukh et al. 2015
    "Vmax_IAH": 0.25,
    "Km_IAH": 412.0,        # mM; Deshmukh et al. 2015, high-capacity/low-affinity branch
    "kt_PENG": 5.0,
    "k_secr": 3.0,
    "k_IPNsecr": 0.8,
    "k_PAACoA_hyd": 0.6,
}

STATES = ["ACV", "IPNc", "IPNp", "PAACoAp", "PENGp", "PENGc"]

ENZYMES = {
    "ACVS": "Vmax_ACVS",
    "IPNS": "Vmax_IPNS",
    "PCL": "Vmax_PCL",
    "IAT": "Vmax_IAT",
    "IAH": "Vmax_IAH",
}

RATE_CAPACITIES = {
    "ACVS": "Vmax_ACVS",
    "IPNS": "Vmax_IPNS",
    "PCL": "Vmax_PCL",
    "IAT": "Vmax_IAT",
    "IAH": "Vmax_IAH",
    "IPN_transport": "kt_IPN",
    "PenG_transport": "kt_PENG",
    "PenG_secretion": "k_secr",
    "IPN_secretion": "k_IPNsecr",
    "PAA_CoA_hydrolysis": "k_PAACoA_hyd",
}

GENE_REACTION_MAP = {
    "pcbAB": ("ACVS",),
    "pcbC": ("IPNS",),
    "penDE": ("IAT", "IAH"),
}


def acvs_precursor_factor(p, pools=POOLS):
    AAA = pools["AAA"]
    CYS = pools["CYS"]
    VAL = pools["VAL"]
    return (
        (AAA / (p["Km_AAA"] + AAA))
        * (CYS / (p["Km_CYS"] + CYS))
        * (VAL / (p["Km_VAL"] + VAL))
    )


def fluxes(y, PAA, p, pools=POOLS):
    ACV, IPNc, IPNp, PAACoAp, PENGp, PENGc = y
    O2 = pools["O2"]

    # Literature-supported product feedback: ACV inhibits ACVS.
    v_ACVS = (
        p["Vmax_ACVS"]
        * acvs_precursor_factor(p, pools)
        * (1.0 / (1.0 + ACV / p["Ki_ACV_ACVS"]))
    )
    v_IPNS = p["Vmax_IPNS"] * (ACV / (p["Km_ACV"] + ACV)) * (O2 / (p["Km_O2"] + O2))
    v_tIPN = p["kt_IPN"] * (IPNc - IPNp)
    # Deshmukh et al. found PCL to be best represented as first order in PAA
    # with competitive PA-CoA feedback; the capacity coefficient remains model-scale.
    v_PCL = p["Vmax_PCL"] * PAA * (1.0 / (1.0 + PAACoAp / p["Ki_PAACoA_PCL"]))
    v_IAT = p["Vmax_IAT"] * (IPNp / (p["Km_IPNp"] + IPNp)) * (PAACoAp / (p["Km_PAACoA"] + PAACoAp))
    v_IAH = p["Vmax_IAH"] * (IPNp / (p["Km_IAH"] + IPNp))
    v_tPENG = p["kt_PENG"] * (PENGp - PENGc)
    v_secr = p["k_secr"] * PENGc
    v_IPNsecr = p["k_IPNsecr"] * IPNc
    v_PAAChyd = p["k_PAACoA_hyd"] * PAACoAp

    return {
        "ACVS": v_ACVS,
        "IPNS": v_IPNS,
        "tIPN": v_tIPN,
        "PCL": v_PCL,
        "IAT": v_IAT,
        "IAH": v_IAH,
        "tPENG": v_tPENG,
        "secr": v_secr,
        "IPNsecr": v_IPNsecr,
        "PAAChyd": v_PAAChyd,
    }


def rhs(t, y, PAA, p, pools=POOLS):
    f = fluxes(y, PAA, p, pools)
    return [
        f["ACVS"] - f["IPNS"],
        f["IPNS"] - f["tIPN"] - f["IPNsecr"],
        f["tIPN"] - f["IAT"] - f["IAH"],
        f["PCL"] - f["IAT"] - f["PAAChyd"],
        f["IAT"] - f["tPENG"],
        f["tPENG"] - f["secr"],
    ]


def steady_state(PAA, p=DEFAULT_PARAMS, y0=None, tmax=300, method="LSODA", pools=POOLS):
    if y0 is None:
        y0 = [0.05, 0.02, 0.02, 0.05, 0.02, 0.01]
    y0 = np.clip(np.asarray(y0, dtype=float), 0.0, 60.0)

    # Direct bounded solution of the steady-state mass balances. Dynamic
    # trajectories are integrated separately where a time course is required.
    fun = lambda yy: np.asarray(rhs(0.0, yy, PAA, p, pools), dtype=float)
    try:
        opt = least_squares(
            fun, y0, bounds=(np.zeros(6), np.full(6, 60.0)),
            xtol=1e-9, ftol=1e-9, gtol=1e-9, max_nfev=120
        )
        y = opt.x
        if opt.success and np.linalg.norm(fun(y), ord=np.inf) < 1e-6:
            return y, fluxes(y, PAA, p, pools)
    except Exception:
        pass
    return None, None

def scale_expression(p, expr_fold, couple_pende=True):
    q = copy.deepcopy(p)
    if "ACVS" in expr_fold:
        q["Vmax_ACVS"] *= expr_fold["ACVS"]
    if "IPNS" in expr_fold:
        q["Vmax_IPNS"] *= expr_fold["IPNS"]
    if "IAT" in expr_fold:
        q["Vmax_IAT"] *= expr_fold["IAT"]
        if couple_pende:
            q["Vmax_IAH"] *= expr_fold["IAT"]
    return q
