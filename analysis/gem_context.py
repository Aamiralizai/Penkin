#!/usr/bin/env python3
"""Per-penicillin stoichiometric budget summed from the iAL1006 reactions of
the committed pathway (ACVS r0814, IPNS r0812, PCL r0747, IAT r0813).

Net consumption of each tracked species is summed over the four reactions;
pathway intermediates (ACV, IPN, phenylacetyl-CoA) cancel and are not tracked.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from penkin.paths import RESULTS_DIR

SBML = os.path.join(ROOT, "penkin", "data", "iAL1006.xml")
OUT = os.path.join(RESULTS_DIR, "gem_budget.json")

CORE = ["r0814", "r0812", "r0747", "r0813"]
TRACKED = {
    "ATP": {"M_ATP", "M_ATPp"},
    "O2": {"M_O2", "M_O2p"},
    "CoA": {"M_COA", "M_COAp"},
    "alpha-aminoadipate": {"M_AMA", "M_AMAp"},
    "cysteine": {"M_CYS"},
    "valine": {"M_VAL"},
    "phenylacetate": {"M_PAA", "M_PAAp"},
}


def main():
    import libsbml
    model = libsbml.readSBML(SBML).getModel()
    net = {k: 0.0 for k in TRACKED}
    gross = {k: 0.0 for k in TRACKED}
    for r in model.getListOfReactions():
        if r.getId().replace("R_", "") not in CORE:
            continue
        for sign, refs in ((1.0, r.getListOfReactants()), (-1.0, r.getListOfProducts())):
            for sr in refs:
                for key, ids in TRACKED.items():
                    if sr.getSpecies() in ids:
                        net[key] += sign * sr.getStoichiometry()
                        if sign > 0:
                            gross[key] += sr.getStoichiometry()
    out = {
        "reactions": CORE,
        "gross_consumed_per_penicillin_G": gross,
        "net_consumed_per_penicillin_G": net,
        "note": ("Summed from iAL1006 stoichiometry; no flux-balance simulation is performed. "
                 "Gross counts reactant consumption; net subtracts products, so CoA and the "
                 "alpha-aminoadipate released by the acyltransferase (r0813) cancel."),
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(OUT)
    return out


if __name__ == "__main__":
    main()
