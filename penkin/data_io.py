import gzip
import os
import xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")

PATHWAY_PROBES = {
    "ACVS": "Pc21g21390_s_at",
    "IPNS": "Pc21g21380_at",
    "IAT": "Pc21g21370_at",
}

COND = {
    "HIGH_noPAA": [0, 1, 2],
    "HIGH_PAA": [3, 4, 5, 6],
    "LOW_noPAA": [7, 8, 9],
    "LOW_PAA": [10, 11, 12],
}

CORE_RXNS = {
    "r0814": "ACVS",
    "r0812": "IPNS",
    "r0747": "PCL",
    "r0813": "IAT",
    "r0803": "IAH",
    "r1271": "IPN_transport",
    "r1281": "PenG_transport",
}


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _load_topology_xml(sbml_path):
    tree = ET.parse(sbml_path)
    root = tree.getroot()
    reactions = [x for x in root.iter() if _local(x.tag) == "reaction"]
    species = [x for x in root.iter() if _local(x.tag) == "species"]
    found = {}

    for r in reactions:
        rid = r.attrib.get("id", "").replace("R_", "")
        if rid not in CORE_RXNS:
            continue
        reactants = {}
        products = {}
        for node in r:
            kind = _local(node.tag)
            if kind not in {"listOfReactants", "listOfProducts"}:
                continue
            target = reactants if kind == "listOfReactants" else products
            for sr in node:
                if _local(sr.tag) != "speciesReference":
                    continue
                target[sr.attrib.get("species")] = float(sr.attrib.get("stoichiometry", 1.0))
        found[CORE_RXNS[rid]] = {
            "id": rid,
            "name": r.attrib.get("name", ""),
            "reactants": reactants,
            "products": products,
        }

    return {
        "n_reactions": len(reactions),
        "n_species": len(species),
        "pathway": found,
    }


def load_topology(sbml_path=None, require=True):
    if sbml_path is None:
        sbml_path = os.path.join(DATA, "iAL1006.xml")

    try:
        import libsbml
    except ImportError:
        topo = _load_topology_xml(sbml_path)
    else:
        doc = libsbml.readSBML(sbml_path)
        model = doc.getModel()
        if model is None:
            raise ValueError(f"Could not parse SBML model at {sbml_path}")
        found = {}
        for r in model.getListOfReactions():
            rid = r.getId().replace("R_", "")
            if rid in CORE_RXNS:
                found[CORE_RXNS[rid]] = {
                    "id": rid,
                    "name": r.getName(),
                    "reactants": {sr.getSpecies(): sr.getStoichiometry() for sr in r.getListOfReactants()},
                    "products": {sr.getSpecies(): sr.getStoichiometry() for sr in r.getListOfProducts()},
                }
        topo = {
            "n_reactions": model.getNumReactions(),
            "n_species": model.getNumSpecies(),
            "pathway": found,
        }

    missing = set(CORE_RXNS.values()) - set(topo["pathway"])
    if missing and require:
        raise ValueError(f"iAL1006 is missing expected pathway reactions: {sorted(missing)}")
    return topo


def load_expression(gse_path=None):
    if gse_path is None:
        for name in ["GSE9825_series_matrix_txt.gz", "GSE9825_series_matrix.txt.gz"]:
            cand = os.path.join(DATA, name)
            if os.path.exists(cand):
                gse_path = cand
                break
    if gse_path is None or not os.path.exists(gse_path):
        raise FileNotFoundError("GSE9825 series matrix not found in data/.")

    rows = {}
    with gzip.open(gse_path, "rt") as fh:
        block = False
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                block = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if block:
                parts = line.rstrip("\n").split("\t")
                try:
                    rows[parts[0].strip('"')] = np.array([float(x) for x in parts[1:]])
                except ValueError:
                    pass

    fold = {}
    reps = {}
    for enzyme, probe in PATHWAY_PROBES.items():
        hi = rows[probe][COND["HIGH_PAA"]]
        lo = rows[probe][COND["LOW_PAA"]]
        fold[enzyme] = float(hi.mean() / lo.mean())
        reps[enzyme] = {"high": hi, "low": lo}
    return fold, reps


def bootstrap_expression(reps, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    out = {}
    for enzyme, d in reps.items():
        hi = d["high"]
        lo = d["low"]
        vals = [
            rng.choice(hi, hi.size, replace=True).mean()
            / rng.choice(lo, lo.size, replace=True).mean()
            for _ in range(n)
        ]
        out[enzyme] = np.asarray(vals)
    return out
