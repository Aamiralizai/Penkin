#!/usr/bin/env python3
"""Compare the outputs in results/ with the values reported in the paper.

Usage
-----
    python verify_reproduction.py                 # check results/ against the reference
    python verify_reproduction.py --rtol 0.05     # looser tolerance
    python verify_reproduction.py --write-reference

The reference file (``reference/reference_values.json``) holds the
manuscript values produced by ``python run_all.py --full``. Exit status is 0
when every check passes and 1 otherwise.

Tolerances
----------
Sampling is seeded, so a rerun on the same platform reproduces these values
essentially exactly. A 2% relative tolerance (default) absorbs differences in
BLAS builds, SciPy versions and platform floating point while still detecting
a change in configuration or model structure.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REFERENCE = ROOT / "reference" / "reference_values.json"

# Each check is (label, file, path, kind). ``path`` walks nested JSON keys.
# kind: "float" compares with the relative tolerance, "int" must match exactly,
# "unity" must sit within 1e-4 of 1.0 (the flux-control summation theorem).
CHECKS = [
    # --- reference ensemble and producer prediction -----------------------
    ("ensemble candidates",        "eval_results.json", ["n_candidates"], "int"),
    ("ensemble admitted",          "eval_results.json", ["n_accepted"], "int"),
    ("models evaluated",           "eval_results.json", ["n_models_evaluated"], "int"),
    ("producer ratio (median)",    "eval_results.json", ["producer_ratio", "median"], "float"),
    ("producer fraction high>low", "eval_results.json", ["producer_fraction_high_gt_low"], "float"),

    # --- flux-control coefficients, low-producer state --------------------
    ("C^J ACVS (low)",  "eval_results.json", ["control_low", "ACVS", "median"], "float"),
    ("C^J IPNS (low)",  "eval_results.json", ["control_low", "IPNS", "median"], "float"),
    ("C^J IAT (low)",   "eval_results.json", ["control_low", "IAT", "median"], "float"),
    ("C^J PCL (low)",   "eval_results.json", ["control_low", "PCL", "median"], "float"),

    # --- top-control fractions (abstract and Results) ---------------------
    ("top-control fraction ACVS (low)", "eval_results.json",
     ["top_control_fraction_low", "ACVS"], "float"),
    ("top-control fraction IPNS (low)", "eval_results.json",
     ["top_control_fraction_low", "IPNS"], "float"),
    ("top-control fraction IAT (low)",  "eval_results.json",
     ["top_control_fraction_low", "IAT"], "float"),

    # --- flux-control coefficients, high-producer state -------------------
    ("C^J ACVS (high)", "eval_results.json", ["control_high", "ACVS", "median"], "float"),
    ("C^J IPNS (high)", "eval_results.json", ["control_high", "IPNS", "median"], "float"),
    ("C^J IAT (high)",  "eval_results.json", ["control_high", "IAT", "median"], "float"),

    # --- summation theorem over all ten rate capacities -------------------
    ("MCA summation (low)",  "eval_results.json", ["control_sum_all_rates_low", "median"], "unity"),
    ("MCA summation (high)", "eval_results.json", ["control_sum_all_rates_high", "median"], "unity"),

    # --- coupled penDE response -------------------------------------------
    ("penDE 3x (low)",  "eval_results.json", ["penDE_3x_coupled_low", "median"], "float"),
    ("penDE 3x (high)", "eval_results.json", ["penDE_3x_coupled_high", "median"], "float"),

    # --- perturbation and pairwise design ---------------------------------
    ("ACVS 3x (low)",   "perturbation_design.json", ["low_state_scans", "ACVS", "3.0", "median"], "float"),
    ("IPNS 3x (low)",   "perturbation_design.json", ["low_state_scans", "IPNS", "3.0", "median"], "float"),
    ("penDE 3x scan (low)", "perturbation_design.json", ["low_state_scans", "PenDE", "3.0", "median"], "float"),
    ("ACVS 3x (high)",  "perturbation_design.json", ["high_state_scans", "ACVS", "3.0", "median"], "float"),
    ("IPNS 3x (high)",  "perturbation_design.json", ["high_state_scans", "IPNS", "3.0", "median"], "float"),
    ("pair ACVS+IPNS",  "perturbation_design.json", ["pairwise_3x_low_state", "ACVS", "IPNS", "median"], "float"),
    ("pair ACVS+penDE", "perturbation_design.json", ["pairwise_3x_low_state", "ACVS", "PenDE", "median"], "float"),

    # --- structural robustness, paired cohort -----------------------------
    ("structural baseline ACVS", "structural.json", ["baseline", "control_summary", "ACVS", "median"], "float"),
    ("structural baseline IPNS", "structural.json", ["baseline", "control_summary", "IPNS", "median"], "float"),
    ("feed-forward ACVS",        "structural.json", ["legacy_feedforward", "variant_on_pairs", "control_summary", "ACVS", "median"], "float"),
    ("feed-forward evaluable n", "structural.json", ["legacy_feedforward", "paired_n"], "int"),
    ("classical Ki ACVS",        "structural.json", ["ACVS_classical_Ki", "variant_on_pairs", "control_summary", "ACVS", "median"], "float"),

    # --- broad-prior stress test ------------------------------------------
    ("broad top-fraction ACVS", "broad_uncertainty.json", ["top_fraction", "ACVS"], "float"),
    ("broad top-fraction IPNS", "broad_uncertainty.json", ["top_fraction", "IPNS"], "float"),
    ("broad top-fraction IAT",  "broad_uncertainty.json", ["top_fraction", "IAT"], "float"),

    # --- acceptance audit ---------------------------------------------------
    ("acceptance sampled",  "acceptance_audit.json", ["stage_counts", "sampled"], "int"),
    ("acceptance admitted", "acceptance_audit.json", ["stage_counts", "accepted"], "int"),
    ("feed-forward steady states", "acceptance_audit.json", ["feedforward_stage_counts", "reference_solver_converged"], "int"),
    ("feed-forward admitted", "acceptance_audit.json", ["feedforward_stage_counts", "accepted"], "int"),

    # --- iAL1006 stoichiometric budget -----------------------------------------
    ("ATP per penicillin G", "gem_budget.json", ["gross_consumed_per_penicillin_G", "ATP"], "float"),
    ("O2 per penicillin G", "gem_budget.json", ["gross_consumed_per_penicillin_G", "O2"], "float"),
]

# External-validation metrics live in a CSV keyed by (dataset, architecture, mapping, metric).
CSV_CHECKS = [
    ("Theilgaard baseline RMSE (activities)",
     "Theilgaard2001", "baseline", "measured_ACVS_IPNS", "RMSE"),
    ("Theilgaard baseline RMSE (+penDE copy)",
     "Theilgaard2001", "baseline", "measured_plus_penDE_copy_proxy", "RMSE"),
    ("Theilgaard feed-forward RMSE (activities)",
     "Theilgaard2001", "legacy_feedforward", "measured_ACVS_IPNS", "RMSE"),
    ("Janoska baseline RMSE",
     "Janoska2023", "baseline", "oxygen", "RMSE"),
    ("Janoska feed-forward RMSE",
     "Janoska2023", "legacy_feedforward", "oxygen", "RMSE"),
    ("Nijland protein-informed Spearman",
     "Nijland2010", "baseline", "protein_informed", "Spearman"),
]
CSV_FILE = "literature_validation_metrics.csv"


def dig(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def load_json(name):
    p = RESULTS / name
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def read_csv_metrics():
    p = RESULTS / CSV_FILE
    if not p.exists():
        return {}
    out = {}
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["dataset"], row["architecture"], row["mapping"], row["metric"])
            out[key] = float(row["value"])
    return out


def collect():
    """Read every checked value out of results/. Missing values come back None."""
    values = {}
    cache = {}
    for label, fname, path, kind in CHECKS:
        if fname not in cache:
            cache[fname] = load_json(fname)
        doc = cache[fname]
        try:
            values[label] = {"value": dig(doc, path), "kind": kind}
        except (KeyError, TypeError):
            values[label] = {"value": None, "kind": kind}
    metrics = read_csv_metrics()
    for label, *key in CSV_CHECKS:
        values[label] = {"value": metrics.get(tuple(key)), "kind": "float"}
    return values


def write_reference():
    values = collect()
    missing = [k for k, v in values.items() if v["value"] is None]
    if missing:
        print("Cannot write a reference while these values are missing:")
        for m in missing:
            print(f"  - {m}")
        print("\nRun `python run_all.py --full` first.")
        return 1
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": (
            "Manuscript values from `python run_all.py --full` "
            "(8,000-candidate reference ensemble, seed 7). Regenerate with "
            "`python verify_reproduction.py --write-reference` only when the "
            "reported values are intended to change."
        ),
        "values": {k: v["value"] for k, v in values.items()},
    }
    REFERENCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {REFERENCE.relative_to(ROOT)} ({len(values)} values)")
    return 0


def compare(rtol):
    if not REFERENCE.exists():
        print(f"No reference file at {REFERENCE.relative_to(ROOT)}.")
        return 1
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))["values"]
    cur = collect()

    width = max(len(k) for k in ref)
    print(f"Penkin reproduction check  (relative tolerance {rtol:.1%})")
    print("=" * (width + 42))
    print(f"{'value':<{width}}  {'reference':>12}  {'this run':>12}  status")
    print("-" * (width + 42))

    failures, missing = [], []
    for label, expected in ref.items():
        entry = cur.get(label)
        got = entry["value"] if entry else None
        kind = entry["kind"] if entry else "float"

        if got is None:
            status, missing = "MISSING", missing + [label]
            shown = "-"
        else:
            if kind == "int":
                ok = int(got) == int(expected)
            elif kind == "unity":
                ok = abs(got - 1.0) < 1e-4
            else:
                denom = abs(expected) if abs(expected) > 1e-12 else 1.0
                ok = abs(got - expected) / denom <= rtol
            status = "ok" if ok else "FAIL"
            if not ok:
                failures.append((label, expected, got))
            shown = f"{got:.6g}"
        print(f"{label:<{width}}  {expected:>12.6g}  {shown:>12}  {status}")

    print("-" * (width + 42))
    total = len(ref)
    print(f"{total - len(failures) - len(missing)}/{total} values reproduced")

    if missing:
        print(f"\n{len(missing)} value(s) missing from results/ — "
              "run `python run_all.py --full` first.")
    if failures:
        print(f"\n{len(failures)} value(s) outside tolerance:")
        for label, expected, got in failures:
            denom = abs(expected) if abs(expected) > 1e-12 else 1.0
            print(f"  {label}: reference {expected:.6g}, this run {got:.6g} "
                  f"({(got - expected) / denom:+.2%})")
    if not failures and not missing:
        print("\nAll manuscript-facing values reproduced.")
    return 1 if (failures or missing) else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rtol", type=float, default=0.02,
                    help="relative tolerance for floating-point checks (default 0.02)")
    ap.add_argument("--write-reference", action="store_true",
                    help="overwrite reference/reference_values.json from results/")
    args = ap.parse_args()
    sys.exit(write_reference() if args.write_reference else compare(args.rtol))


if __name__ == "__main__":
    main()
