#!/usr/bin/env python3
"""Run the complete Penkin workflow.

    python run_all.py --full     # reproduces every value and figure in the paper
    python run_all.py --quick    # reduced-size smoke test (a few minutes)

--full writes to results/ and figures/, then runs the test suite and
verify_reproduction.py against reference/reference_values.json.

--quick uses small ensembles and writes to results_quick/ and figures_quick/,
so the reference outputs in results/ and figures/ are never overwritten by a
reduced-size run. Quick-mode numbers are not expected to match the paper.

--skip-raw-geo omits the optional raw-CEL quality-control step
(analysis/geo_preprocess.py); the published values do not depend on it.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))

# Analysis sizes. The full configuration is the one used in the manuscript.
FULL = dict(
    acceptance_n=8000,                        # acceptance audit
    core_try=8000, core_sub=None,             # reference ensemble, all admissible models
    structural_try=8000, structural_sub=300,  # paired structural comparison
    perturb_try=8000, perturb_sub=300,        # perturbation and pairwise design
    broad_n=300,                              # broad-prior stress test
    lit_try=2000, lit_sub=250,                # Theilgaard, Janoska, Nijland
    weber_try=800, weber_sub=200,             # Weber et al. 2012
    ki_try=1000, ki_sub=100,                  # Ki_ACV profile
)
QUICK = dict(
    acceptance_n=200,
    core_try=200, core_sub=15,
    structural_try=200, structural_sub=15,
    perturb_try=200, perturb_sub=15,
    broad_n=40,
    lit_try=100, lit_sub=20,
    weber_try=100, weber_sub=20,
    ki_try=100, ki_sub=10,
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--full', action='store_true', help='manuscript configuration')
    mode.add_argument('--quick', action='store_true', help='reduced-size smoke test')
    ap.add_argument('--skip-raw-geo', action='store_true', help='skip the optional raw CEL QC step')
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = FULL if args.full else QUICK
    if args.quick:
        # Must be set before any analysis module is imported.
        os.environ['PENKIN_RESULTS_DIR'] = os.path.join(ROOT, 'results_quick')
        os.environ['PENKIN_FIGURES_DIR'] = os.path.join(ROOT, 'figures_quick')
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    from penkin.paths import RESULTS_DIR, FIGURES_DIR
    from analysis import geo_preprocess
    from analysis.acceptance_audit import main as acceptance_audit
    from analysis.evaluate_model import main as evaluate_model
    from analysis.structural_robustness import main as structural_robustness
    from analysis.broad_uncertainty import main as broad_uncertainty
    from analysis.literature_validation import run_validation as literature_validation
    from analysis.weber2012_validation import run_validation as weber_validation
    from analysis.ki_acv_profile import main as ki_acv_profile
    from analysis.ki_acv_identifiability import main as ki_acv_identifiability
    from analysis.model_comparison import main as model_comparison
    from analysis.perturbation_design import main as perturbation_design

    def script(name, *extra):
        subprocess.check_call([sys.executable, os.path.join(ROOT, 'analysis', name), *extra])

    print(f"Penkin workflow ({'full' if args.full else 'quick'}); outputs -> {RESULTS_DIR}, {FIGURES_DIR}")

    argv = sys.argv[:]
    try:
        sys.argv = ['geo_preprocess.py'] + (['--skip-raw'] if args.skip_raw_geo else [])
        geo_preprocess.main()
    finally:
        sys.argv = argv

    acceptance_audit(n_try=cfg['acceptance_n'], seed=7)
    evaluate_model(n_try=cfg['core_try'], n_sub=cfg['core_sub'])
    structural_robustness(n_try=cfg['structural_try'], n_sub=cfg['structural_sub'], seed=1)
    broad_uncertainty(n=cfg['broad_n'], seed=19)
    literature_validation(n_try=cfg['lit_try'], n_sub=cfg['lit_sub'], seed=41)
    weber_validation(n_try=cfg['weber_try'], n_sub=cfg['weber_sub'], seed=23)
    ki_acv_profile(n_try=cfg['ki_try'], n_sub=cfg['ki_sub'], seed=73)
    if args.full:
        ki_acv_identifiability()
    model_comparison(seed=0)
    script('gene_reaction_map.py')
    script('gem_context.py')
    script('figure1_architecture.py')
    fig_sub = -1 if cfg['core_sub'] is None else cfg['core_sub']
    script('reproduce_figures.py', '--n-try', str(cfg['core_try']), '--n-sub', str(fig_sub))
    perturbation_design(n_try=cfg['perturb_try'], n_sub=cfg['perturb_sub'], seed=29)
    script('figure6_weber.py')
    script('figure7_structural.py')

    import pytest
    rc = pytest.main(['-q', os.path.join(ROOT, 'tests')])
    if rc != 0:
        raise SystemExit(rc)

    if args.full:
        rc = subprocess.call([sys.executable, os.path.join(ROOT, 'verify_reproduction.py')])
        if rc != 0:
            raise SystemExit(rc)
    print(f'Penkin workflow completed. Outputs are in {RESULTS_DIR} and {FIGURES_DIR}.')


if __name__ == '__main__':
    main()
