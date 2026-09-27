import csv
import json
from pathlib import Path

from analysis.model_comparison import wilson
from penkin.ensemble import FREE
from penkin.model import ENZYMES

ROOT = Path(__file__).resolve().parents[1]


def test_uncertainty_range_table_covers_all_free_parameters():
    with open(ROOT / 'penkin' / 'data' / 'canonical_uncertainty_ranges.csv', newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert {r['parameter'] for r in rows} == set(FREE)


def test_wilson_interval_for_small_n_accuracy():
    lo, hi = wilson(13, 13)
    assert 0.77 < lo < 0.78
    assert 0.99 < hi <= 1.0


def test_reference_ensemble_outputs():
    """results/eval_results.json holds the full 8,000-model reference run.
    ACV synthetase carries the largest flux-control coefficient in 61.6% of
    admissible models and isopenicillin-N synthase in 32.7%."""
    res = json.loads((ROOT / 'results' / 'eval_results.json').read_text())
    assert res['n_candidates'] == 8000
    assert res['n_models_evaluated'] == res['n_accepted']
    frac = res['top_control_fraction_low']
    assert set(frac) == set(ENZYMES)
    assert abs(sum(frac.values()) - 1.0) < 1e-9
    assert round(frac['ACVS'] * 100, 1) == 61.6
    assert round(frac['IPNS'] * 100, 1) == 32.7

    ref = json.loads((ROOT / 'reference' / 'reference_values.json').read_text())['values']
    assert ref['top-control fraction ACVS (low)'] == frac['ACVS']


def test_figure3_plots_numerical_zeros_at_a_labelled_floor():
    """Figure 3b plots concentrations below 1e-8 mM at a 1e-8 mM floor and says
    so on the panel; these values are numerical zeros, not estimates."""
    src = (ROOT / 'analysis' / 'reproduce_figures.py').read_text()
    assert 'np.maximum(vals, 1e-8)' in src
    assert 'plotted at $10^{-8}$' in src


def test_feedforward_attrition_is_recorded():
    """With ACV product feedback removed, 2,752 of the 8,000 reference
    candidates have no steady state (5,202 would be admitted)."""
    res = json.loads((ROOT / 'results' / 'acceptance_audit.json').read_text())
    ff = res['feedforward_stage_counts']
    assert ff['sampled'] == 8000
    assert ff['sampled'] - ff['reference_solver_converged'] == 2752
    assert ff['accepted'] == 5202
