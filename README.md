# Penkin

A six-state ensemble kinetic model of the committed penicillin G biosynthetic
pathway in *Penicillium chrysogenum* (*P. rubens*).

Reaction topology is taken from the iAL1006 genome-scale model, rate laws and
kinetic constants are taken from the primary experimental literature, and
parameter uncertainty is propagated over an 8,000-member ensemble. Predictions
are compared with four external datasets that are not used to build the model.

This repository reproduces every numerical value and figure in the
accompanying manuscript.

## Installation

```bash
git clone https://github.com/Aamiralizai/Penkin.git
cd Penkin

conda env create -f environment.yml
conda activate penkin
pip install -e .
```

Without conda (Python 3.11 or later):

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt # exact versions used for the paper
pip install -e .
```

## Reproducing the paper

```bash
python run_all.py --full
```

This regenerates `results/` and `figures/`, runs the test suite, and runs
`verify_reproduction.py`, which compares 46 reported values against
`reference/reference_values.json`. The run takes roughly 30–60 minutes on a
laptop.

| Command | Purpose |
|---|---|
| `python run_all.py --full` | Full workflow in the manuscript configuration |
| `python run_all.py --quick` | Reduced-size smoke test (a few minutes); writes to `results_quick/` and `figures_quick/` and does not touch the reference outputs |
| `python verify_reproduction.py` | Compare `results/` with the published values |
| `python -m pytest -q` | Test suite |

Quick-mode numbers come from small ensembles and are not expected to match the
paper.

### Ensemble sizes used in each analysis

| Analysis | Script | Candidates sampled | Models analysed |
|---|---|---|---|
| Reference ensemble, flux control, producer ratio | `evaluate_model.py` | 8,000 | all 8,000 admitted |
| Acceptance audit | `acceptance_audit.py` | 8,000 | 8,000 |
| Structural robustness (paired) | `structural_robustness.py` | 8,000 | 300 |
| Perturbation and pairwise design | `perturbation_design.py` | 8,000 | 300 |
| Broad-prior stress test (wider ranges) | `broad_uncertainty.py` | 300 | 296 valid |
| Theilgaard, Janoska, Nijland validation | `literature_validation.py` | 2,000 | 250 |
| Weber et al. 2012 validation | `weber2012_validation.py` | 800 | 200 |
| Ki(ACV) profile | `ki_acv_profile.py` | 1,000 | 100 |

All sampling is seeded (the reference ensemble uses seed 7; the other seeds are fixed in `run_all.py`), so reruns reproduce the reported values.

## Headline results

| Quantity | Low-producer state | High-producer state |
|---|---|---|
| Flux control, ACV synthetase (median) | 0.581 | 0.487 |
| Flux control, isopenicillin-N synthase (median) | 0.328 | 0.277 |
| Flux control, acyltransferase (median) | 0.043 | 0.053 |
| Flux-control summation (10 rate capacities) | 1.0000005 | 1.0000005 |
| Three-fold ACVS overexpression | 1.46× | 1.31× |
| Three-fold coupled penDE overexpression | 1.02× | 1.02× |

ACV synthetase carries the largest flux-control coefficient in 61.6% of
admissible models and isopenicillin-N synthase in 32.7%. The strongest paired
intervention is combined ACVS and IPNS overexpression (2.65×), ahead of ACVS
with penDE (1.70×).

All 8,000 sampled candidates satisfy the admission criteria (a steady state at
5 mM phenylacetate, penicillin G secretion above 1e-4, all intermediates below
10 mM); `analysis/acceptance_audit.py` reports the attrition at each criterion.

Because iAL1006 assigns reactions r0803 and r0813 to the same gene product
(Pc21g21370, *penDE*), the acyltransferase and amidohydrolase capacities are
always scaled together.

## Repository layout

```
penkin/       model, ensemble, design and I/O modules, plus all input data
analysis/     one script per analysis; each writes to results/ and figures/
scripts/      fetch_geo_raw.py (re-download and verify the raw GEO files)
tests/        test suite
results/      reference outputs of `python run_all.py --full`
figures/      reference figures (PNG and SVG)
reference/    published values checked by verify_reproduction.py
```

Individual analyses can be run on their own, for example:

```bash
python analysis/evaluate_model.py        # control coefficients, producer ratio
python analysis/acceptance_audit.py      # admission criteria and attrition
python analysis/structural_robustness.py # paired structural comparison
python analysis/literature_validation.py # Theilgaard, Janoska, Nijland
python analysis/weber2012_validation.py  # Weber et al. 2012 penDE and PCL series
```

## Data

All input data are included, so the workflow runs without network access.

| Source | Location | Reference |
|---|---|---|
| iAL1006 genome-scale model | `penkin/data/iAL1006.xml` | Ågren et al. 2013, *PLoS Comput Biol* 9:e1002980 |
| GSE9825 series matrix | `penkin/data/GSE9825_series_matrix_txt.gz` | van den Berg et al. 2008, *Nat Biotechnol* 26:1161 |
| GSE9825 and GSE12632 raw CEL files | `penkin/data/geo_raw/` | NCBI GEO |
| Digitized validation data | `penkin/data/*.csv` | `penkin/data/validation_source_provenance.csv` |
| Parameter provenance | `penkin/data/parameter_provenance.csv` | one row per parameter |
| Uncertainty ranges | `penkin/data/canonical_uncertainty_ranges.csv` | range and justification per free parameter |

The raw CEL files are used only for an optional quality-control and
normalisation-sensitivity step; the published values do not depend on them.
That step can be skipped with `python run_all.py --full --skip-raw-geo`.

## License

MIT. See `LICENSE` for the terms covering redistributed third-party data.
