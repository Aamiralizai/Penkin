"""Output locations.

Analysis scripts write to ``results/`` and ``figures/`` at the repository root.
``run_all.py --quick`` redirects them to ``results_quick/`` and
``figures_quick/`` through the environment variables below, so a reduced-size
test run never overwrites the reference outputs.
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.abspath(os.environ.get("PENKIN_RESULTS_DIR", os.path.join(ROOT, "results")))
FIGURES_DIR = os.path.abspath(os.environ.get("PENKIN_FIGURES_DIR", os.path.join(ROOT, "figures")))
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
