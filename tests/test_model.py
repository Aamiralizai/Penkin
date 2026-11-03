import numpy as np
from penkin.model import DEFAULT_PARAMS, ENZYMES, GENE_REACTION_MAP, RATE_CAPACITIES, scale_expression, steady_state
from penkin.variants import steady_variant, variant_fluxes


def test_steady_state_physiological():
    y, f = steady_state(5.0)
    assert y is not None
    assert np.all(y >= -1e-6)
    assert np.all(y < 60)
    assert f["secr"] > 0


def test_paa_dependence():
    _, f5 = steady_state(5.0)
    _, f0 = steady_state(0.0)
    assert f0["secr"] < 0.05 * f5["secr"]


def test_enzyme_map():
    assert set(ENZYMES) == {"ACVS", "IPNS", "PCL", "IAT", "IAH"}
    assert "PenG_secretion" in RATE_CAPACITIES


def test_pende_coupling():
    p = scale_expression(DEFAULT_PARAMS, {"IAT": 2.0}, couple_pende=True)
    assert p["Vmax_IAT"] == 2.0 * DEFAULT_PARAMS["Vmax_IAT"]
    assert p["Vmax_IAH"] == 2.0 * DEFAULT_PARAMS["Vmax_IAH"]
    assert GENE_REACTION_MAP["penDE"] == ("IAT", "IAH")


def test_pcl_feedback_unit_conversion():
    assert np.isclose(DEFAULT_PARAMS["Ki_PAACoA_PCL"], 0.0039)


def test_pcl_no_feedback_variant_removes_baseline_feedback():
    y, _ = steady_state(5.0)
    f0 = variant_fluxes(y, 5.0, DEFAULT_PARAMS, kind="baseline")
    fn = variant_fluxes(y, 5.0, DEFAULT_PARAMS, kind="PCL_no_feedback")
    assert fn["PCL"] > f0["PCL"]


def test_baseline_variant_solver_matches_main_solver():
    y1, f1 = steady_state(5.0)
    y2, f2 = steady_variant(5.0, DEFAULT_PARAMS, kind="baseline")
    assert np.allclose(y1, y2, rtol=1e-7, atol=1e-9)
    assert np.isclose(f1["secr"], f2["secr"], rtol=1e-7, atol=1e-10)
