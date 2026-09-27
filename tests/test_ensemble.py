from penkin.ensemble import fit_ensemble, predict_producer_ratio


def test_fit_and_predict_small():
    acc = fit_ensemble(n_try=200, verbose=False)
    assert len(acc) > 10
    pred, _ = predict_producer_ratio(acc, {"ACVS": 2.21, "IPNS": 2.15, "IAT": 1.49}, n_sub=50)
    assert 0.0 <= pred["frac_high_gt_low"] <= 1.0
    assert len(pred["ratios"]) > 0
