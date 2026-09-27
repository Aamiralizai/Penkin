from analysis.weber2012_validation import PENDE_FOLDS, PCL_FOLDS, load_digitized


def test_validation_scan_ranges():
    assert 20.0 in PENDE_FOLDS
    assert 40.0 in PENDE_FOLDS
    assert PCL_FOLDS == [1.0, 3.0, 10.0, 40.0]


def test_digitized_weber_data():
    rows = load_digitized()
    assert len(rows) == 33
    assert {r["figure"] for r in rows} == {3, 6}
    assert any(r["figure"] == 3 and r["panel"] == "A" for r in rows)
    assert any(r["figure"] == 3 and r["panel"] == "B" for r in rows)
    assert any(r["figure"] == 6 for r in rows)
