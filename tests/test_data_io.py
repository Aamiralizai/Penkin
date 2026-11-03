from penkin import data_io


def test_topology_extraction():
    topo = data_io.load_topology()
    assert topo["n_reactions"] > 1000
    for enz in ["ACVS", "IPNS", "IAT", "IAH"]:
        assert enz in topo["pathway"]


def test_expression_loads():
    fold, reps = data_io.load_expression()
    assert set(fold) == {"ACVS", "IPNS", "IAT"}
    assert all(reps[e]["high"].size >= 3 for e in fold)
