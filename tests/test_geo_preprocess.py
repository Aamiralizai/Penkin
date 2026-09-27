import os
import json
import pytest
from penkin.affymetrix import read_xda_cdf, read_cel, parse_series_matrix

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
DATA=os.path.join(ROOT,'penkin','data','geo_raw')

# The raw CEL and CDF archives are fetched on demand (see
# penkin/data/geo_raw/README.md) because they serve only the optional QC
# branch. Skip rather than fail when they are absent; the series-matrix test
# below covers the data path used for the published values and always runs.
_RAW_FILES=[os.path.join(DATA,'GSE9825_RAW','GPL6225.CDF.gz'),
            os.path.join(DATA,'GSE9825_RAW','GSM247945.CEL.gz'),
            os.path.join(DATA,'GSE12632_RAW','GSM315912.CEL.gz')]
requires_raw=pytest.mark.skipif(
    not all(os.path.exists(p) for p in _RAW_FILES),
    reason="raw GEO CEL files absent; run scripts/fetch_geo_raw.py to enable")


@requires_raw
def test_cdf_and_both_cel_formats():
    cdf=read_xda_cdf(os.path.join(DATA,'GSE9825_RAW','GPL6225.CDF.gz'))
    assert cdf['rows']==716 and cdf['cols']==716
    assert len(cdf['units']['Pc21g21390_s_at']['pm'])==11
    a=read_cel(os.path.join(DATA,'GSE9825_RAW','GSM247945.CEL.gz'))
    b=read_cel(os.path.join(DATA,'GSE12632_RAW','GSM315912.CEL.gz'))
    assert a.shape==(716,716) and b.shape==(716,716)
    assert a[0,0] > 0 and b[0,0] > 0

def test_gse9825_series_matrix_key_ratios():
    ids,titles,probes,mat=parse_series_matrix(os.path.join(DATA,'GSE9825_series_matrix.txt.gz'))
    p={x:i for i,x in enumerate(probes)}; s={x:i for i,x in enumerate(ids)}
    hi=[s[x] for x in ['GSM247948','GSM247949','GSM247950','GSM247951']]
    lo=[s[x] for x in ['GSM247955','GSM247956','GSM247957']]
    import numpy as np
    expected={'Pc21g21390_s_at':2.212305825242719,'Pc21g21380_at':2.1460718184108147,'Pc21g21370_at':1.4857963875205253}
    for probe,target in expected.items():
        v=mat[p[probe]]
        ratio=np.mean(v[hi])/np.mean(v[lo])
        assert ratio == pytest.approx(target, rel=1e-8)
