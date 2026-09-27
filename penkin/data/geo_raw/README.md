# GEO data

Transcriptome data for GSE9825 and GSE12632 from NCBI Gene Expression Omnibus
(public domain).

| File | Used for |
|---|---|
| `GSE9825_series_matrix.txt.gz` | Transcript-to-capacity analysis |
| `GSE12632_series_matrix.txt.gz` | Normalisation-sensitivity comparison |
| `GSE9825_family.soft.gz`, `GSE12632_family.soft.gz` | Sample annotation |
| `GSE9825_RAW/`, `GSE12632_RAW/` | Optional raw-CEL QC and normalisation-sensitivity step (one CDF and 13 CEL files per series) |

The published values do not depend on the raw CEL files; the step that uses
them can be skipped with `python run_all.py --full --skip-raw-geo`.

`CHECKSUMS.txt` records the SHA-256 of the 28 raw files used for the published
analysis. To verify them, or to download them again from GEO:

```bash
python scripts/fetch_geo_raw.py --verify   # check the files present
python scripts/fetch_geo_raw.py            # download and verify
```

## Sources

- **GSE9825**: van den Berg et al. (2008) *Nat Biotechnol* 26:1161–1168.
  Glucose-limited chemostat transcriptomes of the high-producing strain
  DS17690 and the low-producing strain Wisconsin 54-1255, each with and
  without phenylacetate; 13 arrays.
- **GSE12632**: companion series on the same platform (GPL6225), used for the
  normalisation-sensitivity check.
