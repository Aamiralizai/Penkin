#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from penkin.paths import RESULTS_DIR, FIGURES_DIR

from penkin.affymetrix import (
    infer_geo_group,
    load_pm_matrix,
    parse_series_matrix,
    quantile_normalize_log2,
    summarize_probesets,
)

DATA = os.path.join(ROOT, 'penkin', 'data', 'geo_raw')
RESULTS = os.path.join(RESULTS_DIR, 'geo')
FIGURES = os.path.join(FIGURES_DIR, 'geo')

KEY_PROBES = {
    'ACVS_pcbAB': 'Pc21g21390_s_at',
    'IPNS_pcbC': 'Pc21g21380_at',
    'PenDE_penDE': 'Pc21g21370_at',
    'PCL_phl': 'Pc22g14900_at',
}


def group_indices(sample_ids, titles):
    out = defaultdict(list)
    for i, (sid, title) in enumerate(zip(sample_ids, titles)):
        strain, paa = infer_geo_group(title)
        out[f'{strain}_{paa}'].append(i)
    return dict(out)


def fold(a, ia, ib, logged=False):
    if logged:
        return float(2 ** (np.nanmean(a[ia]) - np.nanmean(a[ib])))
    return float(np.nanmean(a[ia]) / np.nanmean(a[ib]))


def bootstrap_fold(a, ia, ib, logged=False, n=5000, seed=17):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        aa = rng.choice(ia, len(ia), replace=True)
        bb = rng.choice(ib, len(ib), replace=True)
        vals.append(fold(a, aa, bb, logged=logged))
    q = np.percentile(vals, [2.5, 25, 50, 75, 97.5])
    return {'median': float(q[2]), 'q025': float(q[0]), 'q25': float(q[1]), 'q75': float(q[3]), 'q975': float(q[4])}


def pca(x):
    z = x - np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(z, axis=1, keepdims=True)
    keep = (sd[:, 0] > 1e-8) & np.all(np.isfinite(z), axis=1)
    z = z[keep]
    # Samples x features, center samples across genes only through SVD of gene-centered data
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    scores = (np.diag(s) @ vt).T
    var = s**2 / np.sum(s**2)
    return scores[:, :2], var[:2]


def analyze_dataset(gse, raw=False):
    sm = os.path.join(DATA, f'{gse}_series_matrix.txt.gz')
    sample_ids, titles, probes, mat = parse_series_matrix(sm)
    groups = group_indices(sample_ids, titles)
    probe_to_i = {p:i for i,p in enumerate(probes)}

    out = {
        'dataset': gse,
        'samples': [{'id':sid, 'title':title, 'strain':infer_geo_group(title)[0], 'PAA':infer_geo_group(title)[1]} for sid,title in zip(sample_ids,titles)],
        'groups': {k:[sample_ids[i] for i in v] for k,v in groups.items()},
        'series_matrix': {'key_probes': {}},
    }
    for label, probe in KEY_PROBES.items():
        if probe not in probe_to_i:
            continue
        v = mat[probe_to_i[probe]]
        rec = {'probe': probe, 'values': {sid: float(v[i]) for i,sid in enumerate(sample_ids)}}
        if gse == 'GSE9825':
            hi_p = groups.get('DS17690_+PAA', [])
            lo_p = groups.get('Wisconsin54-1255_+PAA', [])
            hi_m = groups.get('DS17690_-PAA', [])
            lo_m = groups.get('Wisconsin54-1255_-PAA', [])
            if hi_p and lo_p:
                rec['DS17690_vs_Wisconsin_plusPAA'] = fold(v, hi_p, lo_p, logged=False)
                rec['bootstrap_plusPAA'] = bootstrap_fold(v, hi_p, lo_p, logged=False)
            if hi_m and lo_m:
                rec['DS17690_vs_Wisconsin_minusPAA'] = fold(v, hi_m, lo_m, logged=False)
                rec['bootstrap_minusPAA'] = bootstrap_fold(v, hi_m, lo_m, logged=False)
        elif gse == 'GSE12632':
            ds_p = groups.get('DS17690_+PAA', [])
            cf_p = groups.get('DS50661_+PAA', [])
            ds_m = groups.get('DS17690_-PAA', [])
            cf_m = groups.get('DS50661_-PAA', [])
            if ds_p and cf_p:
                rec['DS17690_vs_clusterfree_plusPAA'] = fold(v, ds_p, cf_p, logged=False)
            if ds_m and cf_m:
                rec['DS17690_vs_clusterfree_minusPAA'] = fold(v, ds_m, cf_m, logged=False)
            if ds_p and ds_m:
                rec['DS17690_PAA_effect'] = fold(v, ds_p, ds_m, logged=False)
            if cf_p and cf_m:
                rec['DS50661_PAA_effect'] = fold(v, cf_p, cf_m, logged=False)
        out['series_matrix']['key_probes'][label] = rec

    if raw:
        raw_dir = os.path.join(DATA, f'{gse}_RAW')
        cdf = os.path.join(raw_dir, 'GPL6225.CDF.gz')
        cels = [os.path.join(raw_dir, sid + '.CEL.gz') for sid in sample_ids]
        missing = [p for p in cels if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f'{len(missing)} raw CEL file(s) for {gse} not found in {raw_dir} '
                f'(first: {os.path.basename(missing[0])}). Download them with '
                '`python scripts/fetch_geo_raw.py`, or skip this optional QC step '
                'with `python run_all.py --skip-raw-geo`.')
        raw_ids, pm, slices, qc = load_pm_matrix(cels, cdf)
        qn = quantile_normalize_log2(pm)
        ps_names, expr = summarize_probesets(qn, slices, method='mean')
        ps_i = {p:i for i,p in enumerate(ps_names)}
        corr = np.corrcoef(qn.T)
        scores, var = pca(expr)
        raw_rec = {
            'method': 'PM-only log2 + across-array quantile normalization + mean probe summarization (sensitivity analysis; not MAS5/GCOS and not claimed as exact RMA)',
            'qc': qc,
            'sample_correlation_min': float(np.min(corr[np.triu_indices_from(corr,1)])),
            'sample_correlation_median': float(np.median(corr[np.triu_indices_from(corr,1)])),
            'pca_variance_PC1': float(var[0]),
            'pca_variance_PC2': float(var[1]),
            'key_probes': {},
        }
        for label, probe in KEY_PROBES.items():
            if probe not in ps_i:
                continue
            v = expr[ps_i[probe]]
            rec = {'probe':probe, 'log2_values': {sid: float(v[i]) for i,sid in enumerate(sample_ids)}}
            if gse == 'GSE9825':
                hi_p = groups.get('DS17690_+PAA', []); lo_p = groups.get('Wisconsin54-1255_+PAA', [])
                hi_m = groups.get('DS17690_-PAA', []); lo_m = groups.get('Wisconsin54-1255_-PAA', [])
                if hi_p and lo_p:
                    rec['DS17690_vs_Wisconsin_plusPAA'] = fold(v, hi_p, lo_p, logged=True)
                if hi_m and lo_m:
                    rec['DS17690_vs_Wisconsin_minusPAA'] = fold(v, hi_m, lo_m, logged=True)
            elif gse == 'GSE12632':
                ds_p = groups.get('DS17690_+PAA', []); cf_p = groups.get('DS50661_+PAA', [])
                ds_m = groups.get('DS17690_-PAA', []); cf_m = groups.get('DS50661_-PAA', [])
                if ds_p and cf_p:
                    rec['DS17690_vs_clusterfree_plusPAA'] = fold(v, ds_p, cf_p, logged=True)
                if ds_m and cf_m:
                    rec['DS17690_vs_clusterfree_minusPAA'] = fold(v, ds_m, cf_m, logged=True)
                if ds_p and ds_m:
                    rec['DS17690_PAA_effect'] = fold(v, ds_p, ds_m, logged=True)
                if cf_p and cf_m:
                    rec['DS50661_PAA_effect'] = fold(v, cf_p, cf_m, logged=True)
            raw_rec['key_probes'][label] = rec
        out['raw_pm_qn_sensitivity'] = raw_rec

        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, f'{gse}_raw_qc.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=qc[0].keys()); w.writeheader(); w.writerows(qc)

        # PCA figure
        os.makedirs(FIGURES, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.4,5.0))
        for i,(sid,title) in enumerate(zip(sample_ids,titles)):
            strain,paa = infer_geo_group(title)
            ax.scatter(scores[i,0], scores[i,1], s=46)
            ax.text(scores[i,0], scores[i,1], sid.replace('GSM',''), fontsize=7, ha='left', va='bottom')
        ax.set_xlabel(f'PC1 ({100*var[0]:.1f}%)')
        ax.set_ylabel(f'PC2 ({100*var[1]:.1f}%)')
        ax.set_title(f'{gse} raw CEL PM-QN sensitivity PCA')
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES, f'{gse}_PMQN_PCA.png'), dpi=220)
        fig.savefig(os.path.join(FIGURES, f'{gse}_PMQN_PCA.svg'))
        plt.close(fig)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-raw', action='store_true')
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    all_out = {}
    for gse in ['GSE9825','GSE12632']:
        print(f'[geo] {gse}')
        all_out[gse] = analyze_dataset(gse, raw=not args.skip_raw)
        with open(os.path.join(RESULTS, f'{gse}_preprocessing.json'), 'w') as fh:
            json.dump(all_out[gse], fh, indent=2)
    with open(os.path.join(RESULTS, 'geo_preprocessing_summary.json'), 'w') as fh:
        json.dump(all_out, fh, indent=2)

    # compact comparison table
    rows=[]
    for gse, rec in all_out.items():
        for label, x in rec['series_matrix']['key_probes'].items():
            row={'dataset':gse,'gene_activity':label,'probe':x['probe']}
            for k,v in x.items():
                if isinstance(v,(float,int)):
                    row['GCOS_'+k]=v
            if 'raw_pm_qn_sensitivity' in rec and label in rec['raw_pm_qn_sensitivity']['key_probes']:
                for k,v in rec['raw_pm_qn_sensitivity']['key_probes'][label].items():
                    if isinstance(v,(float,int)):
                        row['PMQN_'+k]=v
            rows.append(row)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with open(os.path.join(RESULTS,'key_gene_fold_changes.csv'),'w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(rows)

if __name__ == '__main__':
    main()
