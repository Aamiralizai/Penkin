#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from penkin.paths import RESULTS_DIR, FIGURES_DIR

from penkin.data_io import load_expression
from penkin.ensemble import fit_ensemble, predict_producer_ratio
from penkin.variants import scale_cluster, steady_variant
try:
    from .external_validation_suite import validate_janoska, validate_nijland, summary
except ImportError:
    from external_validation_suite import validate_janoska, validate_nijland, summary

DATA = os.path.join(ROOT, 'penkin', 'data')
RESULTS = RESULTS_DIR
FIGURES = FIGURES_DIR


def read_csv(name):
    with open(os.path.join(DATA, name), newline='') as fh:
        return list(csv.DictReader(fh))


def sample_models(accepted, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(accepted), min(n, len(accepted)), replace=False)
    return [accepted[i] for i in idx]


def theilgaard_exact(sub):
    rows = read_csv('theilgaard2001_exact_tableIII.csv')
    ref = rows[0]
    ref_acvs = float(ref['ACVS_nkat_gprot'])
    ref_ipns = float(ref['IPNS_ukat_gprot'])
    transforms = rows[1:]
    out = {}
    for kind in ['baseline', 'legacy_feedforward', 'reversible_IPNS']:
        mode = {}
        for mapping in ['measured_ACVS_IPNS', 'measured_plus_penDE_copy_proxy']:
            pred_by_strain = {}
            med = []
            obs = []
            for row in transforms:
                a = float(row['ACVS_nkat_gprot']) / ref_acvs
                i = float(row['IPNS_ukat_gprot']) / ref_ipns
                d = 1.0 if mapping == 'measured_ACVS_IPNS' else float(row['penDE_copy']) / float(ref['penDE_copy'])
                vals = []
                acv_vals = []
                for r in sub:
                    y0, f0 = steady_variant(5.0, r['p'], r['y_lo'], kind=kind)
                    if y0 is None or f0['secr'] <= 0 or y0[0] <= 0:
                        continue
                    q = scale_cluster(r['p'], acvs=a, ipns=i, pende=d, pcl=1.0)
                    y, f = steady_variant(5.0, q, y0, kind=kind)
                    if y is not None and f['secr'] >= 0:
                        vals.append(f['secr'] / f0['secr'])
                        acv_vals.append(y[0] / y0[0])
                s = summary(vals)
                pred_by_strain[row['strain']] = {
                    'observed_rp_relative': float(row['rp_relative']),
                    'ACVS_activity_relative': a,
                    'IPNS_activity_relative': i,
                    'penDE_copy_relative_used': d,
                    'predicted_rp_relative': s,
                    'predicted_ACV_relative': summary(acv_vals),
                }
                if s['median'] is not None:
                    med.append(s['median']); obs.append(float(row['rp_relative']))
            med = np.asarray(med); obs = np.asarray(obs)
            rmse = float(np.sqrt(np.mean((med-obs)**2)))
            rho = float(spearmanr(obs, med).statistic) if len(obs) >= 3 else None
            mode[mapping] = {'strains': pred_by_strain, 'rmse': rmse, 'spearman': rho}
        out[kind] = mode
    return {
        'source': 'Theilgaard et al. 2001 Table III',
        'note': 'ACVS and IPNS are measured enzyme activities. penDE activity was not measured; the copy-proxy mapping is explicitly a sensitivity analysis.',
        'architectures': out,
    }


def gse9825_check(accepted, n_sub, seed):
    expr, reps = load_expression()
    pred, _ = predict_producer_ratio(accepted, expr, n_sub=n_sub, seed=seed, couple_pende=True)
    return {
        'expression_scaling': {k: float(v) for k,v in expr.items()},
        'predicted_high_low': summary(pred['ratios']),
        'fraction_high_gt_low': float(pred['frac_high_gt_low']),
    }


def context_sources():
    nas = read_csv('nasution2008_exact_summary.csv')
    har = read_csv('harris2009_exact_table2.csv')
    vg = read_csv('van_gulik2000_exact_example.csv')
    return {
        'Nasution_2008': {
            'rows': nas,
            'model_mapping': 'context/limitation only: Penkin v4 has no explicit ATP or NADPH state',
            'key_source_supported_interpretation': 'Penicillin flux tracked energy/redox status more clearly than alpha-AAA or valine across the reported steady-state perturbations.'
        },
        'Harris_2009_GSE12632': {
            'rows': har,
            'model_mapping': 'independent transcriptomic/physiological context; cluster deletion is structurally expected to abolish PenG in Penkin',
        },
        'van_Gulik_2000': {
            'example_flux_rows': vg,
            'model_mapping': 'context/limitation only: primary-metabolism and NADPH supply are outside the six-state Penkin core',
            'key_source_supported_interpretation': 'Large central-carbon flux redistribution across carbon sources had little effect on PenG productivity, whereas increased NADPH demand reduced production.'
        }
    }


LABEL = {'baseline': 'Baseline', 'legacy_feedforward': 'Feed-forward (no ACV feedback)', 'reversible_IPNS': 'Reversible IPNS'}


def make_figure(theil, jan, nij):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    # Theilgaard observed vs predicted, measured + copy proxy
    ax=axes[0]
    obs=[]
    for kind, marker in [('baseline','o'),('legacy_feedforward','s'),('reversible_IPNS','^')]:
        rec=theil['architectures'][kind]['measured_plus_penDE_copy_proxy']
        x=[]; y=[]
        for st,v in rec['strains'].items():
            x.append(v['observed_rp_relative']); y.append(v['predicted_rp_relative']['median'])
        ax.scatter(x,y,label=f'{LABEL[kind]} (RMSE {rec["rmse"]:.2f})',marker=marker)
        obs.extend(x)
    lim=[0.5,3.8]; ax.plot(lim,lim,ls='--',lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Theilgaard observed relative productivity'); ax.set_ylabel('Penkin predicted relative flux'); ax.set_title('A  Theilgaard 2001')
    ax.legend(fontsize=7)
    # Janoska
    ax=axes[1]
    do=np.array([float(k) for k in jan['experimental'].keys()]); exp=np.array([jan['experimental'][str(x)] for x in do])
    order=np.argsort(do); do=do[order]; exp=exp[order]
    ax.plot(do,exp,'o-',label='published trajectory (approx.)')
    for kind, marker in [('baseline','o'),('legacy_feedforward','s'),('reversible_IPNS','^')]:
        vals=np.array([jan['architectures'][kind]['penicillin_relative'][str(float(x))]['median'] for x in do])
        ax.plot(do,vals,marker=marker,label=f'{LABEL[kind]} (RMSE {jan["architectures"][kind]["rmse_vs_approx_published_endpoints"]:.2f})')
    ax.set_xscale('log'); ax.set_xlabel('Dissolved O$_2$ (mM)'); ax.set_ylabel('Relative penicillin rate'); ax.set_title('B  Janoska 2023'); ax.legend(fontsize=7)
    # Nijland
    ax=axes[2]
    copies=np.array(nij['copy_numbers']); exp=np.array(nij['experimental_penicillin_relative_to_1copy'])
    cp=np.array([nij['copy_proportional_prediction'][str(int(c))]['median'] for c in copies])
    pp=np.array([nij['protein_informed_prediction'][str(int(c))]['median'] for c in copies])
    ax.plot(copies,exp,'o-',label='experiment (digitized)'); ax.plot(copies,cp,'s-',label='copy-proportional'); ax.plot(copies,pp,'^-',label='protein-informed')
    ax.set_xlabel('Penicillin cluster copies'); ax.set_ylabel('Relative PenV'); ax.set_title('C  Nijland 2010'); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES,'FigureS_literature_validation.png'),dpi=240)
    fig.savefig(os.path.join(FIGURES,'FigureS_literature_validation.svg'))
    plt.close(fig)


# Manuscript configuration.
MANUSCRIPT_N_TRY = 2000
MANUSCRIPT_N_SUB = 250


def run_validation(n_try=MANUSCRIPT_N_TRY, n_sub=MANUSCRIPT_N_SUB, seed=41):
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    accepted = fit_ensemble(n_try=n_try, seed=seed)
    sub = sample_models(accepted, n_sub, seed)
    expr, _ = load_expression()
    theil = theilgaard_exact(sub)
    jan = validate_janoska(sub, expr)
    nij = validate_nijland(sub)
    gse = gse9825_check(accepted, n_sub, seed + 1)
    context = context_sources()
    out = {
        'run': {'n_try': n_try, 'accepted': len(accepted), 'n_sub': len(sub), 'seed': seed},
        'GSE9825': gse,
        'Theilgaard2001': theil,
        'Janoska2023': jan,
        'Nijland2010': nij,
        'context_sources': context,
    }
    with open(os.path.join(RESULTS, 'literature_validation.json'), 'w') as fh:
        json.dump(out, fh, indent=2)

    rows = []
    for kind in ['baseline', 'legacy_feedforward', 'reversible_IPNS']:
        for mapping in ['measured_ACVS_IPNS', 'measured_plus_penDE_copy_proxy']:
            rec = theil['architectures'][kind][mapping]
            rows.append({'dataset': 'Theilgaard2001', 'architecture': kind, 'mapping': mapping, 'metric': 'RMSE', 'value': rec['rmse']})
            rows.append({'dataset': 'Theilgaard2001', 'architecture': kind, 'mapping': mapping, 'metric': 'Spearman', 'value': rec['spearman']})
        rows.append({'dataset': 'Janoska2023', 'architecture': kind, 'mapping': 'oxygen', 'metric': 'RMSE', 'value': jan['architectures'][kind]['rmse_vs_approx_published_endpoints']})
    rows += [
        {'dataset': 'Nijland2010', 'architecture': 'baseline', 'mapping': 'copy_proportional', 'metric': 'Spearman', 'value': nij['spearman_penicillin_copy_proportional']},
        {'dataset': 'Nijland2010', 'architecture': 'baseline', 'mapping': 'protein_informed', 'metric': 'Spearman', 'value': nij['spearman_penicillin_protein_informed']},
        {'dataset': 'GSE9825', 'architecture': 'baseline', 'mapping': 'GCOS_expression', 'metric': 'median_high_low_flux', 'value': gse['predicted_high_low']['median']},
    ]
    with open(os.path.join(RESULTS, 'literature_validation_metrics.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['dataset', 'architecture', 'mapping', 'metric', 'value'])
        w.writeheader()
        w.writerows(rows)
    make_figure(theil, jan, nij)
    path = os.path.join(RESULTS, 'literature_validation.json')
    print(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-try', type=int, default=MANUSCRIPT_N_TRY)
    ap.add_argument('--n-sub', type=int, default=MANUSCRIPT_N_SUB)
    ap.add_argument('--seed', type=int, default=41)
    args = ap.parse_args()
    run_validation(args.n_try, args.n_sub, args.seed)


if __name__ == '__main__':
    main()
