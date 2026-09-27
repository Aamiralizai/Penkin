#!/usr/bin/env python3
import copy
import csv
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from penkin.paths import RESULTS_DIR, FIGURES_DIR

from penkin.ensemble import fit_ensemble
from penkin.model import ENZYMES
from penkin.variants import steady_variant

OUT = os.path.join(RESULTS_DIR, 'structural.json')
RAW = os.path.join(RESULTS_DIR, 'structural_full_distributions.csv')
SUMMARY = os.path.join(RESULTS_DIR, 'structural_summary.csv')
VARIANTS = ['legacy_feedforward', 'reversible_IPNS', 'PCL_no_feedback', 'ACVS_classical_Ki', 'IAH_classical_4mM', 'combined']
ALL_KINDS = ['baseline'] + VARIANTS


def coefficient(kind, p, y, enzyme, d=1e-3):
    par = ENZYMES[enzyme]
    pp, pm = copy.deepcopy(p), copy.deepcopy(p)
    pp[par] *= 1.0 + d
    pm[par] *= 1.0 - d
    yp, fp = steady_variant(5.0, pp, y, kind=kind)
    ym, fm = steady_variant(5.0, pm, y, kind=kind)
    if yp is None or ym is None or fp['secr'] <= 0 or fm['secr'] <= 0:
        return None
    return float((np.log(fp['secr'])-np.log(fm['secr'])) /
                 (np.log(1.0+d)-np.log(1.0-d)))


def evaluate(kind, p, y_start=None):
    y, f = steady_variant(5.0, p, y_start, kind=kind)
    if y is None or f['secr'] <= 0:
        return None
    c = {e: coefficient(kind, p, y, e) for e in ENZYMES}
    if any(v is None for v in c.values()):
        return None
    J0 = f['secr']
    effects = {}
    for fold in (0.5, 2.0):
        q = copy.deepcopy(p); q['Vmax_IPNS'] *= fold
        yy, ff = steady_variant(5.0, q, y, kind=kind)
        if yy is None or ff['secr'] <= 0:
            return None
        effects[str(fold)] = {
            'flux_ratio': float(ff['secr']/J0),
            'ACV_ratio': float(yy[0]/y[0]) if y[0] > 0 else np.nan,
        }
    return {'control': c, 'ACV': float(y[0]), 'IPNS_effect': effects}


def summary(a):
    a=np.asarray(a,float); a=a[np.isfinite(a)]
    if not len(a): return {'n':0,'median':None,'q05':None,'q25':None,'q75':None,'q95':None}
    return {'n':int(len(a)), 'median':float(np.median(a)), 'q05':float(np.percentile(a,5)),
            'q25':float(np.percentile(a,25)), 'q75':float(np.percentile(a,75)), 'q95':float(np.percentile(a,95))}


def summarize_records(records):
    controls={e:[r['control'][e] for r in records] for e in ENZYMES}
    top=[max(r['control'], key=r['control'].get) for r in records]
    return {
        'n':len(records),
        'control_summary':{e:summary(controls[e]) for e in ENZYMES},
        'top_fraction':{e:float(np.mean(np.asarray(top)==e)) if records else None for e in ENZYMES},
        'ACV_summary':summary([r['ACV'] for r in records]),
        'IPNS_0.5x_flux_summary':summary([r['IPNS_effect']['0.5']['flux_ratio'] for r in records]),
        'IPNS_0.5x_ACV_summary':summary([r['IPNS_effect']['0.5']['ACV_ratio'] for r in records]),
        'IPNS_2x_flux_summary':summary([r['IPNS_effect']['2.0']['flux_ratio'] for r in records]),
        'IPNS_2x_ACV_summary':summary([r['IPNS_effect']['2.0']['ACV_ratio'] for r in records]),
    }


def flat_row(model_id, kind, rec, evaluable=True):
    row = {'model_id': model_id, 'variant': kind, 'evaluable': int(bool(evaluable))}
    for e in ENZYMES:
        row[f'control_{e}'] = rec['control'][e] if rec is not None else np.nan
    row.update({
        'ACV_mM': rec['ACV'] if rec is not None else np.nan,
        'IPNS_0.5x_flux_ratio': rec['IPNS_effect']['0.5']['flux_ratio'] if rec is not None else np.nan,
        'IPNS_0.5x_ACV_ratio': rec['IPNS_effect']['0.5']['ACV_ratio'] if rec is not None else np.nan,
        'IPNS_2x_flux_ratio': rec['IPNS_effect']['2.0']['flux_ratio'] if rec is not None else np.nan,
        'IPNS_2x_ACV_ratio': rec['IPNS_effect']['2.0']['ACV_ratio'] if rec is not None else np.nan,
    })
    return row


# Manuscript configuration. Every structural variant uses the same 300 parameter
# vectors sampled from the 8,000-candidate canonical ensemble.
MANUSCRIPT_N_TRY = 8000
MANUSCRIPT_N_SUB = 300


def main(n_try=MANUSCRIPT_N_TRY, n_sub=MANUSCRIPT_N_SUB, seed=1):
    accepted=fit_ensemble(n_try=n_try, seed=7, verbose=False)
    rng=np.random.default_rng(seed)
    idx=rng.choice(len(accepted), min(n_sub,len(accepted)), replace=False)
    cohort=[accepted[i] for i in idx]

    baseline=[]
    raw_rows=[]
    for model_id, r in enumerate(cohort):
        rec=evaluate('baseline', r['p'], r['y_lo'])
        raw_rows.append(flat_row(model_id, 'baseline', rec, rec is not None))
        if rec is not None:
            baseline.append((model_id, r, rec))

    out={'sampling':{'n_candidates':n_try,'n_accepted':len(accepted),'canonical_sample':len(cohort),
                     'baseline_evaluable':len(baseline),'ensemble_seed':7,'sample_seed':seed,
                     'design':'All structural comparisons start from the same acceptance-filtered baseline ensemble and the same sampled parameter vectors used by the main MCA design.'},
         'baseline':summarize_records([rec for _,_,rec in baseline])}

    for kind in VARIANTS:
        paired_base=[]; paired_var=[]
        deltas={e:[] for e in ENZYMES}
        base_by_id = {mid:brec for mid,_,brec in baseline}
        record_by_id = {}
        for model_id, r, brec in baseline:
            vrec=evaluate(kind, r['p'], r['y_lo'])
            record_by_id[model_id]=vrec
            if vrec is None: continue
            paired_base.append(brec); paired_var.append(vrec)
            for e in ENZYMES: deltas[e].append(vrec['control'][e]-brec['control'][e])
        for model_id in range(len(cohort)):
            vrec = record_by_id.get(model_id)
            raw_rows.append(flat_row(model_id, kind, vrec, vrec is not None))
        out[kind]={
            'paired_n':len(paired_var),
            'paired_fraction_of_baseline':len(paired_var)/len(baseline) if baseline else 0,
            'baseline_on_pairs':summarize_records(paired_base),
            'variant_on_pairs':summarize_records(paired_var),
            'paired_control_delta':{e:summary(deltas[e]) for e in ENZYMES},
        }
    out['notes']={
        'legacy_feedforward':'ACV product feedback removed from ACVS.',
        'reversible_IPNS':'Hypothetical reverse-IPNS stress test; reverse coefficient is not a fitted literature constant.',
        'PCL_no_feedback':'PA-CoA feedback removed from the baseline PCL law.',
        'ACVS_classical_Ki':'Historical Nielsen-style ACV-feedback constant fixed at 12.5 mM.',
        'IAH_classical_4mM':'Historical Pissarra-style IAH IPN affinity fixed at 4 mM.',
        'combined':'Reverse-IPNS stress test plus removal of PCL feedback.',
        'interpretation':'Use paired vectors for structural effects. The accompanying CSV exposes full per-model control, ACV, and IPNS-response distributions rather than only medians.'
    }
    with open(OUT,'w') as fh: json.dump(out,fh,indent=2)

    fields=['model_id','variant','evaluable']+[f'control_{e}' for e in ENZYMES]+[
        'ACV_mM','IPNS_0.5x_flux_ratio','IPNS_0.5x_ACV_ratio','IPNS_2x_flux_ratio','IPNS_2x_ACV_ratio']
    with open(RAW,'w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(raw_rows)

    # Compact summary table for manuscript/SI transfer.
    summary_rows=[]
    def add_summary(kind, block, paired_n, frac):
        row={'variant':kind,'n_evaluable':paired_n,'fraction_evaluable':frac}
        for e in ENZYMES:
            s=block['control_summary'][e]
            row[f'{e}_control_median']=s['median']; row[f'{e}_control_q05']=s['q05']; row[f'{e}_control_q95']=s['q95']
        for key in ['ACV_summary','IPNS_0.5x_flux_summary','IPNS_0.5x_ACV_summary','IPNS_2x_flux_summary','IPNS_2x_ACV_summary']:
            row[f'{key}_median']=block[key]['median']; row[f'{key}_q05']=block[key]['q05']; row[f'{key}_q95']=block[key]['q95']
        summary_rows.append(row)
    add_summary('baseline',out['baseline'],out['baseline']['n'],1.0)
    for kind in VARIANTS:
        add_summary(kind,out[kind]['variant_on_pairs'],out[kind]['paired_n'],out[kind]['paired_fraction_of_baseline'])
    keys=[]
    for r in summary_rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(SUMMARY,'w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=keys); w.writeheader(); w.writerows(summary_rows)

    print(OUT); print(RAW); print(SUMMARY)
    return out

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--n-try',type=int,default=MANUSCRIPT_N_TRY); ap.add_argument('--n-sub',type=int,default=MANUSCRIPT_N_SUB); ap.add_argument('--seed',type=int,default=1)
    a=ap.parse_args(); main(a.n_try,a.n_sub,a.seed)
