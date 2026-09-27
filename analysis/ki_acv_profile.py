#!/usr/bin/env python3
import copy
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
from penkin.ensemble import FREE, MIN_REFERENCE_SECRETION, MAX_INTERMEDIATE_MM
from penkin.model import DEFAULT_PARAMS, POOLS, steady_state, scale_expression
from penkin.variants import steady_variant, scale_cluster

DATA = os.path.join(ROOT, 'penkin', 'data')
RESULTS = RESULTS_DIR
FIGURES = FIGURES_DIR
OUT_JSON = os.path.join(RESULTS, 'ki_acv_profile.json')
OUT_CSV = os.path.join(RESULTS, 'ki_acv_profile.csv')
OUT_PNG = os.path.join(FIGURES, 'FigureS_Ki_ACV_profile.png')
OUT_SVG = os.path.join(FIGURES, 'FigureS_Ki_ACV_profile.svg')


def read_csv(name):
    with open(os.path.join(DATA, name), newline='') as fh:
        return list(csv.DictReader(fh))


def sample_without_ki(rng):
    p = copy.deepcopy(DEFAULT_PARAMS)
    for k, (lo, hi) in FREE.items():
        if k == 'Ki_ACV_ACVS':
            continue
        p[k] = DEFAULT_PARAMS[k] * np.exp(rng.uniform(np.log(lo), np.log(hi)))
    p['Ki_ACV_ACVS'] = DEFAULT_PARAMS['Ki_ACV_ACVS']
    return p


def accepted(p):
    y, f = steady_state(5.0, p)
    if y is None or f['secr'] < MIN_REFERENCE_SECRETION:
        return None
    if np.any(np.asarray(y[:4]) > MAX_INTERMEDIATE_MM):
        return None
    return {'p': p, 'y_lo': list(y), 'J_lo': float(f['secr'])}


def build_cohort(n_try=1000, n_sub=100, seed=73):
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(n_try):
        r = accepted(sample_without_ki(rng))
        if r is not None:
            acc.append(r)
    if not acc:
        return [], 0
    idx = rng.choice(len(acc), min(n_sub, len(acc)), replace=False)
    return [acc[i] for i in idx], len(acc)


def control_coeff(p, y, par, d=1e-3):
    pp, pm = copy.deepcopy(p), copy.deepcopy(p)
    pp[par] *= 1.0 + d
    pm[par] *= 1.0 - d
    yp, fp = steady_state(5.0, pp, y)
    ym, fm = steady_state(5.0, pm, y)
    if yp is None or ym is None or fp['secr'] <= 0 or fm['secr'] <= 0:
        return np.nan
    return float((np.log(fp['secr']) - np.log(fm['secr'])) /
                 (np.log(1.0+d) - np.log(1.0-d)))


def summarize(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0, 'median': None, 'q25': None, 'q75': None}
    return {'n': int(len(a)), 'median': float(np.median(a)),
            'q25': float(np.percentile(a,25)), 'q75': float(np.percentile(a,75))}


def janoska_rmse(records, expr):
    rows = read_csv('janoska2022_oxygen_summary.csv')
    do = [float(r['DO_mM']) for r in rows]
    obs = np.array([float(r['relative_penicillin_rate']) for r in rows])
    pred = {x: [] for x in do}
    acv013, ipn013 = [], []
    for r in records:
        ph = scale_expression(r['p'], expr, couple_pende=True)
        pref = dict(POOLS); pref['O2'] = 0.136
        y0, f0 = steady_variant(5.0, ph, r['y_lo'], pools=pref, kind='baseline')
        if y0 is None or f0['secr'] <= 0:
            continue
        for x in do:
            pools = dict(POOLS); pools['O2'] = x
            y, f = steady_variant(5.0, ph, y0, pools=pools, kind='baseline')
            if y is None or f['secr'] < 0:
                continue
            pred[x].append(f['secr']/f0['secr'])
            if x == 0.013:
                acv013.append(y[0]/y0[0])
                ipn013.append(y[2]/y0[2])
    med = np.array([np.median(pred[x]) if pred[x] else np.nan for x in do])
    ok = np.isfinite(med)
    rmse = float(np.sqrt(np.mean((med[ok]-obs[ok])**2))) if ok.any() else np.nan
    return rmse, float(np.mean(np.asarray(acv013)>1)) if acv013 else np.nan, float(np.mean(np.asarray(ipn013)<1)) if ipn013 else np.nan


def theilgaard_rmse(records):
    rows = read_csv('theilgaard2001_exact_tableIII.csv')
    ref = rows[0]
    ref_a = float(ref['ACVS_nkat_gprot']); ref_i = float(ref['IPNS_ukat_gprot']); ref_d = float(ref['penDE_copy'])
    obs, preds = [], []
    for row in rows[1:]:
        a = float(row['ACVS_nkat_gprot'])/ref_a
        i = float(row['IPNS_ukat_gprot'])/ref_i
        d = float(row['penDE_copy'])/ref_d
        vals=[]
        for r in records:
            y0, f0 = steady_variant(5.0, r['p'], r['y_lo'], kind='baseline')
            if y0 is None or f0['secr'] <= 0:
                continue
            q = scale_cluster(r['p'], acvs=a, ipns=i, pende=d, pcl=1.0)
            y, f = steady_variant(5.0, q, y0, kind='baseline')
            if y is not None and f['secr'] >= 0:
                vals.append(f['secr']/f0['secr'])
        if vals:
            obs.append(float(row['rp_relative'])); preds.append(float(np.median(vals)))
    obs=np.asarray(obs); preds=np.asarray(preds)
    if not len(obs):
        return np.nan, np.nan
    rmse=float(np.sqrt(np.mean((preds-obs)**2)))
    rho=float(spearmanr(obs,preds).statistic) if len(obs)>=3 else np.nan
    return rmse, rho


def nijland_spearman(records):
    rows=[r for r in read_csv('nijland2010_digitized.csv') if int(float(r['copy_number']))>=1]
    copies=[int(float(r['copy_number'])) for r in rows]
    exp=np.array([float(r['penicillin_V_mmol_gDW']) for r in rows]); exp=exp/exp[0]
    pred={c:[] for c in copies}
    for r in records:
        y0,f0=steady_variant(5.0,r['p'],r['y_lo'],kind='baseline')
        if y0 is None or f0['secr']<=0: continue
        for row,c in zip(rows,copies):
            a=float(row['ACVS_protein_pct'])/float(rows[0]['ACVS_protein_pct'])
            i=float(row['IPNS_protein_pct'])/float(rows[0]['IPNS_protein_pct'])
            d=float(row['IAT_protein_pct'])/float(rows[0]['IAT_protein_pct'])
            pcl=float(row['PCL_protein_pct'])/float(rows[0]['PCL_protein_pct'])
            q=scale_cluster(r['p'],acvs=a,ipns=i,pende=d,pcl=pcl)
            y,f=steady_variant(5.0,q,y0,kind='baseline')
            if y is not None and f['secr']>0: pred[c].append(f['secr']/f0['secr'])
    med=np.array([np.median(pred[c]) if pred[c] else np.nan for c in copies])
    ok=np.isfinite(med)
    return float(spearmanr(exp[ok],med[ok]).statistic) if ok.sum()>=3 else np.nan


def gse_ratio(records, expr):
    vals=[]
    for r in records:
        ph=scale_expression(r['p'],expr,couple_pende=True)
        y,f=steady_state(5.0,ph,r['y_lo'])
        if y is not None and f['secr']>0: vals.append(f['secr']/r['J_lo'])
    return summarize(vals)


def profile(n_try=1000, n_sub=100, seed=73, grid=None):
    if grid is None:
        grid=[0.16,0.20,0.24,0.28,0.32,0.36,0.40,0.44,0.48,0.52,0.54,0.56,0.60,0.64,0.68,0.72]
    cohort,nacc=build_cohort(n_try,n_sub,seed)
    expr,_=load_expression()
    rows=[]
    details={}
    for ki in grid:
        records=[]
        acvs=[]; ipns=[]
        for r0 in cohort:
            p=copy.deepcopy(r0['p']); p['Ki_ACV_ACVS']=float(ki)
            r=accepted(p)
            if r is None: continue
            records.append(r)
            y=np.asarray(r['y_lo'],float)
            acvs.append(control_coeff(p,y,'Vmax_ACVS'))
            ipns.append(control_coeff(p,y,'Vmax_IPNS'))
        jrmse,acvfrac,ipnfrac=janoska_rmse(records,expr)
        trmse,trho=theilgaard_rmse(records)
        nrho=nijland_spearman(records)
        gse=gse_ratio(records,expr)
        a=summarize(acvs); i=summarize(ipns)
        d={'Ki_ACV_mM':float(ki),'n_valid':len(records),'valid_fraction':len(records)/len(cohort) if cohort else 0,
           'ACVS_control':a,'IPNS_control':i,'Janoska_RMSE':jrmse,'Janoska_ACV_increase_fraction_0.013':acvfrac,
           'Janoska_IPN_decrease_fraction_0.013':ipnfrac,'Theilgaard_RMSE':trmse,'Theilgaard_Spearman':trho,
           'Nijland_Spearman_protein_informed':nrho,'GSE9825_high_low_ratio':gse}
        details[str(ki)]=d
        rows.append(d)
    finite=[r for r in rows if np.isfinite(r['Janoska_RMSE'])]
    best=min(finite,key=lambda r:r['Janoska_RMSE']) if finite else None
    minrmse=best['Janoska_RMSE'] if best else np.nan
    near=[r['Ki_ACV_mM'] for r in finite if r['Janoska_RMSE'] <= minrmse*1.10] if minrmse>0 else []
    out={'design':{'n_try':n_try,'n_accepted_at_0.44':nacc,'n_profiled':len(cohort),'seed':seed,
                   'grid_mM':grid,'calibration_dataset':'Janoska oxygen perturbation only',
                   'holdout_diagnostics':['Theilgaard 2001','Nijland 2010'],
                   'nonindependent_consistency':'GSE9825 producer scaling',
                   'note':'The same non-Ki parameter vectors are used at every Ki value. Ki is profiled, not jointly refitted.'},
         'best_by_Janoska_RMSE':best,
         'near_optimal_10pct_Ki_mM':near,
         'literature_reference':{'Deshmukh_2015_mM':0.44,'Deshmukh_SD_mM':0.28,'earlier_report_mM':0.54},
         'profile':details}
    return out,rows


def make_figure(out, rows):
    x=np.array([r['Ki_ACV_mM'] for r in rows])
    j=np.array([r['Janoska_RMSE'] for r in rows])
    t=np.array([r['Theilgaard_RMSE'] for r in rows])
    n=np.array([r['Nijland_Spearman_protein_informed'] for r in rows])
    ac=np.array([r['ACVS_control']['median'] for r in rows])
    ip=np.array([r['IPNS_control']['median'] for r in rows])
    valid=np.array([r['valid_fraction'] for r in rows])
    fig,ax=plt.subplots(2,2,figsize=(10.5,7.8))
    ax[0,0].plot(x,j,marker='o',label='Janoska RMSE (calibration)')
    ax[0,0].plot(x,t,marker='s',label='Theilgaard RMSE (holdout diagnostic)')
    ax[0,0].axvline(0.44,ls='--',lw=.9,label='Deshmukh 0.44 mM')
    ax[0,0].axvline(0.54,ls=':',lw=.9,label='Earlier 0.54 mM')
    ax[0,0].set_xlabel('ACV-feedback Ki (mM)'); ax[0,0].set_ylabel('RMSE'); ax[0,0].set_title('A  External-fit profile',loc='left',fontweight='bold'); ax[0,0].legend(frameon=False,fontsize=8)
    ax[0,1].plot(x,ac,marker='o',label='ACVS')
    ax[0,1].plot(x,ip,marker='s',label='IPNS')
    ax[0,1].axvspan(0.44,0.54,alpha=.12)
    ax[0,1].set_xlabel('ACV-feedback Ki (mM)'); ax[0,1].set_ylabel('Median flux-control coefficient'); ax[0,1].set_title('B  Control redistribution',loc='left',fontweight='bold'); ax[0,1].legend(frameon=False)
    ax[1,0].plot(x,n,marker='o')
    ax[1,0].axvspan(0.44,0.54,alpha=.12)
    ax[1,0].set_xlabel('ACV-feedback Ki (mM)'); ax[1,0].set_ylabel('Spearman rho'); ax[1,0].set_ylim(-1.05,1.05); ax[1,0].set_title('C  Nijland holdout rank agreement',loc='left',fontweight='bold')
    ax[1,1].plot(x,valid,marker='o')
    ax[1,1].axvspan(0.44,0.54,alpha=.12)
    ax[1,1].set_xlabel('ACV-feedback Ki (mM)'); ax[1,1].set_ylabel('Fraction of paired cohort valid'); ax[1,1].set_ylim(0,1.05); ax[1,1].set_title('D  Physiological validity',loc='left',fontweight='bold')
    fig.tight_layout(); fig.savefig(OUT_PNG,dpi=300,bbox_inches='tight'); fig.savefig(OUT_SVG,bbox_inches='tight'); plt.close(fig)


def main(n_try=1000,n_sub=100,seed=73):
    out,rows=profile(n_try,n_sub,seed)
    os.makedirs(RESULTS,exist_ok=True); os.makedirs(FIGURES,exist_ok=True)
    with open(OUT_JSON,'w') as fh: json.dump(out,fh,indent=2)
    with open(OUT_CSV,'w',newline='') as fh:
        fields=['Ki_ACV_mM','n_valid','valid_fraction','Janoska_RMSE','Theilgaard_RMSE','Theilgaard_Spearman','Nijland_Spearman_protein_informed']
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r[k] for k in fields})
    make_figure(out,rows)
    print(OUT_JSON); print(OUT_CSV); print(OUT_PNG)
    return out

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--n-try',type=int,default=1000)
    ap.add_argument('--n-sub',type=int,default=100)
    ap.add_argument('--seed',type=int,default=73)
    a=ap.parse_args(); main(a.n_try,a.n_sub,a.seed)
