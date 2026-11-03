#!/usr/bin/env python3
"""Audit every canonical ensemble-admission criterion and prior-vs-accepted distributions."""
import argparse
import csv
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')); sys.path.insert(0,ROOT)
from penkin.paths import RESULTS_DIR, FIGURES_DIR
from penkin import plotstyle
plotstyle.apply()
from penkin.ensemble import FREE, Y0, PAA_REFERENCE, MIN_REFERENCE_SECRETION, MAX_INTERMEDIATE_MM, sample_candidate
from penkin.model import DEFAULT_PARAMS, steady_state
from penkin.variants import steady_variant

OUT=os.path.join(RESULTS_DIR,'acceptance_audit.json')
CSV=os.path.join(RESULTS_DIR,'acceptance_prior_vs_accepted.csv')
FIG=os.path.join(FIGURES_DIR,'FigureS_acceptance_audit')


def summ(a):
    a=np.asarray(a,float); a=a[np.isfinite(a)]
    if not len(a): return {'n':0,'median':None,'q05':None,'q25':None,'q75':None,'q95':None}
    return {'n':int(len(a)),'median':float(np.median(a)),'q05':float(np.percentile(a,5)),
            'q25':float(np.percentile(a,25)),'q75':float(np.percentile(a,75)),'q95':float(np.percentile(a,95))}


def main(n_try=8000,seed=7):
    rng=np.random.default_rng(seed)
    rows=[]; counts={'sampled':0,'reference_solver_converged':0,'secretion_threshold_pass':0,'intermediate_bound_pass':0,'accepted':0}
    fail={'reference_solver':0,'secretion_threshold':0,'intermediate_bound':0}
    prior={k:[] for k in FREE}; acc={k:[] for k in FREE}
    # Same candidates screened under the feed-forward architecture (ACV product
    # feedback removed), to show that rejection there is architectural.
    ff={'sampled':0,'reference_solver_converged':0,'secretion_threshold_pass':0,'intermediate_bound_pass':0,'accepted':0}
    for i in range(n_try):
        p=sample_candidate(rng); counts['sampled']+=1
        factors={k:p[k]/DEFAULT_PARAMS[k] for k in FREE}
        for k,v in factors.items(): prior[k].append(v)
        ff['sampled']+=1
        yf,fff=steady_variant(PAA_REFERENCE,p,Y0,kind='legacy_feedforward')
        if yf is not None and fff is not None:
            ff['reference_solver_converged']+=1
            if fff['secr']>=MIN_REFERENCE_SECRETION:
                ff['secretion_threshold_pass']+=1
                if not np.any(yf[:4]>MAX_INTERMEDIATE_MM):
                    ff['intermediate_bound_pass']+=1; ff['accepted']+=1
        y5,f5=steady_state(PAA_REFERENCE,p,Y0)
        stage='accepted'; accepted=False; J=np.nan; maxint=np.nan
        if y5 is None or f5 is None:
            fail['reference_solver']+=1; stage='reference_solver'
        else:
            counts['reference_solver_converged']+=1; J=float(f5['secr']); maxint=float(np.max(y5[:4]))
            if J < MIN_REFERENCE_SECRETION:
                fail['secretion_threshold']+=1; stage='secretion_threshold'
            else:
                counts['secretion_threshold_pass']+=1
                if np.any(y5[:4] > MAX_INTERMEDIATE_MM):
                    fail['intermediate_bound']+=1; stage='intermediate_bound'
                else:
                    counts['intermediate_bound_pass']+=1; counts['accepted']+=1; accepted=True
                    for k,v in factors.items(): acc[k].append(v)
        row={'candidate_id':i,'accepted':int(accepted),'fail_stage':stage,'reference_secretion':J,'max_first4_intermediate_mM':maxint}
        row.update({f'factor_{k}':v for k,v in factors.items()}); rows.append(row)

    out={
        'sampling':{'n_candidates':n_try,'seed':seed,'reference_PAA_mM':PAA_REFERENCE},
        'criteria':{
            'reference_solver':'steady-state solve must converge',
            'secretion_threshold':f'PenG secretion >= {MIN_REFERENCE_SECRETION:g}',
            'intermediate_bound':f'first four modeled intermediates <= {MAX_INTERMEDIATE_MM:g} mM',
            'PAA_zero_diagnostic':'not an admission filter because baseline PCL is first-order in PAA and therefore structurally zero at PAA=0',
        },
        'stage_counts':counts,
        'failure_attribution':fail,
        'feedforward_stage_counts':ff,
        'prior_summary':{k:summ(v) for k,v in prior.items()},
        'accepted_summary':{k:summ(v) for k,v in acc.items()},
        'interpretation':'This audit quantifies the effect of each admission criterion. If all candidates pass, prior and accepted distributions coincide; this demonstrates that the reference acceptance criteria are permissive rather than data-fitting filters.'
    }
    os.makedirs(os.path.dirname(OUT),exist_ok=True); os.makedirs(os.path.dirname(FIG),exist_ok=True)
    with open(OUT,'w') as fh: json.dump(out,fh,indent=2)
    fields=['candidate_id','accepted','fail_stage','reference_secretion','max_first4_intermediate_mM']+[f'factor_{k}' for k in FREE]
    with open(CSV,'w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(rows)

    # A: criterion attrition. B-M: prior vs accepted ECDFs for every free parameter.
    fig,axes=plt.subplots(7,2,figsize=(plotstyle.WIDTH,16.5)); axes=axes.ravel()
    stages=['sampled','reference_solver_converged','secretion_threshold_pass','intermediate_bound_pass','accepted']
    vals=[counts[s] for s in stages]
    axes[0].bar(range(len(stages)),vals); axes[0].set_xticks(range(len(stages))); axes[0].set_xticklabels(['sampled','solver','secretion','bounds','accepted'],rotation=35,ha='right',fontsize=7)
    axes[0].set_ylabel('candidate count'); axes[0].set_ylim(0,max(vals)*1.15); axes[0].set_title('a  Admission-criterion attrition',loc='left',fontweight='bold')
    for i,v in enumerate(vals): axes[0].text(i,v,f'{v}',ha='center',va='bottom',fontsize=7)
    for j,k in enumerate(FREE, start=1):
        a=np.sort(np.asarray(prior[k])); b=np.sort(np.asarray(acc[k]));
        axes[j].plot(a,np.arange(1,len(a)+1)/len(a),label='prior')
        if len(b): axes[j].plot(b,np.arange(1,len(b)+1)/len(b),ls='--',label='accepted')
        axes[j].set_xscale('log'); axes[j].set_xlabel('factor vs reference',fontsize=7); axes[j].set_ylabel('ECDF',fontsize=7); axes[j].tick_params(labelsize=7)
        axes[j].set_title(f'{chr(97+j)}  {k}',loc='left',fontweight='bold',fontsize=9)
        if j==1: axes[j].legend(frameon=False,fontsize=7)
    for j in range(1+len(FREE),len(axes)): axes[j].axis('off')
    fig.tight_layout()
    for ext in ['png','svg']: fig.savefig(FIG+'.'+ext,dpi=300 if ext=='png' else None,bbox_inches='tight')
    plt.close(fig)
    print(OUT); print(CSV); print(FIG+'.png')
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--n-try',type=int,default=8000); ap.add_argument('--seed',type=int,default=7); a=ap.parse_args(); main(a.n_try,a.seed)
