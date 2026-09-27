#!/usr/bin/env python3
"""Limited transcriptomic classification benchmark with leakage-safe cross-validation.

This is intentionally reported as a small-sample classification benchmark, not as a
validation score for the mechanistic ODE model.
"""
import argparse
import gzip
import json
import math
import os
import sys
import warnings
import numpy as np

warnings.filterwarnings('ignore')
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')); sys.path.insert(0,ROOT)
from penkin.paths import RESULTS_DIR, FIGURES_DIR
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

DATA=os.path.join(ROOT,'penkin','data','GSE9825_series_matrix_txt.gz')
OUT=os.path.join(RESULTS_DIR,'model_comparison.json')


def wilson(k,n,z=1.959963984540054):
    if n<=0:return [None,None]
    p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return [max(0.0,center-half),min(1.0,center+half)]


def metric(pred,y):
    pred=np.asarray(pred,int); y=np.asarray(y,int); k=int(np.sum(pred==y)); n=int(len(y)); ci=wilson(k,n)
    return {'n_correct':k,'n_total':n,'accuracy':k/n,'wilson_95_ci':ci}


def load_data():
    rows={}
    with gzip.open(DATA,'rt') as fh:
        block=False
        for line in fh:
            if line.startswith('!series_matrix_table_begin'): block=True; continue
            if line.startswith('!series_matrix_table_end'): break
            if block:
                p=line.rstrip('\n').split('\t')
                try: rows[p[0].strip('"')]=np.array([float(x) for x in p[1:]])
                except ValueError: pass
    M=np.array([rows[p] for p in rows]).T
    probe={'ACVS':'Pc21g21390_s_at','IPNS':'Pc21g21380_at','IAT':'Pc21g21370_at'}
    X3=np.array([rows[probe[e]] for e in probe]).T
    Xg=np.log2(M+1)
    y=np.array([1]*7+[0]*6)  # high-producing strain vs low-producing strain
    # Four biological condition groups: high-PAA-, high-PAA+, low-PAA-, low-PAA+.
    groups=np.array([0,0,0,1,1,1,1,2,2,2,3,3,3])
    return X3,Xg,y,groups,probe


def cv_predictions(model_fn,X,y,groups=None):
    splitter=LeaveOneGroupOut() if groups is not None else LeaveOneOut()
    it=splitter.split(X,y,groups) if groups is not None else splitter.split(X,y)
    pred=np.zeros(len(y),dtype=int)
    folds=[]
    for fold_id,(tr,te) in enumerate(it):
        # Scaling is fit only on each training fold to prevent information leakage.
        sc=StandardScaler().fit(X[tr]); model=model_fn(); model.fit(sc.transform(X[tr]),y[tr]); pred[te]=model.predict(sc.transform(X[te]))
        folds.append({'fold':fold_id,'train_indices':tr.tolist(),'test_indices':te.tolist()})
    return pred,folds


def main(seed=0):
    X3,Xg,y,groups,probe=load_data()
    models={
        'logistic_regression':lambda: LogisticRegression(max_iter=1000,random_state=seed),
        'random_forest':lambda: RandomForestClassifier(n_estimators=200,random_state=seed),
        'mlp':lambda: MLPClassifier(hidden_layer_sizes=(64,32),max_iter=2000,random_state=seed),
    }
    out={
        'task':'limited producer-status classification benchmark',
        'interpretation':'Exploratory small-n transcriptomic classification only; not an accuracy estimate for the mechanistic ODE model.',
        'n_samples':int(len(y)),
        'class_counts':{'high':int(np.sum(y==1)),'low':int(np.sum(y==0))},
        'features':{'3_pathway':list(probe.values()),'genome_wide':'all 15,531 deposited probe-set rows after log2(signal+1)'},
        'preprocessing':'StandardScaler is fitted within each training fold only. Genome-wide values are log2(signal+1); pathway-probe values retain deposited linear signal before fold-wise scaling.',
        'random_seed':seed,
        'fold_definitions':{
            'leave_one_sample_out':'13 folds; one deposited array held out per fold',
            'leave_one_condition_out':'4 folds defined by high/low strain × PAA-/PAA+ biological condition groups; all arrays in one condition are held out together',
        },
        'uncertainty':'Wilson 95% confidence intervals are reported for sample-level accuracy (n=13); these intervals remain wide because the dataset is small.',
        'models':{},
    }
    fold_manifest={}
    for name,fn in models.items():
        out['models'][name]={}
        for feat,X in [('3_pathway',X3),('genome_wide',Xg)]:
            p1,f1=cv_predictions(fn,X,y,None); p2,f2=cv_predictions(fn,X,y,groups)
            sc=StandardScaler().fit(X); m_all=fn(); m_all.fit(sc.transform(X),y)
            out['models'][name][feat]={
                'training':metric(m_all.predict(sc.transform(X)).astype(int),y),
                'leave_one_sample_out':metric(p1,y),
                'leave_one_condition_out':metric(p2,y),
            }
            fold_manifest.setdefault('leave_one_sample_out',f1)
            fold_manifest.setdefault('leave_one_condition_out',f2)
    out['fold_manifest']=fold_manifest
    with open(OUT,'w') as fh: json.dump(out,fh,indent=2)
    plot(out)
    print(OUT); return out


def plot(out):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[('logistic_regression','Logistic\nregression'),('random_forest','Random\nforest'),('mlp','MLP')]
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,(feat,title) in zip(axes,[('3_pathway','a  3 pathway probe sets'),('genome_wide','b  Genome-wide (15,531 probe sets)')]):
        x=np.arange(len(names)); w=0.35
        for off,key,lab in [(-w/2,'leave_one_sample_out','LOSO-CV'),(w/2,'leave_one_condition_out','LOCO-CV')]:
            vals=[out['models'][m][feat][key]['accuracy'] for m,_ in names]
            bars=ax.bar(x+off,vals,w,label=lab)
            for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.01,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([n for _,n in names]); ax.set_ylim(0,1.08)
        ax.set_title(title,loc='left',fontweight='bold')
    axes[0].set_ylabel('Classification accuracy'); axes[0].legend(frameon=False,ncol=2,loc='upper right',bbox_to_anchor=(1.0,1.12))
    fig.tight_layout()
    for ext in ('png','svg'):
        fig.savefig(os.path.join(FIGURES_DIR,'FigureS_model_comparison.'+ext),dpi=300 if ext=='png' else None,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=0); a=ap.parse_args(); main(a.seed)
