#!/usr/bin/env python3
import argparse, copy, json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')); sys.path.insert(0,ROOT)
from penkin.paths import RESULTS_DIR, FIGURES_DIR
from penkin import plotstyle
plotstyle.apply()
from penkin import ensemble, data_io
from penkin.model import steady_state, scale_expression

OUT=os.path.join(RESULTS_DIR,'perturbation_design.json')
FIGPNG=os.path.join(FIGURES_DIR,'Figure5.png'); FIGSVG=os.path.join(FIGURES_DIR,'Figure5.svg')
ACTIONS=['ACVS','IPNS','PCL','PenDE']; FOLDS=np.array([0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0,10.0])

def summary(a):
    a=np.asarray(a,float); a=a[np.isfinite(a)]
    if not len(a): return {'n':0,'median':None,'q25':None,'q75':None}
    return {'n':int(len(a)),'median':float(np.median(a)),'q25':float(np.percentile(a,25)),'q75':float(np.percentile(a,75))}

def apply_action(p, action, fold):
    q=copy.deepcopy(p)
    if action=='PenDE': q['Vmax_IAT']*=fold; q['Vmax_IAH']*=fold
    else: q['Vmax_'+action]*=fold
    return q

def main(n_try=8000,n_sub=300,seed=29):
    acc=ensemble.fit_ensemble(n_try=n_try,seed=7,verbose=False)
    rng=np.random.default_rng(seed); idx=rng.choice(len(acc),min(n_sub,len(acc)),replace=False); cohort=[acc[i] for i in idx]
    expr,_=data_io.load_expression()
    states=[]
    for r in cohort:
        pL=r['p']; yL=np.asarray(r['y_lo'],float); JL=float(r['J_lo'])
        pH=scale_expression(pL,expr,couple_pende=True); yH,fH=steady_state(5.0,pH,yL)
        states.append((pL,yL,JL,pH,yH,float(fH['secr']) if fH is not None else np.nan))
    def scan(action,high=False):
        out={}
        for fo in FOLDS:
            vals=[]
            for pL,yL,JL,pH,yH,JH in states:
                p0,y0,J0=(pH,yH,JH) if high else (pL,yL,JL)
                if y0 is None or not np.isfinite(J0) or J0<=0: continue
                y,f=steady_state(5.0,apply_action(p0,action,float(fo)),y0)
                if y is not None and f is not None and f['secr']>0: vals.append(f['secr']/J0)
            out[str(float(fo))]=summary(vals)
        return out
    low={a:scan(a,False) for a in ACTIONS}; high={a:scan(a,True) for a in ACTIONS}
    pair={}; mat=np.full((4,4),np.nan); cnt=np.zeros((4,4),int)
    for i,a in enumerate(ACTIONS):
        pair[a]={}
        for j,b in enumerate(ACTIONS):
            vals=[]
            for pL,yL,JL,*_ in states:
                q=apply_action(pL,a,3.0)
                if b!=a: q=apply_action(q,b,3.0)
                y,f=steady_state(5.0,q,yL)
                if y is not None and f is not None and f['secr']>0: vals.append(f['secr']/JL)
            s=summary(vals); pair[a][b]=s; mat[i,j]=s['median']; cnt[i,j]=s['n']
    for i in range(4):
        for j in range(i+1,4):
            a,b=ACTIONS[i],ACTIONS[j]; vals=[]
            for pL,yL,JL,*_ in states:
                y,f=steady_state(5.0,apply_action(apply_action(pL,a,3.0),b,3.0),yL)
                if y is not None and f is not None and f['secr']>0: vals.append(f['secr']/JL)
            s=summary(vals); pair[a][b]=s; pair[b][a]=dict(s); mat[i,j]=mat[j,i]=s['median']; cnt[i,j]=cnt[j,i]=s['n']
    out={'n_candidates':n_try,'n_accepted':len(acc),'n_models_sampled':len(cohort),'ensemble_seed':7,'sample_seed':seed,
         'activity_folds':FOLDS.tolist(),'low_state_scans':low,'high_state_scans':high,'pairwise_3x_low_state':pair}
    with open(OUT,'w') as fh: json.dump(out,fh,indent=2)
    fig,ax=plt.subplots(2,2,figsize=plotstyle.size(2))
    for a in ACTIONS:
        med=np.array([low[a][str(float(x))]['median'] for x in FOLDS]); q1=np.array([low[a][str(float(x))]['q25'] for x in FOLDS]); q3=np.array([low[a][str(float(x))]['q75'] for x in FOLDS])
        ax[0,0].plot(FOLDS,med,marker='o',label=a); ax[0,0].fill_between(FOLDS,q1,q3,alpha=.14)
    ax[0,0].set_xscale('log'); ax[0,0].axhline(1,ls='--',lw=.8); ax[0,0].set_xlabel('Activity scaling'); ax[0,0].set_ylabel('PenG flux / reference'); ax[0,0].set_title('a  Low-producer activity scans',loc='left',fontweight='bold'); ax[0,0].legend(frameon=False,ncol=2)
    x=np.arange(4); w=.36; l3=[low[a]['3.0']['median'] for a in ACTIONS]; h3=[high[a]['3.0']['median'] for a in ACTIONS]
    ax[0,1].bar(x-w/2,l3,width=w,label='Low producer'); ax[0,1].bar(x+w/2,h3,width=w,label='High producer'); ax[0,1].axhline(1,ls='--',lw=.8); ax[0,1].set_xticks(x); ax[0,1].set_xticklabels(ACTIONS); ax[0,1].set_ylabel('PenG flux / state reference'); ax[0,1].set_title('b  Three-fold responses',loc='left',fontweight='bold'); ax[0,1].legend(frameon=False)
    im=ax[1,0].imshow(mat,aspect='auto'); ax[1,0].set_xticks(x); ax[1,0].set_xticklabels(ACTIONS,rotation=35,ha='right'); ax[1,0].set_yticks(x); ax[1,0].set_yticklabels(ACTIONS)
    for i in range(4):
        for j in range(4): ax[1,0].text(j,i,f'{mat[i,j]:.2f}\n(n={cnt[i,j]})',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax[1,0],fraction=.046,label='PenG flux / low reference'); ax[1,0].set_title('c  Pairwise three-fold perturbations',loc='left',fontweight='bold')
    med=np.array([high['PenDE'][str(float(z))]['median'] for z in FOLDS]); q1=np.array([high['PenDE'][str(float(z))]['q25'] for z in FOLDS]); q3=np.array([high['PenDE'][str(float(z))]['q75'] for z in FOLDS])
    ax[1,1].plot(FOLDS,med,marker='o'); ax[1,1].fill_between(FOLDS,q1,q3,alpha=.14); ax[1,1].set_xscale('log'); ax[1,1].axhline(1,ls='--',lw=.8); ax[1,1].set_xlabel('Coupled PenDE activity scaling'); ax[1,1].set_ylabel('High-producer PenG flux / reference'); ax[1,1].set_title('d  High-producer PenDE response',loc='left',fontweight='bold')
    fig.tight_layout(); fig.savefig(FIGPNG,dpi=300,bbox_inches='tight'); fig.savefig(FIGSVG,bbox_inches='tight'); plt.close(fig)
    print(OUT); print(FIGPNG); return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--n-try',type=int,default=8000); ap.add_argument('--n-sub',type=int,default=300); ap.add_argument('--seed',type=int,default=29); a=ap.parse_args(); main(a.n_try,a.n_sub,a.seed)
