#!/usr/bin/env python3
import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
import sys
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from penkin.paths import RESULTS_DIR, FIGURES_DIR
from penkin import plotstyle
plotstyle.apply()
S=json.load(open(os.path.join(RESULTS_DIR,'structural.json')))
RAW=os.path.join(RESULTS_DIR,'structural_full_distributions.csv')

kinds=['baseline','legacy_feedforward','reversible_IPNS','PCL_no_feedback','ACVS_classical_Ki','IAH_classical_4mM','combined']
labels=['Baseline','Feed-forward\n(no ACV fb)','Reversible\nIPNS','No PCL\nfeedback','ACVS\nKi=12.5 mM','IAH\nKm=4 mM','Combined']
short=['Base','FF','revIPNS','noPCLfb','ACVSKi','IAHKm','Combined']
ENZ=['ACVS','IPNS','PCL','IAT','IAH']

rows=[]
with open(RAW,newline='') as fh:
    for r in csv.DictReader(fh):
        r['evaluable']=int(r['evaluable'])
        for k,v in list(r.items()):
            if k not in ('variant','model_id','evaluable'):
                try:r[k]=float(v)
                except:r[k]=np.nan
        rows.append(r)

def vals(kind,key):
    return np.asarray([r[key] for r in rows if r['variant']==kind and r['evaluable']==1 and np.isfinite(r[key])],float)

def box(ax,key,title,ylabel,refline=None):
    data=[vals(k,key) for k in kinds]
    ax.boxplot(data,showfliers=False,tick_labels=short)
    ax.tick_params(axis='x',labelrotation=35,labelsize=7)
    if refline is not None: ax.axhline(refline,ls='--',lw=.8)
    ax.set_ylabel(ylabel); ax.set_title(title,loc='left',fontweight='bold')

# Two columns, three rows: a b / c d / e f.
fig,grid=plt.subplots(3,2,figsize=plotstyle.size(3))
ax={(0,0):grid[0,0],(0,1):grid[0,1],(0,2):grid[1,0],(1,0):grid[1,1],(1,1):grid[2,0],(1,2):grid[2,1]}
box(ax[0,0],'control_ACVS','a  Full ACVS control distributions','Flux-control coefficient',0)
box(ax[0,1],'control_IPNS','b  Full IPNS control distributions','Flux-control coefficient',0)

# PenG response to IPNS perturbation: both 0.5x and 2x per architecture.
x=np.arange(len(kinds)); w=.35
med05=[np.nanmedian(vals(k,'IPNS_0.5x_flux_ratio')) if len(vals(k,'IPNS_0.5x_flux_ratio')) else np.nan for k in kinds]
med20=[np.nanmedian(vals(k,'IPNS_2x_flux_ratio')) if len(vals(k,'IPNS_2x_flux_ratio')) else np.nan for k in kinds]
ax[0,2].bar(x-w/2,med05,w,label='0.5× IPNS'); ax[0,2].bar(x+w/2,med20,w,label='2× IPNS'); ax[0,2].axhline(1,ls='--',lw=.8)
ax[0,2].set_xticks(x); ax[0,2].set_xticklabels(short,rotation=35,ha='right',fontsize=7); ax[0,2].set_ylabel('PenG flux / architecture reference'); ax[0,2].set_title('c  IPNS perturbation response',loc='left',fontweight='bold'); ax[0,2].set_ylim(0,1.38); ax[0,2].legend(frameon=False,fontsize=8,ncol=2,loc='upper left')

# ACV response under IPNS perturbation: full distributions at 0.5x and 2x.
data=[]; positions=[]; tickpos=[]
for i,k in enumerate(kinds):
    data.extend([vals(k,'IPNS_0.5x_ACV_ratio'),vals(k,'IPNS_2x_ACV_ratio')]); positions.extend([i*3+1,i*3+2]); tickpos.append(i*3+1.5)
ax[1,0].boxplot(data,positions=positions,widths=.7,showfliers=False)
ax[1,0].axhline(1,ls='--',lw=.8); ax[1,0].set_xticks(tickpos); ax[1,0].set_xticklabels(short,rotation=35,ha='right',fontsize=7); ax[1,0].set_ylabel('ACV / architecture reference'); ax[1,0].set_title('d  ACV response to 0.5× / 2× IPNS',loc='left',fontweight='bold')
ax[1,0].text(.01,.98,'left box = 0.5×; right box = 2×',transform=ax[1,0].transAxes,va='top',fontsize=7.5)

# Paired median control changes for ACVS/IPNS relative to baseline on the same evaluable vectors.
variants=kinds[1:]; xx=np.arange(len(variants))
da=[S[v]['paired_control_delta']['ACVS']['median'] for v in variants]
di=[S[v]['paired_control_delta']['IPNS']['median'] for v in variants]
ax[1,1].bar(xx-w/2,da,w,label='ACVS'); ax[1,1].bar(xx+w/2,di,w,label='IPNS'); ax[1,1].axhline(0,lw=.8)
ax[1,1].set_xticks(xx); ax[1,1].set_xticklabels(short[1:],rotation=35,ha='right',fontsize=7); ax[1,1].set_ylabel('Median paired change in control'); ax[1,1].set_title('e  Paired architecture effect',loc='left',fontweight='bold'); ax[1,1].legend(frameon=False,fontsize=8)

fr=[1.0]+[S[v]['paired_fraction_of_baseline'] for v in variants]; n=[S['baseline']['n']]+[S[v]['paired_n'] for v in variants]
ax[1,2].bar(np.arange(len(kinds)),fr); ax[1,2].set_ylim(0,1.05); ax[1,2].set_xticks(np.arange(len(kinds))); ax[1,2].set_xticklabels(short,rotation=35,ha='right',fontsize=7); ax[1,2].set_ylabel('Fraction evaluable'); ax[1,2].set_title('f  Valid models by architecture',loc='left',fontweight='bold')
for i,(f,nn) in enumerate(zip(fr,n)): ax[1,2].text(i,min(f+.03,1.02),f'n={nn}',ha='center',fontsize=7)

fig.tight_layout()
for ext in ['png','svg']: fig.savefig(os.path.join(FIGURES_DIR,'Figure7.'+ext),dpi=300 if ext=='png' else None,bbox_inches='tight')
plt.close(fig)

# Supplementary full-control figure: every enzyme, not only ACVS/IPNS.
fig,ax=plt.subplots(3,2,figsize=plotstyle.size(3)); axes=ax.ravel()
for i,e in enumerate(ENZ):
    data=[vals(k,f'control_{e}') for k in kinds]
    axes[i].boxplot(data,showfliers=False,tick_labels=short); axes[i].tick_params(axis='x',labelrotation=35,labelsize=7); axes[i].axhline(0,lw=.8)
    axes[i].set_ylabel('Flux-control coefficient'); axes[i].set_title(f'{chr(97+i)}  {e} control distribution',loc='left',fontweight='bold')
# Sixth panel shows absolute ACV architecture distributions.
data=[vals(k,'ACV_mM') for k in kinds]
axes[5].boxplot(data,showfliers=False,tick_labels=short); axes[5].tick_params(axis='x',labelrotation=35,labelsize=7); axes[5].set_ylabel('ACV (mM)'); axes[5].set_title('f  Steady-state ACV by architecture',loc='left',fontweight='bold')
fig.tight_layout()
for ext in ['png','svg']: fig.savefig(os.path.join(FIGURES_DIR,'FigureS_structural_full_distributions.'+ext),dpi=300 if ext=='png' else None,bbox_inches='tight')
plt.close(fig)
print(os.path.join(FIGURES_DIR,'Figure7.png'))
print(os.path.join(FIGURES_DIR,'FigureS_structural_full_distributions.png'))
