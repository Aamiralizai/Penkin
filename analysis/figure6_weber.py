#!/usr/bin/env python3
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
import sys
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from penkin.paths import RESULTS_DIR, FIGURES_DIR
J=json.load(open(os.path.join(RESULTS_DIR,'weber2012_validation.json')))
E=pd.read_csv(os.path.join(RESULTS_DIR,'weber2012_digitized_relative.csv'))
fig,ax=plt.subplots(3,2,figsize=(11,12))
for strain,g in E[(E.target=='penDE')&(E.metric=='penicillin_V')].groupby('strain'):
 g=g.sort_values('gene_copy_mid'); ax[0,0].plot(g.gene_copy_mid,g.relative_to_parent,marker='o',label=strain)
ax[0,0].axhline(1,lw=.8); ax[0,0].set(xlabel='penDE copy number',ylabel='Penicillin V / parental'); ax[0,0].set_title('A  Weber et al.: penDE',loc='left',fontweight='bold'); ax[0,0].legend(frameon=False)
f=np.array(sorted(float(k) for k in J['penDE_activity_scan'])); m=np.array([J['penDE_activity_scan'][str(x)]['penicillin']['median'] for x in f]); q1=np.array([J['penDE_activity_scan'][str(x)]['penicillin']['q25'] for x in f]); q3=np.array([J['penDE_activity_scan'][str(x)]['penicillin']['q75'] for x in f])
ax[0,1].plot(f,m,marker='o'); ax[0,1].fill_between(f,q1,q3,alpha=.18); ax[0,1].set_xscale('log'); ax[0,1].axhline(1,lw=.8); ax[0,1].set(xlabel='Coupled PenDE activity scaling',ylabel='PenG flux / reference'); ax[0,1].set_title('B  Penkin: PenDE',loc='left',fontweight='bold')
for strain,g in E[(E.target=='phl')&(E.metric=='penicillin_V')].groupby('strain'):
 g=g.sort_values('gene_copy_mid'); ax[1,0].plot(g.gene_copy_mid,g.relative_to_parent,marker='o',label=strain)
ax[1,0].axhline(1,lw=.8); ax[1,0].set(xlabel='phl/PCL copy number',ylabel='Penicillin V / parental'); ax[1,0].set_title('C  Weber et al.: PCL/phl',loc='left',fontweight='bold'); ax[1,0].legend(frameon=False)
f2=np.array(sorted(float(k) for k in J['PCL_activity_scan'])); m2=np.array([J['PCL_activity_scan'][str(x)]['penicillin']['median'] for x in f2]); q12=np.array([J['PCL_activity_scan'][str(x)]['penicillin']['q25'] for x in f2]); q32=np.array([J['PCL_activity_scan'][str(x)]['penicillin']['q75'] for x in f2])
ax[1,1].plot(f2,m2,marker='o'); ax[1,1].fill_between(f2,q12,q32,alpha=.18); ax[1,1].set_xscale('log'); ax[1,1].axhline(1,lw=.8); ax[1,1].set(xlabel='PCL activity scaling',ylabel='PenG flux / reference'); ax[1,1].set_title('D  Penkin: PCL',loc='left',fontweight='bold')
for strain,g in E[(E.target=='penDE')&(E.metric=='6_APA')].groupby('strain'):
 g=g.sort_values('gene_copy_mid'); ax[2,0].plot(g.gene_copy_mid,g.relative_to_parent,marker='o',label=strain)
ax[2,0].axhline(1,lw=.8); ax[2,0].set(xlabel='penDE copy number',ylabel='6-APA / parental'); ax[2,0].set_title('E  Weber et al.: 6-APA',loc='left',fontweight='bold'); ax[2,0].legend(frameon=False)
m=np.array([J['penDE_activity_scan'][str(x)]['IAH_flux']['median'] for x in f]); q1=np.array([J['penDE_activity_scan'][str(x)]['IAH_flux']['q25'] for x in f]); q3=np.array([J['penDE_activity_scan'][str(x)]['IAH_flux']['q75'] for x in f])
ax[2,1].plot(f,m,marker='o'); ax[2,1].fill_between(f,q1,q3,alpha=.18); ax[2,1].set_xscale('log'); ax[2,1].axhline(1,lw=.8); ax[2,1].set(xlabel='Coupled PenDE activity scaling',ylabel='IAH branch flux / reference'); ax[2,1].set_title('F  Penkin: IAH-branch proxy',loc='left',fontweight='bold')
fig.tight_layout();
for ext in ['png','svg']: fig.savefig(os.path.join(FIGURES_DIR,'Figure6.'+ext),dpi=300 if ext=='png' else None,bbox_inches='tight')
plt.close(fig); print(os.path.join(FIGURES_DIR,'Figure6.png'))
