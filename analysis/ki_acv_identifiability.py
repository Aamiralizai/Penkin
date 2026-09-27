#!/usr/bin/env python3
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from penkin.paths import RESULTS_DIR, FIGURES_DIR

from analysis.ki_acv_profile import profile

RESULTS = RESULTS_DIR
FIGURES = FIGURES_DIR

FINE_GRID = [0.44, 0.46, 0.48, 0.50, 0.52, 0.54]
BROAD_GRID = [0.16, 0.28, 0.36, 0.44, 0.49, 0.54, 0.60, 0.72]
MULTISEEDS = [31, 73, 97]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    fine, fine_rows = profile(n_try=300, n_sub=30, seed=31, grid=FINE_GRID)
    with open(os.path.join(RESULTS, 'ki_acv_profile_044_054.json'), 'w') as fh:
        json.dump(fine, fh, indent=2)
    with open(os.path.join(RESULTS, 'ki_acv_profile_044_054.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['Ki_mM','Janoska_RMSE','Theilgaard_RMSE','Nijland_rho','ACVS_control_median','IPNS_control_median'])
        for r in fine_rows:
            w.writerow([r['Ki_ACV_mM'],r['Janoska_RMSE'],r['Theilgaard_RMSE'],r['Nijland_Spearman_protein_informed'],r['ACVS_control']['median'],r['IPNS_control']['median']])

    multiseed = []
    for seed in MULTISEEDS:
        out, rows = profile(n_try=200, n_sub=20, seed=seed, grid=BROAD_GRID)
        multiseed.append({'seed':seed,'best':out['best_by_Janoska_RMSE']['Ki_ACV_mM'],
                          'min_rmse':out['best_by_Janoska_RMSE']['Janoska_RMSE'],
                          'near':out['near_optimal_10pct_Ki_mM'],'rows':rows})
    with open(os.path.join(RESULTS, 'ki_acv_profile_multiseed.json'), 'w') as fh:
        json.dump(multiseed, fh, indent=2)

    best = fine['best_by_Janoska_RMSE']
    first = fine['profile'][str(FINE_GRID[0])]
    last = fine['profile'][str(FINE_GRID[-1])]
    conclusion = {
        'fine_grid_mM': FINE_GRID,
        'numerical_minimum_seed31_mM': best['Ki_ACV_mM'],
        'minimum_Janoska_RMSE': best['Janoska_RMSE'],
        'Janoska_RMSE_at_0.44': first['Janoska_RMSE'],
        'Janoska_RMSE_at_0.54': last['Janoska_RMSE'],
        'relative_RMSE_difference_0.44_vs_min_percent': (first['Janoska_RMSE']/best['Janoska_RMSE']-1)*100,
        'relative_RMSE_difference_0.54_vs_min_percent': (last['Janoska_RMSE']/best['Janoska_RMSE']-1)*100,
        'multiseed_best_grid_points_mM': [s['best'] for s in multiseed],
        'multiseed_all_tested_values_within_10pct_of_seed_specific_minimum': all(set(s['near']) == set(BROAD_GRID) for s in multiseed),
        'interpretation': 'The profile does not identify a unique Ki within the literature-supported region. The shallow numerical minimum is seed-dependent and is not used to replace the literature anchor with a fitted point estimate.'
    }
    with open(os.path.join(RESULTS, 'ki_acv_profile_conclusion.json'), 'w') as fh:
        json.dump(conclusion, fh, indent=2)

    agg=[]
    for ki in BROAD_GRID:
        vals=[]
        for s in multiseed:
            for r in s['rows']:
                if abs(r['Ki_ACV_mM']-ki) < 1e-12:
                    vals.append(r['Janoska_RMSE'])
        agg.append((ki,float(np.median(vals)),float(np.min(vals)),float(np.max(vals))))

    x=np.array([r['Ki_ACV_mM'] for r in fine_rows])
    j=np.array([r['Janoska_RMSE'] for r in fine_rows])
    t=np.array([r['Theilgaard_RMSE'] for r in fine_rows])
    n=np.array([r['Nijland_Spearman_protein_informed'] for r in fine_rows])
    ac=np.array([r['ACVS_control']['median'] for r in fine_rows])
    ip=np.array([r['IPNS_control']['median'] for r in fine_rows])

    fig,ax=plt.subplots(2,2,figsize=(10.5,7.8))
    ax[0,0].plot(x,j,marker='o')
    ax[0,0].axvline(0.44,ls='--',lw=.8)
    ax[0,0].axvline(0.54,ls=':',lw=.8)
    ax[0,0].set_xlabel('ACV-feedback Ki (mM)')
    ax[0,0].set_ylabel('Janoska RMSE')
    ax[0,0].set_title('A  Fine profile within literature region',loc='left',fontweight='bold')

    ax[0,1].plot(x,ac,marker='o',label='ACVS')
    ax[0,1].plot(x,ip,marker='s',label='IPNS')
    ax[0,1].set_xlabel('ACV-feedback Ki (mM)')
    ax[0,1].set_ylabel('Median flux-control coefficient')
    ax[0,1].set_title('B  Control remains distributed',loc='left',fontweight='bold')
    ax[0,1].legend(frameon=False)

    xx=np.array([a[0] for a in agg]); med=np.array([a[1] for a in agg]); lo=np.array([a[2] for a in agg]); hi=np.array([a[3] for a in agg])
    ax[1,0].plot(xx,med,marker='o')
    ax[1,0].fill_between(xx,lo,hi,alpha=.2)
    ax[1,0].axvspan(.44,.54,alpha=.12)
    ax[1,0].set_xlabel('ACV-feedback Ki (mM)')
    ax[1,0].set_ylabel('Janoska RMSE')
    ax[1,0].set_title('C  Multi-seed profile is shallow',loc='left',fontweight='bold')

    ax[1,1].plot(x,t,marker='o',label='Theilgaard RMSE')
    ax[1,1].set_xlabel('ACV-feedback Ki (mM)')
    ax[1,1].set_ylabel('Theilgaard RMSE')
    ax2=ax[1,1].twinx()
    ax2.plot(x,n,marker='s')
    ax2.set_ylabel('Nijland Spearman rho')
    ax[1,1].set_title('D  Holdout diagnostics',loc='left',fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES,'FigureS_Ki_ACV_identifiability.png'),dpi=300,bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES,'FigureS_Ki_ACV_identifiability.svg'),bbox_inches='tight')
    plt.close(fig)

    print(os.path.join(RESULTS, 'ki_acv_profile_conclusion.json'))
    print(os.path.join(FIGURES, 'FigureS_Ki_ACV_identifiability.png'))
    return conclusion


if __name__ == '__main__':
    main()
