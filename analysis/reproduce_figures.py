#!/usr/bin/env python3


import os, sys, copy, gzip, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from penkin import data_io, ensemble, design
from penkin.model import DEFAULT_PARAMS as P, rhs, fluxes, steady_state, STATES, ENZYMES
from scipy.integrate import solve_ivp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from penkin.paths import RESULTS_DIR, FIGURES_DIR
from penkin import plotstyle
plotstyle.apply()
HERE = FIGURES_DIR
os.makedirs(HERE, exist_ok=True)
COL = {"ACVS": "#c1272d", "IPNS": "#0072b2", "PCL": "#009e73", "IAT": "#d55e00", "IAH": "#7b3294"}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
def lab(ax, s): ax.set_title(s, loc="left", fontweight="bold")
def _J(PAA, p, y0):
    y, f = steady_state(PAA, p, y0)
    return f['secr'] if f is not None else np.nan
def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"),
                    dpi=170 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


# Manuscript configuration. n_sub=None evaluates the
# control analysis on the full admissible ensemble, matching evaluate_model.py.
MANUSCRIPT_N_TRY = 8000
MANUSCRIPT_N_SUB = None


def compute(n_try=MANUSCRIPT_N_TRY, n_sub=MANUSCRIPT_N_SUB):
    d = {}
    
    rows = {}
    with gzip.open(os.path.join(ROOT, "penkin", "data", "GSE9825_series_matrix_txt.gz"), "rt") as fh:
        blk = False
        for line in fh:
            if line.startswith("!series_matrix_table_begin"): blk = True; continue
            if line.startswith("!series_matrix_table_end"): break
            if blk:
                p = line.rstrip("\n").split("\t")
                try: rows[p[0].strip('"')] = np.array([float(x) for x in p[1:]])
                except ValueError: pass
    M = np.array([rows[p] for p in rows]); L = np.log2(M + 1)
    top = np.argsort(L.var(1))[::-1][:2000]; X = L[top].T - L[top].T.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    d["PC"] = (U * S)[:, :3]; d["ev"] = (S**2 / (S**2).sum())[:3]
    d["strain"] = np.array(["high"]*7 + ["low"]*6); d["paa"] = np.array([0,0,0,1,1,1,1,0,0,0,1,1,1])
    core = {"pcbAB\n(ACVS)": "Pc21g21390_s_at", "pcbC\n(IPNS)": "Pc21g21380_at", "penDE\n(IAT)": "Pc21g21370_at"}
    d["H"] = np.array([rows[v] for v in core.values()]); d["hm_genes"] = list(core)
    HP, LP = [3,4,5,6], [10,11,12]
    hi = np.array(M[:, HP].mean(1)); lo = np.array(M[:, LP].mean(1)); d["lfc"] = np.log2((hi+1)/(lo+1))
    for e, pr in [("acvs","Pc21g21390_s_at"),("ipns","Pc21g21380_at"),("iat","Pc21g21370_at")]:
        d[e] = np.log2(rows[pr][HP].mean()/rows[pr][LP].mean())
    fold, reps = data_io.load_expression(); boot = data_io.bootstrap_expression(reps)
    d["expr_point"] = [fold[e] for e in ["ACVS","IPNS","IAT"]]
    d["expr_ci"] = [[np.percentile(boot[e],2.5), np.percentile(boot[e],97.5)] for e in ["ACVS","IPNS","IAT"]]
    
    y0,_ = steady_state(0.0)
    sol = solve_ivp(rhs, [0,4], y0, args=(5.0,P), method="LSODA", rtol=1e-8, atol=1e-11, dense_output=True)
    t = np.linspace(0,4,500); Y = sol.sol(t)
    d["t"] = t; d["Y"] = Y; d["Jt"] = np.array([fluxes(Y[:,i],5.0,P)["secr"] for i in range(len(t))])
    d["paas"] = np.linspace(0,10,40); d["Jp"] = np.array([_J(pa, P, None) for pa in d["paas"]])
    y,f = steady_state(5.0); d["flux_ss"] = [f[k] for k in ["ACVS","IPNS","IAT","IAH","secr"]]
    
    acc = ensemble.fit_ensemble(n_try=n_try, verbose=True)
    if n_sub is None:
        n_sub = len(acc)
    d["n_acc"] = len(acc)
    pred, sub = ensemble.predict_producer_ratio(acc, fold, boot, n_sub=n_sub)
    d["ratios"] = pred["ratios"]; d["boot_ratios"] = pred["boot_ratios"]; d["frac"] = pred["frac_high_gt_low"]
    d["param"] = np.array([[a["p"]["Vmax_"+e] / P["Vmax_"+e] for e in ENZYMES] for a in acc])
    res = design.analyze(acc, n_sub=n_sub, include_all_rates=False)
    for e in ENZYMES:
        d[f"FCC_{e}"] = res["FCC"][e]; d[f"g3_{e}"] = res["gain3"][e]; d[f"gd_{e}"] = res["gain_del"][e]
    
    yr, fr = steady_state(5.0); Jr = fr["secr"]
    folds = np.array([0.1,0.25,0.5,0.75,1,1.5,2,3,5,10]); d["folds"] = folds
    actions = ["ACVS", "IPNS", "PCL", "penDE"]
    def scaled_action(p0, action, fold):
        p2 = copy.deepcopy(p0)
        if action == "penDE":
            p2["Vmax_IAT"] *= fold
            p2["Vmax_IAH"] *= fold
        else:
            p2[ENZYMES[action]] *= fold
        return p2
    for action in actions:
        d[f"scan_{action}"] = np.array([_J(5.0, scaled_action(P, action, fo), yr)/Jr for fo in folds])
    d["g3_penDE"] = res["penDE_gain3"]
    Dm = np.ones((len(actions), len(actions)))
    for i,a in enumerate(actions):
        for j,b in enumerate(actions):
            p2 = scaled_action(P, a, 3.0)
            if i != j:
                p2 = scaled_action(p2, b, 3.0)
            Dm[i,j] = _J(5.0, p2, yr)/Jr
    d["double"] = Dm; d["names"] = actions
    ipf = np.linspace(0.3,3,25); acv=[]; acv0=[]
    for fo in ipf:
        p2 = copy.deepcopy(P); p2["Vmax_ACVS"]*=2.21; p2["Vmax_IPNS"]*=fo
        r = steady_state(5.0, p2, yr); acv.append(r[0][0] if r[0] is not None else np.nan)
        p3 = copy.deepcopy(P); p3["Vmax_IPNS"]*=fo
        r = steady_state(5.0, p3, yr); acv0.append(r[0][0] if r[0] is not None else np.nan)
    d["ipf"]=ipf; d["acv"]=np.array(acv); d["acv0"]=np.array(acv0)
    np.savez(os.path.join(RESULTS_DIR,"figure_data.npz"), **d)
    return d


def render(d):
    ENZ = list(ENZYMES)
    
    # Figure 1 is generated separately by analysis/figure1_architecture.py.


    
    fig,ax = plt.subplots(2,2,figsize=plotstyle.size(2)); fig.patch.set_facecolor("white")
    for s,mk in [("high","o"),("low","s")]:
        for p,c in [(1,"#c1272d"),(0,"#6699cc")]:
            m=(d["strain"]==s)&(d["paa"]==p); ax[0,0].scatter(d["PC"][m,0],d["PC"][m,1],marker=mk,s=90,c=c,edgecolor="k",lw=.6,label=f"{s} {'+PAA' if p else '-PAA'}")
    ax[0,0].set_xlabel(f"PC1 ({d['ev'][0]*100:.0f}%)"); ax[0,0].set_ylabel(f"PC2 ({d['ev'][1]*100:.0f}%)"); ax[0,0].legend(fontsize=8,frameon=False,loc="best"); lab(ax[0,0],"a  Transcriptome PCA (13 samples)")
    Hn=np.log2(d["H"]/d["H"].mean(1,keepdims=True)); im=ax[0,1].imshow(Hn,aspect="auto",cmap="RdBu_r",vmin=-1.5,vmax=1.5)
    ax[0,1].set_yticks(range(3)); ax[0,1].set_yticklabels(d["hm_genes"]); ax[0,1].set_xticks([]); plt.colorbar(im,ax=ax[0,1],label="log2 rel. expr",fraction=0.046); lab(ax[0,1],"b  Penicillin BGC expression (13 arrays)")
    gn=["pcbAB","pcbC","penDE"]; ep=d["expr_point"]; eci=d["expr_ci"]
    err=[[ep[i]-eci[i][0] for i in range(3)],[eci[i][1]-ep[i] for i in range(3)]]
    ax[1,0].bar(gn,ep,yerr=err,capsize=5,color=[COL["ACVS"],COL["IPNS"],COL["IAT"]]); ax[1,0].axhline(1,ls="--",color="gray"); ax[1,0].set_ylabel("expr fold-change high/low (+PAA)"); lab(ax[1,0],"c  Up-regulation (95% bootstrap CI)")
    for i,v in enumerate(ep): ax[1,0].text(i,eci[i][1]+.05,f"{v:.2f}x",ha="center",fontsize=9)
    lfc=d["lfc"][np.isfinite(d["lfc"])]; ax[1,1].hist(lfc,bins=80,color="#bbb")
    for e,c in [("acvs",COL["ACVS"]),("ipns",COL["IPNS"]),("iat",COL["IAT"])]: ax[1,1].axvline(d[e],color=c,lw=2,label=f"{e.upper()} ({np.mean(lfc<d[e])*100:.0f}th pct)")
    ax[1,1].set_xlabel("log2 fold-change high/low"); ax[1,1].set_ylabel("probe-set count"); ax[1,1].legend(fontsize=8,frameon=False); lab(ax[1,1],"d  Pathway vs genome-wide (15,531 probe sets)")
    save(fig,"Figure2")

    
    fig,ax = plt.subplots(2,2,figsize=plotstyle.size(2)); fig.patch.set_facecolor("white")
    ax[0,0].plot(d["t"]*60,d["Jt"],color=COL["ACVS"],lw=2.3); ax[0,0].set_xlabel("time after PAA step (min)"); ax[0,0].set_ylabel("PenG secretion flux"); ax[0,0].margins(x=0); lab(ax[0,0],"a  PAA stimulus-response")
    st=list(STATES)
    for s,c in [("ACV","#c1272d"),("IPNp","#0072b2"),("PAACoAp","#009e73"),("PENGc","#d55e00")]:
        vals = np.asarray(d["Y"][st.index(s)], dtype=float)
        vals_plot = np.maximum(vals, 1e-8)  # values below 1e-8 mM are numerical zero for log-scale plotting
        ax[0,1].plot(d["t"]*60, vals_plot, label=s, lw=1.8, color=c)
    ax[0,1].set_xlabel("time (min)"); ax[0,1].set_ylabel("concentration (mM; log scale)"); ax[0,1].set_yscale("log"); ax[0,1].set_ylim(bottom=1e-8); ax[0,1].legend(fontsize=8,frameon=False); ax[0,1].margins(x=0); ax[0,1].text(0.02,0.03,r"values < $10^{-8}$ mM plotted at $10^{-8}$",transform=ax[0,1].transAxes,fontsize=7.5,color="#666"); lab(ax[0,1],"b  Intermediate dynamics")
    ax[1,0].plot(d["paas"],d["Jp"],"-o",ms=4,color="#333"); ax[1,0].set_xlabel("PAA (mM)"); ax[1,0].set_ylabel("steady-state PenG flux"); lab(ax[1,0],"c  PAA dose-response")
    ax[1,1].bar(["ACVS","IPNS","IAT","IAH","secr"],d["flux_ss"],color=[COL["ACVS"],COL["IPNS"],COL["IAT"],COL["IAH"],"#555"]); ax[1,1].set_ylabel("flux"); lab(ax[1,1],"d  Steady-state fluxes")
    save(fig,"Figure3")

    
    fig,ax = plt.subplots(2,2,figsize=plotstyle.size(2)); fig.patch.set_facecolor("white")
    bp=ax[0,0].violinplot([d["param"][:,i] for i in range(5)],showmedians=True)
    for i,b in enumerate(bp["bodies"]): b.set_facecolor(list(COL.values())[i]); b.set_alpha(.6)
    ax[0,0].set_xticks(range(1,6)); ax[0,0].set_xticklabels(ENZ); ax[0,0].axhline(1,ls=":",color="gray"); ax[0,0].set_ylabel("sampled Vmax / reference value"); lab(ax[0,0],f"a  Admissible ensemble (n={d['n_acc']})")
    bp=ax[0,1].boxplot([d[f"FCC_{e}"] for e in ENZ],tick_labels=ENZ,patch_artist=True,showfliers=False)
    for patch,e in zip(bp["boxes"],ENZ): patch.set_facecolor(COL[e]); patch.set_alpha(.7)
    ax[0,1].axhline(0,color="k",lw=.7); ax[0,1].set_ylabel(r"flux-control coefficient $C^{J}_{E_i}$"); lab(ax[0,1],"b  Flux-control coefficients (low-producer state)")
    ax[1,0].hist(d["ratios"],bins=35,color="#c1272d",alpha=.8); ax[1,0].axvline(1,ls="--",color="k",label="high=low"); ax[1,0].axvline(np.median(d["ratios"]),color="darkred",label=f"median {np.median(d['ratios']):.2f}x")
    ax[1,0].set_xlabel("predicted high/low ratio"); ax[1,0].set_ylabel("model count"); ax[1,0].legend(fontsize=8,frameon=False); lab(ax[1,0],f"c  Producer-ratio simulation ({d['frac']*100:.0f}% >1)")
    ax[1,1].hist(d["ratios"],bins=30,color="#888",alpha=.6,density=True,label="parameter unc."); ax[1,1].hist(d["boot_ratios"],bins=30,color="#c1272d",alpha=.5,density=True,label="+transcript bootstrap")
    ax[1,1].axvline(1,ls="--",color="k"); ax[1,1].set_xlabel("predicted high/low ratio"); ax[1,1].set_ylabel("density"); ax[1,1].legend(fontsize=8,frameon=False); lab(ax[1,1],"d  Uncertainty propagation")
    save(fig,"Figure4")

    
    fig,ax = plt.subplots(2,2,figsize=plotstyle.size(2)); fig.patch.set_facecolor("white")
    action_colors={"ACVS":COL["ACVS"],"IPNS":COL["IPNS"],"PCL":COL["PCL"],"penDE":COL["IAT"]}
    actions=["ACVS","IPNS","PCL","penDE"]
    for e in actions: ax[0,0].plot(d["folds"],d[f"scan_{e}"],"-o",ms=4,color=action_colors[e],label=e)
    ax[0,0].axhline(1,ls="--",color="gray"); ax[0,0].axvline(1,ls=":",color="gray"); ax[0,0].set_xscale("log"); ax[0,0].set_xlabel("activity scaling (fold vs WT)"); ax[0,0].set_ylabel("PenG flux (rel. WT)")
    ax[0,0].set_xticks([0.1,0.25,0.5,1,2,5,10]); ax[0,0].set_xticklabels([0.1,0.25,0.5,1,2,5,10],fontsize=8); ax[0,0].legend(fontsize=8,frameon=False,ncol=2); lab(ax[0,0],"a  Single-action scan")
    g3={"ACVS":d["g3_ACVS"],"IPNS":d["g3_IPNS"],"PCL":d["g3_PCL"],"penDE":d["g3_penDE"]}; rank=sorted(actions,key=lambda e:np.median(g3[e]))
    med=[np.median(g3[e]) for e in rank]; q1=[med[i]-np.percentile(g3[rank[i]],25) for i in range(len(rank))]; q3=[np.percentile(g3[rank[i]],75)-med[i] for i in range(len(rank))]
    ax[0,1].barh(rank,med,xerr=[q1,q3],color=[action_colors[e] for e in rank],capsize=4); ax[0,1].axvline(1,ls="--",color="gray"); ax[0,1].set_xlabel("PenG fold at 3x (median±IQR)")
    for i,e in enumerate(rank): ax[0,1].text(med[i]+q3[i]+.05,i,f"P={np.mean(g3[e]>1.02):.2f}",va="center",fontsize=8)
    lab(ax[0,1],"b  Three-fold perturbation response")
    Dm=d["double"]; nm=list(d["names"]); im=ax[1,0].imshow(Dm,cmap="YlOrRd",vmin=1,vmax=np.nanmax(Dm))
    ax[1,0].set_xticks(range(len(nm))); ax[1,0].set_xticklabels(nm,rotation=45,ha="right"); ax[1,0].set_yticks(range(len(nm))); ax[1,0].set_yticklabels(nm)
    for i in range(len(nm)):
        for j in range(len(nm)):
            if np.isfinite(Dm[i,j]): ax[1,0].text(j,i,f"{Dm[i,j]:.1f}",ha="center",va="center",fontsize=8,color="white" if Dm[i,j]>2 else "black")
    plt.colorbar(im,ax=ax[1,0],label="PenG fold vs WT",fraction=0.046); lab(ax[1,0],"c  Paired three-fold perturbations")
    axb=ax[1,1]; axb.plot(d["ipf"],d["acv"],"-o",ms=4,color=COL["IPNS"],label="ACV (ACVS boosted)"); axb.plot(d["ipf"],d["acv0"],"--",color="#88a",label="ACV (no boost)")
    axb.axhline(1,ls=":",color="gray"); axb.set_yscale("log"); axb.set_xlabel("IPNS level (fold vs WT)"); axb.set_ylabel("ACV tripeptide (mM, log)",color=COL["IPNS"]); axb.axvspan(2.0,2.4,alpha=.15,color="green"); axb.legend(fontsize=8,frameon=False,loc="upper right"); lab(axb,"d  ACV response to IPNS scaling")
    save(fig,"FigureS_perturbation_overview")
    print("rendered Figures 2-4 (png+svg); Figure 1 is produced by figure1_architecture.py and Figure 5 by perturbation_design.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--plot-only", action="store_true"); ap.add_argument("--n-try", type=int, default=MANUSCRIPT_N_TRY); ap.add_argument("--n-sub", type=int, default=-1, help="-1 uses the full admissible ensemble"); a = ap.parse_args()
    fd = os.path.join(RESULTS_DIR, "figure_data.npz")
    if a.plot_only and os.path.exists(fd):
        data = dict(np.load(fd, allow_pickle=True))
    else:
        data = compute(n_try=a.n_try, n_sub=(None if a.n_sub is None or a.n_sub < 0 else a.n_sub))
    render(data)
    print(f"accepted_models={int(data['n_acc'])}")
