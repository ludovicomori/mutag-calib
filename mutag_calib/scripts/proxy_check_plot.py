#!/usr/bin/env python
"""Shape comparison: full HH->4b signal vs. b-tagged QCD-MuEnriched proxy.

For a set of Xbb-vs-QCD taggers, overlay the (unit-area normalised) signal score
distribution with the b-tagged MuEnriched proxy at a series of tau21 upper cuts,
plus a proxy/signal ratio panel.  For every proxy curve a binned
Kolmogorov-Smirnov statistic w.r.t. the signal reference is computed and shown
in the legend.

Consolidates the three per-rundir copies that used to live at
    rundir/pt_reweight_check_HH_signal_vs_bkg_btagged_*/proxy_check_plot.py

Examples
--------
# default 2024 gloParT check output, all three taggers
proxy_check_plot.py --infile rundir/<run>/output_all.coffea --outdir rundir/<run>/plots_check_proxy

# same, but restricted to one nmu category of the CartesianSelection
proxy_check_plot.py --infile .../output_all.coffea --nmu nmu-1-matched-unique
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coffea.util import load

DEFAULT_VARS = [
    "FatJetGood_particleNet_XbbVsQCD",
    "FatJetGood_particleNetLegacy_XbbVsQCD",
    "FatJetGood_globalParT3_XbbVsQCD",
]
SIG_PREFIX = "GluGluHHto4B_Par-c2-0p00-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8"
DEFAULT_BKG_SAMPLES = [
    "QCD_MuEnriched__QCD_MuEnriched_bb",
    "QCD_MuEnriched__QCD_MuEnriched_b",
]
TAU21_CUTS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--infile", default="./output_all.coffea",
                   help="input coffea file (default: %(default)s)")
    p.add_argument("--outdir", default="./plots_check_proxy",
                   help="output directory (default: %(default)s)")
    p.add_argument("--vars", nargs="+", default=DEFAULT_VARS,
                   help="tagger variables to plot (default: the three XbbVsQCD taggers)")
    p.add_argument("--sig-flavors", nargs="*", default=[], metavar="FLAV",
                   help="signal flavour subsamples to sum, e.g. l c b cc bb; "
                        "empty (default) = use the un-split signal sample")
    p.add_argument("--bkg-samples", nargs="+", default=DEFAULT_BKG_SAMPLES,
                   help="background (sub)samples summed for the proxy (default: MuEnriched b + bb)")
    p.add_argument("--nmu", default=None,
                   help="nmu category of the CartesianSelection to select, e.g. "
                        "'nmu-1', 'nmu-1-matched', 'nmu-1-matched-unique', 'nmu-2'. "
                        "When set, every tau21 category is combined with this nmu bin "
                        "(category name '<tau21>_<nmu>'). Omit for outputs without the nmu dimension.")
    p.add_argument("--rebin", type=int, default=2,
                   help="merge this many 0.01-wide score bins into one (default: %(default)s)")
    p.add_argument("--logy", action="store_true", help="log scale on the y axis")
    return p.parse_args()


def rebin(vals, edges, factor):
    """Merge groups of `factor` consecutive bins: sum contents, keep every factor-th edge."""
    if factor <= 1:
        return np.asarray(vals, dtype=float), np.asarray(edges, dtype=float)
    vals = np.asarray(vals, dtype=float)
    edges = np.asarray(edges, dtype=float)
    ngroups = len(vals) // factor
    trimmed = ngroups * factor
    new_vals = vals[:trimmed].reshape(ngroups, factor).sum(axis=1)
    new_edges = edges[:trimmed + 1:factor]
    return new_vals, new_edges


def ks_stat(hist_a, hist_b, threshold=0.0):
    """Binned Kolmogorov-Smirnov statistic between two histograms on identical
    binning: the max absolute difference of their (normalised) cumulative
    distributions. Returns nan if either histogram is empty."""
    thresh_bin = int(len(hist_a)*threshold)
    a = np.asarray(hist_a, dtype=float)[thresh_bin:]
    b = np.asarray(hist_b, dtype=float)[thresh_bin:]
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return np.nan
    return float(np.max(np.abs(np.cumsum(a) / sa - np.cumsum(b) / sb)))


def get_1d(d, sample, cat, factor):
    """Sum over all datasets of a sample, pick `cat` (+ nominal variation if present),
    return (values, edges) after rebinning."""
    vals = None
    edges = None
    for _ds, h in d[sample].items():
        sel = {"cat": cat}
        if "variation" in [ax.name for ax in h.axes]:
            sel["variation"] = "nominal"
        hh = h[sel]
        v = hh.values()
        vals = v.copy() if vals is None else vals + v
        edges = hh.axes[0].edges
    return rebin(vals, edges, factor)


def main():
    args = parse_args()
    suffix = f"_{args.nmu}" if args.nmu else ""

    if args.sig_flavors:
        sig_samples = [f"{SIG_PREFIX}__{SIG_PREFIX}_{f}" for f in args.sig_flavors]
        sig_desc = "+".join(args.sig_flavors)
    else:
        sig_samples = [SIG_PREFIX]
        sig_desc = "all subcat."

    tau21_cats = [f"tau21-{x}" for x in TAU21_CUTS]

    o = load(args.infile)
    os.makedirs(args.outdir, exist_ok=True)

    for VAR in args.vars:
        d = o["variables"][VAR]

        # ---- signal: tau21-1 (optionally within one nmu bin), flavours summed ----
        sig_vals = None
        for s in sig_samples:
            v, edges = get_1d(d, s, f"tau21-1{suffix}", args.rebin)
            sig_vals = v.copy() if sig_vals is None else sig_vals + v
        sig_edges = edges

        # ---- background: b-tagged MuEnriched proxy, per tau21 cut ----
        bkg = {}
        for cat in tau21_cats:
            tot = None
            for s in args.bkg_samples:
                v, edges = get_1d(d, s, f"{cat}{suffix}", args.rebin)
                tot = v.copy() if tot is None else tot + v
            bkg[cat] = tot

        centers = 0.5 * (sig_edges[:-1] + sig_edges[1:])
        width = np.diff(sig_edges)

        def norm(v):
            integ = np.sum(v * width)
            return v / integ if integ > 0 else v

        # ---------------- plot ----------------
        fig, (ax, rax) = plt.subplots(
            2, 1, figsize=(11, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07},
        )

        sig_n = norm(sig_vals)
        nmu_lbl = f", {args.nmu}" if args.nmu else ""
        ax.stairs(sig_n, sig_edges, color="black", lw=2.4,
                  label=f"Signal: GluGluHH→4B ({sig_desc}), τ₂₁<1.0{nmu_lbl}", zorder=10)
        ax.stairs(sig_n, sig_edges, baseline=0, fill=True, color="black", alpha=0.10, zorder=1)

        cmap = matplotlib.colormaps["gist_rainbow"]
        for i, cat in enumerate(tau21_cats):
            cutval = cat.split("-")[1]
            cutval = "1.0" if cutval == "1" else cutval
            frac = i / (len(tau21_cats) - 1)
            ks = ks_stat(bkg[cat], sig_vals)
            ks08 = ks_stat(bkg[cat], sig_vals, threshold=0.8)
            ks09 = ks_stat(bkg[cat], sig_vals, threshold=0.9)
            ax.stairs(norm(bkg[cat]), sig_edges, color=cmap(frac), lw=1.7,
                      label=f"MC (b), τ₂₁<{cutval} (KS={ks:.3f} | KS[0.8,1]={ks08:.3f} | KS[0.9,1]={ks09:.3f})")

        ax.set_ylabel("a.u. (normalised to unit area)")
        if args.logy:
            ax.set_yscale("log")
        ax.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        title = f"{VAR}  —  shape comparison: full signal vs. b-tagged MuEnriched proxy"
        if args.nmu:
            title += f"  [{args.nmu}]"
        ax.set_title(title)
        ax.grid(alpha=0.25)

        # ratio: proxy / signal
        for i, cat in enumerate(tau21_cats):
            frac = i / (len(tau21_cats) - 1)
            r = np.divide(norm(bkg[cat]), sig_n, out=np.full_like(sig_n, np.nan), where=sig_n > 0)
            rax.stairs(r, sig_edges, color=cmap(frac), lw=1.4)
        rax.axhline(1.0, color="black", lw=1.2, ls="--")
        rax.set_ylabel("proxy / signal")
        rax.set_xlabel(d[sig_samples[0]][list(d[sig_samples[0]].keys())[0]].axes[-1].label)
        rax.set_ylim(0, 2.5)
        rax.grid(alpha=0.25)

        scale = "log" if args.logy else "linear"
        tag = f"_{args.nmu}" if args.nmu else ""
        out_png = os.path.join(
            args.outdir,
            f"proxy_check_{VAR}_signal_vs_btagged_MuEnriched{tag}_{scale}.png",
        )
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out_png)

        # quick numeric summary: integral of proxy above score 0.9 vs signal, + KS
        hi = centers > 0.9
        print(f"{'cat':>10} | frac(score>0.9) |    KS |  KS[0.8,1] | KS[0.9,1]")
        print(f"{'SIGNAL':>10} | {np.sum(sig_n[hi] * width[hi]):>13.3f} |   --- |   --- |   ---")
        for cat in tau21_cats:
            bn = norm(bkg[cat])
            print(f"{cat:>10} | {np.sum(bn[hi] * width[hi]):>13.3f} | {ks_stat(bkg[cat], sig_vals):.3f} | {ks_stat(bkg[cat], sig_vals, threshold=0.8):.3f} | {ks_stat(bkg[cat], sig_vals, threshold=0.9):.3f}")


if __name__ == "__main__":
    main()
