#(myenv) [ganiendo@lxplus941 tau21_0p30]$ cat make_combine_plots.py 
#!/usr/bin/env python3
import argparse, os, numpy as np, uproot, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# Higher-res + larger text
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 14,
    "axes.linewidth": 1.2,
})

# Exact RGBs (l, c, b)
COL_L = (181/255,  28/255,  16/255)
COL_C = (254/255, 157/255,  43/255)
COL_B = ( 61/255, 135/255, 209/255)
COL_TOTAL = (0.1, 0.1, 0.1)
COL_DATA  = (0, 0, 0)
COL_BAND  = (0.3, 0.3, 0.3)

def fetch_hist(d, key):
    h = d[key]
    vals = h.values(flow=False).astype(float)
    edges = h.axes[0].edges(flow=False).astype(float)
    var = h.variances(flow=False)
    err = np.sqrt(var).astype(float) if var is not None else np.zeros_like(vals, float)
    return vals, edges, err

def fetch_graph_asymm(d):
    if "data" not in d: return (None,)*4
    g = d["data"]
    # Try direct members (common in uproot for TGraphAsymmErrors)
    def m(name):
        try: return np.asarray(g.member(name), dtype=float)
        except Exception: return None
    x, y = m("fX"), m("fY")
    ylo, yhi = m("fEYlow"), m("fEYhigh")
    # Fallbacks
    if x is None or y is None:
        try:
            x, y = g.values()
            x = np.asarray(x, float); y = np.asarray(y, float)
        except Exception:
            return (None,)*4
    if ylo is None or yhi is None:
        try:
            ey = g.errors("both", "y")
            ylo = np.asarray(ey[0], float); yhi = np.asarray(ey[1], float)
        except Exception:
            ylo = np.zeros_like(y, float); yhi = np.zeros_like(y, float)
    return x, y, ylo, yhi

def _pad_to_edges(y, edges):
    if y is None: return None
    if len(edges) == len(y):       # already N+1
        return y
    if len(edges) == len(y) + 1:   # N -> N+1 (step='post')
        return np.r_[y, y[-1]]
    raise ValueError(f"Unexpected lengths: len(edges)={len(edges)}, len(y)={len(y)}")

def stairs_fill(ax, y, edges, color, baseline=None, alpha=1.0, zorder=1):
    if baseline is None:
        baseline = np.zeros_like(y)
    y_pad    = _pad_to_edges(y, edges)
    base_pad = _pad_to_edges(baseline, edges)
    ax.fill_between(
        edges, base_pad, base_pad + y_pad,
        step="post", facecolor=color, alpha=alpha, linewidth=0, zorder=zorder
    )

def draw_total_band(ax, ytot, edges, sigma, color=COL_BAND, alpha=0.25, zorder=2):
    y = np.clip(ytot, 0.0, None)
    s = np.clip(sigma, 0.0, None)
    y_pad  = _pad_to_edges(y, edges)
    lo_pad = _pad_to_edges(np.clip(y - s, 0.0, None), edges)
    hi_pad = _pad_to_edges(y + s, edges)
    ax.fill_between(edges, lo_pad, hi_pad, step="post",
                    facecolor=color, alpha=alpha, linewidth=0, zorder=zorder)

def ratio_band(ax, ytot, edges, sigma, alpha=0.25, zorder=2):
    y = np.clip(ytot, 0.0, None)
    s = np.clip(sigma, 0.0, None)
    safe = y > 0
    rhi = np.ones_like(y); rlo = np.ones_like(y)
    rhi[safe] = 1.0 + s[safe]/y[safe]
    rlo[safe] = 1.0 - s[safe]/y[safe]
    ax.fill_between(
        edges, _pad_to_edges(rlo, edges), _pad_to_edges(rhi, edges),
        step="post", facecolor=COL_BAND, alpha=alpha, linewidth=0, zorder=zorder
    )

def get_sigma_from_cov(d, nbins):
    cov = d["total_covar"].values(flow=False)
    ny, nx = cov.shape
    m = min(ny, nx, nbins)
    diag = np.zeros(nbins, float)
    diag[:m] = np.maximum(np.diag(cov)[:m], 0.0)
    return np.sqrt(diag)

def nice_axes(ax):
    # All four spines visible; inward ticks on all sides; include left ticks.
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(which="both", direction="in",
                   top=True, right=True, left=True, bottom=True, length=6)
    ax.tick_params(which="minor", length=3)
    ax.minorticks_on()

def label_axes_main(ax):
    # X label right-aligned; Y label on LEFT (default), centered vertically.
    ax.set_xlabel(r'$\log(\sum m_{\mathrm{SV}}^{\mathrm{corr}})$', x=1.0, ha='right')
    ax.set_ylabel('Counts')

def plot_one(f, group, channel, out_png, era, ratio_ylim=(0.4, 1.6)):
    d = f[f"{group}"][channel]

    b, edges, _ = fetch_hist(d, f"b_{era}")
    c, _, _     = fetch_hist(d, f"c_{era}")
    l, _, _     = fetch_hist(d, f"light_{era}")
    tot, _, _   = fetch_hist(d, "total")
    sigma       = get_sigma_from_cov(d, len(tot))

    x_data, y_data, ylo, yhi = fetch_graph_asymm(d)

    fig = plt.figure(figsize=(8.0, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.5], hspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    rx = fig.add_subplot(gs[1, 0], sharex=ax)

    nice_axes(ax); nice_axes(rx)
    ax.tick_params(labelbottom=False)
    label_axes_main(ax)
    rx.set_xlabel(r'$\log(\sum m_{\mathrm{SV}}^{\mathrm{corr}})$', x=1.0, ha='right')
    rx.set_ylabel("Data/MC")  # left side, centered (default)

    fig.text(
        0.13, 0.88,
        r"$\mathbf{CMS}$ $\mathit{Simulation\ Preliminary}$",
        ha="left", va="bottom",
        fontsize=18
    )
    fig.text(
        0.9, 0.88,
        "(13.6 TeV)",
        ha="right", va="bottom",
        fontsize=18
    )

    rx.set_ylim(*ratio_ylim)

    # stack (b bottom, then c, then l)
    stairs_fill(ax, b, edges, COL_B, baseline=None, alpha=1.0, zorder=1)
    stairs_fill(ax, c, edges, COL_C, baseline=b,    alpha=1.0, zorder=2)
    stairs_fill(ax, l, edges, COL_L, baseline=b+c,  alpha=1.0, zorder=3)

    # total + syst
    # ax.step(edges, _pad_to_edges(tot, edges), where="post",
    #        color=COL_TOTAL, lw=1.2, zorder=4, label="total")
    draw_total_band(ax, tot, edges, sigma, color=COL_BAND, alpha=0.25, zorder=2.5)

    # data
    if x_data is not None:
        ax.errorbar(x_data, y_data, yerr=[ylo, yhi],
                    fmt='o', color=COL_DATA, ms=4.5, lw=1.0, capsize=0, zorder=5, label="Data")

    # legend (upper right). Add headroom so it doesn't collide with stack.
    # Compute a safe ymax using total+syst and data points.
    ymax_stack = np.max(tot + sigma) if tot.size else 0.0
    if x_data is not None and y_data.size:
        ymax_data = np.nanmax(y_data + (yhi if yhi is not None else 0))
        ymax = max(ymax_stack, ymax_data)
    else:
        ymax = ymax_stack
    ax.set_ylim(0, ymax * 1.35)  # extra room for legend

    ax.legend(
        handles=[
            mpl.patches.Patch(facecolor=COL_L, edgecolor='none', label='l'),
            mpl.patches.Patch(facecolor=COL_C, edgecolor='none', label='c'),
            mpl.patches.Patch(facecolor=COL_B, edgecolor='none', label='b'),
            #mpl.lines.Line2D([], [], color=COL_TOTAL, lw=1.2, label='total'),
            mpl.patches.Patch(facecolor=COL_BAND, alpha=0.25, edgecolor='none', label='total unc.'),
            mpl.lines.Line2D([], [], marker='o', color=COL_DATA, lw=0, label='Data')
        ],
        loc="upper right", frameon=False, ncol=2
    )

    # ratio
    if x_data is not None:
        idx  = np.searchsorted(edges, x_data, side="right") - 1
        good = (idx >= 0) & (idx < len(tot)) & (tot[idx] > 0)
        r    = np.full_like(y_data, np.nan, float)
        r[good] = y_data[good] / tot[idx][good]
        rlo = np.zeros_like(r); rhi = np.zeros_like(r)
        rlo[good] = ylo[good] / tot[idx][good]
        rhi[good] = yhi[good] / tot[idx][good]

        ratio_band(rx, tot, edges, sigma, alpha=0.25, zorder=1.5)
        rx.axhline(1.0, color=COL_TOTAL, lw=1.0, zorder=1)
        if np.any(good):
            rx.errorbar(x_data[good], r[good], yerr=[rlo[good], rhi[good]],
                        fmt='o', color=COL_DATA, ms=4.0, lw=1.0, capsize=0, zorder=3)
    else:
        ratio_band(rx, tot, edges, sigma, alpha=0.25, zorder=1.5)
        rx.axhline(1.0, color=COL_TOTAL, lw=1.0, zorder=1)

    ax.set_xlim(edges[0], edges[-1])

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--passch", required=True)
    ap.add_argument("--failch", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--postfit-group", default="shapes_fit_s",
                    choices=["shapes_fit_s","shapes_fit_b"])
    ap.add_argument("--ratio-min", type=float, default=0.4)
    ap.add_argument("--ratio-max", type=float, default=1.6)
    args = ap.parse_args()
    rmin, rmax = args.ratio_min, args.ratio_max
    era = args.passch.split("_pass_")[-1]

    with uproot.open(args.file) as f:
        plot_one(f, "shapes_prefit", args.passch,
                 os.path.join(args.outdir, "prefit_pass_bcl_data_band.pdf"), era, (rmin, rmax))
        plot_one(f, "shapes_prefit", args.passch,
                 os.path.join(args.outdir, "prefit_pass_bcl_data_band.png"), era, (rmin, rmax))
        plot_one(f, "shapes_prefit", args.failch,
                 os.path.join(args.outdir, "prefit_fail_bcl_data_band.pdf"), era, (rmin, rmax))
        plot_one(f, "shapes_prefit", args.failch,
                 os.path.join(args.outdir, "prefit_fail_bcl_data_band.png"), era, (rmin, rmax))
        plot_one(f, args.postfit_group, args.passch,
                 os.path.join(args.outdir, "postfit_sb_pass_bcl_data_band.pdf"), era, (rmin, rmax))
        plot_one(f, args.postfit_group, args.passch,
                 os.path.join(args.outdir, "postfit_sb_pass_bcl_data_band.png"), era, (rmin, rmax))
        plot_one(f, args.postfit_group, args.failch,
                 os.path.join(args.outdir, "postfit_sb_fail_bcl_data_band.pdf"), era, (rmin, rmax))
        plot_one(f, args.postfit_group, args.failch,
                 os.path.join(args.outdir, "postfit_sb_fail_bcl_data_band.png"), era, (rmin, rmax))

if __name__ == "__main__":
    main()