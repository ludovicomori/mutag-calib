#!/usr/bin/env python3
import os, json, math
import numpy as np
import matplotlib.pyplot as plt
import ROOT
import argparse
import re
import correctionlib.schemav2 as cs
import gzip
import rich
import yaml
from pathlib import Path

from allowed_categories import ALLOWED_CATEGORIES_SF_PLOT

# TAU21_VALUES = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
TAU21_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40]
# TAU21_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
# TAU21_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
# TAU21_VALUES = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
TAU21_CENTRAL = 0.30

# Fallback tagger name / WP score thresholds, used whenever --wp-config can't
# provide them (missing file, missing year/tagger, or a purity not listed).
# The yaml layout (mutag_calibration.wp.<year>.<tagger>) is still evolving,
# so these keep the script working even if it changes or is unavailable.
DEFAULT_WP_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "params" / "mutag_calibration_HHbbbb_2024.yaml"
DEFAULT_TAGGER = "globalParT3_XbbVsQCD"
DEFAULT_WP_THRESHOLDS = {"LP": 0.3, "MP": 0.95, "HP": 0.975, "VHP": 0.99}


def load_wp_config(config_path, year):
    """Return (tagger, {purity_label: lower_edge_score}) parsed from a
    mutag_calibration params yaml (mutag_calibration.wp.<year>.<tagger>).

    A working-point entry can be a single score ("HHbbbb_WP_VHP": 0.99) or a
    "lo-hi" range ("HHbbbb_LP": "0.3-0.95"), in which case the lower edge is
    used. Falls back to DEFAULT_TAGGER / DEFAULT_WP_THRESHOLDS for the tagger,
    the whole dict, or individual purities that can't be resolved.
    """
    tagger = DEFAULT_TAGGER
    thresholds = dict(DEFAULT_WP_THRESHOLDS)
    if not config_path:
        return tagger, thresholds

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        mutag_cfg = cfg["mutag_calibration"]
        tagger = mutag_cfg["taggers"][0]
        wp_dict = mutag_cfg["wp"][year][tagger]
    except (FileNotFoundError, KeyError, TypeError, IndexError) as e:
        print(f"[WARN] Could not read working points from {config_path} (year={year}): {e}. Using defaults.")
        return tagger, thresholds

    for key, value in wp_dict.items():
        purity = key.split("_")[-1]
        lo = value.split("-")[0] if isinstance(value, str) and "-" in value else value
        try:
            thresholds[purity] = float(lo)
        except (TypeError, ValueError):
            print(f"[WARN] Could not parse working point '{key}': {value!r}, skipping.")
    return tagger, thresholds


# read the scale factor from fitResults.json
def read_r(path, sf_type="b"):
    with open(path) as f:
        d = json.load(f)
    if sf_type == "b":
        return d["SF_b"], d["SF_b_errUp"], d["SF_b_errDown"]
    elif sf_type == "c":
        return d["SF_c"], d["SF_c_errUp"], d["SF_c_errDown"]
    elif sf_type == "c_fail":
        return d["SF_c_fail"], d["SF_c_fail_errUp"], d["SF_c_fail_errDown"]

# extract r from fit results
def collect_results(base_dir, ALLOWED_CATEGORIES, sf_type="b"):
    data = {}
    for year in sorted(os.listdir(base_dir)):
        for cat in ALLOWED_CATEGORIES:
            base = os.path.join(base_dir, year, cat)
            if not os.path.isdir(base):
                continue

            data.setdefault(year, {})[cat] = {}

            for t in TAU21_VALUES:
                tdir = f"tau21_{t:.2f}".replace(".", "p")
                fjson = os.path.join(base, tdir, "fitResults.json")
                if not os.path.exists(fjson):
                    continue

                r, eup, edown = read_r(fjson, sf_type=sf_type)
                data[year][cat][t] = (r, eup, edown)

            # tau21 = 0.30, MC reweighted to data (always-on systematic)
            tdir_rw = "tau21_0p30_reweight"
            fjson_rw = os.path.join(base, tdir_rw, "fitResults.json")
            if os.path.exists(fjson_rw):
                r_rw, eup_rw, edn_rw = read_r(fjson_rw, sf_type=sf_type)
                data[year][cat]["0.30_reweight"] = (r_rw, eup_rw, edn_rw)

            # tau21 = 0.30, b-proxy reweighted to the HH4b signal shape
            tdir_rw_sig = "tau21_0p30_reweight_signal"
            fjson_rw_sig = os.path.join(base, tdir_rw_sig, "fitResults.json")
            if os.path.exists(fjson_rw_sig):
                r_rs, eup_rs, edn_rs = read_r(fjson_rw_sig, sf_type=sf_type)
                data[year][cat]["0.30_reweight_signal"] = (r_rs, eup_rs, edn_rs)
    return data

# compute tau21 cut-variation uncertainty
def compute_tau21_unc(results):
    r0, _, _ = results[TAU21_CENTRAL]
    diffs = [abs(results[t][0] - r0) for t in TAU21_VALUES if t != TAU21_CENTRAL]
    return max(diffs)

# always-on systematic: MC reweighted to data at the nominal tau21 cut
def compute_reweight_unc(results):
    if "0.30_reweight" not in results:
        return 0.0
    r0, _, _ = results[TAU21_CENTRAL]
    r_rw, _, _ = results["0.30_reweight"]
    return abs(r_rw - r0)

# alternative to compute_tau21_unc: b-proxy reweighted to the signal shape
def compute_reweight_signal_unc(results):
    if "0.30_reweight_signal" not in results:
        return 0.0
    r0, _, _ = results[TAU21_CENTRAL]
    r_rs, _, _ = results["0.30_reweight_signal"]
    return abs(r_rs - r0)

# tau21 uncertainty is already a nuisance in the combine fit (see create_datacards.py),
# so it is contained in the fit error and must not be added again
def compute_internalised_unc(results):
    return 0.0

# selectable via --error-method: combined (in quadrature) with the always-on
# compute_reweight_unc to form the total up/down uncertainty
ERROR_METHOD_INFO = {
    "tau21": {
        "compute": compute_tau21_unc,
        "root_label": "#tau_{21}^{cut}",
        "column_header": r"$\tau_{21}^\mathrm{cut}$",
        "description": (
            r"$\tau_{21}^\mathrm{cut}$ is the systematic uncertainty related to the choice of the $\tau_{21}$ cut "
            r"used in the event selection (max difference between nominal $\tau_{21}$ cut at "
            f"{TAU21_CENTRAL:.2f}" r" and variations at "
            + ", ".join(f"{t:.2f}" for t in TAU21_VALUES if t != TAU21_CENTRAL) + ")"
        ),
    },
    "reweight_signal": {
        "compute": compute_reweight_signal_unc,
        "root_label": "reweight-to-signal",
        "column_header": r"$\mathrm{reweight\mbox{-}to\mbox{-}signal}$",
        "description": (
            r"$\mathrm{reweight\mbox{-}to\mbox{-}signal}$ is the systematic uncertainty related to reweighting the "
            r"b-proxy shape to the HH4b signal shape at the nominal $\tau_{21}$ cut of "
            f"{TAU21_CENTRAL:.2f}" r" (difference between the SF with and without this reweighting)"
        ),
    },
    "internalised": {
        "compute": compute_internalised_unc,
        "root_label": None,
        "column_header": r"$\tau_{21}^\mathrm{cut}$ (in fit)",
        "description": (
            r"$\tau_{21}^\mathrm{cut}$ is the systematic uncertainty related to the choice of the $\tau_{21}$ cut, "
            r"which is included as a nuisance parameter in the Combine fit and therefore already contained in "
            r"$\mathrm{err_{fit}}$ (not added again)"
        ),
    },
}

# helper function to get pT label from category
def pt_label_from_category(cat):
    m = re.search(r"Pt-(\d+)to(\d+|Inf)", cat)
    if not m:
        return cat

    lo, hi = m.group(1), m.group(2)

    if hi == "Inf":
        return r"p_{T} \geq %s" % (lo)
    else:
        return r"p_{T} = [%s, %s]" % (lo, hi)
    # return r"p_{T} = [%s, %s]" % (lo, hi)

# helper function to set dynamic y range
def set_dynamic_y_range(graph, y, err_up, err_dn, n_sigma=1.5, fixed_range=None):
    # fixed_range=(lo, hi) overrides the automatic range with hard limits
    if fixed_range is not None:
        lo, hi = fixed_range
        graph.GetYaxis().SetRangeUser(lo, hi)
        return lo
    max_err = max(
        max(err_up) if err_up else 0,
        max(err_dn) if err_dn else 0
    )
    margin = n_sigma * max_err
    ymin = min(y[i] - err_dn[i] for i in range(len(y)))
    ymin = math.floor((ymin - margin) * 100) / 100
    ymax = max(y[i] + err_up[i] for i in range(len(y)))
    ymax = math.ceil((ymax + margin) * 100) / 100
    graph.GetYaxis().SetRangeUser(ymin - margin, ymax + margin)
    return (ymin - margin)

# plot SFs vs tau21 cut
def plot_r_vs_tau21(year, cat, results, outdir, sf_type):
    tau = sorted(t for t in results if t in TAU21_VALUES)
    r   = [results[t][0] for t in tau]
    eup = [results[t][1] for t in tau]
    edn = [results[t][2] for t in tau]
    outname = outdir
    sf = sf_type

    plot_r_vs_tau21_ROOT(
        year     = year,
        category = cat,
        tau      = tau,
        r        = r,
        err_up   = eup,
        err_dn   = edn,
        outname  = outname,
        sf_type  = sf
    )

def plot_r_vs_tau21_ROOT(year, category, tau, r, err_up, err_dn, outname, sf_type):
    os.makedirs(os.path.dirname(outname), exist_ok=True)

    ROOT.gStyle.SetOptStat(0)
    n = len(tau)
    x = list(range(1, n+1))
    exl = [0]*n
    exh = [0]*n

    g = ROOT.TGraphAsymmErrors(n)
    for i in range(n):
        g.SetPoint(i, x[i], r[i])
        g.SetPointError(i, exl[i], exh[i], err_dn[i], err_up[i])

    g.SetMarkerStyle(20)
    g.SetMarkerSize(1.3)
    g.SetLineWidth(1)
    g.SetMarkerColor(ROOT.kBlue+1)
    g.SetLineColor(ROOT.kBlue+1)

    c = ROOT.TCanvas("c", "", 1600, 1200)
    c.SetMargin(0.12, 0.05, 0.15, 0.08)

    g.SetTitle("")
    g.GetXaxis().SetTitle("#tau_{21}")
    g.GetXaxis().SetTitleSize(0.05)
    g.GetXaxis().SetTitleOffset(1.1)
    g.GetXaxis().SetLimits(0.5, n+0.5)
    g.GetXaxis().SetNdivisions(5, 0, 0)
    g.GetXaxis().SetLabelSize(0.03)
    g.GetXaxis().SetLabelOffset(999)
    g.GetYaxis().SetTitle(f"SF_{{{sf_type}}}")
    g.GetYaxis().SetTitleSize(0.05)
    g.GetYaxis().SetTitleOffset(0.9)
    y_margin = set_dynamic_y_range(g, r, err_up, err_dn, n_sigma=1.5)  # , fixed_range=(0.9, 1.1))
    g.GetYaxis().SetNdivisions(120, 0, 0)

    c.SetGrid()
    g.Draw("AP")

    labels = [f"{t:.2f}" for t in tau]
    latex = ROOT.TLatex()
    latex.SetTextAlign(22)
    latex.SetTextSize(0.04)
    if sf_type == "b":
        for i, label in enumerate(labels):
            latex.DrawLatex(i+1, y_margin - 0.008, label)
    else:
        for i, label in enumerate(labels):
            latex.DrawLatex(i+1, y_margin - 0.05, label)

    # CMS Preliminary
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.05)
    latex.DrawLatex(0.25, 0.94, "#bf{CMS} #it{Preliminary}")
    latex.SetTextSize(0.03)
    latex.DrawLatex(0.90, 0.94, year)

    leg_label = pt_label_from_category(category)
    leg = ROOT.TLegend(0.65, 0.80, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    leg.AddEntry(g, f"{leg_label} [GeV]", "lp")
    leg.Draw()

    c.Update()
    c.SaveAs(outname)
    c.Close()

# plot SFs for tau21 = 0.30 per each year
def plot_r_vs_category(year, data, outdir, ALLOWED_CATEGORIES, sf_type, error_method="tau21"):
    compute_chosen_unc = ERROR_METHOD_INFO[error_method]["compute"]
    cats = [c for c in ALLOWED_CATEGORIES if c in data]
    x = np.arange(len(cats))
    r, eup, edn, eup_tot, edn_tot, chosen_err, rw_err = [], [], [], [], [], [], []
    for cat in cats:
        res = data[cat]
        r0, eu, ed = res[TAU21_CENTRAL]
        d_chosen = compute_chosen_unc(res)
        d_rw  = compute_reweight_unc(res)
        r.append(r0)
        eup.append(eu)
        edn.append(ed)
        chosen_err.append(d_chosen)
        rw_err.append(d_rw)
    outname = outdir
    sf = sf_type

    plot_r_vs_category_ROOT(
        year    = year,
        cats    = cats,
        r       = r,
        err_fit_up  = eup,
        err_fit_dn  = edn,
        chosen_err  = chosen_err,
        rw_err      = rw_err,
        outname = outname,
        sf_type  = sf,
        error_method = error_method
    )

    return {
        cat: {"error_method": error_method, "chosen_unc": c, "reweight_unc": rw}
        for cat, c, rw in zip(cats, chosen_err, rw_err)
    }

def plot_r_vs_category_ROOT(year, cats, r, err_fit_up, err_fit_dn, chosen_err, rw_err, outname, sf_type, error_method="tau21"):
    os.makedirs(os.path.dirname(outname), exist_ok=True)

    ROOT.gStyle.SetOptStat(0)
    n = len(cats)
    x = list(range(1, n+1))
    ex = [0]*n
    err_up_tot = [math.sqrt(err_fit_up[i]**2 + chosen_err[i]**2 + rw_err[i]**2) for i in range(n)]
    err_dn_tot = [math.sqrt(err_fit_dn[i]**2 + chosen_err[i]**2 + rw_err[i]**2) for i in range(n)]
    # g_tau = ROOT.TGraphAsymmErrors(n)
    g_tot = ROOT.TGraphAsymmErrors(n)

    for i in range(n):
        g_tot.SetPoint(i, x[i], r[i])
        g_tot.SetPointError(i, ex[i], ex[i], err_dn_tot[i], err_up_tot[i])

    g_tot.SetMarkerStyle(20)
    g_tot.SetMarkerSize(1.3)
    g_tot.SetLineWidth(1)
    g_tot.SetMarkerColor(ROOT.kBlue+1)
    g_tot.SetLineColor(ROOT.kBlue+1)

    c = ROOT.TCanvas("c", "", 1600, 1200)
    c.SetMargin(0.12, 0.05, 0.15, 0.08)

    g_tot.SetTitle("")
    g_tot.GetXaxis().SetTitle("p_{T} category [GeV]")
    g_tot.GetXaxis().SetTitleSize(0.04)
    g_tot.GetXaxis().SetTitleOffset(1.3)
    g_tot.GetXaxis().SetLimits(0.5, n+0.5)
    g_tot.GetXaxis().SetNdivisions(3, 0, 0)
    g_tot.GetXaxis().SetLabelSize(0.03)
    g_tot.GetXaxis().SetLabelOffset(999)
    g_tot.GetYaxis().SetTitle(f"SF_{{{sf_type}}}")
    g_tot.GetYaxis().SetTitleSize(0.05)
    g_tot.GetYaxis().SetTitleOffset(0.9)
    y_margin = set_dynamic_y_range(g_tot, r, err_up_tot, err_dn_tot, n_sigma=1.5)  # , fixed_range=(0.8, 1.2))
    g_tot.GetYaxis().SetNdivisions(120, 0, 0)

    c.SetGrid()
    # g_tau.Draw("AE2")
    g_tot.Draw("AP")

    err_box = [math.sqrt(chosen_err[i]**2 + rw_err[i]**2) for i in range(n)]
    boxes = []
    for i in range(n):
        x1 = x[i] - 0.02
        x2 = x[i] + 0.02
        y1 = r[i] - err_box[i]
        y2 = r[i] + err_box[i]
        box = ROOT.TBox(x1, y1, x2, y2)
        box.SetFillColor(ROOT.kRed+1)
        box.SetFillStyle(3004)
        box.SetLineWidth(0)
        box.Draw("same")
        boxes.append(box)

    # first line: pT window per category; second line: trailing category word
    # (cat.split('_')[-1]), merged into a single label over neighbouring
    # categories that share the same word.
    pt_labels, cat_words = [], []
    for cat in cats:
        m = re.search(r"Pt-(\d+)to(\d+|Inf)", cat)
        lo, hi = m.group(1), m.group(2)
        pt_labels.append(f"[{lo}, {hi}]")
        cat_words.append(cat.split('_')[-1])

    latex = ROOT.TLatex()
    latex.SetTextAlign(22)
    latex.SetTextSize(0.03)

    # vertical placement: first line just below the axis, second line one step lower
    y_line1 = y_margin - (0.01 if sf_type == "b" else 0.05)
    hist = g_tot.GetHistogram()
    yspan = (hist.GetMaximum() - hist.GetMinimum()) if hist else (max(r) - min(r) or 1.0)
    y_line2 = y_line1 - 0.06 * yspan

    for i, label in enumerate(pt_labels):
        latex.DrawLatex(i + 1, y_line1, label)

    i = 0
    while i < len(cat_words):
        j = i
        while j + 1 < len(cat_words) and cat_words[j + 1] == cat_words[i]:
            j += 1
        latex.DrawLatex(((i + 1) + (j + 1)) / 2.0, y_line2, cat_words[i])
        i = j + 1

    # CMS Preliminary
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.05)
    latex.DrawLatex(0.25, 0.94, "#bf{CMS} #it{Preliminary}")
    latex.SetTextSize(0.03)
    latex.DrawLatex(0.90, 0.94, year)

    leg = ROOT.TLegend(0.65, 0.80, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    # leg.AddEntry(g_tau, "#tau_{21} syst.", "f")
    chosen_label = ERROR_METHOD_INFO[error_method]["root_label"]
    box_label = "#tau_{21}^{reweight}" if chosen_label is None else f"{chosen_label} #oplus #tau_{{21}}^{{reweight}}"
    leg.AddEntry(g_tot, f"fit #oplus {box_label}", "lp")
    leg.AddEntry(boxes[0], box_label, "f")
    leg.Draw()

    c.Update()
    c.SaveAs(outname)
    c.Close()

def save_latex_table(data, output_dir, ALLOWED_CATEGORIES, sf_type="b", cat_coll="normal_category", error_method="tau21"):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"SF{sf_type}_{cat_coll}_{error_method}_table.tex")
    method_info = ERROR_METHOD_INFO[error_method]

    with open(filename, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write(
            "year & $\\mathrm{m_{SD}}$ [GeV] & category $p_\\mathrm{T}$ [GeV] & $\\mathrm{SF_{nominal}}$ & "
            f"$\\mathrm{{err_{{fit}}}}$ & {method_info['column_header']} & $\\tau_{{21}}^\\mathrm{{reweight}}$ & "
            "$\\sigma_\\mathrm{tot}$ \\\\\n"
        )
        f.write("\\hline\n")

        for year in data.keys():
            f.write("\\hline\n")
            for cat in ALLOWED_CATEGORIES:
                if cat not in data[year]:
                    continue
                res = data[year][cat]
                r0, err_up, err_dn = res[TAU21_CENTRAL]
                chosen_unc = method_info["compute"](res)
                reweight_unc = compute_reweight_unc(res)
                total_unc = math.sqrt(max(err_up, err_dn)**2 + chosen_unc**2 + reweight_unc**2)

                # scrittura riga tabella
                year_label = year.replace("_", " ")

                m_msd = re.search(r"msd-(\d+)to(\d+|Inf)", cat)
                if m_msd:
                    msd_lo, msd_hi = m_msd.group(1), m_msd.group(2)
                    msd_label = f"[{msd_lo}, $\\infty$]" if msd_hi == "Inf" else f"[{msd_lo}, {msd_hi}]"
                else:
                    msd_label = cat

                m = re.search(r"Pt-(\d+)to(\d+|Inf)", cat)
                if m:
                    lo, hi = m.group(1), m.group(2)
                    if hi == "Inf":
                        cat_label = f"[{lo}, $\\infty$]"
                    else:
                        cat_label = f"[{lo}, {hi}]"
                else:
                    cat_label = cat
                f.write(f"{year_label} & {msd_label} & {cat_label} & {r0:.3f} & {max(err_up, err_dn):.3f} & {chosen_unc:.3f} & {reweight_unc:.3f} & {total_unc:.3f} \\\\\n")
                f.write("\\hline\n")

        f.write("\\end{tabular}\n")
        f.write(f"""
        \\caption{{Scale factors $\\mathrm{{SF}}_\\mathrm{{{sf_type}}}$ for ParticleNet XbbVsQCD tagger WP = 0.75.
        $\\mathrm{{err_{{fit}}}}$ is the error coming from Combine fit, so statistics and systematics (pileup, lumi, isr, fsr, JER, JES, syst on light and c jets, Madgraph/Pythia QCD),
        {method_info['description']}, $\\tau_{{21}}^\\mathrm{{reweight}}$ is the systematic uncertainty related to the SF obtained after reweight
        of MC to data (difference between the SF at nominal $\\tau_{{21}}$ cut at 0.30 with and without the reweight).}}\n
        """)
        f.write("\\end{table}\n")

    print(f"[OK] LaTeX table saved to {filename}")

def save_correctionlib_json(data, output_dir, ALLOWED_CATEGORIES, sf_type="b", cat_coll="normal_category",
                             error_method="tau21", wp_config=DEFAULT_WP_CONFIG, config_year="2024"):
    compute_chosen_unc = ERROR_METHOD_INFO[error_method]["compute"]
    keys = ["central", "up", "down", "up_rew", "down_rew", "up_tau21", "down_tau21",
            "up_reweight_signal", "down_reweight_signal", "up_internalised", "down_internalised"]
    correct_dict = {key: {} for key in keys}
    up_keys = [k for k in keys if k.startswith("up")]
    down_keys = [k for k in keys if k.startswith("down")]
    if sf_type == "b":
        flow = {key: 0.85 for key in down_keys} | {key: 1.15 for key in up_keys}
    else:
        flow = {key: 0.6 for key in down_keys} | {key: 1.4 for key in up_keys}
    flow["central"] = 1.0

    tagger, wp_thresholds = load_wp_config(wp_config, config_year)
    purities_present = []

    for year in data.keys():
        for cat in ALLOWED_CATEGORIES:
            if cat not in data[year]:
                # e.g. no LP or no HP fit results for this category collection
                continue
            purity = cat.split("_")[-1]
            if purity not in purities_present:
                purities_present.append(purity)
            m = re.search(r"Pt-(\d+)to(\d+|Inf)", cat)
            pt = m.group(1)  # I am taking lower bound, as upper bound should always be covered by next bin
            res = data[year][cat]
            r0, err_up, err_dn = res[TAU21_CENTRAL]
            tau21_unc = compute_tau21_unc(res)
            reweight_unc = compute_reweight_unc(res)
            reweight_signal_unc = compute_reweight_signal_unc(res)
            chosen_unc = compute_chosen_unc(res)
            total_up = math.sqrt(err_up**2 + chosen_unc**2 + reweight_unc**2)
            total_down = math.sqrt(err_dn**2 + chosen_unc**2 + reweight_unc**2)
            if purity not in correct_dict["central"]:
                for val in correct_dict.values():
                    val[purity] = {}
            correct_dict["central"][purity][pt] = r0
            correct_dict["up"][purity][pt] = r0 + total_up
            correct_dict["down"][purity][pt] = r0 - total_down
            correct_dict["up_rew"][purity][pt] = r0 + reweight_unc
            correct_dict["down_rew"][purity][pt] = r0 - reweight_unc
            correct_dict["up_tau21"][purity][pt] = r0 + tau21_unc
            correct_dict["down_tau21"][purity][pt] = r0 - tau21_unc
            correct_dict["up_reweight_signal"][purity][pt] = r0 + reweight_signal_unc
            correct_dict["down_reweight_signal"][purity][pt] = r0 - reweight_signal_unc
            correct_dict["up_internalised"][purity][pt] = r0 + chosen_unc
            correct_dict["down_internalised"][purity][pt] = r0 - chosen_unc

    if not purities_present:
        raise ValueError(
            f"None of ALLOWED_CATEGORIES for '{cat_coll}' were found in the collected results; nothing to save."
        )

    wp_items = sorted(
        ((wp, wp_thresholds[wp]) for wp in purities_present if wp in wp_thresholds),
        key=lambda kv: kv[1],
    )
    missing_wp = [wp for wp in purities_present if wp not in wp_thresholds]
    if missing_wp:
        print(f"[WARN] No score threshold (config or default) for working point(s) {missing_wp}; "
              f"they will be omitted from the {tagger}_{sf_type}_wp_values correction.")

    corrections = []
    if wp_items:
        wp_names = ", ".join(wp for wp, _ in wp_items)
        corr_wp = cs.Correction(
                name=f"{tagger}_{sf_type}_wp_values",
                description=f"Extract working point values (lower limits) for the bb-jet discrimination for {tagger}. "
                            f"Working points included: {wp_names}.",
                inputs=[cs.Variable(name="working_point", type="string", description="Working points or purity regions used for discrimination")],
                output=cs.Variable(name="value", type="real", description="Lower edge of the score window for the given working point."),
                version=1,
                data=cs.Category(
                    nodetype="category",
                    input="working_point",
                    content=[
                        cs.CategoryItem(
                            key=wp,
                            value=val,
                            )
                        for wp, val in wp_items
                        ],
                    default=0.0
                    )
                )
        corrections.append(corr_wp)
    else:
        print(f"[WARN] No working point had a resolvable threshold; skipping the {tagger}_{sf_type}_wp_values correction.")

    corr_full = cs.Correction(
            name=f"{tagger}_{sf_type}_{cat_coll}",
            version=1,
            inputs=[
                cs.Variable(name="systematic", type="string",
                            description=f"'central' for nominal SF. 'up/down' for total SF variation "
                                        f"(reweight #oplus {error_method}). Other 'up/down_X' for additional uncertainty breakdown."),
                cs.Variable(name="working_point", type="string", description="/".join(purities_present)),
                cs.Variable(name="pt", type="real", description="FatJet pT"),
                ],
            output=cs.Variable(name="weight", type="real"),
            data=cs.Category(
                nodetype="category",
                input="systematic",
                content=[
                    cs.CategoryItem(
                        key=var,
                        value=cs.Category(
                            nodetype="category",
                            input="working_point",
                            content=[
                                cs.CategoryItem(
                                    key=purity,
                                    value=cs.Binning(
                                        nodetype="binning",
                                        input="pt",
                                        edges=list(correct_dict[var][purity].keys()) + [9999.0],
                                        content=list(correct_dict[var][purity].values()),
                                        flow=flow[var]
                                        )
                                    )
                                for purity in correct_dict[var].keys()
                                ],
                            default=flow[var]
                            )
                        )
                    for var in correct_dict.keys()
                    ]
                )
            )
    corrections.append(corr_full)
    for corr in corrections:
        rich.print(corr)
    cset = cs.CorrectionSet(
            schema_version=2,
            description=f"AK8 bbtag scale factors for {tagger}",
            corrections=corrections,
            )
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"bbtag_AK8_scale_factors_for_{tagger}_{cat_coll}_{error_method}.json")
    with open(filename, "w") as fout:
        fout.write(cset.model_dump_json(exclude_unset=True, indent=4))
    with gzip.open(f"{filename}.gzip", "wt") as fout:
        fout.write(cset.model_dump_json(exclude_unset=True, indent=4))
    return corrections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="Base directory containing fit results")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for SFs_plots")
    parser.add_argument("--SF-type", "-sf", default="b", help="Type of scale factor: b for SF_b, c for SF_c (default: b)")
    parser.add_argument("--tau21", "-t21", default="normal", help="tau21 collection scheme. options ['normal', 'all'] (default: 'normal')")
    parser.add_argument("--error-method", "-em", choices=list(ERROR_METHOD_INFO.keys()), default="internalised",
                         help="Systematic combined (in quadrature) with the always-on tau21-reweight uncertainty "
                              "and the fit error to form the total up/down uncertainty: 'tau21' uses the tau21-cut "
                              "variation, 'reweight_signal' uses the b-proxy reweight-to-signal variation, "
                              "'internalised' adds nothing because the tau21 uncertainty is already a nuisance "
                              "in the combine fit (default: 'tau21')")
    parser.add_argument("--wp-config", default=str(DEFAULT_WP_CONFIG),
                         help="YAML file to read the tagger name and working-point score thresholds from, for the "
                              f"correctionlib output (default: {DEFAULT_WP_CONFIG}). Missing file/year/tagger/purity "
                              f"entries fall back to hardcoded defaults ({DEFAULT_WP_THRESHOLDS}).")
    parser.add_argument("--config-year", default="2024",
                         help="Year key to look up in --wp-config's mutag_calibration.wp section (default: '2024')")
    args = parser.parse_args()


    base_dir = args.base_dir
    sf_type = args.SF_type
    error_method = args.error_method

    for category_collection, ALLOWED_CATEGORIES in ALLOWED_CATEGORIES_SF_PLOT.items():
        data = collect_results(base_dir, ALLOWED_CATEGORIES=ALLOWED_CATEGORIES, sf_type=sf_type)
        for year in data:
            year_out = f"{args.output_dir}/{year}"

            for cat, res in data[year].items():
                plot_r_vs_tau21(year, cat, res, os.path.join(year_out, f"SF{sf_type}_vs_tau21_{cat}_{category_collection}.pdf"), sf_type)
                plot_r_vs_tau21(year, cat, res, os.path.join(year_out, f"SF{sf_type}_vs_tau21_{cat}_{category_collection}.png"), sf_type)
                print(f"[OK] Plotted SF vs tau21 for {year} {cat}")

            sys_errors = plot_r_vs_category(year, data[year], os.path.join(year_out, f"SF{sf_type}_{category_collection}_{error_method}_vs_category_tau21_0p30.pdf"), ALLOWED_CATEGORIES, sf_type, error_method=error_method)
            sys_errors = plot_r_vs_category(year, data[year], os.path.join(year_out, f"SF{sf_type}_{category_collection}_{error_method}_vs_category_tau21_0p30.png"), ALLOWED_CATEGORIES, sf_type, error_method=error_method)
            print(f"[OK] Plotted SF vs category for {year}")

            # salva errore sistematico scelto (tau21 o reweight_signal) + reweight sempre attivo
            with open(os.path.join(year_out, f"SF{sf_type}_{category_collection}_{error_method}_sys.json"), "w") as f:
                json.dump(sys_errors, f, indent=2)
            print(f"[OK] Saved {error_method} uncertainties for {year}")

        save_latex_table(data, args.output_dir, ALLOWED_CATEGORIES, sf_type=sf_type, cat_coll=category_collection, error_method=error_method)
        save_correctionlib_json(data, args.output_dir, ALLOWED_CATEGORIES, sf_type=sf_type, cat_coll=category_collection,
                                 error_method=error_method, wp_config=args.wp_config, config_year=args.config_year)


if __name__ == "__main__":
    main()
