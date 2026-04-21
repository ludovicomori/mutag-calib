#!/usr/bin/env python3
import os, json, math
import numpy as np
import matplotlib.pyplot as plt
import ROOT
import argparse
import re

ALLOWED_CATEGORIES = [
    "msd-80to170_Pt-300to350_particleNet_XbbVsQCD-HHbbtt",
    "msd-80to170_Pt-350to425_particleNet_XbbVsQCD-HHbbtt",
    "msd-80to170_Pt-425toInf_particleNet_XbbVsQCD-HHbbtt",
    "msd-30toInf_Pt-300to350_particleNet_XbbVsQCD-HHbbgg",
    "msd-30toInf_Pt-350to425_particleNet_XbbVsQCD-HHbbgg",
    "msd-30toInf_Pt-425toInf_particleNet_XbbVsQCD-HHbbgg",
    "msd-30toInf_Pt-300to350_globalParT3_XbbVsQCD-HHbbgg",
    "msd-30toInf_Pt-350to425_globalParT3_XbbVsQCD-HHbbgg",
    "msd-30toInf_Pt-425toInf_globalParT3_XbbVsQCD-HHbbgg",
]

TAU21_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40]
TAU21_CENTRAL = 0.30


# read r from fitResults.json
def read_r(path, sf_type="b"):
    with open(path) as f:
        d = json.load(f)
    if sf_type == "b":
        return d["r"], d["r_errUp"], d["r_errDown"]
    elif sf_type == "c":
        return d["SF_c"], d["SF_c_errUp"], d["SF_c_errDown"]

# extract r from fit results
def collect_results(base_dir, sf_type="b"):
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

            # tau21 = 0.30 reweight
            tdir_rw = "tau21_0p30_reweight"
            fjson_rw = os.path.join(base, tdir_rw, "fitResults.json")
            if os.path.exists(fjson_rw):
                r_rw, eup_rw, edn_rw = read_r(fjson_rw, sf_type=sf_type)
                data[year][cat]["0.30_reweight"] = (r_rw, eup_rw, edn_rw)
    return data

# compute tau21 uncertainty
def compute_tau21_unc(results):
    r0, _, _ = results[TAU21_CENTRAL]
    diffs = [abs(results[t][0] - r0) for t in TAU21_VALUES if t != TAU21_CENTRAL]
    return max(diffs)

def compute_reweight_unc(results):
    if "0.30_reweight" not in results:
        return 0.0
    r0, _, _ = results[TAU21_CENTRAL]
    r_rw, _, _ = results["0.30_reweight"]
    return abs(r_rw - r0)

# helper function to get pT label from category
def pt_label_from_category(cat):
    m = re.search(r"Pt-(\d+)to(\d+|Inf)", cat)
    if not m:
        return cat

    lo, hi = m.group(1), m.group(2)

    if hi == "Inf":
        return r"p_{T} \geq %s" % lo
    else:
        return r"p_{T} = [%s, %s]" % (lo, hi)
    # return r"p_{T} = [%s, %s]" % (lo, hi)

# helper function to set dynamic y range
def set_dynamic_y_range(graph, y, err_up, err_dn, n_sigma=1.5):
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
    y_margin = set_dynamic_y_range(g, r, err_up, err_dn, n_sigma=1.5)
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
def plot_r_vs_category(year, data, outdir, sf_type):
    cats = [c for c in ALLOWED_CATEGORIES if c in data]
    x = np.arange(len(cats))
    r, eup, edn, eup_tot, edn_tot, tau_err, rw_err = [], [], [], [], [], [], []
    for cat in cats:
        res = data[cat]
        r0, eu, ed = res[TAU21_CENTRAL]
        d_tau = compute_tau21_unc(res)
        d_rw  = compute_reweight_unc(res)
        r.append(r0)
        eup.append(eu)
        edn.append(ed)
        tau_err.append(d_tau)
        rw_err.append(d_rw)
    outname = outdir
    sf = sf_type

    plot_r_vs_category_ROOT(
        year    = year,
        cats    = cats,
        r       = r,
        err_fit_up  = eup,
        err_fit_dn  = edn,
        tau21_err   = tau_err,
        rw_err      = rw_err,
        outname = outname,
        sf_type  = sf
    )

    return dict(zip(cats, tau_err))

def plot_r_vs_category_ROOT(year, cats, r, err_fit_up, err_fit_dn, tau21_err, rw_err, outname, sf_type):
    os.makedirs(os.path.dirname(outname), exist_ok=True)

    ROOT.gStyle.SetOptStat(0)
    n = len(cats)
    x = list(range(1, n+1))
    ex = [0]*n
    err_up_tot = [math.sqrt(err_fit_up[i]**2 + tau21_err[i]**2 + rw_err[i]**2) for i in range(n)]
    err_dn_tot = [math.sqrt(err_fit_dn[i]**2 + tau21_err[i]**2 + rw_err[i]**2) for i in range(n)]
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
    y_margin = set_dynamic_y_range(g_tot, r, err_up_tot, err_dn_tot, n_sigma=1.5)
    g_tot.GetYaxis().SetNdivisions(120, 0, 0)

    c.SetGrid()
    # g_tau.Draw("AE2")
    g_tot.Draw("AP")

    err_box = [math.sqrt(tau21_err[i]**2 + rw_err[i]**2) for i in range(n)]
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

    labels = []
    for cat in cats:
        m = re.search(r"Pt-(\d+)to(\d+|Inf)", cat)
        lo, hi = m.group(1), m.group(2)
        labels.append(f"[{lo}, {hi}]")
    latex = ROOT.TLatex()
    latex.SetTextAlign(22)
    latex.SetTextSize(0.04)
    if sf_type == "b":
        for i, label in enumerate(labels):
            latex.DrawLatex(i+1, y_margin - 0.01, label)
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

    leg = ROOT.TLegend(0.65, 0.80, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    # leg.AddEntry(g_tau, "#tau_{21} syst.", "f")
    leg.AddEntry(g_tot, "fit #oplus #tau_{21}", "lp")
    leg.AddEntry(boxes[0], "#tau_{21}^{cut} #oplus #tau_{21}^{reweight}", "f")
    leg.Draw()

    c.Update()
    c.SaveAs(outname)
    c.Close()

def save_latex_table(data, output_dir, sf_type="b"):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"SF{sf_type}_table.tex")

    with open(filename, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("year & category $p_\\mathrm{T}$ [GeV] & $\\mathrm{SF_{nominal}}$ & $\\mathrm{err_{fit}}$ & $\\tau_{21}^\\mathrm{{cut}}$ & $\\tau_{21}^\\mathrm{reweight}$ & $\\sigma_\\mathrm{tot}$ \\\\\n")
        f.write("\\hline\n")

        for year in sorted(data.keys()):
            f.write("\\hline\n")
            for cat in ALLOWED_CATEGORIES:
                if cat not in data[year]:
                    continue
                res = data[year][cat]
                r0, err_up, err_dn = res[TAU21_CENTRAL]
                tau21_unc = compute_tau21_unc(res)
                reweight_unc = compute_reweight_unc(res)
                total_unc = math.sqrt(max(err_up, err_dn)**2 + tau21_unc**2 + reweight_unc**2)

                # scrittura riga tabella
                year_label = year.replace("_", " ")
                m = re.search(r"Pt-(\d+)to(\d+|Inf)", cat)
                if m:
                    lo, hi = m.group(1), m.group(2)
                    if hi == "Inf":
                        cat_label = f"[{lo}, $\\infty$]"
                    else:
                        cat_label = f"[{lo}, {hi}]"
                else:
                    cat_label = cat
                f.write(f"{year_label} & {cat_label} & {r0:.3f} & {max(err_up, err_dn):.3f} & {tau21_unc:.3f} & {reweight_unc:.3f} & {total_unc:.3f} \\\\\n")
                f.write("\\hline\n")

        f.write("\\end{tabular}\n")
        f.write(f"""
        \\caption{{Scale factors $\\mathrm{{SF}}_\\mathrm{{{sf_type}}}$ for $\\mathrm{{m}}_\\mathrm{{SD}}$ $\\in$ [80, 170] GeV for ParticleNet XbbVsQCD tagger WP = 0.75.
        $\\mathrm{{err_{{fit}}}}$ is the error coming from Combine fit, so statistics and systematics (pileup, lumi, isr, fsr, JER, JES, syst on light and c jets, Madgraph/Pythia QCD),
        $\\tau_{{21}}^\\mathrm{{cut}}$ is the systematic uncertainty related to the choice of the $\\tau_{{21}}$ cut used in the event selection (max difference between nominal 
        $\\tau_{{21}}$ cut at 0.30 and variations at 0.20, 0.25, 0.35, 0.40), $\\tau_{{21}}^\\mathrm{{reweight}}$ is the systematic uncertainty related to the SF obtained after reweight 
        of MC to data (difference between the SF at nominal $\\tau_{{21}}$ cut at 0.30 with and without the reweight).}}\n
        """)
        f.write("\\end{table}\n")

    print(f"[OK] LaTeX table saved to {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="Base directory containing fit results")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for SFs_plots")
    parser.add_argument("--SF-type", "-sf", default="b", help="Type of scale factor: b for SF_b, c for SF_c (default: b)")
    args = parser.parse_args()

    base_dir = args.base_dir
    sf_type = args.SF_type
    data = collect_results(base_dir, sf_type=sf_type)

    for year in data:
        year_out = f"{args.output_dir}/{year}"

        for cat, res in data[year].items():
            plot_r_vs_tau21(year, cat, res, os.path.join(year_out, f"SF{sf_type}_vs_tau21_{cat}.pdf"), sf_type)
            plot_r_vs_tau21(year, cat, res, os.path.join(year_out, f"SF{sf_type}_vs_tau21_{cat}.png"), sf_type)
            print(f"[OK] Plotted SF vs tau21 for {year} {cat}")

        tau21_errors = plot_r_vs_category(year, data[year], os.path.join(year_out, f"SF{sf_type}_vs_category_tau21_0p30.pdf"), sf_type)
        tau21_errors = plot_r_vs_category(year, data[year], os.path.join(year_out, f"SF{sf_type}_vs_category_tau21_0p30.png"), sf_type)
        print(f"[OK] Plotted SF vs category for {year}")

        # salva errore tau21
        with open(os.path.join(year_out, f"SF{sf_type}_tau21_sys.json"), "w") as f:
            json.dump(tau21_errors, f, indent=2)
        print(f"[OK] Saved tau21 uncertainties for {year}")

    save_latex_table(data, args.output_dir, sf_type=sf_type)


if __name__ == "__main__":
    main()
