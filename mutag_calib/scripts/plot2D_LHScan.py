import argparse
import os
import re
import ROOT

lumi_map = {
    "2022_preEE":    "7.98/\\mathrm{{fb}}",
    "2022_postEE":   "26.67/\\mathrm{{fb}}",
    "2023_preBPix":  "17.79/\\mathrm{{fb}}",
    "2023_postBPix": "9.45/\\mathrm{{fb}}",
    "2024":          "108.96/\\mathrm{{fb}}",
}


def plot2D_LHScan_pyroot(input_file, year, category, output_dir):

    f = ROOT.TFile.Open(input_file)
    limit = f.Get("limit")

    print("Entries nel tree:", limit.GetEntries())

    c = ROOT.TCanvas("c", "c", 1200, 1000)
    c.SetLeftMargin(0.1)
    c.SetRightMargin(0.15)
    c.SetBottomMargin(0.12)
    c.SetTopMargin(0.1)

    # IDENTICO A COMBINE
    limit.Draw(
        "2*deltaNLL:SF_c:r>>h(20,0,2,20,0,2)",
        "",
        "prof colz"
    )

    h = ROOT.gROOT.FindObject("h")
    print("Tipo oggetto:", type(h))

    ROOT.TGaxis.SetMaxDigits(3)
    ROOT.gStyle.SetNumberContours(255)
    h.SetTitle("")
    h.GetXaxis().SetTitle("SF_{b}")
    h.GetXaxis().SetTitleOffset(1.2)
    h.GetXaxis().SetTitleSize(0.04)
    h.GetYaxis().SetTitle("SF_{c}")
    h.GetYaxis().SetTitleOffset(1.0)
    h.GetYaxis().SetTitleSize(0.04)
    h.GetZaxis().SetTitle("-2 #Delta log #Lambda")

    # Best fit point
    limit.Draw(
        "SF_c:r",
        "quantileExpected == -1",
        "P same"
    )

    best = ROOT.gROOT.FindObject("Graph")
    best.SetMarkerStyle(34)
    best.SetMarkerSize(3)
    best.SetMarkerColor(ROOT.kBlack)
    best.Draw("P same")

    # Contour 1 sigma
    h68 = h.Clone("h68")
    h68.SetContour(1)
    h68.SetContourLevel(1, 2.30)
    h68.SetLineWidth(3)
    h68.SetLineColor(ROOT.kBlack)
    h68.SetFillStyle(0)
    h68.Draw("CONT3 SAME")

    # CMS Preliminary
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.05)
    latex.DrawLatex(0.12, 0.91, "#bf{CMS} #it{Preliminary}")

    # Lumi
    match = re.search(r"Pt-(\d+)to(\d+|Inf)", category)
    if not match:
        raise RuntimeError(f"Cannot extract pT from category: {category}")
    pt_low  = match.group(1)
    pt_high = match.group(2)
    if pt_high == "Inf":
        pt_str = f"[{pt_low}, +\\infty]"
    else:
        pt_str = f"[{pt_low}, {pt_high}]"
    lumi_str = lumi_map[year]
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.5, 0.91, f"{lumi_str}, \\, \\mathrm{{p_{{T}}}} = {pt_str} \\, \\mathrm{{GeV}}")

    print("Min:", h.GetMinimum())
    print("Max:", h.GetMaximum())

    ROOT.gStyle.SetOptStat(0)
    c.SaveAs(f"{output_dir}/scan2D_LH_{year}_Pt_{pt_low}to{pt_high}.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="file with scan2D")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for scan2D plot")
    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output_dir
    parts = os.path.normpath(input_file).split(os.sep)
    year = parts[parts.index("datacards") + 1]
    category = parts[parts.index("datacards") + 2]

    plot2D_LHScan_pyroot(input_file, year, category, output_dir)


if __name__ == "__main__":
    main()
