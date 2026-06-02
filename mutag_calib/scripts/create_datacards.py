#!/usr/bin/env python

"""
Script to create combine datacards from pocketcoffea output for mutag calibration.

This script creates datacards for each tagger working point and pt bin combination,
with pass and fail categories. It organizes the datacards following the structure
defined in the fit templates configuration.
"""

import os
import argparse
import yaml
from pathlib import Path
from collections import defaultdict
import uproot
from coffea.util import load
from hist import Hist
from hist.axis import StrCategory
import numpy as np
import matplotlib.pyplot as plt

from pocket_coffea.utils.stat import MCProcess, DataProcess, SystematicUncertainty, MCProcesses, DataProcesses, Systematics
from pocket_coffea.utils.stat.combine import combine_datacards

# Import the configuration to get the same parameters
import mutag_calib
from mutag_calib.utils.stat.datacard_mutag import DatacardMutag

LUMI_YAML = Path(__file__).parent.parent / "configs" / "params" / "lumi_systematics.yaml"
with open(LUMI_YAML) as f:
    lumi_cfg = yaml.safe_load(f)
lumi_sys_values = lumi_cfg["lumi_systematics"]


def define_processes(samples, years):
    """Define MC and data processes for the analysis."""
    
    # Define MC processes based on flavor tagging
    mc_processes = MCProcesses([
        MCProcess(
            name="light",
            samples=samples["light"],  # Will be populated from sample names ending with _l
            years=years,
            is_signal=False,
            has_rateParam=True,
        ),
        MCProcess(
            name="c",
            samples=samples["c"],  # Will be populated from sample names ending with _c or _cc  
            years=years,
            is_signal=False,
            has_rateParam=True,
        ),
        MCProcess(
            name="b",
            samples=samples["b"],  # Will be populated from sample names ending with _b or _bb
            years=years,
            is_signal=True,        # b-jets are the signal process for the calibration
            has_rateParam=True,
        )
    ])
    
    # Define data process
    data_processes = DataProcesses([
        DataProcess(
            name="data_obs",
            samples=[],  # Will be populated from sample names starting with DATA_
            years=years,
        )
    ])
    
    return mc_processes, data_processes


def categorize_samples(cutflow):
    """Categorize samples based on their names."""
    light_samples = set()
    c_samples = set()
    b_samples = set()
    data_samples = set()

    baseline_category = "inclusive"

    for dataset, samples_dict in cutflow[baseline_category].items():
        for sample_name in samples_dict.keys():
            if sample_name.startswith("QCD_Madgraph_"):  # we don't want QCD Madgraph samples in the datacards, we use it for systematics
                continue
            elif sample_name.startswith("DATA_"):
                data_samples.add(sample_name)
            elif sample_name.endswith("_l"):
                light_samples.add(sample_name)
            elif sample_name.endswith(("_c", "_cc")):
                c_samples.add(sample_name)
            elif sample_name.endswith(("_b", "_bb")):
                b_samples.add(sample_name)
    
    return {
        "light": sorted(list(light_samples)),
        "c": sorted(list(c_samples)),
        "b": sorted(list(b_samples)),
        "data_obs": sorted(list(data_samples))
    }


def define_systematics(years, mc_process_names):
    """Define systematic uncertainties."""
    year = years[0]
    lumi_value = lumi_sys_values[year]

    systematics = Systematics([
        # Add basic systematic uncertainties
        SystematicUncertainty(
            name="lumi", 
            typ="lnN",
            processes=mc_process_names,
            value=lumi_value,
            years=years,
        ),
        SystematicUncertainty(
            name="pileup",
            typ="shape",
            processes={name : 1.0 for name in mc_process_names},
            years=years,
        ),
        SystematicUncertainty(
            name="QCD_MuEnriched_ratio",
            typ="shape",
            processes={name: 1.0 for name in mc_process_names},
            years=years,
        ),
        SystematicUncertainty(
            name="sf_partonshower_fsr",
            typ="shape",
            processes={name: 1.0 for name in mc_process_names},
            years=years,
        ),
        SystematicUncertainty(
            name="sf_partonshower_isr",
            typ="shape",
            processes={name: 1.0 for name in mc_process_names},
            years=years,
        ),
        SystematicUncertainty(
            name="AK8PFPuppi_JER",
            typ="shape",
            processes={name: 1.0 for name in mc_process_names},
            years=years,
        ),
        SystematicUncertainty(
            name="AK8PFPuppi_JES_Total",
            typ="shape",
            processes={name: 1.0 for name in mc_process_names},
            years=years,
        ),
        # Add process-specific uncertainties
        SystematicUncertainty(
            name="light_norm",
            typ="lnN",
            processes=["light"],
            value=1.20,  # 20% normalization uncertainty for light jets
            years=years,
        ),
        SystematicUncertainty(
            name="c_norm", 
            typ="lnN",
            processes=["c"],
            value=1.20,  # 20% normalization uncertainty for c jets
            years=years,
        ),
    ])
    
    return systematics

def get_passfail_ratio(datacards):
    """Calculate the pass/fail ratio for each category and tau21 cut value.
    
    Args:
        datacards: Nested dictionary of format datacards[category][tau21] = datacard
    
    Returns:
        Dictionary of pass/fail ratios organized by parent category and tau21 cut
    """
    sumw_percat = defaultdict(dict)
    # First level: categories
    for cat in datacards:
        # Second level: tau21 cuts
        for tau21, datacard in datacards[cat].items():
            shape_histograms = datacard.create_shape_histogram_dict(is_data=False)
            sumw_percat[cat][tau21] = {
                process_name.split("_nominal")[0]: hist.values().sum() 
                for process_name, hist in shape_histograms.items()
            }

    passfail_ratio = defaultdict(lambda: defaultdict(dict))
    parent_categories = set(['-'.join(cat.split("-")[:-1]) for cat in datacards.keys()])
    
    for parent_cat in parent_categories:
        for tau21 in datacards[f"{parent_cat}-pass"].keys():
            sumw_pass = sumw_percat[f"{parent_cat}-pass"][tau21]
            sumw_fail = sumw_percat[f"{parent_cat}-fail"][tau21]
            for flavor in sumw_pass.keys():
                passfail_ratio[parent_cat][tau21][flavor] = float(sumw_pass[flavor] / sumw_fail[flavor])

    return dict(passfail_ratio)

def add_Madgraph_systematic(histogram_logsumSVmass_tau21):
    """Function to add the Madgraph systematic uncertainty to the histogram."""
    # select the qcd_samples
    qcd_samples = [s for s in histogram_logsumSVmass_tau21.keys() if s.startswith("QCD_")]
    print(f"QCD samples: {qcd_samples}\n")
    # extract the flavors
    flavors = {s.split("__")[1].split("_")[-1] for s in qcd_samples if "__" in s and len(s.split("__")[1].split("_")) >= 2}
    print(f"flavors: {flavors}\n")
    for flav in flavors:
        print(f"\nProcessing flavor: {flav}")
        mu_name = f"QCD_MuEnriched__QCD_MuEnriched_{flav}"
        # mg_name = f"QCD_MuEnriched__QCD_MuEnriched_{flav}"  # proof that the code is working as I think
        mg_name = f"QCD_Madgraph__QCD_Madgraph_{flav}"
        if mu_name not in histogram_logsumSVmass_tau21 or mg_name not in histogram_logsumSVmass_tau21:
            print(f"------- {mu_name} or {mg_name} not in histogram_logsumSVmass_tau21 -------\n")
            continue
        mu_datasets = histogram_logsumSVmass_tau21[mu_name]
        mg_datasets = histogram_logsumSVmass_tau21[mg_name]
        mu_total = None
        mg_total = None
        for h in mu_datasets.values():
            h_nom = h[:, "nominal", :, :]
            mu_total = h_nom if mu_total is None else mu_total + h_nom
        for h in mg_datasets.values():
            h_nom = h[:, "nominal", :, :]
            mg_total = h_nom if mg_total is None else mg_total + h_nom
        print(f"mu_total.axes: {mu_total.axes}")
        print(f"mg_total.axes: {mg_total.axes}")
        # plot_mu_vs_mg(mu_total, mg_total, flav, tau21_cut=0.3)  # see the difference in mu and mg shapes
        mu_vals = mu_total.values(flow=True)
        mg_vals = mg_total.values(flow=True)
        mu_int = mu_vals.sum()
        mg_int = mg_vals.sum()
        scale = mu_int / mg_int
        mg_vals_scaled = mg_vals * scale
        mask = mu_vals > 1e-9
        ratio = np.ones_like(mu_vals)
        ratio[mask] = mg_vals_scaled[mask] / mu_vals[mask]
        ratio = np.clip(ratio, 0, 2)
        for dataset, h_mu in mu_datasets.items():
            print(f"\ndataset: {dataset}")
            current_vars = list(h_mu.axes["variation"])  # variations already present
            print(f"current_vars: {current_vars}")
            new_vars = current_vars + [f"QCD_MuEnriched_ratioUp", f"QCD_MuEnriched_ratioDown"]  # adding new variations
            print(f"new_vars: {new_vars}")
            new_vars_axis = StrCategory(new_vars, name="variation")  # new variation axis
            new_hist = Hist(
                h_mu.axes["cat"],
                new_vars_axis,
                h_mu.axes["FatJetGood.logsumcorrSVmass"],
                h_mu.axes["FatJetGood.tau21"],
                storage=h_mu.storage_type()
            )
            for v in current_vars:
                new_hist.view(flow=True)[:, new_hist.axes["variation"].index(v), :, :] = h_mu.view(flow=True)[:, h_mu.axes["variation"].index(v), :, :]
            nom_idx = h_mu.axes["variation"].index("nominal")
            up_idx = new_hist.axes["variation"].index("QCD_MuEnriched_ratioUp")
            down_idx = new_hist.axes["variation"].index("QCD_MuEnriched_ratioDown")
            nom_view = h_mu.view(flow=True)[:, nom_idx, :, :]
            up_view = new_hist.view(flow=True)[:, up_idx, :, :]
            down_view = new_hist.view(flow=True)[:, down_idx, :, :]
            up_view["value"] = nom_view["value"] * ratio
            up_view["variance"] = nom_view["variance"] * ratio**2
            down_view["value"] = nom_view["value"] * (2 - ratio)
            down_view["variance"] = nom_view["variance"] * (2 - ratio)**2
            print("Old sum:", h_mu.values(flow=True).sum())
            print("New sum:", new_hist.values(flow=True).sum())
            nominal_integral = new_hist.view(flow=True)[:, nom_idx, :, :].sum().value
            up_integral = new_hist.view(flow=True)[:, up_idx, :, :].sum().value
            down_integral = new_hist.view(flow=True)[:, down_idx, :, :].sum().value
            up_factor = nominal_integral / up_integral
            down_factor = nominal_integral / down_integral
            new_hist.view(flow=True)[:, up_idx, :, :] *= up_factor
            new_hist.view(flow=True)[:, down_idx, :, :] *= down_factor
            histogram_logsumSVmass_tau21[mu_name][dataset] = new_hist
        # plot nominal vs up vs down variations
        mu_datasets = histogram_logsumSVmass_tau21[mu_name]
        nom_total = None
        up_total = None
        down_total = None
        for h in mu_datasets.values():
            h_nom = h[:, "nominal", :, :]
            nom_total = h_nom if nom_total is None else nom_total + h_nom
        for h in mu_datasets.values():
            h_nom = h[:, "QCD_MuEnriched_ratioUp", :, :]
            up_total = h_nom if up_total is None else up_total + h_nom
        for h in mu_datasets.values():
            h_nom = h[:, "QCD_MuEnriched_ratioDown", :, :]
            down_total = h_nom if down_total is None else down_total + h_nom
        plot_nom_vs_up_vs_down(nom_total, up_total, down_total, flav, tau21_cut=0.3)
        nom_vals = nom_total.values(flow=True)
        up_vals = up_total.values(flow=True)
        down_vals = down_total.values(flow=True)
        nom_int = nom_vals.sum()
        up_int = up_vals.sum()
        down_int = down_vals.sum()
        print(f"nom_int: {nom_int}\nup_int: {up_int}\ndown_int: {down_int}")

def add_Madgraph_systematic_1d(histo_1d, cat):
    qcd_samples = [s for s in histo_1d.keys() if s.startswith("QCD_")]
    print(f"QCD samples: {qcd_samples}\n")
    flavors = {s.split("__")[1].split("_")[-1] for s in qcd_samples if "__" in s and len(s.split("__")[1].split("_")) >= 2}
    print(f"flavors: {flavors}\n")
    for flav in flavors:
        print(f"\nProcessing flavor: {flav}")
        mu_name = f"QCD_MuEnriched__QCD_MuEnriched_{flav}"
        mg_name = f"QCD_Madgraph__QCD_Madgraph_{flav}"
        if mu_name not in histo_1d or mg_name not in histo_1d:
            print(f"Skipping {flav}: {mu_name} or {mg_name} not found in histo_1d")
            continue
        mu_datasets = histo_1d[mu_name]
        mg_datasets = histo_1d[mg_name]
        mu_total = None
        mg_total = None
        for h in mu_datasets.values():
            h_nom = h[cat, "nominal", :]
            mu_total = h_nom if mu_total is None else mu_total + h_nom
        for h in mg_datasets.values():
            h_nom = h[cat, "nominal", :]
            mg_total = h_nom if mg_total is None else mg_total + h_nom
        mu_vals = mu_total.values(flow=True)
        mg_vals = mg_total.values(flow=True)
        scale = mu_vals.sum() / mg_vals.sum()
        mg_vals_scaled = mg_vals * scale
        mask = mu_vals > 1e-9
        ratio = np.ones_like(mu_vals)
        ratio[mask] = mg_vals_scaled[mask] / mu_vals[mask]
        ratio = np.clip(ratio, 0, 2)
        for dataset, h_mu in mu_datasets.items():
            print(f"\ndataset: {dataset}")
            current_vars = list(h_mu.axes["variation"])
            # print(f"current_vars: {current_vars}")
            new_vars = current_vars + [f"QCD_MuEnriched_ratioUp", f"QCD_MuEnriched_ratioDown"]
            # print(f"new_vars: {new_vars}")
            new_vars_axis = StrCategory(new_vars, name="variation")
            new_hist = Hist(
                h_mu.axes["cat"],
                new_vars_axis,
                h_mu.axes["FatJetGood.logsumcorrSVmass"],
                storage=h_mu.storage_type()
            )
            for v in current_vars:
                idx_old = h_mu.axes["variation"].index(v)
                idx_new = new_hist.axes["variation"].index(v)
                cat_idx = h_mu.axes["cat"].index(cat)
                new_hist.view(flow=True)[cat_idx, idx_new, :] = h_mu.view(flow=True)[cat_idx, idx_old, :]
            cat_idx = h_mu.axes["cat"].index(cat)
            nom_idx = h_mu.axes["variation"].index("nominal")
            up_idx = new_hist.axes["variation"].index("QCD_MuEnriched_ratioUp")
            down_idx = new_hist.axes["variation"].index("QCD_MuEnriched_ratioDown")
            nom_view = h_mu.view(flow=True)[cat_idx, nom_idx, :]
            up_view = new_hist.view(flow=True)[cat_idx, up_idx, :]
            down_view = new_hist.view(flow=True)[cat_idx, down_idx, :]
            up_view["value"] = nom_view["value"] * ratio
            up_view["variance"] = nom_view["variance"] * ratio**2
            down_view["value"] = nom_view["value"] * (2 - ratio)
            down_view["variance"] = nom_view["variance"] * (2 - ratio)**2
            nominal_integral = new_hist.view(flow=True)[cat_idx, nom_idx, :].sum().value
            up_integral = new_hist.view(flow=True)[cat_idx, up_idx, :].sum().value
            down_integral = new_hist.view(flow=True)[cat_idx, down_idx, :].sum().value
            print(f"nominal_integral = {nominal_integral}\nup_integral = {up_integral}\ndown_integral = {down_integral}")
            if up_integral > 0:
                up_factor = nominal_integral / up_integral
                new_hist.view(flow=True)[cat_idx, up_idx, :] *= up_factor
            else:
                print(f"Warning: up_integral = 0, skipping renormalization")
            if down_integral > 0:
                down_factor = nominal_integral / down_integral
                new_hist.view(flow=True)[cat_idx, down_idx, :] *= down_factor
            else:
                print(f"Warning: down_integral = 0, skipping renormalization")
            histo_1d[mu_name][dataset] = new_hist

def plot_tau21_mu_vs_mg(histogram_tau21, cat="pt300msd80to170"):
    qcd_samples = [s for s in histogram_tau21.keys() if s.startswith("QCD_")]
    flavors = {s.split("__")[1].split("_")[-1] for s in qcd_samples if "__" in s and len(s.split("__")[1].split("_")) >= 2}
    for flav in flavors:
        mu_name = f"QCD_MuEnriched__QCD_MuEnriched_{flav}"
        mg_name = f"QCD_Madgraph__QCD_Madgraph_{flav}"
        if mu_name not in histogram_tau21 or mg_name not in histogram_tau21:
            print(f"------- {mu_name} or {mg_name} not in histogram_tau21 -------\n")
            continue
        mu_datasets = histogram_tau21[mu_name]
        mg_datasets = histogram_tau21[mg_name]
        mu_total = None
        mg_total = None
        for h in mu_datasets.values():
            h_nom = h[cat, "nominal", :]
            mu_total = h_nom if mu_total is None else mu_total + h_nom
        for h in mg_datasets.values():
            h_nom = h[cat, "nominal", :]
            mg_total = h_nom if mg_total is None else mg_total + h_nom
        mu_vals = mu_total.values(flow=False)
        mg_vals = mg_total.values(flow=False)
        mu_vals = mu_vals / mu_vals.sum()
        mg_vals = mg_vals / mg_vals.sum()
        edges = mu_total.axes[0].edges
        centers = 0.5 * (edges[1:] + edges[:-1])
        plt.figure(figsize=(7,5))
        plt.step(centers, mu_vals, where="mid", label="MuEnriched")
        plt.step(centers, mg_vals, where="mid", label="Madgraph")
        plt.xlabel("tau21")
        plt.ylabel("Normalized events")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"tau21_mu_vs_mg_{flav}.png")
        plt.close()
        print(f"Saved tau21_mu_vs_mg_{flav}.png")

def plot_mu_vs_mg(mu_total, mg_total, flavour, tau21_cut=0.3):
    ax_tau21 = mu_total.axes["FatJetGood.tau21"]
    bin_stop = next(i for i, edge in enumerate(ax_tau21.edges[1:]) if edge > tau21_cut)
    mu_1d = mu_total.integrate(ax_tau21.name, 0, bin_stop)
    mg_1d = mg_total.integrate(ax_tau21.name, 0, bin_stop)
    mu_1d = mu_1d.project("FatJetGood.logsumcorrSVmass")
    mg_1d = mg_1d.project("FatJetGood.logsumcorrSVmass")
    ax_mass = mu_1d.axes["FatJetGood.logsumcorrSVmass"]
    edges = ax_mass.edges
    centers = 0.5 * (edges[1:] + edges[:-1])
    mu_vals = mu_1d.values()
    mg_vals = mg_1d.values()
    plt.figure(figsize=(7,5))
    plt.step(centers, mu_vals, where="mid", label="MuEnriched")
    plt.step(centers, mg_vals, where="mid", label="Madgraph")
    plt.xlabel("FatJetGood.logsumcorrSVmass")
    plt.ylabel("Events")
    plt.legend()
    plt.title(f"tau21 < {tau21_cut}")
    plt.tight_layout()
    outfile = f"mu_vs_mg_tau21_{tau21_cut}_{flavour}.png"
    plt.savefig(outfile)
    plt.close()
    print(f"Plot saved in: {outfile}")

def plot_nom_vs_up_vs_down(nom_total, up_total, down_total, flavour, tau21_cut=0.3):
    ax_tau21 = nom_total.axes["FatJetGood.tau21"]
    bin_stop = next(i for i, edge in enumerate(ax_tau21.edges[1:]) if edge > tau21_cut)
    nom_1d = nom_total.integrate(ax_tau21.name, 0, bin_stop)
    up_1d = up_total.integrate(ax_tau21.name, 0, bin_stop)
    down_1d = down_total.integrate(ax_tau21.name, 0, bin_stop)
    nom_1d = nom_1d.project("FatJetGood.logsumcorrSVmass")
    up_1d = up_1d.project("FatJetGood.logsumcorrSVmass")
    down_1d = down_1d.project("FatJetGood.logsumcorrSVmass")
    nom_plot_int = nom_1d.values().sum()
    up_plot_int = up_1d.values().sum()
    down_plot_int = down_1d.values().sum()
    print(f"nom_plot_int: {nom_plot_int}\nup_plot_int: {up_plot_int}\ndown_plot_int: {down_plot_int}")
    up_1d *= nom_plot_int / up_plot_int
    down_1d *= nom_plot_int / down_plot_int
    nom_plot_int = nom_1d.values().sum()
    up_plot_int = up_1d.values().sum()
    down_plot_int = down_1d.values().sum()
    print(f"nom_plot_int: {nom_plot_int}\nup_plot_int: {up_plot_int}\ndown_plot_int: {down_plot_int}")
    ax_mass = nom_1d.axes["FatJetGood.logsumcorrSVmass"]
    edges = ax_mass.edges
    centers = 0.5 * (edges[1:] + edges[:-1])
    nom_vals = nom_1d.values()
    up_vals = up_1d.values()
    down_vals = down_1d.values()
    plt.figure(figsize=(7,5))
    plt.step(centers, nom_vals, where="mid", label="nominal")
    plt.step(centers, up_vals, where="mid", label="up")
    plt.step(centers, down_vals, where="mid", label="down")
    plt.xlabel("FatJetGood.logsumcorrSVmass")
    plt.ylabel("Events")
    plt.legend()
    plt.title(f"tau21 < {tau21_cut}")
    plt.tight_layout()
    outfile = f"nom_vs_up_vs_down_tau21_{tau21_cut}_{flavour}.png"
    plt.savefig(outfile)
    plt.close()
    print(f"Plot saved in: {outfile}")

def get_1d_histogram(h2d_dict, tau21_cut):
    """Function to get the 1D histogram from the 2D histogram by integrating over the axis corresponding to tau21."""
    h1d_dict = {}
    for proc, ds_dict in h2d_dict.items():
        print(f"\nProcessing {proc}...")
        print(f"{ds_dict.keys()}\n")
        h1d_dict[proc] = {}
        for ds, histo2d in ds_dict.items():
            # print(f"histo2d.axes = {histo2d.axes}\n")
            ax_tau21 = histo2d.axes["FatJetGood.tau21"]
            bin_stop = next(i for i, edge in enumerate(ax_tau21.edges[1:]) if edge > tau21_cut)
            histo_cut = histo2d.integrate(ax_tau21.name, 0, bin_stop)
            h1d_dict[proc][ds] = histo_cut
    # print(f"{h1d_dict.keys()}\n")
    return h1d_dict


def get_1d_histogram_reweighed(h2d_dict, tau21_cut, samples, years, parent_category):
    """Return 1D histograms with MC (b+c+light) reweighted to data.

    The input 2D histograms are first integrated over the tau21 axis as in
    get_1d_histogram, using the cut ``tau21 < tau21_cut``. Then, for the
    specified year(s) and a given parent category, a bin-by-bin weight is
    computed such that, in the *inclusive pass+fail region* for that parent
    category, the sum of all MC histograms (b + c + light) equals the data
    histogram. Those weights are applied to the MC templates in both the
    corresponding pass and fail categories, leaving data unchanged.

    ``years`` may be a single string or a list of strings: a dataset is
    included if any of the given year tags appears in its name. This
    supports --combined-years mode (e.g. 2025 data + 2024 MC) where the
    MC year tag differs from the primary label year.
    """
    if isinstance(years, str):
        years = [years]

    # Start from the standard 1D histograms
    h1d_dict = get_1d_histogram(h2d_dict, tau21_cut)

    # Find a reference MC histogram to infer axes (data histograms lack a
    # "variation" axis, so we must pick an MC one for the axis template).
    example_hist = None
    for ds_dict in h1d_dict.values():
        for h in ds_dict.values():
            if "variation" in [ax.name for ax in h.axes]:
                example_hist = h
                break
        if example_hist is not None:
            break

    if example_hist is None:
        return h1d_dict

    # Axes: ["cat", "variation", fit_variable]
    cat_axis = example_hist.axes["cat"]
    var_axis = example_hist.axes["variation"]
    fit_axes = [ax for ax in example_hist.axes if ax.name not in ("cat", "variation")]
    if len(fit_axes) != 1:
        raise RuntimeError("Expected exactly one fit variable axis after tau21 integration")
    fit_axis = fit_axes[0]

    # Identify pass and fail categories for this parent category
    pass_cat_label = f"{parent_category}-pass"
    fail_cat_label = f"{parent_category}-fail"
    cat_indices = []
    for label in (pass_cat_label, fail_cat_label):
        try:
            idx = cat_axis.index(label)
            cat_indices.append(idx)
        except KeyError:
            continue

    if not cat_indices:
        # Nothing to reweight for this parent category
        return h1d_dict

    n_fit_bins = len(fit_axis.edges) - 1
    nom_index = var_axis.index("nominal")

    # Define MC and data sample sets
    mc_sample_names = set(samples["light"] + samples["c"] + samples["b"])
    data_sample_names = set(samples["data_obs"])

    mc_sum = np.zeros(n_fit_bins, dtype=float)
    data_sum = np.zeros(n_fit_bins, dtype=float)

    # Build inclusive (pass+fail) distributions for the requested year(s) and parent category
    for proc_name, ds_dict in h1d_dict.items():
        for ds, h in ds_dict.items():
            # Restrict to the datasets whose name matches any of the requested years
            if not any(y in ds for y in years):
                continue

            # For weight-storage histograms, ``h.view`` returns a structured
            # array with (value, variance). We only want the values here.
            view = h.view(flow=False)
            values_view = view["value"]

            # Handle both cases:
            #  - 3D: (n_cat, n_var, n_fit)
            #  - 2D: (n_cat, n_fit)  (no explicit variation axis)
            if values_view.ndim == 3:
                # Sum over the pass and fail categories of this parent,
                # keeping only the nominal variation
                proj = values_view[cat_indices, nom_index, :].sum(axis=0)
            elif values_view.ndim == 2:
                # No variation axis: treat the existing values as nominal
                proj = values_view[cat_indices, :].sum(axis=0)
            else:
                raise RuntimeError(
                    f"Unsupported histogram dimensionality {values_view.ndim} in reweighting (expected 2 or 3)"
                )

            if proc_name in mc_sample_names:
                mc_sum += proj
            elif proc_name in data_sample_names:
                data_sum += proj

    # Compute bin-by-bin weights; default to 1 when MC is zero
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(mc_sum > 0.0, data_sum / mc_sum, 1.0)
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)

    # Apply weights to all MC templates (all variations) in the pass+fail
    # categories of this parent
    for proc_name, ds_dict in h1d_dict.items():
        if proc_name not in mc_sample_names:
            continue
        for ds, h in ds_dict.items():
            if not any(y in ds for y in years):
                continue
            view = h.view(flow=False)

            # For weight-storage histograms, scale values and variances
            # separately: v -> w*v, var -> w^2*var.
            values_view = view["value"]

            if values_view.ndim == 3:
                # Axes: (cat, variation, fit)
                scale = weights[np.newaxis, np.newaxis, :]
                view["value"][cat_indices, :, :] *= scale
                view["variance"][cat_indices, :, :] *= scale ** 2
            elif values_view.ndim == 2:
                # Axes: (cat, fit) – no explicit variation axis
                scale = weights[np.newaxis, :]
                view["value"][cat_indices, :] *= scale
                view["variance"][cat_indices, :] *= scale ** 2
            else:
                raise RuntimeError(
                    f"Unsupported histogram dimensionality {values_view.ndim} in reweighting (expected 2 or 3)"
                )

    return h1d_dict

def print_report(successful_categories, failed_categories):
    for d_cat in successful_categories:
        print(f"✅ Year: {d_cat['year']}, Category: {d_cat['category']}, Folder: {d_cat['folder']}")
    for d_cat in failed_categories:
        print(f"❌ Year: {d_cat['year']}, Category: {d_cat['category']}, Error: {d_cat['error']}")

    # Summary printout counting successes and failures rate
    ncat = len(successful_categories) + len(failed_categories)
    print("\nSummary Report:")
    print(f"Total categories processed: {ncat}")
    print(f"✅  Successful: {len(successful_categories)} / {ncat}")
    print(f"❌  Failed: {len(failed_categories)} / {ncat}")

# Helper function to extract the tau21 string for directory naming
get_tau21_str = lambda x: f"tau21_{x:.2f}".replace('.', 'p')


def main():
    parser = argparse.ArgumentParser(description="Create combine datacards from pocketcoffea output")
    parser.add_argument("input_file", help="Path to the pocketcoffea output .coffea file")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory for datacards")
    parser.add_argument("--variable", default="FatJetGood_logsumcorrSVmass_tau21", help="Variable to use for the fit")
    parser.add_argument("--years", nargs="+", default=["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix"],
                       help="Years to include in the analysis")
    parser.add_argument("--combined-years", action="store_true", default=False,
                       help="Treat all years as a single combined measurement (e.g. 2025 data + 2024 MC)")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Enable verbose output")
    args = parser.parse_args()
    
    # Load the coffea output
    print(f"Loading coffea output from {args.input_file}\n")
    output = load(args.input_file)
    
    # Extract histograms, cutflow, and metadata
    histograms = output["variables"]
    # hist_tau21 = histograms["FatJetGood_tau21"]
    # plot_tau21_mu_vs_mg(hist_tau21, cat="pt300msd80to170")
    cutflow = output["cutflow"]
    datasets_metadata = output["datasets_metadata"]
    categories = [cat for cat in cutflow.keys() if cat.startswith('msd')]
    
    # Categorize samples
    samples = categorize_samples(cutflow)
    print(f"\nFound samples: {samples}")
    print(f"Available histograms: {list(histograms.keys())}\n")

    successful_categories = []
    failed_categories = []

    # In combined mode, treat all years as one measurement (e.g. 2025 data + 2024 MC)
    if args.combined_years:
        year_groups = [args.years]  # single group with all years
    else:
        year_groups = [[year] for year in args.years]  # one group per year

    for years_group in year_groups:
        year = years_group[0]  # primary year label for output naming
        # Define processes and systematics
        mc_processes, data_processes = define_processes(samples, years_group)
        print(f"MC processes: {mc_processes.items()}")
        print(f"DATA processes: {data_processes.items()}\n")
        
        # Update process samples based on what we found
        for process_name, process in mc_processes.items():
            process.samples = samples[process_name]
        
        for process_name, process in data_processes.items():
            process.samples = samples[process_name]

        # Add the variation QCD_Madgraph/QCD_MuEnriched to the Hist
        # add_Madgraph_systematic(histograms[args.variable])
        
        systematics = define_systematics(years_group, [p_name for p_name, p in mc_processes.items()])
        print(f"systematics: {systematics}\n")
        
        # Create output directory
        if args.output_dir is None:
            args.output_dir = str(Path(args.input_file).parent / "datacards")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Dictionary to store all datacards for combination
        all_datacards = defaultdict(dict)
        # Additional datacards using MC reweighted to data for tau21 < 0.30
        all_datacards_reweight = defaultdict(dict)
        
        # Create datacards for each combination
        for cat in categories:
            print(f"\ncategory: {cat}")

            for tau21 in [0.2, 0.25, 0.3, 0.35, 0.4]:
                print(f"\n\nCreating datacard: Year: {year}\tCategory: {cat}\ttau21 < {tau21}")
                
                # Get the 1D histogram by integrating over tau21 axis with a specific cut: tau21 < tau21_cut
                histo_1d = get_1d_histogram(histograms[args.variable], tau21)
                # Add the variation QCD_Madgraph/QCD_MuEnriched to the Hist
                add_Madgraph_systematic_1d(histo_1d, cat)
                print("\n")
                # Create datacard
                datacard = DatacardMutag(
                    histograms=histo_1d,
                    datasets_metadata=datasets_metadata,
                    cutflow=cutflow,
                    years=years_group,
                    mc_processes=mc_processes,
                    data_processes=data_processes,
                    systematics=systematics,
                    category=cat,  # Category string matching the multicuts structure
                    bin_suffix=year,
                    verbose=args.verbose
                )
                
                # Store for combination between pass and fail regions
                all_datacards[cat][tau21] = datacard

                # For tau21 < 0.30, also create a datacard where MC
                # templates are reweighted to data in the inclusive
                # (pass+fail) region for the corresponding parent
                # category, to define an external systematic.
                if abs(tau21 - 0.3) < 1e-6:
                    print(f"\n\nCreating datacard: Year: {year}\tCategory: {cat}\ttau21 < {tau21} reweighed")
                    parent_category = "-".join(cat.split("-")[:-1])
                    histo_1d_rew = get_1d_histogram_reweighed(
                        histograms[args.variable], tau21, samples, years_group, parent_category
                    )
                    # Add the variation QCD_Madgraph/QCD_MuEnriched to the Hist
                    add_Madgraph_systematic_1d(histo_1d_rew, cat)
                    print("\n")
                    datacard_rew = DatacardMutag(
                        histograms=histo_1d_rew,
                        datasets_metadata=datasets_metadata,
                        cutflow=cutflow,
                        years=years_group,
                        mc_processes=mc_processes,
                        data_processes=data_processes,
                        systematics=systematics,
                        category=cat,
                        bin_suffix=year,
                        verbose=args.verbose,
                    )
                    all_datacards_reweight[cat][tau21] = datacard_rew
                
        passfail_ratio = get_passfail_ratio(all_datacards)

        # Loop over categories again to dump datacards modified with pass/fail ratios
        parent_categories = set()
        for cat in categories:
            for tau21 in [0.2, 0.25, 0.3, 0.35, 0.4]:
                # Extract parent category (without pass/fail)
                parent_category = '-'.join(cat.split("-")[:-1])
                parent_categories.add(parent_category)
                region = cat.split("-")[-1]
                # Create directory for this category
                tau21_str = get_tau21_str(tau21)
                category_dir = output_dir / year / parent_category / tau21_str / region
                category_dir.mkdir(parents=True, exist_ok=True)
                datacard = all_datacards[cat][tau21]

                # Modify action of rateParam for fail regions by passing the passfail_ratio argument
                if cat.endswith("-pass"):
                    kwargs = {"directory" : str(category_dir)}
                elif cat.endswith("-fail"):
                    parent_cat = '-'.join(cat.split("-")[:-1])
                    kwargs = {"directory" : str(category_dir), "passfail_ratio" : passfail_ratio[parent_cat][tau21]}
                try:
                    datacard.dump(**kwargs)
                    successful_categories.append({"year": year, "category": cat, "folder": str(category_dir)})
                except Exception as e:
                    print(f"Failed to create datacard for Year: {year}, Category: {cat}")
                    print(str(e))
                    failed_categories.append({"year": year, "category": cat, "error": str(e)})

                # For tau21 < 0.30, also dump the reweighted datacards
                if abs(tau21 - 0.3) < 1e-6 and cat in all_datacards_reweight and tau21 in all_datacards_reweight[cat]:
                    reweight_tau21_str = f"{tau21_str}_reweight"
                    reweight_category_dir = output_dir / year / parent_category / reweight_tau21_str / region
                    reweight_category_dir.mkdir(parents=True, exist_ok=True)
                    datacard_rew = all_datacards_reweight[cat][tau21]

                    if cat.endswith("-pass"):
                        kwargs_rew = {"directory": str(reweight_category_dir)}
                    elif cat.endswith("-fail"):
                        parent_cat = '-'.join(cat.split("-")[:-1])
                        kwargs_rew = {"directory": str(reweight_category_dir), "passfail_ratio": passfail_ratio[parent_cat][tau21]}
                    else:
                        kwargs_rew = {"directory": str(reweight_category_dir)}

                    try:
                        datacard_rew.dump(**kwargs_rew)
                        successful_categories.append({"year": year, "category": f"{cat}_reweight", "folder": str(reweight_category_dir)})
                    except Exception as e:
                        print(f"Failed to create reweighted datacard for Year: {year}, Category: {cat}")
                        print(str(e))
                        failed_categories.append({"year": year, "category": f"{cat}_reweight", "error": str(e)})

        # Create combined datacard for pass+fail regions, for each parent category
        for parent_cat in parent_categories:
            for tau21 in [0.2, 0.25, 0.3, 0.35, 0.4]:
                print(f"\nCreating combined datacard for category: {parent_cat} with tau21 < {tau21} (pass + fail)")
                tau21_str = get_tau21_str(tau21)
                directory = output_dir / year / parent_cat / tau21_str
                combine_datacards(
                    datacards={f"{region}/datacard.txt": all_datacards[f"{parent_cat}-{region}"][tau21] for region in ["pass", "fail"]},
                    directory=directory
                )
                # Save pass/fail ratio to a YAML file
                filename = directory / "passfail_ratio.yaml"
                print(f"Saving pass/fail ratio to {filename}")
                with open(filename, "w") as f:
                    yaml.dump({"passfail_ratio" : passfail_ratio[parent_cat][tau21]}, f, indent=4)

                print(f"Combined datacard saved in {directory}")

                # For tau21 < 0.30, also create the combined reweighted datacard
                if abs(tau21 - 0.3) < 1e-6:
                    reweight_tau21_str = f"{tau21_str}_reweight"
                    directory_rew = output_dir / year / parent_cat / reweight_tau21_str
                    print(f"\nCreating combined reweighted datacard for category: {parent_cat} with tau21 < {tau21} (pass + fail)")
                    combine_datacards(
                        datacards={f"{region}/datacard.txt": all_datacards_reweight[f"{parent_cat}-{region}"][tau21] for region in ["pass", "fail"]},
                        directory=directory_rew,
                    )
                    filename_rew = directory_rew / "passfail_ratio.yaml"
                    print(f"Saving pass/fail ratio to {filename_rew}")
                    with open(filename_rew, "w") as f:
                        yaml.dump({"passfail_ratio": passfail_ratio[parent_cat][tau21]}, f, indent=4)
                    print(f"Combined reweighted datacard saved in {directory_rew}")

    # Print summary report
    print_report(successful_categories, failed_categories)

if __name__ == "__main__":
    main()
