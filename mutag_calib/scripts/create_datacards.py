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
    signal_samples = set()

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
            elif "GluGlu" in sample_name:  # HH signal sample(s), not split into flavor subsamples
                signal_samples.add(sample_name)

    return {
        "light": sorted(list(light_samples)),
        "c": sorted(list(c_samples)),
        "b": sorted(list(b_samples)),
        "data_obs": sorted(list(data_samples)),
        "signal": sorted(list(signal_samples)),
    }


def define_systematics(years, mc_process_names, tau21_var_region=None):
    """Define systematic uncertainties."""
    year = years[0]
    lumi_value = lumi_sys_values[year]

    syst_list = [
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
            processes={name: 1.0 for name in mc_process_names},
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
    ]
    if tau21_var_region is not None:
        # Separate (uncorrelated) tau21 nuisance per region: tau21_var_pass / tau21_var_fail
        syst_list.append(
                SystematicUncertainty(
                    name=f"tau21_var_{tau21_var_region}",
                    typ="shape",
                    processes={name: 1.0 for name in mc_process_names},
                    years=years,
                    )
                )
    systematics = Systematics(syst_list)

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
            # Region-specific systematics (e.g. tau21_var_pass / tau21_var_fail) only
            # exist in one region, so restrict to the keys shared by pass and fail (& operation removes all entries not shared by both lists)
            for flavor in sumw_pass.keys() & sumw_fail.keys():
                passfail_ratio[parent_cat][tau21][flavor] = float(sumw_pass[flavor] / sumw_fail[flavor])

    return dict(passfail_ratio)

TAU21_NOMINAL = 0.30
TAU21_VAR_DOWN = 0.20
TAU21_VAR_UP = 0.40


def add_tau21_variation_1d(histo_nom, histo_down, histo_up, cat, mc_processes, year,
                           syst_name="tau21_var", shape_only=True, integral_eps=1e-6):
    """Add a tau21 working-point shape systematic to the 1D histograms.

    histo_nom:  1D histos at the nominal cut (tau21 < 0.30)
    histo_down: 1D histos at tau21 < 0.20 -> used as <syst_name>Down
    histo_up:   1D histos at tau21 < 0.40 -> used as <syst_name>Up

    If shape_only is True, the Up/Down templates are rescaled so that, for each
    process (summed over its samples and the datasets of `year`) in category `cat`,
    they have the same integral as the nominal.
    """
    up_name, down_name = f"{syst_name}Up", f"{syst_name}Down"

    def _nom_view(h):
        return h.view(flow=True)[:, h.axes["variation"].index("nominal"), :]

    for process_name, process in mc_processes.items():
        sample_names = [s for s in process.samples if s in histo_nom]
        if not sample_names:
            continue

        # Per-process integrals in this category (only datasets of this year)
        sums = {"nominal": 0.0, "up": 0.0, "down": 0.0}
        for s in sample_names:
            for ds, h in histo_nom[s].items():
                if year not in ds:
                    continue
                ci = h.axes["cat"].index(cat)
                sums["nominal"] += np.nansum(_nom_view(h)[ci]["value"])
                sums["up"] += np.nansum(_nom_view(histo_up[s][ds])[ci]["value"])
                sums["down"] += np.nansum(_nom_view(histo_down[s][ds])[ci]["value"])

        def _factor(key):
            if not shape_only:
                return 1.0
            if sums["nominal"] > integral_eps and sums[key] > integral_eps:
                return sums["nominal"] / sums[key]
            return None  # not enough stats -> variation = nominal

        f_up, f_down = _factor("up"), _factor("down")
        print(f"[{syst_name}] {process_name} in {cat}: nominal={sums['nominal']:.3f}, "
              f"up={sums['up']:.3f} (x{f_up}), down={sums['down']:.3f} (x{f_down})")

        for s in sample_names:
            for ds, h in histo_nom[s].items():
                current_vars = list(h.axes["variation"])
                if up_name in current_vars:
                    continue  # already added
                new_axis = StrCategory(current_vars + [up_name, down_name], name="variation")
                other_axes = [ax for ax in h.axes if ax.name not in ("cat", "variation")]
                new_hist = Hist(h.axes["cat"], new_axis, *other_axes, storage=h.storage_type())

                old_view = h.view(flow=True)
                new_view = new_hist.view(flow=True)
                for v in current_vars:
                    new_view[:, new_axis.index(v), :] = old_view[:, h.axes["variation"].index(v), :]

                up_idx = new_axis.index(up_name)
                down_idx = new_axis.index(down_name)
                nom = old_view[:, h.axes["variation"].index("nominal"), :]

                # Default: variations = nominal everywhere (other cats, other years)
                for idx in (up_idx, down_idx):
                    new_view[:, idx, :]["value"] = nom["value"]
                    new_view[:, idx, :]["variance"] = nom["variance"]

                # Fill the real variation only for this category and year
                if year in ds:
                    ci = h.axes["cat"].index(cat)
                    for idx, src, f in ((up_idx, histo_up, f_up), (down_idx, histo_down, f_down)):
                        if f is None:
                            continue
                        src_nom = _nom_view(src[s][ds])[ci]
                        new_view[ci, idx, :]["value"] = np.nan_to_num(src_nom["value"] * f)
                        new_view[ci, idx, :]["variance"] = np.nan_to_num(src_nom["variance"] * f**2)

                histo_nom[s][ds] = new_hist


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

def add_Madgraph_systematic_1d(histo_1d, cat, integral_eps=1e-6, ratio_source=None):
    """Function to add the Madgraph systematic uncertainty to the histogram.

    integral_eps: below this integral (summed events in a given cat/dataset),
    a variation is treated as "no data" and the systematic is set to have
    no effect (up == down == nominal) instead of being renormalized, which
    is what produces `nan`/`inf` when the process is (almost) empty in a
    given category.
    """
    if ratio_source is None:
        ratio_source = histo_1d

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
        mu_ratio_src = ratio_source[mu_name]
        mg_ratio_src = ratio_source[mg_name]
        mu_total = None
        mg_total = None
        for h in mu_ratio_src.values():
            h_nom = h[cat, "nominal", :]
            mu_total = h_nom if mu_total is None else mu_total + h_nom
        for h in mg_ratio_src.values():
            h_nom = h[cat, "nominal", :]
            mg_total = h_nom if mg_total is None else mg_total + h_nom
        mu_vals = mu_total.values(flow=True)
        mg_vals = mg_total.values(flow=True)

        mu_sum = mu_vals.sum()
        mg_sum = mg_vals.sum()
        if mu_sum <= integral_eps or mg_sum <= integral_eps:
            print(f"Skipping flavor {flav} in cat {cat}: MuEnriched or Madgraph "
                  f"total is ~0 (mu_sum={mu_sum}, mg_sum={mg_sum})")
            ratio = None  # signal "no meaningful ratio" below
        else:
            scale = mu_sum / mg_sum
            mg_vals_scaled = mg_vals * scale
            mask = mu_vals > 1e-9
            ratio = np.ones_like(mu_vals)
            ratio[mask] = mg_vals_scaled[mask] / mu_vals[mask]
            ratio = np.clip(ratio, 0, 2)
            ratio = np.nan_to_num(ratio, nan=1.0, posinf=2.0, neginf=0.0)

        for dataset, h_mu in mu_datasets.items():
            print(f"\ndataset: {dataset}")
            current_vars = list(h_mu.axes["variation"])
            new_vars = current_vars + ["QCD_MuEnriched_ratioUp", "QCD_MuEnriched_ratioDown"]
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

            nom_integral = np.nansum(nom_view["value"])

            if ratio is None or nom_integral <= integral_eps:
                # Nothing meaningful to reweight for this dataset/category:
                # make the systematic a no-op (up == down == nominal) instead
                # of dividing by ~0.
                up_view["value"] = nom_view["value"]
                up_view["variance"] = nom_view["variance"]
                down_view["value"] = nom_view["value"]
                down_view["variance"] = nom_view["variance"]
                histo_1d[mu_name][dataset] = new_hist
                continue

            up_view["value"] = nom_view["value"] * ratio
            up_view["variance"] = nom_view["variance"] * ratio**2
            down_view["value"] = nom_view["value"] * (2 - ratio)
            down_view["variance"] = nom_view["variance"] * (2 - ratio)**2

            nominal_integral = np.nansum(new_hist.view(flow=True)[cat_idx, nom_idx, :]["value"])
            up_integral = np.nansum(up_view["value"])
            down_integral = np.nansum(down_view["value"])
            print(f"nominal_integral = {nominal_integral}\nup_integral = {up_integral}\ndown_integral = {down_integral}")

            if up_integral > integral_eps:
                up_factor = nominal_integral / up_integral
                new_hist.view(flow=True)[cat_idx, up_idx, :]["value"] *= up_factor
                new_hist.view(flow=True)[cat_idx, up_idx, :]["variance"] *= up_factor**2
            else:
                print("Warning: up_integral ~ 0, skipping renormalization, forcing up = nominal")
                up_view["value"] = nom_view["value"]
                up_view["variance"] = nom_view["variance"]

            if down_integral > integral_eps:
                down_factor = nominal_integral / down_integral
                new_hist.view(flow=True)[cat_idx, down_idx, :]["value"] *= down_factor
                new_hist.view(flow=True)[cat_idx, down_idx, :]["variance"] *= down_factor**2
            else:
                print("Warning: down_integral ~ 0, skipping renormalization, forcing down = nominal")
                down_view["value"] = nom_view["value"]
                down_view["variance"] = nom_view["variance"]

            # Final belt-and-braces: kill any nan/inf that could still have
            # crept in (e.g. from earlier variations copied via `current_vars`).
            new_hist.view(flow=True)[cat_idx, :, :]["value"] = np.nan_to_num(
                new_hist.view(flow=True)[cat_idx, :, :]["value"], nan=0.0, posinf=0.0, neginf=0.0
            )
            new_hist.view(flow=True)[cat_idx, :, :]["variance"] = np.nan_to_num(
                new_hist.view(flow=True)[cat_idx, :, :]["variance"], nan=0.0, posinf=0.0, neginf=0.0
            )

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
    if tau21_cut >= ax_tau21.edges[-1]:
        bin_stop = len(ax_tau21.edges) - 2
    else:
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
    if tau21_cut >= ax_tau21.edges[-1]:
        bin_stop = len(ax_tau21.edges) - 2
    else:
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
            if tau21_cut >= ax_tau21.edges[-1]:
                bin_stop = len(ax_tau21.edges) - 2
            else:
                bin_stop = next(i for i, edge in enumerate(ax_tau21.edges[1:]) if edge > tau21_cut)
            histo_cut = histo2d.integrate(ax_tau21.name, 0, bin_stop)
            h1d_dict[proc][ds] = histo_cut
    # print(f"{h1d_dict.keys()}\n")
    return h1d_dict


def get_1d_histogram_reweighed(h2d_dict, tau21_cut, samples, year, parent_category, fixed_weights=None, return_weights=False):
    """Return 1D histograms with MC (b+c+light) reweighted to data.

    The input 2D histograms are first integrated over the tau21 axis as in
    get_1d_histogram, using the cut ``tau21 < tau21_cut``. Then, for the
    specified year and a given parent category, a bin-by-bin weight is
    computed such that, in the *inclusive pass+fail region* for that parent
    category, the sum of all MC histograms (b + c + light) equals the data
    histogram. Those weights are applied to the MC templates in both the
    corresponding pass and fail categories, leaving data unchanged.
    """

    def _ret(histo, weight):
        return (histo, weight) if return_weights else histo
    
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
        return _ret(h1d_dict, None)

    # Axes: ["cat", "variation", fit_variable]
    cat_axis = example_hist.axes["cat"]
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
        return _ret(h1d_dict, None)

    n_fit_bins = len(fit_axis.edges) - 1

    # Define MC and data sample sets
    mc_sample_names = set(samples["light"] + samples["c"] + samples["b"])
    data_sample_names = set(samples["data_obs"])

    if fixed_weights is not None:
        weights = np.asarray(fixed_weights, dtype=float)
        if weights.shape != (n_fit_bins,):
            raise RuntimeError(f"fixed_weights has shape {weights.shape}, expected ({n_fit_bins},)")

    else:
        mc_sum = np.zeros(n_fit_bins, dtype=float)
        data_sum = np.zeros(n_fit_bins, dtype=float)
        # Build inclusive (pass+fail) distributions for the requested year and parent category
        for proc_name, ds_dict in h1d_dict.items():
            for ds, h in ds_dict.items():
                # Restrict to the datasets of the current year
                if year not in ds:
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
                    var_axis = h.axes["variation"]
                    nom_index = var_axis.index("nominal")
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
            if year not in ds:
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

    return _ret(h1d_dict, weights)


def _merge_low_stat_bin_groups(arr_a, arr_b, min_yield):
    """Group contiguous bin indices [0, len(arr_a)) so that, within every
    group, both ``arr_a`` and ``arr_b`` sum to at least ``min_yield``.

    Bins are accumulated left to right; a group is closed as soon as both
    running sums reach the threshold. Any leftover low-statistics tail (not
    enough to form its own group) is merged into the last group instead of
    being left under-threshold.

    Returns a list of (start, stop) tuples (stop exclusive) covering the
    full range.
    """
    n_bins = len(arr_a)
    groups = []
    start = 0
    running_a = 0.0
    running_b = 0.0
    for i in range(n_bins):
        running_a += arr_a[i]
        running_b += arr_b[i]
        if running_a >= min_yield and running_b >= min_yield:
            groups.append((start, i + 1))
            start = i + 1
            running_a = 0.0
            running_b = 0.0
    if start < n_bins:
        if groups:
            groups[-1] = (groups[-1][0], n_bins)
        else:
            groups.append((start, n_bins))
    return groups


def get_1d_histogram_reweighted_to_signal(
    h2d_dict, tau21_cut, samples, year, parent_category, min_bin_yield=50.0
):
    """Return 1D histograms with the b-flavor MC (the b-tag-calibration proxy)
    shape-corrected, bin-by-bin, to match the shape of the HH signal
    sample(s) (``samples["signal"]``), plus a dict of diagnostic info for
    plotting and saving.

    This mirrors ``get_1d_histogram_reweighed``, but is a *shape-only*
    correction: weights are derived once in the *inclusive pass+fail
    region* of ``parent_category`` for the given ``year``, from the ratio
    of the signal and b-proxy histograms after normalizing both to the
    same total, and applied to every "b" MC template (all variations) in
    the corresponding pass and fail categories. This preserves the
    b-proxy's overall normalization (its total yield is unchanged) and
    only reshapes it to match the signal; a plain bin-by-bin ratio would
    instead force the b-proxy to become numerically equal to the signal
    (a far more drastic, normalization-changing correction). Bins are
    first merged (using the original fine binning) so that both the
    b-proxy and the signal histogram have at least ``min_bin_yield``
    summed weight in every merged bin - keeping the statistical
    uncertainty on the derived weights under control. This merged binning
    is used *only* to derive the per-group weights: each original
    (fine-binned) bin then inherits the weight of the merged group it
    belongs to (a step function over the merged groups), so the returned
    histograms keep the original binning instead of the coarser one used
    to derive the weights.

    Returns:
        (h1d_dict, info) where h1d_dict is the reweighted 1D histogram
        dict on the *original* binning (same nested structure as
        get_1d_histogram), and
        info is None if there was nothing to reweight (parent category not
        present, or no signal/b-proxy statistics), otherwise a dict with:
            "bin_edges": merged bin edges (list of float, length n+1)
            "weights": per-bin shape-correction factor (list of float, length n)
            "weights_stat_unc": statistical uncertainty on each weight
            "b_proxy_yield_pre": merged b-proxy yield before reweighting
            "signal_yield": merged signal yield (raw, not normalized to the
                b-proxy total - use together with "weights" to see the shape)
    """

    h1d_dict = get_1d_histogram(h2d_dict, tau21_cut)
    # The signal shape is always taken fully inclusive in tau21 (cut = 1.0,
    # i.e. no tau21 selection): tau21 is a proxy-purification knob and must
    # never be applied to the actual signal when deriving the target shape.
    h1d_dict_signal = get_1d_histogram(h2d_dict, 1.0)

    example_hist = None
    for ds_dict in h1d_dict.values():
        for h in ds_dict.values():
            if "variation" in [ax.name for ax in h.axes]:
                example_hist = h
                break
        if example_hist is not None:
            break

    if example_hist is None:
        return h1d_dict, None

    cat_axis = example_hist.axes["cat"]
    fit_axes = [ax for ax in example_hist.axes if ax.name not in ("cat", "variation")]
    if len(fit_axes) != 1:
        raise RuntimeError("Expected exactly one fit variable axis after tau21 integration")
    fit_axis = fit_axes[0]
    original_edges = np.asarray(fit_axis.edges, dtype=float)

    pass_cat_label = f"{parent_category}-pass"
    fail_cat_label = f"{parent_category}-fail"
    cat_indices = []
    for label in (pass_cat_label, fail_cat_label):
        try:
            cat_indices.append(cat_axis.index(label))
        except KeyError:
            continue

    if not cat_indices:
        return h1d_dict, None

    n_fit_bins = len(original_edges) - 1

    b_sample_names = set(samples["b"])
    signal_sample_names = set(samples["signal"])

    if not signal_sample_names:
        print("No signal samples found (samples['signal'] is empty); skipping reweight-to-signal.")
        return h1d_dict, None

    b_sum = np.zeros(n_fit_bins, dtype=float)
    b_var = np.zeros(n_fit_bins, dtype=float)
    signal_sum = np.zeros(n_fit_bins, dtype=float)
    signal_var = np.zeros(n_fit_bins, dtype=float)

    def _accumulate(source_dict, wanted_names, sum_arr, var_arr):
        for proc_name, ds_dict in source_dict.items():
            if proc_name not in wanted_names:
                continue
            for ds, h in ds_dict.items():
                if year not in ds:
                    continue
                view = h.view(flow=False)
                values_view = view["value"]
                variances_view = view["variance"]

                if values_view.ndim == 3:
                    var_axis = h.axes["variation"]
                    nom_index = var_axis.index("nominal")
                    proj_val = values_view[cat_indices, nom_index, :].sum(axis=0)
                    proj_var = variances_view[cat_indices, nom_index, :].sum(axis=0)
                elif values_view.ndim == 2:
                    proj_val = values_view[cat_indices, :].sum(axis=0)
                    proj_var = variances_view[cat_indices, :].sum(axis=0)
                else:
                    raise RuntimeError(
                        f"Unsupported histogram dimensionality {values_view.ndim} in reweighting (expected 2 or 3)"
                    )

                sum_arr += proj_val
                var_arr += proj_var

    # b-proxy: at the requested tau21 cut (the selection being purified)
    _accumulate(h1d_dict, b_sample_names, b_sum, b_var)
    # signal: always fully inclusive in tau21
    _accumulate(h1d_dict_signal, signal_sample_names, signal_sum, signal_var)

    if signal_sum.sum() <= 0:
        print(f"Signal histogram is empty for year {year}, parent category {parent_category}; skipping reweight-to-signal.")
        return h1d_dict, None

    # Merge low-statistics bins so both the b-proxy and the signal have at
    # least `min_bin_yield` summed weight in every merged bin.
    groups = _merge_low_stat_bin_groups(signal_sum, b_sum, min_bin_yield)
    merged_edges = [original_edges[groups[0][0]]] + [original_edges[stop] for _, stop in groups]

    def _merge(arr):
        return np.array([arr[start:stop].sum() for start, stop in groups])

    b_sum_m = _merge(b_sum)
    b_var_m = _merge(b_var)
    signal_sum_m = _merge(signal_sum)
    signal_var_m = _merge(signal_var)

    total_b = b_sum_m.sum()
    total_signal = signal_sum_m.sum()
    if total_b <= 0:
        print(f"b-proxy histogram is empty for year {year}, parent category {parent_category}; skipping reweight-to-signal.")
        return h1d_dict, None
    # Shape-only correction: normalize both histograms to the same total
    # before taking the bin ratio, so the b-proxy keeps its own overall
    # normalization and only its *shape* is corrected to match the signal.
    # A plain bin-by-bin ratio (signal/b-proxy) would instead force the
    # b-proxy to become numerically equal to the (much smaller) signal
    # yield, which is a far more drastic change than intended here.
    norm_factor = total_b / total_signal

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_ratio = np.where(b_sum_m > 0.0, signal_sum_m / b_sum_m, 1.0)
        raw_ratio = np.nan_to_num(raw_ratio, nan=1.0, posinf=1.0, neginf=1.0)
        weights = raw_ratio * norm_factor

        # Relative uncertainty is scale-invariant, so the per-bin relative
        # uncertainty of the shape-corrected weight is the same as that of
        # the raw ratio (the shared normalization factor is treated as
        # exact, i.e. dominated by the much larger inclusive statistics).
        rel_unc_signal = np.where(signal_sum_m > 0, np.sqrt(signal_var_m) / signal_sum_m, 0.0)
        rel_unc_b = np.where(b_sum_m > 0, np.sqrt(b_var_m) / b_sum_m, 0.0)
        weights_stat_unc = weights * np.sqrt(rel_unc_signal ** 2 + rel_unc_b ** 2)
        weights_stat_unc = np.nan_to_num(weights_stat_unc, nan=0.0, posinf=0.0, neginf=0.0)

    info = {
        "bin_edges": [float(x) for x in merged_edges],
        "weights": [float(x) for x in weights],
        "weights_stat_unc": [float(x) for x in weights_stat_unc],
        "b_proxy_yield_pre": [float(x) for x in b_sum_m],
        "signal_yield": [float(x) for x in signal_sum_m],
    }

    # Broadcast the coarse, low-noise per-group weights back onto the
    # original fine binning (each fine bin inherits its group's weight, i.e.
    # a step function over the merged groups) so the returned histograms
    # keep the original binning instead of the merged one used only to
    # derive stable weights.
    fine_weights = np.empty(n_fit_bins, dtype=float)
    for weight, (start, stop) in zip(weights, groups):
        fine_weights[start:stop] = weight

    # Apply weights to the "b" MC templates (all variations) in the pass+fail
    # categories of this parent.
    for proc_name, ds_dict in h1d_dict.items():
        if proc_name not in b_sample_names:
            continue
        for ds, h in ds_dict.items():
            if year not in ds:
                continue
            view = h.view(flow=False)
            values_view = view["value"]

            if values_view.ndim == 3:
                scale = fine_weights[np.newaxis, np.newaxis, :]
                view["value"][cat_indices, :, :] *= scale
                view["variance"][cat_indices, :, :] *= scale ** 2
            elif values_view.ndim == 2:
                scale = fine_weights[np.newaxis, :]
                view["value"][cat_indices, :] *= scale
                view["variance"][cat_indices, :] *= scale ** 2
            else:
                raise RuntimeError(
                    f"Unsupported histogram dimensionality {values_view.ndim} in reweighting (expected 2 or 3)"
                )

    return h1d_dict, info


def plot_reweight_to_signal(info, year, parent_category, tau21_cut, outdir):
    """Plot b-proxy pre-/post-reweighting compared with the signal shape,
    plus the per-bin reweighting factors with their statistical
    uncertainty, and save the figure to ``outdir``.
    Printing normalised histogram to see only shape differences.
    """
    edges = np.array(info["bin_edges"])
    b_pre = np.array(info["b_proxy_yield_pre"])
    signal = np.array(info["signal_yield"])
    weights = np.array(info["weights"])
    weights_unc = np.array(info["weights_stat_unc"])
    b_post = b_pre * weights
    centers = 0.5 * (edges[1:] + edges[:-1])

    b_pre = b_pre/np.sum(b_pre)
    signal = signal/np.sum(signal)
    b_post = b_post/np.sum(b_post)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.stairs(b_pre, edges, label="b-proxy (pre-reweight)", color="tab:blue")
    ax1.stairs(b_post, edges, label="b-proxy (post-reweight)", color="tab:orange")
    ax1.stairs(signal, edges, label="Signal (HH4b)", color="black")
    ax1.set_yscale("log")
    ax1.set_ylabel("Events")
    ax1.set_title(f"{parent_category}, {year}, b-proxy tau21 < {tau21_cut:.2f} (inclusive pass+fail)")
    ax1.legend()

    ax2.errorbar(centers, weights, yerr=weights_unc, fmt="o", color="tab:red", label="signal / b-proxy")
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax2.set_yscale("log")
    ax2.set_ylabel("Weight")
    ax2.set_xlabel(r"log($\sum({m^{corr}_{SV}})$)")
    ax2.legend()

    plt.tight_layout()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"reweight_to_signal_{parent_category}_{year}.png"
    plt.savefig(outfile)
    plt.close(fig)
    print(f"Saved reweight-to-signal comparison plot: {outfile}")
    return outfile


def sanitize_shape_variations(histo_1d, mc_sample_names, epsilon=1e-9):
    """Enforce nominal/variation consistency and remove nan/inf.

    Rule: wherever the nominal value in a bin is ~0, force every variation
    in that bin to 0 as well. This prevents Combine's kappa/extra-norm
    computation (integral(up)/integral(nominal)) from ever seeing a
    variation with content in a bin where the nominal has none, which is
    the most common source of `Bogus norm nan`.
    """
    for proc_name, ds_dict in histo_1d.items():
        if proc_name not in mc_sample_names:
            continue
        for ds, h in ds_dict.items():
            if "variation" not in [ax.name for ax in h.axes]:
                continue
            var_axis = h.axes["variation"]
            if "nominal" not in list(var_axis):
                continue
            nom_idx = var_axis.index("nominal")
            view = h.view(flow=True)

            # nan/inf cleanup everywhere first
            view["value"] = np.nan_to_num(view["value"], nan=0.0, posinf=0.0, neginf=0.0)
            view["variance"] = np.nan_to_num(view["variance"], nan=0.0, posinf=0.0, neginf=0.0)

            nom_values = view["value"][:, nom_idx, :]
            zero_mask = nom_values <= epsilon  # shape: (n_cat, n_fit)

            for v_idx in range(view.shape[1]):
                if v_idx == nom_idx:
                    continue
                view["value"][:, v_idx, :] = np.where(zero_mask, 0.0, view["value"][:, v_idx, :])
                view["variance"][:, v_idx, :] = np.where(zero_mask, 0.0, view["variance"][:, v_idx, :])

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
    parser.add_argument("-f","--filter-category", default="", help="Substring that must be contained in category to produce datacard.")
    parser.add_argument("--min-reweight-signal-yield", type=float, default=50.0,
                       help="Minimum summed yield (in both the b-proxy and the signal histogram) required to keep a bin "
                            "un-merged when deriving the b-proxy-to-signal reweighting.")
    parser.add_argument("--no-tau21-var", dest="tau21_var", action="store_false", default=True,
                        help="Disable the tau21_var_pass/tau21_var_fail shape nuisances")
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
    if not args.filter_category:
        categories = [cat for cat in cutflow.keys() if cat.startswith('msd')]
    else:
        categories = [cat for cat in cutflow.keys() if cat.startswith('msd') and args.filter_category in cat]

    
    # Categorize samples
    samples = categorize_samples(cutflow)
    print(f"\nFound samples: {samples}")
    print(f"Available histograms: {list(histograms.keys())}\n")

    successful_categories = []
    failed_categories = []

    for year in args.years:
        # Define processes and systematics
        mc_processes, data_processes = define_processes(samples, [year])
        print(f"year: {year}")
        print(f"MC processes: {mc_processes.items()}")
        print(f"DATA processes: {data_processes.items()}\n")
        
        # Update process samples based on what we found
        for process_name, process in mc_processes.items():
            process.samples = samples[process_name]
        
        for process_name, process in data_processes.items():
            process.samples = samples[process_name]

        # Add the variation QCD_Madgraph/QCD_MuEnriched to the Hist
        # add_Madgraph_systematic(histograms[args.variable])

        mc_names = [p_name for p_name, p in mc_processes.items()]
        systematics = define_systematics([year], mc_names)
        systematics_tau21_by_region = {
            region: define_systematics([year], mc_names, tau21_var_region=region)
            for region in ["pass", "fail"]
        }
        print(f"systematics: {systematics}\n")
        print(f"systematics_tau21: {systematics_tau21_by_region}\n")

        # Create output directory
        if args.output_dir is None:
            args.output_dir = str(Path(args.input_file).parent / "datacards")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Dictionary to store all datacards for combination
        all_datacards = defaultdict(dict)
        # Additional datacards using MC reweighted to data for tau21 < 0.30
        all_datacards_reweight = defaultdict(dict)
        # Additional datacards using the "b" proxy reweighted to the HH4b signal shape for tau21 < 0.30
        all_datacards_reweight_signal = defaultdict(dict)
        
        # Create datacards for each combination
        for cat in categories:
            print(f"\ncategory: {cat}")
            region = cat.split("-")[-1]
            tau21_syst_name = f"tau21_var_{region}"
            systematics_tau21 = systematics_tau21_by_region[region] if args.tau21_var else systematics

            # for tau21 in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
            for tau21 in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
            # for tau21 in [0.60]:
            # for tau21 in [0.2, 0.25, 0.3, 0.35, 0.4]:
                print(f"\n\nCreating datacard: Year: {year}\tCategory: {cat}\ttau21 < {tau21}")
                
                is_tau21_nominal = abs(tau21 - TAU21_NOMINAL) < 1e-6
                mc_sample_names = set(samples["light"] + samples["c"] + samples["b"])
                # Get the 1D histogram by integrating over tau21 axis with a specific cut: tau21 < tau21_cut
                histo_1d = get_1d_histogram(histograms[args.variable], tau21)
                # Add the variation QCD_Madgraph/QCD_MuEnriched to the Hist
                add_Madgraph_systematic_1d(histo_1d, cat)
                if is_tau21_nominal and args.tau21_var:
                    add_tau21_variation_1d(
                        histo_1d,
                        histo_down=get_1d_histogram(histograms[args.variable], TAU21_VAR_DOWN),
                        histo_up=get_1d_histogram(histograms[args.variable], TAU21_VAR_UP),
                        cat=cat, mc_processes=mc_processes, year=year,
                        syst_name=tau21_syst_name,
                    )
                sanitize_shape_variations(histo_1d, mc_sample_names)
                print("\n")
                # Create datacard
                datacard = DatacardMutag(
                    histograms=histo_1d,
                    datasets_metadata=datasets_metadata,
                    cutflow=cutflow,
                    years=[year],
                    mc_processes=mc_processes,
                    data_processes=data_processes,
                    systematics=systematics_tau21 if is_tau21_nominal else systematics,
                    category=cat,  # Category string matching the multicuts structure
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
                    histo_1d_raw = get_1d_histogram(histograms[args.variable], tau21)
                    histo_1d_rew, rew_weights = get_1d_histogram_reweighed(
                        histograms[args.variable], tau21, samples, year, parent_category, return_weights=True,
                    )
                    # Add the variation QCD_Madgraph/QCD_MuEnriched to the Hist
                    add_Madgraph_systematic_1d(histo_1d_rew, cat, ratio_source=histo_1d_raw)
                    if args.tau21_var:
                        if rew_weights is not None:
                            histo_down_rew = get_1d_histogram_reweighed(histograms[args.variable], TAU21_VAR_DOWN, samples, year, parent_category, fixed_weights=rew_weights)
                            histo_up_rew = get_1d_histogram_reweighed(histograms[args.variable], TAU21_VAR_UP, samples, year, parent_category, fixed_weights=rew_weights)
                        else:
                            histo_down_rew = get_1d_histogram(histograms[args.variable], TAU21_VAR_DOWN)
                            histo_up_rew = get_1d_histogram(histograms[args.variable], TAU21_VAR_UP)

                        add_tau21_variation_1d(
                            histo_1d_rew,
                            histo_down=histo_down_rew,
                            histo_up=histo_up_rew,
                            cat=cat, mc_processes=mc_processes, year=year,
                            syst_name=tau21_syst_name,
                        )
                    sanitize_shape_variations(histo_1d_rew, mc_sample_names)

                    print("\n")
                    datacard_rew = DatacardMutag(
                        histograms=histo_1d_rew,
                        datasets_metadata=datasets_metadata,
                        cutflow=cutflow,
                        years=[year],
                        mc_processes=mc_processes,
                        data_processes=data_processes,
                        systematics=systematics_tau21,
                        category=cat,
                        verbose=args.verbose,
                    )
                    all_datacards_reweight[cat][tau21] = datacard_rew

                    # Also reweight the "b" proxy templates (bin-by-bin) to
                    # the HH4b signal shape, merging low-statistics bins
                    # first, and produce a comparison plot + saved
                    # reweighting values.
                    print(f"\n\nCreating datacard: Year: {year}\tCategory: {cat}\ttau21 < {tau21} reweighed to signal")
                    histo_1d_rew_sig, reweight_sig_info = get_1d_histogram_reweighted_to_signal(
                        histograms[args.variable], tau21, samples, year, parent_category,
                        min_bin_yield=args.min_reweight_signal_yield,
                    )
                    if reweight_sig_info is not None:
                        add_Madgraph_systematic_1d(histo_1d_rew_sig, cat, ratio_source=histo_1d_raw)
                        print("\n")
                        datacard_rew_sig = DatacardMutag(
                            histograms=histo_1d_rew_sig,
                            datasets_metadata=datasets_metadata,
                            cutflow=cutflow,
                            years=[year],
                            mc_processes=mc_processes,
                            data_processes=data_processes,
                            systematics=systematics,
                            category=cat,
                            verbose=args.verbose,
                        )
                        all_datacards_reweight_signal[cat][tau21] = datacard_rew_sig

                        # Only produce the diagnostic plot/YAML once per
                        # (year, parent_category), not once per pass/fail.
                        if cat.endswith("-pass"):
                            reweight_sig_dir = output_dir / year / parent_category / f"{get_tau21_str(tau21)}_reweight_signal"
                            reweight_sig_dir.mkdir(parents=True, exist_ok=True)
                            plot_reweight_to_signal(reweight_sig_info, year, parent_category, tau21, reweight_sig_dir)
                            info_file = reweight_sig_dir / "reweight_to_signal.yaml"
                            print(f"Saving reweight-to-signal values to {info_file}")
                            with open(info_file, "w") as f:
                                yaml.dump(reweight_sig_info, f, indent=4)

        passfail_ratio = get_passfail_ratio(all_datacards)

        # Loop over categories again to dump datacards modified with pass/fail ratios
        parent_categories = set()
        for cat in categories:
            # for tau21 in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
            for tau21 in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
            # for tau21 in [0.60]:
            # for tau21 in [0.2, 0.25, 0.3, 0.35, 0.4]:
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

                # For tau21 < 0.30, also dump the signal-reweighted datacards (if any were produced)
                if abs(tau21 - 0.3) < 1e-6 and cat in all_datacards_reweight_signal and tau21 in all_datacards_reweight_signal[cat]:
                    reweight_sig_tau21_str = f"{tau21_str}_reweight_signal"
                    reweight_sig_category_dir = output_dir / year / parent_category / reweight_sig_tau21_str / region
                    reweight_sig_category_dir.mkdir(parents=True, exist_ok=True)
                    datacard_rew_sig = all_datacards_reweight_signal[cat][tau21]

                    if cat.endswith("-pass"):
                        kwargs_rew_sig = {"directory": str(reweight_sig_category_dir)}
                    elif cat.endswith("-fail"):
                        parent_cat = '-'.join(cat.split("-")[:-1])
                        kwargs_rew_sig = {"directory": str(reweight_sig_category_dir), "passfail_ratio": passfail_ratio[parent_cat][tau21]}
                    else:
                        kwargs_rew_sig = {"directory": str(reweight_sig_category_dir)}

                    try:
                        datacard_rew_sig.dump(**kwargs_rew_sig)
                        successful_categories.append({"year": year, "category": f"{cat}_reweight_signal", "folder": str(reweight_sig_category_dir)})
                    except Exception as e:
                        print(f"Failed to create signal-reweighted datacard for Year: {year}, Category: {cat}")
                        print(str(e))
                        failed_categories.append({"year": year, "category": f"{cat}_reweight_signal", "error": str(e)})

        # Create combined datacard for pass+fail regions, for each parent category
        for parent_cat in parent_categories:
            # for tau21 in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
            for tau21 in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
            # for tau21 in [0.60]:
            # for tau21 in [0.2, 0.25, 0.3, 0.35, 0.4]:
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

                # For tau21 < 0.30, also create the combined signal-reweighted datacard (if both regions were produced)
                if abs(tau21 - 0.3) < 1e-6 and all(
                    f"{parent_cat}-{region}" in all_datacards_reweight_signal
                    and tau21 in all_datacards_reweight_signal[f"{parent_cat}-{region}"]
                    for region in ["pass", "fail"]
                ):
                    reweight_sig_tau21_str = f"{tau21_str}_reweight_signal"
                    directory_rew_sig = output_dir / year / parent_cat / reweight_sig_tau21_str
                    print(f"\nCreating combined signal-reweighted datacard for category: {parent_cat} with tau21 < {tau21} (pass + fail)")
                    combine_datacards(
                        datacards={f"{region}/datacard.txt": all_datacards_reweight_signal[f"{parent_cat}-{region}"][tau21] for region in ["pass", "fail"]},
                        directory=directory_rew_sig,
                    )
                    filename_rew_sig = directory_rew_sig / "passfail_ratio.yaml"
                    print(f"Saving pass/fail ratio to {filename_rew_sig}")
                    with open(filename_rew_sig, "w") as f:
                        yaml.dump({"passfail_ratio": passfail_ratio[parent_cat][tau21]}, f, indent=4)
                    print(f"Combined signal-reweighted datacard saved in {directory_rew_sig}")

    # Print summary report
    print_report(successful_categories, failed_categories)

if __name__ == "__main__":
    main()
