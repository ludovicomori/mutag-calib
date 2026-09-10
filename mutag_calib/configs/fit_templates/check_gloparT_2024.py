from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_eq, get_nObj_min, get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.categorization import CartesianSelection, MultiCut

from pocket_coffea.lib.calibrators.common.common import JetsCalibrator, JetsSoftdropMassCalibrator
from pocket_coffea.lib.weights.common.common import common_weights
from pocket_coffea.parameters.histograms import *
import mutag_calib
from mutag_calib.configs.fatjet_base.custom.cuts import get_ptmsd, get_ptmsd_window, get_two_jet_ptmsd, get_mregbin, get_nObj_minmsd, get_flavor, get_ptbin, get_msdbin, get_tau21, mutag_fatjet_sel, mutag_fatjet_sel_matched
from mutag_calib.configs.fatjet_base.custom.functions import get_inclusive_wp, get_tagger_pass
from mutag_calib.configs.fatjet_base.custom.weights import SF_trigger_prescale
import mutag_calib.workflows.mutag_oneMuAK8_processor as workflow
from mutag_calib.workflows.mutag_oneMuAK8_processor import mutagAnalysisOneMuonInAK8Processor
import numpy as np
import os

localdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Optional boosted HH->4b BDT evaluation (ggF + VBF outputs).
# Set BDT_MODEL = None to disable and run this config exactly as before.
# When set, the mutagAnalysisOneMuonInAK8BDTProcessor rebuilds the BDT inputs
# (di-Higgs system + AK4 VBF jets) from the two leading FatJetGood and stores
# per-event scores; the 2D score-vs-tau21 histograms below are then added.
# ---------------------------------------------------------------------------
BDT_MODEL = "/work/tharte/datasets/bdts/boosted/bdt_trainings_run3/25Feb5_v13_glopartv2_rawmass/trained_bdt.model"

if BDT_MODEL:
    import mutag_calib.workflows.mutag_bdt_processor as bdt_workflow
    from mutag_calib.workflows.mutag_bdt_processor import mutagAnalysisOneMuonInAK8BDTProcessor

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                f"{localdir}/params/object_preselection_HHbbbb_glopartcheck.yaml",
                                                f"{localdir}/params/jets_calibration.yaml",
                                                f"{localdir}/params/triggers_run3.yaml",
                                                f"{localdir}/params/triggers_prescales_run3.yaml",
                                                f"{localdir}/params/ptetatau21_reweighting_HHbbbb.yaml",
                                                # f"{localdir}/params/mutag_calibration_HHbbbb_2024.yaml",
                                                f"{localdir}/params/mutag_calibration_HHbbbb_2024_glopartcheck.yaml",
                                                f"{localdir}/params/plotting_style.yaml",
                                                update=True)

if BDT_MODEL:
    # JetVBF / JetNearFatJet object_preselection blocks for the AK4 VBF-jet
    # reconstruction used to build the BDT inputs.
    parameters = defaults.merge_parameters_from_files(
        parameters, f"{localdir}/params/bdt_vbf_jets.yaml", update=True
    )

samples = [
    "GluGluHHto4B_Par-c2-0p00-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8",
    # "QCD_Madgraph",
    # "QCD_MuEnriched",
    # "VJets",
    # "TTto4Q",
    # "SingleTop",
    # "DATA_BTagMu"
]

subsamples = {}
for s in filter(lambda x: 'DATA_BTagMu' not in x and 'GluGlu' not in x, samples):
    subsamples[s] = {f"{s}_{f}" : [get_flavor(f)] for f in ['l', 'c', 'b', 'cc', 'bb']}
# subsamples = {}
# for s in filter(lambda x: 'GluGlu' not in x and 'DATA_BTag' not in x, samples):
#     subsamples[s] = {f"{s}_{str(f).replace('.', '_')}": [get_tau21(f), get_flavor('b')] for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

variables = {
    #**count_hist(name="nFatJetGood", coll="FatJetGood",bins=10, start=0, stop=10),
    #**count_hist(coll="FatJetGoodNMuon1",bins=10, start=0, stop=10),
    #**count_hist(coll="FatJetGoodNMuon2",bins=10, start=0, stop=10),
    #**count_hist(coll="FatJetGoodNMuonSJ1",bins=10, start=0, stop=10),
    #**count_hist(coll="FatJetGoodNMuonSJUnique1",bins=10, start=0, stop=10),
}

#collections = ["FatJetGoodNMuon1", "FatJetGoodNMuon2", "FatJetGoodNMuonSJ1", "FatJetGoodNMuonSJUnique1"]
collections = ["FatJetGood"]

for coll in collections:
    # variables.update(**fatjet_hists(coll=coll))
    variables[f"{coll}_pt"] = HistConf([Axis(name=f"{coll}_pt", coll=coll, field="pt",
                                                    label=r"FatJet $p_{T}$ [GeV]", bins=list(range(250, 1010, 10)))]
    )
    variables[f"{coll}_msoftdrop"] = HistConf([Axis(name=f"{coll}_msoftdrop", coll=coll, field="msoftdrop",
                                                           label=r"FatJet $m_{SD}$ [GeV]", bins=list(range(0, 410, 10)))]
    )
    variables[f"{coll}_tau21"] = HistConf([Axis(name=f"{coll}_tau21", coll=coll, field="tau21",
                                                           label=r"FatJet $\tau_{21}$", bins=[0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 
                                                           0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1])]
    )
    variables[f"{coll}_nSVMatchedToFatJetGood"] = HistConf([Axis(name=f"{coll}_nSVMatchedToFatJetGood", coll=coll, field="nSVMatchedToFatJetGood",
                                                           label=r"nSVMatchedToFatJetGood", bins=[0,1,2,3,4,5,6,7,8,9,10,11,12])]
    )
    variables[f"{coll}_logsumcorrSVmass"] = HistConf(
        [ Axis(coll="FatJetGood", field="logsumcorrSVmass", label=r"log($\sum({m^{corr}_{SV}})$)", bins=42, start=-2.4, stop=6) ]
    )
    variables[f"{coll}_sv1mass"] = HistConf([Axis(name=f"{coll}_sv1mass", coll=coll, field="sv1mass",
                                                           label=r"FatJet $m_{SV1}$ [GeV]", bins=list(range(0, 410, 50)))]
    )
    # variables[f"{coll}_logsumcorrSVmass_tau21"] = HistConf(
    #     [ Axis(coll="FatJetGood", field="logsumcorrSVmass", label=r"log($\sum({m^{corr}_{SV}})$)", bins=42, start=-2.4, stop=6),
    #       Axis(coll="FatJetGood", field="tau21", label=r"$\tau_{21}$", type="variable", bins=[0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1]) ]
    # )
    variables[f"{coll}_globalParT3_XbbVsQCD"] = HistConf([Axis(name=f"{coll}_globalParT3_XbbVsQCD", coll=coll, field="globalParT3_XbbVsQCD",
                                                    label=r"FatJet globalParT3_XbbVsQCD score", bins=np.linspace(0,1,101).tolist())]
    )
    variables[f"{coll}_particleNet_XbbVsQCD"] = HistConf([Axis(name=f"{coll}_particleNet_XbbVsQCD", coll=coll, field="particleNet_XbbVsQCD",
                                                    label=r"FatJet particleNet_XbbVsQCD score", bins=np.linspace(0,1,101).tolist())]
    )
    variables[f"{coll}_particleNetLegacy_XbbVsQCD"] = HistConf([Axis(name=f"{coll}_particleNetLegacy_XbbVsQCD", coll=coll, field="particleNetLegacy_XbbVsQCD",
                                                    label=r"FatJet particleNet_XbbVsQCD score", bins=np.linspace(0,1,101).tolist())]
    )

if BDT_MODEL:
    # Event-level boosted-BDT score vs. leading / subleading Higgs (AK8) tau21.
    # Filled by mutagAnalysisOneMuonInAK8BDTProcessor; NaN for <2 FatJetGood.
    _bdt_score_bins = np.linspace(0, 1, 51).tolist()
    _tau21_bins = [0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                   0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1]
    for _score_field, _score_label, _tag in [
        ("boosted_bdt_score", r"$BDT_{ggF}$ score", "ggF"),
        ("boosted_bdt_vbf_score", r"$BDT_{VBF}$ score", "VBF"),
    ]:
        for _higgs, _tau21_field in [
            ("HiggsLeading", "HiggsLeading_tau21"),
            ("HiggsSubLeading", "HiggsSubLeading_tau21"),
        ]:
            variables[f"bdt_{_tag}_vs_{_higgs}_tau21"] = HistConf(
                [
                    Axis(name=f"bdt_{_tag}_score", coll="events", field=_score_field,
                         label=_score_label, bins=_bdt_score_bins),
                    Axis(name=f"{_higgs}_tau21", coll="events", field=_tau21_field,
                         label=fr"{_higgs} $\tau_{{21}}$", type="variable", bins=_tau21_bins),
                ],
                # The tau21/nmu MultiCuts cut on the FatJetGood collection, so every
                # category mask is 2D (per-jet). These BDT variables are per-event
                # (dim=1), so the 2D masks must be collapsed to the event axis: an
                # event enters a category if any relevant FatJetGood passes (OR).
                collapse_2D_masks=True,
                collapse_2D_masks_mode="OR",
            )

# Build dictionary of workflow options
# We reweigh histograms differently depending whether:
# - The histogram contains the whole jet collection ("all")
workflow_options = {
    "histograms_to_reweigh" : {
        "by_pos" : {
            "all" : [name for name in variables.keys() if name.startswith("FatJetGood_") and not name.endswith(("_1", "_2"))]
        }
    }
}

if BDT_MODEL:
    workflow_options["bdt_model"] = BDT_MODEL

taggers = parameters["mutag_calibration"]["taggers"]

# Note: Here we assume that the pt binning and WPs are the same for all the eras!
# To be changed in the future if the WP is a function of the data taking year
pt_binning = parameters["mutag_calibration"]["pt_binning"]["2024"]
msd_binning = parameters["mutag_calibration"]["msd_binning"]["2024"]
tau21_binning = parameters["mutag_calibration"]["tau21_binning"]["2024"]
wp_dict = parameters["mutag_calibration"]["wp"]["2024"]

common_cats = {
    "inclusive" : [passthrough],
    # "pt300msd40" : [get_ptmsd(300., 40.)],
    # "pt300msd60" : [get_ptmsd(300., 60.)],
    # "pt300msd80" : [get_ptmsd(300., 80.)],
    # "pt300msd100" : [get_ptmsd(300., 100.)],
    # "pt300msd80to170" : [get_ptmsd_window(300., 80., 170.)],
    # "pt250msd50mreg50to200bbtag05": [get_ptmsd(250., 50.), get_mregbin(50., 200.), get_tagger_pass(["btag"], 0.05)],
    "pt250msd50mreg50to200": [get_ptmsd(250., 50.), get_mregbin(50., 200.)],
}

# Define cuts to select bins in pt
cuts_pt = []
cuts_names_pt = []
for pt_low, pt_high in pt_binning:
    cuts_pt.append(get_ptbin(pt_low, pt_high))
    cuts_names_pt.append(f'Pt-{pt_low}to{pt_high}')

# Define cuts to select bins in msoftdrop
cuts_msd = []
cuts_names_msd = []
for msd_low, msd_high in msd_binning:
    cuts_msd.append(get_msdbin(msd_low, msd_high))
    cuts_names_msd.append(f'msd-{msd_low}to{msd_high}')

# Define cuts to select bins in msoftdrop
cuts_tau21 = []
cuts_names_tau21 = []
for tau21 in tau21_binning:
    cuts_tau21.append(get_tau21(tau21))
    cuts_names_tau21.append(f'tau21-{tau21}')

# Define cuts to select bins in msoftdrop
cuts_nmu = []
cuts_names_nmu = []
for nmu in [0, 1, 2]:
    if nmu>0:
        cuts_nmu.append(mutag_fatjet_sel_matched(nmu, unique=False))
        cuts_names_nmu.append(f"nmu-{nmu}-matched")
        cuts_nmu.append(mutag_fatjet_sel_matched(nmu, unique=True))
        cuts_names_nmu.append(f"nmu-{nmu}-matched-unique")
        # add matched to subjet
    cuts_nmu.append(mutag_fatjet_sel(nmu))
    cuts_names_nmu.append(f"nmu-{nmu}")

    # only apply nmu cut


# Define cuts to select bins in tagger WPs
cuts_tagger = []
cuts_names_tagger = []
for tagger in taggers:
    for wp, wp_value in wp_dict[tagger].items():
        for region in ["pass", "fail"]:
            if "-" in str(wp_value):
                wp_low, wp_high = wp_value.split("-")
                cuts_tagger.append(get_inclusive_wp(tagger, (float(wp_low), float(wp_high)), region))
            else:
                cuts_tagger.append(get_inclusive_wp(tagger, float(wp_value), region))
            cuts_names_tagger.append(f"{tagger}-{wp}-{region}")

# Define multicuts for pt, msd and tagger WPs
multicuts = [
    MultiCut(name="tau21",
             cuts=cuts_tau21,
             cuts_names=cuts_names_tau21),
    MultiCut(name="nmu",
             cuts=cuts_nmu,
             cuts_names=cuts_names_nmu),
    # MultiCut(name="msd",
    #          cuts=cuts_msd,
    #          cuts_names=cuts_names_msd),
    # MultiCut(name="pt",
    #          cuts=cuts_pt,
    #          cuts_names=cuts_names_pt),
    # MultiCut(name="tagger",
    #          cuts=cuts_tagger,
    #          cuts_names=cuts_names_tagger),
]

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [# "datasets/MC_QCD_MuEnriched_run3_redirector.json",
                  # "datasets/MC_QCD_Madgraph_run3_redirector.json",
                  # "datasets/MC_VJets_run3_redirector.json",
                  # "datasets/MC_TTto4Q_run3_redirector.json",
                  # "datasets/MC_singletop_run3_redirector.json",
                  # "datasets/DATA_BTagMu_run3_redirector.json"],
                  "datasets/skimmed_dataset_definition_madgraph.json",
                   "datasets/skimmed_dataset_definition_VJets2024.json",
                  "datasets/skimmed_dataset_definition_2024.json",
                   "datasets/signal_GluGluHHto4B_Par_boosted_skimmed_2024.json",
                  "datasets/skimmed_dataset_definition.json"],
        "filter" : {
            "samples": samples,
            "samples_exclude" : [],
            "year": [
                # '2022_preEE',
                # '2022_postEE',
                # '2023_preBPix',
                # '2023_postBPix',
                '2024'
            ]
        },
        "subsamples": subsamples
    },

    workflow = mutagAnalysisOneMuonInAK8BDTProcessor if BDT_MODEL else mutagAnalysisOneMuonInAK8Processor,
    workflow_options = workflow_options,

    skim = [get_nPVgood(1),
            eventFlags,
            goldenJson,
            get_nObj_min(1, 200., "FatJet"),
            get_nObj_minmsd(1, 30., "FatJet"),
            get_nObj_min(1, 3., "Muon"),
            get_HLTsel()],

    preselections = [get_nObj_min(1, parameters.object_preselection["FatJet"]["pt"], "FatJetGood")],
    categories = CartesianSelection(multicuts=multicuts, common_cats=common_cats),

    weights_classes = common_weights + [SF_trigger_prescale],
    weights = {
        "common": {
            "inclusive": ["genWeight","lumi","XS","sf_trigger_prescale",
                          "pileup"],
            "bycategory" : {
            }
        },
        "bysample": {
            "QCD_Madgraph": {
                "inclusive": ["sf_partonshower_isr", "sf_partonshower_fsr"],
                "bycategory" : {
                }
            },
            "VJets": {
                "inclusive": ["sf_partonshower_isr", "sf_partonshower_fsr"],
                "bycategory" : {
                }
            },
            "TTto4Q": {
                "inclusive": ["sf_partonshower_isr", "sf_partonshower_fsr"],
                "bycategory" : {
                }
            },
            "SingleTop": {
                "inclusive": ["sf_partonshower_isr", "sf_partonshower_fsr"],
                "bycategory" : {
                }
            }
        }
    },

    calibrators = [JetsCalibrator, JetsSoftdropMassCalibrator],
    variations = {
        "weights": {
            "common": {
                "inclusive": ["pileup"],
                "bycategory" : {
                }
            },
            "bysample": {
                "QCD_Madgraph": {
                    "inclusive": ["sf_partonshower_isr", "sf_partonshower_fsr"],
                    "bycategory": {
                    }
                },
                "VJets": {
                    "inclusive": ["sf_partonshower_isr", "sf_partonshower_fsr"],
                    "bycategory": {
                    }
                },
                "TTto4Q": {
                    "inclusive": ["sf_partonshower_isr", "sf_partonshower_fsr"],
                    "bycategory": {
                    }
                },
                "SingleTop": {
                    "inclusive": ["sf_partonshower_isr", "sf_partonshower_fsr"],
                    "bycategory": {
                    }
                }
            }   
        },
        "shape": {
            "common": {
                "inclusive" : ["jet_calibration"]
            }
        }
    },

    variables = variables,

    columns = {}
)

# Registering custom functions
import cloudpickle
cloudpickle.register_pickle_by_value(workflow)
if BDT_MODEL:
    cloudpickle.register_pickle_by_value(bdt_workflow)
cloudpickle.register_pickle_by_value(mutag_calib)
