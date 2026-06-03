"""
Run 2 UL GloParT fit templates for HH->bbgg scale factors.

Uses globalParT3_XbbVsQCD tagger with WP 0.6 (same as 2024).
Runs on NanoAODv15 reprocessed Run 2 UL samples.

IMPORTANT: Before running, ensure:
  1. globalParT3_Xbb field exists in Run 2 NanoAODv15 files (Step 0 verification)
  2. Dataset JSONs have been built: pocket-coffea build-datasets --cfg datasets/datasets_definitions_*_run2.json -o ...
  3. pT reweighting maps exist (Step 4): FatJetGood_pt_eta_tau21_{year}_reweighting.json

TODO: GloParT WP 0.6 may need tuning for Run 2 (13 TeV vs 13.6 TeV).

Usage:
  pocket-coffea run --cfg mutag_calib/configs/fit_templates/fit_templates_HHbbgg_run2_glopart.py \\
    -o fit_templates_HHbbgg_run2_glopart -e dask@lxplus \\
    --custom-run-options mutag_calib/configs/params/run_options.yaml --process-separately
"""

from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_eq, get_nObj_min, get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.categorization import CartesianSelection, MultiCut

from pocket_coffea.lib.calibrators.common.common import JetsCalibrator
from mutag_calib.lib.calibrators import FixedJetsSoftdropMassCalibrator as JetsSoftdropMassCalibrator
from pocket_coffea.lib.weights.common.common import common_weights
from pocket_coffea.parameters.histograms import *
import mutag_calib
from mutag_calib.configs.fatjet_base.custom.cuts import get_ptmsd, get_ptmsd_window, get_nObj_minmsd, get_flavor, get_ptbin, get_msdbin
from mutag_calib.configs.fatjet_base.custom.functions import get_inclusive_wp
from mutag_calib.configs.fatjet_base.custom.weights import SF_trigger_prescale
import mutag_calib.workflows.mutag_oneMuAK8_processor as workflow
from mutag_calib.workflows.mutag_oneMuAK8_processor import mutagAnalysisOneMuonInAK8Processor
import os

localdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                f"{localdir}/params/object_preselection.yaml",
                                                f"{localdir}/params/jets_calibration.yaml",
                                                f"{localdir}/params/triggers_run2.yaml",
                                                f"{localdir}/params/triggers_prescales_run2.yaml",
                                                f"{localdir}/params/ptetatau21_reweighting_HHbbgg_run2.yaml",
                                                f"{localdir}/params/mutag_calibration_HHbbgg.yaml",
                                                f"{localdir}/params/plotting_style.yaml",
                                                update=True)

# Override Run 2 AK8 JECs with preliminary v15 PUPPI corrections
from omegaconf import OmegaConf
_jec_base = "/afs/cern.ch/user/i/izisopou/public/jme/json_files_ULRun2_PUPPI"
_ak8_jec_v15 = {
    "2018": {
        "json_path": f"{_jec_base}/Run2Summer20UL18/fatJet_jerc.json.gz",
        "jec_mc": "Summer20UL18_V1_MC",
        "jec_data": {"A": "Summer20UL18_RunA_V1_DATA", "B": "Summer20UL18_RunB_V1_DATA",
                      "C": "Summer20UL18_RunC_V1_DATA", "D": "Summer20UL18_RunD_V1_DATA"},
        "jer": "Summer19UL18_JRV2_MC", "level": "L1L2L3Res",
    },
    "2017": {
        "json_path": f"{_jec_base}/Run2Summer20UL17/fatJet_jerc.json.gz",
        "jec_mc": "Summer20UL17_V1_MC",
        "jec_data": {"B": "Summer20UL17_RunB_V1_DATA", "C": "Summer20UL17_RunC_V1_DATA",
                      "D": "Summer20UL17_RunD_V1_DATA", "E": "Summer20UL17_RunE_V1_DATA",
                      "F": "Summer20UL17_RunF_V1_DATA"},
        "jer": "Summer19UL17_JRV3_MC", "level": "L1L2L3Res",
    },
    "2016_PostVFP": {
        "json_path": f"{_jec_base}/Run2Summer20UL16/fatJet_jerc.json.gz",
        "jec_mc": "Summer20UL16_V1_MC",
        "jec_data": "Summer20UL16_RunFGH_V1_DATA",
        "jer": "Summer20UL16_JRV3_MC", "level": "L1L2L3Res",
    },
    "2016_PreVFP": {
        "json_path": f"{_jec_base}/Run2Summer20UL16APV/fatJet_jerc.json.gz",
        "jec_mc": "Summer20UL16APV_V1_MC",
        "jec_data": {"B": "Summer20UL16APV_RunBCD_V1_DATA", "C": "Summer20UL16APV_RunBCD_V1_DATA",
                      "D": "Summer20UL16APV_RunBCD_V1_DATA", "E": "Summer20UL16APV_RunEF_V1_DATA",
                      "F": "Summer20UL16APV_RunEF_V1_DATA"},
        "jer": "Summer20UL16APV_JRV3_MC", "level": "L1L2L3Res",
    },
}
for _year, _cfg in _ak8_jec_v15.items():
    OmegaConf.update(parameters, f"jets_calibration.jet_types.AK8PFPuppi.{_year}", OmegaConf.create(_cfg), merge=False)

# Also override AK4PFPuppi for softdrop mass subjet corrections (NanoAODv15 uses PUPPI subjets)
_ak4_jec_v15 = {
    "2018": {
        "json_path": f"{_jec_base}/Run2Summer20UL18/jet_jerc.json.gz",
        "jec_mc": "Summer20UL18_V1_MC",
        "jec_data": {"A": "Summer20UL18_RunA_V1_DATA", "B": "Summer20UL18_RunB_V1_DATA",
                      "C": "Summer20UL18_RunC_V1_DATA", "D": "Summer20UL18_RunD_V1_DATA"},
        "jer": "Summer19UL18_JRV2_MC", "level": "L1L2L3Res",
    },
    "2017": {
        "json_path": f"{_jec_base}/Run2Summer20UL17/jet_jerc.json.gz",
        "jec_mc": "Summer20UL17_V1_MC",
        "jec_data": {"B": "Summer20UL17_RunB_V1_DATA", "C": "Summer20UL17_RunC_V1_DATA",
                      "D": "Summer20UL17_RunD_V1_DATA", "E": "Summer20UL17_RunE_V1_DATA",
                      "F": "Summer20UL17_RunF_V1_DATA"},
        "jer": "Summer19UL17_JRV3_MC", "level": "L1L2L3Res",
    },
    "2016_PostVFP": {
        "json_path": f"{_jec_base}/Run2Summer20UL16/jet_jerc.json.gz",
        "jec_mc": "Summer20UL16_V1_MC",
        "jec_data": "Summer20UL16_RunFGH_V1_DATA",
        "jer": "Summer20UL16_JRV3_MC", "level": "L1L2L3Res",
    },
    "2016_PreVFP": {
        "json_path": f"{_jec_base}/Run2Summer20UL16APV/jet_jerc.json.gz",
        "jec_mc": "Summer20UL16APV_V1_MC",
        "jec_data": {"B": "Summer20UL16APV_RunBCD_V1_DATA", "C": "Summer20UL16APV_RunBCD_V1_DATA",
                      "D": "Summer20UL16APV_RunBCD_V1_DATA", "E": "Summer20UL16APV_RunEF_V1_DATA",
                      "F": "Summer20UL16APV_RunEF_V1_DATA"},
        "jer": "Summer20UL16APV_JRV3_MC", "level": "L1L2L3Res",
    },
}
for _year, _cfg in _ak4_jec_v15.items():
    OmegaConf.update(parameters, f"jets_calibration.jet_types.AK4PFPuppi.{_year}", OmegaConf.create(_cfg), merge=False)

# Add AK4PFPuppi variations for Run 2 (PocketCoffea defaults only have Run 3)
_run2_ak4puppi_variations = ["JES_Total", "JER"]
for _year in ["2016_PreVFP", "2016_PostVFP", "2017", "2018"]:
    OmegaConf.update(parameters, f"jets_calibration.variations.AK4PFPuppi.{_year}", _run2_ak4puppi_variations, merge=False)

samples = [
    "QCD_MuEnriched",
    "QCD_Madgraph",
    "VJets",
    "TTto4Q",
    "SingleTop",
    "DATA_BTagMu"
]
subsamples = {}
for s in filter(lambda x: 'DATA_BTagMu' not in x, samples):
    subsamples[s] = {f"{s}_{f}" : [get_flavor(f)] for f in ['l', 'c', 'b', 'cc', 'bb']}

variables = {}

collections = ["FatJetGood"]

for coll in collections:
    variables.update(**fatjet_hists(coll=coll))
    variables[f"{coll}_pt"] = HistConf([Axis(name=f"{coll}_pt", coll=coll, field="pt",
                                                    label=r"FatJet $p_{T}$ [GeV]", bins=list(range(300, 1010, 10)))]
    )
    variables[f"{coll}_msoftdrop"] = HistConf([Axis(name=f"{coll}_msoftdrop", coll=coll, field="msoftdrop",
                                                           label=r"FatJet $m_{SD}$ [GeV]", bins=list(range(0, 410, 10)))]
    )
    variables[f"{coll}_msoftdrop_raw"] = HistConf([Axis(name=f"{coll}_msoftdrop_raw", coll=coll, field="msoftdrop_raw",
                                                           label=r"FatJet $m_{SD}$ [GeV]", bins=list(range(0, 410, 10)))]
    )
    variables[f"{coll}_tau21"] = HistConf([Axis(name=f"{coll}_tau21", coll=coll, field="tau21",
                                                           label=r"FatJet $\tau_{21}$", bins=[0, 0.20, 0.25, 0.30, 0.35,
                                                           0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1])]
    )
    variables[f"{coll}_logsumcorrSVmass"] = HistConf(
        [ Axis(coll="FatJetGood", field="logsumcorrSVmass", label=r"log($\sum({m^{corr}_{SV}})$)", bins=42, start=-2.4, stop=6) ]
    )
    variables[f"{coll}_logsumcorrSVmass_tau21"] = HistConf(
        [ Axis(coll="FatJetGood", field="logsumcorrSVmass", label=r"log($\sum({m^{corr}_{SV}})$)", bins=42, start=-2.4, stop=6),
          Axis(coll="FatJetGood", field="tau21", label=r"$\tau_{21}$", type="variable", bins=[0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1]) ]
    )

# Build dictionary of workflow options
workflow_options = {
    "histograms_to_reweigh" : {
        "by_pos" : {
            "all" : [name for name in variables.keys() if name.startswith("FatJetGood_") and not name.endswith(("_1", "_2"))]
        }
    }
}

# Use only GloParT tagger for Run 2 (same as 2024)
# IMPORTANT: Do NOT add globalParT3_XbbVsQCD to global taggers list in YAML
taggers = ["globalParT3_XbbVsQCD"]

# Years to process — must share the same pt_binning.
# 2017/2018 use [300,400,450,inf], 2016 uses [300,350,425,inf].
# Run years with different binning as separate jobs.
year_filter = [
    '2018',
    #'2017',
    #'2016_PostVFP',
    #'2016_PreVFP'
]

pt_binning = parameters["mutag_calibration"]["pt_binning"][year_filter[0]]
wp_dict = parameters["mutag_calibration"]["wp"][year_filter[0]]
msd_binning = parameters["mutag_calibration"]["msd_binning"][year_filter[0]]

common_cats = {
    "inclusive" : [passthrough],
    "pt300msd30" : [get_ptmsd(300., 30.)],
    "pt300msd40" : [get_ptmsd(300., 40.)],
    "pt300msd60" : [get_ptmsd(300., 60.)],
    "pt300msd80" : [get_ptmsd(300., 80.)],
    "pt300msd100" : [get_ptmsd(300., 100.)],
    "pt300msd30to210" : [get_ptmsd_window(300., 30., 210.)],
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

# Define cuts to select bins in tagger WPs
cuts_tagger = []
cuts_names_tagger = []
for tagger in taggers:
    for wp, wp_value in wp_dict[tagger].items():
        for region in ["pass", "fail"]:
            cuts_tagger.append(get_inclusive_wp(tagger, wp_value, region))
            cuts_names_tagger.append(f"{tagger}-{wp}-{region}")

# Define multicuts for pt, msd and tagger WPs
multicuts = [
    MultiCut(name="msd",
             cuts=cuts_msd,
             cuts_names=cuts_names_msd),
    MultiCut(name="pt",
             cuts=cuts_pt,
             cuts_names=cuts_names_pt),
    MultiCut(name="tagger",
             cuts=cuts_tagger,
             cuts_names=cuts_names_tagger),
]

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": ["datasets/MC_QCD_MuEnriched_run2.json",
                  "datasets/MC_QCD_Madgraph_run2.json",
                  "datasets/MC_VJets_run2.json",
                  "datasets/MC_TTto4Q_run2.json",
                  "datasets/MC_singletop_run2.json",
                  "datasets/DATA_BTagMu_run2.json"],
        "filter" : {
            "samples": samples,
            "samples_exclude" : [],
            "year": year_filter
        },
        "subsamples": subsamples
    },

    workflow = mutagAnalysisOneMuonInAK8Processor,
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
cloudpickle.register_pickle_by_value(mutag_calib)
