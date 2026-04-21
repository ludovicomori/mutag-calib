from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_eq, get_nObj_min, get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.categorization import CartesianSelection, MultiCut

from pocket_coffea.lib.calibrators.common.common import JetsCalibrator, JetsSoftdropMassCalibrator
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
                                                f"{localdir}/params/triggers_run3.yaml",
                                                f"{localdir}/params/triggers_prescales_run3.yaml",
                                                f"{localdir}/params/ptetatau21_reweighting_HHbbgg.yaml",
                                                f"{localdir}/params/mutag_calibration_HHbbgg.yaml",
                                                f"{localdir}/params/plotting_style.yaml",
                                                update=True)

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

# Use only GloParT tagger for 2024
taggers = ["globalParT3_XbbVsQCD"]

# 2024-specific binning and WPs
pt_binning = parameters["mutag_calibration"]["pt_binning"]["2024"]
wp_dict = parameters["mutag_calibration"]["wp"]["2024"]
msd_binning = parameters["mutag_calibration"]["msd_binning"]["2024"]

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
        "jsons": ["datasets/MC_QCD_MuEnriched_run3_redirector.json",
                  "datasets/MC_QCD_Madgraph_run3_redirector.json",
                  "datasets/MC_VJets_run3_redirector.json",
                  "datasets/MC_TTto4Q_run3_redirector.json",
                  "datasets/MC_singletop_run3_redirector.json",
                  "datasets/DATA_BTagMu_run3_redirector.json"],
        "filter" : {
            "samples": samples,
            "samples_exclude" : [],
            "year": ['2024']
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
