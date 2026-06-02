"""
Run 2 UL pT reweighting configuration.

Produces 3D reweighting maps (pT, eta, tau21) for Run 2 UL NanoAODv15 samples.
Does NOT include QCD_Madgraph (same as Run 3 ptreweighting).

Prescale values in triggers_prescales_run2.yaml computed via brilcalc.

Usage:
  pocket-coffea run --cfg mutag_calib/configs/pt_reweighting/ptreweighting_run2.py \\
    -o pt_reweighting_run2 -e dask@lxplus \\
    --custom-run-options mutag_calib/configs/params/run_options.yaml --process-separately
  python merge_no_postprocess.py pt_reweighting_run2/output_all.coffea pt_reweighting_run2/*.coffea
  python mutag_calib/scripts/compute_3d_reweighting.py -i pt_reweighting_run2/output_all.coffea \\
    -o pt_reweighting_run2/3d_reweighting --overwrite --test
  cp pt_reweighting_run2/3d_reweighting/FatJetGood_pt_eta_* mutag_calib/configs/params/ptetatau21_reweighting/
"""

from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_eq, get_nObj_min, get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough

from pocket_coffea.lib.calibrators.common.common import JetsCalibrator
from mutag_calib.lib.calibrators import FixedJetsSoftdropMassCalibrator as JetsSoftdropMassCalibrator
from pocket_coffea.lib.weights.common.common import common_weights
from pocket_coffea.parameters.histograms import *
import mutag_calib
from mutag_calib.configs.fatjet_base.custom.cuts import get_ptmsd, get_ptmsd_window, get_nObj_minmsd, get_flavor
from mutag_calib.configs.fatjet_base.custom.functions import get_inclusive_wp
from mutag_calib.configs.fatjet_base.custom.weights import SF_trigger_prescale
import mutag_calib.workflows.pt_reweighting as workflow
from mutag_calib.workflows.pt_reweighting import ptReweightProcessor
import numpy as np
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
    variables[f"{coll}_pt_eta"] = HistConf(
        [ Axis(name=f"{coll}_pos", coll=coll, field="pos", type="int", label=r"FatJet position", bins=2, start=0, stop=2),
          Axis(name=f"{coll}_pt", coll=coll, field="pt", type="variable", label=r"FatJet $p_{T}$ [GeV]",
               bins=[300., 320., 340., 360., 380., 400., 450., 500., 550., 600., 700., 800., 900., 2500.]),
          Axis(name=f"{coll}_eta", coll=coll, field="eta", type="variable", label=r"FatJet $\eta$",
               bins=[-5, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 5]) ]
    )
    variables[f"{coll}_pt_eta_tau21"] = HistConf(
        [ Axis(name=f"{coll}_pos", coll=coll, field="pos", type="int", label=r"FatJet position", bins=2, start=0, stop=2),
          Axis(name=f"{coll}_pt", coll=coll, field="pt", type="variable", label=r"FatJet $p_{T}$ [GeV]",
               bins=[300., 320., 340., 360., 380., 400., 450., 500., 550., 600., 700., 800., 900., 2500.]),
          Axis(name=f"{coll}_eta", coll=coll, field="eta", type="variable", label=r"FatJet $\eta$",
               bins=[-5, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 5]),
          Axis(name=f"{coll}_tau21", coll=coll, field="tau21", type="variable", label=r"FatJet $\tau_{21}$",
               bins=[0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]) ]
    )
    variables[f"{coll}_pt_eta_tau21_bintau05"] = HistConf(
        [ Axis(name=f"{coll}_pos", coll=coll, field="pos", type="int", label=r"FatJet position", bins=2, start=0, stop=2),
          Axis(name=f"{coll}_pt", coll=coll, field="pt", type="variable", label=r"FatJet $p_{T}$ [GeV]",
               bins=[300., 320., 340., 360., 380., 400., 450., 500., 550., 600., 700., 800., 900., 2500.]),
          Axis(name=f"{coll}_eta", coll=coll, field="eta", type="variable", label=r"FatJet $\eta$",
               bins=[-5, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 5]),
          Axis(name=f"{coll}_tau21", coll=coll, field="tau21", type="variable", label=r"FatJet $\tau_{21}$",
               bins=[0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1]) ]
    )

cfg = Configurator(
    parameters = parameters,
    datasets = {
         "jsons": ["datasets/MC_VJets_run2.json",
                   "datasets/MC_TTto4Q_run2.json",
                   "datasets/MC_singletop_run2.json",
                   "datasets/DATA_BTagMu_run2.json",
                   "datasets/MC_QCD_MuEnriched_run2.json"
                   ],
        "filter" : {
            "samples": samples,
            "samples_exclude" : [],
            "year": [
                '2018',
                '2017',
                '2016_PostVFP',
                '2016_PreVFP'
            ]
        },
        "subsamples": subsamples
    },

    workflow = ptReweightProcessor,
    workflow_options = {},

    skim = [get_nPVgood(1),
            eventFlags,
            goldenJson,
            get_nObj_min(1, 200., "FatJet"),
            get_nObj_minmsd(1, 30., "FatJet"),
            get_nObj_min(1, 3., "Muon"),
            get_HLTsel()],

    preselections = [get_nObj_min(1, parameters.object_preselection["FatJet"]["pt"], "FatJetGood")],
    categories = {
        "pt300msd30" : [get_ptmsd(300., 30.)],
        "pt300msd80" : [get_ptmsd(300., 80.)],
        "pt300msd30to210" : [get_ptmsd_window(300., 30., 210.)],
        "pt300msd80to170" : [get_ptmsd_window(300., 80., 170.)],
    },

    weights_classes = common_weights + [SF_trigger_prescale],
    weights = {
        "common": {
            "inclusive": ["genWeight","lumi","XS","sf_trigger_prescale",
                          "pileup"],
            "bycategory" : {
            }
        },
        "bysample": {
        }
    },

    # Rho field alias injected in fatjet_base.py process_extra_after_skim for Run 2 NanoAODv15
    calibrators = [JetsCalibrator, JetsSoftdropMassCalibrator],
    variations = {
        "weights": {
            "common": {
                "inclusive": [],
                "bycategory" : {
                }
            },
            "bysample": {
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
