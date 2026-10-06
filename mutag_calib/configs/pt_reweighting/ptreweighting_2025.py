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
                                                f"{localdir}/params/triggers_run3.yaml",
                                                f"{localdir}/params/triggers_prescales_run3.yaml",
                                                f"{localdir}/params/plotting_style.yaml",
                                                update=True)

# Override PocketCoffea defaults for 2025 (container doesn't have 2025 entries)
from omegaconf import OmegaConf

# NanoAOD version
OmegaConf.update(parameters, "default_nano_version.2025", 15, merge=False)

# Luminosity per era (from PdmV TWiki, Golden JSON column, in pb^-1)
OmegaConf.update(parameters, "lumi.picobarns.2025", OmegaConf.create({
    "C": 21590, "D": 25350, "E": 14050, "F": 26690, "G": 22280, "tot": 110190
}), merge=False)

# Golden JSON
OmegaConf.update(parameters, "lumi.goldenJSON.2025",
    "/cvmfs/cms-griddata.cern.ch/cat/metadata/DC/Collisions25/latest/Cert_Collisions2025_391658_398903_Golden.json",
    merge=False)

# Pileup weights
OmegaConf.update(parameters, "pileupJSONfiles.2025", OmegaConf.create({
    "file": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2025/puWeights2025.json.gz",
    "name": "Collisions25_Prompt_goldenJSON"
}), merge=False)

# Override 2024 MC pileup to use 2025 data conditions
# (MC has year="2024" but must be reweighted to 2025 pileup profile)
OmegaConf.update(parameters, "pileupJSONfiles.2024", OmegaConf.create({
    "file": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2025/puWeights2025.json.gz",
    "name": "Collisions25_Prompt_goldenJSON"
}), merge=False)

# Event flags (same as 2024)
OmegaConf.update(parameters, "event_flags.2025", OmegaConf.create([
    "goodVertices", "globalSuperTightHalo2016Filter", "EcalDeadCellTriggerPrimitiveFilter",
    "BadPFMuonFilter", "BadPFMuonDzFilter", "hfNoisyHitsFilter", "eeBadScFilter", "ecalBadCalibFilter"
]), merge=False)
OmegaConf.update(parameters, "event_flags_data.2025", OmegaConf.create(["eeBadScFilter"]), merge=False)

# AK4PFPuppi JEC for 2025 (needed by softdrop mass calibrator for subjet corrections)
OmegaConf.update(parameters, "jets_calibration.jet_types.AK4PFPuppi.2025", OmegaConf.create({
    "json_path": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-25Prompt-Winter25-NanoAODv15/latest/jet_jerc.json.gz",
    "jec_mc": "Winter25Prompt25_V3_MC",
    "jec_data": "Winter25Prompt25_V3_DATA",
    "jer": "Summer23BPixPrompt23_RunD_JRV1_MC",
    "level": "L1L2L3Res",
}), merge=False)

# AK4PFPuppi variations for 2025
OmegaConf.update(parameters, "jets_calibration.variations.AK4PFPuppi.2025",
    ["JES_Total", "JER"], merge=False)

# Jet ID JSON for 2025 — JME conveners confirmed (via Emmanouil) that the 2024
# jet ID JSON is valid for 2025. No dedicated 2025 JSON exists on CVMFS yet.
OmegaConf.update(parameters, "jet_scale_factors.jet_id.2025",
    "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest/jetid.json.gz",
    merge=False)

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
         "jsons": ["datasets/MC_VJets_run3_redirector.json",
                   "datasets/MC_TTto4Q_run3_redirector.json",
                   "datasets/MC_singletop_run3_redirector.json",
                   "datasets/DATA_BTagMu_2025.json",
                   "datasets/MC_QCD_MuEnriched_run3_redirector.json"
                   ],
        "filter" : {
            "samples": samples,
            "samples_exclude" : [],
            "year": ['2024', '2025']
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
