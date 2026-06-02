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
                                                f"{localdir}/params/triggers_run3.yaml",
                                                f"{localdir}/params/triggers_prescales_run3.yaml",
                                                f"{localdir}/params/ptetatau21_reweighting_HHbbgg.yaml",
                                                f"{localdir}/params/mutag_calibration_HHbbgg.yaml",
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

# Override 2024 reweighting map file path: workflow looks up by MC year (= '2024'),
# but for 2025 SF derivation we want the map derived from 2025 data + 2024 MC.
OmegaConf.update(parameters, "ptetatau21_reweighting.2024.file",
    f"{localdir}/params/ptetatau21_reweighting/HHbbgg/FatJetGood_pt_eta_tau21_2025_reweighting.json",
    merge=False)

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

# Use GloParT tagger for 2025
taggers = ["globalParT3_XbbVsQCD"]

# 2025 binning and WPs (same as 2024)
pt_binning = parameters["mutag_calibration"]["pt_binning"]["2025"]
wp_dict = parameters["mutag_calibration"]["wp"]["2025"]
msd_binning = parameters["mutag_calibration"]["msd_binning"]["2025"]

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
                  "datasets/DATA_BTagMu_2025.json"],
        "filter" : {
            "samples": samples,
            "samples_exclude" : [],
            "year": ['2024', '2025']
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
