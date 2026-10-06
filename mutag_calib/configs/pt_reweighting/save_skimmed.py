from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_eq, get_nObj_min, get_HLTsel, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough

from pocket_coffea.lib.calibrators.common.common import JetsCalibrator, JetsSoftdropMassCalibrator
from pocket_coffea.lib.weights.common.common import common_weights
from pocket_coffea.parameters.histograms import *
import mutag_calib
from mutag_calib.configs.fatjet_base.custom.cuts import get_ptmsd, get_ptmsd_window, get_two_jet_ptmsd, get_mregbin, get_nObj_minmsd, get_flavor
from mutag_calib.configs.fatjet_base.custom.functions import get_inclusive_wp, get_tagger_pass
from mutag_calib.configs.fatjet_base.custom.weights import SF_trigger_prescale
import mutag_calib.workflows.pt_reweighting as workflow
from mutag_calib.workflows.pt_reweighting import ptReweightProcessorSkimonly
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


cfg = Configurator(
    save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/mutag_samples/",
    parameters = parameters,
    datasets = {
         "jsons": ["datasets/MC_VJets_run3.json",
                   # "datasets/MC_TTto4Q_run3.json",
                   # "datasets/MC_singletop_run3.json",
                   # "datasets/DATA_BTagMu_run3.json",
                   # "datasets/MC_QCD_MuEnriched_run3.json"
                   # "datasets/MC_VJets_run3_2024.json",
                   # "datasets/MC_TTto4Q_run3_2024.json",
                   # "datasets/MC_singletop_run3_2024.json",
                   # "datasets/DATA_BTagMu_run3_2024.json",
                   # "datasets/MC_QCD_Madgraph_run3.json"
                   # "datasets/MC_QCD_MuEnriched_run3_2024.json"
                   ],
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

    workflow = ptReweightProcessorSkimonly,
    workflow_options = {"skim_only": True},

    skim = [get_nPVgood(1),
            eventFlags,
            goldenJson,
            get_nObj_min(1, 200., "FatJet"),
            get_nObj_minmsd(1, 30., "FatJet"),
            get_nObj_min(1, 3., "Muon"),
            get_HLTsel()
            ],

    preselections=[
        #
    ],
    categories={
        #
    },
    weights={
        "common": {
            "inclusive": [
                "genWeight",
                "lumi",
                "XS",
            ],
            "bycategory": {},
        },
        "bysample": {},
    },
    variations={
        "weights": {
            "common": {
                "inclusive": [],
                "bycategory": {},
            },
            "bysample": {},
        }
    },
    variables={
        #
    },
    columns={
        #
    },
)

# Registering custom functions
import cloudpickle
cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(mutag_calib)
