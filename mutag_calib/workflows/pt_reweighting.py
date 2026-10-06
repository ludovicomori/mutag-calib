import awkward as ak

from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.categorization import StandardSelection
from mutag_calib.lib.sv import *
from mutag_calib.configs.fatjet_base.custom.cuts import get_ptmsd, mutag_fatjet_sel, mutag_subjet_sel
from mutag_calib.workflows.fatjet_base import fatjetBaseProcessor


class ptReweightProcessor(fatjetBaseProcessor):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)
        self.pt_eta_2d_maps = [
            'FatJetGood_pt_eta',
            #'FatJetGoodNMuon1_pt_eta',
            #'FatJetGoodNMuon2_pt_eta',
            #'FatJetGoodNMuonSJ1_pt_eta',
            #'FatJetGoodNMuonSJUnique1_pt_eta',
        ]
        self.pt_eta_tau21_3d_maps = [
            'FatJetGood_pt_eta_tau21', 'FatJetGood_pt_eta_tau21_bintau05',
            #'FatJetGoodNMuon1_pt_eta_tau21', 'FatJetGoodNMuon1_pt_eta_tau21_bintau05',
            #'FatJetGoodNMuon2_pt_eta_tau21', 'FatJetGoodNMuon2_pt_eta_tau21_bintau05',
            #'FatJetGoodNMuonSJ1_pt_eta_tau21', 'FatJetGoodNMuonSJ1_pt_eta_tau21_bintau05',
            #'FatJetGoodNMuonSJUnique1_pt_eta_tau21', 'FatJetGoodNMuonSJUnique1_pt_eta_tau21_bintau05',
        ]
        for histname in self.pt_eta_2d_maps + self.pt_eta_tau21_3d_maps:
            if not histname in self.cfg.variables.keys():
                raise Exception(f"'{histname}' is not present in the histogram keys.")

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation)

        # Restrict analysis to leading and subleading jets only
        self.events["FatJetGood"] = self.events.FatJetGood[ak.local_index(self.events.FatJetGood, axis=1) < 2]

        # Label leading and subleading AK8 jets BEFORE muon tagging selection
        # Leading: pos=0, Subleading: pos=1
        self.events["FatJetGood"] = ak.with_field(self.events["FatJetGood"], ak.local_index(self.events["FatJetGood"], axis=1), "pos")

        # Build 4 distinct AK8 jet collections with 4 different muon tagging scenarios
        cuts_mutag = {
            "FatJetGoodNMuon1" : [mutag_fatjet_sel(nmu=self.params.object_preselection["FatJet"]["nmu"])],
            #"FatJetGoodNMuon2" : [mutag_fatjet_sel(nmu=2)],
            #"FatJetGoodNMuonSJ1" : [mutag_subjet_sel(unique_matching=False)],
            #"FatJetGoodNMuonSJUnique1" : [mutag_subjet_sel(unique_matching=True)],
        }
        selection_mutag = StandardSelection(cuts_mutag)
        selection_mutag.prepare(
            events=self.events,
            processor_params=self.params
        )
        self.events["FatJetGood"] = self.events.FatJetGood[selection_mutag.get_mask("FatJetGoodNMuon1")]
        #self._ak8jet_collections = list(cuts_mutag.keys())
        #for coll in self._ak8jet_collections:
        #    mask_mutag = selection_mutag.get_mask(coll)

        #    # Apply muon tagging to AK8 jet collection
        #    self.events[coll] = self.events.FatJetGood[mask_mutag]

        # Define a new field called btag, that depends on what we will cut on later on.
        if "2024" in self._year:
            tagger = "globalParT3_XbbVsQCD"
        else:
            tagger = "particleNet_XbbVsQCD"
        self.events["FatJetGood"] = ak.with_field(self.events["FatJetGood"], self.events["FatJetGood"][tagger], "btag")
        self.events["FatJetGoodJet0"] = ak.pad_none(self.events.FatJetGood, 1, clip=True)[:, 0]
        self.events["FatJetGoodJet1"] = ak.pad_none(self.events.FatJetGood, 2, clip=True)[:, 1]


class ptReweightProcessorSkimonly(fatjetBaseProcessor):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation)
