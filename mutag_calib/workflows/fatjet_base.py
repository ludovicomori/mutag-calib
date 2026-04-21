import awkward as ak
from collections import defaultdict

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.jets import jet_selection

from mutag_calib.lib.leptons import lepton_selection_noniso
from mutag_calib.lib.sv import get_corrmass, sv_matched_to_fatjet, get_sumcorrmass, get_sv1mass
from mutag_calib.lib.muon_matching import muons_matched_to_fatjet, muon_matched_to_subjet

class fatjetBaseProcessor(BaseProcessorABC):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)
        # Define dictionary to save fatjet JER seeds
        self.output_format.update({"seed_fatjet_chunk": defaultdict(str)})

        # Additional axis for the year
        #self.custom_axes.append(
        #    Axis(
        #        coll="metadata",
        #        field="year",
        #        name="year",
        #        bins=set(sorted(self.cfg.years)),
        #        type="strcat",
        #        growth=False,
        #        label="Year",
        #    )
        #)

    def process_extra_after_skim(self):
        super().process_extra_after_skim()
        # Save raw softdrop mass before any calibration as a new field of FatJet
        self.events["FatJet"] = ak.with_field(
            self.events.FatJet, self.events.FatJet.msoftdrop, "msoftdrop_raw"
        )
        # If the FatJet collection does not have nBHadrons and nCHadrons fields
        # Consider only fatjets with a matched GenJet, in order to retrieve nBHadrons and nCHadrons
        if self._isMC:
            # If the nBHadrons and nCHadrons branches are already present, no need to check genJetAK8Idx
            if ('nBHadrons' not in self.events.FatJet.fields) or ('nCHadrons' not in self.events.FatJet.fields):
                mask_matched_genjet = self.events.FatJet.genJetAK8Idx >= 0
                self.events["FatJet"] = self.events.FatJet[mask_matched_genjet]

            # Since NanoAODv15 the nBHadrons and nCHadrons variables are stored in the GenJetAK8 collection
            # so we need to match the FatJet to the GenJetAK8 and copy them if not already present
            if (not 'nBHadrons' in self.events.FatJet.fields) or (not 'nCHadrons' in self.events.FatJet.fields):
                matched_genjets = self.events.GenJetAK8[self.events.FatJet.genJetAK8Idx]
            if not 'nBHadrons' in self.events.FatJet.fields:
                self.events["FatJet"] = ak.with_field(
                    self.events.FatJet,
                    matched_genjets.nBHadrons,
                    "nBHadrons"
                )
            if not 'nCHadrons' in self.events.FatJet.fields:
                self.events["FatJet"] = ak.with_field(
                    self.events.FatJet,
                    matched_genjets.nCHadrons,
                    "nCHadrons"
                )

    def apply_object_preselection(self, variation):
        '''
        The ttHbb processor cleans
          - Electrons
          - Muons
          - Jets -> JetGood
          - BJet -> BJetGood

        '''
        # Include the supercluster pseudorapidity variable
        electron_etaSC = self.events.Electron.eta + self.events.Electron.deltaEtaSC
        self.events["Electron"] = ak.with_field(
            self.events.Electron, electron_etaSC, "etaSC"
        )
        # Build masks for selection of muons, electrons, jets, fatjets

        ################################################
        # Dedicated Muon selection for mutag final state
        self.events["MuonGood"] = lepton_selection_noniso(
            self.events, "Muon", self.params
        )

        self.events["FatJetGood"], self.fatjetGoodMask = jet_selection(
            self.events, "FatJet", self.params, self._year
        )

        # Select here events with at least one FatJetGood
        # WARNING: Here we are applying a per-event selection asking for at least one AK8 jet in the event
        # In this way, we can compute the number of MuonGood matched to the FatJetGood and its subjets
        self.events = self.events[ak.num(self.events.FatJetGood) >= 1]

        # Uniquely match muons to leading and subleading subjets
        # The shape of these masks is the same as the self.events.FatJetGood collection:
        # the first object are the muons matched to the leading subjet,
        # the second object are the muons that are matched to the subleading subjet, while
        # the third object are the dimuon pairs constructed from the matched muons

        self.events["MuonGoodMatchedToFatJetGood"] = muons_matched_to_fatjet(self.events)
        #muon_matched_to_leading_subjet = muon_matched_to_subjet(self.events, pos=0, unique=False)
        #muon_matched_to_subleading_subjet = muon_matched_to_subjet(self.events, pos=1, unique=False)
        #self.events["MuonGoodMatchedToSubJet"] = ak.concatenate((muon_matched_to_leading_subjet, muon_matched_to_subleading_subjet), axis=2)
        #muon_matched_uniquely_to_leading_subjet = muon_matched_to_subjet(self.events, pos=0, unique=True)
        #muon_matched_uniquely_to_subleading_subjet = muon_matched_to_subjet(self.events, pos=1, unique=True)
        #self.events["MuonGoodMatchedUniquelyToSubJet"] = ak.concatenate((muon_matched_uniquely_to_leading_subjet, muon_matched_uniquely_to_subleading_subjet), axis=2)

        fatjet_fields = {
            "tau21" : self.events.FatJetGood.tau2 / self.events.FatJetGood.tau1,
            #"nSubJet" : ak.count(events.FatJetGood.subjets.pt, axis=2),
            "nMuonGoodMatchedToFatJetGood" : ak.count(self.events["MuonGoodMatchedToFatJetGood"].pt, axis=2),
            #"nMuonGoodMatchedToSubJet" : ak.count(self.events["MuonGoodMatchedToSubJet"].pt, axis=2),
            #"nMuonGoodMatchedUniquelyToSubJet" : ak.count(self.events["MuonGoodMatchedUniquelyToSubJet"].pt, axis=2)
        }
        # Compute GloParT XbbVsQCD discriminator (only available in NanoAODv15, i.e. 2024)
        if "globalParT3_Xbb" in self.events.FatJetGood.fields:
            Xbb = self.events.FatJetGood.globalParT3_Xbb
            QCD = self.events.FatJetGood.globalParT3_QCD
            fatjet_fields["globalParT3_XbbVsQCD"] = ak.where(
                (Xbb + QCD) > 0,
                Xbb / (Xbb + QCD),
                -999.0,
            )
        for field, value in fatjet_fields.items():
            self.events["FatJetGood"] = ak.with_field(self.events.FatJetGood, value, field)

    def count_objects(self, variation):
        self.events["nMuonGood"] = ak.num(self.events.MuonGood)
        self.events["nFatJetGood"] = ak.num(self.events.FatJetGood)
        self.events["nSV"] = ak.num(self.events.SV)

    def define_common_variables_after_presel(self, variation):

        # Correct SV mass for mis-aligment between the SV momentum and the PV-SV direction.
        # This takes into account particles that are not reconstructed in the SV reconstruction.
        # The corrected mass is assigned to the SV.mass branch
        self.events.SV = ak.with_field(self.events.SV, get_corrmass(self.events.SV), "mass")
        # Match SV to AK8 jets
        self.events["SVMatchedToFatJetGood"] = sv_matched_to_fatjet(self.events)

        # Compute final particleNetMD scores
        # Xbb = self.events.FatJetGood.particleNetMD_Xbb
        # Xcc = self.events.FatJetGood.particleNetMD_Xcc
        # QCD = self.events.FatJetGood.particleNetMD_QCD
        sumcorrSVmass, logsumcorrSVmass = get_sumcorrmass(self.events.SVMatchedToFatJetGood)
        # Order SV by dxySig and compute the leading SV mass and its log
        #index_max_pt = ak.argsort(self.events.SVMatchedToFatJetGood.pt, ascending=False)
        index_max_dxySig = ak.argsort(self.events.SVMatchedToFatJetGood.dxySig, ascending=False)
        sv1mass, logsv1mass = get_sv1mass(self.events.SVMatchedToFatJetGood[index_max_dxySig])
        fatjet_fields = {
            "nSVMatchedToFatJetGood": ak.count(self.events["SVMatchedToFatJetGood"].pt, axis=2),
            # "particleNetMD_Xbb_QCD" : Xbb / (Xbb + QCD),
            # "particleNetMD_Xcc_QCD" : Xcc / (Xcc + QCD),
            "sumcorrSVmass" : sumcorrSVmass,
            "logsumcorrSVmass" : logsumcorrSVmass,
            "sv1mass" : sv1mass,
            "logsv1mass" : logsv1mass,
        }

        for field, value in fatjet_fields.items():
            self.events["FatJetGood"] = ak.with_field(self.events.FatJetGood, value, field)

    def fill_column_accumulators(self, variation):
        pass
