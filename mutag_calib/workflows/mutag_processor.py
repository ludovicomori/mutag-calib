from collections import defaultdict
import awkward as ak
import correctionlib

from mutag_calib.workflows.fatjet_base import fatjetBaseProcessor
from pocket_coffea.utils.configurator import Configurator
from mutag_calib.lib.sv import *

class mutagAnalysisProcessor(fatjetBaseProcessor):
    def __init__(self, cfg: Configurator):
        super().__init__(cfg)
        if not "histograms_to_reweigh" in self.cfg.workflow_options.keys():
            raise Exception("The entry of the config file 'workflow_options' does not contain a key 'histograms_to_reweigh'. Please specify it in the config file.")
        self.histograms_to_reweigh = self.cfg.workflow_options["histograms_to_reweigh"]
        self.weight_3d = defaultdict(dict)
        self.custom_histogram_weights = {}

    def apply_object_preselection(self, variation):
        super().apply_object_preselection(variation)

        # Restrict analysis to leading and subleading jets only
        self.events["FatJetGood"] = self.events.FatJetGood[ak.local_index(self.events.FatJetGood, axis=1) < 2]

        # Label leading and subleading AK8 jets BEFORE muon tagging selection
        # Leading: pos=0, Subleading: pos=1
        self.events["FatJetGood"] = ak.with_field(self.events["FatJetGood"], ak.local_index(self.events["FatJetGood"], axis=1), "pos")

    def ptetatau21_reweighting(self, variation):
        '''Correction of jets observable by a 3D reweighting based on (pT, eta, tau21).
        The function stores the nominal, up and down weights in self.weight_3d,
        where the up/down variations are computed considering the statistical uncertainty on data and MC.'''
        cset = correctionlib.CorrectionSet.from_file(self.params["ptetatau21_reweighting"][self._year]["file"])
        assert len(list(cset.keys())) == 1, "The correction file should contain only one correction."
        key = list(cset.keys())[0]
        corr = cset[key]

        cat = self.params["ptetatau21_reweighting"][self._year]["category"]
        nfatjet  = ak.num(self.events.FatJetGood.pt)
        pos = ak.flatten(self.events.FatJetGood.pos)
        pt = ak.flatten(self.events.FatJetGood.pt)
        eta = ak.flatten(self.events.FatJetGood.eta)
        tau21 = ak.flatten(self.events.FatJetGood.tau21)

        weight_dict = {"all" : {}, "1" : {}, "2" : {}}
        for var in ["nominal", "statUp", "statDown"]:
            w = corr.evaluate(cat, variation, var, pos, pt, eta, tau21)
            weight_dict["all"][var] = ak.unflatten(w, nfatjet)
            # Here we build the flattened custom weights for the leading and subleading jet collections.
            # In order for the length of the weights array to match the number of the per-event mask,
            # we set the weight to be 1 for the events that does not contain a jet with pos=0(1)
            weight_dict["1"][var] = ak.fill_none(ak.firsts(weight_dict["all"]["nominal"][self.events.FatJetGood.pos == 0]), 1)
            weight_dict["2"][var] = ak.fill_none(ak.firsts(weight_dict["all"]["nominal"][self.events.FatJetGood.pos == 1]), 1)

        # Here we store the dictionary for the custom weights, with the following content:
        # - pos = "all": the jagged array of weights of the full jet collection
        # - pos = "1": the flat array of weights of the leading jet collection
        # - pos = "2": the flat array of weights of the subleading jet collection
        # For each of the 3 cases, the nominal, up and down variations are stored.
        for pos, weight in weight_dict.items():
            self.weight_3d[pos] = {
                "nominal" : weight["nominal"],
                "sf_ptetatau21_reweightingUp" : weight["statUp"],
                "sf_ptetatau21_reweightingDown" : weight["statDown"]
            }

    def process_extra_after_presel(self, variation):
        if self._sample in ["QCD_MuEnriched", "QCD_Madgraph"]:
            self.ptetatau21_reweighting(variation)
            for pos, hists in self.histograms_to_reweigh["by_pos"].items():
                for histname in hists:
                    self.custom_histogram_weights[histname] = self.weight_3d[pos]["nominal"]

    # We redefine the fill_histograms method in order to use the custom histogram weights
    def fill_histograms(self, variation):
        '''Function which fill the histograms for each category and variation,
        throught the HistManager.
        '''
        # Filling the autofill=True histogram automatically
        # Calling hist manager with the subsample masks
        self.hists_manager.fill_histograms(
            self.events,
            self._categories,
            subsamples=self._subsamples[self._sample],
            shape_variation=variation,
            custom_fields=self.custom_histogram_fields,
            custom_weight=self.custom_histogram_weights  # Here we pass the custom weights to the hist manager: this will apply our custom 3D weights during filling
        )
        # Saving the output for each sample/subsample
        for subs in self._subsamples[self._sample].keys():
            # When we loop on all the subsample we need to format correctly the output if
            # there are no subsamples
            if self._hasSubsamples:
                name = f"{self._sample}__{subs}"
            else:
                name = self._sample
            for var, H in self.hists_manager.get_histograms(subs).items():
                self.output["variables"][var][name][self._dataset] = H

    def fill_column_accumulators(self, variation):
        pass
