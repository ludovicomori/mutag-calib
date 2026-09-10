"""Optional boosted HH->4b BDT evaluation on top of the mu-tagged AK8 workflow.

``mutagAnalysisOneMuonInAK8BDTProcessor`` extends
``mutagAnalysisOneMuonInAK8Processor`` with an evaluation of the boosted ggF/VBF
BDT (the ``25Feb5_v13_glopartv2_rawmass`` training used by
``AnalysisConfigs_bossted_dev/configs/VBF_HH4b_boosted``).

It is completely opt-in: the extra reconstruction and the model evaluation only
run when ``workflow_options["bdt_model"]`` is set to a model path. With no
``bdt_model`` this class behaves exactly like its parent, so a config can select
it unconditionally and toggle the BDT on/off through ``workflow_options`` alone.

The BDT inputs are rebuilt exactly as in the boosted VBF workflow:
  * leading / subleading ``FatJetGood`` -> ``HiggsLeading`` / ``HiggsSubLeading``
  * 2024 regressed mass ``mass_raw * globalParT3_massCorrGeneric``
  * ``Tau3OverTau2``, ``btagBBTXbb`` (+ ``disc_TXbb`` digitisation), ``divHHmass``
  * ``HH`` four-vector and the leading/subleading pT ratio
  * AK4 ``JetGoodVBF`` -> ``get_lead_mjj_jet_pair`` -> ``mjj/detaJetGoodVBFEnergyOrdered``
  * AK4 ``JetGoodCloseToFatJet`` -> ``dRclosestVBF`` / ``massclosestVBF``

The per-event outputs ``boosted_bdt_score`` (ggF) and ``boosted_bdt_vbf_score``
(VBF) are stored on ``self.events``, set to ``NaN`` for events with fewer than
two ``FatJetGood``. ``HiggsLeading_tau21`` / ``HiggsSubLeading_tau21`` are stored
alongside so the config can histogram score vs. tau21 in 2D.
"""

import warnings

import awkward as ak
import numpy as np

from mutag_calib.workflows.mutag_oneMuAK8_processor import mutagAnalysisOneMuonInAK8Processor
from mutag_calib.lib.bdt import get_default_bdt_inputs, evaluate_bdt, disc_TXbb
from mutag_calib.lib.vbf_reco import add_fields, custom_jet_selection, get_lead_mjj_jet_pair


class mutagAnalysisOneMuonInAK8BDTProcessor(mutagAnalysisOneMuonInAK8Processor):

    def __init__(self, cfg):
        super().__init__(cfg)
        # Opt-in switch: no bdt_model -> this processor is a no-op wrapper.
        self.bdt_model = self.cfg.workflow_options.get("bdt_model", None)

    # ------------------------------------------------------------------
    # helpers (ported 1:1 from configs/VBF_HH4b_boosted/workflow.py)
    # ------------------------------------------------------------------
    def _regressed_mass(self, fatjet):
        """2024 regressed AK8 mass, as in the boosted workflow apply_object_preselection."""
        if ("mass_raw" in fatjet.fields) and ("globalParT3_massCorrGeneric" in fatjet.fields):
            return fatjet.mass_raw * fatjet.globalParT3_massCorrGeneric
        warnings.warn(
            "mutag BDT: 'mass_raw'/'globalParT3_massCorrGeneric' not found on FatJetGood; "
            "falling back to FatJetGood.mass for the BDT H1Mass input."
        )
        return fatjet.mass

    def _txbb(self, fatjet):
        """globalParT3 Xbb-vs-QCD, as used for btagBBTXbb in the boosted workflow."""
        xbb = fatjet.globalParT3_Xbb
        qcd = fatjet.globalParT3_QCD
        return xbb / (xbb + qcd)

    def _build_vbf_ak4_collections(self):
        """Reproduce the AK4 VBF-jet collections of the boosted workflow.

        Needs ``self.events["FatJetGoodSelected"]`` (the <=2 tagged AK8 jets) for
        the dR cleaning done inside ``custom_jet_selection``.
        """
        # VBF AK4 jets (boosted branch: from the full Jet collection, no forward-jet veto)
        self.events["JetGoodVBF"], _ = custom_jet_selection(
            self.events, "Jet", "JetVBF", self.params, year=self._year,
            pt_type="pt", pt_cut_name="pt", forward_jet_veto=False,
        )
        self.events["JetGoodVBF"] = self.events.JetGoodVBF[
            ak.argsort(self.events.JetGoodVBF.pt, axis=1, ascending=False)
        ]
        self.events["JetGoodVBFEnergyOrdered"] = get_lead_mjj_jet_pair(self.events, "JetGoodVBF")

        for jet_coll in ["JetGoodVBF", "JetGoodVBFEnergyOrdered"]:
            padded = ak.pad_none(self.events[jet_coll], 2, axis=1)
            self.events[f"mjj{jet_coll}"] = ak.fill_none((padded[:, 0] + padded[:, 1]).mass, -999.0)
            self.events[f"deta{jet_coll}"] = ak.fill_none(
                abs(padded[:, 0].eta - padded[:, 1].eta), -999.0
            )

        # AK4 jets near the two AK8 Higgs candidates
        self.events["JetGoodCloseToFatJet"], _ = custom_jet_selection(
            self.events, "Jet", "JetNearFatJet", self.params, year=self._year,
            pt_type="pt", pt_cut_name="pt", forward_jet_veto=False,
        )

    def _nearest_ak4(self, higgs):
        """AK4 JetGoodCloseToFatJet closest in dR to the given Higgs candidate."""
        jets = self.events["JetGoodCloseToFatJet"]
        return ak.firsts(jets[ak.argsort(jets.delta_r(higgs), axis=1, ascending=True)])

    def _build_bdt_event_variables(self):
        """Populate self.events with every field get_default_bdt_inputs() reads."""
        fatjets = ak.pad_none(self.events.FatJetGood, 2, axis=1, clip=True)
        h1 = fatjets[:, 0]
        h2 = fatjets[:, 1]

        # --- per-Higgs derived quantities (build_boosted_variables) ---
        hh = add_fields(h1 + h2)

        near1 = self._nearest_ak4(h1)
        near2 = self._nearest_ak4(h2)

        def _decorate(h, near):
            h = ak.with_field(h, ak.fill_none(h.tau3 / h.tau2, -999), "Tau3OverTau2")
            h = ak.with_field(h, self._regressed_mass(h), "mass")
            h = ak.with_field(h, ak.fill_none(h.pt / hh.mass, -999), "divHHmass")
            txbb = self._txbb(h)
            h = ak.with_field(h, txbb, "btagBBTXbb")
            h = ak.with_field(h, disc_TXbb(txbb), "btagBBTXbb_dig")
            h = ak.with_field(h, ak.fill_none(h.delta_r(near), -999), "dRclosestVBF")
            h = ak.with_field(h, ak.fill_none((h + near).mass, -999), "massclosestVBF")
            return h

        h1 = _decorate(h1, near1)
        h2 = _decorate(h2, near2)

        self.events["HiggsLeading"] = h1
        self.events["HiggsSubLeading"] = h2
        self.events["HH"] = hh
        self.events["HiggsLeadingByHiggsSubLeadingPt"] = h1.pt / h2.pt

    # ------------------------------------------------------------------
    def process_extra_after_presel(self, variation):
        super().process_extra_after_presel(variation)

        if not self.bdt_model:
            return

        self.events["FatJetGoodSelected"] = self.events.FatJetGood[:, :2]

        self._build_vbf_ak4_collections()
        self._build_bdt_event_variables()

        bdt_inputs = get_default_bdt_inputs(self.events)
        ggf_score, vbf_score = evaluate_bdt(self.bdt_model, bdt_inputs)

        # The BDT is only meaningful for a two-AK8-jet (di-Higgs) topology.
        n_fatjet = ak.to_numpy(ak.num(self.events.FatJetGood, axis=1))
        valid = n_fatjet >= 2
        ggf_score = np.where(valid, ggf_score, np.nan)
        vbf_score = np.where(valid, vbf_score, np.nan)

        self.events["boosted_bdt_score"] = ggf_score
        self.events["boosted_bdt_vbf_score"] = vbf_score
        self.events["HiggsLeading_tau21"] = ak.fill_none(self.events.HiggsLeading.tau21, np.nan)
        self.events["HiggsSubLeading_tau21"] = ak.fill_none(self.events.HiggsSubLeading.tau21, np.nan)
