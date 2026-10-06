"""AK4 VBF-jet reconstruction helpers for the optional boosted-BDT evaluation.

Vendored (verbatim) from ``AnalysisConfigs_bossted_dev``:
  - ``add_fields``                        <- ``utils_configs/basic_functions.py``
  - ``object_cleaning_mask``              <- ``utils_configs/custom_cut_functions.py``
  - ``custom_jet_selection``              <- ``utils_configs/custom_cut_functions.py``
  - ``get_forward_jet_veto``              <- ``utils_configs/custom_cut_functions.py``
  - ``get_lead_mjj_jet_pair``            <- ``utils_configs/reconstruct_higgs_candidates.py``

These reproduce, unchanged, the AK4 VBF-jet selection used by
``configs/VBF_HH4b_boosted/workflow.py`` so the BDT inputs are built identically.
Only used by ``mutag_calib.workflows.mutag_bdt_processor`` and only when a
``bdt_model`` is configured; nothing here runs otherwise.
"""

import copy

import awkward as ak
import numpy as np
import vector

from pocket_coffea.lib.jets import jet_selection

vector.register_awkward()


# ---------------------------------------------------------------------------
# utils_configs/basic_functions.py
# ---------------------------------------------------------------------------
def add_fields(collection, fields=None, four_vec="PtEtaPhiMLorentzVector"):
    if fields == "all":
        fields = list(collection.fields)
        for field in ["pt", "eta", "phi", "mass"]:
            if field not in fields:
                fields.append(field)

        # remove 3d fields
        fields = [f for f in fields if getattr(collection, f).ndim <= 2]

    elif fields is None:
        fields = ["pt", "eta", "phi", "mass"]
        fields_add = [
            "pt_raw",
            "mass_raw",
            "PNetRegPtRawRes",
            "PNetRegPtRawCorr",
            "PNetRegPtRawCorrNeutrino",
            "btagPNetB",
            "index",
        ]
        for field in fields_add:
            if field in list(collection.fields):
                fields.append(field)

    if not isinstance(fields, list):
        raise ValueError("fields must be a list of fields or 'all' or None")

    if four_vec == "PtEtaPhiMLorentzVector":
        fields_dict = {field: getattr(collection, field) for field in fields}
        collection = ak.zip(
            fields_dict,
            with_name="PtEtaPhiMLorentzVector",
        )
    elif four_vec == "Momentum4D":
        fields_dict = {field: getattr(collection, field) for field in fields}
        collection = ak.zip(
            fields_dict,
            with_name="Momentum4D",
        )
    else:
        for field in fields:
            collection = ak.with_field(collection, getattr(collection, field), field)

    return collection


# ---------------------------------------------------------------------------
# utils_configs/custom_cut_functions.py
# ---------------------------------------------------------------------------
def object_cleaning_mask(obj, cleaning_collection, dr_min=0.4):
    # here I create a deltaR matrix between jets and cleaning collection the output shape is (njets, ncleaning)
    dR = obj[:, :, None].delta_r(cleaning_collection[:, None, :])

    # then I check if the jets are within dR min of ANY cleaning object
    dR_mask = dR < dr_min
    dR_mask_jets = ~ak.any(dR_mask, axis=2)

    return dR_mask_jets


def custom_jet_selection(
    events,
    jet_type,
    jet_type_obj_presel,
    params,
    year,
    leptons_collection="",
    jet_tagger="",
    pt_type="pt",
    pt_cut_name="pt",
    forward_jet_veto=False,
):
    """
    Custom jet selection function to apply selection on different pt types.
    Args:
        events: awkward array with events
        jet_type: str, type of jet to select (e.g. "Jet")
        params: configuration parameters
        year: year of the data-taking period
        leptons_collection: str, type of leptons to consider for overlap removal
        jet_tagger: str, jet tagger to use
        pt_type: str, type of pt to apply the cut on (e.g. "pt", "pt_default", "pt_regressed")
        pt_cut_name: str, name of the pt cut in the params (e.g. "pt", "pt_tight")
    """
    jet_type_default = "Jet" if not "FatJet" in jet_type else "FatJet"

    # create a copy of params to avoid modifying the original one
    # and put in the collection the AK4PFPuppi to compute the jetId
    # because whatever jet_type is passed the jetId should be computed with AK4PFPuppi
    # since it's the one having the jetId stored
    # copy also the object_preselection to modify it
    params_copy = copy.copy(params)
    params_copy.object_preselection[jet_type_default] = params_copy.object_preselection[
        jet_type_obj_presel
    ].copy()
    params_copy.object_preselection[jet_type_default]["pt"] = (
        params_copy.object_preselection[jet_type_obj_presel][pt_cut_name]
    )
    params_copy.jets_calibration.collection[year] = {"AK4PFPuppi": jet_type_default}

    # create a copy of events to avoid modifying the original one
    # replace the jet_type_default collection with the jet_type one
    events_copy = copy.copy(events)
    events_copy[jet_type_default] = ak.with_field(
        events_copy[jet_type],
        events_copy[jet_type][pt_type],
        "pt",
    )

    _, selection_mask = jet_selection(
        events_copy,
        jet_type_default,
        params_copy,
        year,
        leptons_collection,
        jet_tagger,
    )

    if forward_jet_veto:
        # Apply forward jet veto
        _, forward_mask = get_forward_jet_veto(events, jet_type, pt_type)
        mask = selection_mask & forward_mask
    else:
        mask = selection_mask

    obj_param = params_copy.object_preselection[jet_type_obj_presel]
    if "dr_jet" in obj_param.keys():
        clean_jet_coll = obj_param["clean_jet_coll"] if "clean_jet_coll" in obj_param.keys() else "FatJetGoodSelected"
        mask = mask & object_cleaning_mask(events_copy[jet_type_default], events_copy[clean_jet_coll], obj_param["dr_jet"])
    if "dr_lep" in obj_param.keys():
        eles = events_copy[obj_param["clean_ele_coll"] if "clean_ele_coll" in obj_param.keys() else "Electron"]
        muons = events_copy[obj_param["clean_mu_coll"] if "clean_mu_coll" in obj_param.keys() else "Muon"]
        if "clean_ele_pt" in obj_param.keys():
            eles = eles[eles["pt"] > obj_param["clean_ele_pt"]]
        if "clean_mu_pt" in obj_param.keys():
            muons = muons[muons["pt"] > obj_param["clean_mu_pt"]]
        mask = mask & object_cleaning_mask(events_copy[jet_type_default], eles, obj_param["dr_lep"])
        mask = mask & object_cleaning_mask(events_copy[jet_type_default], muons, obj_param["dr_lep"])

    # remove copies
    del params_copy
    del events_copy

    return events[jet_type][mask], mask


def get_forward_jet_veto(events, jet_type, pt_type):
    # jets rejected if pT < 50 GeV and  2.5 < |eta| < 3
    eta_range = (abs(events[jet_type].eta) < 2.5) | (abs(events[jet_type].eta) > 3.0)
    pt_mask = events[jet_type][pt_type] > 50

    mask = eta_range | pt_mask

    return events[jet_type][mask], mask


# ---------------------------------------------------------------------------
# utils_configs/reconstruct_higgs_candidates.py
# ---------------------------------------------------------------------------
def get_lead_mjj_jet_pair(events, jet_coll):
    """Choose the two jets with the highest mjj."""
    jet = events[jet_coll]

    # Adds none jets to events that have less than 2 jets
    # jet_padded for each event can be either
    # [None, None], [Jet, None], [Jet, Jet, ...]
    jet_padded = ak.pad_none(jet, 2)

    # get combinations of jet and choose the one with highest mjj
    jet_combinations = ak.combinations(jet_padded, 2)
    jet_combinations_mass = (jet_combinations["0"] + jet_combinations["1"]).mass
    jet_combinations_mass_padded = ak.fill_none(jet_combinations_mass, -np.inf)
    jet_combinations_mass_max_idx = ak.to_numpy(
        ak.argsort(jet_combinations_mass_padded, axis=1, ascending=False)[:, 0]
    )
    jets_max_mass = jet_combinations[
        ak.local_index(jet_combinations, axis=0), jet_combinations_mass_max_idx
    ]

    # get the two jets with the highest mjj
    lead_mjj_jet_0 = ak.unflatten(
        jets_max_mass["0"],
        1,
    )
    lead_mjj_jet_1 = ak.unflatten(
        jets_max_mass["1"],
        1,
    )
    lead_mjj_jet_pair = ak.with_name(
        ak.concatenate([lead_mjj_jet_0, lead_mjj_jet_1], axis=1),
        name="PtEtaPhiMCandidate",
    )
    lead_mjj_jet_pair = add_fields(lead_mjj_jet_pair, "all")

    energy = ak.fill_none(lead_mjj_jet_pair.energy, -np.inf)
    # order the jets according to the energy
    lead_mjj_jet_pair = lead_mjj_jet_pair[
        ak.argsort(energy, axis=1, ascending=False)
    ]

    return lead_mjj_jet_pair
