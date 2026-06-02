"""Workaround subclass for PocketCoffea's JetsSoftdropMassCalibrator.

Fixes two issues:
1. PocketCoffea case bug: checks "2016_preVFP" (lowercase) but year labels are "2016_PreVFP".
2. PocketCoffea calibrator chain bug: calibrate() returns the entire FatJet collection
   (captured at init time with uncorrected pt), which overwrites the JEC-corrected pt
   from JetsCalibrator. Fix: only return the msoftdrop field via "FatJet.msoftdrop" format.
"""

import copy
from pocket_coffea.lib.calibrators.common.common import JetsSoftdropMassCalibrator
from pocket_coffea.lib.jets import msoftdrop_correction
from pocket_coffea.utils.utils import get_nano_version


class FixedJetsSoftdropMassCalibrator(JetsSoftdropMassCalibrator):
    """JetsSoftdropMassCalibrator with fixed 2016 year labels and field-level replacement."""

    def initialize(self, events):
        # Load the calibration of each jet type requested by the parameters
        for jet_type, jet_coll_name in self.jet_calib_param.collection[self.year].items():
            # Calibrate only AK8 jets
            if jet_type in ["AK8PFPuppi"]:
                # NanoAODv15 uses PUPPI subjets for all years
                subjet_type = "AK4PFPuppi"
            else:
                print("WARNING: JetsSoftdropMassCalibrator only supports AK8PFPuppi jets for softdrop mass calibration." +
                      f" Jet type {jet_type} will be skipped.")
                continue

            # Check if the collection is enabled in the parameters
            if self.isMC:
                if (self.jet_calib_param.apply_jec_msoftdrop_MC[self.year][jet_type] == False):
                    continue
            else:
                if self.jet_calib_param.apply_jec_msoftdrop_Data[self.year][jet_type] == False:
                    continue

            # Register as field-level calibration so the manager only replaces msoftdrop,
            # not the entire FatJet collection (which would overwrite JEC-corrected pt).
            self.calibrated_collections.append(f"{jet_coll_name}.msoftdrop")

            # N.B.: since the correction is applied to the subjets of the AK8 jet, the parameters for AK4 jets are passed
            self.jets_calibrated[jet_coll_name] = msoftdrop_correction(
                calib_params=self.jet_calib_param.jet_types[subjet_type][self._year],
                variations=self.jet_calib_param.variations[subjet_type][self._year],
                events=events,
                subjet_type = subjet_type,   # AK4PFPuppi
                jet_coll_name=jet_coll_name, # AK8PFPuppi
                chunk_metadata={
                    "year": self._year,
                    "isMC": self.metadata["isMC"],
                    "era": self.metadata["era"] if "era" in self.metadata else None,
                    "nano_version": get_nano_version(events, self.params, self._year),
                },
                jec_syst=self.do_variations
            )
            self.jets_calibrated_types.append(jet_type)

        assert len(self.jets_calibrated_types) > 0, "No jet types were calibrated in JetsSoftdropMassCalibrator. Please check the configuration."

        available_jet_variations = []
        # N.B.: JES variations are not yet implemented for msoftdrop
        self._variations = list(sorted(set(available_jet_variations)))

    def calibrate(self, events, orig_colls, variation, already_applied_calibrators=None):
        out = {}
        for jet_coll_name, jets in self.jets_calibrated.items():
            # Only return the msoftdrop field, not the entire collection.
            # This prevents overwriting JEC-corrected pt from JetsCalibrator.
            out[f"{jet_coll_name}.msoftdrop"] = jets.msoftdrop

        if variation == "nominal" or variation not in self._variations:
            return out

        # # Shape variations for msoftdrop (not yet implemented upstream)
        # variation_parts = variation.split("_")
        # jet_type = variation_parts[0]
        # if jet_type not in self.jet_calib_param.collection[self.year]:
        #     raise ValueError(f"Jet type {jet_type} not found in the parameters for year {self.year}.")
        # jet_coll_name = self.jet_calib_param.collection[self.year][jet_type]
        # if jet_coll_name not in self.jets_calibrated:
        #     raise ValueError(f"Jet collection {jet_coll_name} not found in the calibrated jets.")
        #
        # if variation.endswith("Up"):
        #     variation_type = "_".join(variation_parts[1:])[:-2]
        #     out[f"{jet_coll_name}.msoftdrop"] = self.jets_calibrated[jet_coll_name][f"msoftdrop_{variation_type}_up"]
        # elif variation.endswith("Down"):
        #     variation_type = "_".join(variation_parts[1:])[:-4]
        #     out[f"{jet_coll_name}.msoftdrop"] = self.jets_calibrated[jet_coll_name][f"msoftdrop_{variation_type}_down"]
        #
        # return out
