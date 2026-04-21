import os
import uproot
from pocket_coffea.utils.stat import MCProcess, Datacard

class DatacardMutag(Datacard):
    """Custom Datacard class for mutag calibration.
    This class extends the base Datacard class to implement specific features
    needed for the mutag calibration analysis, such as handling pass/fail categories.
    Since the effect of the rate parameter is different in the fail region,
    we need to modify the rate parameter section accordingly.
    The effect of the rate parameter in the fail region is:
    SF * (1 + (1 - SF) * R)
    where R is the pass/fail ratio."""
    def get_passfail_formula(self, process : MCProcess, year: str, passfail_ratio: dict) -> str:
        """
        Get the formula for the rate parameter based on the pass/fail ratio.
        This function implement the different effect of the rate parameter in the fail region.
        :param process: The MC process for which to get the formula.
        :type process: MCProcess
        :param year: The year for which to get the formula.
        :type year: str
        :param passfail_ratio: Dictionary containing the pass/fail ratios
        :type passfail_ratio: dict
        :returns: The formula string for the rate parameter.
        :rtype: str
        """
        key = f"{process.name}_{year}"
        if passfail_ratio is None or key not in passfail_ratio:
            raise KeyError(f"Pass/fail ratio for '{key}' not found in passfail_ratio")
        # format ratio with reasonable precision
        ratio = float(passfail_ratio[key])
        ratio_str = f"{ratio:.6g}"
        # return the expression part that will follow the channel/process entry in the rateParam line
        # e.g. "* proc_2022 (((1-@0)*0.8)+1)"
        return f"(((1-@0)*{ratio_str})+1)"


    def rate_parameters_section(self, passfail_ratio=None) -> str:
        """
        Generate the rate parameters section of the datacard.
        :param passfail_ratio: Optional dictionary of passfail_ratio for the rate parameters.
        :type passfail_ratio: dict, optional
        """
        content = ""
        for process in self.mc_processes.values():
            for year in process.years:
                if process.has_rateParam:
                    if process.is_signal:
                        rate_param_name = "r"
                    else:
                        rate_param_name = f"SF_{process.name}"
                    line = rate_param_name.ljust(self.adjust_syst_colum)
                    line += "rateParam".ljust(self.adjust_columns)
                    if passfail_ratio is None:
                        line += f"* {process.name}_{year} 1 [0,5]".ljust(
                            self.adjust_columns
                        )
                    else:
                        formula = self.get_passfail_formula(process, year, passfail_ratio)
                        line += f"* {process.name}_{year} {formula} {rate_param_name}".ljust(
                            self.adjust_columns
                        )
                    line += self.linesep
                    content += line
        return content
    
    def content(self, shapes_filename: str, passfail_ratio : dict = None) -> str:
        """
        Generate the content of the datacard.

        :param shapes_filename: The filename of the root file containing the shape histograms.
        :type shapes_filename: str

        :returns: Content of the datacard as a string.
        :rtype: str
        """
        content = self.preamble()
        content += self.sectionsep + self.linesep

        content += self.shape_section(shapes_name=shapes_filename)
        content += self.sectionsep + self.linesep

        content += self.observation_section()
        content += self.sectionsep + self.linesep

        content += self.expectation_section()
        content += self.sectionsep + self.linesep

        content += self.systematics_section()
        content += self.sectionsep + self.linesep

        content += self.rate_parameters_section(passfail_ratio=passfail_ratio)
        content += self.sectionsep + self.linesep

        if self.mcstat:
            content += self.mcstat_section()
            content += self.sectionsep + self.linesep

        return content

    def dump(
        self,
        directory: os.PathLike,
        card_name: str = "datacard.txt",
        shapes_name: str = "shapes.root",
        passfail_ratio: dict = None,
    ) -> None:
        """Dump datacard and shapes to a directory.

        :param directory: Directory to dump the datacard and shapes
        :type directory: os.PathLike
        :param card_name: name of the datacard file, defaults to "datacard.txt"
        :type card_name: str, optional
        :param shapes_filename: name of the shapes file, defaults to "shapes.root"
        :type shapes_filename: str, optional
        """

        card_file = os.path.join(directory, card_name)
        shapes_file = os.path.join(directory, shapes_name)

        os.makedirs(directory, exist_ok=True)

        with open(card_file, "w") as card:
            card.write(self.content(shapes_filename=shapes_name, passfail_ratio=passfail_ratio))

        shape_histograms = self.create_shape_histogram_dict(is_data=False)
        if self.has_data:
            shape_histograms_data = self.create_shape_histogram_dict(is_data=True)
        with uproot.recreate(shapes_file) as root_file:
            if self.has_data:
                for shape, histogram in shape_histograms_data.items():
                    root_file[shape] = histogram
            for shape, histogram in shape_histograms.items():
                root_file[shape] = histogram

    @property
    def bin(self) -> str:
        """Name of the bin in the datacard"""
        bin_name = self.category.replace('-', '_')
        if self.bin_prefix:
            bin_name = f"{self.bin_prefix}_{bin_name}"
        if self.bin_suffix:
            bin_name = f"{bin_name}_{self.bin_suffix}"
        return bin_name
