# Analysis workflow for calibration of boosted jets with the mu-tagged method
Repository for the calibration of large-radius jets with the mu-tagged method based on the [PocketCoffea](https://pocketcoffea.readthedocs.io/en/latest/index.html) analysis workflow.

## Installation
This package requires PocketCoffea >=0.9.8 as specified in the requirements.
The recommended running mode is to use the PocketCoffea singularity image available in `/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general`.
In order to run the mu-tagged method analysis written in PocketCoffea, we create a virtual environment and install both the `PocketCoffea` and `mutag-calib` packages,
by running the following commands:

```bash

# Clone locally the mutag-calib repository
git clone git@github.com:mmarchegiani/mutag-calib.git

#Enter the Singularity image
apptainer shell --bind /afs -B /cvmfs/cms.cern.ch -B /cvmfs/cms-griddata.cern.ch \
         --bind /tmp  --bind /eos/cms/ -B /etc/sysconfig/ngbauth-submit \
         -B ${XDG_RUNTIME_DIR}  --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc"  \
         /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-latest

# Create a local virtual environment using the packages defined in the apptainer image
python -m venv --system-site-packages myenv

# Activate the environment
source myenv/bin/activate

# Install mutag-calib in EDITABLE mode
cd mutag-calib
pip install -e .

# Set the PYTHONPATH to make sure the editable mutag-calib installation is picked up
export PYTHONPATH=`pwd`
```

Now the environment is setup. In order to use this environment in the future, just run:
```bash
# Enter the Singularity image
apptainer shell --bind /afs -B /cvmfs/cms.cern.ch -B /cvmfs/cms-griddata.cern.ch \
         --bind /tmp  --bind /eos/cms/ -B /etc/sysconfig/ngbauth-submit \
         -B ${XDG_RUNTIME_DIR}  --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc"  \
         /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-latest
# Activate the environment
source myenv/bin/activate

# Set the PYTHONPATH to make sure the editable mutag-calib installation is picked up
export PYTHONPATH=`pwd`
```


> [!TIP]
> 
> To avoid typing the long `apptainer shell` command every time, you can create an alias in your `~/.bashrc` file:
>
> ```bash
> alias sing='apptainer shell --bind /afs -B /cvmfs/cms.cern.ch -B /cvmfs/cms-griddata.cern.ch --bind /tmp --bind /eos/cms/ -B /etc/sysconfig/ngbauth-submit -B ${XDG_RUNTIME_DIR} --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc" /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-latest'
> ```
>
> After adding this alias, you can simply type `sing` to enter the Singularity image!

## Workflows
The folder `workflows` contains different PocketCoffea workflows that are needed to perform the analysis.
Here is their description:

- `pt_reweighting.py`: defines the `ptReweightProcessor` to compute the MC-to-data reweighting based on the 3D map in jet ($p_T$, $\eta$, $\tau_{21}$). It needs to be run once to compute the 3D reweighting map, that is then saved and used as a parameter for the next workflows.
- `fatjet_base.py`: defines the `fatjetBaseProcessor` that applies the custom object preselection needed for the calibration, creates all the branches that are counting the number of muons within the AK8 jet and creates branches specific to the taggers.
- `mutag_processor.py`: defines the `mutagAnalysisProcessor` on top of `fatjetBaseProcessor`. It applies the $p_T$ and $M_{SD}$ cuts on the AK8 jet collection and applies the 3D reweighting to MC.
- `mutag_oneMuAK8_processor.py`: defines the `mutagAnalysisOneMuonInAK8Processor` on top of `mutagAnalysisProcessor`. It applies the selection on AK8 jets based on the number of soft muons contained in the AK8 jet (>=1 soft muon within the AK8 jet).

## Analysis steps
### Step 0: produce datasets definitions
The folder `datasets` contains the `datasets_definitions_*.json` files which contain the metadata of all the Run 3 datasets used for the calibration, including DAS names, dataset and sample names, data/MC flag and cross-sections.
These files can be generated using the interactive `dataset-discovery-cli` command of PocketCoffea to query and select datasets interactively directly from DAS.

Datasets to be included for Run 3:

- [x] 2022+2023
- [ ] 2024
  - MC
    - [ ] QCD_MuEnriched
    - [ ] V+jets
    - [ ] Single top fully hadronic and semileptonic
    - [ ] ttbar fully hadronic
  - Data
    - [] BTagMu
  - MC for systematics
    - [x] QCD_HT
  - Validations
    - [ ] ggH(bb) and ggH(cc) signals (for validations)

⚠️ **Warning:** before entering in the singularity image, you need to run the command to get the GRID certificate:
```bash
voms-proxy-init --rfc --voms cms -valid 192:00
```
Now the GRID certificate is setup and you can enter in the singularity image.

To create json datasets, run the `build-datasets` command for each dataset definition file in the `datasets` folder:
```bash
pocket-coffea build-datasets --cfg datasets/datasets_definitions_DATA_BTagMu_run3.json -o
pocket-coffea build-datasets --cfg datasets/datasets_definitions_QCD_MuEnriched_run3.json -o
pocket-coffea build-datasets --cfg datasets/datasets_definitions_VJets_run3.json -o
pocket-coffea build-datasets --cfg datasets/datasets_definitions_TTto4Q_run3.json -o
pocket-coffea build-datasets --cfg datasets/datasets_definitions_singletop_semileptonic.json -o
pocket-coffea build-datasets --cfg datasets/datasets_definitions_singletop_fullyhadronic.json -o
pocket-coffea build-datasets --cfg datasets/datasets_definitions_singletop_s-channel.json -o
```

Restricting the dataset source in Europe (recommended for working from lxplus):
```bash
pocket-coffea build-datasets --cfg datasets/datasets_definitions_DATA_BTagMu_run3.json -o -rs 'T[123]_(FR|IT|DE|BE|CH|UK)_\w+'
pocket-coffea build-datasets --cfg datasets/datasets_definitions_QCD_MuEnriched_run3.json -o -rs 'T[123]_(FR|IT|DE|BE|CH|UK)_\w+'
pocket-coffea build-datasets --cfg datasets/datasets_definitions_VJets_run3.json -o -rs 'T[123]_(FR|IT|DE|BE|CH|UK)_\w+'
pocket-coffea build-datasets --cfg datasets/datasets_definitions_TTto4Q_run3.json -o -rs 'T[123]_(FR|IT|DE|BE|CH|UK)_\w+'
pocket-coffea build-datasets --cfg datasets/datasets_definitions_singletop_semileptonic.json -o -rs 'T[123]_(FR|IT|DE|BE|CH|UK)_\w+'
pocket-coffea build-datasets --cfg datasets/datasets_definitions_singletop_fullyhadronic.json -o -rs 'T[123]_(FR|IT|DE|BE|CH|UK)_\w+'
pocket-coffea build-datasets --cfg datasets/datasets_definitions_singletop_s-channel.json -o -rs 'T[123]_(FR|IT|DE|BE|CH|UK)_\w+'
```

It is recommended to re-run the `build-datasets` command once in a while to update the file list of the datasets, as files from specific sites that are available in a given moment might not be available later on.

For more detailed instructions on how to create datasets in PocketCoffea, please follow [this guide](https://pocketcoffea.readthedocs.io/en/latest/datasets.html#datasets-handling).

Once the json datasets are created, the configuration files have to be updated to run on the newly defined datasets.

### Step 0: update the trigger paths
In order to run the analysis on Run 3 data, the trigger paths specified in the `mutag_calib/configs/params/triggers.yaml` file have to be updated with the Run 3 trigger paths. Trigger paths included:
- [x] `2022_preEE`
- [x] `2022_postEE`
- [x] `2023_preBPix`
- [x] `2023_postBPix`
- [x] `2024`

### Step 1: compute 3D reweighting based on jet $p_T$, $\eta$, $\tau_{21}$
Run jobs on DATA, QCD, V+jets and top datasets:
```bash
pocket-coffea run --cfg mutag_calib/configs/pt_reweighting/ptreweighting_run3.py -o pt_reweighting -e dask@lxplus -ro mutag_calib/configs/params/run_options.yaml --process-separately 
```

> [!TIP]
> 
> The argument `--process-separately` will save one output .coffea file for each dataset. This is convenient because your processor will often crash on a specific dataset. In order to avoid losing precious CPU time for a single faulty dataset, it is better to save an output file for each dataset and then merge all the files into a single one after processing.
> It is possible to merge datasets belonging to a specific sample while still procesing separately the remaining datasets. This is possible by passing custom run options with the `-ro` argument. For instance, by adding these lines to the `mutag_calib/configs/params/run_options.yaml` file:
> ```yaml
> group-samples:
>   TTto4Q:
>     - "TTto4Q"
>   Vjets:
>     - "VJets"
>   SingleTop:
>     - "SingleTop"
> ```
> Single output files including all data-taking years will be saved: `output_TTto4Q.coffea`, `output_Vjets.coffea` and `output_SingleTop.coffea`. Instead for QCD and DATA, one output file for each dataset and data-taking year will be saved, for example: `output_QCD_PT-170to300_MuEnrichedPt5_Pt-170to300_2023_preBPix.coffea`.
> 

After merging the output files with the `merge-outputs` command, it is possible to plot all the histograms and produce the 3D reweighting map.

Produce the plots of all 1D histograms with:
```bash
cd pt_reweighting
make-plots -i output_all.coffea
```
The plots are saved in the `plots` subfolder. 

Produce 3D reweighting map with:
```bash
python mutag_calib/scripts/compute_3d_reweighting.py -i pt_reweighting/output_all.coffea -o pt_reweighting/3d_reweighting --test
```

### Step 2: Run jobs to produce fit templates in all the tagger categories
To apply the 3D reweighting to the QCD MC sample, it is necessary to include the 3D reweighting maps in the parameters.
For each data taking year, the corresponding file computed in Step 1 has to be specified in the `mutag_calib/configs/params/ptetatau21_reweighting.yaml` parameter file.
Dedicated parameter files can be found in the folder `mutag_calib/configs/params`.
For example, `mutag_calib/configs/params/mutag_calibration.yaml`, is used to specify all the parameters related to the mu-tagged calibration for the HHbbtt analysis. In this file are specified the AK8 taggers to calibrate, the $p_T$ and $M_{SD}$ binning and the tagger working points.
Example:
```yaml
mutag_calibration:
  taggers:
    - particleNet_XbbVsQCD
  pt_binning:
    2022_preEE:
      - [300, 'Inf']
      - [300, 350]
      - [350, 425]
      - [425, 'Inf']
  msd_binning:
    2022_preEE:
      - [40, 'Inf']
      - [80, 170]
  wp:
    2022_preEE:
      particleNet_XbbVsQCD:
        HHbbtt: 0.75
```

Launch jobs to produce fit templates including 3D reweighting of QCD sample (**command specific to the HHbbtt analysis**):
```bash
pocket-coffea run --cfg mutag_calib/configs/fit_templates/fit_templates_HHbbtt.py -o fit_templates_HHbbtt -e dask@lxplus --custom-run-options mutag_calib/configs/params/run_options.yaml --process-separately
```

Similarly for the **HHbbgg analysis**:
```bash
pocket-coffea run --cfg mutag_calib/configs/fit_templates/fit_templates_HHbbgg.py -o fit_templates_HHbbgg -e dask@lxplus --custom-run-options mutag_calib/configs/params/run_options.yaml --process-separately
```

Merge outputs and produce plots with similar commands as in Step 2.

### Step 3: Produce fit shapes and combine datacards
In order to produce the fit shapes and combine datacards for the fit, a dedicated script can be run with the following command:
```bash
python mutag_calib/scripts/create_datacards.py fit_templates/output_all.coffea --years 2022_preEE 2022_postEE 2023_preBPix 2023_postBPix
```
The script will create a folder `fit_templates/datacards` containing a system of subfolders organized by data-taking era, category, $\tau_{21}$ cut and pass/fail regions. The folder will have the following structure:
```bash
├── 2022_postEE
│   ├── msd-30to210_Pt-300toInf_particleNet_XbbVsQCD-HHbbgg
│   │   ├── tau21_0p20
│   │   │   ├── combine_cards.sh
│   │   │   ├── fail
│   │   │   │   ├── datacard.txt
│   │   │   │   └── shapes.root
│   │   │   ├── pass
│   │   │   │   ├── datacard.txt
│   │   │   │   └── shapes.root
│   │   │   └── passfail_ratio.yaml
│   │   ├── tau21_0p25
│   │   │   ├── combine_cards.sh
│   │   │   ├── fail
│   │   │   │   ├── datacard.txt
│   │   │   │   └── shapes.root
│   │   │   ├── pass
│   │   │   │   ├── datacard.txt
│   │   │   │   └── shapes.root
│   │   │   └── passfail_ratio.yaml
```
The datacards for the pass and fail regions of the tagger are stored in separate subfolders of the $\tau_{21}$ folder, `pass` and `fail`.

### Step 4: Run combine fit
In order to run the maximum likelihood fit to extract the AK8 jet scale factors, it is necessary to use the [Combine](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/) package. This package is based on RooFit and it runs outside of the Python environment used for the production of the fit templates. A dedicated CMSSW installation needs to be setup to use the Combine statistical tool.

Do a fresh login to a lxplus machine without entering in the PocketCoffea singularity image. If you are in the singularity image, type `exit` to exit the container.
In a clean environment, setup the combine tool following the recommendations in the [Combine documentation](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/#combine-v10-recommended-version). The same commands can be found in the `setup_combine.sh` script. You can run it with the following command:
```bash
source setup_combine.sh
```
This will setup a `CMSSW_14_1_0_pre4` installation and build the necessary packages. The installation might take a few minutes.

Once the installation is completed, you can use the newly installed CMSSW environment with the usual command:
```bash
cd CMSSW_14_1_0_pre4/src
cmsenv
cd -
```

Move to the folder of the category and $\tau_{21}$ cut that you are interested to calibrate. A bash script `combine_cards.sh` is provided to combine the datacards in the pass and fail regions. Run the `combine_cards.sh` script with the following command:
```bash
source combine_cards.sh
```
This script will create the Combine workspace file, `workspace.root`, and the combined datacard text file, `datacard_combined.txt`.

Now you can run the usual Combine commands to perform maximum likelihood fits and extract the Xbb scale factors!

Example Combine scripts to extract the SF can be found in the `mutag_calib/scripts/fit` folder. Example:
```bash
combine -M FitDiagnostics -d workspace.root --saveWorkspace \
--name .msd-80to170_Pt-300toInf_particleNet_XbbVsQCD-HHbbtt --cminDefaultMinimizerStrategy 2 \
--robustFit=1 --saveShapes --saveWithUncertainties --saveOverallShapes \
--redefineSignalPOIs=r,SF_c,SF_light --setParameters SF_light=1 --freezeParameters SF_light \
--robustHesse=1 --stepSize=0.001 --X-rtd=MINIMIZER_analytic --X-rtd MINIMIZER_MaxCalls=9999999 \
--cminFallbackAlgo Minuit2,Migrad,0:0.2 --X-rtd FITTER_NEW_CROSSING_ALGO --X-rtd FITTER_NEVER_GIVE_UP --X-rtd FITTER_BOUND
```

Note that the `--redefineSignalPOIs=r,SF_c,SF_light` is added to the combine command in order to perform a fit with 3 POIs to measure the b, c and light SFs simultaneously. For the calibration of bb taggers such as `ParticleNetXbbVsQCD`, it is sometimes necessary to freeze the light SF by adding `--setParameters SF_light=1 --freezeParameters SF_light`, since in the pass region of this tagger the light jets are negligible and the light SF cannot be measured.

#### Systematic uncertainties

The script create_datacards.py defines the following systematic uncertainties in the datacards:

| Systematic uncertainty       | Description                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------------|
| lumi                        | Integrated luminosity uncertainty (lnN) applied to all MC processes, year-dependent value.       |
| pileup                      | Shape uncertainty associated to pileup reweighting of MC.                                         |
| QCD_MuEnriched_ratio        | Shape uncertainty accounting for the difference between QCD simulation in Pythia / Madgraph. |
| sf_partonshower_isr         | Shape uncertainty from initial-state radiation (ISR) parton-shower variations.                          |
| sf_partonshower_fsr         | Shape uncertainty from final-state radiation (FSR) parton-shower variations.                            |
| AK8PFPuppi_JER              | Shape uncertainty from AK8 PFPuppi jet energy resolution (JER) variations.                              |
| AK8PFPuppi_JES_Total        | Shape uncertainty from AK8 PFPuppi total jet energy scale (JES) variations.                             |
| light_norm                  | 20% normalization (lnN) uncertainty on the light-flavour QCD component.                           |
| c_norm                      | 20% normalization (lnN) uncertainty on the charm-flavour QCD component.                           |

In addition, the following "external" systematic uncertainties are summed in quadrature to the SF obtained from the Combine fit:

| External systematic uncertainty       | Description                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------------|
| tau21                        | Maximum difference between the nominal SF (tau21 < 0.3) and any of the alternative SF (tau21 < 0.2, 0.25, 0.35, 0.4), to account for the dependence on the tau21 cut.        |
| Data / MC reweighting                        | Difference between the nominal SF (tau21 < 0.3) and the SF obtained by reweighting the fit variable in the inclusive pass+fail region, to account for the residual mismodelling in the fit variable.        |

