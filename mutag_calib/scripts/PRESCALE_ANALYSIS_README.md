# Trigger Prescale Analysis Scripts

This directory contains scripts for analyzing trigger prescale factors from the JSON correction files referenced in `triggers_prescales_run3.yaml`.

## Overview

The trigger prescale system in CMS reduces the rate of certain triggers by only accepting a fraction of events. The prescale factor indicates how many events are skipped - for example, a prescale of 2 means only every 2nd event is accepted, while a prescale of 0 means the trigger is completely disabled.

## Generating prescale files

### 1. `dump_prescale.py` - Compute prescales from brilcalc

Queries brilcalc for each run in a golden JSON to produce per-run correctionlib JSON files. Must be run **on lxplus outside any container** (uses the brilws Singularity container on CVMFS).

```bash
# Golden JSONs are provided by PocketCoffea's defaults (pocket_coffea/parameters/datacert/)
# Find the path with: python3 -c "import pocket_coffea, os; print(os.path.join(os.path.dirname(pocket_coffea.__file__), 'parameters', 'datacert'))"

python3 mutag_calib/scripts/dump_prescale.py \
  --lumimask "$(python3 -c "import pocket_coffea, os; print(os.path.join(os.path.dirname(pocket_coffea.__file__), 'parameters', 'datacert', 'Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt'))")" \
  --HLT "BTagMu_AK8Jet300_Mu5,BTagMu_AK4Jet300_Mu5" \
  --output-dir mutag_calib/configs/params/prescales/run2/ \
  --force
```

### 2. `generate_prescale_yaml.py` - Generate average prescale YAML

Reads the per-run JSON files and computes lumi-range-weighted average prescales, outputting a YAML file compatible with `scale_factors.py`.

```bash
python3 mutag_calib/scripts/generate_prescale_yaml.py \
  --config mutag_calib/configs/params/triggers_prescales_2024.yaml \
  -o average_prescales.yaml
```

The input config must use the JSON-path format (e.g. `triggers_prescales_2024.yaml`), where values point to per-run JSON files rather than scalar numbers.

## Analyzing prescale files

### 3. `analyze_prescales.py`

This is the main analysis script that processes all prescale JSON files and calculates various statistics.

**Features:**
- Loads and parses all prescale JSON files from the configuration
- Calculates average prescale factors by run, HLT path, and overall
- Handles both constant prescales and luminosity-section-dependent prescales
- Exports detailed results to CSV files

**Usage:**
```bash
# Analyze all years and trigger groups
python mutag_calib/scripts/analyze_prescales.py

# Analyze specific year
python mutag_calib/scripts/analyze_prescales.py --year 2022_preEE

# Analyze specific trigger group
python mutag_calib/scripts/analyze_prescales.py --trigger-group BTagMu

# Custom output directory
python mutag_calib/scripts/analyze_prescales.py --output-dir my_analysis
```

**Or use the helper script:**
```bash
# Analyze all data
./run_prescale_analysis.sh

# Analyze specific year and trigger group
./run_prescale_analysis.sh 2022_preEE BTagMu
```

### 4. `interactive_prescale_analysis.py` - Detailed Analysis

This script provides more detailed analysis capabilities and can generate plots.

**Features:**
- Interactive analysis of specific run ranges
- Luminosity section pattern analysis
- Visualization plots (histograms, trends by run, etc.)
- Detailed statistics for specific periods

**Usage:**
```bash
# Basic analysis with plots
python mutag_calib/scripts/interactive_prescale_analysis.py --plot

# Analyze specific run range
python mutag_calib/scripts/interactive_prescale_analysis.py --run-start 355000 --run-end 356000

# Generate plots in custom directory
python mutag_calib/scripts/interactive_prescale_analysis.py --plot --output-dir my_plots
```

## Output Files

### From `analyze_prescales.py`:

- **`prescale_raw_data.csv`**: All prescale entries with metadata
- **`averages_by_hlt_path.csv`**: Average prescales by HLT trigger path
- **`averages_by_run.csv`**: Average prescales by run number
- **`averages_by_run_and_path.csv`**: Average prescales by run and HLT path combination
- **`overall_statistics.json`**: Overall statistics summary

### From `interactive_prescale_analysis.py` (with --plot):

- **`prescale_distribution.png`**: Histogram of prescale values
- **`prescales_by_run.png`**: Trend of average prescales by run number
- **`prescales_by_hlt_path.png`**: Average prescales by HLT path

## Understanding the Data

### JSON File Structure

The prescale JSON files follow the correctionlib format with this structure:
```
run_number → HLT_path → luminosity_sections → prescale_weight
```

### Prescale Values

- **0**: Trigger is disabled (unprescaled but off)
- **1**: No prescaling (trigger fires on every qualifying event)
- **2**: Accept every 2nd event (50% rate reduction)
- **N**: Accept every Nth event (rate reduced by factor of N)

### Luminosity Section Binning

Some prescales are constant for an entire run, while others change during the run based on luminosity sections. The scripts handle both cases:

- **Constant prescales**: Applied to entire run
- **Binned prescales**: Different prescales for different lumi section ranges within a run

## Example Analysis Workflow

1. **Run comprehensive analysis:**
   ```bash
   ./run_prescale_analysis.sh
   ```

2. **Examine specific problematic periods:**
   ```bash
   python mutag_calib/scripts/interactive_prescale_analysis.py --run-start 355000 --run-end 356000 --plot
   ```

3. **Check results:**
   - Look at `averages_by_run.csv` to identify runs with unusual prescales
   - Examine `prescale_distribution.png` to understand the overall pattern
   - Use `averages_by_hlt_path.csv` to compare different triggers

## Configuration

The scripts automatically read the prescale file paths from:
```
mutag_calib/configs/params/triggers_prescales_run3.yaml
```

The YAML file contains mappings like:
```yaml
HLT_triggers_prescales:
  2022_preEE:
    BTagMu:
      BTagMu_AK8Jet300_Mu5: ${config_dir:}/prescales/ps_weight_BTagMu_AK8Jet300_Mu5_run355374_362760.json
```

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib (for plotting)
- seaborn (for plotting)
- PyYAML

Install with:
```bash
pip install pandas numpy matplotlib seaborn pyyaml
```

## Tips

1. **Zero prescales**: A high fraction of zero prescales indicates periods where triggers were disabled
2. **Prescale trends**: Look for patterns in prescales by run number - this often correlates with luminosity conditions
3. **HLT path comparison**: Different HLT paths may have very different prescale patterns
4. **Luminosity correlation**: Higher luminosity periods typically have higher prescales to manage trigger rates
