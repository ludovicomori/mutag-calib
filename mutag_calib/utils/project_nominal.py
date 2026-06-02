"""Harmonize variation axes across MC samples so make-plots can group them.
The issue: different MC samples have different variation axis entries (e.g., QCD_MuEnriched
has QCD_MuEnriched_ratio while VJets has sf_partonshower_isr/fsr). hist.Stack requires
identical axes, so grouping fails. This script rebuilds each MC histogram keeping only
the 'nominal' entry on the variation axis (but preserving the axis itself).
Optional --merge-years for the special mixed-year case (e.g. 2025 SFs from 2024 MC + 2025
data). PocketCoffea splits plots by datataking_period, so MC-only and data-only plots are
produced. This flag merges the specified year buckets in datasets_metadata so make-plots
overlays them on a single plot.
Usage:
    python project_nominal.py fit_templates_HHbbgg_2024_glopart/output_all.coffea
    # Creates output_all_nominal.coffea in the same directory
    # Mixed-year case (2024 MC + 2025 data merged under label '2025'):
    python project_nominal.py pt_reweighting_2025/output_all.coffea --merge-years 2024 2025 --merged-label 2025
"""
import sys
import os
import argparse
import hist
from coffea.util import load, save


def keep_nominal_only(h):
    """Rebuild histogram keeping only 'nominal' on the variation axis."""
    ax_names = [ax.name for ax in h.axes]
    if 'variation' not in ax_names:
        return h

    # Slice to nominal — this removes the variation axis
    h_nom = h[{'variation': 'nominal'}]

    # Rebuild with a variation axis containing only 'nominal'
    var_ax_idx = ax_names.index('variation')
    new_axes = []
    for i, ax in enumerate(h.axes):
        if i == var_ax_idx:
            new_axes.append(hist.axis.StrCategory(['nominal'], name='variation', growth=False))
        else:
            new_axes.append(ax)

    h_new = hist.Hist(*new_axes, storage=h.storage_type())
    # Use integer indexing: bin 0 = 'nominal' (the only bin)
    idx = [slice(None)] * h_new.ndim
    idx[var_ax_idx] = 0
    h_new.view()[tuple(idx)] = h_nom.view()
    return h_new


def merge_year_buckets(output, years_to_merge, merged_label):
    """Merge entries in datasets_metadata['by_datataking_period'] for the mixed-year case.
    For 2025 SFs, MC has year='2024' and data has year='2025'. PocketCoffea groups plots
    by datataking_period so they end up in separate plots. This merges the buckets so
    they appear together.
    """
    metadata = output.get('datasets_metadata', {})
    period_dict = metadata.get('by_datataking_period', {})
    missing = [y for y in years_to_merge if y not in period_dict]
    if missing:
        print(f"  WARNING: years not found in by_datataking_period: {missing}")
        print(f"  Available: {list(period_dict.keys())}")
        return

    # Build merged sample dict: {sample_name: set_of_datasets}
    merged = {}
    for y in years_to_merge:
        for sample_name, datasets in period_dict[y].items():
            if sample_name not in merged:
                merged[sample_name] = set()
            # datasets may be a list, set, or dict — normalize to iterable of names
            if isinstance(datasets, dict):
                merged[sample_name].update(datasets.keys())
            else:
                merged[sample_name].update(datasets)

    # Convert sets back to original container type (list)
    merged_final = {s: sorted(ds) for s, ds in merged.items()}

    # Remove the original year keys, add merged
    new_period = {k: v for k, v in period_dict.items() if k not in years_to_merge}
    new_period[merged_label] = merged_final
    metadata['by_datataking_period'] = new_period
    print(f"  Merged years {years_to_merge} -> '{merged_label}': "
          f"{len(merged_final)} samples, "
          f"{sum(len(ds) for ds in merged_final.values())} datasets total")


def main():
    parser = argparse.ArgumentParser(description="Project to nominal variation; optionally merge year buckets.")
    parser.add_argument('infile', help="Input coffea file")
    parser.add_argument('--merge-years', nargs='+', default=None,
                        help="List of year labels to merge (e.g. 2024 2025).")
    parser.add_argument('--merged-label', default=None,
                        help="Label for the merged year (default: first of --merge-years).")
    args = parser.parse_args()

    infile = args.infile
    dirname = os.path.dirname(infile)
    basename = os.path.basename(infile).replace('.coffea', '_nominal.coffea')
    outfile = os.path.join(dirname, basename) if dirname else basename

    print(f"Loading {infile}...")
    output = load(infile)

    if args.merge_years:
        merged_label = args.merged_label or args.merge_years[0]
        print(f"\nMerging year buckets {args.merge_years} -> '{merged_label}'...")
        merge_year_buckets(output, args.merge_years, merged_label)

    # Debug: inspect structure of first variable
    first_var = list(output['variables'].keys())[0]
    print(f"\n=== Debug: structure of '{first_var}' ===")
    var_dict = output['variables'][first_var]
    print(f"  Type: {type(var_dict)}")
    if isinstance(var_dict, dict):
        for k, v in list(var_dict.items())[:3]:
            print(f"  Key: {k}, Type: {type(v)}")
            if isinstance(v, dict):
                for k2, v2 in list(v.items())[:2]:
                    print(f"    Key: {k2}, Type: {type(v2)}")
                    if hasattr(v2, 'axes'):
                        print(f"    Axes: {[(ax.name, type(ax).__name__, len(ax)) for ax in v2.axes]}")
                        if 'variation' in [ax.name for ax in v2.axes]:
                            var_ax = [ax for ax in v2.axes if ax.name == 'variation'][0]
                            print(f"    Variation entries: {list(var_ax)}")
            elif hasattr(v, 'axes'):
                print(f"    Axes: {[(ax.name, type(ax).__name__, len(ax)) for ax in v.axes]}")
                if 'variation' in [ax.name for ax in v.axes]:
                    var_ax = [ax for ax in v.axes if ax.name == 'variation'][0]
                    print(f"    Variation entries: {list(var_ax)}")
    elif hasattr(var_dict, 'axes'):
        print(f"  Axes: {[(ax.name, type(ax).__name__, len(ax)) for ax in var_dict.axes]}")
    print("=== End debug ===\n")

    print("Harmonizing variation axes to nominal-only...")
    n_projected = 0
    n_skipped = 0
    for varname, var_dict in output['variables'].items():
        if isinstance(var_dict, dict):
            for sample_name, sample_dict in var_dict.items():
                if isinstance(sample_dict, dict):
                    for dataset_name, h in sample_dict.items():
                        if not hasattr(h, 'axes'):
                            continue
                        ax_names = [ax.name for ax in h.axes]
                        if 'variation' in ax_names:
                            try:
                                output['variables'][varname][sample_name][dataset_name] = keep_nominal_only(h)
                                n_projected += 1
                            except Exception as e:
                                print(f"  WARNING: failed for {sample_name}/{dataset_name}/{varname}: {e}")
                        else:
                            n_skipped += 1
                elif hasattr(sample_dict, 'axes'):
                    ax_names = [ax.name for ax in sample_dict.axes]
                    if 'variation' in ax_names:
                        try:
                            output['variables'][varname][sample_name] = keep_nominal_only(sample_dict)
                            n_projected += 1
                        except Exception as e:
                            print(f"  WARNING: failed for {sample_name}/{varname}: {e}")
                    else:
                        n_skipped += 1

    print(f"Projected {n_projected} histograms to nominal-only.")
    print(f"Saving to {outfile}...")
    save(output, outfile)
    print("Done.")


if __name__ == '__main__':
    main()