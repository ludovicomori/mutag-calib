"""Merge per-job coffea files, then run PocketCoffea postprocess (metadata + MC rescaling).

Condor batch jobs use partial configs with do_postprocessing=False, so each output_job_*.coffea
has unscaled MC histograms (large integrals). Official merge-outputs loads the full
configurator and postprocess() applies rescale_sumgenweights once — that is required here too.

If you used dask@lxplus --process-separately with the normal config, each shard was already
postprocessed and rescaling again would double-divide: pass --donot-scale in that case.

Usage:
    python merge_no_postprocess.py output_all.coffea path/to/output_job_*.coffea
    python merge_no_postprocess.py output_all.coffea path/to/*.coffea --donot-scale
"""
import sys
import os
import cloudpickle
from coffea.util import load, save
from coffea.processor import accumulate


def main():
    argv = list(sys.argv)
    donotscale = "--donot-scale" in argv
    if donotscale:
        argv.remove("--donot-scale")

    if len(argv) < 3:
        print("Usage: python merge_no_postprocess.py <output.coffea> <input1.coffea> [...] [--donot-scale]")
        sys.exit(1)

    outfile = argv[1]
    infiles = argv[2:]

    # Exclude the output file and any previous merged files from inputs (in case of glob overlap)
    infiles = [f for f in infiles if os.path.abspath(f) != os.path.abspath(outfile)]
    infiles = [f for f in infiles if not os.path.basename(f).startswith('output_all')]

    print(f"Merging {len(infiles)} files into {outfile}")

    merged = None
    for f in sorted(infiles):
        print(f"  Loading {os.path.basename(f)}...")
        chunk = load(f)
        if merged is None:
            merged = chunk
        else:
            merged = accumulate([merged, chunk])
        del chunk

    if merged is None:
        print("No input files to merge.")
        sys.exit(1)

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(sorted(infiles)[0])), "configurator.pkl")
    if not os.path.isfile(cfg_path):
        print(f"Missing {cfg_path}; need configurator.pkl next to inputs for postprocess (make-plots metadata).")
        sys.exit(1)
    with open(cfg_path, "rb") as f:
        configurator = cloudpickle.load(f)
    processor = configurator.processor_instance
    wo = processor.workflow_options
    if wo is None:
        processor.cfg.workflow_options = {}
        processor.workflow_options = processor.cfg.workflow_options
        wo = processor.workflow_options
    wo["donotscale_sumgenweights"] = donotscale
    if donotscale:
        print("Postprocess: metadata + skip sum_genweights rescale (--donot-scale)")
    else:
        print("Postprocess: metadata + rescale_sumgenweights (default; needed for condor shards)")
    merged = processor.postprocess(merged)

    print(f"Saving to {outfile}...")
    save(merged, outfile)
    print("Done.")


if __name__ == '__main__':
    main()
