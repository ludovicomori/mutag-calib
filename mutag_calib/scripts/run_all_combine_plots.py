#!/usr/bin/env python3
import os
import subprocess
import argparse
import re

ALLOWED_CATEGORIES = {
    "msd-80to170_Pt-300to350_particleNet_XbbVsQCD-HHbbtt",
    "msd-80to170_Pt-350to425_particleNet_XbbVsQCD-HHbbtt",
    "msd-80to170_Pt-425toInf_particleNet_XbbVsQCD-HHbbtt",
    "msd-30toInf_Pt-300to350_particleNet_XbbVsQCD-HHbbgg",
    "msd-30toInf_Pt-350to425_particleNet_XbbVsQCD-HHbbgg",
    "msd-30toInf_Pt-425toInf_particleNet_XbbVsQCD-HHbbgg",
    "msd-30toInf_Pt-300to350_globalParT3_XbbVsQCD-HHbbgg",
    "msd-30toInf_Pt-350to425_globalParT3_XbbVsQCD-HHbbgg",
    "msd-30toInf_Pt-425toInf_globalParT3_XbbVsQCD-HHbbgg",
}

FIT_RE = re.compile(r"fitDiagnostics\.(.+)\.root")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="Base directory containing datacards")
    parser.add_argument("-o", "--output-dir", dest="output_dir", help="Output directory for plots", default=None)
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MAKE_PLOTS = os.path.join(SCRIPT_DIR, "make_combine_plots.py")

    for year in sorted(os.listdir(BASE_DIR)):
        year_dir = os.path.join(BASE_DIR, year)
        if not os.path.isdir(year_dir):
            continue

        for category in sorted(os.listdir(year_dir)):
            if category not in ALLOWED_CATEGORIES:
                continue

            cat_dir = os.path.join(year_dir, category)
            if not os.path.isdir(cat_dir):
                continue

            for tau21 in sorted(os.listdir(cat_dir)):
                tau_dir = os.path.join(cat_dir, tau21)
                if not os.path.isdir(tau_dir):
                    continue

                fit_file = None
                tag = None
                for f in os.listdir(tau_dir):
                    m = FIT_RE.match(f)
                    if m:
                        fit_file = f
                        tag = m.group(1)
                        break

                if fit_file is None:
                    continue

                print(f"\n Processing {tau_dir}")

                # costruzione pass/fail channel
                base = tag.replace("-", "_")
                passch = f"{base}_pass_{year}"
                failch = f"{base}_fail_{year}"

                outdir_base = args.output_dir
                outdir = os.path.join(outdir_base, year, category, tau21)

                cmd = [
                    "python3", MAKE_PLOTS,
                    "--file", fit_file,
                    "--passch", passch,
                    "--failch", failch,
                    "--outdir", outdir
                ]

                print("  ", " ".join(cmd))
                subprocess.run(cmd, cwd=tau_dir, check=True)
                

if __name__ == "__main__":
    main()
