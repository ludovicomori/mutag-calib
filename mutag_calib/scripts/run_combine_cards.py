#!/usr/bin/env python3

import os
import subprocess
import argparse

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="Base directory containing datacards")
    args = parser.parse_args()

    BASE_DIR = args.base_dir

    for year in sorted(os.listdir(BASE_DIR)):
        year_path = os.path.join(BASE_DIR, year)
        if not os.path.isdir(year_path):
            continue

        print(f"\n=== Year/Era: {year} ===")

        for category in sorted(os.listdir(year_path)):
            if category not in ALLOWED_CATEGORIES:
                continue

            category_path = os.path.join(year_path, category)
            if not os.path.isdir(category_path):
                continue

            print(f"\n  -> Category: {category}")

            for cut in sorted(os.listdir(category_path)):
                cut_path = os.path.join(category_path, cut)
                if not os.path.isdir(cut_path):
                    continue

                script_path = os.path.join(cut_path, "combine_cards.sh")
                if not os.path.isfile(script_path):
                    print(f"     [SKIP] {cut}: combine_cards.sh not found")
                    continue

                print(f"     [RUN] {year}/{category}/{cut}")
                subprocess.run(
                    ["bash", "combine_cards.sh"],
                    cwd=cut_path,
                    check=True
                )


if __name__ == "__main__":
    main()
