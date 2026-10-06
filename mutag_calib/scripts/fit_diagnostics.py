#!/usr/bin/env python3

import argparse
import os
import subprocess
import json
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--split-c", action="store_true",
                    help="datacards have SF_c split into SF_c_pass (frozen to 1) and SF_c_fail (floating)")
parser.add_argument("--c-freeze", action="store_true",
                    help="if true, feeze all SFc factors")
args = parser.parse_args()

if args.split_c:
    pois = "SF_b,SF_c_fail,SF_c_pass,SF_light"
    set_params = "r=1,SF_light=1,SF_c_pass=1"
    frozen = "r,SF_light,SF_c_pass"
    if args.c_freeze:
        set_params = "r=1,SF_light=1,SF_c_pass=1,SF_fail=1"
        frozen = "r,SF_light,SF_c_pass,SF_c_fail"
else:
    pois = "SF_b,SF_c,SF_light"
    set_params = "r=1,SF_light=1"
    frozen = "r,SF_light"
    if args.c_freeze:
        set_params = "r=1,SF_light=1,SF_c=1"
        frozen = "r,SF_light,SF_c"

# sanity checks
if not os.path.isfile("workspace.root"):
    raise RuntimeError("workspace.root not found in current directory")

# category name
category = os.path.basename(os.path.dirname(os.getcwd()))

fit_name = f".{category}"

cmd = [
    "combine", "-M", "FitDiagnostics",
    "-d", "workspace.root",
    # "--saveFitResult",
    "--name", fit_name,
    "--cminDefaultMinimizerStrategy", "0",
    # "--robustFit", "1",
    "--skipBOnlyFit",
    "--saveWorkspace",
    "--saveShapes",
    "--saveWithUncertainties",
    "--saveOverallShapes",
    "--redefineSignalPOIs", pois,
    "--setParameters", set_params,
    "--freezeParameters", frozen,
    "--ignoreCovWarning",
    # "--robustHesse", "1",
    # "--stepSize", "0.001",
    # "--X-rtd", "MINIMIZER_analytic",
    # "--X-rtd", "MINIMIZER_MaxCalls=9999999",
    # fallbacks only used if the default fit fails (e.g. Migrad strategy 0 overshooting on the first step)
    "--cminFallbackAlgo", "Minuit2,Simplex,0:0.1",
    "--cminFallbackAlgo", "Minuit2,Migrad,1:0.1",
    # "--X-rtd", "FITTER_NEW_CROSSING_ALGO",
    # "--X-rtd", "FITTER_NEVER_GIVE_UP",
    # "--X-rtd", "FITTER_BOUND",
]

logfile = f"fitDiagnostics{fit_name}.log"

print(f"[INFO] Running FitDiagnostics for category: {category}")
print(f"[INFO] Log file: {logfile}")

with open(logfile, "w") as log:
    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

# save basic status
status = {
    "category": category,
    "cwd": os.getcwd(),
    "command": " ".join(cmd),
    "returncode": result.returncode,
    "timestamp": datetime.now().isoformat(),
    "fit_root_file": f"fitDiagnostics{fit_name}.root",
    "combine_output": f"higgsCombine{fit_name}.FitDiagnostics.mH120.root",
}

with open("fit_status.json", "w") as f:
    json.dump(status, f, indent=2)

if result.returncode != 0:
    print("[ERROR] FitDiagnostics FAILED")
else:
    print("[OK] FitDiagnostics completed successfully")
