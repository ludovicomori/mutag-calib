#!/usr/bin/env python3

import os
import subprocess
import json
from datetime import datetime

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
    "--cminDefaultMinimizerStrategy", "1",
    # "--robustFit", "1",
    "--saveWorkspace",
    "--saveShapes",
    "--saveWithUncertainties",
    "--saveOverallShapes",
    "--redefineSignalPOIs", "r,SF_c,SF_light",
    "--setParameters", "SF_light=1",
    "--freezeParameters", "SF_light",
    "--rMin", "0",
    "--rMax", "10",
    # "--robustHesse", "1",
    # "--stepSize", "0.001",
    # "--X-rtd", "MINIMIZER_analytic",
    # "--X-rtd", "MINIMIZER_MaxCalls=9999999",
    # "--cminFallbackAlgo", "Minuit2,Migrad,0:0.2",
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
