#!/usr/bin/env python3

import os
import ROOT
import pandas as pd
import json
import glob

ROOT.gROOT.SetBatch(True)

POIS = ["r", "SF_c", "SF_light"]

# find fitDiagnostics file
fit_files = glob.glob("fitDiagnostics.*.root")
if len(fit_files) != 1:
    raise RuntimeError(f"Found {len(fit_files)} fitDiagnostics files: {fit_files}")

fit_file = fit_files[0]

# infer metadata
cut = os.path.basename(os.getcwd())
category = os.path.basename(os.path.dirname(os.getcwd()))
year = os.path.basename(os.path.dirname(os.path.dirname(os.getcwd())))

# open ROOT file
f = ROOT.TFile.Open(fit_file)
fit_s = f.Get("fit_s")
if not fit_s:
    raise RuntimeError("fit_s not found in ROOT file")

pars = fit_s.floatParsFinal()

row = {
    "year": year,
    "category": category,
    "cut": cut,
}

# extract POIs
for poi in POIS:
    par = pars.find(poi)
    if not par:
        print(f"[WARN] POI {poi} not found")
        continue
    row[poi] = par.getVal()
    row[f"{poi}_errUp"] = par.getErrorHi()
    row[f"{poi}_errDown"] = abs(par.getErrorLo())

# save outputs
df = pd.DataFrame([row])
df.to_csv("fitResults.csv", index=False)

with open("fitResults.json", "w") as fjson:
    json.dump(row, fjson, indent=2)

print("[OK] Results saved in fitResults.csv and fitResults.json")

