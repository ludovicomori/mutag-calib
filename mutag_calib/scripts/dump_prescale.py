import json, os, argparse, pandas
import correctionlib.schemav2 as cs
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

try:
    from BTVNanoCommissioning.helpers.BTA_helper import BTA_HLT
except ImportError:
    BTA_HLT = None

# Resolve default lumimask from PocketCoffea's bundled datacert files
_POCKETCOFFEA_DATACERT = None
try:
    from pocket_coffea.parameters import defaults as _pcf_defaults
    _POCKETCOFFEA_DATACERT = os.path.join(os.path.dirname(_pcf_defaults.__file__), "datacert")
except ImportError:
    pass

_DEFAULT_LUMIMASK = (
    os.path.join(_POCKETCOFFEA_DATACERT, "Cert_Collisions2022_355100_362760_Golden.json")
    if _POCKETCOFFEA_DATACERT else None
)

### NOTICE The scripts only works on lxplus...

parser = argparse.ArgumentParser(description="Create prescale weights(lxplus)")

parser.add_argument(
    "-l",
    "--lumimask",
    default=_DEFAULT_LUMIMASK,
    required=_DEFAULT_LUMIMASK is None,
    help="lumimask to generate prescale weights (default: PocketCoffea's 2022 golden JSON)",
)
parser.add_argument("-H", "--HLT", default=None, type=str, help="Which HLT is used")
parser.add_argument("-v", "--verbose", action="store_true", help="debugging")
parser.add_argument("-t", "--test", action="store_true", help="test with only 5 runs")
parser.add_argument("-f", "--force", action="store_true", help="recreate .csv")
parser.add_argument(
    "-o",
    "--output-dir",
    default="src/BTVNanoCommissioning/data/Prescales",
    help="output directory for prescale CSV and JSON files",
)

### NOTICE The scripts only works on lxplus...


def process_run(ir_run):
    ir, run = ir_run
    tmpfile = f".tmp_{ir}.csv"
    os.system(
        f"singularity -s exec --env PYTHONPATH=/home/bril/.local/lib/python3.10/site-packages "
        f"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cloud/brilws-docker:latest "
        f"brilcalc trg --prescale --hltpath 'HLT_{HLT}_v*' -r {run} --output-style csv &>{tmpfile}"
    )
    return pandas.read_csv(tmpfile)


def get_prescale(HLT, lumimask, verbose=False, test=False, force=False, outdir="src/BTVNanoCommissioning/data/Prescales"):
    # os.system("source /cvmfs/cms-bril.cern.ch/cms-lumi-pog/brilws-docker/brilws-env")
    prescales = pandas.DataFrame()
    runs = json.load(open(lumimask))
    runs = list(runs.keys())
    if test:
        runs = runs[: min(len(runs), 5)]

    os.makedirs(outdir, exist_ok=True)
    outcsv = f"{outdir}/HLTinfo_{HLT}_run{runs[0]}_{runs[-1]}.csv"
    if force or not os.path.exists(outcsv):
        with ThreadPoolExecutor() as executor:
            dfs = list(
                tqdm(executor.map(process_run, enumerate(runs)), total=len(runs))
            )

        prescales = pandas.concat(dfs, ignore_index=True)
        # Filter out brilcalc error rows (e.g. "No hltpathl1seed mapping found")
        prescales = prescales[prescales["# run"].apply(lambda x: str(x).isdigit())]
        prescales["# run"] = prescales["# run"].astype(int)
        # Filter out disabled triggers (prescale=0)
        prescales = prescales[prescales["totprescval"] != 0]
        prescales.to_csv(outcsv)
        os.system(f"rm -rf .tmp_*.csv")

        if verbose:
            print("prescales :", prescales)
    else:
        prescales = pandas.read_csv(outcsv)
    return prescales


### code from Hsin-Wei: https://github.com/cms-btv-pog/BTVNanoCommissioning/blob/f2a5db0e325c9b26d220089a49ddb8f73682f846/prescales.ipynb
## read prescale


def get_ps(ps, verbose=False):
    if len(ps) != 1:
        print(ps)
        print("Length of ps after selection ", len(ps))
        raise ValueError(ps)
    if verbose:
        print("Final prescale weight: ", ps.iloc[0]["totprescval"])
    return float(ps.iloc[0]["totprescval"])


def build_lumibins(ps, verbose=False):
    ##### to sort as bin edges properly, starting lumi sections need to be stored as floats
    if verbose:
        print("Path: ", ps["hltpath/prescval"], ps["totprescval"])
    edges = sorted(set(ps["cmsls"].astype(float)))
    if len(edges) == 1:
        return get_ps(ps)
    elif len(edges) > 1:
        edges.append("inf")
        if verbose:
            print("Lumi bin edges: ", list(zip(edges[:-1], edges[1:])))
        content = [
            get_ps(
                ps[
                    (ps["cmsls"].astype(float) >= lo)
                    & (ps["cmsls"].astype(float) < float(hi))
                ]
            )
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
        if verbose:
            print("Prescales: ", content)
        return cs.Binning.parse_obj(
            {
                "nodetype": "binning",
                "input": "lumi",
                "edges": edges,
                "content": content,
                "flow": "clamp",
            }
        )


def build_runs(ps, HLT_paths, verbose=False):
    runs = sorted(ps["# run"].unique())
    if verbose:
        print("Selected ", len(runs), ": ", runs)
    return cs.Category.parse_obj(
        {
            "nodetype": "category",
            "input": "run",
            "content": [
                {
                    "key": int(run),
                    "value": build_paths(ps[ps["# run"] == run], HLT_paths, verbose),
                }
                for run in runs
            ],
        }
    )


def build_paths(ps, HLT_paths, verbose=False):
    if verbose:
        print("Run: ", ps["# run"].iloc[0], type(ps["# run"].iloc[0]))
    # paths are unique bc of hltpath/lumi --> make array of path name separate
    paths = [HLT_paths]
    if verbose:
        print("Type of path key: ", type(paths[0]), paths)
    return cs.Category.parse_obj(
        {
            "nodetype": "category",
            "input": "path",
            "content": [
                {"key": str(path), "value": build_lumibins(ps, verbose)}
                for path in paths
            ],
        }
    )


if __name__ == "__main__":
    args = parser.parse_args()
    if args.HLT is None:
        if BTA_HLT is None:
            raise RuntimeError(
                "No --HLT specified and BTVNanoCommissioning not installed. "
                "Please pass --HLT 'trigger1,trigger2,...' explicitly."
            )
        args.HLT = BTA_HLT
    else:
        if "," in args.HLT:
            args.HLT = args.HLT.split(",")
        else:
            args.HLT = [args.HLT]

    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    os.system(
        "source /cvmfs/cms-bril.cern.ch/cms-lumi-pog/brilws-docker/brilws-env; which brilcalc"
    )

    for HLT in args.HLT:
        print("HLT : ", HLT)
        ps_csvData = get_prescale(
            HLT, args.lumimask, args.verbose, args.test, args.force, outdir
        )
        psCorr = cs.Correction.parse_obj(
            {
                "version": 2,
                "name": "prescaleWeight",
                "inputs": [
                    {"name": "run", "type": "int"},
                    {"name": "path", "type": "string"},
                    {"name": "lumi", "type": "real"},
                ],
                "output": {"name": "weight", "type": "real"},
                "data": build_runs(ps_csvData, "HLT_" + HLT, args.verbose),
            }
        )
        cset = cs.CorrectionSet(
            schema_version=2,
            corrections=[psCorr],
            description=f"prescales for HLT_{HLT}",
        )
        runs = json.load(open(args.lumimask))

        runs = list(runs.keys())
        with open(
            f"{outdir}/ps_weight_{HLT}_run{runs[0]}_{runs[-1]}.json",
            "w",
        ) as f:
            f.write(cset.json(exclude_unset=True))
