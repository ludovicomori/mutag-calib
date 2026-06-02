#!/usr/bin/env python3
"""Remove ROOT file URLs from a PocketCoffea pickled Configurator (e.g. config_job_N.pkl).

These pickles embed lambdas (skims/cuts) and workflow refs; **stdlib pickle cannot round-trip**
them (`Can't pickle ... lambda`). Use **cloudpickle** on write (default), matching PocketCoffea.

Usage:
  export PYTHONPATH="/afs/cern.ch/user/l/lumori/mutag/mutag-calib:${PYTHONPATH}"
  python remove_bad_files_from_pkl.py /path/to/config_job_196.pkl --bad SUBSTRING
  python remove_bad_files_from_pkl.py config.pkl --bad BAD1 --bad BAD2 --dry-run

Optional ``--stdlib-pickle`` tries canonicalizing globals then ``pickle.dump`` (fragile).
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import sys
import types

try:
    from omegaconf import DictConfig as _OmegaDictConfig
    from omegaconf import ListConfig as _OmegaListConfig
except ImportError:
    _OmegaDictConfig = None  # type: ignore[misc, assignment]
    _OmegaListConfig = None  # type: ignore[misc, assignment]


def _get_files_list(spec):
    """Return mutable list backing this fileset entry, or None."""
    if isinstance(spec, dict) and "files" in spec:
        files = spec["files"]
        return files if isinstance(files, list) else None
    files = getattr(spec, "files", None)
    return files if isinstance(files, list) else None


def _set_files_list(spec, new_files: list) -> None:
    if isinstance(spec, dict):
        spec["files"] = new_files
    else:
        setattr(spec, "files", new_files)


def _resolve_module_singleton(obj: object) -> object:
    """Map a class or module-level function to the object in ``sys.modules`` (same as ``import``).

    ``pickle`` stores globals by reference; unpickling can leave *different* class/function
    objects than ``import`` would return, and ``pickle.dump`` then fails with
    "it's not the same object as …". Resolving by ``module`` + ``qualname`` fixes that.
    """
    if isinstance(obj, type):
        modn, qn = obj.__module__, obj.__qualname__
    elif isinstance(obj, types.FunctionType):
        modn = getattr(obj, "__module__", None) or ""
        qn = getattr(obj, "__qualname__", None) or ""
        if "<locals>" in qn or "lambda" in qn.lower():
            return obj
    else:
        return obj

    if not modn or modn == "builtins" or not qn:
        return obj
    try:
        root = importlib.import_module(modn)
        cur: object = root
        for part in qn.split("."):
            cur = getattr(cur, part)
        return cur
    except Exception:
        return obj


def _canonicalize_graph_for_dump(obj: object, memo: dict[int, object]) -> object:
    """Walk containers and rebind module-backed types/functions to canonical singletons.

    Uses *memo* so duplicate references to the same broken global all map to one canonical
    object (``seen``-only was wrong: second sighting returned the stale handle again).
    """
    oid = id(obj)
    if oid in memo:
        return memo[oid]

    canon = _resolve_module_singleton(obj)
    if canon is not obj:
        memo[oid] = canon
        return canon

    # OmegaConf: traverse only via indexed assignment — never vars()/setattr on containers.
    if _OmegaDictConfig is not None and isinstance(obj, _OmegaDictConfig):
        memo[oid] = obj
        for k in list(obj.keys()):
            try:
                obj[k] = _canonicalize_graph_for_dump(obj[k], memo)
            except Exception:
                pass
        return obj
    if _OmegaListConfig is not None and isinstance(obj, _OmegaListConfig):
        memo[oid] = obj
        for i in range(len(obj)):
            try:
                obj[i] = _canonicalize_graph_for_dump(obj[i], memo)
            except Exception:
                pass
        return obj

    if isinstance(obj, dict):
        memo[oid] = obj
        for k in list(obj.keys()):
            nk = _canonicalize_graph_for_dump(k, memo)
            nv = _canonicalize_graph_for_dump(obj[k], memo)
            if nk is not k:
                del obj[k]
                obj[nk] = nv
            else:
                obj[k] = nv
        return obj
    if isinstance(obj, list):
        memo[oid] = obj
        for i in range(len(obj)):
            obj[i] = _canonicalize_graph_for_dump(obj[i], memo)
        return obj
    if isinstance(obj, tuple):
        res = tuple(_canonicalize_graph_for_dump(x, memo) for x in obj)
        memo[oid] = res
        return res
    if isinstance(obj, (str, int, float, bool, bytes)) or obj is None:
        memo[oid] = obj
        return obj

    # User objects (Configurator, pocket_coffea objects, …)
    memo[oid] = obj
    if hasattr(obj, "__dict__"):
        try:
            for attr, val in list(vars(obj).items()):
                setattr(obj, attr, _canonicalize_graph_for_dump(val, memo))
        except (TypeError, AttributeError):
            pass
    return obj


def strip_bad_urls(filesets: dict, bad_substrings: list[str], dry_run: bool) -> dict[str, int]:
    """Returns mapping sample_name -> number of URLs removed."""
    removed_per_sample: dict[str, int] = {}

    for sample_name, spec in list(filesets.items()):
        files = _get_files_list(spec)
        if files is None:
            continue

        keep = []
        removed = 0
        for url in files:
            s = str(url)
            if any(bad in s for bad in bad_substrings):
                removed += 1
                continue
            keep.append(url)

        if removed:
            removed_per_sample[str(sample_name)] = removed
            if not dry_run:
                _set_files_list(spec, keep)

    return removed_per_sample


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkl", help="Path to pickled Configurator")
    ap.add_argument(
        "--bad",
        action="append",
        dest="bad",
        default=[],
        help="Substring; any file URL containing it is dropped (repeatable)",
    )
    ap.add_argument(
        "--stdlib-pickle",
        action="store_true",
        help=(
            "Use stdlib pickle after canonicalizing module globals (often breaks on lambdas / OmegaConf). "
            "Default is cloudpickle, like PocketCoffea job pickles."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed but do not write the pickle",
    )
    args = ap.parse_args()

    if not args.bad:
        print("error: pass at least one --bad SUBSTRING", file=sys.stderr)
        return 2

    with open(args.pkl, "rb") as f:
        cfg = pickle.load(f)

    filesets = getattr(cfg, "filesets", None)
    if not isinstance(filesets, dict):
        print(f"error: expected cfg.filesets to be dict, got {type(filesets)}", file=sys.stderr)
        return 1

    removed = strip_bad_urls(filesets, args.bad, dry_run=args.dry_run)

    if not removed:
        print("No matching URLs found; nothing to do.")
        return 0

    print("Removed URLs (per dataset key in cfg.filesets):")
    for k, v in sorted(removed.items()):
        print(f"  {k}: {v}")

    if args.dry_run:
        print("--dry-run: not writing pickle")
        return 0

    out_path = args.pkl
    tmp_path = args.pkl + ".tmp"

    if args.stdlib_pickle:
        _canonicalize_graph_for_dump(cfg, {})
        with open(tmp_path, "wb") as f:
            pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)
        backend = "stdlib pickle"
    else:
        try:
            import cloudpickle
        except ImportError:
            print(
                "error: cloudpickle is required for default save (configs contain lambdas). "
                "Install cloudpickle or pass --stdlib-pickle (fragile).",
                file=sys.stderr,
            )
            return 1
        with open(tmp_path, "wb") as f:
            cloudpickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(tmp_path, "rb") as f:
            cloudpickle.load(f)
        backend = "cloudpickle"

    # atomic replace avoids leaving a truncated .pkl on crash mid-write
    import os

    os.replace(tmp_path, out_path)
    print(f"Wrote updated pickle ({backend}) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())