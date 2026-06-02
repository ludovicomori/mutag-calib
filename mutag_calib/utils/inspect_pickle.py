#!/usr/bin/env python3
"""Inspect a PocketCoffea pickled ``Configurator`` (e.g. ``config_job_N.pkl``).

Walks ``cfg.filesets`` (where chunk URLs actually live) and prints file counts.
Optionally asserts that given substrings never appear in any ROOT URL.

Usage:
  export PYTHONPATH="/afs/cern.ch/user/l/lumori/mutag/mutag-calib:${PYTHONPATH}"
  python inspect_pickle.py pt_reweighting/job/config_job_196.pkl
  python inspect_pickle.py pt_reweighting/job/config_job_196.pkl \\
      --must-not-contain a7d8632e-0a31-4d72-9d55-65df40527939
"""

from __future__ import annotations

import argparse
import pickle
import sys

try:
    import cloudpickle
except ImportError:
    cloudpickle = None


def _get_files_list(spec: object):
    """Same convention as ``remove_bad_files_from_pkl``."""
    if isinstance(spec, dict) and "files" in spec:
        files = spec["files"]
        return files if isinstance(files, list) else None
    files = getattr(spec, "files", None)
    return files if isinstance(files, list) else None


def _load_cfg(path: str):
    with open(path, "rb") as f:
        raw = f.read()
    print("file bytes:", len(raw))
    try:
        return pickle.loads(raw)
    except Exception as e_pick:
        if cloudpickle is None:
            print(f"pickle.loads failed ({e_pick}); install cloudpickle and retry.", file=sys.stderr)
            raise
        try:
            return cloudpickle.loads(raw)
        except Exception as e_cloud:
            print(f"pickle.loads failed: {e_pick}", file=sys.stderr)
            print(f"cloudpickle.loads failed: {e_cloud}", file=sys.stderr)
            raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "pkl",
        nargs="?",
        default="/afs/cern.ch/user/l/lumori/mutag/mutag-calib/pt_reweighting/job/config_job_196.pkl",
        help="Path to pickled Configurator",
    )
    ap.add_argument(
        "--must-not-contain",
        action="append",
        dest="forbidden",
        default=[],
        help="Substring that must not appear in any file URL (repeatable); exit 1 if found",
    )
    args = ap.parse_args()

    cfg = _load_cfg(args.pkl)

    print("top type:", type(cfg))
    print("top repr:", repr(cfg)[:500])

    filesets = getattr(cfg, "filesets", None)
    if not isinstance(filesets, dict):
        print(f"error: expected cfg.filesets dict, got {type(filesets)}", file=sys.stderr)
        return 1

    print("\n--- cfg.filesets ---")
    total_files = 0
    all_urls_sample: list[tuple[str, str]] = []

    for sample_name in sorted(filesets.keys()):
        spec = filesets[sample_name]
        files = _get_files_list(spec)
        if files is None:
            print(f"  {sample_name}: <no files list> type(spec)={type(spec).__name__}")
            continue
        n = len(files)
        total_files += n
        first = str(files[0])[:140] + ("…" if len(str(files[0])) > 140 else "") if files else "<empty>"
        print(f"  {sample_name}: n_files={n}  first={first!r}")
        for u in files:
            all_urls_sample.append((sample_name, str(u)))

    print(f"\nTotal ROOT URLs across filesets: {total_files}")

    if args.forbidden:
        print("\n--- forbidden substring check ---")
        failed = False
        for sub in args.forbidden:
            hits = [(s, url) for s, url in all_urls_sample if sub in url]
            if hits:
                failed = True
                print(f"FAIL: found {len(hits)} URL(s) containing {sub!r}")
                for s, url in hits[:25]:
                    print(f"  [{s}] {url}")
                if len(hits) > 25:
                    print(f"  ... and {len(hits) - 25} more")
            else:
                print(f"OK: no URL contains {sub!r}")
        return 1 if failed else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
