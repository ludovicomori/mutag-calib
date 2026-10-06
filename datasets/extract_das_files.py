#!/usr/bin/env python3
import json
import glob
from pathlib import Path
from typing import Any, List


def extract_das_names(obj: Any, out: List[str]) -> None:
    """
    Recursively walk a nested JSON-like structure (dict/list)
    and collect strings from any key named 'das_names' whose value is a list.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "das_names" and isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        out.append(item)
            else:
                extract_das_names(v, out)
    elif isinstance(obj, list):
        for item in obj:
            extract_das_names(item, out)


def main() -> None:
    # Change these if you want
    input_dir = Path(".")          # directory to search in
    pattern = "datasets_definitions_VJets*"  # file prefix/pattern
    output_file = Path("das_names_vjets.txt")

    files = sorted(glob.glob(f"{input_dir}/{pattern}"))
    if not files:
        raise SystemExit(f"No files found matching: {input_dir / pattern}")

    all_names: List[str] = []
    bad_files: List[str] = []

    for fn in files:
        path = Path(fn)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            bad_files.append(f"{fn}: {e}")
            continue

        extract_das_names(data, all_names)

    # Optional: deduplicate + stable sort
    unique_names = sorted(set(all_names))

    output_file.write_text("\n".join(unique_names) + ("\n" if unique_names else ""), encoding="utf-8")

    print(f"Scanned {len(files)} files")
    print(f"Extracted {len(all_names)} das_names entries ({len(unique_names)} unique)")
    print(f"Wrote: {output_file}")

    if bad_files:
        print("\nWARNING: Some files could not be parsed as JSON:")
        for msg in bad_files:
            print("  -", msg)


if __name__ == "__main__":
    main()
