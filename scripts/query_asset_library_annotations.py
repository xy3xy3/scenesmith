#!/usr/bin/env python3
"""Query generated asset-library annotations by HSSD id without heavy deps."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scenesmith"
    / "scenebenchmark_critic"
    / "asset_library_annotations.py"
)


def _load_lookup_module():
    spec = importlib.util.spec_from_file_location("asset_library_annotations", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hssd_id", help="40-character HSSD id, optionally prefixed with hssd:")
    ap.add_argument("--lookup", type=Path, default=None)
    ap.add_argument(
        "--self-emission-only",
        action="store_true",
        help="return only the lightweight self_emission annotation",
    )
    args = ap.parse_args()

    module = _load_lookup_module()
    store = module.AssetLibraryAnnotationStore(args.lookup or module.DEFAULT_LOOKUP)
    if args.self_emission_only:
        record = store.get_self_emission_annotations(args.hssd_id)
        if record is None:
            raise KeyError(f"HSSD id not found: {args.hssd_id}")
    else:
        record = store.require(args.hssd_id)
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
