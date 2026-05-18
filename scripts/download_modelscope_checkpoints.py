#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys

from pathlib import Path

import yaml

from modelscope import snapshot_download

SAM3_MODEL_ID = "facebook/sam3"
SAM3D_MODEL_ID = "facebook/sam-3d-objects"
SAM3_REQUIRED = ["sam3.pt"]
SAM3D_PIPELINE = "checkpoints/pipeline.yaml"
CHECKPOINT_SUFFIXES = (".yaml", ".ckpt", ".pt", ".safetensors")


def download_repo_subset(
    model_id: str,
    allow_patterns: list[str],
    cache_dir: Path,
    local_dir: Path,
) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Syncing {model_id} via ModelScope SDK")
    print(f"  allow_patterns={allow_patterns}")
    snapshot_download(
        model_id=model_id,
        repo_type="model",
        cache_dir=str(cache_dir),
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        max_workers=4,
    )
    return local_dir


def extract_pipeline_artifacts(pipeline_path: Path) -> list[str]:
    data = yaml.safe_load(pipeline_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected pipeline format in {pipeline_path}")

    remote_paths: set[str] = {SAM3D_PIPELINE}
    for value in data.values():
        if not isinstance(value, str):
            continue
        if not value.endswith(CHECKPOINT_SUFFIXES):
            continue
        # pipeline.yaml stores checkpoint paths relative to checkpoints/
        remote_paths.add(f"checkpoints/{value}")
    return sorted(remote_paths)


def sync_required_files(
    repo_root: Path,
    remote_paths: list[str],
    output_dir: Path,
) -> None:
    for remote_path in remote_paths:
        source = repo_root / remote_path
        dest = dest_for_remote_path(output_dir, remote_path)

        if not source.is_file():
            if dest.is_file():
                print(f"Ready: {dest}")
                continue
            raise FileNotFoundError(
                f"ModelScope SDK did not materialize required file: {source}"
            )

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        print(f"Ready: {dest}")


def dest_for_remote_path(output_dir: Path, remote_path: str) -> Path:
    if remote_path.startswith("checkpoints/"):
        return output_dir / Path(remote_path).name
    return output_dir / remote_path


def filter_missing_remote_paths(output_dir: Path, remote_paths: list[str]) -> list[str]:
    missing: list[str] = []
    for remote_path in remote_paths:
        if not dest_for_remote_path(output_dir, remote_path).is_file():
            missing.append(remote_path)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="external/checkpoints",
        help="Directory where checkpoint files should be stored",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print which checkpoint files would be downloaded",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / "modelscope-cache"
    mirror_dir = output_dir / ".modelscope-sdk"
    sam3_local_dir = mirror_dir / "facebook_sam3"
    sam3d_local_dir = mirror_dir / "facebook_sam-3d-objects"

    # Download the tiny pipeline manifest first so we only request checkpoints
    # that the current pipeline actually references.
    download_repo_subset(
        model_id=SAM3D_MODEL_ID,
        allow_patterns=[SAM3D_PIPELINE],
        cache_dir=cache_dir,
        local_dir=sam3d_local_dir,
    )
    pipeline_path = sam3d_local_dir / SAM3D_PIPELINE
    sam3d_required = extract_pipeline_artifacts(pipeline_path)

    if args.list_only:
        print(f"{SAM3_MODEL_ID}:")
        for path in SAM3_REQUIRED:
            print(f"  {path}")
        print(f"{SAM3D_MODEL_ID}:")
        for path in sam3d_required:
            print(f"  {path}")
        return 0

    missing_sam3 = filter_missing_remote_paths(output_dir, SAM3_REQUIRED)
    if missing_sam3:
        download_repo_subset(
            model_id=SAM3_MODEL_ID,
            allow_patterns=missing_sam3,
            cache_dir=cache_dir,
            local_dir=sam3_local_dir,
        )
    else:
        print("All SAM3 files already exist in output directory; skipping SDK download")
    sync_required_files(
        repo_root=sam3_local_dir,
        remote_paths=SAM3_REQUIRED,
        output_dir=output_dir,
    )

    missing_sam3d = filter_missing_remote_paths(output_dir, sam3d_required)
    if missing_sam3d:
        download_repo_subset(
            model_id=SAM3D_MODEL_ID,
            allow_patterns=missing_sam3d,
            cache_dir=cache_dir,
            local_dir=sam3d_local_dir,
        )
    else:
        print(
            "All SAM 3D Objects files already exist in output directory; "
            "skipping SDK download"
        )
    sync_required_files(
        repo_root=sam3d_local_dir,
        remote_paths=sam3d_required,
        output_dir=output_dir,
    )

    (output_dir / ".sam3d_objects_downloaded").touch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
