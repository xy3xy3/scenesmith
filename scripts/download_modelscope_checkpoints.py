#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import uuid

from pathlib import Path

import requests

from modelscope.hub.api import HubApi, ModelScopeConfig
from modelscope.hub.file_download import get_file_download_url
from tqdm.auto import tqdm

SAM3_MODEL_ID = "facebook/sam3"
SAM3D_MODEL_ID = "facebook/sam-3d-objects"
RETRY_TIMES = 3
CHUNK_SIZE = 1024 * 1024


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_repo_files(api: HubApi, model_id: str) -> tuple[str, str, list[dict]]:
    endpoint = api.get_endpoint_for_read(
        repo_id=model_id, repo_type="model", token=None
    )
    cookies = api.get_cookies()
    revision = api.get_valid_revision(
        model_id, revision=None, cookies=cookies, endpoint=endpoint
    )
    repo_files = api.get_model_files(
        model_id=model_id,
        revision=revision,
        recursive=True,
        use_cookies=False if cookies is None else cookies,
        endpoint=endpoint,
    )
    return endpoint, revision, repo_files


def build_targets(api: HubApi) -> list[dict]:
    targets: list[dict] = []

    endpoint, revision, repo_files = describe_repo_files(api, SAM3_MODEL_ID)
    for repo_file in repo_files:
        if repo_file.get("Type") == "tree":
            continue
        if repo_file.get("Path") == "sam3.pt":
            targets.append(
                {
                    "model_id": SAM3_MODEL_ID,
                    "endpoint": endpoint,
                    "revision": revision,
                    "repo_file": repo_file,
                    "dest_rel": Path("sam3.pt"),
                }
            )
            break
    else:
        raise FileNotFoundError(f"sam3.pt not found in {SAM3_MODEL_ID}")

    endpoint, revision, repo_files = describe_repo_files(api, SAM3D_MODEL_ID)
    found_any = False
    for repo_file in repo_files:
        if repo_file.get("Type") == "tree":
            continue
        remote_path = repo_file.get("Path", "")
        if not remote_path.startswith("checkpoints/"):
            continue
        found_any = True
        targets.append(
            {
                "model_id": SAM3D_MODEL_ID,
                "endpoint": endpoint,
                "revision": revision,
                "repo_file": repo_file,
                "dest_rel": Path(remote_path.removeprefix("checkpoints/")),
            }
        )
    if not found_any:
        raise FileNotFoundError(f"No checkpoint files found in {SAM3D_MODEL_ID}")

    return targets


def verify_existing_file(path: Path, expected_hash: str, expected_size: int) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size != expected_size:
        return False
    actual_hash = compute_sha256(path)
    return actual_hash == expected_hash


def legacy_cache_candidates(
    output_dir: Path, model_id: str, remote_path: str
) -> list[Path]:
    owner, name = model_id.split("/", 1)
    return [
        output_dir / "modelscope-cache" / owner / name / remote_path,
        output_dir / "modelscope-cache" / "._____temp" / owner / name / remote_path,
    ]


def adopt_existing_downloads(
    output_dir: Path,
    model_id: str,
    remote_path: str,
    dest_path: Path,
    expected_hash: str,
    expected_size: int,
) -> None:
    part_path = dest_path.with_name(dest_path.name + ".part")

    if dest_path.exists() and not verify_existing_file(
        dest_path, expected_hash, expected_size
    ):
        current_size = dest_path.stat().st_size
        if current_size < expected_size and not part_path.exists():
            print(f"Reusing incomplete file as resume source: {dest_path}")
            shutil.move(str(dest_path), str(part_path))

    for candidate in legacy_cache_candidates(output_dir, model_id, remote_path):
        if not candidate.exists():
            continue

        if verify_existing_file(candidate, expected_hash, expected_size):
            if not verify_existing_file(dest_path, expected_hash, expected_size):
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Promoting verified legacy download: {candidate}")
                shutil.copy2(candidate, dest_path)
            return

        candidate_size = candidate.stat().st_size
        if candidate_size < expected_size and not part_path.exists():
            part_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Reusing legacy partial download: {candidate}")
            shutil.copy2(candidate, part_path)
            return


def download_with_resume(
    url: str,
    dest_path: Path,
    expected_hash: str,
    expected_size: int,
    cookies,
    headers: dict[str, str],
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_name(dest_path.name + ".part")

    if verify_existing_file(dest_path, expected_hash, expected_size):
        print(f"Skip verified file: {dest_path}")
        return

    if part_path.exists() and part_path.stat().st_size > expected_size:
        print(f"Discard oversized partial file: {part_path}")
        part_path.unlink()

    for attempt in range(1, RETRY_TIMES + 1):
        if part_path.exists() and part_path.stat().st_size == expected_size:
            if compute_sha256(part_path) == expected_hash:
                shutil.move(str(part_path), str(dest_path))
                print(f"Promoted completed partial file: {dest_path}")
                return
            print(f"Discard corrupt completed partial file: {part_path}")
            part_path.unlink()

        resume_from = part_path.stat().st_size if part_path.exists() else 0
        request_headers = dict(headers)
        mode = "ab" if resume_from > 0 else "wb"
        if resume_from > 0:
            request_headers["Range"] = f"bytes={resume_from}-{expected_size - 1}"

        if resume_from > 0:
            print(f"Resuming {dest_path.name} from {resume_from / (1024 ** 3):.2f} GiB")
        else:
            print(f"Downloading {dest_path.name}")

        try:
            with requests.get(
                url,
                stream=True,
                headers=request_headers,
                cookies=cookies,
                timeout=(30, 3600),
            ) as response:
                response.raise_for_status()

                if resume_from > 0 and response.status_code == 200:
                    print(
                        f"Server ignored Range for {dest_path.name}, restarting this file"
                    )
                    part_path.unlink(missing_ok=True)
                    resume_from = 0
                    mode = "wb"

                progress = tqdm(
                    total=expected_size,
                    initial=resume_from,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=dest_path.name,
                )
                with part_path.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        progress.update(len(chunk))
                progress.close()

            actual_size = part_path.stat().st_size
            if actual_size != expected_size:
                raise IOError(
                    f"Size mismatch for {dest_path.name}: expected {expected_size}, got {actual_size}"
                )

            actual_hash = compute_sha256(part_path)
            if actual_hash != expected_hash:
                raise IOError(
                    f"SHA256 mismatch for {dest_path.name}: expected {expected_hash}, got {actual_hash}"
                )

            shutil.move(str(part_path), str(dest_path))
            print(f"Downloaded and verified: {dest_path}")
            return
        except Exception as exc:
            print(f"Attempt {attempt}/{RETRY_TIMES} failed for {dest_path.name}: {exc}")
            if attempt == RETRY_TIMES:
                raise
            time.sleep(min(5 * attempt, 15))

    raise RuntimeError(f"Exhausted retries for {dest_path}")


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
        help="Only print remote checkpoint file metadata without downloading",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    api = HubApi()
    headers = {
        "user-agent": ModelScopeConfig.get_user_agent(user_agent=None),
        "X-Request-ID": uuid.uuid4().hex,
    }
    cookies = api.get_cookies()

    targets = build_targets(api)

    if args.list_only:
        for target in targets:
            repo_file = target["repo_file"]
            print(
                f"{target['model_id']} :: {repo_file['Path']} -> "
                f"{target['dest_rel']} ({repo_file['Size']} bytes)"
            )
        return 0

    for target in targets:
        repo_file = target["repo_file"]
        dest_path = output_dir / target["dest_rel"]
        adopt_existing_downloads(
            output_dir=output_dir,
            model_id=target["model_id"],
            remote_path=repo_file["Path"],
            dest_path=dest_path,
            expected_hash=repo_file["Sha256"],
            expected_size=repo_file["Size"],
        )
        url = get_file_download_url(
            target["model_id"],
            repo_file["Path"],
            target["revision"],
            target["endpoint"],
        )
        download_with_resume(
            url=url,
            dest_path=dest_path,
            expected_hash=repo_file["Sha256"],
            expected_size=repo_file["Size"],
            cookies=cookies,
            headers=headers,
        )

    (output_dir / ".sam3d_objects_downloaded").touch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
