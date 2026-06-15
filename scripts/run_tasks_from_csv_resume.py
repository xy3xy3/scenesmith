#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


def repo_root_from_this_file() -> Path:
    # .../scenesmith/scripts/run_tasks_from_csv_resume.py -> .../scenesmith
    return Path(__file__).resolve().parents[1]


def _hydra_quote(value: str) -> str:
    """Quote a string for Hydra override parsing (keeps spaces safe)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


@dataclass(frozen=True)
class TaskRow:
    task_id: str
    prompt: str


def iter_task_rows(csv_path: Path) -> Iterable[TaskRow]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {csv_path}")
        if "id" not in reader.fieldnames or "instruction" not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain 'id' and 'instruction' columns; got {reader.fieldnames}"
            )
        for row in reader:
            task_id = (row.get("id") or "").strip()
            prompt = (row.get("instruction") or "").strip()
            if not task_id or not prompt:
                continue
            yield TaskRow(task_id=task_id, prompt=prompt)


PIPELINE_STAGES = [
    "floor_plan",
    "furniture",
    "wall_mounted",
    "ceiling_mounted",
    "manipuland",
]


def _room_dirs(scene_dir: Path) -> list[Path]:
    rooms: list[Path] = []
    for p in scene_dir.iterdir():
        if not p.is_dir() or not p.name.startswith("room_"):
            continue
        # Scene-level directory that unfortunately matches the room_ prefix.
        if p.name == "room_geometry":
            continue
        # Real room dirs always have scene_states/ (checkpoint root).
        if not (p / "scene_states").is_dir():
            continue
        rooms.append(p)
    return sorted(rooms)


def _all_rooms_have(scene_dir: Path, rel_path: Path) -> bool:
    rooms = _room_dirs(scene_dir)
    if not rooms:
        return False
    return all((room / rel_path).exists() for room in rooms)


def compute_resume_stage(output_dir: Path) -> tuple[str, bool]:
    """Infer which pipeline stage to resume from based on existing checkpoints.

    Returns:
        (start_stage, done)
    """
    scene_dir = output_dir / "scene_000"
    house_layout_path = scene_dir / "house_layout.json"
    # Floor plan stage is considered complete only if core scene-level artifacts exist.
    if not house_layout_path.exists():
        return ("floor_plan", False)
    if not (scene_dir / "room_geometry").is_dir():
        return ("floor_plan", False)
    if not (scene_dir / "floor_plans").is_dir():
        return ("floor_plan", False)

    if not _all_rooms_have(
        scene_dir, Path("scene_states/scene_after_furniture/scene_state.json")
    ):
        return ("furniture", False)
    if not _all_rooms_have(
        scene_dir, Path("scene_states/scene_after_wall_objects/scene_state.json")
    ):
        return ("wall_mounted", False)
    if not _all_rooms_have(
        scene_dir, Path("scene_states/scene_after_ceiling_objects/scene_state.json")
    ):
        return ("ceiling_mounted", False)
    if not _all_rooms_have(
        scene_dir, Path("scene_states/final_scene/scene_state.json")
    ):
        return ("manipuland", False)

    return ("manipuland", True)


DEFAULT_OVERRIDES: list[str] = [
    "floor_plan_agent.mode=room",
    "furniture_agent.asset_manager.general_asset_source=hssd",
    "wall_agent.asset_manager.general_asset_source=hssd",
    "ceiling_agent.asset_manager.general_asset_source=hssd",
    "manipuland_agent.asset_manager.general_asset_source=hssd",
]


def run_one_task(
    *,
    repo_root: Path,
    task_id: str,
    prompt: str,
    basedir: Path,
    skip_done: bool,
    dry_run: bool,
    extra_overrides: Sequence[str],
    use_default_overrides: bool,
) -> int:
    output_dir = (basedir / task_id).resolve()
    start_stage, done = compute_resume_stage(output_dir)

    if skip_done and done:
        print(f"[id {task_id}] done; skipping ({output_dir})")
        return 0

    if start_stage not in PIPELINE_STAGES:
        raise ValueError(f"Invalid inferred start_stage={start_stage!r}")

    cmd: list[str] = [
        sys.executable,
        "main.py",
        f"+name=task_{task_id}",
        f"+prompt={_hydra_quote(prompt)}",
        f"hydra.run.dir={output_dir.as_posix()}",
        f"experiment.pipeline.start_stage={start_stage}",
        "experiment.pipeline.stop_stage=manipuland",
    ]
    if use_default_overrides:
        cmd.extend(DEFAULT_OVERRIDES)
    cmd.extend(list(extra_overrides))

    print(f"[id {task_id}] output_dir={output_dir} resume_from={start_stage}")
    if dry_run:
        print("DRY_RUN:", " ".join(repr(c) for c in cmd))
        return 0

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.run(cmd, cwd=str(repo_root), env=env)
    return int(proc.returncode)


def main() -> int:
    repo_root = repo_root_from_this_file()

    parser = argparse.ArgumentParser(
        description=(
            "Batch-run SceneSmith prompts from task_instructions.csv with per-id outputs/ID "
            "directories and automatic resume from the last incomplete pipeline stage."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=repo_root / "task_instructions.csv",
        help="CSV path (must include columns: id, instruction).",
    )
    parser.add_argument(
        "--basedir",
        type=Path,
        default=repo_root / "outputs",
        help="Base output directory; each task writes into basedir/<id>/scene_000/...",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="Only run rows with numeric id >= this value (default: 1).",
    )
    parser.add_argument(
        "--only-id",
        action="append",
        default=[],
        help="Only run specific id(s). Repeatable.",
    )
    parser.add_argument(
        "--no-skip-done",
        action="store_true",
        help="Do not skip completed ids (force re-run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing.",
    )
    parser.add_argument(
        "--no-default-overrides",
        action="store_true",
        help="Do not apply built-in overrides (asset sources + room mode).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override(s) forwarded to main.py. Repeatable.",
    )

    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    basedir = args.basedir.expanduser().resolve()
    basedir.mkdir(parents=True, exist_ok=True)

    only_ids = {str(x).strip() for x in args.only_id if str(x).strip()}
    if only_ids:
        print(f"Filtering to ids: {sorted(only_ids)}")

    for row in iter_task_rows(csv_path):
        try:
            row_id_int = int(row.task_id)
        except ValueError:
            row_id_int = None

        if row_id_int is not None and row_id_int < args.start_id:
            continue
        if only_ids and row.task_id not in only_ids:
            continue

        rc = run_one_task(
            repo_root=repo_root,
            task_id=row.task_id,
            prompt=row.prompt,
            basedir=basedir,
            skip_done=not args.no_skip_done,
            dry_run=args.dry_run,
            extra_overrides=args.override,
            use_default_overrides=not args.no_default_overrides,
        )
        if rc != 0:
            print(f"[id {row.task_id}] exited rc={rc}")
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
