#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, open_dict


DEFAULT_BATCH_DIR = Path(
    "outputs/critic_probe/2026-07-01_16-31-19/critic_on/batch_001"
)
DEFAULT_BASE_URL = "http://127.0.0.1:8002/v1"
DEFAULT_MODEL = "unsloth/Qwen3.6-27B-GGUF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the batch_001 final bedroom scene against a local OpenAI-"
            "compatible VLM and check whether bed-against-wall issues are surfaced."
        )
    )
    parser.add_argument(
        "--scene-state",
        type=Path,
        default=None,
        help=(
            "Path to the final scene_state.json to evaluate. If omitted, the script "
            "auto-discovers a single final_scene/scene_state.json under --batch-dir."
        ),
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=DEFAULT_BATCH_DIR,
        help="Path to the batch directory containing resolved_config.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for replay outputs. If omitted, defaults to "
            "outputs/local_qwen_final_scene_test/<batch_name>."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL for the local llama-server.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model id exposed by the local llama-server.",
    )
    parser.add_argument(
        "--mode",
        choices=("original", "benchmark", "both"),
        default="both",
        help="Which critique path to run.",
    )
    parser.add_argument(
        "--placement-style",
        choices=("natural", "perfect"),
        default="natural",
        help="Placement style passed to the original furniture critic replay.",
    )
    return parser.parse_args()


def _abs(path: Path) -> Path:
    return path.expanduser().resolve()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_resolved_config(batch_dir: Path) -> Any:
    return OmegaConf.load(batch_dir / "resolved_config.yaml")


def _discover_scene_state(batch_dir: Path) -> Path:
    matches = sorted(batch_dir.glob("scene_*/room_*/scene_states/final_scene/scene_state.json"))
    if not matches:
        raise FileNotFoundError(
            f"No final_scene/scene_state.json found under batch dir: {batch_dir}"
        )
    if len(matches) > 1:
        joined = "\n".join(str(match) for match in matches)
        raise RuntimeError(
            "Multiple final scenes found; pass --scene-state explicitly:\n"
            f"{joined}"
        )
    return matches[0]


def _default_output_dir(batch_dir: Path) -> Path:
    return Path("outputs/local_qwen_final_scene_test") / batch_dir.name


def _configure_local_model(cfg: Any, *, base_url: str, model: str) -> Any:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(cfg):
        cfg.openai.base_url = base_url
        cfg.openai.model = model
        cfg.openai.use_responses = False

        critic_cfg = cfg.experiment.scenebenchmark_critic
        critic_cfg.enabled = True
        critic_cfg.room_stage_hooks = ["final_scene"]
        critic_cfg.house_stage_hooks = []
        critic_cfg.fd_relation_proposer_mode = "vlm"

        asset_cfg = critic_cfg.asset_annotation
        asset_cfg.enabled = True
        asset_cfg.backend = "vlm"
        asset_cfg.model = model
        asset_cfg.skip_existing = False
        asset_cfg.refresh = True
        asset_cfg.write_back = True
        asset_cfg.write_files = True
        asset_cfg.write_scene_state = False
    return cfg


def _run_original_critic(
    *,
    scene_state: Path,
    output_dir: Path,
    base_url: str,
    model: str,
    placement_style: str,
) -> dict[str, Any]:
    target_dir = output_dir / "original_critic"
    target_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "dummy")
    env["OPENAI_BASE_URL"] = base_url
    cmd = [
        sys.executable,
        str(_repo_root() / "scripts/run_single_scene_original_critic.py"),
        "--scene-state",
        str(scene_state),
        "--output-dir",
        str(target_dir),
        "--model",
        model,
        "--agent-type",
        "furniture",
        "--placement-style",
        placement_style,
    ]
    completed = subprocess.run(
        cmd,
        cwd=_repo_root(),
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    critic_text_path = target_dir / "exports/furniture/critic_text.md"
    critic_text = (
        critic_text_path.read_text(encoding="utf-8")
        if critic_text_path.exists()
        else ""
    )
    return {
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "critic_text_path": str(critic_text_path) if critic_text_path.exists() else None,
        "mentions_bed": "bed" in critic_text.lower(),
        "flags_bed_issue": _flags_bed_issue(critic_text),
        "bed_excerpt": _extract_bed_excerpt(critic_text),
    }


def _run_benchmark_critic(
    *,
    scene_state: Path,
    batch_dir: Path,
    output_dir: Path,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    from scenesmith.experiments.indoor_scene_generation import _load_room_scene_state
    from scenesmith.scenebenchmark_critic.api import write_room_stage_report

    target_dir = output_dir / "scenebenchmark"
    target_dir.mkdir(parents=True, exist_ok=True)
    cfg = _configure_local_model(
        _load_resolved_config(batch_dir),
        base_url=base_url,
        model=model,
    )
    scene = _load_room_scene_state(scene_state)
    payload = write_room_stage_report(
        scene,
        target_dir,
        config=cfg,
        raw_config=cfg,
        stage="final_scene",
    )
    results = payload.get("results", []) if payload else []
    issues = [
        result
        for result in results
        if str(result.get("label") or "").lower() in {"fail", "degraded", "unknown"}
    ]
    bed_issues = [
        issue
        for issue in issues
        if "bed" in json.dumps(issue, ensure_ascii=False).lower()
    ]
    return {
        "report_path": str(target_dir / "scenebenchmark_critic.md"),
        "json_path": str(target_dir / "scenebenchmark_critic.json"),
        "issue_count": len(issues),
        "result_count": len(results),
        "bed_issue_count": len(bed_issues),
        "bed_issues": bed_issues,
    }


def _flags_bed_issue(text: str) -> bool:
    lowered = text.lower()
    return "bed" in lowered and any(
        marker in lowered
        for marker in (
            "not against",
            "too far from wall",
            "away from wall",
            "bed issue",
            "problem with the bed",
            "bed should",
            "bed needs",
            "headboard should",
        )
    )


def _extract_bed_excerpt(text: str) -> str | None:
    for paragraph in text.split("\n\n"):
        if "bed" in paragraph.lower():
            return paragraph.strip()
    return None


def main() -> int:
    args = parse_args()
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    os.environ["OPENAI_BASE_URL"] = args.base_url
    batch_dir = _abs(args.batch_dir)
    scene_state = _abs(args.scene_state) if args.scene_state else _discover_scene_state(batch_dir)
    output_dir = _abs(args.output_dir) if args.output_dir else _abs(_default_output_dir(batch_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not scene_state.exists():
        raise FileNotFoundError(f"scene_state not found: {scene_state}")
    if not (batch_dir / "resolved_config.yaml").exists():
        raise FileNotFoundError(f"resolved_config.yaml not found under: {batch_dir}")

    summary: dict[str, Any] = {
        "scene_state": str(scene_state),
        "batch_dir": str(batch_dir),
        "base_url": args.base_url,
        "model": args.model,
        "mode": args.mode,
        "python": sys.executable,
    }

    if args.mode in {"original", "both"}:
        summary["original_critic"] = _run_original_critic(
            scene_state=scene_state,
            output_dir=output_dir,
            base_url=args.base_url,
            model=args.model,
            placement_style=args.placement_style,
        )

    if args.mode in {"benchmark", "both"}:
        summary["scenebenchmark"] = _run_benchmark_critic(
            scene_state=scene_state,
            batch_dir=batch_dir,
            output_dir=output_dir,
            base_url=args.base_url,
            model=args.model,
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nText reports:")
    if "original_critic" in summary:
        print(
            "  original critic: "
            f"{summary['original_critic'].get('critic_text_path') or 'not generated'}"
        )
    if "scenebenchmark" in summary:
        print(
            "  scenebenchmark: "
            f"{summary['scenebenchmark'].get('report_path') or 'not generated'}"
        )
    print(f"\nWrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
