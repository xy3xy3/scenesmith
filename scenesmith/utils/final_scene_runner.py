"""CLI helpers for rerunning the SceneBenchmark critic on final scenes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, open_dict

from scenesmith.utils.omegaconf import register_resolvers

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun the SceneBenchmark critic on saved final_scene outputs. "
            "Reports are written in place by default."
        )
    )
    parser.add_argument(
        "target",
        type=Path,
        help=(
            "Path to a critic output root, batch dir, scene dir, room dir, "
            "final_scene dir, or final_scene/scene_state.json."
        ),
    )
    parser.add_argument(
        "--resolved-config",
        type=Path,
        default=None,
        help=(
            "Optional resolved_config.yaml to use for all scenes. If omitted, "
            "the script auto-discovers the nearest resolved_config.yaml."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional directory for report copies. If omitted, reports are "
            "written directly into each final_scene directory."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="Custom OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--model",
        "--model-name",
        dest="model",
        default=None,
        help="Model name used by SceneBenchmark VLM calls.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key passed through OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--use-responses",
        "--is-response",
        dest="use_responses",
        choices=("true", "false"),
        default=None,
        help="Whether to use the Responses API.",
    )
    parser.add_argument(
        "--write-scene-state",
        action="store_true",
        help=(
            "Also write asset-annotation hints back into scene_state.json. "
            "Disabled by default to avoid mutating the source scene."
        ),
    )
    return parser.parse_args(argv)


def discover_final_scene_states(target: Path) -> list[Path]:
    target = target.expanduser().resolve()
    if target.is_file():
        if target.name != "scene_state.json":
            raise ValueError(f"Expected scene_state.json, got: {target}")
        return [target]
    if not target.exists():
        raise FileNotFoundError(f"Target path does not exist: {target}")

    direct_candidates = [
        target / "scene_state.json",
        target / "final_scene" / "scene_state.json",
        target / "scene_states" / "final_scene" / "scene_state.json",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return [candidate]

    patterns = (
        "room_*/scene_states/final_scene/scene_state.json",
        "scene_*/room_*/scene_states/final_scene/scene_state.json",
        "**/scene_states/final_scene/scene_state.json",
    )
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(target.glob(pattern))

    unique_matches = sorted({path.resolve() for path in matches if path.is_file()})
    if not unique_matches:
        raise FileNotFoundError(
            "Could not find any final_scene/scene_state.json under "
            f"{target}"
        )
    return unique_matches


def find_nearest_resolved_config(path: Path) -> Path | None:
    current = path.expanduser().resolve()
    for candidate in (current, *current.parents):
        resolved_config = candidate / "resolved_config.yaml"
        if resolved_config.is_file():
            return resolved_config
    return None


def relative_output_subdir(scene_state_path: Path, target_root: Path) -> Path:
    target_root = target_root.expanduser().resolve()
    scene_state_path = scene_state_path.expanduser().resolve()
    try:
        return scene_state_path.parent.relative_to(target_root)
    except ValueError:
        pass

    parts = scene_state_path.parts
    tail = parts[-5:-1]
    if tail:
        return Path(*tail)
    return Path(scene_state_path.parent.name)


def load_base_cfg(scene_state_path: Path, resolved_config_path: Path | None) -> Any:
    register_resolvers()
    if resolved_config_path is not None:
        return OmegaConf.load(resolved_config_path)

    discovered = find_nearest_resolved_config(scene_state_path)
    if discovered is not None:
        return OmegaConf.load(discovered)

    return _load_default_cfg()


def apply_runtime_overrides(
    cfg: Any,
    *,
    base_url: str | None,
    model: str | None,
    use_responses: bool | None,
    write_scene_state: bool,
) -> Any:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(cfg):
        if "openai" not in cfg or cfg.openai is None:
            cfg.openai = OmegaConf.create({})
        if base_url:
            cfg.openai.base_url = base_url
        if use_responses is not None:
            cfg.openai.use_responses = use_responses
        if model:
            cfg.openai.model = model

        if "experiment" not in cfg or cfg.experiment is None:
            cfg.experiment = OmegaConf.create({})
        if "scenebenchmark_critic" not in cfg.experiment:
            cfg.experiment.scenebenchmark_critic = OmegaConf.create({})

        critic_cfg = cfg.experiment.scenebenchmark_critic
        critic_cfg.enabled = True
        critic_cfg.room_stage_hooks = ["final_scene"]
        critic_cfg.house_stage_hooks = []

        if "asset_annotation" not in critic_cfg or critic_cfg.asset_annotation is None:
            critic_cfg.asset_annotation = OmegaConf.create({})
        asset_cfg = critic_cfg.asset_annotation
        if model:
            asset_cfg.model = model
        asset_cfg.skip_existing = False
        asset_cfg.refresh = True
        asset_cfg.write_scene_state = write_scene_state
    return cfg


def evaluate_final_scene(
    scene_state_path: Path,
    *,
    cfg: Any,
    output_dir: Path,
) -> dict[str, Any]:
    from scenesmith.experiments.indoor_scene_generation import _load_room_scene_state
    from scenesmith.scenebenchmark_critic.api import write_room_stage_report

    scene = _load_room_scene_state(scene_state_path)
    payload = write_room_stage_report(
        scene,
        output_dir,
        config=cfg,
        raw_config=cfg,
        stage="final_scene",
    )
    if payload is None:
        raise RuntimeError(f"SceneBenchmark critic did not run for {scene_state_path}")

    scene_summary = ((payload.get("summary") or {}).get("scene_summary")) or {}
    result = {
        "scene_state_path": str(scene_state_path),
        "output_dir": str(output_dir),
        "report_json": str(output_dir / "scenebenchmark_critic.json"),
        "report_md": str(output_dir / "scenebenchmark_critic.md"),
        "scope": payload.get("scope"),
        "stage": payload.get("stage"),
        "score": scene_summary.get("score"),
        "checks": scene_summary.get("total_checks"),
        "pass": scene_summary.get("pass"),
        "degraded": scene_summary.get("degraded"),
        "fail": scene_summary.get("fail"),
        "unknown": scene_summary.get("unknown"),
    }
    return result


def run_final_scene_critic(
    *,
    target: Path,
    resolved_config: Path | None = None,
    output_root: Path | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    use_responses: bool | None = None,
    write_scene_state: bool = False,
) -> list[dict[str, Any]]:
    _configure_environment(
        api_key=api_key,
        base_url=base_url,
        use_responses=use_responses,
    )
    resolved_config = (
        resolved_config.expanduser().resolve() if resolved_config is not None else None
    )
    target = target.expanduser().resolve()
    scene_state_paths = discover_final_scene_states(target)

    results: list[dict[str, Any]] = []
    for scene_state_path in scene_state_paths:
        base_cfg = load_base_cfg(scene_state_path, resolved_config)
        cfg = apply_runtime_overrides(
            base_cfg,
            base_url=base_url,
            model=model,
            use_responses=use_responses,
            write_scene_state=write_scene_state,
        )
        if output_root is None:
            report_dir = scene_state_path.parent
        else:
            report_dir = output_root.expanduser().resolve() / relative_output_subdir(
                scene_state_path, target
            )
        results.append(
            evaluate_final_scene(
                scene_state_path,
                cfg=cfg,
                output_dir=report_dir,
            )
        )

    if output_root is not None:
        summary_path = output_root.expanduser().resolve() / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return results


def format_results_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "score\tfail\tdegraded\tpass\tchecks\tscene_state",
    ]
    for item in results:
        score = item.get("score")
        score_text = "n/a" if score is None else f"{float(score):.3f}"
        lines.append(
            "\t".join(
                [
                    score_text,
                    str(item.get("fail")),
                    str(item.get("degraded")),
                    str(item.get("pass")),
                    str(item.get("checks")),
                    str(item.get("scene_state_path")),
                ]
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_final_scene_critic(
        target=args.target,
        resolved_config=args.resolved_config,
        output_root=args.output_root,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        use_responses=_bool_from_arg(args.use_responses),
        write_scene_state=args.write_scene_state,
    )
    print(format_results_table(results))
    return 0


def _bool_from_arg(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "true"


def _configure_environment(
    *,
    api_key: str | None,
    base_url: str | None,
    use_responses: bool | None,
) -> None:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif not os.environ.get("OPENAI_API_KEY") and _looks_local_base_url(base_url):
        os.environ["OPENAI_API_KEY"] = "dummy"

    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if use_responses is not None:
        os.environ["OPENAI_USE_RESPONSES"] = str(use_responses).lower()


def _looks_local_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    return "127.0.0.1" in base_url or "localhost" in base_url


def _load_group(config_dir: Path, group: str, name: str) -> Any:
    group_cfg = OmegaConf.load(config_dir / group / f"{name}.yaml")
    nested_defaults = group_cfg.get("defaults", [])
    merged = OmegaConf.create()
    for nested_item in nested_defaults:
        if isinstance(nested_item, str):
            parent_cfg = OmegaConf.load(config_dir / group / f"{nested_item}.yaml")
            merged = OmegaConf.merge(merged, parent_cfg)
    merged = OmegaConf.merge(merged, group_cfg)
    if "defaults" in merged:
        with open_dict(merged):
            del merged["defaults"]
    return merged


def _load_default_cfg() -> Any:
    config_dir = REPO_ROOT / "configurations"
    root_cfg = OmegaConf.load(config_dir / "config.yaml")
    cfg = OmegaConf.merge(
        root_cfg,
        {
            "experiment": _load_group(
                config_dir, "experiment", "indoor_scene_generation"
            ),
            "floor_plan_agent": _load_group(
                config_dir, "floor_plan_agent", "stateful_floor_plan_agent"
            ),
            "furniture_agent": _load_group(
                config_dir, "furniture_agent", "stateful_furniture_agent"
            ),
            "wall_agent": _load_group(config_dir, "wall_agent", "stateful_wall_agent"),
            "ceiling_agent": _load_group(
                config_dir, "ceiling_agent", "stateful_ceiling_agent"
            ),
            "manipuland_agent": _load_group(
                config_dir, "manipuland_agent", "stateful_manipuland_agent"
            ),
        },
    )
    if "defaults" in cfg:
        with open_dict(cfg):
            del cfg["defaults"]
    OmegaConf.resolve(cfg)
    return cfg
