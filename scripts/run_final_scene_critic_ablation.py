#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_TARGET = Path(
    "/data/task3_2/L202500266_hrk/code/scenesmith/outputs/critic_probe/"
    "critic_probe_4rooms_2026-07-02_11-45-52/critic_off/batch_001/scene_001"
)

console_logger = logging.getLogger(__name__)


def _load_runtime_dependencies() -> None:
    global ConsoleLogger
    global FileLoggingContext
    global OmegaConf
    global StatefulFurnitureAgent
    global VisionTools
    global _load_room_scene_state
    global critic_config_from_any
    global evaluate_room_scene
    global format_prompt_context
    global open_dict
    global register_resolvers
    global scores_to_dict
    global set_tracing_disabled
    global write_report
    global yaml

    import yaml as yaml_module

    from agents.tracing import set_tracing_disabled as tracing_disabled_fn
    from omegaconf import OmegaConf as omega_conf_cls
    from omegaconf import open_dict as omega_open_dict

    from scenesmith.agent_utils.scoring import scores_to_dict as scores_to_dict_fn
    from scenesmith.experiments.indoor_scene_generation import (
        _load_room_scene_state as load_room_scene_state_fn,
    )
    from scenesmith.furniture_agents.stateful_furniture_agent import (
        StatefulFurnitureAgent as StatefulFurnitureAgentCls,
    )
    from scenesmith.furniture_agents.tools.vision_tools import (
        VisionTools as VisionToolsCls,
    )
    from scenesmith.scenebenchmark_critic import (
        evaluate_room_scene as evaluate_room_scene_fn,
    )
    from scenesmith.scenebenchmark_critic import (
        format_prompt_context as format_prompt_context_fn,
    )
    from scenesmith.scenebenchmark_critic.config import (
        critic_config_from_any as critic_config_from_any_fn,
    )
    from scenesmith.scenebenchmark_critic.reports import write_report as write_report_fn
    from scenesmith.utils.logging import ConsoleLogger as ConsoleLoggerCls
    from scenesmith.utils.logging import FileLoggingContext as FileLoggingContextCls
    from scenesmith.utils.omegaconf import register_resolvers as register_resolvers_fn

    yaml = yaml_module
    set_tracing_disabled = tracing_disabled_fn
    OmegaConf = omega_conf_cls
    open_dict = omega_open_dict
    scores_to_dict = scores_to_dict_fn
    _load_room_scene_state = load_room_scene_state_fn
    StatefulFurnitureAgent = StatefulFurnitureAgentCls
    VisionTools = VisionToolsCls
    evaluate_room_scene = evaluate_room_scene_fn
    format_prompt_context = format_prompt_context_fn
    critic_config_from_any = critic_config_from_any_fn
    write_report = write_report_fn
    ConsoleLogger = ConsoleLoggerCls
    FileLoggingContext = FileLoggingContextCls
    register_resolvers = register_resolvers_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run final-scene critic ablations: original SceneSmith LLM critic "
            "without SceneBenchmark, and the same critic with SceneBenchmark "
            "context injected into the critic prompt."
        )
    )
    parser.add_argument(
        "target",
        nargs="?",
        type=Path,
        default=DEFAULT_TARGET,
        help=(
            "Scene result directory, room directory, or final_scene/scene_state.json. "
            f"Default: {DEFAULT_TARGET}"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible base URL. Omit to use the default OpenAI endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key. Defaults to OPENAI_API_KEY, or 'dummy' for local endpoints.",
    )
    parser.add_argument("--model", required=True, help="Critic model name.")
    parser.add_argument(
        "--use-responses",
        choices=("true", "false"),
        default=None,
        help=(
            "Whether the OpenAI Agents provider should use the Responses API. "
            "Use false for many OpenAI-compatible local servers."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("original", "injected", "both"),
        default="both",
        help="Which critic variant to run.",
    )
    parser.add_argument(
        "--placement-style",
        choices=("natural", "perfect"),
        default="natural",
        help="Placement style passed into the original furniture critic prompt.",
    )
    parser.add_argument(
        "--max-issues-for-prompt",
        type=int,
        default=8,
        help="Maximum SceneBenchmark issues injected into the LLM critic prompt.",
    )
    parser.add_argument(
        "--fd-relation-proposer-mode",
        choices=("template", "vlm"),
        default="template",
        help=(
            "SceneBenchmark functional-dependency relation proposer. Template is "
            "deterministic and avoids extra VLM calls."
        ),
    )
    parser.add_argument(
        "--asset-annotation",
        action="store_true",
        help="Enable SceneBenchmark asset annotation before rule checks.",
    )
    parser.add_argument(
        "--asset-annotation-backend",
        choices=("vlm", "mock"),
        default="vlm",
        help="Backend used only when --asset-annotation is set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate at most N discovered final scenes.",
    )
    return parser.parse_args()


def _bool_from_arg(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "true"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _default_output_dir(target: Path, model: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (
        REPO_ROOT
        / "outputs"
        / "final_scene_critic_ablation"
        / f"{target.expanduser().resolve().name}_{_safe_name(model)}_{stamp}"
    )


def _discover_scene_states(target: Path) -> list[Path]:
    target = target.expanduser().resolve()
    if target.is_file():
        if target.name != "scene_state.json":
            raise ValueError(f"Expected a scene_state.json file, got: {target}")
        return [target]
    if not target.exists():
        raise FileNotFoundError(f"Target path does not exist: {target}")

    direct = target / "scene_states" / "final_scene" / "scene_state.json"
    if direct.exists():
        return [direct]

    final_scene = target / "final_scene" / "scene_state.json"
    if final_scene.exists():
        return [final_scene]

    matches = sorted(target.glob("room_*/scene_states/final_scene/scene_state.json"))
    if matches:
        return matches

    matches = sorted(target.glob("**/scene_states/final_scene/scene_state.json"))
    if matches:
        return matches

    raise FileNotFoundError(
        "Could not find any room final scene_state.json under: " f"{target}"
    )


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


def _load_full_cfg(
    *,
    model: str,
    base_url: str | None,
    use_responses: bool | None,
    scenebenchmark_enabled: bool,
    inject_into_llm_critic: bool,
    max_issues_for_prompt: int,
    fd_relation_proposer_mode: str,
    asset_annotation: bool,
    asset_annotation_backend: str,
) -> Any:
    register_resolvers()
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

    with open_dict(cfg):
        cfg.openai.model = model
        if base_url:
            cfg.openai.base_url = base_url
        if use_responses is not None:
            cfg.openai.use_responses = use_responses

        critic_cfg = cfg.experiment.scenebenchmark_critic
        critic_cfg.enabled = scenebenchmark_enabled
        critic_cfg.inject_into_llm_critic = inject_into_llm_critic
        critic_cfg.max_issues_for_prompt = max_issues_for_prompt
        critic_cfg.fd_relation_proposer_mode = fd_relation_proposer_mode
        critic_cfg.room_stage_hooks = ["final_scene"]
        critic_cfg.house_stage_hooks = []
        critic_cfg.asset_annotation.enabled = asset_annotation
        critic_cfg.asset_annotation.backend = asset_annotation_backend
        critic_cfg.asset_annotation.model = model
    return cfg


def _furniture_agent_cfg(full_cfg: Any) -> Any:
    cfg = OmegaConf.create(
        OmegaConf.to_container(full_cfg.furniture_agent, resolve=True)
    )
    with open_dict(cfg):
        cfg.openai.service_tier = full_cfg.openai.service_tier
        cfg.openai.base_url = full_cfg.openai.base_url
        cfg.openai.use_responses = full_cfg.openai.use_responses
        cfg.openai.model = full_cfg.openai.model
        cfg.scenebenchmark_critic = OmegaConf.create(
            OmegaConf.to_container(
                full_cfg.experiment.scenebenchmark_critic, resolve=True
            )
        )
    return cfg


def _copy_tree(src_dir: Path | None, dst_dir: Path) -> None:
    if src_dir is None or not src_dir.exists():
        return
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_scores(export_dir: Path, scores: Any) -> Path | None:
    if scores is None:
        return None
    path = export_dir / "critic_scores.yaml"
    path.write_text(
        yaml.safe_dump(scores_to_dict(scores), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    raw_path = export_dir / "critic_response.json"
    raw = asdict(scores) if is_dataclass(scores) else scores_to_dict(scores)
    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_scenebenchmark_context(
    *,
    cfg: Any,
    scene_state_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    scene = _load_room_scene_state(scene_state_path)
    payload = evaluate_room_scene(
        scene,
        config=cfg,
        raw_config=cfg,
        stage="llm_critic_furniture",
    )
    report_dir = output_dir / "scenebenchmark_context"
    json_path, md_path = write_report(report_dir, payload)
    prompt_context = format_prompt_context(
        payload,
        max_issues=critic_config_from_any(cfg).max_issues_for_prompt,
    )
    prompt_context_path = report_dir / "prompt_context.md"
    prompt_context_path.write_text(prompt_context, encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "prompt_context_path": str(prompt_context_path),
    }


async def _run_furniture_critic_once(
    *,
    cfg: Any,
    scene_state_path: Path,
    output_dir: Path,
    placement_style: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = ConsoleLogger(output_dir)
    room_id = scene_state_path.parents[2].name.removeprefix("room_")
    export_dir = output_dir / "exports" / "furniture"

    with logger.room_context(room_id):
        scene = _load_room_scene_state(scene_state_path)
        agent = StatefulFurnitureAgent(cfg=cfg, logger=logger)
        try:
            agent.scene = scene
            agent.placement_style = placement_style

            initial_render_dir = VisionTools(
                scene=scene,
                rendering_manager=agent.rendering_manager,
                cfg=cfg,
                blender_server=agent.blender_server,
            ).rendering_manager.render_scene(
                scene,
                blender_server=agent.blender_server,
            )

            critic_tools = agent._create_critic_tools()
            agent.critic = agent._create_critic_agent(scene=scene, tools=critic_tools)
            await agent._request_critique_impl(update_checkpoint=False)

            final_render_dir = (
                agent.rendering_manager.last_render_dir or initial_render_dir
            )
            export_dir.mkdir(parents=True, exist_ok=True)
            _copy_tree(final_render_dir, export_dir / "images")
            _copy_if_exists(scene_state_path, export_dir / "scene_state.json")
            _copy_if_exists(
                scene_state_path.with_name("scene.dmd.yaml"),
                export_dir / "scene.dmd.yaml",
            )

            scores = getattr(agent, "previous_scores", None)
            critic_text = getattr(scores, "critique", "") if scores is not None else ""
            critic_text_path = export_dir / "critic_text.md"
            critic_text_path.write_text(critic_text, encoding="utf-8")
            scores_path = _write_scores(export_dir, scores)

            return {
                "room_id": room_id,
                "scene_state_path": str(scene_state_path),
                "output_dir": str(output_dir),
                "export_dir": str(export_dir),
                "images_dir": str(export_dir / "images"),
                "source_render_dir": str(final_render_dir),
                "critic_text_path": str(critic_text_path),
                "critic_scores_path": str(scores_path) if scores_path else None,
            }
        finally:
            agent.cleanup()


def _write_index(output_dir: Path, summary: dict[str, Any]) -> Path:
    lines = [
        "# Final Scene Critic Ablation",
        "",
        f"- Model: `{summary['model']}`",
        f"- Base URL: `{summary.get('base_url') or 'default OpenAI endpoint'}`",
        f"- Use responses: `{summary.get('use_responses')}`",
        f"- Target: `{summary['target']}`",
        "",
        "## Runs",
        "",
    ]
    for room in summary["rooms"]:
        lines.extend([f"### {room['room_id']}", ""])
        for variant_name, variant in room["variants"].items():
            export_dir = Path(variant["export_dir"])
            rel_text = os.path.relpath(variant["critic_text_path"], output_dir)
            rel_images = os.path.relpath(variant["images_dir"], output_dir)
            lines.extend(
                [
                    f"- `{variant_name}` text: [{rel_text}]({rel_text})",
                    f"- `{variant_name}` images: [{rel_images}]({rel_images})",
                ]
            )
            context = variant.get("scenebenchmark_context")
            if context:
                rel_context = os.path.relpath(context["markdown_path"], output_dir)
                rel_prompt = os.path.relpath(context["prompt_context_path"], output_dir)
                lines.extend(
                    [
                        f"- `{variant_name}` SceneBenchmark report: "
                        f"[{rel_context}]({rel_context})",
                        f"- `{variant_name}` injected prompt context: "
                        f"[{rel_prompt}]({rel_prompt})",
                    ]
                )
            pngs = sorted(export_dir.glob("images/*.png"))
            if pngs:
                lines.append("")
                for png in pngs[:8]:
                    rel_png = os.path.relpath(png, output_dir)
                    lines.append(f"![{png.name}]({rel_png})")
                lines.append("")
        lines.append("")
    index_path = output_dir / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


def _set_openai_env(
    *, base_url: str | None, api_key: str | None, use_responses: bool | None
) -> None:
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        os.environ.setdefault("OPENAI_API_KEY", "dummy")
    if use_responses is not None:
        os.environ["OPENAI_USE_RESPONSES"] = "true" if use_responses else "false"


def main() -> int:
    args = parse_args()
    _load_runtime_dependencies()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    set_tracing_disabled(True)

    target = args.target.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else _default_output_dir(target, args.model)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    use_responses = _bool_from_arg(args.use_responses)
    _set_openai_env(
        base_url=args.base_url,
        api_key=args.api_key,
        use_responses=use_responses,
    )

    scene_states = _discover_scene_states(target)
    if args.limit is not None:
        scene_states = scene_states[: args.limit]

    summary: dict[str, Any] = {
        "target": str(target),
        "output_dir": str(output_dir),
        "model": args.model,
        "base_url": args.base_url,
        "use_responses": use_responses,
        "mode": args.mode,
        "placement_style": args.placement_style,
        "rooms": [],
    }

    with FileLoggingContext(
        log_file_path=output_dir / "experiment.log", suppress_stdout=False
    ):
        console_logger.info("Output directory: %s", output_dir)
        console_logger.info("Discovered %d final scene(s)", len(scene_states))

        for scene_state_path in scene_states:
            room_id = scene_state_path.parents[2].name.removeprefix("room_")
            room_summary: dict[str, Any] = {
                "room_id": room_id,
                "scene_state_path": str(scene_state_path),
                "variants": {},
            }
            room_dir = output_dir / room_id

            if args.mode in {"original", "both"}:
                variant_dir = room_dir / "original_no_scenebenchmark"
                cfg = _furniture_agent_cfg(
                    _load_full_cfg(
                        model=args.model,
                        base_url=args.base_url,
                        use_responses=use_responses,
                        scenebenchmark_enabled=False,
                        inject_into_llm_critic=False,
                        max_issues_for_prompt=args.max_issues_for_prompt,
                        fd_relation_proposer_mode=args.fd_relation_proposer_mode,
                        asset_annotation=False,
                        asset_annotation_backend=args.asset_annotation_backend,
                    )
                )
                room_summary["variants"]["original_no_scenebenchmark"] = asyncio.run(
                    _run_furniture_critic_once(
                        cfg=cfg,
                        scene_state_path=scene_state_path,
                        output_dir=variant_dir,
                        placement_style=args.placement_style,
                    )
                )

            if args.mode in {"injected", "both"}:
                variant_dir = room_dir / "scenebenchmark_injected"
                cfg = _furniture_agent_cfg(
                    _load_full_cfg(
                        model=args.model,
                        base_url=args.base_url,
                        use_responses=use_responses,
                        scenebenchmark_enabled=True,
                        inject_into_llm_critic=True,
                        max_issues_for_prompt=args.max_issues_for_prompt,
                        fd_relation_proposer_mode=args.fd_relation_proposer_mode,
                        asset_annotation=args.asset_annotation,
                        asset_annotation_backend=args.asset_annotation_backend,
                    )
                )
                context_summary = _write_scenebenchmark_context(
                    cfg=cfg,
                    scene_state_path=scene_state_path,
                    output_dir=variant_dir,
                )
                variant_summary = asyncio.run(
                    _run_furniture_critic_once(
                        cfg=cfg,
                        scene_state_path=scene_state_path,
                        output_dir=variant_dir,
                        placement_style=args.placement_style,
                    )
                )
                variant_summary["scenebenchmark_context"] = context_summary
                room_summary["variants"]["scenebenchmark_injected"] = variant_summary

            summary["rooms"].append(room_summary)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    index_path = _write_index(output_dir, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote summary: {summary_path}")
    print(f"Wrote human index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
