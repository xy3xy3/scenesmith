#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from agents.tracing import set_tracing_disabled
from omegaconf import OmegaConf, open_dict

from scenesmith.furniture_agents.tools.vision_tools import VisionTools
from scenesmith.agent_utils.scene_analyzer import FurnitureSelection
from scenesmith.experiments.indoor_scene_generation import _load_room_scene_state
from scenesmith.furniture_agents.stateful_furniture_agent import StatefulFurnitureAgent
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)
from scenesmith.utils.logging import ConsoleLogger, FileLoggingContext
from scenesmith.utils.omegaconf import register_resolvers


console_logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg(model: str) -> Any:
    register_resolvers()
    config_dir = _repo_root() / "configurations"
    root_cfg = OmegaConf.load(config_dir / "config.yaml")

    def load_group(group: str, name: str) -> Any:
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

    cfg = OmegaConf.merge(
        root_cfg,
        {
            "experiment": load_group("experiment", "indoor_scene_generation"),
            "floor_plan_agent": load_group("floor_plan_agent", "stateful_floor_plan_agent"),
            "furniture_agent": load_group("furniture_agent", "stateful_furniture_agent"),
            "wall_agent": load_group("wall_agent", "stateful_wall_agent"),
            "ceiling_agent": load_group("ceiling_agent", "stateful_ceiling_agent"),
            "manipuland_agent": load_group("manipuland_agent", "stateful_manipuland_agent"),
        },
    )

    if "defaults" in cfg:
        with open_dict(cfg):
            del cfg["defaults"]

    OmegaConf.resolve(cfg)
    with open_dict(cfg):
        cfg.openai.model = model
        cfg.openai.use_responses = False
        cfg.experiment.scenebenchmark_critic.enabled = False
    return cfg


def _agent_cfg(full_cfg: Any, agent_type: str) -> Any:
    if agent_type == "furniture":
        cfg = OmegaConf.create(OmegaConf.to_container(full_cfg.furniture_agent, resolve=True))
    elif agent_type == "manipuland":
        cfg = OmegaConf.create(
            OmegaConf.to_container(full_cfg.manipuland_agent, resolve=True)
        )
    else:
        raise ValueError(f"Unsupported agent_type: {agent_type}")

    with open_dict(cfg):
        cfg.openai.service_tier = full_cfg.openai.service_tier
        cfg.openai.base_url = full_cfg.openai.base_url
        cfg.openai.use_responses = full_cfg.openai.use_responses
        cfg.openai.model = full_cfg.openai.model
    return cfg


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_tree_png_yaml(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".yaml", ".md", ".json"}:
            shutil.copy2(path, dst_dir / path.name)


def _write_critic_artifacts(
    *,
    output_dir: Path,
    export_dir: Path,
    render_dir: Path | None,
    scene_state_path: Path,
    agent: Any,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    if render_dir is not None:
        _copy_tree_png_yaml(render_dir, export_dir)
    _copy_if_exists(scene_state_path, export_dir / "scene_state.json")

    previous_scores = getattr(agent, "previous_scores", None)
    if previous_scores is not None:
        critique_text = getattr(previous_scores, "critique", "")
        if critique_text:
            (export_dir / "critic_text.md").write_text(
                critique_text,
                encoding="utf-8",
            )


async def _run_furniture_once(
    *,
    cfg: Any,
    scene_state_path: Path,
    output_dir: Path,
    placement_style: str,
) -> None:
    logger = ConsoleLogger(output_dir)
    room_id = scene_state_path.parents[2].name.removeprefix("room_")

    with logger.room_context(room_id):
        scene = _load_room_scene_state(scene_state_path)
        agent = StatefulFurnitureAgent(cfg=cfg, logger=logger)
        try:
            agent.scene = scene
            agent.placement_style = placement_style

            # Force one deterministic render set up front so we always have the
            # exact images used for comparison, even if the model skips tools.
            render_dir = VisionTools(
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

            export_dir = output_dir / "exports" / "furniture"
            last_render_dir = agent.rendering_manager.last_render_dir or render_dir
            _write_critic_artifacts(
                output_dir=output_dir,
                export_dir=export_dir,
                render_dir=last_render_dir,
                scene_state_path=scene_state_path,
                agent=agent,
            )

            summary = {
                "agent_type": "furniture",
                "model": str(cfg.openai.model),
                "scene_state_path": str(scene_state_path),
                "placement_style": placement_style,
                "render_dir": str(last_render_dir),
                "export_dir": str(export_dir),
            }
            (output_dir / "run_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        finally:
            agent.cleanup()


def _build_default_selection(furniture_id: str) -> FurnitureSelection:
    return FurnitureSelection(
        furniture_id=furniture_id,
        suggested_items="Evaluate the current manipuland arrangement exactly as-is.",
        prompt_constraints="Do not assume missing requested items unless visible evidence supports it.",
        style_notes="Focus on the existing arrangement only.",
        context_furniture_ids=[],
    )


async def _run_manipuland_once(
    *,
    cfg: Any,
    scene_state_path: Path,
    output_dir: Path,
    furniture_id: str,
    placement_style: str,
) -> None:
    logger = ConsoleLogger(output_dir)
    room_id = scene_state_path.parents[2].name.removeprefix("room_")

    with logger.room_context(room_id):
        scene = _load_room_scene_state(scene_state_path)
        furniture = scene.get_object(furniture_id)
        if furniture is None:
            raise ValueError(f"Furniture id not found in scene: {furniture_id}")

        agent = StatefulManipulandAgent(cfg=cfg, logger=logger)
        try:
            agent.scene = scene
            agent.placement_style = placement_style
            agent.current_furniture_id = furniture_id
            agent.current_furniture_selection = _build_default_selection(furniture_id)
            agent._initialize_checkpoint_state()

            furniture_description = (
                furniture.description or furniture.name or str(furniture.object_id)
            )
            agent._setup_furniture_agents(
                furniture_id=furniture_id,
                furniture_description=furniture_description,
            )

            await agent._request_critique_impl(update_checkpoint=False)

            export_dir = output_dir / "exports" / f"manipuland_{furniture_id}"
            last_render_dir = agent.rendering_manager.last_render_dir
            _write_critic_artifacts(
                output_dir=output_dir,
                export_dir=export_dir,
                render_dir=last_render_dir,
                scene_state_path=scene_state_path,
                agent=agent,
            )

            summary = {
                "agent_type": "manipuland",
                "model": str(cfg.openai.model),
                "scene_state_path": str(scene_state_path),
                "placement_style": placement_style,
                "furniture_id": furniture_id,
                "render_dir": str(last_render_dir),
                "export_dir": str(export_dir),
            }
            (output_dir / "run_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        finally:
            agent.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SceneSmith's original LLM critic exactly once on an existing scene_state.json, "
            "without SceneBenchmark critic."
        )
    )
    parser.add_argument("--scene-state", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--agent-type",
        choices=["furniture", "manipuland"],
        default="furniture",
    )
    parser.add_argument(
        "--placement-style",
        choices=["natural", "perfect"],
        default="natural",
    )
    parser.add_argument(
        "--furniture-id",
        help="Required for manipuland critic runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    set_tracing_disabled(True)

    scene_state_path = args.scene_state.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not scene_state_path.exists():
        raise FileNotFoundError(f"scene_state not found: {scene_state_path}")

    if args.agent_type == "manipuland" and not args.furniture_id:
        raise ValueError("--furniture-id is required when --agent-type manipuland")

    full_cfg = _load_cfg(args.model)
    cfg = _agent_cfg(full_cfg, args.agent_type)

    experiment_log_path = output_dir / "experiment.log"
    with FileLoggingContext(log_file_path=experiment_log_path, suppress_stdout=False):
        console_logger.info("Output directory: %s", output_dir)
        console_logger.info("Scene state: %s", scene_state_path)
        console_logger.info("Agent type: %s", args.agent_type)
        console_logger.info("Model: %s", args.model)
        console_logger.info("Placement style: %s", args.placement_style)

        if args.agent_type == "furniture":
            asyncio.run(
                _run_furniture_once(
                    cfg=cfg,
                    scene_state_path=scene_state_path,
                    output_dir=output_dir,
                    placement_style=args.placement_style,
                )
            )
        else:
            asyncio.run(
                _run_manipuland_once(
                    cfg=cfg,
                    scene_state_path=scene_state_path,
                    output_dir=output_dir,
                    furniture_id=args.furniture_id,
                    placement_style=args.placement_style,
                )
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
