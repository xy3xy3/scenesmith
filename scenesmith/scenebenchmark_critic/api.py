"""Public API for embedding the SceneBenchmark critic in SceneSmith."""

from __future__ import annotations

import logging

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from scenesmith.agent_utils.house import HouseScene
from scenesmith.agent_utils.room import ObjectType, RoomScene
from scenesmith.scenebenchmark_critic.adapter import (
    house_scene_to_case_pack,
    room_scene_to_case_pack,
)
from scenesmith.scenebenchmark_critic.asset_annotation import annotate_room_scene
from scenesmith.scenebenchmark_critic.config import CriticConfig, critic_config_from_any
from scenesmith.scenebenchmark_critic.dining_place_setting_alignment import (
    evaluate_dining_place_setting_alignment,
)
from scenesmith.scenebenchmark_critic.dining_seat_distribution import (
    evaluate_dining_seat_distribution,
)
from scenesmith.scenebenchmark_critic.manipuland_completeness import (
    evaluate_manipuland_completeness,
)
from scenesmith.scenebenchmark_critic.media_support_alignment import (
    evaluate_media_support_alignment,
)
from scenesmith.scenebenchmark_critic.room_center_alignment import (
    evaluate_room_center_alignment,
)
from scenesmith.scenebenchmark_critic.orientation_contracts import (
    stabilize_orientation_contracts,
)
from scenesmith.scenebenchmark_critic.reports import (
    build_evaluation_payload,
    format_prompt_context as _format_prompt_context,
    write_report,
)
from scenesmith.scenebenchmark_critic.vendor.rules import run_case_pack_checks

if TYPE_CHECKING:
    from scenesmith.agent_utils.blender.server_manager import BlenderServer

console_logger = logging.getLogger(__name__)


def evaluate_room_scene(
    scene: RoomScene,
    *,
    config: CriticConfig | Any | None = None,
    stage: str = "adhoc",
    raw_config: Any | None = None,
    annotate_assets: bool = True,
    blender_server: "BlenderServer | None" = None,
) -> dict[str, Any]:
    critic_config = _coerce_config(config)
    if annotate_assets:
        annotate_room_scene(
            scene,
            output_dir=scene.scene_dir,
            config=critic_config,
            raw_config=raw_config or config,
            # 2026-07-07: Forward BlenderServer so SceneBenchmark asset annotation
            # can render evidence images instead of silently falling back to [].
            blender_server=blender_server,
            stage=stage,
        )
    case_pack = room_scene_to_case_pack(
        scene, stage=stage, metrics=list(critic_config.metrics)
    )
    stabilize_orientation_contracts(
        case_pack,
        scene,
        critic_config,
        stage=stage,
    )
    results = run_case_pack_checks(case_pack, config=critic_config)
    # 2026-07-14 修改原因：wall-mounted TV 即使避开窗口，也可能被迫偏离 TV
    # stand；将“TV 必须位于 TV stand 上方”作为通用 functional dependency，
    # 让 wall critic 能给出精确、可执行的窗口释放和 TV 居中反馈。
    if "functional_dependency" in critic_config.metrics:
        results.extend(evaluate_media_support_alignment(case_pack))
        # 2026-07-15 修改原因：家具 critic 只检查桌椅之间的关系，无法发现
        # prompt 明确要求“in the center”的主家具被局部 clearance 修复推离房间中心。
        # 增加通用 room-center functional dependency，约束中心锚点并给出成组修复建议。
        results.extend(evaluate_room_center_alignment(case_pack))
    # 2026-07-13 修改原因：dining_set 只约束椅子朝桌，无法发现单椅偏离桌边
    # 中心或同边多椅挤在一侧；追加按桌局部坐标计算的通用座椅分布检查。
    results.extend(evaluate_dining_seat_distribution(case_pack))
    # 2026-07-09 修改原因：餐桌等成组 tabletop manipulands 可能在物理后处理
    # 后被删除；规则报告需要直接暴露必需小物缺失，而不只看几何可达性。
    results.extend(evaluate_manipuland_completeness(case_pack))
    # 2026-07-13 修改原因：库存完整不代表餐位可用；餐盘及配套餐具还必须
    # 与最近离散座椅一对一对应，并位于该座椅正前方。
    results.extend(evaluate_dining_place_setting_alignment(case_pack))
    return build_evaluation_payload(
        case_pack=case_pack,
        results=results,
        stage=stage,
        scope=f"room:{scene.room_id}",
        config=critic_config,
    )


def evaluate_house_scene(
    house: HouseScene,
    *,
    config: CriticConfig | Any | None = None,
    stage: str = "adhoc",
    include_object_types: list[ObjectType] | tuple[ObjectType, ...] | None = None,
) -> dict[str, Any]:
    critic_config = _coerce_config(config)
    case_pack = house_scene_to_case_pack(
        house,
        stage=stage,
        metrics=list(critic_config.metrics),
        include_object_types=include_object_types,
    )
    results = run_case_pack_checks(case_pack, config=critic_config)
    return build_evaluation_payload(
        case_pack=case_pack,
        results=results,
        stage=stage,
        scope="house",
        config=critic_config,
    )


def write_room_stage_report(
    scene: RoomScene,
    output_dir: Path,
    *,
    config: CriticConfig | Any | None = None,
    stage: str,
    raw_config: Any | None = None,
    blender_server: "BlenderServer | None" = None,
) -> dict[str, Any] | None:
    critic_config = _coerce_config(config)
    if not critic_config.enabled or not critic_config.room_stage_enabled(stage):
        return None
    annotate_room_scene(
        scene,
        output_dir=output_dir,
        config=critic_config,
        raw_config=raw_config or config,
        # 2026-07-07: Forward BlenderServer for report-time asset renders too.
        blender_server=blender_server,
        stage=stage,
    )
    payload = evaluate_room_scene(
        scene,
        config=critic_config,
        raw_config=raw_config or config,
        stage=stage,
        annotate_assets=False,
    )
    write_report(output_dir, payload)
    console_logger.info("SceneBenchmark critic report saved to %s", output_dir)
    return payload


def write_house_stage_report(
    house: HouseScene,
    output_dir: Path,
    *,
    config: CriticConfig | Any | None = None,
    stage: str,
    include_object_types: list[ObjectType] | tuple[ObjectType, ...] | None = None,
) -> dict[str, Any] | None:
    critic_config = _coerce_config(config)
    if not critic_config.enabled or not critic_config.house_stage_enabled(stage):
        return None
    payload = evaluate_house_scene(
        house,
        config=critic_config,
        stage=stage,
        include_object_types=include_object_types,
    )
    write_report(output_dir, payload)
    console_logger.info("SceneBenchmark critic report saved to %s", output_dir)
    return payload


def format_prompt_context(
    payload: dict[str, Any], *, max_issues: int | None = None
) -> str:
    if max_issues is None:
        max_issues = 8
    return _format_prompt_context(payload, max_issues=max_issues)


def _coerce_config(config: CriticConfig | Any | None) -> CriticConfig:
    if isinstance(config, CriticConfig):
        return config
    if config is None:
        return CriticConfig()
    return critic_config_from_any(config)
