from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    type: str = "openai"
    model: str = "gpt-4.1-mini"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None


class AgentConfig(BaseModel):
    name: str = "scenebenchmark_vlm_critic"
    retries: int = 1
    system_prompt_file: str = "src/critic/prompts/vlm_critic_system_prompt.txt"
    append_note: str | None = None


class AccessibilityAgentProfile(BaseModel):
    id: str = "default"
    clearance_width_m: float = 0.50
    reach_radius_m: float = 0.75
    arm_origin_height_m: float = 1.10
    locomotion_mode: str = "walk"
    crouch_factor: float | None = None
    eye_height_m: float | None = None


class RunConfig(BaseModel):
    include_render_overview_image: bool = True
    include_all_render_overview_images: bool = True
    include_render_check_image: bool = True
    include_dependency_focus_images: bool = True
    include_reference_overview_image: bool = False
    include_reference_marked_image: bool = False
    prefer_renderer_mode: str = "blender_rebuild"
    metrics: list[str] | None = None
    max_checks: int | None = None
    max_workers: int = 1
    stop_on_error: bool = False
    accessibility_grid_resolution_m: float = 0.05
    accessibility_agent_width_m: float = 0.50
    accessibility_obstacle_height_threshold_m: float = 1.8
    accessibility_access_zone_depth_m: float = 0.55
    accessibility_pass_ratio: float = 0.60
    accessibility_degraded_ratio: float = 0.25
    accessibility_agent_profiles: list[AccessibilityAgentProfile] | None = None
    fd_relation_proposer_mode: str = "template"
    max_fd_relation_proposals: int = 8


class OutputConfig(BaseModel):
    results_file: str = "vlm_results.json"
    save_request_payloads: bool = True


class SettingsConfig(BaseModel):
    temperature: float = 0.1
    max_tokens: int | None = 800
    timeout: float = 120.0


class VLMConfig(BaseModel):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    model_settings: SettingsConfig = Field(default_factory=SettingsConfig)


def load_vlm_config(path: Path) -> VLMConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return VLMConfig.model_validate(data)


def resolve_repo_path(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def model_settings_dict(config: SettingsConfig) -> dict[str, Any]:
    return config.model_dump(exclude_none=True)
