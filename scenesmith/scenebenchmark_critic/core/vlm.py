from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from scenesmith.agent_utils.vlm_service import VLMService
from scenesmith.utils.llm_json import parse_llm_json_object

OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True)
class StructuredAgentResult(Generic[OutputT]):
    output: OutputT


class StructuredVLMAgent(Generic[OutputT]):
    def __init__(
        self,
        config: Any,
        *,
        output_type: type[OutputT],
        system_prompt: str,
        name: str,
    ) -> None:
        self.config = config
        self.output_type = output_type
        self.system_prompt = system_prompt
        self.name = name

    def run_sync(self, prompt: str) -> StructuredAgentResult[OutputT]:
        service = VLMService(cfg=self.config)
        response_text = service.create_completion(
            model=_model_name(self.config),
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            reasoning_effort=_config_value(
                self.config, "reasoning_effort", default="low"
            ),
            verbosity=_config_value(self.config, "verbosity", default="low"),
            response_format={"type": "json_object"},
            vision_detail=_config_value(self.config, "vision_detail", default="auto"),
        )
        payload = parse_llm_json_object(response_text)
        return StructuredAgentResult(output=self.output_type.model_validate(payload))


def build_structured_agent(
    config: Any,
    *,
    output_type: type[OutputT],
    system_prompt: str,
    name: str = "scenebenchmark_structured_agent",
) -> StructuredVLMAgent[OutputT]:
    return StructuredVLMAgent(
        config,
        output_type=output_type,
        system_prompt=system_prompt,
        name=name,
    )


def _model_name(config: Any) -> str:
    for path in (
        ("provider", "model"),
        ("openai", "model"),
        ("asset_annotation", "model"),
    ):
        value = _nested_get(config, path)
        if value:
            return str(value)
    return os.environ.get("MODEL_NAME") or "gpt-4.1-mini"


def _config_value(config: Any, key: str, *, default: str) -> str:
    value = _nested_get(config, ("model_settings", key))
    if value is None:
        value = _nested_get(config, ("openai", key))
    if value is None:
        value = _nested_get(config, ("asset_annotation", key))
    return str(value if value is not None else default)


def _nested_get(config: Any, path: tuple[str, ...]) -> Any:
    current = config
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current
