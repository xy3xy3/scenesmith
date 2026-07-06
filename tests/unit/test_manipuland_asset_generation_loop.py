import json

from scenesmith.agent_utils.asset_manager import (
    AssetGenerationRequest,
    AssetGenerationResult as ManagerAssetGenerationResult,
    FailedAsset,
)
from scenesmith.agent_utils.room import ObjectType
from scenesmith.manipuland_agents.tools.manipuland_tools import ManipulandTools


class _FailingAssetManager:
    def __init__(self) -> None:
        self.calls = 0

    def generate_assets(
        self, request: AssetGenerationRequest
    ) -> ManagerAssetGenerationResult:
        self.calls += 1
        return ManagerAssetGenerationResult(
            successful_assets=[],
            failed_assets=[
                FailedAsset(
                    index=0,
                    description=request.object_descriptions[0],
                    error_message="All generation/retrieval attempts exhausted",
                )
            ],
        )


def _make_tools() -> tuple[ManipulandTools, _FailingAssetManager]:
    manager = _FailingAssetManager()
    tools = object.__new__(ManipulandTools)
    tools.asset_manager = manager
    tools._asset_generation_loop_detection_enabled = True
    tools._asset_generation_repeat_limit = 2
    tools._asset_generation_failure_counts = {}
    tools._asset_generation_failure_messages = {}
    return tools, manager


def test_repeated_failed_asset_generation_is_blocked() -> None:
    tools, manager = _make_tools()
    request = AssetGenerationRequest(
        object_descriptions=["small square stone tile"],
        short_names=["stone_tile"],
        object_type=ObjectType.MANIPULAND,
        desired_dimensions=[[0.07, 0.07, 0.03]],
        style_context=None,
        scene_id="room_dining_room",
    )

    first = json.loads(tools._generate_assets_impl(request))
    second = json.loads(tools._generate_assets_impl(request))
    third = json.loads(tools._generate_assets_impl(request))

    assert first["success"] is False
    assert second["success"] is False
    assert third["success"] is False
    assert manager.calls == 2
    assert third["message"].startswith("Blocked repeated asset generation loop")
    assert "Do not request this exact asset again" in third["failures"]
