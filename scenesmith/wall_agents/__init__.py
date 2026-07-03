"""Wall-mounted object placement agents."""

from typing import Any

__all__ = ["BaseWallAgent", "StatefulWallAgent"]


def __getattr__(name: str) -> Any:
    """Lazily import wall-agent classes so light helpers stay importable in tests."""
    if name == "BaseWallAgent":
        from scenesmith.wall_agents.base_wall_agent import BaseWallAgent

        return BaseWallAgent
    if name == "StatefulWallAgent":
        from scenesmith.wall_agents.stateful_wall_agent import StatefulWallAgent

        return StatefulWallAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
