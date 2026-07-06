from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .params import RenderParams
    from .renderer import BlenderRenderer
    from .server_app import BlenderRenderApp
    from .server_manager import BlenderServer

__all__ = ["RenderParams", "BlenderRenderer", "BlenderRenderApp", "BlenderServer"]


def __getattr__(name: str) -> Any:
    """Lazily import Blender modules so lightweight server utilities avoid bpy."""
    if name == "RenderParams":
        from .params import RenderParams

        return RenderParams
    if name == "BlenderRenderer":
        from .renderer import BlenderRenderer

        return BlenderRenderer
    if name == "BlenderRenderApp":
        from .server_app import BlenderRenderApp

        return BlenderRenderApp
    if name == "BlenderServer":
        from .server_manager import BlenderServer

        return BlenderServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
