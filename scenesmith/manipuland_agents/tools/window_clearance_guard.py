"""Pre-placement guard for manipulands near window clearance zones."""

from __future__ import annotations

from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject


def window_clearance_placement_error(
    *, scene: RoomScene, obj: SceneObject, margin_m: float = 0.005
) -> str | None:
    """Return an error if placing ``obj`` would block a window clearance zone."""
    room_geom = scene.room_geometry
    if room_geom is None or not getattr(room_geom, "openings", None):
        return None
    if obj.object_type in (ObjectType.WALL, ObjectType.FLOOR):
        return None
    if obj.metadata.get("asset_source") == "thin_covering":
        return None
    world_bounds = obj.compute_world_bounds()
    if world_bounds is None:
        return None
    obj_min, obj_max = world_bounds
    object_top = float(obj_max[2])

    for opening in room_geom.openings:
        opening_type = getattr(opening, "opening_type", None)
        if hasattr(opening_type, "value"):
            opening_type = opening_type.value
        if str(opening_type) != "window":
            continue
        zone_min = getattr(opening, "clearance_bbox_min", None)
        zone_max = getattr(opening, "clearance_bbox_max", None)
        sill_height = float(getattr(opening, "sill_height", 0.0) or 0.0)
        if zone_min is None or zone_max is None:
            continue
        if object_top <= sill_height + margin_m:
            continue
        intersects_xy = (
            obj_min[0] < zone_max[0]
            and obj_max[0] > zone_min[0]
            and obj_min[1] < zone_max[1]
            and obj_max[1] > zone_min[1]
        )
        if not intersects_xy:
            continue
        # 2026-07-09 修改原因：低窗台旁的 sideboard 上反复生成花瓶/碗等高物体，
        # 随后又被 window_access critic 删除；在工具层提前拒绝明显遮窗放置。
        available_height = max(0.0, sill_height - float(obj_min[2]))
        return (
            f"Placement would block {opening.opening_id}: object top "
            f"{object_top:.3f}m exceeds window sill {sill_height:.3f}m while "
            "intersecting the window clearance zone. Choose a lower-profile "
            f"object no taller than about {available_height:.3f}m here, or use "
            "a different surface/position away from the window."
        )
    return None
