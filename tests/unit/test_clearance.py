"""Unit tests for the functional-clearance critic integration."""

from __future__ import annotations

import math

import pytest

from scenesmith.scenebenchmark_critic import clearance_source


def _sit_record() -> dict:
    return {
        "kind": "nonarticulated",
        "clearance_type": "落座",
        "direction": "前",
        "depth_m": 0.6,
        "width_m": 0.75,
        "height_m": 1.3,
        "confidence": "high",
        "inherits_from_support": False,
    }


def _obj(oid: str, bmin, bmax, *, yaw=0.0, clearance=None, name=None) -> dict:
    meta: dict = {}
    if clearance is not None:
        meta["clearance"] = clearance
    return {
        "id": oid,
        "name": name or oid,
        "yaw_deg": yaw,
        "bbox_world": {"min": list(bmin), "max": list(bmax)},
        "metadata": meta,
    }


def _a_key_with_projectable_clearance() -> str:
    for key, raw in clearance_source._nonartic_index().items():
        cat = str(raw.get("cat") or "").lower()
        if cat in clearance_source._SUPPRESS_FLOOR_CLEARANCE_CATS:
            continue
        rec = clearance_source.get_clearance(key)
        if rec and float(rec.get("depth_m") or 0.0) > 0.0:
            return key
    raise AssertionError("test fixture requires at least one projectable clearance key")


def test_index_loaded():
    s = clearance_source.stats()
    assert clearance_source.available()
    assert s["nonarticulated"] > 0
    assert s["articulated"] > 0


def test_get_clearance_roundtrip_and_miss():
    # pull a real key from the index so the test is not hard-coded to a hash
    key = next(iter(clearance_source._nonartic_index()))
    rec = clearance_source.get_clearance(key)
    assert rec is not None
    assert rec["asset_id"] == key
    assert rec["kind"] in {"nonarticulated", "articulated"}
    assert clearance_source.get_clearance("not-a-real-asset-id") is None
    assert clearance_source.get_clearance(None) is None


def test_project_front_box_extends_toward_plus_y_at_yaw0():
    rec = _sit_record()
    bbox = {"min": [-0.25, -0.25, 0.0], "max": [0.25, 0.25, 0.9]}
    regions = clearance_source.project_keep_clear(rec, bbox, yaw_deg=0.0)
    assert len(regions) == 1
    r = regions[0]
    # facing = +Y at yaw 0 (SceneSmith pose convention): box sits above on Y axis
    assert r["min"][1] == pytest.approx(0.25)
    assert r["max"][1] == pytest.approx(0.85)  # 0.25 + 0.60 depth
    assert r["max"][2] == pytest.approx(1.3)  # height applied


def test_yaw_180_flips_front_to_minus_y():
    axis, sign = clearance_source._front_world_axis(180.0)
    assert axis == 1 and sign == -1  # facing +Y rotated 180 -> -Y


def test_ring_direction_reserves_four_sides():
    rec = _sit_record()
    rec.update({"clearance_type": "接近", "direction": "四周", "depth_m": 0.45})
    bbox = {"min": [-0.2, -0.2, 0.0], "max": [0.2, 0.2, 1.0]}
    regions = clearance_source.project_keep_clear(rec, bbox)
    assert len(regions) == 4
    assert {r["side"] for r in regions} == {"+x", "-x", "+y", "-y"}


def test_vertical_clearance_sits_above_footprint():
    rec = {
        "clearance_type": "上方站立",
        "direction": "上",
        "depth_m": 0.0,
        "height_m": 1.9,
        "inherits_from_support": False,
    }
    bbox = {"min": [0.0, 0.0, 0.0], "max": [1.0, 0.6, 0.02]}
    regions = clearance_source.project_keep_clear(rec, bbox)
    assert len(regions) == 1
    assert regions[0]["side"] == "above"
    assert regions[0]["min"][2] == pytest.approx(0.02)
    assert regions[0]["max"][2] == pytest.approx(0.02 + 1.9)


def test_inherited_item_reserves_nothing():
    rec = _sit_record()
    rec["inherits_from_support"] = True
    bbox = {"min": [0, 0, 0], "max": [0.1, 0.1, 0.3]}
    assert clearance_source.project_keep_clear(rec, bbox) == []


def test_intruding_object_fails_clearance_check():
    chair = _obj("chair", [-0.25, -0.25, 0.0], [0.25, 0.25, 0.9],
                 clearance=_sit_record(), name="armchair")
    table = _obj("table", [-0.3, 0.3, 0.0], [0.3, 0.7, 0.7])  # in front (+Y)
    checks = clearance_source.build_clearance_checks({"chair": chair, "table": table})
    assert len(checks) == 1
    result = clearance_source.evaluate_clearance(checks[0])
    assert result["label"] == "fail"
    assert result["primary_object"] == "chair"
    assert result["blocking_objects"] == ["table"]
    assert result["confidence"] == pytest.approx(0.9)  # high


def test_structural_floor_is_not_clearance_blocker():
    # 2026-07-07: Room structure overlaps most keep-clear AABBs at z=0, but it
    # is the support/constraint frame, not an object intruding into clearance.
    chair = _obj("chair", [-0.25, -0.25, 0.0], [0.25, 0.25, 0.9],
                 clearance=_sit_record(), name="armchair")
    floor = _obj("floor_living_room", [-5.0, -5.0, -0.02], [5.0, 5.0, 0.03])
    floor["category"] = "floor"
    floor["object_type"] = "floor"
    checks = clearance_source.build_clearance_checks({"chair": chair, "floor": floor})
    assert len(checks) == 1
    result = clearance_source.evaluate_clearance(checks[0])
    assert result["label"] == "pass"
    assert result["blocking_objects"] == []


def test_clear_layout_passes():
    chair = _obj("chair", [-0.25, -0.25, 0.0], [0.25, 0.25, 0.9],
                 clearance=_sit_record())
    far = _obj("far", [2, 2, 0], [3, 3, 1])
    checks = clearance_source.build_clearance_checks({"chair": chair, "far": far})
    result = clearance_source.evaluate_clearance(checks[0])
    assert result["label"] == "pass"
    assert result["blocking_objects"] == []


def test_gated_small_item_builds_no_check():
    rec = _sit_record()
    rec["inherits_from_support"] = True
    lamp = _obj("lamp", [0, 0, 0], [0.1, 0.1, 0.3], clearance=rec)
    assert clearance_source.build_clearance_checks({"lamp": lamp}) == []


def test_asset_id_resolves_hssd_mesh_id_first():
    # Real SceneSmith objects carry the HSSD hash under hssd_mesh_id, not asset_id.
    assert clearance_source.asset_id_from_metadata(
        {"hssd_mesh_id": "abc", "asset_id": "xyz"}
    ) == "abc"
    assert clearance_source.asset_id_from_metadata({"asset_id": "xyz"}) == "xyz"
    assert clearance_source.asset_id_from_metadata({"object_id": "oid"}) == "oid"
    assert clearance_source.asset_id_from_metadata({}) is None
    assert clearance_source.asset_id_from_metadata(None) is None


def test_get_clearance_for_metadata_uses_real_join_key():
    # A real key from the index, placed under hssd_mesh_id, must resolve.
    key = next(iter(clearance_source._nonartic_index()))
    rec = clearance_source.get_clearance_for_metadata({"hssd_mesh_id": key})
    assert rec is not None and rec["asset_id"] == key
    assert clearance_source.get_clearance_for_metadata({"unrelated": key}) is None


def test_build_clearance_checks_resolves_hssd_mesh_id_without_asset_annotation():
    # 2026-07-07: Regression for direct critic runs where asset_annotation is
    # disabled and generated HSSD objects only carry metadata.hssd_mesh_id.
    key = _a_key_with_projectable_clearance()
    obj = _obj("hssd_obj", [0, 0, 0], [1, 1, 1])
    obj["metadata"]["hssd_mesh_id"] = key
    checks = clearance_source.build_clearance_checks({"hssd_obj": obj})
    assert len(checks) == 1
    assert checks[0]["subject_id"] == "hssd_obj"


# --- asset-level placement-aware clearance policy ---------------------------


def _raw(**kw) -> dict:
    base = dict(
        type="接近", dir="四周", depth=0.45, width=0.7, height=1.8,
        conf="med", inherits=False, bbox=[1.0, 2.0, 1.0], cat="x",
    )
    base.update(kw)
    return base


def test_policy_suppresses_bed_ring_clearance():
    # A bed is anchored furniture (backs against walls, nightstands abut); its
    # raw four-side ring produces mostly false fails, so floor clearance is
    # suppressed entirely -> no keep-clear region.
    out = clearance_source._apply_asset_clearance_policy(_raw(cat="bed", dir="四周"))
    assert out["depth"] == 0.0
    assert out.get("_policy")
    bbox = {"min": [0.0, 0.0, 0.0], "max": [1.6, 2.0, 0.7]}
    rec = clearance_source.get_clearance(_a_key_with_cat("bed"))
    if rec is not None:  # only assert when a bed asset exists in the index
        assert clearance_source.project_keep_clear(rec, bbox, 0.0) == []


def test_policy_suppresses_wall_mounted_decor():
    out = clearance_source._apply_asset_clearance_policy(
        _raw(cat="wall_art", type="接近", dir="前", depth=0.45)
    )
    assert out["depth"] == 0.0
    assert out.get("_policy")


def test_policy_shrinks_seating_front_to_tuck_depth():
    # chair/sofa front points at the paired table -> shrink to a tuck gap and
    # force a single front side (no ring) so the table is not an intruder.
    chair = clearance_source._apply_asset_clearance_policy(
        _raw(cat="chair", type="落座", dir="前", depth=0.6)
    )
    assert chair["dir"] == "前"
    assert chair["depth"] == clearance_source._SEATING_FRONT_DEPTH_M
    swivel = clearance_source._apply_asset_clearance_policy(
        _raw(cat="swivel_chair", type="落座", dir="四周", depth=0.6)
    )
    assert swivel["dir"] == "前"  # ring dropped
    assert swivel["depth"] == clearance_source._SEATING_FRONT_DEPTH_M


def test_policy_leaves_free_standing_storage_untouched():
    # a dresser's front access clearance is a genuine layout constraint.
    out = clearance_source._apply_asset_clearance_policy(
        _raw(cat="dresser", type="接近", dir="前", depth=0.6)
    )
    assert out["depth"] == 0.6
    assert not out.get("_policy")


def _a_key_with_cat(cat: str):
    for key, na in clearance_source._nonartic_index().items():
        if (na.get("cat") or "") == cat:
            return key
    return "not-a-real-asset-id"


# --- functional-dependency partner exclusion -------------------------------


def test_family_normalization_handles_synset_space_and_instance():
    f = clearance_source._family
    assert f("coffee table") == "surface"        # funeval space form
    assert f("dining_table.n.01") == "surface"   # synset suffix
    assert f("swivel_chair") == "seat"
    assert f("office_chair_0") == "seat"          # instance suffix dropped
    assert f("double_bed") == "bed"
    assert f("wall mirror") == "wall_mirror"      # unmapped -> normalized self


def test_functional_partner_is_not_an_intrusion():
    # A chair whose functional partners include the "surface" family: a table in
    # its keep-clear is an intended adjacency, not a violation.
    rec = {
        "kind": "nonarticulated", "clearance_type": "落座", "direction": "前",
        "depth_m": 0.6, "height_m": 0.0, "inherits_from_support": False,
        "partner_families": ["surface"],
    }
    chair = _obj("chair", [-0.25, -0.25, 0.0], [0.25, 0.25, 0.9], yaw=0.0, clearance=rec)
    table = _obj("table", [-0.3, 0.26, 0.0], [0.3, 0.8, 0.75])  # in front (+Y)
    table["metadata"]["category"] = "dining_table"   # -> surface (a partner)
    checks = clearance_source.build_clearance_checks({"chair": chair, "table": table})
    label = {c["subject_id"]: c["clearance_result"]["label"] for c in checks}
    assert label["chair"] == "pass"  # partner excluded

    # A non-partner object in the same spot is still a real intrusion.
    wall = _obj("wall", [-1.0, 0.7, 0.0], [1.0, 0.8, 2.5])  # in front (+Y)
    wall["metadata"]["category"] = "wall"
    checks2 = clearance_source.build_clearance_checks({"chair": chair, "wall": wall})
    label2 = {c["subject_id"]: c["clearance_result"]["label"] for c in checks2}
    assert label2["chair"] == "fail"
