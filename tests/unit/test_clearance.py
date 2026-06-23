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


def test_project_front_box_extends_toward_minus_y_at_yaw0():
    rec = _sit_record()
    bbox = {"min": [-0.25, -0.25, 0.0], "max": [0.25, 0.25, 0.9]}
    regions = clearance_source.project_keep_clear(rec, bbox, yaw_deg=0.0)
    assert len(regions) == 1
    r = regions[0]
    # front = -Y at yaw 0: box sits below the object on the Y axis
    assert r["min"][1] == pytest.approx(-0.85)  # -0.25 - 0.60 depth
    assert r["max"][1] == pytest.approx(-0.25)
    assert r["max"][2] == pytest.approx(1.3)  # height applied


def test_yaw_180_flips_front_to_plus_y():
    axis, sign = clearance_source._front_world_axis(180.0)
    assert axis == 1 and sign == 1  # front -Y rotated 180 -> +Y


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
    table = _obj("table", [-0.3, -0.7, 0.0], [0.3, -0.3, 0.7])  # in front (-Y)
    checks = clearance_source.build_clearance_checks({"chair": chair, "table": table})
    assert len(checks) == 1
    result = clearance_source.evaluate_clearance(checks[0])
    assert result["label"] == "fail"
    assert result["primary_object"] == "chair"
    assert result["blocking_objects"] == ["table"]
    assert result["confidence"] == pytest.approx(0.9)  # high


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
