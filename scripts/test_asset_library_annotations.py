#!/usr/bin/env python3
"""Standalone test for the portable asset-library annotation lookup.

Runs without pydrake/bpy (loads the module directly), so it can be executed on
a fresh clone. Verifies the bundled lookup is self-sufficient and returns the
merged annotation families for the yz asset library.

Run:  python scripts/test_asset_library_annotations.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scenesmith"
    / "scenebenchmark_critic"
    / "asset_library_annotations.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("asset_library_annotations", MODULE_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    m = load_module()
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        if cond:
            print(f"  PASS {name}")
        else:
            print(f"  FAIL {name}")
            failures.append(name)

    # 1) default store loads bundled lookup
    store = m.AssetLibraryAnnotationStore()
    total = len(store._load())
    check("bundled_lookup_count==10963", total == 10963)

    # 2) a static seating asset returns verified front axis (orientation fix)
    chair = store.require("005b673d5ded782d54be17969159917dfc9ae0f2")
    cf = chair.get("canonical_front") or {}
    check("chair_has_front_axis", cf.get("asset_local_front_axis") is not None)
    check("chair_front_verified", cf.get("validation_status") == "geometry_axis_verified")
    check("chair_has_orientation_axis", cf.get("canonical_orientation_axis") is not None)
    check("chair_orientation_semantic",
          cf.get("canonical_orientation_is_semantic_front") is True)
    chair_hints = chair.get("scenebenchmark_functional_hints") or {}
    check("chair_fd_sa_hints", bool(chair_hints))
    check("chair_scenebenchmark_sittable",
          "sittable" in chair_hints.get("functional_categories", []))
    check("chair_scenebenchmark_access_required",
          chair_hints.get("accessibility_policy") == "required")
    check("chair_scenebenchmark_scene_type",
          chair_hints.get("scene_object_type") == "furniture")
    check("chair_scenebenchmark_front_hint",
          chair_hints.get("front_hint") == "front")
    check("chair_scenebenchmark_fd_targets",
          "table" in chair_hints.get("target_relation", []))

    # 2b) every asset has a placement orientation axis; fallback axes are
    # explicitly marked non-semantic.
    fallback = store.require("000bbe302b53fd2904e7ae92e1516a18f29de02d")
    fcf = fallback.get("canonical_front") or {}
    check("fallback_has_orientation_axis",
          fcf.get("canonical_orientation_axis") is not None)
    check("fallback_not_semantic_front",
          fcf.get("canonical_orientation_is_semantic_front") is False)
    fallback_hints = fallback.get("scenebenchmark_functional_hints") or {}
    check("fallback_scenebenchmark_scene_type",
          fallback_hints.get("scene_object_type") == "manipuland")
    check("fallback_scenebenchmark_fd_metric",
          (fallback_hints.get("metric_relevance") or {}).get("functional_dependency") == 1.0)

    # 3) PM replacement realization present in post_replacement
    pm = store.require("0022fa6ee5a44330e765488919803155b1a9e88c")
    pr = pm.get("post_replacement") or {}
    check("pm_realization", pr.get("realization_kind") == "pm_replacement")
    check("pm_match_id", pr.get("match_id") == "102177")

    # 3b) articulated storage/media assets expose FD/SA-normalized open/support
    # affordances and runtime-friendly relation targets.
    tv_stand = store.require("0074b6bf5758fd7186157fd3a8d53afe3e200cba")
    tv_hints = tv_stand.get("scenebenchmark_functional_hints") or {}
    check("tv_stand_openable_supportable",
          {"openable", "supportable"}.issubset(tv_hints.get("functional_categories", [])))
    check("tv_stand_storage_surface",
          tv_hints.get("category_group") == "storage_surface")
    check("tv_stand_relation_target",
          "television_receiver" in tv_hints.get("target_relation", []))
    check("tv_stand_openable_height",
          (tv_hints.get("interaction_height_m") or {}).get("openable") == 0.9)

    # 4) inline interaction_clearance is authoritative (portable)
    check("inline_clearance", bool(chair.get("interaction_clearance")))
    check("inline_keep_clear", chair["interaction_clearance"].get("has_keep_clear") is True)

    # 5) PORTABILITY: works with all external roots absent
    portable = m.AssetLibraryAnnotationStore(
        unified_affordance_dir="/nonexistent/a",
        operation_space_dir="/nonexistent/b",
        nonartic_clearance_v2_path="/nonexistent/c.json",
        official_combined_clearance_path="/nonexistent/d.json",
        hssd_articulation_clearance_run_path="/nonexistent/e.json",
        hssd_clearance_voxel_results_path="/nonexistent/f.json",
    )
    pr2 = portable.require("005b673d5ded782d54be17969159917dfc9ae0f2")
    check("portable_returns_record", pr2.get("category") is not None)
    check("portable_keeps_inline_clearance",
          pr2["interaction_clearance"].get("has_keep_clear") is True)
    check("portable_keeps_front_axis",
          (pr2.get("canonical_front") or {}).get("asset_local_front_axis") is not None)

    # 6) category search works
    hits = portable.search_category("chair", limit=5)
    check("search_returns_hits", len(hits) == 5)

    # 7) all bundled assets carry the SceneBenchmark FD/SA normalized envelope.
    all_records = portable._load()
    check("all_records_have_fd_sa",
          all(bool(r.get("scenebenchmark_fd_sa")) for r in all_records.values()))
    check("all_records_have_functional_hints",
          all(bool(r.get("scenebenchmark_functional_hints")) for r in all_records.values()))
    check("all_records_have_scene_object_type",
          all((r.get("scenebenchmark_functional_hints") or {}).get("scene_object_type")
              for r in all_records.values()))
    check("all_records_have_access_policy",
          all((r.get("scenebenchmark_functional_hints") or {}).get("accessibility_policy")
              for r in all_records.values()))

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
