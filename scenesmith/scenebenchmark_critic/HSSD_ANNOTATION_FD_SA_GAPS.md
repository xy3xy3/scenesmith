# HSSD Annotations for SceneBenchmark FD/SA

日期：2026-07-07  
原因：梳理如果 SceneBenchmark `functional_dependency` 和
`spatial_accessibility` 改为依赖 hssd-annotations，当前还缺哪些可消费字段，
以及需要协作者补齐的数据形态。

## 当前结论

`interaction_clearance` 已经直接接入 HSSD 标注；但
`functional_dependency`（FD）和 `spatial_accessibility`（SA）目前还没有自动
从 `asset_annotation_data/hssd_annotation_lookup.json.gz` hydrate 到
`SceneObject.metadata.functional_hints`。

当前 FD/SA 主要消费的是 case pack object 里的：

- `category` / `category_norm`
- `bbox_world` / `footprint_world` / `yaw_deg`
- `functional_hints.functional_categories`
- `functional_hints.candidate_affordances`
- `functional_hints.accessibility_policy`
- `functional_hints.scene_object_type`
- `functional_hints.category_group`
- `functional_hints.access_sides` / `front_hint` / `front_face`
- `functional_hints.attachment_dependencies`
- `functional_hints.orientation_dependencies`
- `functional_hints.explicit_target_relation`
- `metadata.functional_dependencies`
- `support_regions`

相关入口：

- `scenesmith/scenebenchmark_critic/adapter.py:487` builds
  `functional_hints` for each object.
- `scenesmith/scenebenchmark_critic/adapter.py:525` writes those hints into the
  case-pack object.
- `scenesmith/scenebenchmark_critic/adapter.py:1125` passes through metadata
  keys into `functional_hints`.
- `scenesmith/scenebenchmark_critic/adapter.py:1174` converts
  `functional_dependencies` metadata into scene geometry relations.
- `scenesmith/scenebenchmark_critic/checks.py:67` starts SA check generation.
- `scenesmith/scenebenchmark_critic/checks.py:95` starts FD check generation.
- `scenesmith/scenebenchmark_critic/vendor/scenebenchmark/critic/geometry.py:117`
  reads affordances from `functional_hints.functional_categories` or
  `candidate_affordances`.
- `scenesmith/scenebenchmark_critic/asset_annotation.py:731` shows the current
  VLM/mock asset annotation write-back format FD/SA already know how to consume.

## Current HSSD Lookup Status

The portable lookup exists and is self-contained for the light fields:

- `scenesmith/scenebenchmark_critic/asset_library_annotations.py:16` defines
  `asset_annotation_data/hssd_annotation_lookup.json.gz`.
- `scenesmith/scenebenchmark_critic/asset_library_annotations.py:264` loads the
  bundled record and adds optional external enrichment stubs.
- `scenesmith/scenebenchmark_critic/asset_library_annotations.py:356` exposes
  `get_hssd_asset_annotations(hssd_id)`.
- `scenesmith/scenebenchmark_critic/README.md:306` documents the bundled fields:
  `interaction_clearance`, `post_replacement`, `canonical_front`,
  `relation_priors`, `placement_dof`,
  `clearance_intrusion_whitelist_refs`, and `environment_anchors`.

Important: the current bundled lookup has `affordance_ref`, but does not inline
SceneBenchmark-ready `functional_categories` / `candidate_affordances`. External
affordance and operation-space layers are treated as optional enrichment in
`scenesmith/scenebenchmark_critic/asset_library_annotations.py:267`.

## Data Gaps To Ask The HSSD-Annotations Owner To Fill

### 1. SceneBenchmark-Normalized Affordance Labels

Needed by SA and FD semantic checks.

Current consumer:

- `scenesmith/scenebenchmark_critic/vendor/scenebenchmark/critic/geometry.py:117`
  reads only `functional_hints.functional_categories` or
  `functional_hints.candidate_affordances`.
- `scenesmith/scenebenchmark_critic/checks.py:191` creates SA checks only when
  affordances intersect `sittable`, `openable`, `supportable`, `sleepable`,
  `graspable`.

Need from hssd-annotations:

```json
{
  "functional_categories": ["sittable", "supportable"],
  "candidate_affordances": ["sittable", "supportable"],
  "affordance_confidence": 0.82,
  "affordance_source": "hssd_annotations"
}
```

Minimum vocabulary should match current SceneBenchmark labels:

- `sittable`
- `sleepable`
- `supportable`
- `openable`
- `containable`
- `toggleable`
- `graspable`

Without this inline normalized field, SA still falls back to local heuristic text
matching in `scenesmith/scenebenchmark_critic/adapter.py:876`, or to VLM/mock
asset annotation write-back in `scenesmith/scenebenchmark_critic/asset_annotation.py:740`.

### 2. Accessibility Policy And Subject Filtering

Needed by SA to decide whether an object should be checked at all.

Current consumer:

- `scenesmith/scenebenchmark_critic/checks.py:208` reads
  `functional_hints.accessibility_policy`.
- `scenesmith/scenebenchmark_critic/checks.py:216` excludes ceiling-mounted
  objects.
- `scenesmith/scenebenchmark_critic/checks.py:232` reads
  `functional_hints.scene_object_type`.
- `scenesmith/scenebenchmark_critic/checks.py:308` drops small/non-actionable
  objects using size, category, `category_group`, and affordances.

Need from hssd-annotations:

```json
{
  "accessibility_policy": "required",
  "scene_object_type": "furniture",
  "mobility_class": "semi_movable",
  "category_group": "seating",
  "benchmark_relevance": "functional"
}
```

Allowed `accessibility_policy` values:

- `required`
- `optional`
- `ignored`

Allowed `scene_object_type` values:

- `furniture`
- `wall_mounted`
- `ceiling_mounted`
- `manipuland`
- `unknown`

The bundled `week27_asset_policy` already contains some of these fields, but
many sample records still have `scene_object_type: "unknown"`. The collaborator
should make these fields complete and SceneBenchmark-normalized.

### 3. Front / Access Direction In SceneBenchmark Terms

Needed by SA interaction faces and by orientation-sensitive FD checks.

Current consumer:

- `scenesmith/scenebenchmark_critic/adapter.py:522` builds
  `interaction_faces`.
- `scenesmith/scenebenchmark_critic/adapter.py:947` maps `front_face` to
  `front_hint`.
- `scenesmith/scenebenchmark_critic/adapter.py:952` can derive `front_hint` from
  `access_direction` / `access_directions`.
- `scenesmith/scenebenchmark_critic/asset_annotation.py:743` writes
  `front_hint` / `front_face`.

Need from hssd-annotations:

```json
{
  "front_face": "front",
  "front_hint": "front",
  "access_sides": ["front"],
  "access_direction": "front",
  "asset_local_front_axis": [0.0, 1.0, 0.0],
  "front_confidence": 0.9
}
```

If only `canonical_front.asset_local_front_axis` is provided, we still need a
hydrator-side mapping into SceneBenchmark's side tokens (`front`, `back`,
`left`, `right`, `top`, `bottom`) or a ready-made `front_hint`.

### 4. Functional Relation Priors In Runtime-Friendly Form

Needed by FD to generate object-object checks without VLM.

Current consumer:

- `scenesmith/scenebenchmark_critic/adapter.py:1174` reads
  `metadata.functional_dependencies` or
  `metadata.functional_hints.functional_dependencies`.
- `scenesmith/scenebenchmark_critic/checks.py:95` creates FD checks from
  metadata relations in `scene_geometry.relations`.
- `scenesmith/scenebenchmark_critic/checks.py:380` creates FD checks from
  `functional_hints.explicit_target_relation`.
- `scenesmith/scenebenchmark_critic/checks.py:530` creates FD checks from
  `attachment_dependencies` and `orientation_dependencies`.

The current lookup has `relation_priors` and `environment_anchors`, but FD needs
a normalized form that can be resolved against actual scene instances.

Need from hssd-annotations:

```json
{
  "target_relation": ["dining_table", "desk"],
  "explicit_target_relation": ["dining_table", "desk"],
  "functional_dependencies": [
    {
      "relation_type": "used_with",
      "target_category": "dining_table",
      "target_kind": "object_category",
      "distance_range_m": [0.0, 1.2],
      "relative_facing": "front_faces_target",
      "confidence": 0.78,
      "reason": "chair is used with a table or desk"
    }
  ],
  "attachment_dependencies": [
    {
      "relation_type": "against",
      "target_kind": "environment_anchor",
      "environment_anchor": "wall",
      "distance_range_m": [0.0, 0.12],
      "confidence": 0.8
    }
  ],
  "orientation_dependencies": [
    {
      "relation_type": "front_faces",
      "target_category": "television_receiver",
      "confidence": 0.75
    }
  ]
}
```

Do not require concrete `target_ids` in the asset annotation; those are
scene-instance ids and must be resolved by SceneSmith at runtime. The asset
annotation should provide target categories, environment anchors, relation type,
distance range, facing/height constraints, and confidence.

### 5. Support / Interaction Surface Details

Needed by FD support checks and by SA target selection.

Current consumer:

- `scenesmith/scenebenchmark_critic/adapter.py:489` reads metadata
  `support_regions`.
- `scenesmith/scenebenchmark_critic/adapter.py:501` writes
  `support_region_summary`.
- `scenesmith/scenebenchmark_critic/vendor/scenebenchmark/metrics/functional_dependency/support_scoring.py:103`
  scores against `support_regions`.
- `scenesmith/scenebenchmark_critic/vendor/scenebenchmark/metrics/functional_dependency/support_scoring.py:186`
  reads `functional_hints.interaction_height_m`.
- `scenesmith/scenebenchmark_critic/vendor/scenebenchmark/critic/geometry.py:634`
  reads `functional_hints.interaction_surface_map`.

Need from hssd-annotations when available:

```json
{
  "interaction_surface_map": {
    "supportable": ["top"],
    "openable": ["front"]
  },
  "interaction_height_m": {
    "supportable": 0.74,
    "openable": 0.9
  },
  "support_regions": [
    {
      "region_id": "asset_top",
      "surface_type": "top",
      "local_min": [-0.5, -0.3, 0.72],
      "local_max": [0.5, 0.3, 0.78],
      "confidence": 0.8
    }
  ]
}
```

If support regions are too heavy to inline for all assets, at least provide
`interaction_surface_map` and `interaction_height_m` inline. Full local support
regions can remain optional enrichment.

### 6. Metric Relevance / Suppression

Needed to avoid over-checking decorative or irrelevant objects.

Current consumer:

- `scenesmith/scenebenchmark_critic/checks.py:260` reads
  `functional_hints.metric_relevance`.
- `scenesmith/scenebenchmark_critic/adapter.py:925` clears affordances when
  asset annotation marks an object as non-functional.
- `scenesmith/scenebenchmark_critic/asset_annotation.py:754` writes
  `benchmark_relevance`.

Need from hssd-annotations:

```json
{
  "benchmark_relevance": "functional",
  "metric_relevance": {
    "spatial_accessibility": 1.0,
    "functional_dependency": 0.7,
    "interaction_clearance": 0.8
  }
}
```

For decorative/wall-only assets, set `benchmark_relevance` to `decorative` or
`noise`, and set SA/FD relevance to `0.0` or mark
`accessibility_policy: "ignored"`.

## Runtime Work Still Needed In SceneSmith

These are not requests for the hssd-annotations owner; they are our integration
tasks.

1. Add an HSSD annotation hydrator before `adapter._functional_hints()`.
   It should resolve `metadata.hssd_mesh_id` / `metadata.asset_id`, call
   `get_hssd_asset_annotations()`, and merge normalized fields into
   `SceneObject.metadata.functional_hints`.

2. Map bundled HSSD fields to current consumer fields:

   - `week27_asset_policy.accessibility_policy` ->
     `functional_hints.accessibility_policy`
   - `week27_asset_policy.access_sides` -> `functional_hints.access_sides`
   - `week27_asset_policy.scene_object_type` ->
     `functional_hints.scene_object_type`
   - `canonical_front` -> `front_hint` / `front_face`
   - normalized affordance labels -> `functional_categories` and
     `candidate_affordances`
   - relation priors -> `target_relation`, `explicit_target_relation`,
     `attachment_dependencies`, `orientation_dependencies`, or
     `functional_dependencies`

3. Keep scene-instance resolution inside SceneSmith. HSSD annotations should not
   name concrete `target_ids`; SceneSmith should choose actual nearby targets
   from `scene_geometry.objects`.

## Bottom Line For The Collaborator

To remove VLM-on-the-fly dependency for HSSD assets, the highest-value missing
fields are:

1. Inline SceneBenchmark-normalized affordance labels.
2. Complete `scene_object_type`, `category_group`, `mobility_class`, and
   `accessibility_policy`.
3. Ready-to-consume `front_hint` / `access_sides` in SceneBenchmark side tokens.
4. Runtime-friendly relation/dependency priors with target categories and
   constraints, not concrete scene ids.
5. Inline support/interaction surface summaries, at least
   `interaction_surface_map` and `interaction_height_m`.
6. Metric relevance / suppression flags for decorative and non-functional assets.
