# HSSD FD/SA Annotation Gap Response

日期：2026-07-07
分支：`yz`
提交：`0cf4377 feat(critic): hydrate HSSD FD SA annotations`

## 已处理

已根据 `HSSD_ANNOTATION_FD_SA_GAPS.md` 在 `yz` 分支补齐 HSSD lookup 对
SceneBenchmark `functional_dependency` 和 `spatial_accessibility` 的可消费字段。

现在 `asset_annotation_data/hssd_annotation_lookup.json.gz` 中 10,963 条 HSSD
记录均包含：

- `scenebenchmark_fd_sa`
- `scenebenchmark_functional_hints`

其中 `scenebenchmark_functional_hints` 已内联以下字段：

- `functional_categories`
- `candidate_affordances`
- `accessibility_policy`
- `scene_object_type`
- `mobility_class`
- `category_group`
- `front_hint` / `front_face`
- `access_sides`
- `target_relation` / `explicit_target_relation`
- `functional_dependencies`
- `attachment_dependencies`
- `orientation_dependencies`
- `interaction_surface_map`
- `interaction_height_m`
- `metric_relevance`

## Runtime 接入

`adapter.py` 已增加 HSSD annotation hydrator。

当 SceneSmith 对象 metadata 中存在以下任一字段时：

- `hssd_mesh_id`
- `asset_id`
- `object_id`

adapter 会查 `get_hssd_asset_annotations()`，并把
`scenebenchmark_functional_hints` 合并进 case-pack object 的
`functional_hints`。显式 scene metadata 仍优先于 HSSD 默认标注。

## 验证

已运行：

```bash
python -m py_compile scenesmith/scenebenchmark_critic/asset_library_annotations.py scenesmith/scenebenchmark_critic/adapter.py
python scripts/test_asset_library_annotations.py
```

结果：`ALL PASS`

额外真实查询样例：

`0074b6bf5758fd7186157fd3a8d53afe3e200cba`，category `tv stand`

返回：

```json
{
  "functional_categories": ["containable", "openable", "supportable"],
  "scene_object_type": "furniture",
  "category_group": "storage_surface",
  "accessibility_policy": "required",
  "front_hint": "front",
  "access_sides": ["front", "top"],
  "target_relation": [
    "television_receiver",
    "video_game_console",
    "dvd_player",
    "loudspeaker"
  ],
  "interaction_height_m": {
    "containable": 0.9,
    "openable": 0.9,
    "supportable": 0.8
  },
  "metric_relevance": {
    "functional_dependency": 1.0,
    "interaction_clearance": 0.9,
    "spatial_accessibility": 1.0
  }
}
```

## 注意

资产标注只提供 target category / environment anchor / constraint，不写具体
scene instance id。具体对象解析仍保留在 SceneSmith runtime 中完成。
