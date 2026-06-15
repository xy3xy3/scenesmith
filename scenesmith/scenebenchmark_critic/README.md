# SceneBenchmark Critic for SceneSmith

This package embeds the SceneBenchmark rule critic subset directly inside
SceneSmith. It is designed for SceneSmith developers who already have
`RoomScene` or `HouseScene` objects and want lightweight geometry feedback
without adding the full SceneBenchmark runtime or a VLM dependency.

## What It Checks

- `spatial_accessibility`: whether objects with functional affordances such as
  sittable, sleepable, openable, or supportable are spatially usable.
- `functional_dependency`: whether placed objects have a valid supporting
  surface relationship, for example a mug on a nightstand surface.

The v1 adapter derives affordance hints from SceneSmith object names,
descriptions, metadata, support surfaces, and `placement_info`.

## Enable During Generation

The critic is disabled by default. Enable it with Hydra overrides:

```bash
uv run python main.py \
  +name=critic_demo \
  "experiment.tasks=[generate_scenes,evaluate_scenes]" \
  experiment.scenebenchmark_critic.enabled=true \
  "experiment.prompts=['A bedroom with a nightstand and a mug.','A classroom with desks and chairs.','A living room with a sofa, rug, plants, and a mug.']"
```

Online generation requires the same LLM/API credentials as the rest of
SceneSmith. Reports are written next to the generated SceneSmith checkpoints:

- `room_*/scene_states/scene_after_furniture/scenebenchmark_critic.json|md`
- `room_*/scene_states/final_scene/scenebenchmark_critic.json|md`
- `combined_house_after_furniture/scenebenchmark_critic.json|md`
- `combined_house/scenebenchmark_critic.json|md`

## Re-evaluate Existing Outputs

To refresh reports for an existing output directory:

```bash
uv run python main.py \
  +name=critic_eval \
  "experiment.tasks=[evaluate_scenes]" \
  experiment.scenebenchmark_critic.enabled=true \
  hydra.run.dir=/path/to/existing/output_dir
```

`evaluate_scenes` scans saved `scene_state.json` and `house_state.json` files,
then overwrites the critic reports in place.

## Direct Python API

```python
from scenesmith.scenebenchmark_critic import (
    CriticConfig,
    evaluate_room_scene,
    format_prompt_context,
    write_room_stage_report,
)

config = CriticConfig(
    enabled=True,
    metrics=("spatial_accessibility", "functional_dependency"),
)

payload = evaluate_room_scene(room_scene, config=config, stage="final_scene")
context = format_prompt_context(payload, max_issues=8)
write_room_stage_report(room_scene, stage_dir, config=config, stage="final_scene")
```

The payload contains the adapted SceneBenchmark-style `case_pack`, rule
`results`, stage metadata, and summary counts. The Markdown report is intended
for quick human inspection; the JSON report is stable enough for downstream
automation.

## Integration Notes

- The critic is report-only by default. `hard_gate` metadata is recorded, but v1
  does not roll back or rewrite SceneSmith scenes.
- LLM critic prompt injection only runs for furniture and manipuland agents.
- Combined house furniture-stage reports filter out manipulands so stage-level
  checks match the objects that have actually been placed at that point.
- The embedded `vendor/rules.py` file is the local rule subset; it intentionally
  avoids importing the external SceneBenchmark repo.
