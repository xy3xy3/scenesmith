# SceneBenchmark Critic for SceneSmith

This package embeds the SceneBenchmark `spatial_accessibility` and
`functional_dependency` rule critics directly inside SceneSmith. It is designed
for SceneSmith developers who already have `RoomScene` objects and want
geometry feedback without adding the full SceneBenchmark runtime or a VLM
dependency. Combined-house helpers remain available for compatibility and
explicit opt-in workflows, but the supported SceneSmith path is room-first.

The default SceneSmith integration is single-room: it evaluates room
checkpoints and leaves combined-house reports disabled unless
`house_stage_hooks` is overridden explicitly.

## What It Checks

- `spatial_accessibility`: SceneBenchmark's grid/reach based evaluator for
  functional access zones, obstacle masks, connected stance area, and reach
  distance.
- `functional_dependency`: SceneBenchmark's template-proposed and rule-scored
  dependency relations, including object-on-support, lamp-surface,
  seating-work-surface, seating-media, bed-nightstand, dining/workstation, and
  related near/facing relations.

The adapter derives categories, affordance hints, object function profiles,
support regions, room polygons, and placement relations from SceneSmith object
names, descriptions, metadata, support surfaces, and `placement_info`.
Asset annotation fields such as `front_face`, `access_direction`,
`operation_space`, `target_relation`, and `explicit_target_relation` are
preserved; annotated front/access directions are also mapped into
SceneBenchmark-style interaction faces so spatial accessibility checks use the
annotated operating side. Asset annotations with `benchmark_relevance` set to a
non-functional value suppress functional affordance checks, matching
SceneBenchmark's converter behavior. `affordances`, `functional_categories`,
and `candidate_affordances` are all accepted as affordance inputs, and metadata
functional dependencies plus `support_regions`/`support_region` may live either
at the object metadata root or inside `metadata.functional_hints`.
When annotations are absent, the adapter mirrors the SceneBenchmark demo
converter's lightweight defaults: it normalizes room-prefixed instance names
such as `bedroom_nightstand_1_f0_c`, infers `category_keywords`,
`front_hint`, `target_relation`, and `metric_relevance`, and writes
`support_region_summary` for detected or SceneSmith-provided support regions.
SceneSmith also materializes SceneBenchmark's grouped functional-dependency
checks for dining sets, workstations, and multi-nightstand bedside pairs when
the corresponding objects are present in a single room.

The vendored rule code is copied from `~/proj/SceneBenchmark/src` under
`vendor/scenebenchmark/`. SceneSmith uses SceneBenchmark's deterministic
template FD proposer by default. If `fd_relation_proposer_mode` is explicitly
set to `vlm`, `hybrid`, or `auto`, the vendored FD proposer entrypoint is used;
when the optional SceneBenchmark VLM stack is unavailable it falls back to the
template proposer. The full SceneBenchmark rendering/request pipeline is not
used by the default SceneSmith integration. See `vendor/README.md` for the
source manifest and intentional vendoring differences.

## Enable During Generation

The critic is disabled by default. Enable it with Hydra overrides. By default
this refreshes room reports only:

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

Combined-house reports are opt-in:

- `combined_house_after_furniture/scenebenchmark_critic.json|md`
- `combined_house/scenebenchmark_critic.json|md`

For a memory-safe single-room smoke test, keep generation serial, keep the
template FD proposer, and stop after furniture placement:

```bash
uv run python main.py \
  +name=scenebenchmark_critic_memsafe_smoke \
  "experiment.tasks=[generate_scenes]" \
  experiment.scenebenchmark_critic.enabled=true \
  "experiment.scenebenchmark_critic.room_stage_hooks=[scene_after_furniture]" \
  experiment.scenebenchmark_critic.fd_relation_proposer_mode=template \
  experiment.num_workers=1 \
  experiment.pipeline.parallel_rooms=false \
  experiment.pipeline.max_parallel_rooms=1 \
  experiment.pipeline.stop_stage=furniture \
  furniture_agent.asset_manager.router.parallel_workers=1 \
  "experiment.prompts=['A small bedroom with a bed and a nightstand beside it.']"
```

If credentials are stored in a local `config.yml` with `openai_api_key` and
`openai_base_url`, use a small wrapper so secrets stay out of the shell history:

```bash
uv run python - <<'PY'
import os
import runpy
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path("config.yml").read_text(encoding="utf-8")) or {}
os.environ["OPENAI_API_KEY"] = str(config["openai_api_key"])
os.environ["OPENAI_BASE_URL"] = str(config["openai_base_url"])
os.environ["OPENAI_USE_RESPONSES"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

sys.argv = [
    "main.py",
    "+name=scenebenchmark_critic_memsafe_smoke",
    "experiment.tasks=[generate_scenes]",
    "experiment.scenebenchmark_critic.enabled=true",
    "experiment.scenebenchmark_critic.room_stage_hooks=[scene_after_furniture]",
    "experiment.scenebenchmark_critic.fd_relation_proposer_mode=template",
    "experiment.num_workers=1",
    "experiment.pipeline.parallel_rooms=false",
    "experiment.pipeline.max_parallel_rooms=1",
    "experiment.pipeline.stop_stage=furniture",
    "furniture_agent.asset_manager.router.parallel_workers=1",
    "openai.use_responses=false",
    "experiment.prompts=['A small bedroom with a bed and a nightstand beside it.']",
]
runpy.run_path("main.py", run_name="__main__")
PY
```

To refresh the smoke-test report without regenerating the scene, rerun the same
output directory with `experiment.tasks=[evaluate_scenes]`.

After the smoke test, verify that at least the room-stage reports exist:

```bash
find /path/to/output_dir \
  -path '*/scene_states/scene_after_furniture/scenebenchmark_critic.json' \
  -o -path '*/scene_states/scene_after_furniture/scenebenchmark_critic.md'
```

Then summarize the JSON reports without requiring `jq`:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

root = Path("/path/to/output_dir")
for report_path in sorted(root.glob("scene_*/room_*/scene_states/*/scenebenchmark_critic.json")):
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    checks = payload.get("case_pack", {}).get("checks", [])
    proposer_checks = [
        check
        for check in checks
        if check.get("check_source") == "fd_relation_proposer"
    ]
    print(report_path)
    print("  scope:", payload.get("scope"), "stage:", payload.get("stage"))
    print("  metrics:", sorted((payload.get("summary", {}).get("metric_summary") or {})))
    print("  total_checks:", payload.get("summary", {}).get("scene_summary", {}).get("total_checks"))
    print("  fd_proposer_checks:", len(proposer_checks))
PY
```

For the single-room smoke configuration above, those reports are the primary
manual acceptance artifacts.

## Re-evaluate Existing Outputs

To refresh reports for an existing output directory:

```bash
uv run python main.py \
  +name=critic_eval \
  "experiment.tasks=[evaluate_scenes]" \
  experiment.scenebenchmark_critic.enabled=true \
  hydra.run.dir=/path/to/existing/output_dir
```

With credentials in `config.yml`, use the same wrapper pattern:

```bash
uv run python - <<'PY'
import os
import runpy
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path("config.yml").read_text(encoding="utf-8")) or {}
os.environ["OPENAI_API_KEY"] = str(config["openai_api_key"])
os.environ["OPENAI_BASE_URL"] = str(config["openai_base_url"])
os.environ["OPENAI_USE_RESPONSES"] = "false"

sys.argv = [
    "main.py",
    "+name=critic_eval_config",
    "experiment.tasks=[evaluate_scenes]",
    "experiment.scenebenchmark_critic.enabled=true",
    "openai.use_responses=false",
    "hydra.run.dir=/path/to/existing/output_dir",
]
runpy.run_path("main.py", run_name="__main__")
PY
```

`evaluate_scenes` scans saved room `scene_state.json` files and overwrites the
room critic reports in place. Combined-house `house_state.json` reports are
refreshed only when `house_stage_hooks` is set to a non-empty list.

## Direct Python API

```python
from scenesmith.scenebenchmark_critic import (
    CriticConfig,
    evaluate_room_scene,
    format_prompt_context,
    room_scene_to_case_pack,
    write_room_stage_report,
)

config = CriticConfig(
    enabled=True,
    metrics=("spatial_accessibility", "functional_dependency"),
)

case_pack = room_scene_to_case_pack(room_scene, stage="final_scene")
payload = evaluate_room_scene(room_scene, config=config, stage="final_scene")
context = format_prompt_context(payload, max_issues=8)
write_room_stage_report(room_scene, stage_dir, config=config, stage="final_scene")
```

`evaluate_room_scene()` runs immediately for ad hoc or scripted checks. The
`write_*_stage_report()` helpers are the stage-hook entrypoints and honor
`enabled` plus the configured stage hook lists.

`room_scene_to_case_pack()` exposes the adapted SceneBenchmark-style geometry
for debugging or downstream automation. The evaluation payload includes that
`case_pack`, rule `results`, stage metadata, and summary counts. The Markdown
report is intended for quick human inspection; the JSON report is stable enough
for downstream automation.

## Integration Notes

- The critic is report-only by default. `hard_gate` metadata is recorded, but v1
  does not roll back or rewrite SceneSmith scenes.
- LLM critic prompt injection only runs for furniture and manipuland agents.
- Combined house reports are still available through `write_house_stage_report`
  or non-empty `house_stage_hooks`, but the default integration is room-only.
- Combined house furniture-stage reports filter out manipulands so stage-level
  checks match the objects that have actually been placed at that point.
- `vendor/rules.py` is a bridge into the vendored SceneBenchmark modules; it
  intentionally avoids importing the external SceneBenchmark repo at runtime.
