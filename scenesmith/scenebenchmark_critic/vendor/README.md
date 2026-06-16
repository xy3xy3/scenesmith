# Vendored SceneBenchmark Rule Critics

This directory embeds the SceneBenchmark rule code needed by SceneSmith's
single-room critic integration.

## Source Scope

Copied from `~/proj/SceneBenchmark/src`:

- `critic/accessibility.py`
- `critic/config.py`
- `critic/dependency.py`
- `critic/geometry.py`
- `critic/models.py`
- `metrics/base.py`
- all Python files under `metrics/functional_dependency/`
- all Python files under `metrics/spatial_accessibility/`

The vendored package intentionally does not include the full SceneBenchmark
rendering, request-building, case-pack generation, selection, or service
runtime. SceneSmith provides its own `RoomScene` adapter and report writer.

## Intentional Differences

- Imports are rewritten from top-level `critic.*` and `metrics.*` modules to the
  local `scenesmith.scenebenchmark_critic.vendor.scenebenchmark.*` package.
- `functional_dependency.proposer` keeps the SceneBenchmark VLM entrypoint, but
  imports the optional agent stack defensively. The default SceneSmith config
  uses the deterministic template proposer.
- Lazy exports in metric package `__init__.py` files are package-qualified so
  they do not depend on the external SceneBenchmark repo being on `PYTHONPATH`.
- `vendor/rules.py` is a SceneSmith bridge that normalizes result payloads and
  exposes the room critic configuration knobs.

## Guardrails

`tests/unit/test_scenebenchmark_critic.py` checks that the vendored source
manifest is complete, that vendored modules import without external
SceneBenchmark paths, that both FD/SA wrapper entrypoints run locally, and that
core rule bodies match the upstream SceneBenchmark checkout through an optional
AST parity test when `~/proj/SceneBenchmark/src` is available.
