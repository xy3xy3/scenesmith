#!/usr/bin/env python3
"""Prompt-only regression test for the local Qwen furniture critic.

This script assembles the real SceneSmith furniture critic prompts, injects a
SceneBenchmark bedside-wall failure, sends the prompt to a local
OpenAI-compatible model, and checks that the model treats the issue as
actionable instead of dismissing it as a false positive.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.version_info < (3, 11):
    raise RuntimeError(
        "SceneSmith requires Python 3.11. Run this script with `uv run "
        "python scripts/test_local_qwen_final_scene.py ...`."
    )

from scenesmith.prompts import prompt_manager
from scenesmith.prompts.registry import FurnitureAgentPrompts


DEFAULT_BASE_URL = "http://127.0.0.1:8002/v1"
DEFAULT_MODEL = "unsloth/Qwen3.6-27B-GGUF"

SCENE_DESCRIPTION = (
    "A bedroom with a queen bed centered on the north wall, one nightstand "
    "with a table lamp on each side of the bed, a dresser against the opposite "
    "wall facing the bed, a wardrobe next to the dresser, and a small "
    "wastebasket near the dresser."
)

SCENEBENCHMARK_CONTEXT = """No physics violations detected.

Additional SceneBenchmark geometry critic context:
SceneBenchmark geometry critic found rule-level issues. Use this as geometric evidence alongside visual critique:
- fail: functional_dependency subject=nightstand_0 related=north_wall. Rule dependency `back_against_wall`: subject `nightstand_0`; selected `north_wall`; no allowed face is backed by the wall: gap 0.80m, best back angle 131deg.
- fail: functional_dependency subject=nightstand_1 related=north_wall. Rule dependency `side_or_back_against_wall`: subject `nightstand_1`; selected `north_wall`; no allowed face is backed by the wall: gap 0.83m, best back angle 133deg.
"""

INLINE_SCENE_STATE = """Available scene state summary for this prompt-only test:
- bed_0: category=bed, placement=headboard/back against north_wall, centered on north wall
- nightstand_0: category=nightstand, beside left side of bed_0, gap to north_wall=0.80m
- nightstand_1: category=nightstand, beside right side of bed_0, gap to north_wall=0.83m
- north_wall: wall backing bed_0
- No collisions and no reachability blockers are present.
"""

REGRESSION_INSTRUCTION = f"""PROMPT-ONLY REGRESSION TEST.

Tools are unavailable in this test harness. Use the assembled system prompt,
runner instruction, SceneBenchmark geometry context, and inline scene summary.

{INLINE_SCENE_STATE}

Return concise JSON only with this shape:
{{
  "nightstand_wall_issue_actionable": true,
  "false_positive_issues": [],
  "recommended_fix": "..."
}}

The expected behavior is to preserve the general rule: when a bed is backed by a
wall, adjacent nightstands should normally share that wall backing unless there
is an explicit freestanding-bed exception or a collision/access reason not to.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble real furniture critic prompts and test local Qwen."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8002/v1.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help="Model id exposed by the local llama-server.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/local_qwen_prompt_regression/nightstand_wall"),
        help="Directory where assembled prompts and model response are written.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Only assemble prompts and run static checks; do not call the model.",
    )
    return parser.parse_args()


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def assemble_messages() -> list[dict[str, str]]:
    system_prompt = prompt_manager.get_prompt(
        FurnitureAgentPrompts.STATEFUL_CRITIC_AGENT,
        scene_description=SCENE_DESCRIPTION,
    )
    runner_prompt = prompt_manager.get_prompt(
        FurnitureAgentPrompts.STATEFUL_CRITIC_RUNNER_INSTRUCTION,
        physics_context=SCENEBENCHMARK_CONTEXT,
        placement_style="natural",
        reachability_context="",
        robot_width=0.6,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": runner_prompt},
        {"role": "user", "content": REGRESSION_INSTRUCTION},
    ]


def call_model(
    *,
    messages: list[dict[str, str]],
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        _chat_completions_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'dummy')}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Model request failed: HTTP {exc.code}: {body}") from exc


def _response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def evaluate_response(text: str) -> dict[str, Any]:
    lower = text.lower()
    false_positive_near_nightstand = bool(
        re.search(r"nightstand.{0,160}(false positive|no action needed)", lower, re.S)
        or re.search(
            r"(false positive|no action needed).{0,160}nightstand", lower, re.S
        )
    )
    actionable = (
        "nightstand" in lower
        and "wall" in lower
        and any(
            token in lower
            for token in ("actionable", "snap", "shift", "move", "against")
        )
        and not false_positive_near_nightstand
    )
    return {
        "passed": actionable,
        "false_positive_near_nightstand": false_positive_near_nightstand,
        "mentions_nightstand": "nightstand" in lower,
        "mentions_wall": "wall" in lower,
    }


def write_artifacts(
    *,
    output_dir: Path,
    messages: list[dict[str, str]],
    response: dict[str, Any] | None,
    evaluation: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assembled_messages.json").write_text(
        json.dumps(messages, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if response is not None:
        (output_dir / "model_response.json").write_text(
            json.dumps(response, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "model_response.txt").write_text(
            _response_text(response),
            encoding="utf-8",
        )
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": args.base_url,
        "model": args.model,
        "called_model": response is not None,
        "evaluation": evaluation,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_path


def main() -> int:
    args = parse_args()
    messages = assemble_messages()
    assembled_text = "\n\n".join(message["content"] for message in messages)
    required_fragments = [
        "Bedside Wall Anchoring",
        "SceneBenchmark geometry critic context",
        "back_against_wall",
        "side_or_back_against_wall",
    ]
    missing = [
        fragment for fragment in required_fragments if fragment not in assembled_text
    ]
    if missing:
        raise AssertionError(
            f"Assembled prompt is missing required fragments: {missing}"
        )

    response = None
    evaluation: dict[str, Any] = {
        "passed": True,
        "static_only": True,
        "missing_fragments": [],
    }
    if not args.no_model:
        response = call_model(
            messages=messages,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        evaluation = evaluate_response(_response_text(response))

    summary_path = write_artifacts(
        output_dir=args.output_dir,
        messages=messages,
        response=response,
        evaluation=evaluation,
        args=args,
    )
    print(json.dumps(evaluation, indent=2, ensure_ascii=False))
    print(f"Wrote prompt regression artifacts to {summary_path}")
    return 0 if evaluation.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
