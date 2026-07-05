#!/usr/bin/env python3
"""Replay HSSD orientation prompts against saved side-view renders.

2026-06-30: Added to validate prompt changes for mesh canonicalization front
view selection without rerunning Blender asset generation.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from openai import OpenAI
from PIL import Image, ImageDraw


DEFAULT_SCENE_ROOT = Path(
    "/data/task3_2/L202500266_hrk/code/scenesmith/outputs/critic_probe/"
    "2026-06-30_11-15-32/critic_on/batch_001/scene_001/room_bedroom"
)
DEFAULT_PROMPT = (
    Path(__file__).resolve().parents[1]
    / "scenesmith/prompts/data/mesh_physics/hssd.yaml"
)


@dataclass(frozen=True)
class Case:
    name: str
    object_type: str
    mesh_id: str
    expected_index: int
    note: str


CASES = [
    Case("desk_chair", "furniture", "0d29db65c42d5ca9021037c0b73914e9d63e0096", 3, "seat front opposite backrest"),
    Case("study_desk", "furniture", "eb6aa6ead94ef785c022795b36da765ac1c1cbe1", 3, "drawer/knee side face-on"),
    Case("nightstand", "furniture", "cb73b8d7a7eba06c5070a71263b39705724e9f7b", 3, "drawer/open shelf face-on"),
    Case("wardrobe", "furniture", "a9967a984dbe294a7b789d969c5f1ed6a9dbca5c", 3, "doors face-on"),
    Case("bed", "furniture", "e2e72c270d6dae987e03d7f2ce4c3e185e51a1fc", 3, "foot/end view opposite the headboard"),
    Case("digital_alarm_clock", "manipuland", "568a7d49f3f327c9c8ce5f4b47291a44edbb7aa5", 3, "display face-on"),
    Case("laptop_computer", "manipuland", "61304cfe7c7c7c57463ac6297b6e02cf622cbf75", 3, "user-facing screen/keyboard"),
]


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def load_prompt(path: Path) -> str:
    data = yaml.safe_load(path.read_text())
    return str(data["prompt"])


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def case_image_paths(scene_root: Path, case: Case) -> list[Path]:
    debug_dir = (
        scene_root
        / "generated_assets"
        / case.object_type
        / "debug"
        / case.mesh_id
    )
    paths = [debug_dir / f"{idx}_side.png" for idx in range(4)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing side renders for {case.name}: {missing}")
    return paths


def build_messages(prompt: str, case: Case, image_paths: list[Path]) -> list[dict[str, Any]]:
    text = (
        f"{prompt}\n\n"
        f"Please analyze these 4 HSSD side-view renders for asset "
        f"'{case.name}'. The expected semantic front cue is: {case.note}. "
        "Return the JSON object only."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for idx, path in enumerate(image_paths):
        content.append(
            {
                "type": "text",
                "text": (
                    f"IMAGE_INDEX={idx}. Use this exact 0-based number if this "
                    "following image shows the front face."
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image(path)}",
                    "detail": "high",
                },
            }
        )
    return [{"role": "user", "content": content}]


def build_contact_sheet(image_paths: list[Path], output_path: Path) -> Path:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images) + 70
    sheet = Image.new("RGB", (cell_w * len(images), cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, image in enumerate(images):
        x = idx * cell_w
        sheet.paste(image, (x + (cell_w - image.width) // 2, 70))
        draw.rectangle([x + 8, 8, x + 230, 62], fill="white", outline="black", width=4)
        draw.text((x + 20, 18), f"IMAGE {idx}", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def build_contact_sheet_messages(
    prompt: str,
    case: Case,
    image_paths: list[Path],
    work_dir: Path,
) -> list[dict[str, Any]]:
    sheet_path = build_contact_sheet(image_paths, work_dir / f"{case.name}_sheet.png")
    text = (
        f"{prompt}\n\n"
        f"Please analyze this single contact sheet for asset '{case.name}'. "
        f"The four panels are labeled IMAGE 0, IMAGE 1, IMAGE 2, IMAGE 3. "
        f"The expected semantic front cue is: {case.note}. "
        "Return front_view_image_index as the 0-based panel number only."
    )
    content: list[dict[str, Any]] = [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image(sheet_path)}",
                "detail": "high",
            },
        },
    ]
    return [{"role": "user", "content": content}]


def run_case(
    client: OpenAI,
    model: str,
    prompt: str,
    scene_root: Path,
    case: Case,
    mode: str,
    work_dir: Path,
) -> dict[str, Any]:
    image_paths = case_image_paths(scene_root, case)
    messages = (
        build_contact_sheet_messages(prompt, case, image_paths, work_dir)
        if mode == "contact-sheet"
        else build_messages(prompt, case, image_paths)
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    parsed = extract_json(content)
    orientation = parsed.get("canonical_orientation", {})
    predicted = int(orientation.get("front_view_image_index", -1))
    return {
        "case": case.name,
        "expected": case.expected_index,
        "predicted": predicted,
        "ok": predicted == case.expected_index,
        "reason": orientation.get("front_view_reasoning", ""),
        "raw": parsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8002/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "not-needed"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "unsloth/Qwen3.6-27B-GGUF"))
    parser.add_argument("--cases", nargs="*", default=[case.name for case in CASES])
    parser.add_argument("--mode", choices=("separate", "contact-sheet"), default="separate")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/scenesmith_hssd_orientation_replay"))
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = load_prompt(args.prompt)
    selected = {name for name in args.cases}
    cases = [case for case in CASES if case.name in selected]
    unknown = selected - {case.name for case in CASES}
    if unknown:
        raise ValueError(f"Unknown case(s): {sorted(unknown)}")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    results = []
    for case in cases:
        result = run_case(
            client,
            args.model,
            prompt,
            args.scene_root,
            case,
            args.mode,
            args.work_dir,
        )
        results.append(result)
        status = "PASS" if result["ok"] else "FAIL"
        print(
            f"{status:4} {result['case']:<20} "
            f"pred={result['predicted']} expected={result['expected']} "
            f"reason={result['reason']}"
        )
        sys.stdout.flush()

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    passed = sum(1 for result in results if result["ok"])
    print(f"\n{passed}/{len(results)} cases matched expected front_view_image_index.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
