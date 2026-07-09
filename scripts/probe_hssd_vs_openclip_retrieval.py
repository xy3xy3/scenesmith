#!/usr/bin/env python3
"""Compare HSSD embedding retrieval against OpenCLIP retrieval on probe prompts.

The probe uses objects mentioned by scripts/run_single_room_critic_probe.sh,
retrieves the same queries with HSSD Zvec embeddings and OpenCLIP, renders the
top candidates, and writes a Markdown report under outputs/.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import tempfile

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageColor, ImageDraw, ImageFilter

# 2026-07-08: Keep heavy runtime imports lazy so --help and config validation do
# not pay the bpy/OpenCLIP/Torch startup cost before actual retrieval/rendering.
if TYPE_CHECKING:
    from scenesmith.agent_utils.blender.renderer import BlenderRenderer
    from scenesmith.agent_utils.hssd_retrieval.config import HssdConfig
    from scenesmith.agent_utils.hssd_retrieval.retrieval import (
        HssdRetriever,
        RetrievalCandidate,
    )


LOGGER = logging.getLogger("probe_hssd_vs_openclip_retrieval")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE_SCRIPT = PROJECT_ROOT / "scripts" / "run_single_room_critic_probe.sh"
DEFAULT_RENDER_CACHE_ROOT = PROJECT_ROOT / "data" / "hssd_rendered_assets"
DEFAULT_RENDER_VIEWS = ("top", "iso")
SUPPORTED_NAMED_VIEWS = {"top", "bottom", "front", "back", "left", "right", "iso"}

# 2026-07-08: Keep the evaluated object list explicit so report rows remain
# traceable to the fixed single-room critic probe prompts.
PROMPT_OBJECTS: list[dict[str, str]] = [
    {
        "case_id": "living_room_media_bottleneck",
        "object": "sofa",
        "query": "living room sofa",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "living_room_media_bottleneck",
        "object": "TV stand",
        "query": "TV stand",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "living_room_media_bottleneck",
        "object": "television",
        "query": "television",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "living_room_media_bottleneck",
        "object": "coffee table",
        "query": "coffee table",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "living_room_media_bottleneck",
        "object": "armchair",
        "query": "living room armchair",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "living_room_media_bottleneck",
        "object": "floor lamp",
        "query": "floor lamp",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "living_room_media_bottleneck",
        "object": "remote control",
        "query": "remote control",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "living_room_media_bottleneck",
        "object": "magazine",
        "query": "magazine",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "living_room_media_bottleneck",
        "object": "rug",
        "query": "small rug",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "living_room_media_bottleneck",
        "object": "game controller",
        "query": "game controller",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "living_room_media_bottleneck",
        "object": "coaster",
        "query": "drink coaster",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "living_room_media_bottleneck",
        "object": "candle",
        "query": "decorative candle",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "study_desk_access_crunch",
        "object": "desk",
        "query": "study desk",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "study_desk_access_crunch",
        "object": "office chair",
        "query": "office chair",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "study_desk_access_crunch",
        "object": "computer monitor",
        "query": "computer monitor",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "study_desk_access_crunch",
        "object": "guest chair",
        "query": "guest chair",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "study_desk_access_crunch",
        "object": "bookshelf",
        "query": "bookshelf",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "study_desk_access_crunch",
        "object": "desk lamp",
        "query": "desk lamp",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "study_desk_access_crunch",
        "object": "notebook",
        "query": "notebook",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "study_desk_access_crunch",
        "object": "pen holder",
        "query": "pen holder",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "study_desk_access_crunch",
        "object": "trash can",
        "query": "small trash can",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "study_desk_access_crunch",
        "object": "keyboard",
        "query": "computer keyboard",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "study_desk_access_crunch",
        "object": "computer mouse",
        "query": "computer mouse",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "study_desk_access_crunch",
        "object": "headphones",
        "query": "desk headphones",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "study_desk_access_crunch",
        "object": "coffee mug",
        "query": "coffee mug",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "bedroom_bedside_blockage",
        "object": "bed",
        "query": "bed",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "bedroom_bedside_blockage",
        "object": "nightstand",
        "query": "nightstand",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "bedroom_bedside_blockage",
        "object": "table lamp",
        "query": "table lamp",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "bedroom_bedside_blockage",
        "object": "dresser",
        "query": "dresser",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "bedroom_bedside_blockage",
        "object": "wardrobe",
        "query": "wardrobe",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "bedroom_bedside_blockage",
        "object": "alarm clock",
        "query": "alarm clock",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "bedroom_bedside_blockage",
        "object": "book",
        "query": "book",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "bedroom_bedside_blockage",
        "object": "wastebasket",
        "query": "small wastebasket",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "bedroom_bedside_blockage",
        "object": "picture frame",
        "query": "picture frame",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "bedroom_bedside_blockage",
        "object": "smartphone",
        "query": "smartphone",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "bedroom_bedside_blockage",
        "object": "eyeglasses",
        "query": "eyeglasses",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "dining_room_service_squeeze",
        "object": "dining table",
        "query": "dining table",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "dining_room_service_squeeze",
        "object": "dining chair",
        "query": "dining chair",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "dining_room_service_squeeze",
        "object": "sideboard",
        "query": "sideboard",
        "object_type": "FURNITURE",
    },
    {
        "case_id": "dining_room_service_squeeze",
        "object": "plate",
        "query": "plate",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "dining_room_service_squeeze",
        "object": "cutlery",
        "query": "cutlery",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "dining_room_service_squeeze",
        "object": "drinking glass",
        "query": "drinking glass",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "dining_room_service_squeeze",
        "object": "vase with flowers",
        "query": "vase with flowers",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "dining_room_service_squeeze",
        "object": "coaster",
        "query": "coaster",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "dining_room_service_squeeze",
        "object": "fork",
        "query": "dining fork",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "dining_room_service_squeeze",
        "object": "spoon",
        "query": "dining spoon",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "dining_room_service_squeeze",
        "object": "bowl",
        "query": "serving bowl",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "dining_room_service_squeeze",
        "object": "napkin",
        "query": "cloth napkin",
        "object_type": "MANIPULAND",
    },
    {
        "case_id": "dining_room_service_squeeze",
        "object": "salt shaker",
        "query": "salt shaker",
        "object_type": "MANIPULAND",
    },
]


@dataclass(frozen=True)
class QuerySpec:
    case_id: str
    object_name: str
    query: str
    object_type: str


@dataclass
class RenderedResult:
    rank: int
    asset_id: str
    score: float
    name: str
    wordnet_key: str
    image_paths: list[Path]
    montage_path: Path | None
    metadata_path: Path | None = None
    view_sources: dict[str, str] | None = None
    error: str | None = None


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "item"


def relative_to_report(path: Path, report_path: Path) -> str:
    return path.resolve().relative_to(report_path.parent.resolve()).as_posix()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_probe_cases(script_path: Path) -> dict[str, dict[str, str]]:
    """Extract case metadata from run_single_room_critic_probe.sh for reporting."""
    if not script_path.exists():
        LOGGER.warning("Probe script not found: %s", script_path)
        return {}

    case_re = re.compile(r'^\s*"([^"|]+)\|([^"|]+)\|(.+)"\s*$')
    cases: dict[str, dict[str, str]] = {}
    for line in script_path.read_text(encoding="utf-8").splitlines():
        match = case_re.match(line)
        if match is None:
            continue
        case_id, critic_goal, prompt = match.groups()
        cases[case_id] = {"critic_goal": critic_goal, "prompt": prompt}
    return cases


def round_robin_specs(groups: list[list[QuerySpec]]) -> list[QuerySpec]:
    pending = [list(group) for group in groups if group]
    ordered: list[QuerySpec] = []
    while pending:
        next_round: list[list[QuerySpec]] = []
        for group in pending:
            ordered.append(group.pop(0))
            if group:
                next_round.append(group)
        pending = next_round
    return ordered


def prioritize_specs_for_sampling(specs: list[QuerySpec]) -> list[QuerySpec]:
    case_ids = list(dict.fromkeys(spec.case_id for spec in specs))
    manipulands_by_case: list[list[QuerySpec]] = []
    furniture_by_case: list[list[QuerySpec]] = []

    # 2026-07-09 修改原因：烟测只取前几个 query 时，优先覆盖更多 case 和更多小物体，避免样本全落在大件家具上。
    for case_id in case_ids:
        case_specs = [spec for spec in specs if spec.case_id == case_id]
        manipulands_by_case.append(
            [spec for spec in case_specs if spec.object_type == "MANIPULAND"]
        )
        furniture_by_case.append(
            [spec for spec in case_specs if spec.object_type != "MANIPULAND"]
        )

    return round_robin_specs(manipulands_by_case) + round_robin_specs(
        furniture_by_case
    )


def build_query_specs(max_queries: int) -> list[QuerySpec]:
    specs = [
        QuerySpec(
            case_id=item["case_id"],
            object_name=item["object"],
            query=item["query"],
            object_type=item["object_type"],
        )
        for item in PROMPT_OBJECTS
    ]
    if max_queries > 0:
        return prioritize_specs_for_sampling(specs)[:max_queries]
    return specs


def make_hssd_config(
    *,
    data_path: Path,
    preprocessed_path: Path,
    backend: str,
    top_k: int,
    zvec_collection_path: Path,
    zvec_base_url: str,
) -> HssdConfig:
    from scenesmith.agent_utils.hssd_retrieval.config import HssdConfig, HssdZvecConfig

    zvec_config = None
    if backend == "embedding":
        zvec_config = HssdZvecConfig(
            collection_path=zvec_collection_path,
            base_url=zvec_base_url,
        )

    return HssdConfig(
        data_path=data_path,
        preprocessed_path=preprocessed_path,
        retrieval_backend=backend,
        use_top_k=top_k,
        zvec=zvec_config,
    )


def export_candidate_glb(candidate: RetrievalCandidate) -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    glb_path = Path(handle.name)
    handle.close()
    candidate.mesh.export(glb_path)
    return glb_path


def render_named_candidate_views(
    *,
    renderer: BlenderRenderer,
    candidate: RetrievalCandidate,
    output_dir: Path,
    width: int,
    height: int,
    view_names: list[str],
) -> dict[str, Path]:
    glb_path = export_candidate_glb(candidate)
    try:
        # 2026-07-09 修改原因：probe 结果目录只保留 top/iso 这类命名视角，避免再次生成旧的多视角分析图。
        named_paths = renderer.render_named_views_for_embedding(
            mesh_path=glb_path,
            output_dir=output_dir,
            view_names=view_names,
            width=width,
            height=height,
        )
        return {path.stem.lower(): path for path in named_paths}
    finally:
        glb_path.unlink(missing_ok=True)


def composite_render_background(image_paths: list[Path], background_color: str) -> None:
    """Make pale/white assets visible when reports are viewed on white pages."""
    # 2026-07-08: Some HSSD assets render nearly white on transparent/white
    # backgrounds, so flatten every render onto neutral gray with a soft alpha shadow.
    bg_rgb = ImageColor.getrgb(background_color)
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGBA")
        except OSError:
            LOGGER.warning("Could not post-process render image: %s", image_path)
            continue

        alpha = image.getchannel("A")
        background = Image.new("RGBA", image.size, bg_rgb + (255,))
        shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=5))
        shifted_shadow = Image.new("L", image.size, 0)
        shifted_shadow.paste(shadow_alpha, (3, 5))
        shadow_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        shadow_layer.putalpha(shifted_shadow.point(lambda value: int(value * 0.28)))
        flattened = Image.alpha_composite(background, shadow_layer)
        flattened.alpha_composite(image)
        flattened.convert("RGB").save(image_path)


def copy_cached_views(
    *,
    asset_id: str,
    cache_root: Path | None,
    output_dir: Path,
    render_views: list[str],
) -> tuple[dict[str, Path], dict[str, str], list[str]]:
    reused_paths: dict[str, Path] = {}
    view_sources: dict[str, str] = {}
    missing_views: list[str] = []

    for view_name in render_views:
        cache_path = (
            None if cache_root is None else cache_root / asset_id / f"{view_name}.png"
        )
        if cache_path is None or not cache_path.exists():
            missing_views.append(view_name)
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{view_name}.png"
        shutil.copy2(cache_path, output_path)
        reused_paths[view_name] = output_path
        view_sources[view_name] = f"reused:{display_path(cache_path)}"

    return reused_paths, view_sources, missing_views


def prepare_candidate_views(
    *,
    renderer: BlenderRenderer,
    candidate: RetrievalCandidate,
    output_dir: Path,
    cache_root: Path | None,
    render_views: list[str],
    width: int,
    height: int,
) -> tuple[list[Path], dict[str, str]]:
    image_paths_by_view, view_sources, missing_views = copy_cached_views(
        asset_id=candidate.mesh_id,
        cache_root=cache_root,
        output_dir=output_dir,
        render_views=render_views,
    )

    if missing_views:
        rendered_paths = render_named_candidate_views(
            renderer=renderer,
            candidate=candidate,
            output_dir=output_dir,
            width=width,
            height=height,
            view_names=missing_views,
        )
        image_paths_by_view.update(rendered_paths)
        for view_name in missing_views:
            if view_name in rendered_paths:
                view_sources[view_name] = "rendered"

    ordered_paths = [
        image_paths_by_view[view_name]
        for view_name in render_views
        if view_name in image_paths_by_view
    ]
    return ordered_paths, view_sources


def write_result_metadata(
    *,
    output_dir: Path,
    spec: QuerySpec,
    method_key: str,
    result: RenderedResult,
    render_views: list[str],
) -> Path:
    lines = [
        f"Case ID: {spec.case_id}",
        f"Object Name: {spec.object_name}",
        f"Query: {spec.query}",
        f"Object Type: {spec.object_type}",
        f"Method: {method_key}",
        f"Rank: {result.rank}",
        f"Asset ID: {result.asset_id}",
        f"Score: {result.score:.6f}",
        f"Name: {result.name}",
        f"WordNet: {result.wordnet_key}",
    ]

    if result.view_sources:
        for view_name in render_views:
            lines.append(
                f"View {view_name}: {result.view_sources.get(view_name, 'missing')}"
            )

    if result.error:
        lines.append(f"Error: {result.error}")

    metadata_path = output_dir / "metadata.txt"
    metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metadata_path


def make_montage(
    results: list[RenderedResult],
    output_path: Path,
    *,
    thumb_size: int = 220,
    max_views: int = 4,
) -> Path | None:
    rendered = [result for result in results if result.image_paths]
    if not rendered:
        return None

    cols = max(1, min(max_views, max(len(result.image_paths) for result in rendered)))
    rows = len(rendered)
    pad = 12
    label_h = 74
    width = cols * thumb_size + (cols + 1) * pad
    height = rows * (thumb_size + label_h) + (rows + 1) * pad

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for row, result in enumerate(rendered):
        y0 = pad + row * (thumb_size + label_h + pad)
        label = (
            f"#{result.rank} {result.asset_id[:8]} score={result.score:.4f}\n"
            f"{result.name or '(no name)'}\n{result.wordnet_key or ''}"
        )
        draw.text((pad, y0), label, fill="black")
        for col, image_path in enumerate(result.image_paths[:cols]):
            x = pad + col * (thumb_size + pad)
            image_y = y0 + label_h
            try:
                image = Image.open(image_path).convert("RGBA")
            except OSError:
                draw.rectangle(
                    [(x, image_y), (x + thumb_size - 1, image_y + thumb_size - 1)],
                    outline="red",
                    width=2,
                )
                draw.text((x + 8, image_y + 8), "missing image", fill="red")
                continue

            tile = Image.new("RGBA", (thumb_size, thumb_size), "white")
            image.thumbnail((thumb_size, thumb_size))
            offset = ((thumb_size - image.width) // 2, (thumb_size - image.height) // 2)
            tile.alpha_composite(image, dest=offset)
            canvas.paste(tile.convert("RGB"), (x, image_y))
            draw.rectangle(
                [(x, image_y), (x + thumb_size - 1, image_y + thumb_size - 1)],
                outline="gray",
                width=1,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def render_results(
    *,
    retriever: HssdRetriever,
    renderer: BlenderRenderer,
    spec: QuerySpec,
    method_dir: Path,
    method_key: str,
    width: int,
    height: int,
    cache_root: Path | None,
    render_views: list[str],
) -> list[RenderedResult]:
    candidates = retriever.retrieve_multiple(
        description=spec.query,
        object_type=spec.object_type,
        max_candidates=retriever.config.use_top_k,
    )
    rendered_results: list[RenderedResult] = []

    for rank, candidate in enumerate(candidates, start=1):
        metadata = candidate.metadata
        result_dir = method_dir / f"rank_{rank:02d}_{candidate.mesh_id[:12]}"
        result_dir.mkdir(parents=True, exist_ok=True)
        view_sources: dict[str, str] | None = None
        try:
            image_paths, view_sources = prepare_candidate_views(
                renderer=renderer,
                candidate=candidate,
                output_dir=result_dir,
                cache_root=cache_root,
                render_views=render_views,
                width=width,
                height=height,
            )
            error = None
        except Exception as exc:  # pragma: no cover - depends on Blender/runtime data.
            LOGGER.exception("Failed to render %s", candidate.mesh_id)
            image_paths = []
            error = str(exc)

        result = RenderedResult(
            rank=rank,
            asset_id=candidate.mesh_id,
            score=float(candidate.clip_score),
            name=metadata.name,
            wordnet_key=metadata.wordnet_key,
            image_paths=image_paths,
            montage_path=None,
            view_sources=view_sources,
            error=error,
        )
        result.metadata_path = write_result_metadata(
            output_dir=result_dir,
            spec=spec,
            method_key=method_key,
            result=result,
            render_views=render_views,
        )
        rendered_results.append(result)

    montage_path = make_montage(rendered_results, method_dir / "montage.png")
    for result in rendered_results:
        result.montage_path = montage_path

    return rendered_results


def result_to_json(result: RenderedResult, output_dir: Path) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "asset_id": result.asset_id,
        "score": result.score,
        "name": result.name,
        "wordnet_key": result.wordnet_key,
        "image_paths": [
            path.resolve().relative_to(output_dir.resolve()).as_posix()
            for path in result.image_paths
        ],
        "montage_path": (
            result.montage_path.resolve().relative_to(output_dir.resolve()).as_posix()
            if result.montage_path is not None
            else None
        ),
        "metadata_path": (
            result.metadata_path.resolve().relative_to(output_dir.resolve()).as_posix()
            if result.metadata_path is not None
            else None
        ),
        "view_sources": result.view_sources or {},
        "error": result.error,
    }


def write_json(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_path = output_dir / "retrieval_results.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), "utf-8")


def write_markdown(
    *,
    output_dir: Path,
    report_path: Path,
    all_results: list[dict[str, Any]],
    cases: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# HSSD Embedding vs OpenCLIP Retrieval Probe",
        "",
        "Source prompts: `scripts/run_single_room_critic_probe.sh`",
        "",
        "## Run Config",
        "",
        f"- HSSD data: `{args.hssd_data_path}`",
        f"- HSSD preprocessed: `{args.hssd_preprocessed_path}`",
        f"- Zvec collection: `{args.zvec_collection_path}`",
        f"- Zvec embedding URL: `{args.zvec_base_url}`",
        f"- Top K: `{args.top_k}`",
        f"- Render size: `{args.render_width}x{args.render_height}`",
        f"- Render cache root: `{args.render_cache_root}`",
        f"- Render views: `{', '.join(args.render_views)}`",
        "",
        "## Prompt Cases",
        "",
    ]

    for case_id, meta in cases.items():
        lines.extend(
            [
                f"### {case_id}",
                "",
                f"- Critic goal: {meta['critic_goal']}",
                f"- Prompt: {meta['prompt']}",
                "",
            ]
        )

    lines.extend(["## Retrieval Comparison", ""])

    for item in all_results:
        spec = item["query"]
        lines.extend(
            [
                f"### {spec['case_id']} / {spec['object_name']}",
                "",
                f"- Query: `{spec['query']}`",
                f"- Object type: `{spec['object_type']}`",
                "",
                "| HSSD embedding retrieval | OpenCLIP retrieval |",
                "| --- | --- |",
            ]
        )

        embedding_montage = item["methods"]["hssd_embedding"].get("montage_path")
        openclip_montage = item["methods"]["openclip"].get("montage_path")
        embedding_cell = (
            f"![hssd embedding]({relative_to_report(output_dir / embedding_montage, report_path)})"
            if embedding_montage
            else "No render"
        )
        openclip_cell = (
            f"![openclip]({relative_to_report(output_dir / openclip_montage, report_path)})"
            if openclip_montage
            else "No render"
        )
        lines.extend([f"| {embedding_cell} | {openclip_cell} |", ""])

        for method_key, method_label in [
            ("hssd_embedding", "HSSD embedding"),
            ("openclip", "OpenCLIP"),
        ]:
            lines.extend([f"**{method_label} top results**", ""])
            rows = item["methods"][method_key]["results"]
            if not rows:
                lines.extend(["No candidates.", ""])
                continue
            lines.extend(
                [
                    "| Rank | Asset | Score | Name | WordNet |",
                    "| --- | --- | ---: | --- | --- |",
                ]
            )
            for row in rows:
                asset = row["asset_id"]
                lines.append(
                    f"| {row['rank']} | `{asset[:12]}` | {row['score']:.4f} | "
                    f"{row['name']} | `{row['wordnet_key']}` |"
                )
            lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve objects from run_single_room_critic_probe.sh prompts with "
            "HSSD embedding and OpenCLIP backends, render candidates, and write MD."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "hssd_vs_openclip_retrieval" / timestamp,
    )
    parser.add_argument("--probe-script", type=Path, default=DEFAULT_PROBE_SCRIPT)
    parser.add_argument(
        "--hssd-data-path", type=Path, default=PROJECT_ROOT / "data/hssd-models"
    )
    parser.add_argument(
        "--hssd-preprocessed-path",
        type=Path,
        default=PROJECT_ROOT / "data/preprocessed",
    )
    parser.add_argument(
        "--zvec-collection-path",
        type=Path,
        default=PROJECT_ROOT / "data/hssd_zvec_collection",
    )
    parser.add_argument("--zvec-base-url", default="http://127.0.0.1:8014")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Limit objects for a smoke run; 0 means all prompt objects.",
    )
    parser.add_argument("--render-width", type=int, default=384)
    parser.add_argument("--render-height", type=int, default=384)
    parser.add_argument(
        "--render-cache-root",
        type=Path,
        default=DEFAULT_RENDER_CACHE_ROOT,
        help="Reuse named renders from this cache root before falling back to Blender.",
    )
    parser.add_argument(
        "--render-views",
        nargs="+",
        default=list(DEFAULT_RENDER_VIEWS),
        help="Named views to keep per candidate. Default keeps only top and iso.",
    )
    parser.add_argument(
        "--render-background",
        default="#cfd3d8",
        help=(
            "Legacy compatibility flag. Named-view cache/re-render outputs already "
            "use the renderer's neutral background."
        ),
    )
    parser.add_argument("--num-side-views", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument(
        "--include-vertical-views",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.max_queries < 0:
        raise ValueError("--max-queries must be >= 0")
    if args.render_width < 64 or args.render_height < 64:
        raise ValueError("--render-width and --render-height must be >= 64")
    normalized_views: list[str] = []
    for view_name in args.render_views:
        normalized = view_name.strip().lower()
        if normalized not in SUPPORTED_NAMED_VIEWS:
            raise ValueError(
                f"--render-views only supports: {sorted(SUPPORTED_NAMED_VIEWS)}"
            )
        if normalized not in normalized_views:
            normalized_views.append(normalized)
    if not normalized_views:
        raise ValueError("--render-views must include at least one named view")
    args.render_views = normalized_views
    try:
        ImageColor.getrgb(args.render_background)
    except ValueError as exc:
        raise ValueError("--render-background must be a valid PIL color") from exc


def main() -> int:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "hssd_embedding_vs_openclip.md"

    cases = parse_probe_cases(args.probe_script)
    specs = build_query_specs(args.max_queries)
    LOGGER.info("Writing probe output to %s", args.output_dir)
    LOGGER.info("Running %d object queries", len(specs))
    if args.render_cache_root.exists():
        LOGGER.info(
            "Reusing cached HSSD renders from %s for views=%s",
            args.render_cache_root,
            args.render_views,
        )
        render_cache_root: Path | None = args.render_cache_root
    else:
        LOGGER.warning(
            "Render cache root does not exist, will render missing views from meshes: %s",
            args.render_cache_root,
        )
        render_cache_root = None

    # 2026-07-09 修改原因：让 --help/参数校验先完成，再加载 OpenCLIP/Torch，避免轻量命令也卡在重依赖导入上。
    from scenesmith.agent_utils.hssd_retrieval.retrieval import HssdRetriever

    embedding_retriever = HssdRetriever(
        make_hssd_config(
            data_path=args.hssd_data_path,
            preprocessed_path=args.hssd_preprocessed_path,
            backend="embedding",
            top_k=args.top_k,
            zvec_collection_path=args.zvec_collection_path,
            zvec_base_url=args.zvec_base_url,
        )
    )
    openclip_retriever = HssdRetriever(
        make_hssd_config(
            data_path=args.hssd_data_path,
            preprocessed_path=args.hssd_preprocessed_path,
            backend="clip",
            top_k=args.top_k,
            zvec_collection_path=args.zvec_collection_path,
            zvec_base_url=args.zvec_base_url,
        )
    )
    from scenesmith.agent_utils.blender.renderer import BlenderRenderer

    renderer = BlenderRenderer()

    all_results: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        query_slug = f"{index:02d}_{slugify(spec.case_id)}_{slugify(spec.object_name)}"
        query_dir = args.output_dir / query_slug
        LOGGER.info(
            "[%d/%d] %s / %s (%s)",
            index,
            len(specs),
            spec.case_id,
            spec.object_name,
            spec.query,
        )

        method_payloads: dict[str, dict[str, Any]] = {}
        for method_key, retriever in [
            ("hssd_embedding", embedding_retriever),
            ("openclip", openclip_retriever),
        ]:
            method_dir = query_dir / method_key
            results = render_results(
                retriever=retriever,
                renderer=renderer,
                spec=spec,
                method_dir=method_dir,
                method_key=method_key,
                width=args.render_width,
                height=args.render_height,
                cache_root=render_cache_root,
                render_views=args.render_views,
            )
            montage = results[0].montage_path if results else None
            method_payloads[method_key] = {
                "montage_path": (
                    montage.resolve().relative_to(args.output_dir.resolve()).as_posix()
                    if montage is not None
                    else None
                ),
                "results": [
                    result_to_json(result, args.output_dir) for result in results
                ],
            }

        all_results.append(
            {
                "query": {
                    "case_id": spec.case_id,
                    "object_name": spec.object_name,
                    "query": spec.query,
                    "object_type": spec.object_type,
                },
                "methods": method_payloads,
            }
        )

    write_json(all_results, args.output_dir)
    write_markdown(
        output_dir=args.output_dir,
        report_path=report_path,
        all_results=all_results,
        cases=cases,
        args=args,
    )
    LOGGER.info("Wrote Markdown report: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
