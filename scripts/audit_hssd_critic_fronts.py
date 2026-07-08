#!/usr/bin/env python3
"""Audit HSSD chair front annotations against critic lookup metadata.

# 2026-07-08 修改原因：新增 HSSD/critic front 审查脚本，支持用 HSSD embedding
# 抽样椅子资产，并导出多视角渲染与 HTML 报告，方便人工核验正面标注是否正确。
# 2026-07-08 修改原因：zvec/clip 检索改用 HssdZvecSearcher / clip_get_top_k_similar_meshes，
# 走 WordNet object_categories 过滤，避免 wall-art 等非目标类混入检索结果。
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import random
import re
import shutil
import tempfile

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from PIL import Image, ImageEnhance, ImageOps

from scenesmith.agent_utils.blender.renderer import BlenderRenderer
from scenesmith.agent_utils.hssd_retrieval.clip_similarity import (
    get_top_k_similar_meshes as clip_get_top_k,
)
from scenesmith.agent_utils.hssd_retrieval.config import HssdZvecConfig
from scenesmith.agent_utils.hssd_retrieval.data_loader import (
    HssdMeshMetadata,
    HssdPreprocessedData,
    construct_hssd_mesh_path,
    load_preprocessed_data,
)
from scenesmith.agent_utils.hssd_retrieval.zvec_similarity import (
    HssdZvecSearcher,
)
from scenesmith.scenebenchmark_critic.asset_library_annotations import (
    get_hssd_asset_annotations,
)

LOGGER = logging.getLogger(__name__)

# 2026-07-08 修改原因：默认审计从单一 chair query 扩展为多类少量抽样。
DEFAULT_AUDIT_KEYWORDS = ["bed", "chair", "Cabinet", "Monitor", "TV"]

# 2026-07-08 修改原因：zvec 语义召回会把 bedroom/wall-art 等相关但非目标类资产带入。
DEFAULT_KEYWORD_FILTER_TERMS = {
    "bed": ["bed", "mattress", "daybed", "bunk", "headboard"],
    "chair": ["chair", "armchair", "stool", "seat"],
    "cabinet": [
        "cabinet",
        "dresser",
        "wardrobe",
        "armoire",
        "cupboard",
        "sideboard",
        "drawer",
    ],
    "monitor": ["monitor", "screen", "display"],
    "tv": ["tv", "television", "media console", "tv stand"],
}

# 2026-07-08 修改原因：白色 HSSD 资产在默认高亮透明背景下难以人工辨认。
AUDIT_LIGHT_ENERGY = 850.0
AUDIT_BACKGROUND_GREY = 104


VIEW_SPECS: list[tuple[str, np.ndarray, str]] = [
    ("0_top", np.array([0.0, 0.0, 1.0]), "top / +Z"),
    ("1_bottom", np.array([0.0, 0.0, -1.0]), "bottom / -Z"),
    ("2_side", np.array([1.0, 0.0, 0.0]), "side / +X"),
    ("3_side", np.array([0.0, 1.0, 0.0]), "side / +Y"),
    ("4_side", np.array([-1.0, 0.0, 0.0]), "side / -X"),
    ("5_side", np.array([0.0, -1.0, 0.0]), "side / -Y"),
]


@dataclass
class FrontMatch:
    """Best matching rendered view for a front vector."""

    source_name: str
    raw_vector: list[float] | None
    glb_axis: str | None
    blender_vector: list[float] | None
    blender_axis: str | None
    selected_view_name: str | None
    selected_view_label: str | None
    selected_view_index: int | None
    alignment_score: float | None
    note: str | None = None


@dataclass
class AuditTarget:
    """One asset selected for a keyword audit run."""

    mesh_id: str
    score: float
    keyword: str


# 2026-07-08 修改原因：将默认审计关键词映射到 HSSD
# preprocessed_data.object_categories 里的过滤类别，配合
# HssdZvecSearcher / clip_get_top_k 按 WordNet 键过滤。
_KEYWORD_HSSD_CATEGORY: dict[str, str] = {
    "bed": "large_objects",
    "chair": "large_objects",
    "cabinet": "large_objects",
    "monitor": "small_objects",
    "tv": "large_objects",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit HSSD front annotations by sampling assets with HSSD "
            "retrieval, rendering standard views, and generating an HTML report."
        )
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help=(
            "Keyword to audit. Repeat or pass comma-separated values. "
            f"Defaults to: {', '.join(DEFAULT_AUDIT_KEYWORDS)}."
        ),
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Backward-compatible single keyword alias appended to --keyword.",
    )
    parser.add_argument(
        "--search-mode",
        choices=("zvec", "clip"),
        default="zvec",
        help="Retrieval backend. Defaults to zvec + llama.cpp embeddings.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k retrieved assets per keyword to build each sampling pool from.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="How many assets to audit per keyword from the retrieved pool.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used when sampling from the retrieval pool.",
    )
    parser.add_argument(
        "--name-filter",
        default=None,
        help=(
            "Optional comma-separated substring filter on name/wordnet/id. "
            "Overrides the default per-keyword filters."
        ),
    )
    parser.add_argument(
        "--asset-id",
        action="append",
        default=[],
        help="Explicit HSSD asset id to audit. Can be provided multiple times.",
    )
    parser.add_argument(
        "--hssd-root",
        type=Path,
        default=Path("data/hssd-models"),
        help="Path to the HSSD asset root.",
    )
    parser.add_argument(
        "--preprocessed-path",
        type=Path,
        default=Path("data/preprocessed"),
        help="Path to HSSD preprocessed embeddings/metadata.",
    )
    parser.add_argument(
        "--zvec-collection-path",
        type=Path,
        default=Path("data/hssd_zvec_collection"),
        help="Path to the HSSD zvec collection used for semantic retrieval.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default="http://127.0.0.1:8014",
        help="llama.cpp embedding server base URL for zvec retrieval.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hssd_critic_front_audit"),
        help="Directory to store rendered images, JSON, and HTML report.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Render width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Render height in pixels.",
    )
    return parser.parse_args()


def load_hssd_metadata_index(
    preprocessed_path: Path,
) -> tuple[list[str], dict[str, HssdMeshMetadata], HssdPreprocessedData]:
    # 2026-07-08 修改原因：改用 load_preprocessed_data 统一加载，
    # 返回 HssdPreprocessedData 供 zvec/clip searcher 按 WordNet 类别过滤。
    preprocessed = load_preprocessed_data(preprocessed_path=preprocessed_path)
    metadata_by_id: dict[str, HssdMeshMetadata] = {}
    object_ids: list[str] = []
    for wordnet_key, entries in preprocessed.metadata_by_wordnet.items():
        for metadata in entries:
            metadata_by_id[metadata.mesh_id] = metadata
            object_ids.append(metadata.mesh_id)
    return object_ids, metadata_by_id, preprocessed


def _split_filter_terms(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    raw_values = value if isinstance(value, list) else [value]
    terms: list[str] = []
    for raw_value in raw_values:
        for part in str(raw_value).split(","):
            term = part.strip().lower()
            if term:
                terms.append(term)
    return terms


def _default_filter_terms_for_keyword(keyword: str) -> list[str]:
    normalized = keyword.strip().lower()
    return list(DEFAULT_KEYWORD_FILTER_TERMS.get(normalized, [normalized]))


def _effective_name_filter(
    keyword: str, explicit_name_filter: str | None
) -> list[str]:
    if explicit_name_filter is not None:
        return _split_filter_terms(explicit_name_filter)
    return _default_filter_terms_for_keyword(keyword)


def _filter_and_sample_ranked_assets(
    ranked: list[tuple[str, float]],
    metadata_by_id: dict[str, HssdMeshMetadata],
    top_k: int,
    sample_count: int,
    seed: int,
    name_filter: str | list[str] | None,
) -> list[tuple[str, float]]:
    filtered: list[tuple[str, float]] = []
    filter_terms = _split_filter_terms(name_filter)
    # 2026-07-08 修改原因：用 \b 词边界匹配，防止 "bed" 匹配到
    # "bedside" 之类子串，避免 nightstand 等带 bedside 的非床资产误入。
    _filter_re: re.Pattern[str] | None = None
    if filter_terms:
        _filter_re = re.compile(
            r"\b(?:" + "|".join(re.escape(term) for term in filter_terms) + r")\b"
        )
    for mesh_id, score in ranked:
        metadata = metadata_by_id.get(mesh_id)
        haystack = " ".join(
            part
            for part in [
                mesh_id,
                metadata.name if metadata is not None else "",
                metadata.wordnet_key if metadata is not None else "",
            ]
            if part
        ).lower()
        if _filter_re and not _filter_re.search(haystack):
            continue
        filtered.append((mesh_id, score))
        if len(filtered) >= top_k:
            break

    if not filtered:
        raise RuntimeError(
            f"No HSSD assets matched name_filter={filter_terms} "
            f"(word-boundary) within the retrieval pool."
        )

    if sample_count >= len(filtered):
        return filtered

    rng = random.Random(seed)
    sampled = rng.sample(filtered, sample_count)
    sampled.sort(key=lambda item: item[1], reverse=True)
    return sampled


def retrieve_assets_clip(
    query: str,
    metadata_by_id: dict[str, HssdMeshMetadata],
    top_k: int,
    sample_count: int,
    seed: int,
    name_filter: str | list[str] | None,
    preprocessed_data: HssdPreprocessedData,
    hssd_category: str | None,
) -> list[tuple[str, float]]:
    # 2026-07-08 修改原因：改用 clip_get_top_k_similar_meshes，
    # 按 WordNet object_categories 过滤，避免无关类别混入。
    # WordNet 类别过滤 + name_filter 两层过滤会大幅削减候选数。
    fetch_k = max(top_k * 10, sample_count * 10, top_k)
    ranked = clip_get_top_k(
        text_description=query,
        preprocessed_data=preprocessed_data,
        category=hssd_category,
        top_k=fetch_k,
    )
    if not ranked:
        raise RuntimeError(
            f"No HSSD assets matched query='{query}' via clip retrieval."
        )

    return _filter_and_sample_ranked_assets(
        ranked=ranked,
        metadata_by_id=metadata_by_id,
        top_k=top_k,
        sample_count=sample_count,
        seed=seed,
        name_filter=name_filter,
    )


def retrieve_assets_zvec(
    query: str,
    metadata_by_id: dict[str, HssdMeshMetadata],
    top_k: int,
    sample_count: int,
    seed: int,
    name_filter: str | list[str] | None,
    preprocessed_data: HssdPreprocessedData,
    zvec_searcher: HssdZvecSearcher,
    hssd_category: str | None,
) -> list[tuple[str, float]]:
    # 2026-07-08 修改原因：改用 HssdZvecSearcher.get_top_k_similar_meshes，
    # 按 WordNet object_categories 过滤，避免 wall-art 等非目标类混入。
    # 使用较大乘数，因为 searcher 内部有 top_k_factor=4 放大，且
    # WordNet 类别过滤 + name_filter 两层过滤会大幅削减候选数。
    fetch_k = max(top_k * 10, sample_count * 10, top_k)
    ranked = zvec_searcher.get_top_k_similar_meshes(
        text_description=query,
        preprocessed_data=preprocessed_data,
        category=hssd_category,
        top_k=fetch_k,
    )
    if not ranked:
        raise RuntimeError(
            f"No HSSD assets matched query='{query}' via zvec retrieval."
        )

    return _filter_and_sample_ranked_assets(
        ranked=ranked,
        metadata_by_id=metadata_by_id,
        top_k=top_k,
        sample_count=sample_count,
        seed=seed,
        name_filter=name_filter,
    )


def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    return vector / norm


def _vector_to_axis_string(vector: np.ndarray) -> str:
    axis_names = ["X", "Y", "Z"]
    idx = int(np.argmax(np.abs(vector)))
    sign = "+" if float(vector[idx]) >= 0 else "-"
    return f"{sign}{axis_names[idx]}"


def _glb_vector_to_blender_vector(glb_vector: np.ndarray) -> np.ndarray:
    # 2026-07-08 修改原因：Blender glTF import 将 GLB/HSSD (X,Y,Z)
    # 映射为 Blender (X,-Z,Y)，否则椅子 front 会被审计视图反选到椅背。
    return np.array([glb_vector[0], -glb_vector[2], glb_vector[1]], dtype=float)


def parse_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) != 3:
            return None
        try:
            return np.array([float(part) for part in parts], dtype=float)
        except ValueError:
            return None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return np.array([float(part) for part in value], dtype=float)
        except (TypeError, ValueError):
            return None
    return None


def build_front_match(source_name: str, vector_value: Any) -> FrontMatch:
    glb_vector = parse_vector(vector_value)
    if glb_vector is None:
        return FrontMatch(
            source_name=source_name,
            raw_vector=None,
            glb_axis=None,
            blender_vector=None,
            blender_axis=None,
            selected_view_name=None,
            selected_view_label=None,
            selected_view_index=None,
            alignment_score=None,
            note="missing_or_invalid_vector",
        )

    glb_unit = _normalize_vector(glb_vector)
    if glb_unit is None:
        return FrontMatch(
            source_name=source_name,
            raw_vector=glb_vector.tolist(),
            glb_axis=None,
            blender_vector=None,
            blender_axis=None,
            selected_view_name=None,
            selected_view_label=None,
            selected_view_index=None,
            alignment_score=None,
            note="zero_length_vector",
        )

    blender_unit = _normalize_vector(_glb_vector_to_blender_vector(glb_unit))
    if blender_unit is None:
        return FrontMatch(
            source_name=source_name,
            raw_vector=glb_unit.tolist(),
            glb_axis=_vector_to_axis_string(glb_unit),
            blender_vector=None,
            blender_axis=None,
            selected_view_name=None,
            selected_view_label=None,
            selected_view_index=None,
            alignment_score=None,
            note="failed_coordinate_conversion",
        )

    best_name: str | None = None
    best_label: str | None = None
    best_index: int | None = None
    best_score: float | None = None
    for view_index, (view_name, direction, label) in enumerate(VIEW_SPECS):
        score = float(np.dot(blender_unit, direction))
        if best_score is None or score > best_score:
            best_name = view_name
            best_label = label
            best_index = view_index
            best_score = score

    return FrontMatch(
        source_name=source_name,
        raw_vector=glb_unit.tolist(),
        glb_axis=_vector_to_axis_string(glb_unit),
        blender_vector=blender_unit.tolist(),
        blender_axis=_vector_to_axis_string(blender_unit),
        selected_view_name=best_name,
        selected_view_label=best_label,
        selected_view_index=best_index,
        alignment_score=best_score,
    )


def improve_audit_image_visibility(image_path: Path) -> None:
    """Composite transparent renders onto grey and add mild local contrast."""
    image = Image.open(image_path).convert("RGBA")
    alpha = image.getchannel("A")
    background = Image.new(
        "RGB",
        image.size,
        (AUDIT_BACKGROUND_GREY, AUDIT_BACKGROUND_GREY, AUDIT_BACKGROUND_GREY),
    )
    background.paste(image.convert("RGB"), mask=alpha)
    background = ImageOps.autocontrast(background, cutoff=0.2)
    background = ImageEnhance.Contrast(background).enhance(1.18)
    background = ImageEnhance.Sharpness(background).enhance(1.1)
    background.save(image_path)


def render_analysis_views(
    mesh_path: Path,
    output_dir: Path,
    width: int,
    height: int,
) -> dict[str, Path]:
    renderer = BlenderRenderer()
    image_paths = renderer.render_multiview_for_analysis(
        mesh_path=mesh_path,
        output_dir=output_dir,
        elevation_degrees=20.0,
        num_side_views=4,
        include_vertical_views=True,
        light_energy=AUDIT_LIGHT_ENERGY,
        width=width,
        height=height,
    )
    for path in image_paths:
        improve_audit_image_visibility(path)
    return {path.stem: path for path in image_paths}


def _rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return 4x4 rotation matrix mapping source direction onto target.

    # 2026-07-08 修改原因：审查报告里的 clean front/top 应该直接依据 critic
    # front 轴重新渲染，而不是复用 data/hssd_rendered_assets 里的历史参考图。
    """
    source_unit = _normalize_vector(source)
    target_unit = _normalize_vector(target)
    if source_unit is None or target_unit is None:
        return np.eye(4, dtype=float)

    cross = np.cross(source_unit, target_unit)
    dot = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))

    if cross_norm <= 1e-8:
        if dot > 0.999999:
            return np.eye(4, dtype=float)

        orthogonal = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(source_unit[0])) > 0.9:
            orthogonal = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = _normalize_vector(np.cross(source_unit, orthogonal))
        if axis is None:
            return np.eye(4, dtype=float)
        angle = np.pi
    else:
        axis = cross / cross_norm
        angle = float(np.arccos(dot))

    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    t = 1.0 - c
    rotation = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=float,
    )
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    return transform


def render_clean_views_from_front_axis(
    mesh: trimesh.Trimesh,
    glb_front_vector: np.ndarray | None,
    output_dir: Path,
    width: int,
    height: int,
) -> dict[str, str]:
    if glb_front_vector is None:
        return {}

    # 2026-07-08 修改原因：render_named_views_for_embedding 的 front 相机在
    # Blender +Y；该方向对应导入前的 GLB -Z，因此 clean front 需对齐到 GLB -Z。
    target_glb_front = np.array([0.0, 0.0, -1.0], dtype=float)
    aligned_mesh = mesh.copy()
    aligned_mesh.apply_transform(
        _rotation_matrix_from_vectors(glb_front_vector, target_glb_front)
    )

    with tempfile.TemporaryDirectory(prefix="critic_clean_views_") as tmp_dir:
        temp_mesh_path = Path(tmp_dir) / "critic_aligned.glb"
        aligned_mesh.export(temp_mesh_path)

        renderer = BlenderRenderer()
        renderer.render_named_views_for_embedding(
            mesh_path=temp_mesh_path,
            output_dir=output_dir,
            view_names=["front", "top"],
            width=width,
            height=height,
            light_energy=AUDIT_LIGHT_ENERGY,
        )

    result: dict[str, str] = {}
    for view_name in ("front", "top"):
        path = output_dir / f"{view_name}.png"
        if path.exists():
            improve_audit_image_visibility(path)
            result[view_name] = path.name
    return result


def copy_selected_view(
    rendered_views: dict[str, Path],
    match: FrontMatch,
    destination: Path,
) -> str | None:
    if match.selected_view_name is None:
        return None
    source_path = rendered_views.get(match.selected_view_name)
    if source_path is None or not source_path.exists():
        return None
    shutil.copy2(source_path, destination)
    return destination.name


def build_asset_summary(
    mesh_id: str,
    score: float,
    keyword: str,
    metadata: HssdMeshMetadata | None,
    record: dict[str, Any] | None,
    rendered_views: dict[str, Path],
    asset_dir: Path,
    clean_critic_views: dict[str, str],
) -> dict[str, Any]:
    canonical_front = (record or {}).get("canonical_front") or {}
    hints = (record or {}).get("scenebenchmark_functional_hints") or {}

    critic_match = build_front_match(
        source_name="critic_asset_local_front_axis",
        vector_value=canonical_front.get("asset_local_front_axis")
        or hints.get("asset_local_front_axis"),
    )
    hssd_match = build_front_match(
        source_name="hssd_raw_front",
        vector_value=metadata.front if metadata is not None else None,
    )

    critic_selected_rel = copy_selected_view(
        rendered_views=rendered_views,
        match=critic_match,
        destination=asset_dir / "critic_selected.png",
    )
    hssd_selected_rel = copy_selected_view(
        rendered_views=rendered_views,
        match=hssd_match,
        destination=asset_dir / "hssd_selected.png",
    )

    relative_views = {
        name: path.name for name, path in sorted(rendered_views.items(), key=lambda item: item[0])
    }

    return {
        "mesh_id": mesh_id,
        "keyword": keyword,
        "retrieval_score": score,
        "name": metadata.name if metadata is not None else "",
        "wordnet_key": metadata.wordnet_key if metadata is not None else "",
        "hssd_up": metadata.up if metadata is not None else "",
        "hssd_front": metadata.front if metadata is not None else "",
        "critic_front_hint": hints.get("front_hint"),
        "critic_front_face": hints.get("front_face"),
        "critic_front_confidence": hints.get("front_confidence"),
        "critic_validation_status": canonical_front.get("validation_status"),
        "critic_semantic_front": canonical_front.get(
            "canonical_orientation_is_semantic_front"
        ),
        "critic_front_view_index": canonical_front.get("front_view_image_index"),
        "rendered_views": relative_views,
        "critic_match": {
            **critic_match.__dict__,
            "selected_image": critic_selected_rel,
        },
        "hssd_match": {
            **hssd_match.__dict__,
            "selected_image": hssd_selected_rel,
        },
        "critic_clean_front": (
            f"{mesh_id}/{clean_critic_views['front']}"
            if "front" in clean_critic_views
            else None
        ),
        "critic_clean_top": (
            f"{mesh_id}/{clean_critic_views['top']}"
            if "top" in clean_critic_views
            else None
        ),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_asset_info_files(asset_dir: Path, asset: dict[str, Any]) -> None:
    # 2026-07-08 修改原因：单独打开某个 asset 输出目录时，需要直接看到资产名称。
    info = {
        "keyword": asset.get("keyword"),
        "mesh_id": asset.get("mesh_id"),
        "name": asset.get("name"),
        "wordnet_key": asset.get("wordnet_key"),
        "retrieval_score": asset.get("retrieval_score"),
        "critic_front_axis": (asset.get("critic_match") or {}).get("raw_vector"),
        "critic_selected_view": (asset.get("critic_match") or {}).get(
            "selected_view_name"
        ),
        "critic_selected_label": (asset.get("critic_match") or {}).get(
            "selected_view_label"
        ),
    }
    write_json(asset_dir / "asset_info.json", info)

    lines = [
        f"name: {asset.get('name') or ''}",
        f"mesh_id: {asset.get('mesh_id') or ''}",
        f"keyword: {asset.get('keyword') or ''}",
        f"wordnet_key: {asset.get('wordnet_key') or ''}",
    ]
    (asset_dir / "asset_name.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def html_image(rel_path: str | None, label: str, css_class: str = "") -> str:
    if not rel_path:
        return f"<div class='missing {css_class}'>missing {html.escape(label)}</div>"
    class_attr = f" class='{css_class}'" if css_class else ""
    return (
        f"<figure{class_attr}>"
        f"<img src='{html.escape(rel_path)}' alt='{html.escape(label)}'>"
        f"<figcaption>{html.escape(label)}</figcaption>"
        "</figure>"
    )


def write_report(output_dir: Path, run_summary: dict[str, Any]) -> None:
    report_path = output_dir / "report.html"
    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>HSSD Critic Front Audit</title>",
        "<style>",
        "body { font-family: sans-serif; margin: 24px; background: #fafafa; color: #222; }",
        "code { background: #f1f1f1; padding: 2px 4px; }",
        ".asset { background: white; border: 1px solid #ddd; padding: 18px; margin: 20px 0; }",
        ".summary { margin: 0 0 12px 0; line-height: 1.5; }",
        ".views { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; }",
        ".selected { display: grid; grid-template-columns: repeat(4, minmax(220px, 1fr)); gap: 12px; margin: 16px 0; }",
        "figure { margin: 0; border: 1px solid #ccc; background: #fff; padding: 8px; }",
        "figure img { width: 100%; height: auto; display: block; background: #eee; }",
        "figcaption { margin-top: 6px; font-size: 12px; }",
        ".missing { border: 1px dashed #c44; color: #c44; padding: 24px; background: #fff5f5; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>HSSD Critic Front Audit</h1>",
        "<p>",
        f"search_mode=<code>{html.escape(str(run_summary['search_mode']))}</code>, ",
        f"keywords=<code>{html.escape(', '.join(run_summary['keywords']))}</code>, ",
        f"top_k={run_summary['top_k']}, ",
        f"sample_count_per_keyword={run_summary['sample_count_per_keyword']}, ",
        f"total_assets={run_summary['sample_count']}, ",
        f"seed={run_summary['seed']}, ",
        f"name_filter=<code>{html.escape(str(run_summary['name_filter']))}</code>",
        "</p>",
    ]

    for asset in run_summary["assets"]:
        asset_dir = output_dir / asset["mesh_id"]
        lines.extend(
            [
                "<section class='asset'>",
                f"<h2>{html.escape(asset['name'] or asset['mesh_id'])}</h2>",
                "<div class='summary'>",
                f"keyword=<code>{html.escape(str(asset.get('keyword') or ''))}</code><br>",
                f"asset_id=<code>{html.escape(asset['mesh_id'])}</code><br>",
                f"wordnet=<code>{html.escape(asset['wordnet_key'] or '')}</code><br>",
                f"retrieval_score={asset['retrieval_score']:.4f}<br>",
                f"hssd_front=<code>{html.escape(asset['hssd_front'] or '')}</code><br>",
                f"critic_front_hint=<code>{html.escape(str(asset['critic_front_hint']))}</code><br>",
                f"critic_asset_local_front_axis=<code>{html.escape(str(asset['critic_match']['raw_vector']))}</code><br>",
                f"critic_validation_status=<code>{html.escape(str(asset['critic_validation_status']))}</code>",
                "</div>",
                "<div class='selected'>",
                html_image(
                    rel_path=f"{asset['mesh_id']}/{asset['critic_match']['selected_image']}"
                    if asset["critic_match"]["selected_image"]
                    else None,
                    label=(
                        "critic selected view: "
                        f"{asset['critic_match']['selected_view_name']} / "
                        f"{asset['critic_match']['selected_view_label']}"
                    ),
                    css_class="critic",
                ),
                html_image(
                    rel_path=f"{asset['mesh_id']}/{asset['hssd_match']['selected_image']}"
                    if asset["hssd_match"]["selected_image"]
                    else None,
                    label=(
                        "hssd selected view: "
                        f"{asset['hssd_match']['selected_view_name']} / "
                        f"{asset['hssd_match']['selected_view_label']}"
                    ),
                    css_class="hssd",
                ),
                html_image(
                    rel_path=asset["critic_clean_front"],
                    label="critic clean front render",
                    css_class="critic-front",
                ),
                html_image(
                    rel_path=asset["critic_clean_top"],
                    label="critic clean top render",
                    css_class="critic-top",
                ),
                "</div>",
                "<div class='views'>",
            ]
        )
        for view_name, _, label in VIEW_SPECS:
            lines.append(
                html_image(
                    rel_path=f"{asset['mesh_id']}/{asset['rendered_views'].get(view_name)}"
                    if asset["rendered_views"].get(view_name)
                    else None,
                    label=f"{view_name} ({label})",
                )
            )
        lines.extend(["</div>", "</section>"])

        asset_json_path = asset_dir / "summary.json"
        write_json(asset_json_path, asset)

    lines.extend(["</body>", "</html>"])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_audit_keywords(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    for value in [*args.keyword, args.query]:
        if value is None:
            continue
        for part in str(value).split(","):
            keyword = part.strip()
            if keyword:
                values.append(keyword)

    if values:
        return values
    return list(DEFAULT_AUDIT_KEYWORDS)


def resolve_asset_list_for_keyword(
    query: str,
    args: argparse.Namespace,
    metadata_by_id: dict[str, HssdMeshMetadata],
    preprocessed_data: HssdPreprocessedData,
    zvec_searcher: HssdZvecSearcher | None,
) -> list[tuple[str, float]]:
    name_filter = _effective_name_filter(
        keyword=query,
        explicit_name_filter=args.name_filter,
    )
    hssd_category = _KEYWORD_HSSD_CATEGORY.get(query.strip().lower())
    LOGGER.info(
        "Keyword '%s' using name_filter=%s hssd_category=%s",
        query,
        name_filter,
        hssd_category,
    )
    if args.search_mode == "zvec":
        if zvec_searcher is None:
            raise RuntimeError("zvec searcher not initialised")
        return retrieve_assets_zvec(
            query=query,
            metadata_by_id=metadata_by_id,
            top_k=args.top_k,
            sample_count=args.sample_count,
            seed=args.seed,
            name_filter=name_filter,
            preprocessed_data=preprocessed_data,
            zvec_searcher=zvec_searcher,
            hssd_category=hssd_category,
        )
    return retrieve_assets_clip(
        query=query,
        metadata_by_id=metadata_by_id,
        top_k=args.top_k,
        sample_count=args.sample_count,
        seed=args.seed,
        name_filter=name_filter,
        preprocessed_data=preprocessed_data,
        hssd_category=hssd_category,
    )


def resolve_audit_targets(
    args: argparse.Namespace,
    metadata_by_id: dict[str, HssdMeshMetadata],
    preprocessed_data: HssdPreprocessedData,
    zvec_searcher: HssdZvecSearcher | None,
) -> list[AuditTarget]:
    if args.asset_id:
        resolved: list[AuditTarget] = []
        for mesh_id in args.asset_id:
            if mesh_id not in metadata_by_id:
                raise KeyError(f"Unknown HSSD asset id: {mesh_id}")
            resolved.append(AuditTarget(mesh_id=mesh_id, score=0.0, keyword="explicit"))
        return resolved

    targets: list[AuditTarget] = []
    for keyword in parse_audit_keywords(args):
        keyword_assets = resolve_asset_list_for_keyword(
            query=keyword,
            args=args,
            metadata_by_id=metadata_by_id,
            preprocessed_data=preprocessed_data,
            zvec_searcher=zvec_searcher,
        )
        for mesh_id, score in keyword_assets:
            targets.append(AuditTarget(mesh_id=mesh_id, score=score, keyword=keyword))
    return targets


def audit_asset(
    mesh_id: str,
    score: float,
    keyword: str,
    metadata: HssdMeshMetadata | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    mesh_path = construct_hssd_mesh_path(args.hssd_root, mesh_id)
    asset_dir = args.output_dir / mesh_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"hssd_front_audit_{mesh_id[:8]}_") as tmp_dir:
        temp_mesh_path = Path(tmp_dir) / f"{mesh_id}.glb"
        mesh = trimesh.load(mesh_path, force="mesh")
        mesh.export(temp_mesh_path)

        rendered_views = render_analysis_views(
            mesh_path=temp_mesh_path,
            output_dir=asset_dir,
            width=args.width,
            height=args.height,
        )

    record = get_hssd_asset_annotations(mesh_id)
    canonical_front = (record or {}).get("canonical_front") or {}
    hints = (record or {}).get("scenebenchmark_functional_hints") or {}
    critic_front_vector = parse_vector(
        canonical_front.get("asset_local_front_axis")
        or hints.get("asset_local_front_axis")
    )
    clean_critic_views = render_clean_views_from_front_axis(
        mesh=mesh,
        glb_front_vector=critic_front_vector,
        output_dir=asset_dir,
        width=args.width,
        height=args.height,
    )
    summary = build_asset_summary(
        mesh_id=mesh_id,
        score=score,
        keyword=keyword,
        metadata=metadata,
        record=record,
        rendered_views=rendered_views,
        asset_dir=asset_dir,
        clean_critic_views=clean_critic_views,
    )
    write_asset_info_files(asset_dir, summary)
    return summary


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 2026-07-08 修改原因：统一用 load_preprocessed_data 加载元数据和
    # embeddings，供 zvec/clip searcher 按 WordNet 类别过滤。
    _, metadata_by_id, preprocessed_data = load_hssd_metadata_index(
        args.preprocessed_path
    )
    zvec_searcher: HssdZvecSearcher | None = None
    if args.search_mode == "zvec":
        zvec_config = HssdZvecConfig(
            collection_path=args.zvec_collection_path,
            base_url=args.embedding_base_url,
        )
        zvec_searcher = HssdZvecSearcher(zvec_config)

    audit_targets = resolve_audit_targets(
        args, metadata_by_id, preprocessed_data, zvec_searcher
    )
    keywords = ["explicit"] if args.asset_id else parse_audit_keywords(args)
    LOGGER.info(
        "Auditing %d assets across %d keyword(s)",
        len(audit_targets),
        len(keywords),
    )

    asset_summaries: list[dict[str, Any]] = []
    for target in audit_targets:
        LOGGER.info(
            "Rendering audit views for %s (keyword=%s)",
            target.mesh_id,
            target.keyword,
        )
        summary = audit_asset(
            mesh_id=target.mesh_id,
            score=target.score,
            keyword=target.keyword,
            metadata=metadata_by_id.get(target.mesh_id),
            args=args,
        )
        asset_summaries.append(summary)

    run_summary = {
        "search_mode": args.search_mode,
        "keywords": keywords,
        "keyword_filter_terms": {
            keyword: _effective_name_filter(keyword, args.name_filter)
            for keyword in keywords
        },
        "top_k": args.top_k,
        "sample_count_per_keyword": args.sample_count,
        "sample_count": len(asset_summaries),
        "seed": args.seed,
        "name_filter": args.name_filter,
        "assets": asset_summaries,
    }
    write_json(args.output_dir / "report.json", run_summary)
    write_report(args.output_dir, run_summary)
    LOGGER.info("Wrote audit report to %s", args.output_dir / "report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
