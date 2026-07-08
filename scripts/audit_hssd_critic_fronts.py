#!/usr/bin/env python3
"""Audit HSSD chair front annotations against critic lookup metadata.

# 2026-07-08 修改原因：新增 HSSD/critic front 审查脚本，支持用 HSSD embedding
# 抽样椅子资产，并导出多视角渲染与 HTML 报告，方便人工核验正面标注是否正确。
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import random
import shutil
import tempfile
import time
import urllib.error
import urllib.request

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import zvec

from scenesmith.agent_utils.blender.renderer import BlenderRenderer
from scenesmith.agent_utils.clip_embeddings import (
    compute_clip_similarities,
    get_text_embedding,
)
from scenesmith.agent_utils.hssd_retrieval.data_loader import (
    HssdMeshMetadata,
    construct_hssd_mesh_path,
    load_preprocessed_data,
)
from scenesmith.scenebenchmark_critic.asset_library_annotations import (
    get_hssd_asset_annotations,
)

LOGGER = logging.getLogger(__name__)

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


class LlamaTextEmbeddingClient:
    """Minimal llama.cpp embeddings client for zvec text retrieval."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 120.0,
        embd_normalize: int = 2,
        request_retries: int = 2,
        retry_sleep_seconds: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.embd_normalize = embd_normalize
        self.request_retries = request_retries
        self.retry_sleep_seconds = retry_sleep_seconds

    def embed_text(self, text: str) -> list[float]:
        payloads = [
            {"content": text, "embd_normalize": self.embd_normalize},
            {
                "content": {"prompt_string": text, "multimodal_data": []},
                "embd_normalize": self.embd_normalize,
            },
            {"input": text, "embd_normalize": self.embd_normalize},
        ]

        last_error: Exception | None = None
        for payload in payloads:
            try:
                body = self._post_embeddings(payload)
                embedding = extract_embedding(json.loads(body))
                if embedding:
                    return embedding
                last_error = RuntimeError("Empty embedding response")
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to embed text query")

    def _post_embeddings(self, payload: dict[str, Any]) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.request_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"Embedding request failed: HTTP {exc.code}: {detail[:500]}"
                )
                if 400 <= exc.code < 500 and exc.code not in {408, 429}:
                    raise last_error from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = RuntimeError(f"Embedding request failed: {exc}")

            if attempt < self.request_retries:
                time.sleep(self.retry_sleep_seconds * (attempt + 1))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Embedding request failed")


def extract_embedding(response: Any) -> list[float]:
    if isinstance(response, list) and response and isinstance(response[0], dict):
        embedding = response[0].get("embedding") or response[0].get("embeddings")
    elif isinstance(response, list):
        embedding = response
    elif isinstance(response, dict):
        embedding = (
            response.get("embedding")
            or response.get("embeddings")
            or (response.get("data") or [{}])[0].get("embedding")
        )
    else:
        embedding = None

    while (
        isinstance(embedding, list)
        and len(embedding) == 1
        and isinstance(embedding[0], list)
    ):
        embedding = embedding[0]

    if not isinstance(embedding, list):
        return []
    return [float(value) for value in embedding]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit HSSD chair front annotations by sampling assets with HSSD "
            "retrieval, rendering standard views, and generating an HTML report."
        )
    )
    parser.add_argument(
        "--query",
        default="dining chair",
        help="Text query used for HSSD embedding retrieval.",
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
        default=40,
        help="Top-k retrieved assets to build the sampling pool from.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=12,
        help="How many assets to audit from the retrieved pool.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used when sampling from the retrieval pool.",
    )
    parser.add_argument(
        "--name-filter",
        default="chair",
        help="Optional case-insensitive substring filter on name/wordnet/id.",
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
        default=Path("output/hssd_critic_front_audit"),
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
) -> tuple[list[str], dict[str, HssdMeshMetadata]]:
    index_path = preprocessed_path / "hssd_wnsynsetkey_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    with index_path.open("r", encoding="utf-8") as file:
        index_data = json.load(file)

    metadata_by_id: dict[str, HssdMeshMetadata] = {}
    object_ids: list[str] = []
    for wordnet_key, entries in index_data.items():
        for entry in entries:
            metadata = HssdMeshMetadata(
                mesh_id=entry["id"],
                name=entry.get("name", ""),
                up=entry.get("up", ""),
                front=entry.get("front", ""),
                wordnet_key=wordnet_key,
            )
            metadata_by_id[metadata.mesh_id] = metadata
            object_ids.append(metadata.mesh_id)

    return object_ids, metadata_by_id


def load_hssd_clip_index(preprocessed_path: Path) -> tuple[np.ndarray, list[str]]:
    preprocessed = load_preprocessed_data(preprocessed_path=preprocessed_path)
    return preprocessed.clip_embeddings, preprocessed.embedding_index


def _filter_and_sample_ranked_assets(
    ranked: list[tuple[str, float]],
    metadata_by_id: dict[str, HssdMeshMetadata],
    top_k: int,
    sample_count: int,
    seed: int,
    name_filter: str | None,
) -> list[tuple[str, float]]:
    filtered: list[tuple[str, float]] = []
    filter_text = (name_filter or "").strip().lower()
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
        if filter_text and filter_text not in haystack:
            continue
        filtered.append((mesh_id, score))
        if len(filtered) >= top_k:
            break

    if not filtered:
        raise RuntimeError(
            f"No HSSD assets matched name_filter='{name_filter}' within the retrieval pool."
        )

    if sample_count >= len(filtered):
        return filtered

    rng = random.Random(seed)
    sampled = rng.sample(filtered, sample_count)
    sampled.sort(key=lambda item: item[1], reverse=True)
    return sampled


def retrieve_assets_clip(
    query: str,
    embeddings: np.ndarray,
    object_ids: list[str],
    metadata_by_id: dict[str, HssdMeshMetadata],
    top_k: int,
    sample_count: int,
    seed: int,
    name_filter: str | None,
) -> list[tuple[str, float]]:
    query_embedding = get_text_embedding(query)
    similarities = compute_clip_similarities(
        query_embedding=query_embedding,
        embeddings=embeddings,
        indices=list(range(len(object_ids))),
    )

    ranked = [
        (object_ids[idx], float(score))
        for idx, score in sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    ]
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
    name_filter: str | None,
    collection_path: Path,
    embedding_base_url: str,
) -> list[tuple[str, float]]:
    client = LlamaTextEmbeddingClient(base_url=embedding_base_url)
    query_embedding = client.embed_text(query)
    collection = zvec.open(
        path=str(collection_path),
        option=zvec.CollectionOption(read_only=True, enable_mmap=True),
    )
    docs = collection.query(
        queries=zvec.Query(field_name="embedding", vector=query_embedding),
        topk=max(top_k * 4, sample_count * 4, top_k),
        output_fields=["asset_id", "name", "wordnet_key"],
        include_vector=False,
    )

    ranked: list[tuple[str, float]] = []
    seen: set[str] = set()
    for doc in docs:
        fields = doc.fields or {}
        mesh_id = str(fields.get("asset_id") or doc.id or "").strip().lower()
        if not mesh_id or mesh_id in seen:
            continue
        seen.add(mesh_id)
        ranked.append((mesh_id, float(doc.score or 0.0)))

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
    # GLB (X, Y, Z) -> Blender (X, Z, -Y)
    return np.array([glb_vector[0], glb_vector[2], -glb_vector[1]], dtype=float)


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
        width=width,
        height=height,
    )
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

    target_glb_front = np.array([0.0, 0.0, 1.0], dtype=float)
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
        )

    result: dict[str, str] = {}
    for view_name in ("front", "top"):
        path = output_dir / f"{view_name}.png"
        if path.exists():
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
        f"query=<code>{html.escape(str(run_summary['query']))}</code>, ",
        f"top_k={run_summary['top_k']}, ",
        f"sample_count={run_summary['sample_count']}, ",
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


def resolve_asset_list(
    args: argparse.Namespace,
    metadata_by_id: dict[str, HssdMeshMetadata],
) -> list[tuple[str, float]]:
    if args.asset_id:
        resolved: list[tuple[str, float]] = []
        for mesh_id in args.asset_id:
            if mesh_id not in metadata_by_id:
                raise KeyError(f"Unknown HSSD asset id: {mesh_id}")
            resolved.append((mesh_id, 0.0))
        return resolved
    if args.search_mode == "zvec":
        return retrieve_assets_zvec(
            query=args.query,
            metadata_by_id=metadata_by_id,
            top_k=args.top_k,
            sample_count=args.sample_count,
            seed=args.seed,
            name_filter=args.name_filter,
            collection_path=args.zvec_collection_path,
            embedding_base_url=args.embedding_base_url,
        )
    embeddings, object_ids = load_hssd_clip_index(args.preprocessed_path)
    return retrieve_assets_clip(
        query=args.query,
        embeddings=embeddings,
        object_ids=object_ids,
        metadata_by_id=metadata_by_id,
        top_k=args.top_k,
        sample_count=args.sample_count,
        seed=args.seed,
        name_filter=args.name_filter,
    )


def audit_asset(
    mesh_id: str,
    score: float,
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
    return build_asset_summary(
        mesh_id=mesh_id,
        score=score,
        metadata=metadata,
        record=record,
        rendered_views=rendered_views,
        asset_dir=asset_dir,
        clean_critic_views=clean_critic_views,
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, metadata_by_id = load_hssd_metadata_index(args.preprocessed_path)
    assets_to_audit = resolve_asset_list(args, metadata_by_id)
    LOGGER.info("Auditing %d assets", len(assets_to_audit))

    asset_summaries: list[dict[str, Any]] = []
    for mesh_id, score in assets_to_audit:
        LOGGER.info("Rendering audit views for %s", mesh_id)
        summary = audit_asset(
            mesh_id=mesh_id,
            score=score,
            metadata=metadata_by_id.get(mesh_id),
            args=args,
        )
        asset_summaries.append(summary)

    run_summary = {
        "search_mode": args.search_mode,
        "query": args.query,
        "top_k": args.top_k,
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
