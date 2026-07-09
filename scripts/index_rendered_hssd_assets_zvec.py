#!/usr/bin/env python3
"""Index rendered HSSD asset images into a local Zvec collection.

This script expects rendered images to already exist. It groups images by HSSD
asset id, embeds the selected views with a llama.cpp Qwen3-VL-Embedding server,
and upserts one Zvec document per asset.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import gc
import json
import logging
import multiprocessing as mp
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zvec

from tqdm import tqdm

LOGGER = logging.getLogger("index_rendered_hssd_assets_zvec")

# 2026-07-09 修改原因：并行渲染 worker 启动时 Pydantic 会重复打印无害的 Field(repr/frozen) warning，
# 会淹没 ACP 日志里的 tqdm 进度；这里只过滤这类第三方库噪声。
warnings.filterwarnings(
    "ignore",
    message=r"The '(repr|frozen)' attribute .*`?Field\(\)`? function.*",
)

HSSD_ID_RE = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{40})(?![0-9a-fA-F])")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_RETRIEVAL_VIEWS = ("front", "back", "left", "right", "top", "iso")
VIEW_PRIORITY = (*DEFAULT_RETRIEVAL_VIEWS, "bottom", "side", "image")


@dataclass
class HssdMeshMetadata:
    mesh_id: str
    name: str
    up: str
    front: str
    wordnet_key: str


@dataclass
class RenderedAsset:
    asset_id: str
    image_paths: dict[str, Path]
    metadata: HssdMeshMetadata | None
    object_groups: list[str]
    asset_path: Path | None


@dataclass
class RenderJob:
    asset_id: str
    asset_path: Path
    metadata: HssdMeshMetadata | None
    render_root: Path
    view_names: list[str]
    overwrite: bool
    width: int
    height: int


@dataclass
class RenderJobResult:
    asset_id: str
    rendered: bool
    error: str | None = None


_WORKER_RENDERER: Any | None = None


class LlamaEmbeddingClient:
    """Small client for llama.cpp's native /embeddings endpoint."""

    def __init__(
        self,
        base_url: str,
        media_marker: str = "<__media__>",
        timeout_seconds: float = 120.0,
        embd_normalize: int = 2,
        request_retries: int = 2,
        retry_sleep_seconds: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.media_marker = media_marker
        self.timeout_seconds = timeout_seconds
        self.embd_normalize = embd_normalize
        self.request_retries = request_retries
        self.retry_sleep_seconds = retry_sleep_seconds

    def props(self) -> dict[str, Any]:
        with urllib.request.urlopen(
            f"{self.base_url}/props", timeout=self.timeout_seconds
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    def embed_image_text(self, image_path: Path, prompt: str) -> list[float]:
        return self.embed_images_text([image_path], prompt)

    def embed_images_text(self, image_paths: list[Path], prompt: str) -> list[float]:
        if not image_paths:
            raise ValueError("At least one image is required for multimodal embedding")

        prompt_with_marker = prompt
        marker_count = prompt_with_marker.count(self.media_marker)
        if marker_count == 0:
            markers = " ".join(
                f"image_{idx}: {self.media_marker}"
                for idx in range(1, len(image_paths) + 1)
            )
            prompt_with_marker = f"{prompt.rstrip()} {markers}"
        elif marker_count != len(image_paths):
            raise ValueError(
                f"Prompt has {marker_count} media markers but got "
                f"{len(image_paths)} images"
            )

        payload = {
            "content": {
                "prompt_string": prompt_with_marker,
                "multimodal_data": [
                    base64.b64encode(image_path.read_bytes()).decode("ascii")
                    for image_path in image_paths
                ],
            },
            "embd_normalize": self.embd_normalize,
        }
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        body = self._urlopen_with_retries(request, image_paths[0])

        embedding = _extract_embedding(json.loads(body))
        if not embedding:
            raise RuntimeError(f"Empty embedding response for {image_paths}")
        return embedding

    def _urlopen_with_retries(
        self, request: urllib.request.Request, image_path: Path
    ) -> str:
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
                    f"Embedding request failed for {image_path}: "
                    f"HTTP {exc.code}: {detail[:500]}"
                )
                if 400 <= exc.code < 500 and exc.code not in {408, 429}:
                    raise last_error from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = RuntimeError(
                    f"Embedding request failed for {image_path}: {exc}"
                )

            if attempt < self.request_retries:
                time.sleep(self.retry_sleep_seconds * (attempt + 1))

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Embedding request failed for {image_path}")


def _extract_embedding(response: Any) -> list[float]:
    """Handle llama.cpp embedding response variants."""
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


def load_hssd_lookup(
    preprocessed_path: Path | None,
) -> tuple[dict[str, HssdMeshMetadata], dict[str, list[str]]]:
    if preprocessed_path is None:
        return {}, {}

    index_path = preprocessed_path / "hssd_wnsynsetkey_index.json"
    categories_path = preprocessed_path / "object_categories.json"
    if not index_path.exists():
        LOGGER.warning("HSSD metadata index not found: %s", index_path)
        return {}, {}

    with index_path.open("r", encoding="utf-8") as file:
        index_data = json.load(file)

    metadata_by_id: dict[str, HssdMeshMetadata] = {}
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

    groups_by_wordnet: dict[str, list[str]] = {}
    if categories_path.exists():
        with categories_path.open("r", encoding="utf-8") as file:
            categories_data = json.load(file)
        for group, wordnet_keys in categories_data.items():
            if group == "available_categories":
                continue
            for wordnet_key in wordnet_keys:
                groups_by_wordnet.setdefault(wordnet_key, []).append(group)

    return metadata_by_id, groups_by_wordnet


def discover_rendered_assets(
    render_root: Path,
    metadata_by_id: dict[str, HssdMeshMetadata],
    groups_by_wordnet: dict[str, list[str]],
    hssd_root: Path | None,
    include_views: set[str] | None,
    require_metadata: bool = True,
) -> list[RenderedAsset]:
    grouped: dict[str, dict[str, Path]] = {}
    for image_path in sorted(render_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        asset_id, view = infer_asset_id_and_view(image_path, render_root)
        if include_views is not None and view not in include_views:
            continue
        grouped.setdefault(asset_id, {})[view] = image_path

    assets: list[RenderedAsset] = []
    skipped_without_metadata = 0
    for asset_id, image_paths in sorted(grouped.items()):
        metadata = metadata_by_id.get(asset_id)
        if require_metadata and metadata is None:
            skipped_without_metadata += 1
            continue
        object_groups = (
            sorted(groups_by_wordnet.get(metadata.wordnet_key, []))
            if metadata is not None
            else []
        )
        asset_path = construct_hssd_glb_path(hssd_root, asset_id)
        assets.append(
            RenderedAsset(
                asset_id=asset_id,
                image_paths=sort_views(image_paths),
                metadata=metadata,
                object_groups=object_groups,
                asset_path=asset_path,
            )
        )
    if skipped_without_metadata:
        # 2026-07-09 修改原因：HSSD zvec 检索只应索引带 WordNet metadata 的资产，避免无标注渲染图污染重建后的向量库。
        LOGGER.info(
            "Skipped %d rendered asset directories without HSSD metadata",
            skipped_without_metadata,
        )
    return assets


def infer_asset_id_and_view(image_path: Path, render_root: Path) -> tuple[str, str]:
    rel_text = image_path.relative_to(render_root).as_posix()
    match = HSSD_ID_RE.search(rel_text)
    if match:
        asset_id = match.group(1).lower()
    elif image_path.parent != render_root:
        asset_id = image_path.parent.name
    else:
        asset_id = image_path.stem

    return asset_id, infer_view_name(image_path)


def infer_view_name(image_path: Path) -> str:
    text = " ".join(
        part.lower() for part in (*image_path.parts[-3:-1], image_path.stem)
    )
    for view in VIEW_PRIORITY:
        if view != "image" and re.search(rf"(^|[_\-\s]){view}($|[_\-\s])", text):
            return view
    return image_path.stem.lower() or "image"


def sort_views(image_paths: dict[str, Path]) -> dict[str, Path]:
    priority = {view: idx for idx, view in enumerate(VIEW_PRIORITY)}
    return dict(
        sorted(
            image_paths.items(),
            key=lambda item: (priority.get(item[0], len(priority)), item[0]),
        )
    )


def construct_hssd_glb_path(hssd_root: Path | None, asset_id: str) -> Path | None:
    if hssd_root is None or not HSSD_ID_RE.fullmatch(asset_id):
        return None
    path = hssd_root / "objects" / asset_id[0] / f"{asset_id}.glb"
    return path if path.exists() else None


def list_hssd_glb_assets(
    hssd_root: Path,
    metadata_by_id: dict[str, HssdMeshMetadata],
    limit: int | None,
) -> list[tuple[str, Path, HssdMeshMetadata | None]]:
    assets: list[tuple[str, Path, HssdMeshMetadata | None]] = []
    skipped_without_metadata = 0
    for asset_path in sorted((hssd_root / "objects").glob("*/*.glb")):
        asset_id = asset_path.stem.lower()
        metadata = metadata_by_id.get(asset_id)
        if metadata is None:
            skipped_without_metadata += 1
            continue
        assets.append((asset_id, asset_path, metadata))
        if limit is not None and len(assets) >= limit:
            break
    if skipped_without_metadata:
        # 2026-07-09 修改原因：补渲染也跳过无 metadata 的 HSSD GLB，保持渲染集和最终 zvec 索引一致。
        LOGGER.info(
            "Skipped %d HSSD GLB assets without metadata during render discovery",
            skipped_without_metadata,
        )
    return assets


def missing_render_views(
    render_root: Path,
    asset_id: str,
    render_views: list[str],
    overwrite: bool,
) -> list[str]:
    if overwrite:
        return render_views

    asset_dir = render_root / asset_id
    missing: list[str] = []
    for view_name in render_views:
        output_path = asset_dir / f"{view_name}.png"
        if not output_path.exists():
            missing.append(view_name)
    return missing


def _render_hssd_asset_job(job: RenderJob) -> RenderJobResult:
    """Render missing named views for one HSSD asset.

    This function is process-pool friendly: all bpy imports happen inside the
    child process, and the BlenderRenderer instance is cached per worker.
    """
    global _WORKER_RENDERER

    # Local imports keep pure indexing usage lightweight when rendering is skipped.
    import trimesh

    from scenesmith.agent_utils.blender.renderer import BlenderRenderer
    from scenesmith.agent_utils.hssd_retrieval.alignment import (
        apply_hssd_alignment_transform,
    )

    asset_render_dir = job.render_root / job.asset_id
    asset_render_dir.mkdir(parents=True, exist_ok=True)

    if job.overwrite:
        for view_name in job.view_names:
            output_path = asset_render_dir / f"{view_name}.png"
            if output_path.exists():
                output_path.unlink()

    try:
        if _WORKER_RENDERER is None:
            _WORKER_RENDERER = BlenderRenderer()

        with tempfile.TemporaryDirectory(
            prefix=f"hssd_render_{job.asset_id[:8]}_"
        ) as tmp:
            aligned_path = Path(tmp) / f"{job.asset_id}.glb"
            mesh = trimesh.load(job.asset_path, force="mesh")
            if job.metadata is not None:
                mesh = apply_hssd_alignment_transform(mesh, job.metadata)
            mesh.export(aligned_path)
            _WORKER_RENDERER.render_named_views_for_embedding(
                mesh_path=aligned_path,
                output_dir=asset_render_dir,
                view_names=job.view_names,
                width=job.width,
                height=job.height,
            )
        return RenderJobResult(asset_id=job.asset_id, rendered=True)
    except Exception as exc:
        return RenderJobResult(asset_id=job.asset_id, rendered=False, error=str(exc))


def render_hssd_assets_if_needed(
    render_root: Path,
    hssd_root: Path | None,
    metadata_by_id: dict[str, HssdMeshMetadata],
    render_views: list[str],
    limit: int | None,
    overwrite: bool,
    width: int,
    height: int,
    render_workers: int,
) -> int:
    if hssd_root is None:
        raise ValueError("HSSD root is required for rendering")
    if not render_views:
        raise ValueError("At least one render view is required")

    render_root.mkdir(parents=True, exist_ok=True)
    jobs: list[RenderJob] = []
    for asset_id, asset_path, metadata in list_hssd_glb_assets(
        hssd_root, metadata_by_id, limit
    ):
        needed_views = missing_render_views(
            render_root=render_root,
            asset_id=asset_id,
            render_views=render_views,
            overwrite=overwrite,
        )
        if not needed_views:
            continue

        jobs.append(
            RenderJob(
                asset_id=asset_id,
                asset_path=asset_path,
                metadata=metadata,
                render_root=render_root,
                view_names=needed_views,
                overwrite=overwrite,
                width=width,
                height=height,
            )
        )

    if not jobs:
        LOGGER.info("No missing HSSD render views found")
        return 0

    render_workers = max(1, render_workers)
    LOGGER.info(
        "Rendering %d HSSD assets with %d worker(s)",
        len(jobs),
        render_workers,
    )

    # 2026-07-09 修改原因：HSSD 多视角补图很慢，允许多个独立 Blender 子进程并行渲染缺失视角。
    if render_workers == 1:
        results = [
            _render_hssd_asset_job(job)
            for job in tqdm(jobs, desc="Rendering HSSD assets")
        ]
    else:
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=render_workers,
            mp_context=context,
        ) as executor:
            results = list(
                tqdm(
                    executor.map(_render_hssd_asset_job, jobs),
                    total=len(jobs),
                    desc="Rendering HSSD assets",
                )
            )

    rendered_assets = 0
    failed_assets = 0
    for result in results:
        if result.rendered:
            rendered_assets += 1
        else:
            failed_assets += 1
            LOGGER.warning(
                "Failed to render HSSD asset %s: %s",
                result.asset_id,
                result.error,
            )

    if failed_assets:
        LOGGER.warning("Skipped %d assets due to render failures", failed_assets)
    return rendered_assets


def build_asset_content(asset: RenderedAsset) -> str:
    parts = [f"HSSD asset id: {asset.asset_id}."]
    if asset.metadata is not None:
        parts.extend(
            [
                f"Title: {asset.metadata.name}.",
                f"WordNet synset: {asset.metadata.wordnet_key}.",
            ]
        )
        if asset.metadata.up:
            parts.append(f"Original up vector: {asset.metadata.up}.")
        if asset.metadata.front:
            parts.append(f"Original front vector: {asset.metadata.front}.")
    if asset.object_groups:
        parts.append(f"HSSD groups: {', '.join(asset.object_groups)}.")
    parts.append(f"Rendered views: {', '.join(asset.image_paths.keys())}.")
    return " ".join(parts)


def embed_asset_views(
    asset: RenderedAsset,
    client: LlamaEmbeddingClient,
    expected_dimension: int,
    view_embedding_strategy: str,
) -> list[float]:
    if not asset.image_paths:
        raise ValueError(f"No rendered views found for asset {asset.asset_id}")

    content = build_asset_content(asset)
    if view_embedding_strategy == "multi_image":
        image_paths = list(asset.image_paths.values())
        view_markers = " ".join(
            f"{view} view: {client.media_marker}" for view in asset.image_paths
        )
        prompt = (
            "Represent this rendered HSSD asset for semantic visual retrieval. "
            "All images are different views of the same physical object. "
            f"{content} {view_markers}"
        )
        vector = np.asarray(
            client.embed_images_text(image_paths, prompt), dtype=np.float32
        )
        if vector.shape != (expected_dimension,):
            raise ValueError(
                f"Embedding dimension mismatch for asset {asset.asset_id}: "
                f"got {vector.shape}, expected ({expected_dimension},)"
            )
        return vector.astype(np.float32).tolist()

    if view_embedding_strategy != "average":
        raise ValueError(
            f"Unsupported view embedding strategy: {view_embedding_strategy}"
        )

    vectors: list[np.ndarray] = []
    for view, image_path in asset.image_paths.items():
        prompt = (
            "Represent this rendered HSSD asset for semantic visual retrieval. "
            f"{content} Current rendered view: {view}. image: {client.media_marker}"
        )
        vector = np.asarray(
            client.embed_image_text(image_path, prompt), dtype=np.float32
        )
        if vector.shape != (expected_dimension,):
            raise ValueError(
                f"Embedding dimension mismatch for {image_path}: "
                f"got {vector.shape}, expected ({expected_dimension},)"
            )
        vectors.append(vector)

    averaged = np.mean(np.stack(vectors, axis=0), axis=0)
    norm = float(np.linalg.norm(averaged))
    if norm > 0.0:
        averaged = averaged / norm
    return averaged.astype(np.float32).tolist()


def make_schema(
    collection_name: str,
    dimension: int,
    index_type: str,
    enable_fts: bool,
) -> zvec.CollectionSchema:
    if index_type == "flat":
        index_param = zvec.FlatIndexParam(metric_type=zvec.MetricType.COSINE)
    elif index_type == "hnsw":
        index_param = zvec.HnswIndexParam(
            metric_type=zvec.MetricType.COSINE,
            m=32,
            ef_construction=300,
        )
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    content_index = (
        zvec.FtsIndexParam(tokenizer_name="standard", filters=["lowercase"])
        if enable_fts
        else None
    )

    return zvec.CollectionSchema(
        name=collection_name,
        fields=[
            zvec.FieldSchema(
                name="asset_id",
                data_type=zvec.DataType.STRING,
                index_param=zvec.InvertIndexParam(enable_range_optimization=False),
            ),
            zvec.FieldSchema(
                name="name", data_type=zvec.DataType.STRING, nullable=True
            ),
            zvec.FieldSchema(
                name="wordnet_key",
                data_type=zvec.DataType.STRING,
                nullable=True,
                index_param=zvec.InvertIndexParam(enable_range_optimization=False),
            ),
            zvec.FieldSchema(
                name="object_groups",
                data_type=zvec.DataType.ARRAY_STRING,
                nullable=True,
                index_param=zvec.InvertIndexParam(enable_range_optimization=False),
            ),
            zvec.FieldSchema(
                name="views",
                data_type=zvec.DataType.ARRAY_STRING,
                nullable=True,
            ),
            zvec.FieldSchema(
                name="image_paths",
                data_type=zvec.DataType.ARRAY_STRING,
                nullable=True,
            ),
            zvec.FieldSchema(
                name="asset_path",
                data_type=zvec.DataType.STRING,
                nullable=True,
            ),
            zvec.FieldSchema(
                name="content",
                data_type=zvec.DataType.STRING,
                index_param=content_index,
            ),
        ],
        vectors=[
            zvec.VectorSchema(
                name="embedding",
                data_type=zvec.DataType.VECTOR_FP32,
                dimension=dimension,
                index_param=index_param,
            ),
        ],
    )


def open_or_create_collection(
    collection_path: Path,
    schema: zvec.CollectionSchema,
    recreate: bool,
) -> zvec.Collection:
    if recreate and collection_path.exists():
        shutil.rmtree(collection_path)

    if collection_path.exists():
        LOGGER.info("Opening existing Zvec collection: %s", collection_path)
        return zvec.open(path=str(collection_path))

    collection_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Creating Zvec collection: %s", collection_path)
    return zvec.create_and_open(path=str(collection_path), schema=schema)


def make_doc(asset: RenderedAsset, embedding: list[float]) -> zvec.Doc:
    metadata = asset.metadata
    fields = {
        "asset_id": asset.asset_id,
        "name": metadata.name if metadata is not None else "",
        "wordnet_key": metadata.wordnet_key if metadata is not None else "",
        "object_groups": asset.object_groups,
        "views": list(asset.image_paths.keys()),
        "image_paths": [str(path) for path in asset.image_paths.values()],
        "asset_path": str(asset.asset_path) if asset.asset_path is not None else "",
        "content": build_asset_content(asset),
    }
    return zvec.Doc(
        id=asset.asset_id,
        vectors={"embedding": embedding},
        fields=fields,
    )


def embed_asset_doc(
    asset: RenderedAsset,
    client: LlamaEmbeddingClient,
    expected_dimension: int,
    view_embedding_strategy: str,
) -> zvec.Doc:
    embedding = embed_asset_views(
        asset=asset,
        client=client,
        expected_dimension=expected_dimension,
        view_embedding_strategy=view_embedding_strategy,
    )
    return make_doc(asset, embedding)


def flush_docs(collection: zvec.Collection, docs: list[zvec.Doc]) -> int:
    if not docs:
        return 0
    result = collection.upsert(docs)
    statuses = result if isinstance(result, list) else [result]
    failures = [status for status in statuses if not status_ok(status)]
    if failures:
        raise RuntimeError(
            f"Zvec upsert failed for {len(failures)} docs: {failures[:3]}"
        )
    return len(docs)


def status_ok(status: Any) -> bool:
    ok = getattr(status, "ok", None)
    if callable(ok):
        return bool(ok())

    code = getattr(status, "code", None)
    if callable(code):
        return str(code()).endswith(".OK")
    return code in (0, "0", "OK", None)


def parse_include_views(value: str) -> set[str] | None:
    cleaned = value.strip().lower()
    if cleaned in {"", "all", "*"}:
        return None
    return {item.strip() for item in cleaned.split(",") if item.strip()}


def parse_view_list(value: str) -> list[str]:
    cleaned = value.strip().lower()
    if cleaned in {"", "all", "*"}:
        return list(DEFAULT_RETRIEVAL_VIEWS)
    return [item.strip() for item in cleaned.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read rendered HSSD asset images, compute Qwen3-VL embeddings, "
            "and index them into a local Zvec collection."
        )
    )
    parser.add_argument(
        "--render-root",
        type=Path,
        required=True,
        help="Directory containing rendered asset images.",
    )
    parser.add_argument(
        "--collection-path",
        type=Path,
        default=Path("data/hssd_zvec_collection"),
        help="Zvec collection directory to create/open.",
    )
    parser.add_argument(
        "--preprocessed-path",
        type=Path,
        default=Path("data/preprocessed"),
        help="HSSD preprocessed directory for names and WordNet metadata.",
    )
    parser.add_argument(
        "--hssd-root",
        type=Path,
        default=Path("data/hssd-models"),
        help="HSSD model root containing objects/<first-char>/<asset-id>.glb.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8014",
        help="llama.cpp Qwen3-VL-Embedding server base URL.",
    )
    parser.add_argument(
        "--media-marker",
        default="<__media__>",
        help="Media marker configured for llama.cpp multimodal requests.",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=2048,
        help="Expected embedding dimension from Qwen3-VL-Embedding.",
    )
    parser.add_argument(
        "--include-views",
        default=",".join(DEFAULT_RETRIEVAL_VIEWS),
        help=(
            "Comma-separated view names to include, or 'all'. "
            "Default: front,back,left,right,top,iso"
        ),
    )
    parser.add_argument(
        "--render-first",
        action="store_true",
        help=(
            "Render HSSD assets into --render-root before embedding. "
            "Uses clean named views suitable for multimodal retrieval."
        ),
    )
    parser.add_argument(
        "--render-views",
        default=",".join(DEFAULT_RETRIEVAL_VIEWS),
        help=(
            "Comma-separated named views to render when --render-first is set. "
            "Supported: top,front,back,left,right,bottom,iso. "
            "Default: front,back,left,right,top,iso."
        ),
    )
    parser.add_argument(
        "--render-width",
        type=int,
        default=224,
        help="Rendered image width in pixels when --render-first is set.",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=224,
        help="Rendered image height in pixels when --render-first is set.",
    )
    parser.add_argument(
        "--render-overwrite",
        action="store_true",
        help="Re-render requested views even if output PNGs already exist.",
    )
    parser.add_argument(
        "--render-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel Blender worker processes for --render-first. "
            "Use 2-4 only if GPU/CPU memory allows."
        ),
    )
    parser.add_argument(
        "--view-embedding-strategy",
        choices=("multi_image", "average"),
        default="multi_image",
        help=(
            "How to combine multiple rendered views for one asset. "
            "multi_image sends all selected views in one llama.cpp request; "
            "average embeds each view separately and averages normalized vectors."
        ),
    )
    parser.add_argument(
        "--index-type",
        choices=("flat", "hnsw"),
        default="flat",
        help="Zvec vector index type. Flat is exact and fine for ~14k HSSD assets.",
    )
    parser.add_argument(
        "--collection-name",
        default="hssd_rendered_assets",
        help="Zvec collection schema name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of Zvec docs to upsert per batch.",
    )
    parser.add_argument(
        "--embedding-workers",
        type=int,
        default=4,
        help=(
            "Number of concurrent embedding workers. Match this with llama.cpp "
            "PARALLEL for best throughput."
        ),
    )
    parser.add_argument(
        "--embedding-timeout-seconds",
        type=float,
        default=120.0,
        help="Per-request timeout for llama.cpp /embeddings calls.",
    )
    parser.add_argument(
        "--embedding-retries",
        type=int,
        default=2,
        help="Retries for transient embedding request failures.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of assets to index for smoke tests.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        default=True,
        help=(
            "Delete and rebuild the Zvec collection path before indexing. "
            "Default is enabled to avoid stale embeddings."
        ),
    )
    parser.add_argument(
        "--no-recreate",
        dest="recreate",
        action="store_false",
        help="Open the existing Zvec collection and upsert into it.",
    )
    parser.add_argument(
        "--no-fts",
        action="store_true",
        help="Disable full-text index on the content field.",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip collection.optimize() after upserting documents.",
    )
    parser.add_argument(
        "--check-server",
        action="store_true",
        help="Read /props before indexing and verify vision is enabled.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    preprocessed_path = (
        args.preprocessed_path if args.preprocessed_path.exists() else None
    )
    if preprocessed_path is None:
        LOGGER.warning(
            "HSSD preprocessed path not found; indexing with filename metadata"
        )

    hssd_root = args.hssd_root if args.hssd_root.exists() else None
    if hssd_root is None:
        LOGGER.warning("HSSD root not found; asset_path field will be empty")

    metadata_by_id, groups_by_wordnet = load_hssd_lookup(preprocessed_path)
    include_views = parse_include_views(args.include_views)
    render_views = parse_view_list(args.render_views)

    if args.render_first:
        try:
            rendered_assets = render_hssd_assets_if_needed(
                render_root=args.render_root,
                hssd_root=hssd_root,
                metadata_by_id=metadata_by_id,
                render_views=render_views,
                limit=args.limit,
                overwrite=args.render_overwrite,
                width=args.render_width,
                height=args.render_height,
                render_workers=args.render_workers,
            )
        except Exception as exc:
            LOGGER.error("Pre-render stage failed: %s", exc)
            return 2
        LOGGER.info(
            "Pre-render stage completed, rendered %d asset directories",
            rendered_assets,
        )

    if not args.render_root.exists():
        LOGGER.error("Render root does not exist: %s", args.render_root)
        return 2

    assets = discover_rendered_assets(
        render_root=args.render_root,
        metadata_by_id=metadata_by_id,
        groups_by_wordnet=groups_by_wordnet,
        hssd_root=hssd_root,
        include_views=include_views,
        require_metadata=True,
    )
    if args.limit is not None:
        assets = assets[: args.limit]

    if not assets:
        LOGGER.error("No rendered assets found under %s", args.render_root)
        if hssd_root is not None and not args.render_first:
            LOGGER.error(
                "Render root is empty or missing expected views. "
                "Re-run with --render-first to auto-render HSSD assets first."
            )
        return 3

    # 2026-07-09 修改原因：本脚本用于重建 HSSD zvec RAG 库，默认只索引有 metadata 的多视角资产并清空旧库。
    LOGGER.info(
        "Discovered %d rendered assets with metadata using views=%s",
        len(assets),
        sorted(include_views) if include_views is not None else "all",
    )

    embedding_retries = max(0, args.embedding_retries)
    client = LlamaEmbeddingClient(
        base_url=args.base_url,
        media_marker=args.media_marker,
        timeout_seconds=args.embedding_timeout_seconds,
        request_retries=embedding_retries,
    )
    if args.check_server:
        props = client.props()
        modalities = props.get("modalities") or {}
        LOGGER.info("Embedding server modalities: %s", modalities)
        if modalities.get("vision") is not True:
            LOGGER.error("Embedding server does not report vision=true")
            return 4

    schema = make_schema(
        collection_name=args.collection_name,
        dimension=args.embedding_dimension,
        index_type=args.index_type,
        enable_fts=not args.no_fts,
    )
    collection = open_or_create_collection(
        collection_path=args.collection_path,
        schema=schema,
        recreate=args.recreate,
    )

    pending_docs: list[zvec.Doc] = []
    indexed = 0
    embedding_workers = max(1, args.embedding_workers)
    LOGGER.info("Embedding workers: %d", embedding_workers)
    try:
        if embedding_workers == 1:
            for asset in tqdm(assets, desc="Indexing rendered HSSD assets"):
                pending_docs.append(
                    embed_asset_doc(
                        asset=asset,
                        client=client,
                        expected_dimension=args.embedding_dimension,
                        view_embedding_strategy=args.view_embedding_strategy,
                    )
                )
                if len(pending_docs) >= args.batch_size:
                    indexed += flush_docs(collection, pending_docs)
                    pending_docs.clear()
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=embedding_workers
            ) as executor:
                futures = {
                    executor.submit(
                        embed_asset_doc,
                        asset,
                        client,
                        args.embedding_dimension,
                        args.view_embedding_strategy,
                    ): asset
                    for asset in assets
                }
                iterator = tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Indexing rendered HSSD assets",
                )
                for future in iterator:
                    asset = futures[future]
                    try:
                        pending_docs.append(future.result())
                    except Exception as exc:
                        for pending_future in futures:
                            pending_future.cancel()
                        raise RuntimeError(
                            f"Failed to embed asset {asset.asset_id}"
                        ) from exc
                    if len(pending_docs) >= args.batch_size:
                        indexed += flush_docs(collection, pending_docs)
                        pending_docs.clear()

        indexed += flush_docs(collection, pending_docs)
        pending_docs.clear()

        LOGGER.info("Indexed %d assets into %s", indexed, args.collection_path)
        if not args.no_optimize:
            LOGGER.info("Calling collection.optimize()")
            collection.optimize()
        LOGGER.info("Collection stats: %s", collection.stats)
    finally:
        del collection
        gc.collect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
