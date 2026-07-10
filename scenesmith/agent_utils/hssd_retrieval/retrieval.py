"""Main HSSD retrieval logic with two-stage process: CLIP → size ranking."""

import logging

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import trimesh

from scenesmith.agent_utils.hssd_retrieval.alignment import (
    apply_hssd_alignment_transform,
)
from scenesmith.agent_utils.hssd_retrieval.clip_similarity import (
    get_top_k_similar_meshes,
)
from scenesmith.agent_utils.hssd_retrieval.config import HssdConfig
from scenesmith.agent_utils.hssd_retrieval.data_loader import (
    HssdMeshMetadata,
    construct_hssd_mesh_path,
    load_preprocessed_data,
)
from scenesmith.agent_utils.hssd_retrieval.zvec_similarity import HssdZvecSearcher
from scenesmith.agent_utils.manipuland_scale import (
    compute_uniform_scale_fit,
    match_size_profile,
)

console_logger = logging.getLogger(__name__)

_BUNK_BED_REQUEST_TERMS = (
    "bunk",
    "loft bed",
    "lofted bed",
    "two-tier",
    "two tier",
    "stacked bed",
)
_BUNK_BED_METADATA_TERMS = ("bunk_bed", "bunk bed", "loft bed", "lofted bed")
_BED_REQUEST_TERMS = ("bed", "mattress", "duvet", "headboard", "bedding")


class SemanticSearcher(Protocol):
    """Protocol for HSSD semantic candidate generation."""

    def get_top_k_similar_meshes(
        self,
        text_description: str,
        preprocessed_data: object,
        category: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Return semantic candidates ranked best-first."""


@dataclass
class RetrievalCandidate:
    """Candidate mesh for retrieval."""

    mesh_id: str
    """HSSD mesh ID."""

    mesh: trimesh.Trimesh
    """Loaded and transformed mesh."""

    metadata: HssdMeshMetadata
    """HSSD metadata."""

    clip_score: float
    """CLIP similarity score."""

    bbox_score: float
    """Bounding box size difference score (L1 distance)."""

    max_axis_relative_error: float | None = None
    """Uniform-scale fit error used for manipuland aspect-ratio filtering."""


class HssdRetriever:
    """HSSD asset retrieval system.

    Implements two-stage retrieval:
    1. CLIP semantic filtering (select top-K candidates)
    2. Size-based ranking (rank by dimension match)

    Based on HSM (https://arxiv.org/abs/2503.16848).
    """

    def __init__(self, config: HssdConfig, clip_device: str | None = None) -> None:
        """Initialize HSSD retriever.

        Args:
            config: HSSD configuration.
            clip_device: Target device for CLIP model (e.g., "cuda:0"). If None,
                uses default.
        """
        self.config = config
        self.clip_device = clip_device
        self.preprocessed_data = load_preprocessed_data(config.preprocessed_path)
        self.semantic_searcher = self._create_semantic_searcher()
        console_logger.info(f"HSSD retriever initialized (clip_device={clip_device})")

    def _create_semantic_searcher(self) -> SemanticSearcher:
        """Build the semantic search backend selected by config."""
        if self.config.retrieval_backend == "embedding":
            if self.config.zvec is None:
                raise ValueError("Missing Zvec config for embedding retrieval backend")
            return HssdZvecSearcher(self.config.zvec)
        return _ClipSemanticSearcher(device=self.clip_device)

    def _calculate_bbox_score(
        self, target_dimensions: np.ndarray, mesh_extents: np.ndarray
    ) -> float:
        """Calculate bounding box score (L1 distance).

        Args:
            target_dimensions: Desired dimensions (3,).
            mesh_extents: Actual mesh extents (3,).

        Returns:
            L1 distance score (lower is better).
        """
        return float(np.sum(np.abs(target_dimensions - mesh_extents)))

    def _load_and_process_mesh(
        self, mesh_id: str, metadata: HssdMeshMetadata
    ) -> trimesh.Trimesh:
        """Load mesh and apply HSSD alignment transform (if orientation data available).

        Pipeline:
        1. Load GLB from HSSD directory
        2. Apply HSSD alignment transform if metadata contains orientation data
           (original → HSM canonical Y-up Z-forward). If orientation data is
           missing (common for ~99% of meshes), skip alignment and rely on
           downstream canonicalization.

        Note: Mesh remains in Y-up coordinates (HSM canonical) for GLTF export
        compatibility. The canonicalization step will handle coordinate processing
        and normalization (centering, ground placement) in the appropriate
        coordinate system for both aligned and non-aligned meshes.

        Args:
            mesh_id: HSSD mesh ID.
            metadata: HSSD metadata (may have empty orientation fields).

        Returns:
            Mesh in HSM canonical coordinates if alignment data available,
            otherwise in original HSSD coordinates (both Y-up).
        """
        mesh_path = construct_hssd_mesh_path(self.config.data_path, mesh_id)

        mesh = trimesh.load(mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Loaded mesh is not a Trimesh: {type(mesh)}")

        mesh = apply_hssd_alignment_transform(mesh, metadata)

        return mesh

    def retrieve(
        self,
        description: str,
        object_type: str,
        desired_dimensions: np.ndarray | None = None,
        max_axis_relative_error: float | None = None,
    ) -> tuple[trimesh.Trimesh, str, float, HssdMeshMetadata]:
        """Retrieve best matching HSSD mesh for description.

        Two-stage process:
        1. CLIP semantic filtering → top-K candidates
        2. Size-based ranking → best dimension match

        Note: Meshes are loaded with optional HSSD alignment (applied only if
        orientation metadata is available, which is ~1% of meshes). Meshes
        without alignment will be handled by downstream canonicalization.

        Args:
            description: Object description text.
            object_type: Object type (e.g., "FURNITURE", "MANIPULAND").
            desired_dimensions: Optional desired dimensions (width, height, depth).

        Returns:
            Tuple of (mesh, mesh_id, clip_score, metadata) where:
            - mesh: Best matching mesh in Y-up coordinates
            - mesh_id: SHA-1 hash identifying the mesh
            - clip_score: CLIP similarity score (0.0 to 1.0)
            - metadata: HSSD mesh metadata including orientation and WordNet info

        Raises:
            ValueError: If no suitable mesh is found.
        """
        candidates = self.retrieve_multiple(
            description=description,
            object_type=object_type,
            desired_dimensions=desired_dimensions,
            max_axis_relative_error=max_axis_relative_error,
            max_candidates=1,
        )

        if not candidates:
            raise ValueError(
                f"No suitable mesh found for '{description}' (type={object_type})"
            )

        best = candidates[0]
        return best.mesh, best.mesh_id, best.clip_score, best.metadata

    def retrieve_multiple(
        self,
        description: str,
        object_type: str,
        desired_dimensions: np.ndarray | None = None,
        max_axis_relative_error: float | None = None,
        max_candidates: int | None = None,
    ) -> list[RetrievalCandidate]:
        """Retrieve multiple matching HSSD meshes for description.

        Same two-stage process as retrieve(), but returns all candidates
        sorted by bbox_score instead of just the best one.

        Args:
            description: Object description text.
            object_type: Object type (e.g., "FURNITURE", "MANIPULAND").
            desired_dimensions: Optional desired dimensions (width, height, depth).
            max_axis_relative_error: Optional rejection threshold for the actual
                dimensions produced by uniform scaling.
            max_candidates: Maximum candidates to return. If None, returns all
                available (up to use_top_k CLIP candidates).

        Returns:
            List of RetrievalCandidate sorted by bbox_score (best first).
            Empty list if no suitable meshes found.
        """
        console_logger.info(
            f"Retrieving multiple HSSD meshes: description='{description}', "
            f"type={object_type}, dimensions={desired_dimensions}"
        )

        category = self.config.object_type_mapping.get(object_type.upper())
        if category is None:
            console_logger.warning(
                f"Unknown object type: {object_type}. "
                f"Available: {list(self.config.object_type_mapping.keys())}"
            )
            return []

        semantic_searcher = getattr(self, "semantic_searcher", None)
        if semantic_searcher is None:
            semantic_searcher = self._create_semantic_searcher()
            self.semantic_searcher = semantic_searcher

        top_k_meshes = semantic_searcher.get_top_k_similar_meshes(
            text_description=description,
            preprocessed_data=self.preprocessed_data,
            category=category,
            top_k=self.config.use_top_k,
        )

        if not top_k_meshes:
            console_logger.warning(f"No meshes found for category: {category}")
            return []

        console_logger.info(f"Processing {len(top_k_meshes)} CLIP-filtered candidates")

        candidates: list[RetrievalCandidate] = []
        size_profile = match_size_profile(description)

        for mesh_id, clip_score in top_k_meshes:
            metadata = self.preprocessed_data.get_metadata(mesh_id)
            if metadata is None:
                console_logger.warning(f"Metadata not found for mesh {mesh_id}")
                continue
            if _is_unrequested_bunk_bed(description, metadata):
                # 2026-07-10 修改原因：普通 bed 请求中 bunk bed 候选会被 VLM
                # 偶发接受并阻塞 bedroom furniture 回放；只有明确请求 bunk/loft
                # bed 时才允许这类 HSSD 候选进入后续校验。
                console_logger.info(
                    "Rejecting HSSD candidate %s for '%s': metadata indicates bunk bed",
                    mesh_id[:8],
                    description,
                )
                continue
            if _is_low_information_generic_bed(description, metadata):
                # 2026-07-10 修改原因：6-view embedding 索引下普通床检索仍会把
                # name 为空的 bed.n.01 泛化资产排到首位；这类低信息候选容易通过
                # lenient VLM 校验但在 bedroom furniture 回放中卡住转换/碰撞。
                console_logger.info(
                    "Rejecting HSSD candidate %s for '%s': low-information generic bed",
                    mesh_id[:8],
                    description,
                )
                continue

            try:
                mesh = self._load_and_process_mesh(mesh_id, metadata)
            except Exception as e:
                console_logger.warning(
                    f"Failed to load mesh {mesh_id}: {e}", exc_info=True
                )
                continue

            if desired_dimensions is not None:
                mesh_extents = mesh.extents
                bbox_score = self._calculate_bbox_score(
                    target_dimensions=desired_dimensions, mesh_extents=mesh_extents
                )
                uniform_fit_error = None
                if max_axis_relative_error is not None:
                    # 7.3 fix: reject small-object candidates whose original aspect
                    # ratio would remain visibly wrong after the required uniform scale.
                    fit = compute_uniform_scale_fit(
                        current_dimensions=mesh_extents,
                        desired_dimensions=desired_dimensions,
                        footprint_swappable=(
                            size_profile.footprint_swappable
                            if size_profile is not None
                            else False
                        ),
                    )
                    uniform_fit_error = fit.max_axis_relative_error
                    if uniform_fit_error > max_axis_relative_error:
                        console_logger.info(
                            f"Rejecting HSSD candidate {mesh_id[:8]} for "
                            f"'{description}': uniform fit error "
                            f"{uniform_fit_error:.3f} exceeds "
                            f"{max_axis_relative_error:.3f}"
                        )
                        continue
            else:
                bbox_score = 0.0
                uniform_fit_error = None

            candidate = RetrievalCandidate(
                mesh_id=mesh_id,
                mesh=mesh,
                metadata=metadata,
                clip_score=clip_score,
                bbox_score=bbox_score,
                max_axis_relative_error=uniform_fit_error,
            )
            candidates.append(candidate)

            console_logger.debug(
                f"Candidate {mesh_id[:8]}: CLIP={clip_score:.3f}, "
                f"bbox={bbox_score:.3f}, extents={mesh.extents}"
            )

        if not candidates:
            console_logger.warning("No valid candidates found after mesh loading")
            return []

        # Sort by deterministic size fit first when present, then bbox_score.
        candidates.sort(
            key=lambda c: (
                (
                    c.max_axis_relative_error
                    if c.max_axis_relative_error is not None
                    else c.bbox_score
                ),
                c.bbox_score,
            )
        )

        # Limit results if requested.
        if max_candidates is not None and len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        console_logger.info(
            f"Returning {len(candidates)} candidates (sorted by bbox_score)"
        )

        return candidates


def _is_unrequested_bunk_bed(description: str, metadata: HssdMeshMetadata) -> bool:
    request_text = str(description or "").lower()
    if any(term in request_text for term in _BUNK_BED_REQUEST_TERMS):
        return False
    metadata_text = f"{metadata.name} {metadata.wordnet_key}".lower().replace(".", "_")
    return any(term in metadata_text for term in _BUNK_BED_METADATA_TERMS)


def _is_low_information_generic_bed(
    description: str, metadata: HssdMeshMetadata
) -> bool:
    request_text = str(description or "").lower()
    if not any(term in request_text for term in _BED_REQUEST_TERMS):
        return False
    metadata_name = str(metadata.name or "").strip()
    return not metadata_name and str(metadata.wordnet_key or "") == "bed.n.01"


class _ClipSemanticSearcher:
    """Adapter around the existing CLIP similarity search."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device

    def get_top_k_similar_meshes(
        self,
        text_description: str,
        preprocessed_data: object,
        category: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        return get_top_k_similar_meshes(
            text_description=text_description,
            preprocessed_data=preprocessed_data,
            category=category,
            top_k=top_k,
            device=self.device,
        )
