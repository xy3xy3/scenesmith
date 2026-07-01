"""Asset-level VLM annotation support for the embedded SceneBenchmark critic."""

from __future__ import annotations

import hashlib
import json
import logging

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from pydantic import BaseModel, ConfigDict, Field
from pydrake.math import RollPitchYaw

from scenesmith.agent_utils.room import ObjectType, RoomScene, SceneObject
from scenesmith.agent_utils.vlm_service import VLMService
from scenesmith.scenebenchmark_critic import adapter, clearance_source
from scenesmith.utils.llm_json import parse_llm_json_object
from scenesmith.utils.openai import encode_image_to_base64

if TYPE_CHECKING:
    from scenesmith.agent_utils.blender.server_manager import BlenderServer

console_logger = logging.getLogger(__name__)

ANNOTATION_SCHEMA_VERSION = "asset_vlm_annotation@0.3"
TASK_SCHEMA_VERSION = "asset_vlm_annotation_task@0.3"
PROMPT_VERSION = "asset_vlm_annotation_prompt@0.5"

SceneObjectType = Literal[
    "wall_mounted", "manipuland", "ceiling_mounted", "furniture", "unknown"
]
MobilityClass = Literal["fixed", "movable", "semi_movable", "mounted", "unknown"]
AccessibilityPolicy = Literal["required", "optional", "ignored"]

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "backend": "mock",
    "write_back": True,
    "write_files": True,
    "write_scene_state": True,
    "skip_existing": True,
    "refresh": False,
    "output_dir_name": "asset_annotations",
    "request_dir_name": "asset_annotation_requests",
    "render_dir_name": "asset_annotation_renders",
    "object_types": ["furniture", "wall_mounted", "ceiling_mounted", "manipuland"],
    "model": None,
    "reasoning_effort": "low",
    "verbosity": "low",
    "vision_detail": None,
    "render": {
        "enabled": True,
        "num_side_views": 4,
        "include_vertical_views": True,
        "elevation_degrees": 20.0,
        "width": 512,
        "height": 512,
        "timeout": 120.0,
        "show_coordinate_frame": False,
    },
    "merge": {
        "category_override_threshold": 0.75,
        "affordance_add_threshold": 0.70,
        "front_face_threshold": 0.70,
        "scene_object_type_threshold": 0.70,
    },
}

FUNCTIONAL_AFFORDANCES = {
    "sittable",
    "sleepable",
    "supportable",
    "openable",
    "containable",
    "toggleable",
    "graspable",
}
DECORATIVE_CATEGORIES = {
    "art",
    "artwork",
    "canvas",
    "clock",
    "mirror",
    "painting",
    "picture",
    "poster",
    "print",
    "rug",
    "sconce",
    "vase",
    "wall_art",
    "wall_clock",
    "wall_sconce",
}
SMALL_LOOSE_CATEGORIES = {
    "book",
    "bottle",
    "bowl",
    "cup",
    "laptop",
    "mug",
    "plate",
    "remote",
    "tray",
}
SCENE_OBJECT_TYPES = {
    "wall_mounted",
    "manipuland",
    "ceiling_mounted",
    "furniture",
    "unknown",
}
MOBILITY_CLASSES = {"fixed", "movable", "semi_movable", "mounted", "unknown"}
ACCESSIBILITY_POLICIES = {"required", "optional", "ignored"}
ACCESS_FACE_NAMES = {"front", "back", "left", "right", "top", "bottom"}

MOVABLE_SEATING_CATEGORIES = {
    "chair",
    "office_chair",
    "dining_chair",
    "stool",
    "bar_stool",
    "bean_bag",
    "beanbag_chair",
}
SEMI_MOVABLE_FURNITURE_CATEGORIES = {
    "armchair",
    "bench",
    "loveseat",
    "sofa",
    "coffee_table",
    "dining_table",
    "table",
    "desk",
    "nightstand",
    "side_table",
    "tv_stand",
}
WALL_BACKED_CATEGORIES = {
    "bed",
    "bookshelf",
    "cabinet",
    "dresser",
    "refrigerator",
    "shelf",
    "wardrobe",
    "wine_cabinet",
}
WALL_MOUNTED_CATEGORIES = {"wall_cabinet", "wall_shelf", "wall_sconce"}
CEILING_MOUNTED_CATEGORIES = {"ceiling_light", "pendant_light"}


class AffordancePrediction(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str
    anchor: str | None = None
    required_face: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class FrontFacePrediction(BaseModel):
    model_config = ConfigDict(extra="allow")

    view: str | None = None
    reason: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AmbiguityPrediction(BaseModel):
    model_config = ConfigDict(extra="allow")

    is_ambiguous: bool = False
    notes: list[str] = Field(default_factory=list)


class AssetVlmPrediction(BaseModel):
    model_config = ConfigDict(extra="allow")

    canonical_name: str | None = None
    category_norm: str | None = None
    placement_class: str | None = None
    scene_object_type: SceneObjectType = "unknown"
    semantic_size_class: str | None = None
    benchmark_relevance: Literal["functional", "decorative", "noise", "unknown"] = (
        "unknown"
    )
    affordances: list[AffordancePrediction] = Field(default_factory=list)
    front_face: FrontFacePrediction | None = None
    interaction_surface_map: dict[str, list[str]] = Field(default_factory=dict)
    access_type: dict[str, Any] = Field(default_factory=dict)
    mobility_class: MobilityClass = "unknown"
    accessibility_policy: AccessibilityPolicy = "required"
    access_sides: list[str] = Field(default_factory=list)
    attachment_dependencies: list[dict[str, Any]] = Field(default_factory=list)
    orientation_dependencies: list[dict[str, Any]] = Field(default_factory=list)
    interaction_height_m: dict[str, float | None] = Field(default_factory=dict)
    related_categories: list[str] = Field(default_factory=list)
    style_tags: list[str] = Field(default_factory=list)
    ambiguity: AmbiguityPrediction = Field(default_factory=AmbiguityPrediction)
    rationale: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EffectiveAnnotation(BaseModel):
    model_config = ConfigDict(extra="allow")

    category_norm: str | None = None
    placement_class: str | None = None
    scene_object_type: SceneObjectType = "unknown"
    benchmark_relevance: Literal["functional", "decorative", "noise", "unknown"] = (
        "unknown"
    )
    affordances: list[str] = Field(default_factory=list)
    front_face: str | None = None
    interaction_surface_map: dict[str, list[str]] = Field(default_factory=dict)
    access_type: dict[str, Any] = Field(default_factory=dict)
    mobility_class: MobilityClass = "unknown"
    accessibility_policy: AccessibilityPolicy = "required"
    access_sides: list[str] = Field(default_factory=list)
    attachment_dependencies: list[dict[str, Any]] = Field(default_factory=list)
    orientation_dependencies: list[dict[str, Any]] = Field(default_factory=list)
    interaction_height_m: dict[str, float | None] = Field(default_factory=dict)
    related_categories: list[str] = Field(default_factory=list)
    source: str = "heuristic"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    conflict_notes: list[str] = Field(default_factory=list)
    low_confidence_candidates: dict[str, Any] = Field(default_factory=dict)


class ObjectFunctionProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    can_support_top: bool = False
    has_internal_shelf: bool = False
    is_small_placeable: bool = False
    is_seating: bool = False
    is_work_surface: bool = False
    is_media_target: bool = False
    is_bedside_surface: bool = False
    is_sleeping_surface: bool = False


class AssetAnnotation(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = ANNOTATION_SCHEMA_VERSION
    annotation_status: Literal["succeeded", "failed"] = "succeeded"
    object_id: str
    scene_id: str | None = None
    source: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    heuristic_prior: dict[str, Any] = Field(default_factory=dict)
    vlm_prediction: AssetVlmPrediction | None = None
    effective_annotation: EffectiveAnnotation
    object_function_profile: ObjectFunctionProfile = Field(
        default_factory=ObjectFunctionProfile
    )
    clearance: dict[str, Any] | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None


def annotate_room_scene(
    scene: RoomScene,
    *,
    output_dir: Path,
    config: Any,
    raw_config: Any | None = None,
    vlm_service: VLMService | None = None,
    blender_server: "BlenderServer | None" = None,
    stage: str = "adhoc",
) -> dict[str, Any] | None:
    """Annotate scene object assets and write effective hints into metadata."""
    annotation_cfg = _annotation_config(config)
    if not _as_bool(annotation_cfg.get("enabled")):
        return None

    output_dir = Path(output_dir)
    annotation_dir = output_dir / str(annotation_cfg["output_dir_name"])
    request_dir = output_dir / str(annotation_cfg["request_dir_name"])
    render_root = output_dir / str(annotation_cfg["render_dir_name"])
    backend = str(annotation_cfg.get("backend") or "mock").strip().lower()
    selected = _select_objects(scene, annotation_cfg)

    written: list[str] = []
    skipped: list[str] = []
    annotations: list[AssetAnnotation] = []

    for obj in selected:
        annotation_path = annotation_dir / f"{obj.object_id}.yaml"
        if (
            _as_bool(annotation_cfg.get("skip_existing"))
            and not _as_bool(annotation_cfg.get("refresh"))
            and annotation_path.is_file()
        ):
            existing = _load_annotation(annotation_path)
            if existing is not None and existing.annotation_status == "succeeded":
                if _as_bool(annotation_cfg.get("write_back")):
                    write_back_effective_hints(obj, existing)
                skipped.append(str(obj.object_id))
                annotations.append(existing)
                continue

        annotation = annotate_scene_object(
            scene,
            obj,
            config=annotation_cfg,
            raw_config=raw_config,
            vlm_service=vlm_service,
            blender_server=blender_server,
            output_dir=output_dir,
            request_dir=request_dir,
            render_dir=render_root / str(obj.object_id),
            stage=stage,
        )
        annotations.append(annotation)
        if _as_bool(annotation_cfg.get("write_files")):
            annotation_dir.mkdir(parents=True, exist_ok=True)
            _write_yaml(annotation_path, annotation.model_dump(mode="json"))
            written.append(str(annotation_path))
        if _as_bool(annotation_cfg.get("write_back")):
            write_back_effective_hints(obj, annotation)

    if _as_bool(annotation_cfg.get("write_scene_state")):
        state_path = output_dir / "scene_state.json"
        if state_path.exists() and _as_bool(annotation_cfg.get("write_back")):
            state_path.write_text(
                json.dumps(scene.to_state_dict(), indent=2), encoding="utf-8"
            )

    summary = {
        "schema_version": TASK_SCHEMA_VERSION,
        "stage": stage,
        "backend": backend,
        "object_count": len(selected),
        "processed_count": len(written),
        "skipped_count": len(skipped),
        "written": written,
        "skipped": skipped,
    }
    if _as_bool(annotation_cfg.get("write_files")):
        annotation_dir.mkdir(parents=True, exist_ok=True)
        (annotation_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    console_logger.info(
        "SceneBenchmark asset annotation finished for %s object(s) using %s",
        len(selected),
        backend,
    )
    return summary


def annotate_scene_object(
    scene: RoomScene,
    obj: SceneObject,
    *,
    config: dict[str, Any],
    raw_config: Any | None = None,
    vlm_service: VLMService | None = None,
    blender_server: "BlenderServer | None" = None,
    output_dir: Path,
    request_dir: Path,
    render_dir: Path,
    stage: str,
) -> AssetAnnotation:
    heuristic_prior = build_heuristic_prior(scene, obj)
    render_paths = _render_or_collect_images(
        obj=obj,
        config=config,
        blender_server=blender_server,
        render_dir=render_dir,
    )
    request_payload = build_request_payload(
        scene=scene,
        obj=obj,
        heuristic_prior=heuristic_prior,
        image_paths=render_paths,
        config=config,
        stage=stage,
    )
    if _as_bool(config.get("write_files")):
        request_dir.mkdir(parents=True, exist_ok=True)
        request_path = request_dir / f"{obj.object_id}.request.json"
        request_path.write_text(json.dumps(request_payload, indent=2), encoding="utf-8")

    prediction = run_prediction(
        obj=obj,
        heuristic_prior=heuristic_prior,
        request_payload=request_payload,
        image_paths=render_paths,
        config=config,
        raw_config=raw_config,
        vlm_service=vlm_service,
    )
    effective = merge_asset_annotation(
        heuristic_prior, prediction, config.get("merge") or {}
    )
    object_function_profile = _object_function_profile_from_effective_annotation(
        obj, effective
    )
    clearance = clearance_source.get_clearance_for_metadata(obj.metadata)
    if clearance is not None:
        # Mirror into metadata so the adapter carries it into the case_pack
        # geometry without re-querying the provider.
        obj.metadata["clearance"] = clearance
    evidence = {
        "mesh_path": str(obj.geometry_path) if obj.geometry_path else None,
        "image_paths": [str(path) for path in render_paths],
        "evidence_hash": _evidence_hash(obj=obj, image_paths=render_paths),
    }
    return AssetAnnotation(
        object_id=str(obj.object_id),
        scene_id=scene.room_id,
        source={
            "room_id": scene.room_id,
            "object_type": obj.object_type.value,
            "asset_id": obj.metadata.get("asset_id"),
            "asset_source": obj.metadata.get("asset_source"),
        },
        evidence=evidence,
        heuristic_prior=heuristic_prior,
        vlm_prediction=prediction,
        effective_annotation=effective,
        object_function_profile=object_function_profile,
        clearance=clearance,
        provenance={
            "annotator": "scenesmith_scenebenchmark_asset_annotator",
            "backend": str(config.get("backend") or "mock"),
            "model": _model_name(config, raw_config),
            "prompt_version": PROMPT_VERSION,
            "schema_version": TASK_SCHEMA_VERSION,
            "created_at": datetime.now()
            .astimezone()
            .replace(microsecond=0)
            .isoformat(),
            "stage": stage,
            "output_dir": str(output_dir),
        },
    )


def build_heuristic_prior(scene: RoomScene, obj: SceneObject) -> dict[str, Any]:
    category = adapter._category_for_object(obj)
    yaw = RollPitchYaw(obj.transform.rotation()).yaw_angle()
    hints = adapter._functional_hints(
        obj, category, yaw_deg=float(yaw * 180.0 / 3.141592653589793)
    )
    return {
        "category": obj.name,
        "category_norm": category,
        "description": obj.description,
        "placement": _placement_class(obj),
        "scene_object_type": _scene_object_type(obj),
        "bbox_world": _bbox_world(obj),
        "room_id": scene.room_id,
        "room_type": scene.room_type,
        "functional_hints": hints,
        "source_metadata": dict(obj.metadata),
    }


def build_request_payload(
    *,
    scene: RoomScene,
    obj: SceneObject,
    heuristic_prior: dict[str, Any],
    image_paths: list[Path],
    config: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    prompt_text = _build_annotation_prompt(
        scene=scene,
        obj=obj,
        heuristic_prior=heuristic_prior,
        image_paths=image_paths,
        stage=stage,
    )
    return {
        "schema_version": TASK_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "scene_id": scene.room_id,
        "object_id": str(obj.object_id),
        "backend": config.get("backend") or "mock",
        "inputs": {
            "raw_label": obj.name,
            "description": obj.description,
            "object_type": obj.object_type.value,
            "dimensions": (heuristic_prior.get("bbox_world") or {}).get("size"),
            "heuristic_prior": heuristic_prior,
            "image_paths": [str(path) for path in image_paths],
        },
        "prompt_text": prompt_text,
        "expected_output": "AssetVlmPrediction JSON",
    }


def run_prediction(
    *,
    obj: SceneObject,
    heuristic_prior: dict[str, Any],
    request_payload: dict[str, Any],
    image_paths: list[Path],
    config: dict[str, Any],
    raw_config: Any | None = None,
    vlm_service: VLMService | None = None,
) -> AssetVlmPrediction:
    backend = str(config.get("backend") or "mock").strip().lower()
    if backend in {"mock", "dry_run", "template"}:
        return mock_vlm_prediction(obj=obj, heuristic_prior=heuristic_prior)
    if backend not in {"vlm", "openai"}:
        raise ValueError(f"Unsupported asset_annotation.backend={backend!r}")

    service = vlm_service or VLMService(cfg=raw_config)
    messages = _build_vlm_messages(
        prompt_text=str(request_payload.get("prompt_text") or ""),
        image_paths=image_paths,
    )
    response_text = service.create_completion(
        model=_model_name(config, raw_config),
        messages=messages,
        reasoning_effort=str(config.get("reasoning_effort") or "low"),
        verbosity=str(config.get("verbosity") or "low"),
        response_format={"type": "json_object"},
        vision_detail=str(
            config.get("vision_detail")
            or _cfg_get(raw_config, "openai.vision_detail", "auto")
        ),
    )
    # 2026-06-18 hardening: asset-annotation VLM output now uses the shared JSON
    # repair path so fenced or lightly malformed local-model responses survive.
    payload = parse_llm_json_object(response_text)
    return AssetVlmPrediction.model_validate(payload)


def mock_vlm_prediction(
    *, obj: SceneObject, heuristic_prior: dict[str, Any]
) -> AssetVlmPrediction:
    category = str(heuristic_prior.get("category_norm") or "unknown")
    hints = heuristic_prior.get("functional_hints") or {}
    affordance_labels = _unique(
        hints.get("functional_categories") or hints.get("candidate_affordances") or []
    )
    group = str(hints.get("category_group") or "")
    relevance = _benchmark_relevance(category, group, affordance_labels)
    if relevance in {"decorative", "noise"}:
        affordance_labels = []
    confidence = _prediction_confidence(category, relevance, affordance_labels)
    front_face = _front_face(category, affordance_labels, hints)
    scene_object_type = _scene_object_type(obj)
    mobility_class = _mobility_class(
        category, scene_object_type, affordance_labels, hints
    )
    accessibility_policy = _accessibility_policy(
        category, mobility_class, affordance_labels, scene_object_type, hints
    )
    access_sides = _access_sides(category, affordance_labels, front_face, hints)
    return AssetVlmPrediction(
        canonical_name=category.replace("_", " "),
        category_norm=category,
        placement_class=_placement_class(obj),
        scene_object_type=scene_object_type,
        semantic_size_class=_semantic_size_class(heuristic_prior),
        benchmark_relevance=relevance,
        affordances=[
            AffordancePrediction(
                label=label,
                anchor=_anchor_for_affordance(label),
                required_face=_required_face_for_affordance(label, front_face),
                confidence=confidence,
            )
            for label in affordance_labels
        ],
        front_face=FrontFacePrediction(
            view=front_face,
            reason="deterministic mock prediction from SceneSmith heuristic prior",
            confidence=confidence if front_face else 0.0,
        ),
        interaction_surface_map=_interaction_surface_map(affordance_labels, front_face),
        access_type={"primary": _access_type(affordance_labels), "secondary": None},
        mobility_class=mobility_class,
        accessibility_policy=accessibility_policy,
        access_sides=access_sides,
        attachment_dependencies=_attachment_dependencies(
            category, scene_object_type, hints
        ),
        orientation_dependencies=_orientation_dependencies(
            category, affordance_labels, hints
        ),
        interaction_height_m=_interaction_height(heuristic_prior),
        related_categories=list(hints.get("target_relation") or []),
        ambiguity={
            "is_ambiguous": confidence < 0.65,
            "notes": [] if confidence >= 0.65 else ["low semantic prior"],
        },
        rationale=(
            "Mock VLM shim: preserves SceneBenchmark asset annotation semantics "
            "without calling a model."
        ),
        confidence=confidence,
    )


def merge_asset_annotation(
    heuristic_prior: dict[str, Any],
    vlm_prediction: AssetVlmPrediction,
    merge_config: dict[str, Any] | None = None,
) -> EffectiveAnnotation:
    cfg = {**(DEFAULT_CONFIG["merge"]), **(merge_config or {})}
    notes: list[str] = []
    low_confidence: dict[str, Any] = {}
    source_parts = ["heuristic"]

    heuristic_category = _clean(
        heuristic_prior.get("category_norm") or heuristic_prior.get("category")
    )
    category_norm = heuristic_category
    if vlm_prediction.category_norm and vlm_prediction.confidence >= float(
        cfg["category_override_threshold"]
    ):
        vlm_category = _clean(vlm_prediction.category_norm)
        if heuristic_category and heuristic_category != vlm_category:
            notes.append(f"category heuristic={heuristic_category} vlm={vlm_category}")
        category_norm = vlm_category
        source_parts.append("vlm_category")
    elif vlm_prediction.category_norm:
        low_confidence["category_norm"] = {
            "value": _clean(vlm_prediction.category_norm),
            "confidence": vlm_prediction.confidence,
        }

    hints = heuristic_prior.get("functional_hints") or {}
    affordances = _unique(
        hints.get("functional_categories") or hints.get("candidate_affordances") or []
    )
    low_conf_affordances: list[dict[str, Any]] = []
    for prediction in vlm_prediction.affordances:
        label = _clean(prediction.label)
        if not label:
            continue
        if prediction.confidence >= float(cfg["affordance_add_threshold"]):
            if label not in affordances:
                affordances.append(label)
                source_parts.append("vlm_affordance")
        else:
            low_conf_affordances.append(
                {
                    "label": label,
                    "confidence": prediction.confidence,
                    "anchor": prediction.anchor,
                }
            )
    if low_conf_affordances:
        low_confidence["affordances"] = low_conf_affordances

    benchmark_relevance = vlm_prediction.benchmark_relevance or "unknown"
    if benchmark_relevance in {"decorative", "noise"}:
        affordances = []
        source_parts.append("vlm_gate")

    front_face = None
    if vlm_prediction.front_face and vlm_prediction.front_face.view:
        if vlm_prediction.front_face.confidence >= float(cfg["front_face_threshold"]):
            front_face = _clean(vlm_prediction.front_face.view)
            source_parts.append("vlm_front_face")
        else:
            low_confidence["front_face"] = {
                "value": _clean(vlm_prediction.front_face.view),
                "confidence": vlm_prediction.front_face.confidence,
            }
    if front_face is None:
        front_face = _clean(hints.get("front_hint"))

    scene_object_type = _resolve_scene_object_type(
        heuristic_prior,
        vlm_prediction,
        threshold=float(cfg["scene_object_type_threshold"]),
        low_confidence=low_confidence,
    )
    if scene_object_type != "unknown":
        source_parts.append("scene_object_type")

    # External HSSD sidecars can pre-populate these keys in metadata. Treat those
    # as stronger than the VLM/mock fallback so the critic can be upgraded before
    # the final annotation pipeline lands.
    mobility_class = _resolve_mobility_class(hints, vlm_prediction)
    accessibility_policy = _resolve_accessibility_policy(
        hints, vlm_prediction, scene_object_type, affordances
    )
    access_sides = _resolve_access_sides(hints, vlm_prediction)
    attachment_dependencies = _resolve_dependency_list(
        hints, vlm_prediction, "attachment_dependencies"
    )
    orientation_dependencies = _resolve_dependency_list(
        hints, vlm_prediction, "orientation_dependencies"
    )
    if mobility_class != "unknown":
        source_parts.append("mobility_policy")
    if attachment_dependencies or orientation_dependencies:
        source_parts.append("dependency_policy")

    return EffectiveAnnotation(
        category_norm=category_norm,
        placement_class=vlm_prediction.placement_class
        or heuristic_prior.get("placement"),
        scene_object_type=scene_object_type,
        benchmark_relevance=benchmark_relevance,
        affordances=affordances,
        front_face=front_face,
        interaction_surface_map=vlm_prediction.interaction_surface_map,
        access_type=vlm_prediction.access_type,
        mobility_class=mobility_class,
        accessibility_policy=accessibility_policy,
        access_sides=access_sides,
        attachment_dependencies=attachment_dependencies,
        orientation_dependencies=orientation_dependencies,
        interaction_height_m=vlm_prediction.interaction_height_m,
        related_categories=vlm_prediction.related_categories,
        source="+".join(_unique(source_parts)),
        confidence=vlm_prediction.confidence,
        conflict_notes=notes,
        low_confidence_candidates=low_confidence,
    )


def write_back_effective_hints(obj: SceneObject, annotation: AssetAnnotation) -> None:
    effective = annotation.effective_annotation
    obj.metadata["category_norm"] = effective.category_norm or obj.metadata.get(
        "category_norm"
    )
    hints = obj.metadata.setdefault("functional_hints", {})
    if not isinstance(hints, dict):
        hints = {}
        obj.metadata["functional_hints"] = hints
    hints["candidate_affordances"] = list(effective.affordances)
    hints["functional_categories"] = list(effective.affordances)
    hints["affordances"] = list(effective.affordances)
    hints["front_hint"] = effective.front_face
    hints["front_face"] = effective.front_face
    hints["classification_source"] = "asset_annotation"
    hints["classification_confidence"] = effective.confidence
    hints["classification_reason"] = (
        "; ".join(effective.conflict_notes) or effective.source
    )
    hints["asset_annotation_source"] = "scenesmith_vlm_asset_annotator"
    hints["asset_annotation_schema_version"] = ANNOTATION_SCHEMA_VERSION
    hints["scene_object_type"] = effective.scene_object_type
    hints["placement_class"] = effective.placement_class
    hints["benchmark_relevance"] = effective.benchmark_relevance
    hints["mobility_class"] = effective.mobility_class
    hints["accessibility_policy"] = effective.accessibility_policy
    hints["access_sides"] = list(effective.access_sides)
    if effective.attachment_dependencies:
        hints["attachment_dependencies"] = list(effective.attachment_dependencies)
    if effective.orientation_dependencies:
        hints["orientation_dependencies"] = list(effective.orientation_dependencies)
    if effective.interaction_surface_map:
        hints["interaction_surface_map"] = effective.interaction_surface_map
    if effective.access_type:
        hints["access_type"] = effective.access_type
    if effective.interaction_height_m:
        hints["interaction_height_m"] = effective.interaction_height_m
    if effective.related_categories:
        hints["target_relation"] = effective.related_categories
    if effective.low_confidence_candidates:
        hints["low_confidence_candidates"] = effective.low_confidence_candidates
    profile = annotation.object_function_profile.model_dump(mode="json")
    if not any(bool(value) for value in profile.values()):
        profile = _object_function_profile_from_effective_annotation(obj, effective)
    obj.metadata["object_function_profile"] = profile


def _object_function_profile_from_effective_annotation(
    obj: SceneObject, effective: EffectiveAnnotation
) -> dict[str, bool]:
    return adapter._object_function_profile(
        obj, str(effective.category_norm or adapter._category_for_object(obj))
    )


def _annotation_config(config: Any) -> dict[str, Any]:
    if hasattr(config, "asset_annotation"):
        raw = getattr(config, "asset_annotation")
    else:
        raw = _cfg_get(config, "scenebenchmark_critic.asset_annotation", None)
        if raw is None:
            raw = _cfg_get(
                config, "experiment.scenebenchmark_critic.asset_annotation", None
            )
    raw_dict = _to_plain_dict(raw)
    merged = _deep_merge(DEFAULT_CONFIG, raw_dict)
    return merged


def _select_objects(scene: RoomScene, config: dict[str, Any]) -> list[SceneObject]:
    allowed_types = {
        str(item).strip()
        for item in (config.get("object_types") or [])
        if str(item).strip()
    }
    allowlist = {str(item) for item in (config.get("object_ids") or []) if str(item)}
    blocklist = {
        str(item) for item in (config.get("exclude_object_ids") or []) if str(item)
    }
    max_objects = config.get("max_objects")
    selected: list[SceneObject] = []
    for obj in scene.objects.values():
        object_id = str(obj.object_id)
        if allowlist and object_id not in allowlist:
            continue
        if object_id in blocklist:
            continue
        if allowed_types and obj.object_type.value not in allowed_types:
            continue
        if obj.object_type in {ObjectType.WALL, ObjectType.FLOOR}:
            continue
        selected.append(obj)
        if max_objects not in (None, "") and len(selected) >= int(max_objects):
            break
    return selected


def _render_or_collect_images(
    *,
    obj: SceneObject,
    config: dict[str, Any],
    blender_server: "BlenderServer | None",
    render_dir: Path,
) -> list[Path]:
    render_cfg = config.get("render") or {}
    if (
        _as_bool(render_cfg.get("enabled"))
        and blender_server is not None
        and obj.geometry_path is not None
        and Path(obj.geometry_path).is_file()
    ):
        try:
            return blender_server.render_multiview_for_analysis(
                mesh_path=Path(obj.geometry_path),
                output_dir=render_dir,
                elevation_degrees=float(render_cfg.get("elevation_degrees") or 20.0),
                num_side_views=int(render_cfg.get("num_side_views") or 4),
                include_vertical_views=_as_bool(
                    render_cfg.get("include_vertical_views")
                ),
                width=int(render_cfg.get("width") or 512),
                height=int(render_cfg.get("height") or 512),
                timeout=float(render_cfg.get("timeout") or 120.0),
                show_coordinate_frame=_as_bool(render_cfg.get("show_coordinate_frame")),
            )
        except Exception as exc:
            console_logger.warning(
                "Asset annotation render failed for %s: %s", obj.object_id, exc
            )
    if obj.image_path is not None and Path(obj.image_path).is_file():
        return [Path(obj.image_path)]
    return []


def _build_vlm_messages(
    *, prompt_text: str, image_paths: list[Path]
) -> list[dict[str, Any]]:
    system_prompt = (
        "You are an asset-level semantic annotator for 3D indoor scene benchmark "
        "assets. Classify only the object itself, not whether its current "
        "placement is good. Return strict JSON matching the requested schema."
    )
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt_text},
    ]
    for image_path in image_paths:
        if not image_path.is_file():
            continue
        encoded = encode_image_to_base64(image_path)
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            }
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_annotation_prompt(
    *,
    scene: RoomScene,
    obj: SceneObject,
    heuristic_prior: dict[str, Any],
    image_paths: list[Path],
    stage: str,
) -> str:
    hints = heuristic_prior.get("functional_hints") or {}
    schema_hint = {
        "canonical_name": "string or null",
        "category_norm": "string or null",
        "placement_class": "string or null",
        "scene_object_type": "wall_mounted | manipuland | ceiling_mounted | furniture | unknown",
        "semantic_size_class": "string or null",
        "benchmark_relevance": "functional | decorative | noise | unknown",
        "affordances": [
            {
                "label": "string",
                "anchor": "string or null",
                "required_face": "string or null",
                "confidence": "0..1",
            }
        ],
        "front_face": {
            "view": "front|back|left|right|top|bottom|null",
            "reason": "string",
            "confidence": "0..1",
        },
        "interaction_surface_map": {
            "front": ["..."],
            "back": [],
            "left": [],
            "right": [],
            "top": [],
            "bottom": [],
        },
        "access_type": {"primary": "string or null", "secondary": "string or null"},
        "mobility_class": "fixed | movable | semi_movable | mounted | unknown",
        "accessibility_policy": "required | optional | ignored",
        "access_sides": ["front|back|left|right|top|bottom"],
        "attachment_dependencies": [
            {
                "relation_type": "back_against_wall | side_or_back_against_wall | object_on_floor | mounted_to_wall | mounted_to_ceiling",
                "target_kind": "architecture",
                "target_category": "wall|floor|ceiling",
                "subject_face": "front|back|left|right|top|bottom|null",
                "target_face": "front|back|left|right|top|bottom|any|null",
                "max_angle_deg": "number|null",
                "max_distance_m": "number|null",
                "confidence": "0..1",
                "reason": "string",
            }
        ],
        "orientation_dependencies": [
            {
                "relation_type": "seat_faces_surface | furniture_faces_furniture",
                "target_kind": "object",
                "target_category": "string or list",
                "subject_face": "front|back|left|right|top|bottom",
                "target_face": "front|back|left|right|top|bottom|any|null",
                "max_angle_deg": "number|null",
                "max_distance_m": "number|null",
                "confidence": "0..1",
                "reason": "string",
            }
        ],
        "interaction_height_m": {"min": "number|null", "max": "number|null"},
        "related_categories": ["string"],
        "style_tags": ["string"],
        "ambiguity": {"is_ambiguous": "bool", "notes": ["string"]},
        "rationale": "string",
        "confidence": "0..1",
    }
    rule_prior = {
        "normalized_category_prior": heuristic_prior.get("category_norm"),
        "placement_prior": heuristic_prior.get("placement"),
        "scene_object_type_prior": heuristic_prior.get("scene_object_type"),
        "functional_hints_prior": {
            "functional_categories": hints.get("functional_categories") or [],
            "front_hint": hints.get("front_hint"),
            "category_group": hints.get("category_group"),
            "category_keywords": hints.get("category_keywords") or [],
            "target_relation": hints.get("target_relation") or [],
            "metric_relevance": hints.get("metric_relevance") or {},
            "mobility_class": hints.get("mobility_class"),
            "accessibility_policy": hints.get("accessibility_policy"),
            "access_sides": hints.get("access_sides") or [],
            "attachment_dependencies": hints.get("attachment_dependencies") or [],
            "orientation_dependencies": hints.get("orientation_dependencies") or [],
        },
        "geometry_prior": heuristic_prior.get("bbox_world") or {},
    }
    return (
        f"Object ID: {obj.object_id}\n"
        f"Room ID: {scene.room_id}\n"
        f"Room type: {scene.room_type}\n"
        f"Stage: {stage}\n"
        f"Raw category/name: {obj.name}\n"
        f"Normalized category prior: {heuristic_prior.get('category_norm')}\n"
        f"Description: {obj.description or 'none'}\n"
        f"SceneSmith object_type: {obj.object_type.value}\n"
        f"BBox world (m): {json.dumps(heuristic_prior.get('bbox_world'))}\n"
        f"Attached image count: {len(image_paths)}\n\n"
        "Rule/heuristic prior JSON. Treat it as prior evidence, not final truth:\n"
        f"{json.dumps(rule_prior, ensure_ascii=False)}\n\n"
        "Task: classify only the asset itself. Do not judge room placement. "
        "Use exactly one scene_object_type: wall_mounted, ceiling_mounted, "
        "manipuland, furniture, or unknown. For scene_object_type, classify the "
        "object's intended installation/function class, not merely the current "
        "transform. If uncertain, keep low confidence and mark ambiguity notes.\n"
        "For mobility/accessibility, movable seating such as chairs or stools may "
        "be optional rather than a hard spatial-accessibility requirement. For "
        "functional dependencies, follow SceneEval-style face_to orientation: "
        "chairs face desks/tables, and large storage/sleeping furniture often has "
        "a back-against-wall architecture relation. Ignore small-object orientation "
        "dependencies unless they are explicitly visible and important.\n"
        "Return only strict JSON matching this shape:\n"
        f"{json.dumps(schema_hint, ensure_ascii=False)}"
    )


def _bbox_world(obj: SceneObject) -> dict[str, Any]:
    bounds = obj.compute_world_bounds()
    if bounds is None:
        return {}
    bmin, bmax = bounds
    center = (bmin + bmax) / 2.0
    size = bmax - bmin
    return {
        "center": [float(value) for value in center],
        "size": [float(value) for value in size],
        "min": [float(value) for value in bmin],
        "max": [float(value) for value in bmax],
    }


def _placement_class(obj: SceneObject) -> str:
    if obj.object_type == ObjectType.WALL_MOUNTED:
        return "wall_mounted"
    if obj.object_type == ObjectType.CEILING_MOUNTED:
        return "ceiling_mounted"
    if obj.object_type == ObjectType.MANIPULAND:
        return "surface_object" if obj.placement_info else "manipuland"
    if obj.object_type == ObjectType.THIN_COVERING:
        return "thin_covering"
    return "floor_furniture"


def _scene_object_type(obj: SceneObject) -> SceneObjectType:
    if obj.object_type == ObjectType.WALL_MOUNTED:
        return "wall_mounted"
    if obj.object_type == ObjectType.CEILING_MOUNTED:
        return "ceiling_mounted"
    if obj.object_type in {ObjectType.MANIPULAND, ObjectType.THIN_COVERING}:
        return "manipuland"
    if obj.object_type == ObjectType.FURNITURE:
        return "furniture"
    return "unknown"


def _benchmark_relevance(
    category: str, group: str, affordances: list[str]
) -> Literal["functional", "decorative", "noise", "unknown"]:
    if set(affordances) & FUNCTIONAL_AFFORDANCES:
        return "functional"
    if category in DECORATIVE_CATEGORIES or group in {"decor", "soft_furnishing"}:
        return "decorative"
    if not category or category == "unknown":
        return "unknown"
    return "functional" if category not in {"wall", "floor", "ceiling"} else "noise"


def _prediction_confidence(
    category: str, relevance: str, affordances: list[str]
) -> float:
    if category and category != "unknown" and affordances:
        return 0.82
    if relevance in {"decorative", "noise"}:
        return 0.78
    if category and category != "unknown":
        return 0.62
    return 0.35


def _front_face(
    category: str, affordances: list[str], hints: dict[str, Any]
) -> str | None:
    existing = str(hints.get("front_hint") or hints.get("front_face") or "").strip()
    if existing:
        return existing
    if "supportable" in affordances:
        return "top"
    if "openable" in affordances or "sittable" in affordances:
        return "front"
    if category in {"mug", "cup", "bottle", "vase"}:
        return "side"
    return None


def _semantic_size_class(heuristic_prior: dict[str, Any]) -> str | None:
    size = (heuristic_prior.get("bbox_world") or {}).get("size") or []
    if len(size) < 3:
        return None
    max_dim = max(abs(float(value or 0.0)) for value in size[:3])
    if max_dim < 0.25:
        return "small"
    if max_dim < 1.0:
        return "medium"
    return "large"


def _anchor_for_affordance(label: str) -> str | None:
    return {
        "sittable": "seat_surface",
        "sleepable": "sleep_surface",
        "supportable": "top_surface",
        "openable": "front_access",
        "containable": "interior",
        "toggleable": "switch_or_body",
        "graspable": "grasp_region",
    }.get(label)


def _required_face_for_affordance(label: str, front_face: str | None) -> str | None:
    if label in {"openable", "sittable", "toggleable"}:
        return front_face or "front"
    if label in {"supportable", "sleepable"}:
        return "top"
    return front_face


def _interaction_surface_map(
    affordances: list[str], front_face: str | None
) -> dict[str, list[str]]:
    surface_map = {
        key: [] for key in ("front", "back", "left", "right", "top", "bottom")
    }
    face_key = _surface_map_face(front_face)
    if "supportable" in affordances:
        surface_map["top"].append("support_surface")
    if "sleepable" in affordances:
        surface_map["top"].append("sleep_surface")
    if "openable" in affordances:
        surface_map[face_key].append("openable_front")
    if "sittable" in affordances:
        surface_map[face_key].append("seat_access")
        surface_map["top"].append("seat_surface")
    if "graspable" in affordances:
        surface_map[face_key].append("graspable")
    return surface_map


def _surface_map_face(front_face: str | None) -> str:
    face = str(front_face or "").strip().lower()
    if face in {"front", "back", "left", "right", "top", "bottom"}:
        return face
    return "front"


def _access_type(affordances: list[str]) -> str | None:
    if "openable" in affordances:
        return "front_open"
    if "sittable" in affordances:
        return "front_sit"
    if "supportable" in affordances:
        return "top"
    if "graspable" in affordances:
        return "reachable"
    return None


def _mobility_class(
    category: str,
    scene_object_type: SceneObjectType,
    affordances: list[str],
    hints: dict[str, Any],
) -> MobilityClass:
    explicit = _normalize_mobility_class(hints.get("mobility_class"))
    if explicit != "unknown":
        return explicit
    if scene_object_type in {"wall_mounted", "ceiling_mounted"}:
        return "mounted"
    if category in WALL_MOUNTED_CATEGORIES or category in CEILING_MOUNTED_CATEGORIES:
        return "mounted"
    if category in MOVABLE_SEATING_CATEGORIES:
        return "movable"
    if category in SEMI_MOVABLE_FURNITURE_CATEGORIES:
        return "semi_movable"
    if category in WALL_BACKED_CATEGORIES or "openable" in affordances:
        return "fixed"
    return "unknown"


def _accessibility_policy(
    category: str,
    mobility_class: MobilityClass,
    affordances: list[str],
    scene_object_type: SceneObjectType,
    hints: dict[str, Any],
) -> AccessibilityPolicy:
    explicit = _normalize_accessibility_policy(hints.get("accessibility_policy"))
    if explicit is not None:
        return explicit
    if scene_object_type == "ceiling_mounted" or mobility_class == "mounted":
        return "ignored"
    if scene_object_type == "manipuland":
        return "required" if affordances else "ignored"
    if category in SMALL_LOOSE_CATEGORIES or affordances == ["graspable"]:
        return "ignored"
    if mobility_class == "movable" and "sittable" in affordances:
        return "optional"
    if {"openable", "sleepable"} & set(affordances):
        return "required"
    if category in WALL_BACKED_CATEGORIES:
        return "required"
    return "required" if affordances else "ignored"


def _access_sides(
    category: str,
    affordances: list[str],
    front_face: str | None,
    hints: dict[str, Any],
) -> list[str]:
    explicit = _normalize_face_list(hints.get("access_sides"))
    if explicit:
        return explicit
    face = _surface_map_face(front_face)
    sides: list[str] = []
    if "openable" in affordances or "sittable" in affordances:
        sides.append(face)
    if "supportable" in affordances:
        sides.append("top")
    if "sleepable" in affordances:
        sides.extend(["left", "right", face])
    if category in WALL_BACKED_CATEGORIES and "front" not in sides:
        sides.append("front")
    return _unique(sides)


def _attachment_dependencies(
    category: str,
    scene_object_type: SceneObjectType,
    hints: dict[str, Any],
) -> list[dict[str, Any]]:
    explicit = _as_dependency_list(hints.get("attachment_dependencies"))
    if explicit:
        return explicit
    dependencies: list[dict[str, Any]] = []
    if scene_object_type == "furniture":
        dependencies.append(
            {
                "relation_type": "object_on_floor",
                "target_kind": "architecture",
                "target_category": "floor",
                "max_distance_m": 0.12,
                "confidence": 0.74,
                "reason": "large floor furniture should be supported by the floor",
            }
        )
    if scene_object_type == "wall_mounted" or category in WALL_MOUNTED_CATEGORIES:
        return [
            {
                "relation_type": "mounted_to_wall",
                "target_kind": "architecture",
                "target_category": "wall",
                "subject_face": "back",
                "max_angle_deg": 45.0,
                "max_distance_m": 0.18,
                "confidence": 0.72,
                "reason": "wall-mounted assets should remain attached to a wall plane",
            }
        ]
    if scene_object_type == "ceiling_mounted" or category in CEILING_MOUNTED_CATEGORIES:
        return [
            {
                "relation_type": "mounted_to_ceiling",
                "target_kind": "architecture",
                "target_category": "ceiling",
                "max_distance_m": 0.18,
                "confidence": 0.72,
                "reason": "ceiling-mounted assets should remain attached to the ceiling",
            }
        ]
    if category in WALL_BACKED_CATEGORIES:
        dependencies.append(
            {
                "relation_type": "back_against_wall",
                "target_kind": "architecture",
                "target_category": "wall",
                "subject_face": "back",
                "max_angle_deg": 45.0,
                "max_distance_m": 0.25,
                "confidence": 0.68,
                "reason": "large storage/sleeping furniture is normally backed by a wall",
            }
        )
    return dependencies


def _orientation_dependencies(
    category: str, affordances: list[str], hints: dict[str, Any]
) -> list[dict[str, Any]]:
    explicit = _as_dependency_list(hints.get("orientation_dependencies"))
    if explicit:
        return explicit
    if category in MOVABLE_SEATING_CATEGORIES or (
        "sittable" in affordances and category not in {"sofa", "loveseat", "bench"}
    ):
        return [
            {
                "relation_type": "seat_faces_surface",
                "target_kind": "object",
                "target_category": ["desk", "table", "dining_table"],
                "subject_face": "front",
                "target_face": "any",
                "max_angle_deg": 90.0,
                "max_distance_m": 1.5,
                "confidence": 0.70,
                "reason": "SceneEval-style face_to relation for chair/table pairs",
            }
        ]
    return []


def _interaction_height(heuristic_prior: dict[str, Any]) -> dict[str, float | None]:
    bbox = heuristic_prior.get("bbox_world") or {}
    bmin = bbox.get("min") or []
    bmax = bbox.get("max") or []
    if len(bmin) < 3 or len(bmax) < 3:
        return {"min": None, "max": None}
    return {"min": float(bmin[2]), "max": float(bmax[2])}


def _resolve_scene_object_type(
    heuristic_prior: dict[str, Any],
    vlm_prediction: AssetVlmPrediction,
    *,
    threshold: float,
    low_confidence: dict[str, Any],
) -> SceneObjectType:
    metadata = heuristic_prior.get("source_metadata") or {}
    source_type = _normalize_scene_object_type(metadata.get("object_type"))
    if source_type != "unknown":
        return source_type
    hints = heuristic_prior.get("functional_hints") or {}
    existing = _normalize_scene_object_type(hints.get("scene_object_type"))
    if existing != "unknown":
        return existing
    vlm_type = _normalize_scene_object_type(vlm_prediction.scene_object_type)
    if vlm_type != "unknown":
        if vlm_prediction.confidence >= threshold:
            return vlm_type
        low_confidence["scene_object_type"] = {
            "value": vlm_type,
            "confidence": vlm_prediction.confidence,
        }
    return _normalize_scene_object_type(heuristic_prior.get("scene_object_type"))


def _normalize_scene_object_type(value: Any) -> SceneObjectType:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text if text in SCENE_OBJECT_TYPES else "unknown"  # type: ignore[return-value]


def _resolve_mobility_class(
    hints: dict[str, Any], vlm_prediction: AssetVlmPrediction
) -> MobilityClass:
    existing = _normalize_mobility_class(hints.get("mobility_class"))
    if existing != "unknown":
        return existing
    return _normalize_mobility_class(vlm_prediction.mobility_class)


def _normalize_mobility_class(value: Any) -> MobilityClass:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text if text in MOBILITY_CLASSES else "unknown"  # type: ignore[return-value]


def _resolve_accessibility_policy(
    hints: dict[str, Any],
    vlm_prediction: AssetVlmPrediction,
    scene_object_type: SceneObjectType,
    affordances: list[str],
) -> AccessibilityPolicy:
    existing = _normalize_accessibility_policy(hints.get("accessibility_policy"))
    if scene_object_type == "manipuland" and affordances:
        return "required"
    if existing is not None:
        return existing
    return (
        _normalize_accessibility_policy(vlm_prediction.accessibility_policy)
        or "required"
    )


def _normalize_accessibility_policy(value: Any) -> AccessibilityPolicy | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text if text in ACCESSIBILITY_POLICIES else None  # type: ignore[return-value]


def _resolve_access_sides(
    hints: dict[str, Any], vlm_prediction: AssetVlmPrediction
) -> list[str]:
    existing = _normalize_face_list(hints.get("access_sides"))
    if existing:
        return existing
    return _normalize_face_list(vlm_prediction.access_sides)


def _normalize_face_list(raw: Any) -> list[str]:
    values = raw if isinstance(raw, list) else [raw]
    faces: list[str] = []
    for value in values:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if text in ACCESS_FACE_NAMES and text not in faces:
            faces.append(text)
    return faces


def _resolve_dependency_list(
    hints: dict[str, Any], vlm_prediction: AssetVlmPrediction, key: str
) -> list[dict[str, Any]]:
    existing = _as_dependency_list(hints.get(key))
    if existing:
        return existing
    return _as_dependency_list(getattr(vlm_prediction, key, []))


def _as_dependency_list(raw: Any) -> list[dict[str, Any]]:
    values = raw if isinstance(raw, list) else [raw]
    dependencies: list[dict[str, Any]] = []
    for item in values:
        if isinstance(item, dict):
            dependencies.append(dict(item))
    return dependencies


def _model_name(config: dict[str, Any], raw_config: Any | None) -> str:
    configured = str(config.get("model") or "").strip()
    if configured:
        return configured
    return str(_cfg_get(raw_config, "openai.model", "gpt-5.2"))


def _load_annotation(path: Path) -> AssetAnnotation | None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return AssetAnnotation.model_validate(payload)
    except Exception:
        return None


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _evidence_hash(*, obj: SceneObject, image_paths: list[Path]) -> str:
    payload = {
        "object_id": str(obj.object_id),
        "name": obj.name,
        "description": obj.description,
        "geometry_path": str(obj.geometry_path) if obj.geometry_path else None,
        "image_paths": [str(path) for path in image_paths],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _unique(values: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return {key: item for key, item in value.items()}
    return {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _cfg_get(config: Any, dotted_path: str, default: Any = None) -> Any:
    current = config
    for part in dotted_path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
    return current
