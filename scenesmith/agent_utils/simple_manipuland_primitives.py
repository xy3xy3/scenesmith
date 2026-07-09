"""Procedural fallback meshes for simple prompt-critical manipulands.

These meshes are intentionally plain. They are used only after configured asset
strategies fail, so missing or semantically wrong small objects can degrade into
simple-but-correct geometry instead of disappearing from the scene.
"""

from __future__ import annotations

import math
import re
import time

from pathlib import Path

import numpy as np

from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    GLTF2,
    UNSIGNED_INT,
    Accessor,
    Attributes,
    Buffer,
    BufferView,
    Material as GltfMaterial,
    Mesh,
    Node,
    PbrMetallicRoughness,
    Primitive,
    Scene,
)

from scenesmith.agent_utils.asset_router.dataclasses import AssetItem
from scenesmith.agent_utils.manipuland_scale import match_size_profile
from scenesmith.utils.gltf_generation import zup_to_yup_transform


_SUPPORTED_PROFILE_NAMES = {
    "coaster",
    "cutlery",
    "notebook_book",
    "plate_bowl",
    "remote_control",
    "computer_monitor",
    "keyboard",
    "mouse",
    "writing_instrument",
}


def can_generate_simple_manipuland_primitive(item: AssetItem) -> bool:
    """Return True when ``item`` is a conservative primitive fallback match."""
    category = _primitive_category(item)
    return category is not None


def generate_simple_manipuland_primitive(
    item: AssetItem,
    output_dir: Path,
) -> Path | None:
    """Generate a self-contained GLB primitive for a known simple manipuland."""
    category = _primitive_category(item)
    if category is None:
        return None

    width, depth, height = (float(value) for value in item.dimensions)
    mesh = _mesh_for_category(category, width, depth, height)
    if mesh is None:
        return None

    vertices_zup, normals_zup, indices = mesh
    vertices = zup_to_yup_transform(vertices_zup)
    normals = zup_to_yup_transform(normals_zup)
    uvs = np.zeros((len(vertices), 2), dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir / f"{item.short_name}_procedural_primitive_{time.time_ns()}.glb"
    )
    _write_colored_glb(
        vertices=vertices,
        normals=normals,
        uvs=uvs,
        indices=indices,
        output_path=output_path,
        color=_color_for_category(category),
    )
    return output_path


def _primitive_category(item: AssetItem) -> str | None:
    profile = match_size_profile(item.description, item.short_name)
    if profile is not None and profile.name in _SUPPORTED_PROFILE_NAMES:
        if profile.name == "plate_bowl" and _matches(item, (r"\bbowl\b",)):
            return None
        return profile.name

    if _matches(item, (r"\bcoaster(s)?\b", r"\bdrink mat(s)?\b")):
        return "coaster"
    if _matches(
        item,
        (
            r"\bnapkin(s)?\b",
            r"\bplacemat(s)?\b",
            r"\bplace mat(s)?\b",
            r"\btable runner(s)?\b",
            r"\btablecloth(s)?\b",
            r"\blinen(s)?\b",
        ),
    ):
        # 2026-07-09 修改原因：critic probe 中 materials client 关闭时，
        # napkin 会被 router 选成 thin_covering 后直接失败；用保守薄片
        # primitive 保留 prompt-critical 餐桌织物，不依赖材料服务。
        return "table_linen"
    if _matches(
        item,
        (
            r"\bfork(s)?\b",
            r"\bknife\b",
            r"\bknives\b",
            r"\bspoon(s)?\b",
            r"\bcutlery\b",
            r"\butensil(s)?\b",
            r"\bflatware\b",
        ),
    ):
        return "cutlery"
    if _matches(item, (r"\bremote\b", r"\bremote control\b")):
        return "remote_control"
    return None


def _matches(item: AssetItem, patterns: tuple[str, ...]) -> bool:
    haystack = f"{item.description} {item.short_name}".lower()
    return any(re.search(pattern, haystack) for pattern in patterns)


def _mesh_for_category(
    category: str,
    width: float,
    depth: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if category == "coaster":
        return _elliptic_cylinder_mesh(width, depth, height, segments=48)
    if category == "table_linen":
        linen_height = min(max(height, 0.004), 0.012)
        return _combined_boxes(
            [((0.0, 0.0, linen_height / 2.0), (width, depth, linen_height))]
        )
    if category == "plate_bowl":
        return _elliptic_cylinder_mesh(width, depth, height, segments=64)
    if category in {"notebook_book", "remote_control", "keyboard"}:
        return _combined_boxes(
            [((0.0, 0.0, height / 2.0), (width, depth, height))]
        )
    if category == "mouse":
        return _mouse_mesh(width, depth, height)
    if category == "writing_instrument":
        return _writing_instrument_mesh(width, depth, height)
    if category == "cutlery":
        return _cutlery_mesh(width, depth, height)
    if category == "computer_monitor":
        return _monitor_mesh(width, depth, height)
    return None


def _cutlery_mesh(
    width: float,
    depth: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if width >= depth:
        handle_size = (width * 0.72, depth * 0.22, height)
        head_size = (width * 0.22, depth * 0.45, height)
        boxes = [
            ((-width * 0.12, 0.0, height / 2.0), handle_size),
            ((width * 0.36, 0.0, height / 2.0), head_size),
        ]
    else:
        handle_size = (width * 0.22, depth * 0.72, height)
        head_size = (width * 0.45, depth * 0.22, height)
        boxes = [
            ((0.0, -depth * 0.12, height / 2.0), handle_size),
            ((0.0, depth * 0.36, height / 2.0), head_size),
        ]
    return _combined_boxes(boxes)


def _monitor_mesh(
    width: float,
    depth: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    screen_height = height * 0.72
    screen_thickness = min(depth * 0.35, 0.04)
    base_height = max(height * 0.05, 0.01)
    stand_height = height - screen_height - base_height
    boxes = [
        (
            (0.0, 0.0, base_height / 2.0),
            (width * 0.42, depth, base_height),
        ),
        (
            (0.0, 0.0, base_height + stand_height / 2.0),
            (width * 0.08, depth * 0.28, stand_height),
        ),
        (
            (0.0, 0.0, height - screen_height / 2.0),
            (width, screen_thickness, screen_height),
        ),
    ]
    return _combined_boxes(boxes)


def _mouse_mesh(
    width: float,
    depth: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 2026-07-09 修改原因：HSSD 常把 mouse 检索成板/支架；用低矮椭圆
    # primitive 保留“桌面鼠标”语义，避免 designer 反复换词重试。
    body = _elliptic_cylinder_mesh(width, depth, height, segments=48)
    button_bar_height = max(height * 0.12, 0.003)
    button_bar_width = min(width * 0.08, 0.008)
    button = _combined_boxes(
        [
            (
                (0.0, depth * 0.18, height + button_bar_height / 2.0),
                (button_bar_width, depth * 0.45, button_bar_height),
            )
        ]
    )
    return _combine_meshes([body, button])


def _writing_instrument_mesh(
    width: float,
    depth: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 2026-07-09 修改原因：单支 pen/pencil 是常见 prompt-critical 小物；
    # 失败时用细长 primitive，避免错类 HSSD 候选拖慢 critic-on。
    long_axis_is_x = width >= depth
    length = max(width, depth)
    thickness = min(max(min(width, depth), height), 0.02)
    shaft_length = length * 0.86
    tip_length = length - shaft_length

    if long_axis_is_x:
        boxes = [
            (
                (-tip_length / 2.0, 0.0, height / 2.0),
                (shaft_length, thickness, thickness),
            ),
            (
                (shaft_length / 2.0, 0.0, height / 2.0),
                (tip_length, thickness * 0.65, thickness * 0.65),
            ),
        ]
    else:
        boxes = [
            (
                (0.0, -tip_length / 2.0, height / 2.0),
                (thickness, shaft_length, thickness),
            ),
            (
                (0.0, shaft_length / 2.0, height / 2.0),
                (thickness * 0.65, tip_length, thickness * 0.65),
            ),
        ]
    return _combined_boxes(boxes)


def _combine_meshes(
    meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices_parts: list[np.ndarray] = []
    normals_parts: list[np.ndarray] = []
    indices_parts: list[np.ndarray] = []
    vertex_offset = 0
    for vertices, normals, indices in meshes:
        vertices_parts.append(vertices)
        normals_parts.append(normals)
        indices_parts.append(indices + vertex_offset)
        vertex_offset += len(vertices)
    return (
        np.vstack(vertices_parts).astype(np.float32),
        np.vstack(normals_parts).astype(np.float32),
        np.concatenate(indices_parts).astype(np.uint32),
    )


def _combined_boxes(
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices_parts: list[np.ndarray] = []
    normals_parts: list[np.ndarray] = []
    indices_parts: list[np.ndarray] = []
    vertex_offset = 0
    for center, size in boxes:
        vertices, normals, indices = _box_mesh(*size, center=center)
        vertices_parts.append(vertices)
        normals_parts.append(normals)
        indices_parts.append(indices + vertex_offset)
        vertex_offset += len(vertices)
    return (
        np.vstack(vertices_parts).astype(np.float32),
        np.vstack(normals_parts).astype(np.float32),
        np.concatenate(indices_parts).astype(np.uint32),
    )


def _box_mesh(
    width: float,
    depth: float,
    height: float,
    *,
    center: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy, cz = center
    x0, x1 = cx - width / 2.0, cx + width / 2.0
    y0, y1 = cy - depth / 2.0, cy + depth / 2.0
    z0, z1 = cz - height / 2.0, cz + height / 2.0
    vertices = np.array(
        [
            [x0, y1, z0],
            [x1, y1, z0],
            [x1, y1, z1],
            [x0, y1, z1],
            [x1, y0, z0],
            [x0, y0, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z0],
            [x1, y0, z0],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y0, z0],
            [x0, y1, z0],
            [x0, y1, z1],
            [x0, y0, z1],
            [x0, y1, z1],
            [x1, y1, z1],
            [x1, y0, z1],
            [x0, y0, z1],
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
        ],
        dtype=np.float32,
    )
    normals = np.repeat(
        np.array(
            [
                [0, 1, 0],
                [0, -1, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            dtype=np.float32,
        ),
        4,
        axis=0,
    )
    indices = np.array(
        [
            0,
            1,
            2,
            0,
            2,
            3,
            4,
            5,
            6,
            4,
            6,
            7,
            8,
            9,
            10,
            8,
            10,
            11,
            12,
            13,
            14,
            12,
            14,
            15,
            16,
            17,
            18,
            16,
            18,
            19,
            20,
            21,
            22,
            20,
            22,
            23,
        ],
        dtype=np.uint32,
    )
    return vertices, normals, indices


def _elliptic_cylinder_mesh(
    width: float,
    depth: float,
    height: float,
    *,
    segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rx = width / 2.0
    ry = depth / 2.0
    angles = [2.0 * math.pi * i / segments for i in range(segments)]
    vertices: list[list[float]] = [[0.0, 0.0, height], [0.0, 0.0, 0.0]]
    normals: list[list[float]] = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    for z, nz in ((height, 1.0), (0.0, -1.0)):
        for angle in angles:
            vertices.append([rx * math.cos(angle), ry * math.sin(angle), z])
            normals.append([0.0, 0.0, nz])
    side_start = len(vertices)
    for angle in angles:
        nx = math.cos(angle)
        ny = math.sin(angle)
        vertices.append([rx * nx, ry * ny, height])
        normals.append([nx, ny, 0.0])
    for angle in angles:
        nx = math.cos(angle)
        ny = math.sin(angle)
        vertices.append([rx * nx, ry * ny, 0.0])
        normals.append([nx, ny, 0.0])

    indices: list[int] = []
    top_start = 2
    bottom_start = 2 + segments
    for i in range(segments):
        j = (i + 1) % segments
        indices.extend([0, top_start + i, top_start + j])
        indices.extend([1, bottom_start + j, bottom_start + i])
        ti = side_start + i
        tj = side_start + j
        bi = side_start + segments + i
        bj = side_start + segments + j
        indices.extend([ti, bi, bj, ti, bj, tj])

    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )


def _write_colored_glb(
    *,
    vertices: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    indices: np.ndarray,
    output_path: Path,
    color: tuple[float, float, float, float],
) -> None:
    vertices_binary = vertices.astype(np.float32).tobytes()
    normals_binary = normals.astype(np.float32).tobytes()
    uvs_binary = uvs.astype(np.float32).tobytes()
    indices_binary = indices.astype(np.uint32).tobytes()
    buffer_data = vertices_binary + normals_binary + uvs_binary + indices_binary

    buffer_views = [
        BufferView(
            buffer=0,
            byteOffset=0,
            byteLength=len(vertices_binary),
            target=ARRAY_BUFFER,
        ),
        BufferView(
            buffer=0,
            byteOffset=len(vertices_binary),
            byteLength=len(normals_binary),
            target=ARRAY_BUFFER,
        ),
        BufferView(
            buffer=0,
            byteOffset=len(vertices_binary) + len(normals_binary),
            byteLength=len(uvs_binary),
            target=ARRAY_BUFFER,
        ),
        BufferView(
            buffer=0,
            byteOffset=len(vertices_binary) + len(normals_binary) + len(uvs_binary),
            byteLength=len(indices_binary),
            target=ELEMENT_ARRAY_BUFFER,
        ),
    ]
    accessors = [
        Accessor(
            bufferView=0,
            byteOffset=0,
            componentType=FLOAT,
            count=len(vertices),
            type="VEC3",
            min=vertices.min(axis=0).tolist(),
            max=vertices.max(axis=0).tolist(),
        ),
        Accessor(
            bufferView=1,
            byteOffset=0,
            componentType=FLOAT,
            count=len(normals),
            type="VEC3",
        ),
        Accessor(
            bufferView=2,
            byteOffset=0,
            componentType=FLOAT,
            count=len(uvs),
            type="VEC2",
        ),
        Accessor(
            bufferView=3,
            byteOffset=0,
            componentType=UNSIGNED_INT,
            count=len(indices),
            type="SCALAR",
        ),
    ]
    material = GltfMaterial(
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=list(color),
            metallicFactor=0.0,
            roughnessFactor=0.75,
        ),
        doubleSided=False,
    )
    primitive = Primitive(
        attributes=Attributes(POSITION=0, NORMAL=1, TEXCOORD_0=2),
        indices=3,
        material=0,
    )
    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=[Node(mesh=0)],
        meshes=[Mesh(primitives=[primitive])],
        materials=[material],
        accessors=accessors,
        bufferViews=buffer_views,
        buffers=[Buffer(byteLength=len(buffer_data))],
    )
    gltf.set_binary_blob(buffer_data)
    gltf.save(str(output_path))


def _color_for_category(category: str) -> tuple[float, float, float, float]:
    return {
        "coaster": (0.18, 0.16, 0.12, 1.0),
        "table_linen": (0.93, 0.90, 0.84, 1.0),
        "cutlery": (0.75, 0.72, 0.68, 1.0),
        "notebook_book": (0.08, 0.16, 0.50, 1.0),
        "plate_bowl": (0.86, 0.84, 0.78, 1.0),
        "remote_control": (0.03, 0.03, 0.035, 1.0),
        "computer_monitor": (0.02, 0.025, 0.03, 1.0),
        "keyboard": (0.04, 0.04, 0.045, 1.0),
        "mouse": (0.02, 0.02, 0.025, 1.0),
        "writing_instrument": (0.04, 0.035, 0.032, 1.0),
    }.get(category, (0.5, 0.5, 0.5, 1.0))
