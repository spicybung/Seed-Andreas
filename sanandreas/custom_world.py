from __future__ import annotations

import json
import math
import random
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .gta_integration import GameScan, build_augmented_gta_dat
from .textures import MATERIAL_TEXTURE_NAMES
from .generator import GENERATOR_VERSION


RW_STRUCT = 0x01
RW_STRING = 0x02
RW_EXTENSION = 0x03
RW_TEXTURE = 0x06
RW_MATERIAL = 0x07
RW_MATERIAL_LIST = 0x08
RW_FRAME_LIST = 0x0E
RW_GEOMETRY = 0x0F
RW_CLUMP = 0x10
RW_ATOMIC = 0x14
RW_GEOMETRY_LIST = 0x1A
RW_BIN_MESH = 0x050E
RW_FRAME_NAME = 0x0253F2FE
RW_LIBRARY_ID = 0x1803FFFF

GEOMETRY_POSITIONS = 0x00000002
GEOMETRY_TEXTURED = 0x00000004
GEOMETRY_PRELIT = 0x00000008
GEOMETRY_NORMALS = 0x00000010
GEOMETRY_LIGHT = 0x00000020
GEOMETRY_MODULATE_MATERIAL = 0x00000040

MATERIAL_ROAD = 0
MATERIAL_SIDEWALK = 1
MATERIAL_BUILDING = 2
MATERIAL_ROOF = 3
MATERIAL_WINDOW = 4
MATERIAL_VEGETATION = 5
MATERIAL_TRIM = 6
MATERIAL_GRASS = 7
UNSAFE_REPLACEMENT_FLAGS = 0x00000800 | 0x00001000 | 0x00100000


@dataclass
class MeshData:
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    normals: list[tuple[float, float, float]] = field(default_factory=list)
    colors: list[tuple[int, int, int, int]] = field(default_factory=list)
    uvs: list[tuple[float, float]] = field(default_factory=list)
    triangles: list[tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class GeneratedModel:
    model_id: int
    model_name: str
    render_mesh: MeshData
    collision_mesh: MeshData
    building_count: int
    style: str
    txd_name: str = ""
    material_texture_names: tuple[str, ...] = MATERIAL_TEXTURE_NAMES
    building_debug_records: list[object] = field(default_factory=list)


@dataclass
class CustomPlacement:
    model_id: int
    model_name: str
    x: float
    y: float
    z: float
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0


@dataclass
class CustomWorldManifest:
    generator_version: str
    seed: int
    game_directory: str
    mod_directory: str
    target_center: tuple[float, float, float]
    grid_size: int
    tile_size: float
    custom_models: int
    custom_buildings: int
    render_vertices: int
    render_triangles: int
    collision_vertices: int
    collision_triangles: int
    model_id_first: int
    model_id_last: int
    dff_validation: list[str]
    col_validation: list[str]


@dataclass
class CustomGeneratedWorld:
    models: list[GeneratedModel]
    placements: list[CustomPlacement]
    manifest: CustomWorldManifest


class MeshBuilder:
    def __init__(self) -> None:
        self.mesh = MeshData()

    def add_quad(
        self,
        points: tuple[tuple[float, float, float], ...],
        material: int,
        color: tuple[int, int, int, int],
        uvs: tuple[tuple[float, float], ...] | None = None,
    ) -> None:
        if len(points) != 4:
            raise ValueError("A quad requires exactly four points.")
        normal = calculate_normal(points[0], points[1], points[2])
        first_index = len(self.mesh.vertices)
        self.mesh.vertices.extend(points)
        self.mesh.normals.extend([normal] * 4)
        self.mesh.colors.extend([color] * 4)
        self.mesh.uvs.extend(uvs or ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)))
        self.mesh.triangles.append((first_index, first_index + 1, first_index + 2, material))
        self.mesh.triangles.append((first_index, first_index + 2, first_index + 3, material))

    def add_triangle(
        self,
        points: tuple[tuple[float, float, float], ...],
        material: int,
        color: tuple[int, int, int, int],
        uvs: tuple[tuple[float, float], ...] | None = None,
    ) -> None:
        if len(points) != 3:
            raise ValueError("A triangle requires exactly three points.")
        normal = calculate_normal(points[0], points[1], points[2])
        first_index = len(self.mesh.vertices)
        self.mesh.vertices.extend(points)
        self.mesh.normals.extend([normal] * 3)
        self.mesh.colors.extend([color] * 3)
        self.mesh.uvs.extend(uvs or ((0.0, 1.0), (1.0, 1.0), (0.5, 0.0)))
        self.mesh.triangles.append((first_index, first_index + 1, first_index + 2, material))

    def add_double_sided_quad(
        self,
        points: tuple[tuple[float, float, float], ...],
        material: int,
        color: tuple[int, int, int, int],
        uvs: tuple[tuple[float, float], ...] | None = None,
    ) -> None:
        mapping = uvs or ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))
        self.add_quad(points, material, color, mapping)
        self.add_quad(tuple(reversed(points)), material, color, tuple(reversed(mapping)))

    def add_crossed_billboard(
        self,
        center_x: float,
        center_y: float,
        minimum_z: float,
        width: float,
        height: float,
        material: int,
        color: tuple[int, int, int, int],
    ) -> None:
        radius = width * 0.5 / math.sqrt(2.0)
        maximum_z = minimum_z + height
        self.add_double_sided_quad(
            (
                (center_x - radius, center_y - radius, minimum_z),
                (center_x + radius, center_y + radius, minimum_z),
                (center_x + radius, center_y + radius, maximum_z),
                (center_x - radius, center_y - radius, maximum_z),
            ),
            material,
            color,
        )
        self.add_double_sided_quad(
            (
                (center_x - radius, center_y + radius, minimum_z),
                (center_x + radius, center_y - radius, minimum_z),
                (center_x + radius, center_y - radius, maximum_z),
                (center_x - radius, center_y + radius, maximum_z),
            ),
            material,
            color,
        )

    def add_box(
        self,
        center_x: float,
        center_y: float,
        minimum_z: float,
        width: float,
        depth: float,
        height: float,
        side_material: int,
        top_material: int,
        color: tuple[int, int, int, int],
    ) -> None:
        half_width = width * 0.5
        half_depth = depth * 0.5
        x0 = center_x - half_width
        x1 = center_x + half_width
        y0 = center_y - half_depth
        y1 = center_y + half_depth
        z0 = minimum_z
        z1 = minimum_z + height
        dark = shade_color(color, 0.72)
        medium = shade_color(color, 0.84)
        light = shade_color(color, 1.08)

        top_u = max(1.0, width / 8.0)
        top_v = max(1.0, depth / 8.0)
        side_u = max(1.0, width / 6.0)
        depth_u = max(1.0, depth / 6.0)
        side_v = max(1.0, height / 4.0)
        self.add_quad(
            ((x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)),
            side_material,
            dark,
            ((0.0, top_v), (top_u, top_v), (top_u, 0.0), (0.0, 0.0)),
        )
        self.add_quad(
            ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)),
            top_material,
            light,
            ((0.0, top_v), (top_u, top_v), (top_u, 0.0), (0.0, 0.0)),
        )
        self.add_quad(
            ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)),
            side_material,
            medium,
            ((0.0, side_v), (side_u, side_v), (side_u, 0.0), (0.0, 0.0)),
        )
        self.add_quad(
            ((x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)),
            side_material,
            color,
            ((0.0, side_v), (side_u, side_v), (side_u, 0.0), (0.0, 0.0)),
        )
        self.add_quad(
            ((x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)),
            side_material,
            dark,
            ((0.0, side_v), (depth_u, side_v), (depth_u, 0.0), (0.0, 0.0)),
        )
        self.add_quad(
            ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)),
            side_material,
            medium,
            ((0.0, side_v), (depth_u, side_v), (depth_u, 0.0), (0.0, 0.0)),
        )

    def add_polygon_prism(
        self,
        points: tuple[tuple[float, float], ...],
        minimum_z: float,
        height: float,
        side_material: int,
        top_material: int,
        color: tuple[int, int, int, int],
    ) -> None:
        if len(points) < 3:
            raise ValueError("A polygon prism requires at least three points.")
        maximum_z = minimum_z + height
        center_x = sum(point[0] for point in points) / len(points)
        center_y = sum(point[1] for point in points) / len(points)
        center_index = len(self.mesh.vertices)
        self.mesh.vertices.append((center_x, center_y, maximum_z))
        self.mesh.normals.append((0.0, 0.0, 1.0))
        self.mesh.colors.append(shade_color(color, 1.06))
        self.mesh.uvs.append((center_x / 8.0, center_y / 8.0))
        top_indices: list[int] = []
        for x, y in points:
            top_indices.append(len(self.mesh.vertices))
            self.mesh.vertices.append((x, y, maximum_z))
            self.mesh.normals.append((0.0, 0.0, 1.0))
            self.mesh.colors.append(shade_color(color, 1.06))
            self.mesh.uvs.append((x / 8.0, y / 8.0))
        for index in range(len(points)):
            self.mesh.triangles.append(
                (center_index, top_indices[index], top_indices[(index + 1) % len(points)], top_material)
            )
        for index, first in enumerate(points):
            second = points[(index + 1) % len(points)]
            length = math.hypot(second[0] - first[0], second[1] - first[1])
            self.add_quad(
                (
                    (first[0], first[1], minimum_z),
                    (second[0], second[1], minimum_z),
                    (second[0], second[1], maximum_z),
                    (first[0], first[1], maximum_z),
                ),
                side_material,
                shade_color(color, 0.83),
                ((0.0, 1.0), (length / 8.0, 1.0), (length / 8.0, 0.0), (0.0, 0.0)),
            )


def calculate_normal(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    third: tuple[float, float, float],
) -> tuple[float, float, float]:
    first_edge = (second[0] - first[0], second[1] - first[1], second[2] - first[2])
    second_edge = (third[0] - first[0], third[1] - first[1], third[2] - first[2])
    normal = (
        first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
        first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
        first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
    )
    length = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    if length <= 0.000001:
        return 0.0, 0.0, 1.0
    return normal[0] / length, normal[1] / length, normal[2] / length


def shade_color(color: tuple[int, int, int, int], multiplier: float) -> tuple[int, int, int, int]:
    return (
        max(0, min(255, round(color[0] * multiplier))),
        max(0, min(255, round(color[1] * multiplier))),
        max(0, min(255, round(color[2] * multiplier))),
        color[3],
    )


BUILDING_PALETTE = (
    (165, 126, 96, 255),
    (183, 158, 119, 255),
    (126, 145, 153, 255),
    (151, 118, 123, 255),
    (174, 172, 153, 255),
    (113, 133, 122, 255),
    (191, 144, 92, 255),
    (120, 119, 139, 255),
)


def generate_custom_world(
    scan: GameScan,
    seed: int,
    target_x: float = 2600.0,
    target_y: float = -2600.0,
    target_z: float = 28.0,
    grid_size: int = 5,
    tile_size: float = 112.0,
) -> CustomGeneratedWorld:
    if grid_size < 2 or grid_size > 7:
        raise ValueError("Grid size must be between 2 and 7.")
    model_count = grid_size * grid_size
    replacement_models = sorted(
        (
            model
            for model in scan.models.values()
            if model.section == "objs" and not model.flags & UNSAFE_REPLACEMENT_FLAGS
        ),
        key=lambda model: model.model_id,
    )
    if len(replacement_models) < model_count:
        raise RuntimeError(
            f"Only {len(replacement_models)} reusable object definitions were found, but {model_count} are required."
        )
    replacement_models = replacement_models[:model_count]
    model_ids = [model.model_id for model in replacement_models]
    random_source = random.Random(seed)
    models: list[GeneratedModel] = []
    placements: list[CustomPlacement] = []
    start_offset = -((grid_size - 1) * tile_size) * 0.5

    for row in range(grid_size):
        for column in range(grid_size):
            model_index = row * grid_size + column
            model_id = model_ids[model_index]
            model_name = replacement_models[model_index].model_name
            tile_random = random.Random(random_source.getrandbits(64))
            generated_model = generate_city_tile(model_id, model_name, tile_size, tile_random)
            models.append(generated_model)
            placements.append(
                CustomPlacement(
                    model_id=model_id,
                    model_name=model_name,
                    x=target_x + start_offset + column * tile_size,
                    y=target_y + start_offset + row * tile_size,
                    z=target_z,
                )
            )

    dff_validation: list[str] = []
    for model in models:
        dff_data = write_dff(model.model_name, model.render_mesh)
        errors = validate_dff(dff_data)
        if errors:
            raise RuntimeError(f"Generated DFF {model.model_name} failed validation: " + "; ".join(errors))
    dff_validation.append(f"PASS: {len(models)} custom RenderWare DFF clumps validated.")
    dff_validation.append("PASS: Every DFF contains geometry, materials, prelights, normals and a Bin Mesh PLG.")

    col_data = write_col3_archive(models)
    col_errors = validate_col3_archive(col_data, len(models))
    if col_errors:
        raise RuntimeError("Generated COL3 archive failed validation: " + "; ".join(col_errors))
    col_validation = [f"PASS: COL3 archive contains {len(models)} matching custom collision models."]

    manifest = CustomWorldManifest(
        generator_version=GENERATOR_VERSION,
        seed=seed,
        game_directory=scan.game_directory,
        mod_directory="",
        target_center=(target_x, target_y, target_z),
        grid_size=grid_size,
        tile_size=tile_size,
        custom_models=len(models),
        custom_buildings=sum(model.building_count for model in models),
        render_vertices=sum(len(model.render_mesh.vertices) for model in models),
        render_triangles=sum(len(model.render_mesh.triangles) for model in models),
        collision_vertices=sum(len(model.collision_mesh.vertices) for model in models),
        collision_triangles=sum(len(model.collision_mesh.triangles) for model in models),
        model_id_first=model_ids[0],
        model_id_last=model_ids[-1],
        dff_validation=dff_validation,
        col_validation=col_validation,
    )
    return CustomGeneratedWorld(models, placements, manifest)


def allocate_model_ids(occupied_ids: set[int], count: int) -> list[int]:
    available = [model_id for model_id in range(19000, 20000) if model_id not in occupied_ids]
    if len(available) < count:
        available = [model_id for model_id in range(18000, 20000) if model_id not in occupied_ids]
    if len(available) < count:
        raise RuntimeError(f"Could not allocate {count} unused object IDs below the retail 20000-ID boundary.")
    return available[:count]


def generate_city_tile(
    model_id: int,
    model_name: str,
    tile_size: float,
    random_source: random.Random,
) -> GeneratedModel:
    render = MeshBuilder()
    collision = MeshBuilder()
    road_color = (63, 67, 70, 255)
    sidewalk_color = (137, 133, 124, 255)
    grass_color = (76, 108, 71, 255)
    tile_half = tile_size * 0.5
    block_size = tile_size - 22.0

    render.add_box(0.0, 0.0, -0.8, tile_size, tile_size, 0.8, MATERIAL_ROAD, MATERIAL_ROAD, road_color)
    collision.add_box(0.0, 0.0, -0.8, tile_size, tile_size, 0.8, MATERIAL_ROAD, MATERIAL_ROAD, road_color)
    render.add_box(0.0, 0.0, 0.0, block_size, block_size, 0.22, MATERIAL_SIDEWALK, MATERIAL_SIDEWALK, sidewalk_color)
    collision.add_box(0.0, 0.0, 0.0, block_size, block_size, 0.22, MATERIAL_SIDEWALK, MATERIAL_SIDEWALK, sidewalk_color)

    building_count = 0
    style = random_source.choice(("lowrise", "stepped", "dense", "courtyard", "mixed"))
    lot_centers = ((-23.0, -23.0), (23.0, -23.0), (-23.0, 23.0), (23.0, 23.0))

    for lot_index, (lot_x, lot_y) in enumerate(lot_centers):
        if random_source.random() < (0.22 if style != "dense" else 0.08):
            add_park(render, collision, lot_x, lot_y, random_source, grass_color)
            continue
        building_count += 1
        width = random_source.uniform(27.0, 40.0)
        depth = random_source.uniform(27.0, 40.0)
        base_height = random_source.uniform(13.0, 27.0)
        if style in {"stepped", "dense"}:
            base_height += random_source.uniform(8.0, 22.0)
        color = random_source.choice(BUILDING_PALETTE)
        add_procedural_building(render, collision, lot_x, lot_y, width, depth, base_height, color, random_source)

    add_road_markings(render, tile_half, block_size)
    return GeneratedModel(model_id, model_name, render.mesh, collision.mesh, building_count, style)


def add_procedural_building(
    render: MeshBuilder,
    collision: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    base_height: float,
    color: tuple[int, int, int, int],
    random_source: random.Random,
) -> float:
    if base_height < 30.0:
        tier_count = 1 if random_source.random() < 0.82 else 2
    elif base_height < 62.0:
        tier_count = random_source.choice((1, 2, 2))
    else:
        tier_count = random_source.choice((2, 2, 3))
    current_z = 0.22
    current_width = width
    current_depth = depth
    roof_base_width = width
    roof_base_depth = depth
    tier_weights = [random_source.uniform(0.82, 1.18) for _ in range(tier_count)]
    tier_weight_total = sum(tier_weights)
    for tier_index in range(tier_count):
        tier_height = base_height * tier_weights[tier_index] / tier_weight_total
        render.add_box(
            center_x,
            center_y,
            current_z,
            current_width,
            current_depth,
            tier_height,
            MATERIAL_BUILDING,
            MATERIAL_ROOF,
            color,
        )
        collision.add_box(
            center_x,
            center_y,
            current_z,
            current_width,
            current_depth,
            tier_height,
            MATERIAL_BUILDING,
            MATERIAL_ROOF,
            color,
        )
        add_window_bands(render, center_x, center_y, current_z, current_width, current_depth, tier_height, random_source)
        current_z += tier_height
        roof_base_width = current_width
        roof_base_depth = current_depth
        current_width *= random_source.uniform(0.68, 0.88)
        current_depth *= random_source.uniform(0.68, 0.88)

    add_ground_floor_details(render, center_x, center_y, width, depth, color, random_source)

    if base_height < 30.0 and random_source.random() < 0.48:
        add_gable_roof(render, center_x, center_y, current_z, roof_base_width, roof_base_depth, color, random_source)
    elif random_source.random() < 0.72:
        roof_width = min(8.0, current_width * 0.45)
        roof_depth = min(9.0, current_depth * 0.45)
        roof_height = random_source.uniform(2.5, 6.0)
        roof_color = shade_color(color, 0.62)
        render.add_box(center_x, center_y, current_z, roof_width, roof_depth, roof_height, MATERIAL_TRIM, MATERIAL_ROOF, roof_color)
        collision.add_box(center_x, center_y, current_z, roof_width, roof_depth, roof_height, MATERIAL_TRIM, MATERIAL_ROOF, roof_color)
    return current_z


def add_ground_floor_details(
    render: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    color: tuple[int, int, int, int],
    random_source: random.Random,
) -> None:
    front_y = center_y - depth * 0.5 - 0.03
    door_width = min(3.2, max(1.8, width * 0.10))
    door_height = random_source.uniform(2.7, 3.5)
    door_x = center_x + random_source.uniform(-width * 0.22, width * 0.22)
    render.add_quad(
        (
            (door_x - door_width * 0.5, front_y, 0.24),
            (door_x + door_width * 0.5, front_y, 0.24),
            (door_x + door_width * 0.5, front_y, 0.24 + door_height),
            (door_x - door_width * 0.5, front_y, 0.24 + door_height),
        ),
        MATERIAL_TRIM,
        shade_color(color, 0.48),
    )
    if width >= 18.0 and random_source.random() < 0.55:
        awning_width = min(width * 0.62, random_source.uniform(8.0, 16.0))
        render.add_box(
            center_x,
            front_y - 0.65,
            3.25,
            awning_width,
            1.35,
            0.22,
            MATERIAL_TRIM,
            MATERIAL_TRIM,
            random_source.choice(((116, 52, 45, 255), (52, 79, 96, 255), (109, 93, 56, 255))),
        )


def add_gable_roof(
    render: MeshBuilder,
    center_x: float,
    center_y: float,
    minimum_z: float,
    width: float,
    depth: float,
    color: tuple[int, int, int, int],
    random_source: random.Random,
) -> None:
    half_width = width * 0.5
    half_depth = depth * 0.5
    ridge_height = random_source.uniform(2.2, 4.8)
    x0 = center_x - half_width
    x1 = center_x + half_width
    y0 = center_y - half_depth
    y1 = center_y + half_depth
    ridge_x = center_x
    ridge_z = minimum_z + ridge_height
    roof_color = shade_color(color, 0.70)
    render.add_quad(
        ((x0, y0, minimum_z), (ridge_x, y0, ridge_z), (ridge_x, y1, ridge_z), (x0, y1, minimum_z)),
        MATERIAL_ROOF,
        roof_color,
        ((0.0, 0.0), (1.0, 0.0), (1.0, depth / 5.0), (0.0, depth / 5.0)),
    )
    render.add_quad(
        ((ridge_x, y0, ridge_z), (x1, y0, minimum_z), (x1, y1, minimum_z), (ridge_x, y1, ridge_z)),
        MATERIAL_ROOF,
        shade_color(roof_color, 0.86),
        ((0.0, 0.0), (1.0, 0.0), (1.0, depth / 5.0), (0.0, depth / 5.0)),
    )
    render.add_triangle(
        ((x0, y0, minimum_z), (x1, y0, minimum_z), (ridge_x, y0, ridge_z)),
        MATERIAL_BUILDING,
        color,
    )
    render.add_triangle(
        ((x1, y1, minimum_z), (x0, y1, minimum_z), (ridge_x, y1, ridge_z)),
        MATERIAL_BUILDING,
        shade_color(color, 0.86),
    )


def add_window_bands(
    render: MeshBuilder,
    center_x: float,
    center_y: float,
    minimum_z: float,
    width: float,
    depth: float,
    height: float,
    random_source: random.Random,
) -> None:
    # A streamed GTA SA atomic uses 16-bit vertex indices. Represent tall
    # facades with four evenly spaced window bands instead of one quad set per
    # physical floor; the texture and pilasters carry the repeated rhythm.
    physical_floors = max(1, int(height // 4.2))
    floors = min(4, physical_floors)
    columns_x = max(2, min(4, int(width // 7.0)))
    x0 = center_x - width * 0.5
    y0 = center_y - depth * 0.5
    y1 = center_y + depth * 0.5

    for floor in range(floors):
        physical_floor = round(floor * max(0, physical_floors - 1) / max(1, floors - 1))
        window_z0 = minimum_z + 1.5 + physical_floor * (height / physical_floors)
        window_z1 = min(minimum_z + height - 0.8, window_z0 + 1.45)
        if window_z1 <= window_z0:
            continue
        for column in range(columns_x):
            center = x0 + (column + 0.5) * width / columns_x
            half_window = min(1.35, width / columns_x * 0.30)
            window_color = choose_window_color(random_source)
            render.add_quad(
                ((center - half_window, y0 - 0.025, window_z0), (center + half_window, y0 - 0.025, window_z0), (center + half_window, y0 - 0.025, window_z1), (center - half_window, y0 - 0.025, window_z1)),
                MATERIAL_WINDOW,
                window_color,
            )
            if floor % 2 == 0:
                render.add_quad(
                    ((center + half_window, y1 + 0.025, window_z0), (center - half_window, y1 + 0.025, window_z0), (center - half_window, y1 + 0.025, window_z1), (center + half_window, y1 + 0.025, window_z1)),
                    MATERIAL_WINDOW,
                    choose_window_color(random_source),
                )


def choose_window_color(random_source: random.Random) -> tuple[int, int, int, int]:
    if random_source.random() < 0.20:
        return 216, 184, 104, 255
    return random_source.choice(((57, 76, 87, 255), (68, 91, 103, 255), (78, 82, 91, 255)))


def add_park(
    render: MeshBuilder,
    collision: MeshBuilder,
    center_x: float,
    center_y: float,
    random_source: random.Random,
    grass_color: tuple[int, int, int, int],
) -> None:
    render.add_box(center_x, center_y, 0.22, 37.0, 37.0, 0.12, MATERIAL_GRASS, MATERIAL_GRASS, grass_color)
    collision.add_box(center_x, center_y, 0.22, 37.0, 37.0, 0.12, MATERIAL_GRASS, MATERIAL_GRASS, grass_color)
    for offset_x, offset_y in ((-10.0, -9.0), (10.0, 8.0), (-8.0, 10.0)):
        if random_source.random() < 0.25:
            continue
        trunk_color = (91, 67, 44, 255)
        leaf_color = random_source.choice(((62, 94, 62, 255), (72, 105, 68, 255), (58, 87, 64, 255)))
        render.add_box(center_x + offset_x, center_y + offset_y, 0.34, 1.2, 1.2, 5.0, MATERIAL_TRIM, MATERIAL_TRIM, trunk_color)
        render.add_crossed_billboard(
            center_x + offset_x,
            center_y + offset_y,
            3.8,
            10.5,
            11.0,
            MATERIAL_VEGETATION,
            leaf_color,
        )
        collision.add_box(center_x + offset_x, center_y + offset_y, 0.34, 1.2, 1.2, 5.0, MATERIAL_TRIM, MATERIAL_TRIM, trunk_color)


def add_road_markings(render: MeshBuilder, tile_half: float, block_size: float) -> None:
    line_color = (209, 181, 83, 255)
    block_half = block_size * 0.5
    z = 0.015
    render.add_quad(((-tile_half, -block_half - 5.5, z), (tile_half, -block_half - 5.5, z), (tile_half, -block_half - 5.15, z), (-tile_half, -block_half - 5.15, z)), MATERIAL_TRIM, line_color)
    render.add_quad(((-tile_half, block_half + 5.15, z), (tile_half, block_half + 5.15, z), (tile_half, block_half + 5.5, z), (-tile_half, block_half + 5.5, z)), MATERIAL_TRIM, line_color)
    render.add_quad(((-block_half - 5.5, tile_half, z), (-block_half - 5.5, -tile_half, z), (-block_half - 5.15, -tile_half, z), (-block_half - 5.15, tile_half, z)), MATERIAL_TRIM, line_color)
    render.add_quad(((block_half + 5.15, tile_half, z), (block_half + 5.15, -tile_half, z), (block_half + 5.5, -tile_half, z), (block_half + 5.5, tile_half, z)), MATERIAL_TRIM, line_color)


def write_custom_mod_package(staging_directory: Path, world: CustomGeneratedWorld) -> Path:
    mod_directory = staging_directory / "SeedAndreas"
    data_directory = mod_directory / "data"
    map_directory = data_directory / "maps" / "seedandreas"
    model_directory = mod_directory / "models" / "gta3.img"
    map_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    for model in world.models:
        dff_path = model_directory / f"{model.model_name}.dff"
        dff_path.write_bytes(write_dff(model.model_name, model.render_mesh))

    col_path = map_directory / "seedandreas.col"
    col_path.write_bytes(write_col3_archive(world.models))

    ipl_lines = ["inst"]
    for placement in world.placements:
        ipl_lines.append(
            f"{placement.model_id}, {placement.model_name}, 0, "
            f"{placement.x:.6f}, {placement.y:.6f}, {placement.z:.6f}, "
            f"{placement.qx:.8f}, {placement.qy:.8f}, {placement.qz:.8f}, {placement.qw:.8f}, -1"
        )
    ipl_lines.append("end")
    (map_directory / "seedandreas.ipl").write_text("\n".join(ipl_lines) + "\n", encoding="cp1252")

    source_gta_dat = Path(world.manifest.game_directory) / "data" / "gta.dat"
    (data_directory / "gta.dat").write_text(
        build_augmented_gta_dat(
            source_gta_dat.read_text(encoding="cp1252", errors="replace"),
            colfile_line="COLFILE 0 DATA\\MAPS\\SEEDANDREAS\\SEEDANDREAS.COL",
            ipl_line="IPL DATA\\MAPS\\SEEDANDREAS\\SEEDANDREAS.IPL",
        ),
        encoding="cp1252",
    )
    (mod_directory / "seed_andreas_install.json").write_text(json.dumps(asdict(world.manifest), indent=2), encoding="utf-8")
    (mod_directory / "README.txt").write_text(
        "Seed Andreas custom-mesh prototype\n\n"
        f"Seed: {world.manifest.seed}\n"
        f"Custom DFF models: {world.manifest.custom_models}\n"
        f"Procedural buildings: {world.manifest.custom_buildings}\n"
        f"Test city center: {world.manifest.target_center}\n\n"
        "Every visible city tile in this package was generated as new RenderWare geometry. "
        "No retail object placement was shuffled or copied.\n",
        encoding="utf-8",
    )
    write_combined_obj(mod_directory / "seed_andreas_debug.obj", world)
    return mod_directory


def write_dff(
    model_name: str,
    mesh: MeshData,
    material_texture_names: tuple[str, ...] = MATERIAL_TEXTURE_NAMES,
) -> bytes:
    if not mesh.vertices or not mesh.triangles:
        raise ValueError("A DFF mesh requires vertices and triangles.")
    if not (len(mesh.vertices) == len(mesh.normals) == len(mesh.colors) == len(mesh.uvs)):
        raise ValueError("DFF vertices, normals, prelight colors and UV coordinates must have matching counts.")
    validate_mesh_indices(mesh, "DFF")
    if any(not math.isfinite(component) for stream in (mesh.vertices, mesh.normals, mesh.uvs)
           for values in stream for component in values):
        raise ValueError("DFF positions, normals and UV coordinates must be finite.")

    geometry_struct = bytearray()
    flags = (
        GEOMETRY_POSITIONS
        | GEOMETRY_TEXTURED
        | GEOMETRY_PRELIT
        | GEOMETRY_NORMALS
        | GEOMETRY_LIGHT
        | GEOMETRY_MODULATE_MATERIAL
    )
    geometry_struct += struct.pack("<IIII", flags, len(mesh.triangles), len(mesh.vertices), 1)
    for color in mesh.colors:
        geometry_struct += struct.pack("<4B", *tone_map_vertex_color(color))
    for uv in mesh.uvs:
        geometry_struct += struct.pack("<2f", *uv)
    for first, second, third, material in mesh.triangles:
        geometry_struct += struct.pack("<4H", second, first, material, third)
    center, radius = calculate_bounding_sphere(mesh.vertices)
    geometry_struct += struct.pack("<4fII", center[0], center[1], center[2], radius, 1, 1)
    for vertex in mesh.vertices:
        geometry_struct += struct.pack("<3f", *vertex)
    for normal in mesh.normals:
        geometry_struct += struct.pack("<3f", *normal)

    material_count = max(triangle[3] for triangle in mesh.triangles) + 1
    if len(material_texture_names) < material_count:
        raise ValueError(
            f"DFF needs {material_count} material texture names, but only {len(material_texture_names)} were supplied."
        )
    material_list_struct = struct.pack("<I", material_count) + b"".join(struct.pack("<i", -1) for _ in range(material_count))
    material_list_payload = chunk(RW_STRUCT, material_list_struct)
    for material_index in range(material_count):
        material_color = material_base_color(material_index)
        material_struct = struct.pack("<I4BII3f", 0, *material_color, 1, 1, 1.0, 0.0, 1.0)
        material_payload = (
            chunk(RW_STRUCT, material_struct)
            + write_texture_reference(material_texture_names[material_index])
            + chunk(RW_EXTENSION, b"")
        )
        material_list_payload += chunk(RW_MATERIAL, material_payload)

    meshes: dict[int, list[int]] = {material_index: [] for material_index in range(material_count)}
    for first, second, third, material in mesh.triangles:
        meshes[material].extend((first, second, third))
    meshes = {material: indices for material, indices in meshes.items() if indices}
    total_indices = sum(len(indices) for indices in meshes.values())
    bin_mesh = bytearray(struct.pack("<III", 0, len(meshes), total_indices))
    for material, indices in meshes.items():
        bin_mesh += struct.pack("<II", len(indices), material)
        bin_mesh += struct.pack(f"<{len(indices)}I", *indices)

    geometry_payload = (
        chunk(RW_STRUCT, bytes(geometry_struct))
        + chunk(RW_MATERIAL_LIST, material_list_payload)
        + chunk(RW_EXTENSION, chunk(RW_BIN_MESH, bytes(bin_mesh)))
    )
    geometry = chunk(RW_GEOMETRY, geometry_payload)
    geometry_list = chunk(RW_GEOMETRY_LIST, chunk(RW_STRUCT, struct.pack("<I", 1)) + geometry)

    identity_frame = struct.pack(
        "<12fiI",
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        -1,
        0,
    )
    frame_extension = chunk(RW_EXTENSION, chunk(RW_FRAME_NAME, model_name.encode("ascii")))
    frame_list = chunk(RW_FRAME_LIST, chunk(RW_STRUCT, struct.pack("<I", 1) + identity_frame) + frame_extension)
    atomic = chunk(RW_ATOMIC, chunk(RW_STRUCT, struct.pack("<4I", 0, 0, 5, 0)) + chunk(RW_EXTENSION, b""))
    clump_payload = chunk(RW_STRUCT, struct.pack("<3I", 1, 0, 0)) + frame_list + geometry_list + atomic + chunk(RW_EXTENSION, b"")
    return chunk(RW_CLUMP, clump_payload)


def material_base_color(material_index: int) -> tuple[int, int, int, int]:
    colors = (
        (205, 205, 205, 255),
        (198, 198, 198, 255),
        (210, 210, 210, 255),
        (190, 190, 190, 255),
        (196, 196, 196, 255),
        (190, 198, 190, 255),
        (202, 198, 190, 255),
        (190, 205, 184, 255),
    )
    return colors[material_index % len(colors)]


def tone_map_vertex_color(color: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (
        max(0, min(255, round(color[0] * 0.72))),
        max(0, min(255, round(color[1] * 0.72))),
        max(0, min(255, round(color[2] * 0.72))),
        color[3],
    )


def write_texture_reference(texture_name: str) -> bytes:
    filter_and_addressing = chunk(RW_STRUCT, struct.pack("<I", 0x1106))
    name = texture_name.encode("ascii") + b"\0"
    name += b"\0" * ((-len(name)) % 4)
    mask = b"\0\0\0\0"
    return chunk(
        RW_TEXTURE,
        filter_and_addressing
        + chunk(RW_STRING, name)
        + chunk(RW_STRING, mask)
        + chunk(RW_EXTENSION, b""),
    )


def chunk(chunk_type: int, payload: bytes) -> bytes:
    return struct.pack("<III", chunk_type, len(payload), RW_LIBRARY_ID) + payload


def calculate_bounding_sphere(vertices: list[tuple[float, float, float]]) -> tuple[tuple[float, float, float], float]:
    minimum = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
    maximum = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
    center = tuple((minimum[axis] + maximum[axis]) * 0.5 for axis in range(3))
    radius = max(math.dist(center, vertex) for vertex in vertices)
    return center, radius


def write_col3_archive(models: list[GeneratedModel]) -> bytes:
    return b"".join(write_col3_model(model) for model in models)


def write_col3_model(model: GeneratedModel) -> bytes:
    mesh = model.collision_mesh
    validate_mesh_indices(mesh, f"COL3 model {model.model_name}")
    if len(mesh.triangles) > 65535:
        raise ValueError(f"Collision model {model.model_name} exceeds the COL3 16-bit face count limit.")
    compressed_vertices = []
    for vertex in mesh.vertices:
        if any(not math.isfinite(component) for component in vertex):
            raise ValueError(f"Collision vertices in {model.model_name} must be finite.")
        compressed = tuple(round(component * 128.0) for component in vertex)
        if any(component < -32768 or component > 32767 for component in compressed):
            raise ValueError(f"Collision vertex in {model.model_name} exceeds COL3 compressed coordinate range.")
        compressed_vertices.append(compressed)
    vertex_data = b"".join(struct.pack("<3h", *vertex) for vertex in compressed_vertices)
    face_data = b"".join(
        struct.pack("<3HBB", first, second, third, collision_material(material), 0)
        for first, second, third, material in mesh.triangles
    )
    minimum = tuple(min(vertex[axis] for vertex in mesh.vertices) for axis in range(3))
    maximum = tuple(max(vertex[axis] for vertex in mesh.vertices) for axis in range(3))
    center, radius = calculate_bounding_sphere(mesh.vertices)
    bounds = struct.pack("<10f", *minimum, *maximum, *center, radius)

    offset_base = 116
    spheres_offset = offset_base
    boxes_offset = offset_base
    vertices_offset = offset_base
    faces_offset = offset_base + len(vertex_data)
    flags = 2
    collision_header = struct.pack(
        "<HHHBxIIIIIII",
        0,
        0,
        len(mesh.triangles),
        0,
        flags,
        spheres_offset,
        boxes_offset,
        0,
        vertices_offset,
        faces_offset,
        0,
    )
    shadow_header = struct.pack("<III", 0, 0, 0)
    body = bounds + collision_header + shadow_header + vertex_data + face_data
    model_name = model.model_name.encode("ascii")[:21]
    header = struct.pack("<4sI22sH", b"COL3", len(body) + 24, model_name, model.model_id)
    return header + body


def validate_mesh_indices(mesh: MeshData, format_name: str) -> None:
    if not mesh.vertices or not mesh.triangles:
        raise ValueError(f"{format_name} requires vertices and triangles.")
    if len(mesh.vertices) > 65535:
        raise ValueError(f"{format_name} exceeds the 16-bit vertex index limit.")
    for triangle_index, (first, second, third, material) in enumerate(mesh.triangles):
        if any(index < 0 or index >= len(mesh.vertices) for index in (first, second, third)):
            raise ValueError(f"{format_name} triangle {triangle_index} references a missing vertex.")
        if material < 0 or material > 65535:
            raise ValueError(f"{format_name} triangle {triangle_index} has an invalid material index.")


def collision_material(render_material: int) -> int:
    if render_material == MATERIAL_ROAD:
        return 1
    if render_material == MATERIAL_SIDEWALK:
        return 4
    if render_material == MATERIAL_VEGETATION:
        return 9
    return 1


def validate_dff(data: bytes) -> list[str]:
    errors: list[str] = []
    if len(data) < 12:
        return ["file is shorter than one RenderWare chunk header"]
    root_type, root_size, root_version = struct.unpack_from("<III", data, 0)
    if root_type != RW_CLUMP:
        errors.append(f"root chunk is {root_type:#x}, expected clump")
    if root_size != len(data) - 12:
        errors.append("root chunk size does not match file length")
    if root_version != RW_LIBRARY_ID:
        errors.append(f"library ID is {root_version:#x}, expected GTA SA PC {RW_LIBRARY_ID:#x}")
    root_children = read_child_chunks(data, 12, len(data), errors)
    child_types = [item[0] for item in root_children]
    for required in (RW_STRUCT, RW_FRAME_LIST, RW_GEOMETRY_LIST, RW_ATOMIC, RW_EXTENSION):
        if required not in child_types:
            errors.append(f"clump is missing child chunk {required:#x}")
    geometry_lists = [item for item in root_children if item[0] == RW_GEOMETRY_LIST]
    if not geometry_lists:
        return errors
    geometry_list = geometry_lists[0]
    geometry_children = read_child_chunks(data, geometry_list[3], geometry_list[4], errors)
    geometries = [item for item in geometry_children if item[0] == RW_GEOMETRY]
    if len(geometries) != 1:
        errors.append(f"geometry list contains {len(geometries)} geometries, expected one")
        return errors
    geometry = geometries[0]
    children = read_child_chunks(data, geometry[3], geometry[4], errors)
    struct_chunks = [item for item in children if item[0] == RW_STRUCT]
    vertex_count = 0
    triangle_materials: list[int] = []
    if not struct_chunks:
        errors.append("geometry has no struct")
    else:
        struct_chunk = struct_chunks[0]
        if struct_chunk[1] < 16:
            errors.append("geometry struct is too small")
        else:
            flags, triangles, vertices, morph_targets = struct.unpack_from("<IIII", data, struct_chunk[3])
            vertex_count = vertices
            required_flags = GEOMETRY_POSITIONS | GEOMETRY_TEXTURED | GEOMETRY_PRELIT | GEOMETRY_NORMALS
            if flags & required_flags != required_flags:
                errors.append("geometry is missing positions, UVs, prelights or normals")
            if triangles == 0 or vertices == 0 or morph_targets != 1:
                errors.append("geometry counts are invalid")
            expected_size = 40 + vertices * 36 + triangles * 8
            if struct_chunk[1] != expected_size:
                errors.append("geometry vertex/triangle streams do not match their counts")
            else:
                triangle_offset = struct_chunk[3] + 16 + vertices * 12
                for triangle_index in range(triangles):
                    second, first, material, third = struct.unpack_from("<4H", data, triangle_offset + triangle_index * 8)
                    if max(first, second, third) >= vertices:
                        errors.append(f"geometry triangle {triangle_index} references a missing vertex")
                        break
                    triangle_materials.append(material)

    material_lists = [item for item in children if item[0] == RW_MATERIAL_LIST]
    material_count = 0
    if not material_lists:
        errors.append("geometry has no material list")
    else:
        material_children = read_child_chunks(data, material_lists[0][3], material_lists[0][4], errors)
        material_structs = [item for item in material_children if item[0] == RW_STRUCT]
        if not material_structs or material_structs[0][1] < 4:
            errors.append("material list has no count")
        else:
            material_count = struct.unpack_from("<I", data, material_structs[0][3])[0]
            if material_structs[0][1] != 4 + material_count * 4:
                errors.append("material list index table does not match its count")
            if sum(item[0] == RW_MATERIAL for item in material_children) != material_count:
                errors.append("material list count does not match its material chunks")
            if any(material >= material_count for material in triangle_materials):
                errors.append("geometry triangle references a missing material")
    extensions = [item for item in children if item[0] == RW_EXTENSION]
    if not extensions:
        errors.append("geometry has no extension")
    else:
        extension_children = read_child_chunks(data, extensions[0][3], extensions[0][4], errors)
        bin_meshes = [item for item in extension_children if item[0] == RW_BIN_MESH]
        if not bin_meshes:
            errors.append("geometry extension has no Bin Mesh PLG")
        else:
            bin_mesh = bin_meshes[0]
            if bin_mesh[1] < 12:
                errors.append("Bin Mesh PLG has a truncated header")
            else:
                mesh_flags, mesh_count, total_indices = struct.unpack_from("<III", data, bin_mesh[3])
                position = bin_mesh[3] + 12
                counted_indices = 0
                if mesh_flags != 0:
                    errors.append("generated Bin Mesh PLG is not a triangle list")
                for mesh_index in range(mesh_count):
                    if position + 8 > bin_mesh[4]:
                        errors.append("Bin Mesh PLG has a truncated mesh table")
                        break
                    index_count, material = struct.unpack_from("<II", data, position)
                    position += 8
                    if position + index_count * 4 > bin_mesh[4]:
                        errors.append("Bin Mesh PLG has a truncated index stream")
                        break
                    if material >= material_count:
                        errors.append(f"Bin Mesh {mesh_index} references a missing material")
                    if index_count % 3:
                        errors.append(f"Bin Mesh {mesh_index} index count is not divisible by three")
                    indices = struct.unpack_from(f"<{index_count}I", data, position)
                    if any(index >= vertex_count for index in indices):
                        errors.append(f"Bin Mesh {mesh_index} references a missing vertex")
                    counted_indices += index_count
                    position += index_count * 4
                if counted_indices != total_indices or position != bin_mesh[4]:
                    errors.append("Bin Mesh PLG sizes do not match its declared index count")
    return errors


def read_child_chunks(
    data: bytes,
    start: int,
    end: int,
    errors: list[str],
) -> list[tuple[int, int, int, int, int]]:
    chunks: list[tuple[int, int, int, int, int]] = []
    position = start
    while position < end:
        if position + 12 > end:
            errors.append("child chunk header crosses its parent boundary")
            break
        chunk_type, size, version = struct.unpack_from("<III", data, position)
        payload_start = position + 12
        payload_end = payload_start + size
        if payload_end > end:
            errors.append(f"child chunk {chunk_type:#x} crosses its parent boundary")
            break
        chunks.append((chunk_type, size, version, payload_start, payload_end))
        position = payload_end
    if position != end:
        errors.append("child chunks do not exactly consume their parent")
    return chunks


def validate_col3_archive(data: bytes, expected_models: int) -> list[str]:
    errors: list[str] = []
    position = 0
    model_count = 0
    while position < len(data):
        if position + 32 > len(data):
            errors.append("truncated COL3 model header")
            break
        magic, file_size, model_name, model_id = struct.unpack_from("<4sI22sH", data, position)
        if magic != b"COL3":
            errors.append(f"invalid collision magic at {position}: {magic!r}")
            break
        model_end = position + file_size + 8
        if model_end > len(data) or file_size < 112:
            errors.append(f"invalid COL3 size for model {model_id}")
            break
        if not model_name.split(b"\0", 1)[0]:
            errors.append(f"COL3 model {model_id} has no name")
        face_count = struct.unpack_from("<H", data, position + 32 + 40 + 4)[0]
        if face_count == 0:
            errors.append(f"COL3 model {model_id} has no collision faces")
        model_count += 1
        position = model_end
    if model_count != expected_models:
        errors.append(f"COL3 archive contains {model_count} models, expected {expected_models}")
    if position != len(data):
        errors.append("COL3 archive contains trailing or truncated data")
    return errors


def write_combined_obj(path: Path, world: CustomGeneratedWorld) -> None:
    lines = ["# Seed Andreas generated custom geometry debug export"]
    vertex_offset = 1
    for model, placement in zip(world.models, world.placements):
        lines.append(f"o {model.model_name}")
        for x, y, z in model.render_mesh.vertices:
            lines.append(f"v {x + placement.x:.6f} {y + placement.y:.6f} {z + placement.z:.6f}")
        for first, second, third, material in model.render_mesh.triangles:
            lines.append(f"f {first + vertex_offset} {second + vertex_offset} {third + vertex_offset}")
        vertex_offset += len(model.render_mesh.vertices)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
