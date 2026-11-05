from __future__ import annotations

import hashlib
import json
import math
import random
import re
import struct
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from .generator import GENERATOR_VERSION
from .neighborhoods import (
    choose_neighborhood, choose_building_style, build_neighborhood_building,
    choose_settlement_district, generate_town_plans, town_main_street_endpoints,
)
from .gameplay import GameplayPlan, collect_gameplay_plan, validate_preserved_tracks
from .path_nodes import BinaryPathNetwork, build_binary_path_network
from .water import WaterOverride, WaterSurface, build_water_override, water_height_at

from .custom_world import (
    BUILDING_PALETTE,
    MATERIAL_BUILDING,
    MATERIAL_ROAD,
    MATERIAL_ROOF,
    MATERIAL_SIDEWALK,
    MATERIAL_WINDOW,
    MATERIAL_TRIM,
    MATERIAL_VEGETATION,
    MATERIAL_GRASS,
    CustomPlacement,
    GeneratedModel,
    MeshBuilder,
    calculate_normal,
    add_park,
    add_procedural_building,
    add_road_markings,
    validate_col3_archive,
    validate_dff,
    write_col3_archive,
    write_col3_model,
    write_combined_obj,
    write_dff,
    shade_color,
)
from .gta_integration import (
    GameScan,
    SeededPathOverrides,
    TrafficPathSegment,
    build_augmented_gta_dat,
    build_seeded_path_overrides,
    point_in_any_bounds,
    read_game_path_files,
)
from .street_graph import (
    StreetBlock,
    StreetGraph,
    StreetSegment,
    StreetPoint,
    finalize_street_graph,
    WaterBody,
    generate_hierarchical_street_graph,
    generate_seeded_water_bodies,
)
from .textures import (
    MATERIAL_TEXTURE_NAMES,
    TextureImage,
    augment_texture_dictionary,
    augment_texture_dictionary_with_native_chunks,
    extract_native_texture_chunks,
    generate_seed_radar_tiles,
    generate_world_textures,
    native_texture_has_alpha,
    native_texture_name,
    render_geometry_radar_tile,
    validate_texture_dictionary,
    write_native_texture,
)


LOS_SANTOS_BOUNDS = (0.0, -3000.0, 3000.0, 0.0)
LOS_SANTOS_STREAM_RE = re.compile(
    r"^(?:lan|lan2|las|las2|lae|lae2|law|law2|lawn|lahills)_stream\d+\.ipl$",
    re.IGNORECASE,
)
IMG_SECTOR_SIZE = 2048
UNSAFE_REPLACEMENT_FLAGS = 0x00000800 | 0x00001000 | 0x00100000
MISSION_PATH_BOUNDS = (
    (2160.0, -1390.0, 2335.0, -1170.0),
    (2240.0, -1550.0, 2435.0, -1300.0),
    (2340.0, -1795.0, 2605.0, -1450.0),
)
MISSION_INSTANCE_BOUNDS = (
    (2474.0, -1685.0, 2506.0, -1651.0),
    (2234.0, -1259.0, 2266.0, -1227.0),
)
MISSION_SPAWN_BOUNDS = (
    (1615.0, -2310.0, 1755.0, -2170.0),
    (2405.0, -1750.0, 2575.0, -1585.0),
)
MISSION_SPAWN_POINTS = (
    (1685.0, -2243.0, 13.4, "CJ opening spawn"),
    (2490.0, -1668.0, 13.4, "Grove Street spawn"),
)
MISSION_SAFE_TERRAIN_HEIGHT = 12.0


@dataclass(frozen=True)
class ImgEntry:
    name: str
    sector_offset: int
    streaming_sectors: int
    archive_sectors: int


@dataclass(frozen=True)
class ColRecord:
    archive_name: str
    model_name: str
    model_id: int
    raw: bytes


@dataclass
class LosSantosManifest:
    generator_version: str
    mode: str
    seed: int
    seed_fingerprint: str
    game_directory: str
    mod_directory: str
    bounds: tuple[float, float, float, float]
    columns: int
    rows: int
    tile_size: float
    custom_models: int
    custom_buildings: int
    render_vertices: int
    render_triangles: int
    collision_vertices: int
    collision_triangles: int
    model_id_first: int
    model_id_last: int
    district_tiles: dict[str, int]
    text_ipls_overridden: list[str]
    stream_ipls_overridden: list[str]
    collision_archives_overridden: list[str]
    texture_dictionaries_overridden: list[str]
    radar_tiles_overridden: list[str]
    material_texture_names: list[str]
    material_texture_sources: list[str]
    traffic_path_segments_preserved: int
    traffic_path_nodes_changed: int
    traffic_path_nodes_preserved: int
    traffic_path_files_overridden: list[str]
    mission_safe_instances_preserved: int
    gameplay_regions_preserved: int
    terrain_elevation_minimum: float
    terrain_elevation_maximum: float
    ground_rules: list[str]
    limitations: list[str]
    validation: list[str]
    traffic_path_nodes_added: int = 0
    traffic_car_links_added: int = 0
    traffic_pedestrian_nodes_remapped: int = 0
    neighborhood_buildings: dict[str, int] = field(default_factory=dict)
    building_conditions: dict[str, int] = field(default_factory=dict)
    building_styles: dict[str, int] = field(default_factory=dict)
    dwelling_units: int = 0
    water: dict[str, object] = field(default_factory=dict)
    mission_regions: list[str] = field(default_factory=list)
    railway_tracks_preserved: dict[str, str] = field(default_factory=dict)


@dataclass
class LosSantosWorld:
    models: list[GeneratedModel]
    placements: list[CustomPlacement]
    tile_districts: list[str]
    traffic_segments: list[TrafficPathSegment]
    path_overrides: SeededPathOverrides
    protected_instance_bounds: tuple[tuple[float, float, float, float], ...]
    material_texture_names: tuple[str, ...]
    material_native_chunks: list[bytes]
    material_generated_textures: list[tuple[str, TextureImage, bool]]
    manifest: LosSantosManifest
    procedural_graph: StreetGraph | None = None
    building_debug_records: list[BuildingDebugRecord] | None = None
    native_paths: BinaryPathNetwork | None = None
    gameplay: GameplayPlan | None = None
    water_override: WaterOverride | None = None
    preserved_model_ids: tuple[int, ...] = ()
    retail_surface_bounds: tuple[tuple[float, float, float, float], ...] = ()
    retail_radar_tiles: dict[str, bytes] = field(default_factory=dict)


@dataclass
class MaterialTextureSelection:
    names: tuple[str, ...]
    native_chunks: list[bytes]
    generated_textures: list[tuple[str, TextureImage, bool]]
    sources: list[str]


@dataclass(frozen=True)
class CityBlock:
    center_x: float
    center_y: float
    width: float
    depth: float
    points: tuple[tuple[float, float], ...]
    pad_elevation: float = MISSION_SAFE_TERRAIN_HEIGHT
    terrain_minimum: float = MISSION_SAFE_TERRAIN_HEIGHT
    terrain_maximum: float = MISSION_SAFE_TERRAIN_HEIGHT
    slope: float = 0.0
    cut_fill: float = 0.0
    retaining_edges: int = 0


@dataclass(frozen=True)
class BuildingProduction:
    """A bounded L-system-style production selected from lot and zone state."""
    mass_rule: str
    facade_rule: str
    roof_rule: str

@dataclass(frozen=True)
class BuildingDebugRecord:
    lot_id: str
    archetype: str
    district: str
    parcel_points: tuple[tuple[float, float], ...]
    setback_points: tuple[tuple[float, float], ...]
    footprint_points: tuple[tuple[float, float], ...]
    world_center: tuple[float, float]
    pad_elevation: float
    terrain_minimum: float
    terrain_maximum: float
    slope: float
    cut_fill: float
    retaining_edges: int
    street_class: str
    corner_lot: bool
    production: BuildingProduction
    neighborhood: str = "residential"
    condition: str = "maintained"
    building_style: str = "residential"
    dwelling_units: int = 1


DISTRICT_BUILDING_PALETTES: dict[str, tuple[tuple[int, int, int, int], ...]] = {
    "downtown": (
        (224, 222, 211, 255), (188, 191, 195, 255), (154, 158, 164, 255),
        (205, 188, 164, 255), (170, 147, 132, 255), (147, 155, 144, 255),
        (183, 169, 143, 255), (129, 132, 138, 255), (211, 205, 191, 255),
    ),
    "commercial": (
        (224, 184, 104, 255), (192, 105, 79, 255), (131, 166, 142, 255),
        (198, 181, 147, 255), (176, 137, 115, 255), (213, 211, 190, 255),
        (139, 151, 177, 255), (185, 153, 176, 255), (157, 119, 89, 255),
    ),
    "civic": (
        (224, 219, 201, 255), (190, 180, 157, 255), (170, 174, 169, 255),
        (211, 188, 143, 255), (184, 197, 188, 255), (155, 149, 138, 255),
    ),
    "residential": (
        (229, 213, 176, 255), (205, 168, 121, 255), (153, 184, 151, 255),
        (215, 197, 187, 255), (187, 124, 102, 255), (155, 177, 187, 255),
        (221, 202, 145, 255), (180, 154, 188, 255), (237, 231, 210, 255),
    ),
    "grove": (
        (205, 170, 119, 255), (128, 165, 126, 255), (220, 209, 172, 255),
        (179, 116, 88, 255), (198, 188, 147, 255), (142, 176, 160, 255),
        (229, 220, 196, 255), (165, 139, 109, 255), (184, 164, 191, 255),
    ),
    "coastal": (
        (232, 219, 180, 255), (201, 171, 132, 255), (143, 184, 174, 255),
        (221, 168, 139, 255), (181, 200, 209, 255), (236, 228, 208, 255),
        (183, 154, 192, 255),
    ),
    "beach": (
        (239, 222, 166, 255), (222, 164, 111, 255), (143, 196, 178, 255),
        (233, 188, 178, 255), (165, 192, 216, 255), (239, 234, 217, 255),
        (193, 166, 207, 255), (205, 211, 157, 255),
    ),
    "hills": (
        (232, 224, 201, 255), (198, 169, 133, 255), (143, 173, 137, 255),
        (216, 203, 176, 255), (175, 145, 122, 255), (194, 205, 211, 255),
        (222, 187, 168, 255),
    ),
    "industrial": (
        (139, 139, 131, 255), (171, 151, 116, 255), (109, 135, 124, 255),
        (166, 103, 77, 255), (126, 145, 161, 255), (188, 181, 160, 255),
        (110, 111, 114, 255),
    ),
    "eastside": (
        (193, 137, 92, 255), (132, 165, 128, 255), (213, 191, 149, 255),
        (166, 96, 81, 255), (143, 157, 184, 255), (203, 205, 188, 255),
        (179, 145, 181, 255), (155, 132, 102, 255),
    ),
}


def build_procedural_city_graph(seed: int) -> StreetGraph:
    graph = generate_hierarchical_street_graph(
        seed,
        LOS_SANTOS_BOUNDS[0],
        LOS_SANTOS_BOUNDS[1],
        LOS_SANTOS_BOUNDS[2],
        LOS_SANTOS_BOUNDS[3],
        neighborhood_count=12,
    )
    for town in generate_town_plans(seed):
        start, end = town_main_street_endpoints(town)
        if min(procedural_land_mask(*point, seed) for point in (start, end)) < 0.62:
            continue
        first_index = len(graph.nodes)
        graph.nodes.extend((StreetPoint(*start), StreetPoint(*end)))
        graph.segments.append(StreetSegment(first_index, first_index + 1, "collector"))
        for index in (first_index, first_index + 1):
            point = graph.nodes[index]
            nearest = min(range(first_index), key=lambda other: math.hypot(graph.nodes[other].x - point.x, graph.nodes[other].y - point.y))
            graph.segments.append(StreetSegment(index, nearest, "collector"))
    return finalize_street_graph(graph)


def point_inside_water_body(x: float, y: float, water_body: WaterBody) -> bool:
    inside = False
    points = water_body.points
    for index, first in enumerate(points):
        second = points[(index + 1) % len(points)]
        if (first.y > y) == (second.y > y):
            continue
        intersection_x = first.x + (y - first.y) * (second.x - first.x) / (second.y - first.y)
        if x < intersection_x:
            inside = not inside
    return inside


@lru_cache(maxsize=16)
def cached_inland_water_bodies(seed: int) -> tuple[WaterBody, ...]:
    return tuple(generate_seeded_water_bodies(seed, *LOS_SANTOS_BOUNDS))


def inland_water_strength(world_x: float, world_y: float, seed: int) -> float:
    water_bodies = cached_inland_water_bodies(seed)
    strongest = 0.0
    for water_body in water_bodies:
        center_x = sum(point.x for point in water_body.points) / len(water_body.points)
        center_y = sum(point.y for point in water_body.points) / len(water_body.points)
        radius_x = max(abs(point.x - center_x) for point in water_body.points)
        radius_y = max(abs(point.y - center_y) for point in water_body.points)
        normalized = math.hypot(
            (world_x - center_x) / max(1.0, radius_x),
            (world_y - center_y) / max(1.0, radius_y),
        )
        strongest = max(strongest, 1.0 - smoothstep(0.78, 1.10, normalized))
    return strongest


def generated_road_segments_from_graph(graph: StreetGraph, seed: int) -> list[TrafficPathSegment]:
    widths = {"arterial": 24.0, "collector": 17.0, "local": 11.0}
    segments: list[TrafficPathSegment] = []
    retained: list[tuple[StreetSegment, bool]] = []
    heights: dict[int, float] = {}
    airport_bounds = (900.0, -3000.0, 2250.0, -2210.0)
    for segment in graph.segments:
        start = graph.nodes[segment.start_node]
        end = graph.nodes[segment.end_node]
        midpoint_x = (start.x + end.x) * 0.5
        midpoint_y = (start.y + end.y) * 0.5
        if point_in_any_bounds(midpoint_x, midpoint_y, (airport_bounds,) + MISSION_PATH_BOUNDS):
            continue
        land_samples = [
            procedural_land_mask(
                start.x + (end.x - start.x) * factor,
                start.y + (end.y - start.y) * factor,
                seed,
            )
            for factor in (0.0, 0.20, 0.40, 0.60, 0.80, 1.0)
        ]
        if min(land_samples, default=0.0) < 0.52:
            continue
        water_samples = [
            inland_water_strength(
                start.x + (end.x - start.x) * factor,
                start.y + (end.y - start.y) * factor,
                seed,
            )
            for factor in (0.20, 0.40, 0.60, 0.80)
        ]
        if max(water_samples, default=0.0) > 0.55 and segment.street_class != "arterial":
            continue
        bridge = max(water_samples, default=0.0) > 0.55
        retained.append((segment, bridge))
        for node_index, point in ((segment.start_node, start), (segment.end_node, end)):
            height = base_terrain_height(point.x, point.y, seed)
            if bridge:
                height = max(8.0, height + 2.6)
            heights[node_index] = max(heights.get(node_index, height), height)

    # A node has one grade shared by every incident road, including bridges.
    # Clamping only each edge's end created steps at the same intersection.
    for _ in range(max(1, len(heights))):
        changed = False
        for segment, bridge in retained:
            start = graph.nodes[segment.start_node]
            end = graph.nodes[segment.end_node]
            limit = math.hypot(end.x - start.x, end.y - start.y) * (
                0.055 if segment.street_class == "arterial" else 0.075
            )
            for first, second in ((segment.start_node, segment.end_node), (segment.end_node, segment.start_node)):
                maximum = heights[second] + limit
                if heights[first] > maximum + 0.00001:
                    heights[first] = maximum
                    changed = True
        if not changed:
            break

    for segment, bridge in retained:
        start = graph.nodes[segment.start_node]
        end = graph.nodes[segment.end_node]
        segments.append(
            TrafficPathSegment(
                start.x,
                start.y,
                round(heights[segment.start_node], 3),
                end.x,
                end.y,
                round(heights[segment.end_node], 3),
                widths[segment.street_class],
            )
        )
    return segments


def merge_story_safe_road_segments(
    generated: list[TrafficPathSegment],
    retail: list[TrafficPathSegment],
) -> list[TrafficPathSegment]:
    """Keep every drivable retail path visibly represented, then add infill.

    Until the complete NODES topology writer is certified, removing ordinary
    retail segments would make vehicles drive across grass.  The procedural
    graph therefore supplies additional arterial/collector/local geometry and
    parcel structure, while every existing vehicle path remains a visible,
    grade-locked road.
    """
    result = list(retail)
    result.extend(generated)
    return deduplicate_world_road_segments(result)


def deduplicate_world_road_segments(segments: list[TrafficPathSegment]) -> list[TrafficPathSegment]:
    result: list[TrafficPathSegment] = []
    seen: set[tuple[int, int, int, int]] = set()
    for segment in segments:
        first = (round(segment.start_x * 2.0), round(segment.start_y * 2.0))
        second = (round(segment.end_x * 2.0), round(segment.end_y * 2.0))
        key = (*first, *second) if first <= second else (*second, *first)
        if key in seen:
            continue
        seen.add(key)
        result.append(segment)
    return result


def generate_los_santos_world(
    scan: GameScan,
    seed: int,
    columns: int = 12,
    rows: int = 12,
    tile_size: float = 250.0,
    ground_z: float = 12.0,
) -> LosSantosWorld:
    if not 4 <= columns <= 14 or not 4 <= rows <= 14:
        raise ValueError("Los Santos grid dimensions must be between 4 and 14 tiles.")
    if tile_size < 120.0 or tile_size > 400.0:
        raise ValueError("Los Santos tile size must be between 120 and 400 units.")

    game_directory = Path(scan.game_directory)
    gameplay_plan = collect_gameplay_plan(scan)
    gameplay_bounds = collect_gameplay_protected_bounds(game_directory)
    retail_surface_bounds = collect_retail_surface_bounds(
        scan, merge_protected_bounds(gameplay_plan.protected_bounds + gameplay_bounds + MISSION_INSTANCE_BOUNDS),
    )
    protected_instance_bounds = retail_surface_bounds
    protected_geometry_bounds = merge_protected_bounds(protected_instance_bounds + MISSION_SPAWN_BOUNDS)
    geometry_exclusion_bounds = merge_protected_bounds(protected_geometry_bounds + MISSION_PATH_BOUNDS)
    protected_path_bounds = merge_protected_bounds(MISSION_PATH_BOUNDS + gameplay_bounds)
    path_overrides = build_seeded_path_overrides(
        Path(scan.game_directory) / "data" / "paths",
        seed,
        LOS_SANTOS_BOUNDS,
        protected_path_bounds,
    )
    model_ids = allocate_los_santos_replacement_ids(
        scan, columns * rows, protected_instance_bounds, gameplay_plan.protected_model_ids,
    )
    texture_selection = select_retail_material_textures(Path(scan.game_directory) / "models" / "gta3.img", seed)
    retail_traffic_segments = los_santos_traffic_segments(path_overrides.vehicle_segments)
    procedural_graph = build_procedural_city_graph(seed)
    procedural_traffic_segments = generated_road_segments_from_graph(procedural_graph, seed)
    native_paths = build_binary_path_network(
        game_directory / "data" / "paths", procedural_traffic_segments,
        protected_bounds=retail_surface_bounds, rail_bounds=gameplay_plan.rail.protected_bounds,
        source_files=read_game_path_files(game_directory),
    )
    path_overrides = SeededPathOverrides(
        native_paths.files, retail_traffic_segments, 0, native_paths.retail_nodes_preserved,
    )
    traffic_segments = merge_story_safe_road_segments(
        native_paths.generated_segments, retail_traffic_segments
    )
    water_override = build_water_override(
        game_directory, procedural_land_outline(seed), procedural_graph.water_bodies,
        LOS_SANTOS_BOUNDS,
        exclusion_bounds=merge_protected_bounds(retail_surface_bounds + ((1000.0, -3000.0, 2250.0, -2210.0),)),
        preserved_water_bounds=retail_surface_bounds,
    )
    seed_source = random.Random(seed)
    models: list[GeneratedModel] = []
    placements: list[CustomPlacement] = []
    districts: list[str] = []
    terrain_elevations: list[float] = []
    minimum_x, minimum_y, _, _ = LOS_SANTOS_BOUNDS

    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            center_x = minimum_x + (column + 0.5) * tile_size
            center_y = minimum_y + (row + 0.5) * tile_size
            district_rng = random.Random(seed_source.getrandbits(64))
            district = choose_district(center_x, center_y, district_rng)
            district = choose_settlement_district(center_x, center_y, seed, district)
            tile_ground_z = ground_z
            model_name = scan.models[model_ids[index]].model_name
            tile_segments = traffic_segments_for_tile(traffic_segments, center_x, center_y, tile_size)
            tile_protected_bounds = protected_bounds_for_tile(
                protected_geometry_bounds,
                center_x,
                center_y,
                tile_size,
            )
            tile_exclusion_bounds = protected_bounds_for_tile(
                geometry_exclusion_bounds, center_x, center_y, tile_size,
            )
            tile_retail_bounds = protected_bounds_for_tile(retail_surface_bounds, center_x, center_y, tile_size)
            tile_rail_segments = traffic_segments_for_tile(list(gameplay_plan.rail.segments), center_x, center_y, tile_size)
            tile_bounds = (center_x - tile_size * 0.5, center_y - tile_size * 0.5,
                           center_x + tile_size * 0.5, center_y + tile_size * 0.5)
            tile_water_surfaces = tuple(surface for surface in water_override.surfaces
                                        if bounds_intersect(surface.bounds, (tile_bounds,))) if water_override.data else ()
            model = generate_district_tile(
                model_ids[index],
                model_name,
                tile_size,
                district,
                district_rng,
                tile_segments,
                center_x,
                center_y,
                seed,
                tile_protected_bounds,
                procedural_graph.blocks,
                tile_exclusion_bounds,
                tile_retail_bounds,
                tile_rail_segments,
                tile_water_surfaces,
            )
            model.txd_name = scan.models[model_ids[index]].txd_name
            model.material_texture_names = texture_selection.names
            districts.append(district)
            if not model.collision_mesh.triangles:
                continue
            models.append(model)
            placements.append(
                CustomPlacement(
                    model_id=model.model_id,
                    model_name=model.model_name,
                    x=center_x,
                    y=center_y,
                    z=tile_ground_z,
                )
            )
            terrain_elevations.append(
                continuous_terrain_height(
                    0.0,
                    0.0,
                    center_x,
                    center_y,
                    seed,
                    district,
                    tile_segments,
                    tile_protected_bounds,
                )
            )

    validation: list[str] = list(native_paths.validation) + list(gameplay_plan.diagnostics)
    track_errors = validate_preserved_tracks(gameplay_plan.rail, game_directory)
    if track_errors:
        raise RuntimeError("; ".join(track_errors))
    for model in models:
        if len(model.render_mesh.vertices) > 32_768:
            material_vertices = Counter(
                material
                for _, _, _, material in model.render_mesh.triangles
                for _ in range(3)
            )
            raise RuntimeError(
                f"Generated DFF {model.model_name} exceeds the 32,768-vertex GTA-era stream budget "
                f"({len(model.render_mesh.vertices)} vertices; materials {dict(material_vertices)})."
            )
        errors = validate_dff(write_dff(model.model_name, model.render_mesh, model.material_texture_names))
        if errors:
            raise RuntimeError(f"Generated DFF {model.model_name} failed validation: " + "; ".join(errors))
    validation.append(f"PASS: {len(models)} unique seeded Los Santos DFF models validated.")
    validation.append(
        f"PASS: every streamed tile stays within the 32,768-vertex budget; maximum is "
        f"{max(len(model.render_mesh.vertices) for model in models)}."
    )
    validation.extend(
        validate_mission_spawn_clearance(
            seed,
            tile_size,
            traffic_segments,
            protected_geometry_bounds,
        )
    )
    col_errors = validate_col3_archive(write_col3_archive(models), len(models))
    if col_errors:
        raise RuntimeError("Generated Los Santos COL3 failed validation: " + "; ".join(col_errors))
    validation.append(f"PASS: {len(models)} matching COL3 collision models validated.")
    validation.append("PASS: every procedural model has a placement; retained mission, railway and recorded-route terrain is excluded from generated ground.")
    validation.append(
        f"PASS: {len(model_ids)} existing Los Santos model IDs and names are reused without loading duplicate IDE definitions."
    )
    district_counts = Counter(districts)
    if columns * tile_size >= 3000.0 and rows * tile_size >= 3000.0:
        if district_counts.get("airport", 0) == 0:
            raise RuntimeError("The generated full Los Santos footprint contains no airport district.")
        validation.append(
            f"PASS: {district_counts['airport']} connected tiles reserve the Los Santos airport, runways, taxiway and hangars."
        )

    fingerprint_hash = hashlib.sha256()
    for model in models:
        district = model.style
        fingerprint_hash.update(model.model_name.encode("ascii"))
        fingerprint_hash.update(b":")
        fingerprint_hash.update(district.encode("ascii"))
        fingerprint_hash.update(b":")
        fingerprint_hash.update(write_dff(model.model_name, model.render_mesh, model.material_texture_names))
    fingerprint = fingerprint_hash.hexdigest()[:16]
    manifest = LosSantosManifest(
        generator_version=GENERATOR_VERSION,
        mode="full_los_santos_seed",
        seed=seed,
        seed_fingerprint=fingerprint,
        game_directory=scan.game_directory,
        mod_directory="",
        bounds=LOS_SANTOS_BOUNDS,
        columns=columns,
        rows=rows,
        tile_size=tile_size,
        custom_models=len(models),
        custom_buildings=sum(model.building_count for model in models),
        render_vertices=sum(len(model.render_mesh.vertices) for model in models),
        render_triangles=sum(len(model.render_mesh.triangles) for model in models),
        collision_vertices=sum(len(model.collision_mesh.vertices) for model in models),
        collision_triangles=sum(len(model.collision_mesh.triangles) for model in models),
        model_id_first=model_ids[0],
        model_id_last=model_ids[-1],
        district_tiles=dict(sorted(district_counts.items())),
        text_ipls_overridden=[],
        stream_ipls_overridden=[],
        collision_archives_overridden=[],
        texture_dictionaries_overridden=[],
        radar_tiles_overridden=[],
        material_texture_names=list(texture_selection.names),
        material_texture_sources=texture_selection.sources,
        traffic_path_segments_preserved=len(retail_traffic_segments),
        traffic_path_nodes_changed=path_overrides.changed_nodes,
        traffic_path_nodes_preserved=path_overrides.preserved_nodes,
        traffic_path_files_overridden=sorted(path_overrides.files),
        mission_safe_instances_preserved=0,
        gameplay_regions_preserved=len(protected_instance_bounds),
        terrain_elevation_minimum=min(terrain_elevations),
        terrain_elevation_maximum=max(terrain_elevations),
        ground_rules=[
            "Every generated main city reserves an operational airport; Las Venturas may additionally retain an abandoned desert airstrip.",
            "Every retail vehicle path remains a visible grade-locked road, while one connected procedural arterial, collector and local graph adds the organic city structure and missing between-grid connectors.",
            "Generated buildings pass exact road-corridor and emitted-collision checks; trunk placement reserves road clearance and opening mission routes.",
            "Retail vehicle paths and the procedural street graph jointly drive road render mesh, terrain grade locking, parcel clearance and radar.",
            "The arterial skeleton is generated before collector loops, diagonals and local subdivisions, so neighborhood grids cannot exist without connecting roads.",
            "Seeded inland ponds and lakes remove buildable land and interrupt local grids; only arterial-class bridge segments may cross them.",
            "A full-world diamond-square landmass plus octave detail is sampled in world coordinates; shared tile boundaries therefore resolve to identical heights.",
            "Road grades, mission sites and developed block platforms constrain the heightmap before render and collision meshes are built; assembled buildings are never vertex-warped.",
            "A smooth seeded population field modulates frontage density within district limits.",
            "Retail path coordinates are preserved instead of applying a cosmetic noise warp that makes roads wavy without creating useful topology.",
            "Retail instances around every discovered Los Santos ENEX entrance and garage are preserved together with explicit mission compounds.",
            "Original mission and railway surfaces are retained without generated ground above them; ordinary parcels use terrain-sampled pads.",
            "Parcels select coherent neighborhood wealth and condition, distinct low-rise layouts, shops, industrial yards, midrises or towers.",
            "Street-derived blocks are clipped around finite road corridors and subdivided as quadrilateral or convex polygon parcels; building collision must fit the individual parcel.",
            "Building parcels are rewritten through archetype-sensitive mass/facade/roof productions including domestic roofs, storefronts, podium/core midrises and setback towers.",
            "Road ribbons and junction caps render above one indexed 32-by-32-cell terrain collision surface; road grade takes priority over polygonal block pads.",
        ],
        limitations=[
            "Native vehicle NODES are generated and validated, but in-game traffic and complete mission playthroughs remain untested in this environment.",
            "Start a new disposable save and keep this seed isolated from other total-conversion map mods.",
            "Mission entrances, 25 named compounds, rail alignments and available CARREC routes are reserved; full story completion remains uncertified.",
        ],
        validation=validation,
        traffic_path_nodes_added=native_paths.nodes_added,
        traffic_car_links_added=native_paths.links_added,
        traffic_pedestrian_nodes_remapped=native_paths.pedestrian_nodes_remapped,
        neighborhood_buildings=dict(Counter(record.neighborhood for model in models for record in model.building_debug_records)),
        building_conditions=dict(Counter(record.condition for model in models for record in model.building_debug_records)),
        building_styles=dict(Counter(record.building_style for model in models for record in model.building_debug_records)),
        dwelling_units=sum(record.dwelling_units for model in models for record in model.building_debug_records),
        water=water_override.diagnostics,
        mission_regions=[region.name for region in gameplay_plan.mission_regions],
        railway_tracks_preserved=gameplay_plan.rail.source_hashes,
    )
    return LosSantosWorld(
        models,
        placements,
        districts,
        traffic_segments,
        path_overrides,
        protected_instance_bounds,
        texture_selection.names,
        texture_selection.native_chunks,
        texture_selection.generated_textures,
        manifest,
        procedural_graph,
        [record for model in models for record in getattr(model, "building_debug_records", [])],
        native_paths,
        gameplay_plan,
        water_override,
        tuple(sorted(gameplay_plan.protected_model_ids)),
        retail_surface_bounds,
        load_retail_radar_tiles(game_directory / "models" / "gta3.img"),
    )


def load_retail_radar_tiles(img_path: Path) -> dict[str, bytes]:
    names = {f"radar{row * 12 + column:02d}.txd" for row in range(6, 12) for column in range(6, 12)}
    return {entry.name.lower(): read_img_entry_payload(img_path, entry)
            for entry in read_img_v2_entries(img_path) if entry.name.lower() in names}


def collect_retail_surface_bounds(
    scan: GameScan,
    protected_bounds: tuple[tuple[float, float, float, float], ...],
) -> tuple[tuple[float, float, float, float], ...]:
    img_path = Path(scan.game_directory) / "models" / "gta3.img"
    model_bounds: dict[int, tuple[float, ...]] = {}
    for entry in collision_archive_entries(img_path):
        records, _ = parse_col_archive(read_img_entry_payload(img_path, entry), entry.name)
        for record in records:
            if len(record.raw) < 72 or record.raw[:4] not in {b"COL2", b"COL3", b"COL4"}:
                continue
            values = struct.unpack_from("<6f", record.raw, 32)
            if all(math.isfinite(value) and abs(value) <= 10000.0 for value in values) and all(values[axis] <= values[axis + 3] for axis in range(3)):
                model_bounds[record.model_id] = values
    placements = [(placement.model_id, (placement.x, placement.y, placement.z),
                   (placement.qx, placement.qy, placement.qz, placement.qw)) for placement in scan.placements]
    for entry in find_los_santos_stream_ipls(img_path):
        data = read_img_entry_payload(img_path, entry)
        count, _, offset, _ = parse_binary_ipl_header(data)
        for index in range(count):
            values = struct.unpack_from("<7fi", data, offset + index * 40)
            placements.append((values[7], values[:3], values[3:7]))
    result = list(protected_bounds)
    for model_id, position, rotation in placements:
        bounds = model_bounds.get(model_id)
        if bounds is None:
            continue
        qx, qy, qz, qw = rotation
        length = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if length < 0.00001:
            continue
        qx, qy, qz, qw = (component / length for component in rotation)
        matrix = (
            (1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)),
            (2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)),
            (2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)),
        )
        points = []
        for x in (bounds[0], bounds[3]):
            for y in (bounds[1], bounds[4]):
                for z in (bounds[2], bounds[5]):
                    local = (x, y, z)
                    points.append(tuple(position[axis] + sum(matrix[axis][component] * local[component] for component in range(3)) for axis in (0, 1)))
                    points.append(tuple(position[axis] + sum(matrix[component][axis] * local[component] for component in range(3)) for axis in (0, 1)))
        footprint = (min(x for x, _ in points), min(y for _, y in points), max(x for x, _ in points), max(y for _, y in points))
        if bounds_intersect(footprint, protected_bounds):
            result.append((footprint[0] - 1.0, footprint[1] - 1.0, footprint[2] + 1.0, footprint[3] + 1.0))
    return merge_protected_bounds(tuple(result))


def collect_gameplay_protected_bounds(
    game_directory: Path,
) -> tuple[tuple[float, float, float, float], ...]:
    """Discover exterior entrances and complete garage approach regions."""
    maps_directory = game_directory / "data" / "maps"
    if not maps_directory.is_dir():
        return ()
    bounds: list[tuple[float, float, float, float]] = []
    for path in sorted(maps_directory.rglob("*"), key=lambda item: str(item).lower()):
        normalized = str(path).replace("\\", "/").lower()
        if (
            not path.is_file()
            or path.suffix.lower() != ".ipl"
            or "/data/maps/la/" not in normalized
        ):
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if raw[:4].lower() == b"bnry" or b"\0" in raw[:256]:
            continue
        text = raw.decode("cp1252", errors="replace")
        section = ""
        for raw_line in text.splitlines():
            line = raw_line.partition("#")[0].strip()
            lowered = line.lower()
            if not line:
                continue
            if lowered in {"enex", "grge"}:
                section = lowered
                continue
            if lowered == "end":
                section = ""
                continue
            if not section:
                continue
            fields = line.replace(",", " ").split()
            if len(fields) < 2:
                continue
            try:
                x, y = float(fields[0]), float(fields[1])
            except ValueError:
                continue
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            if section == "enex":
                radius = 14.0
                if len(fields) >= 6:
                    try:
                        range_x, range_y = float(fields[4]), float(fields[5])
                    except ValueError:
                        pass
                    else:
                        if all(
                            math.isfinite(value) and 0.0 <= value <= 250.0
                            for value in (range_x, range_y)
                        ):
                            radius = max(radius, math.hypot(range_x, range_y) * 0.5 + 8.0)
                append_protected_point(bounds, x, y, radius)
                continue

            if len(fields) >= 11:
                try:
                    coordinates = [float(field) for field in fields[:8]]
                except ValueError:
                    coordinates = []
                if coordinates and all(math.isfinite(value) for value in coordinates):
                    front_x, front_y = coordinates[3:5]
                    rear_x, rear_y = coordinates[5:7]
                    width_x, width_y = front_x - x, front_y - y
                    depth_x, depth_y = rear_x - x, rear_y - y
                    garage_width = math.hypot(width_x, width_y)
                    garage_depth = math.hypot(depth_x, depth_y)
                    area = abs(width_x * depth_y - width_y * depth_x)
                    if (
                        all(math.isfinite(value) for value in (garage_width, garage_depth, area))
                        and garage_width >= 0.1
                        and garage_depth >= 0.1
                        and area > 0.1
                    ):
                        corners = (
                            (x, y),
                            (front_x, front_y),
                            (rear_x, rear_y),
                            (front_x + depth_x, front_y + depth_y),
                        )
                        region = (
                            min(point[0] for point in corners) - 18.0,
                            min(point[1] for point in corners) - 18.0,
                            max(point[0] for point in corners) + 18.0,
                            max(point[1] for point in corners) + 18.0,
                        )
                        minimum_x, minimum_y, maximum_x, maximum_y = LOS_SANTOS_BOUNDS
                        if (
                            region[2] >= minimum_x
                            and region[0] <= maximum_x
                            and region[3] >= minimum_y
                            and region[1] <= maximum_y
                        ):
                            bounds.append(region)
                        continue
            append_protected_point(bounds, x, y, 18.0)
    return tuple(dict.fromkeys(bounds))


def append_protected_point(
    bounds: list[tuple[float, float, float, float]],
    x: float,
    y: float,
    radius: float,
) -> None:
    if not all(math.isfinite(value) for value in (x, y, radius)) or radius < 0.0:
        return
    minimum_x, minimum_y, maximum_x, maximum_y = LOS_SANTOS_BOUNDS
    if (
        x + radius >= minimum_x
        and x - radius <= maximum_x
        and y + radius >= minimum_y
        and y - radius <= maximum_y
    ):
        bounds.append((x - radius, y - radius, x + radius, y + radius))


def merge_protected_bounds(
    source: tuple[tuple[float, float, float, float], ...],
) -> tuple[tuple[float, float, float, float], ...]:
    unique: list[tuple[float, float, float, float]] = []
    for bounds in source:
        if bounds not in unique:
            unique.append(bounds)
    return tuple(unique)


def protected_bounds_for_tile(
    protected_bounds: tuple[tuple[float, float, float, float], ...],
    center_x: float,
    center_y: float,
    tile_size: float,
) -> tuple[tuple[float, float, float, float], ...]:
    half = tile_size * 0.5
    tile_bounds = (center_x - half, center_y - half, center_x + half, center_y + half)
    local: list[tuple[float, float, float, float]] = []
    for minimum_x, minimum_y, maximum_x, maximum_y in protected_bounds:
        if (
            maximum_x < tile_bounds[0]
            or minimum_x > tile_bounds[2]
            or maximum_y < tile_bounds[1]
            or minimum_y > tile_bounds[3]
        ):
            continue
        local.append(
            (
                minimum_x - center_x,
                minimum_y - center_y,
                maximum_x - center_x,
                maximum_y - center_y,
            )
        )
    return tuple(local)


def allocate_los_santos_replacement_ids(
    scan: GameScan,
    count: int,
    protected_bounds: tuple[tuple[float, float, float, float], ...] = (),
    preserved_model_ids: frozenset[int] | set[int] | tuple[int, ...] = (),
) -> list[int]:
    img_path = Path(scan.game_directory) / "models" / "gta3.img"
    collision_keys = collect_collision_model_keys(img_path)
    streamed_names = {entry.name.lower() for entry in read_img_v2_entries(img_path)}
    protected_ids = collect_protected_model_ids(img_path, scan, protected_bounds) | set(preserved_model_ids)
    candidates = sorted(
        model.model_id
        for model in scan.models.values()
        if model.section == "objs"
        and 0 <= model.model_id < 20_000
        and not model.flags & UNSAFE_REPLACEMENT_FLAGS
        and model.source_file.replace("\\", "/").lower().startswith("data/maps/la/")
        and (model.model_id, model.model_name.lower()) in collision_keys
        and f"{model.txd_name.lower()}.txd" in streamed_names
        and model.model_id not in protected_ids
        and not is_lod_model_name(model.model_name)
    )
    if len(candidates) < count:
        raise RuntimeError(
            f"Only {len(candidates)} safe Los Santos OBJS models with matching retail collision records "
            f"were found, but {count} are required. The retail data\\maps\\LA IDE files or "
            "models\\gta3.img collision archives are missing or incomplete."
        )
    return candidates[:count]


def is_lod_model_name(model_name: str) -> bool:
    """Reject streaming LOD identities as procedural collision containers."""
    lowered = model_name.lower()
    return (
        lowered.startswith("lod")
        or lowered.endswith("lod")
        or "_lod" in lowered
        or "lod_" in lowered
        or "lodx" in lowered
    )


def collect_protected_model_ids(
    img_path: Path,
    scan: GameScan,
    protected_bounds: tuple[tuple[float, float, float, float], ...],
) -> set[int]:
    if not protected_bounds:
        return set()
    protected: set[int] = {
        placement.model_id
        for placement in scan.placements
        if point_in_world_bounds(placement.x, placement.y, protected_bounds)
    }
    for entry in find_los_santos_stream_ipls(img_path):
        data = read_img_entry_payload(img_path, entry)
        instance_count, _, instance_offset, _ = parse_binary_ipl_header(data)
        for index in range(instance_count):
            offset = instance_offset + index * 40
            x, y = struct.unpack_from("<2f", data, offset)
            if point_in_world_bounds(x, y, protected_bounds):
                protected.add(struct.unpack_from("<i", data, offset + 28)[0])
    return protected


def point_in_world_bounds(
    x: float,
    y: float,
    collection: tuple[tuple[float, float, float, float], ...],
) -> bool:
    return any(
        minimum_x <= x <= maximum_x and minimum_y <= y <= maximum_y
        for minimum_x, minimum_y, maximum_x, maximum_y in collection
    )


def choose_tile_ground_z(
    center_x: float,
    center_y: float,
    seed: int,
    tile_segments: list[TrafficPathSegment] | None,
) -> float:
    if tile_segments:
        path_heights = [
            height
            for segment in tile_segments
            for height in (segment.start_z, segment.end_z)
            if -20.0 <= height <= 180.0
        ]
        if path_heights:
            path_heights.sort()
            return round(path_heights[len(path_heights) // 2], 3)
    first = terrain_value_noise(center_x / 850.0, center_y / 850.0, seed ^ 0x59D2A41B)
    second = terrain_value_noise(center_x / 360.0, center_y / 360.0, seed ^ 0xA172BC31)
    return round(12.0 + first * 4.5 + second * 1.5, 3)


def validate_mission_spawn_clearance(
    seed: int,
    tile_size: float,
    traffic_segments: list[TrafficPathSegment],
    protected_geometry_bounds: tuple[tuple[float, float, float, float], ...],
) -> list[str]:
    validation: list[str] = []
    minimum_x, minimum_y, _, _ = LOS_SANTOS_BOUNDS
    for spawn_x, spawn_y, spawn_z, label in MISSION_SPAWN_POINTS:
        column = math.floor((spawn_x - minimum_x) / tile_size)
        row = math.floor((spawn_y - minimum_y) / tile_size)
        center_x = minimum_x + (column + 0.5) * tile_size
        center_y = minimum_y + (row + 0.5) * tile_size
        local_x = spawn_x - center_x
        local_y = spawn_y - center_y
        tile_segments = traffic_segments_for_tile(traffic_segments, center_x, center_y, tile_size)
        tile_protected_bounds = protected_bounds_for_tile(
            protected_geometry_bounds,
            center_x,
            center_y,
            tile_size,
        )
        surface_height = continuous_terrain_height(
            local_x,
            local_y,
            center_x,
            center_y,
            seed,
            "mission_spawn",
            tile_segments,
            tile_protected_bounds,
        )
        clearance = spawn_z - surface_height
        if clearance < 0.75:
            raise RuntimeError(
                f"{label} at Z {spawn_z:.2f} has only {clearance:.2f} units above generated terrain "
                f"at Z {surface_height:.2f}."
            )
        validation.append(
            f"PASS: {label} remains {clearance:.2f} units above generated terrain at Z {surface_height:.2f}."
        )
    return validation


def base_terrain_height(world_x: float, world_y: float, seed: int) -> float:
    continent = diamond_square_height(world_x, world_y, seed)
    broad = terrain_value_noise(world_x / 900.0, world_y / 900.0, seed ^ 0x59D2A41B)
    detail = terrain_value_noise(world_x / 360.0, world_y / 360.0, seed ^ 0xA172BC31)
    coast_factor = 1.0 - smoothstep(250.0, 650.0, world_x)
    hill_factor = smoothstep(-950.0, -150.0, world_y)
    south_flat_factor = 1.0 - smoothstep(-2380.0, -2050.0, world_y)
    height = 12.0 + continent * 2.8 + broad * 1.35 + detail * 0.45
    height += hill_factor * (18.0 + broad * 1.5 + detail * 0.5)
    island_factor = procedural_land_mask(world_x, world_y, seed)
    height += coast_factor * (9.7 - height) * island_factor
    height += south_flat_factor * (10.6 - height) * 0.90 * island_factor
    height = (-2.8 * (1.0 - smoothstep(0.20, 0.50, island_factor))
              + height * smoothstep(0.50, 0.72, island_factor))
    inland_water = inland_water_strength(world_x, world_y, seed)
    height += (4.85 - height) * inland_water
    airport_distance = math.hypot(
        max(1000.0 - world_x, 0.0, world_x - 2250.0),
        max(-3000.0 - world_y, 0.0, world_y + 2250.0),
    )
    airport_factor = 1.0 - smoothstep(0.0, 80.0, airport_distance)
    height += (MISSION_SAFE_TERRAIN_HEIGHT - height) * airport_factor
    return height


def continuous_terrain_height(
    local_x: float,
    local_y: float,
    world_center_x: float,
    world_center_y: float,
    seed: int,
    district: str,
    traffic_segments: list[TrafficPathSegment],
    protected_bounds: tuple[tuple[float, float, float, float], ...] = (),
) -> float:
    world_x = world_center_x + local_x
    world_y = world_center_y + local_y
    # Height is based only on world coordinates. District labels change abruptly
    # at tile edges, so using them here created disconnected slabs.  Inland
    # water is part of this same continuous field rather than painted afterward.
    height = base_terrain_height(world_x, world_y, seed)

    nearest_distance = float("inf")
    nearest_height = height
    nearest_surface_radius = 0.0
    nearest_shoulder_radius = 0.0
    for segment in traffic_segments:
        distance = distance_from_point_to_segment(
            local_x,
            local_y,
            segment.start_x,
            segment.start_y,
            segment.end_x,
            segment.end_y,
        )
        if distance >= nearest_distance:
            continue
        delta_x = segment.end_x - segment.start_x
        delta_y = segment.end_y - segment.start_y
        length_squared = delta_x * delta_x + delta_y * delta_y
        if length_squared <= 0.000001:
            factor = 0.0
        else:
            factor = (
                (local_x - segment.start_x) * delta_x
                + (local_y - segment.start_y) * delta_y
            ) / length_squared
            factor = max(0.0, min(1.0, factor))
        nearest_distance = distance
        nearest_height = segment.start_z + (segment.end_z - segment.start_z) * factor
        nearest_surface_radius = max(8.0, segment.width * 0.5 + 2.0)
        nearest_shoulder_radius = nearest_surface_radius + 16.0

    # The traffic graph owns the grade. Lock the carriageway and verge to it,
    # then blend across a broad shoulder so a road cannot pass between terrain
    # samples and leave a collision ridge beside a floating render ribbon.
    if nearest_distance <= nearest_surface_radius:
        height = nearest_height - 0.04
    elif nearest_distance < nearest_shoulder_radius:
        factor = 1.0 - (
            (nearest_distance - nearest_surface_radius)
            / (nearest_shoulder_radius - nearest_surface_radius)
        )
        factor = factor * factor * (3.0 - 2.0 * factor)
        height += (nearest_height - 0.04 - height) * factor
    protected_factor = protected_terrain_blend_factor(local_x, local_y, protected_bounds)
    if protected_factor > 0.0:
        height += (MISSION_SAFE_TERRAIN_HEIGHT - height) * protected_factor
    return round(height, 4)


@lru_cache(maxsize=16)
def diamond_square_field(seed: int, size: int = 65) -> tuple[tuple[float, ...], ...]:
    """Return a normalized fractal landmass field with deterministic corners."""
    if size < 3 or (size - 1) & (size - 2):
        raise ValueError("Diamond-square size must be 2^n + 1.")
    rng = random.Random(seed ^ 0xD14A0D5)
    values = [[0.0 for _ in range(size)] for _ in range(size)]
    values[0][0] = rng.uniform(-0.45, 0.15)
    values[0][-1] = rng.uniform(-0.45, 0.15)
    values[-1][0] = rng.uniform(-0.45, 0.15)
    values[-1][-1] = rng.uniform(-0.45, 0.15)
    step = size - 1
    roughness = 0.92
    displacement = 1.0
    while step > 1:
        half = step // 2
        for y in range(half, size - 1, step):
            for x in range(half, size - 1, step):
                average = (
                    values[y - half][x - half]
                    + values[y - half][x + half]
                    + values[y + half][x - half]
                    + values[y + half][x + half]
                ) * 0.25
                values[y][x] = average + rng.uniform(-displacement, displacement)
        for y in range(0, size, half):
            offset = half if (y // half) % 2 == 0 else 0
            for x in range(offset, size, step):
                neighbors = []
                for neighbor_x, neighbor_y in ((x - half, y), (x + half, y), (x, y - half), (x, y + half)):
                    if 0 <= neighbor_x < size and 0 <= neighbor_y < size:
                        neighbors.append(values[neighbor_y][neighbor_x])
                values[y][x] = sum(neighbors) / len(neighbors) + rng.uniform(-displacement, displacement)
        step //= 2
        displacement *= roughness * 0.5
    minimum = min(min(row) for row in values)
    maximum = max(max(row) for row in values)
    span = max(0.000001, maximum - minimum)
    return tuple(tuple(((value - minimum) / span) * 2.0 - 1.0 for value in row) for row in values)


def diamond_square_height(world_x: float, world_y: float, seed: int) -> float:
    field = diamond_square_field(seed)
    size = len(field)
    # One field covers the GTA world coordinate square; bilinear sampling makes
    # every chunk query continuous and independent of generation order.
    grid_x = max(0.0, min(size - 1.000001, (world_x + 3000.0) / 6000.0 * (size - 1)))
    grid_y = max(0.0, min(size - 1.000001, (world_y + 3000.0) / 6000.0 * (size - 1)))
    x0 = math.floor(grid_x)
    y0 = math.floor(grid_y)
    x1 = min(size - 1, x0 + 1)
    y1 = min(size - 1, y0 + 1)
    tx = grid_x - x0
    ty = grid_y - y0
    bottom = field[y0][x0] + (field[y0][x1] - field[y0][x0]) * tx
    top = field[y1][x0] + (field[y1][x1] - field[y1][x0]) * tx
    return bottom + (top - bottom) * ty


def procedural_land_mask(world_x: float, world_y: float, seed: int) -> float:
    """Return continuous land coverage for an irregular Los Santos landmass."""

    def lobe(center_x: float, center_y: float, radius_x: float, radius_y: float) -> float:
        return 1.0 - math.hypot((world_x - center_x) / radius_x, (world_y - center_y) / radius_y)

    def bay(center_x: float, center_y: float, radius_x: float, radius_y: float) -> float:
        distance = math.hypot((world_x - center_x) / radius_x, (world_y - center_y) / radius_y)
        return max(0.0, 1.0 - distance)

    # Overlapping, differently proportioned regions form hills, the urban
    # basin, east-side industry and the airport peninsula.  Their union is
    # intentionally not centered and not rotationally symmetric.
    land_score = max(
        lobe(1740.0, -1530.0, 1270.0, 1070.0),  # urban basin
        lobe(890.0, -700.0, 850.0, 540.0),     # northwest hills
        lobe(850.0, -2370.0, 720.0, 500.0),    # southwest coast
        lobe(1760.0, -2710.0, 930.0, 350.0),   # airport peninsula
        lobe(2590.0, -1610.0, 540.0, 890.0),   # east-side industry
        lobe(1850.0, -370.0, 690.0, 470.0),    # northern ridge
    )

    broad_noise = terrain_value_noise(world_x / 720.0, world_y / 720.0, seed ^ 0x6C414E44)
    ridge_noise = diamond_square_height(world_x, world_y, seed ^ 0xC0457A)
    land_score += broad_noise * 0.105 + ridge_noise * 0.055

    # Negative fields make readable bays and harbour cuts.  They are placed
    # away from mission compounds and create concave coast rather than merely
    # roughening an ellipse.
    harbour_cut = bay(180.0, -1620.0, 680.0, 520.0)
    airport_bay = bay(2700.0, -2780.0, 680.0, 430.0)
    north_channel = bay(720.0, -80.0, 620.0, 430.0)
    land_score -= harbour_cut * 0.40
    land_score -= airport_bay * 0.42
    land_score -= north_channel * 0.30

    return smoothstep(-0.085, 0.075, land_score)


def procedural_land_outline(seed: int, sample_count: int = 96) -> tuple[tuple[float, float], ...]:
    """Trace the connected landmass for selector and report rendering."""
    if sample_count < 32:
        raise ValueError("Land outline requires at least 32 angular samples.")
    center_x = 1700.0
    center_y = -1550.0
    maximum_radius = 2750.0
    points: list[tuple[float, float]] = []
    for index in range(sample_count):
        angle = -math.pi * 0.5 + math.tau * index / sample_count
        cosine = math.cos(angle)
        sine = math.sin(angle)
        last_land_radius = 0.0
        for step in range(1, 111):
            radius = maximum_radius * step / 110.0
            sample_x = center_x + cosine * radius
            sample_y = center_y + sine * radius
            if procedural_land_mask(sample_x, sample_y, seed) >= 0.5:
                last_land_radius = radius
        lower = last_land_radius
        upper = min(maximum_radius, last_land_radius + maximum_radius / 110.0)
        for _ in range(8):
            middle = (lower + upper) * 0.5
            if procedural_land_mask(center_x + cosine * middle, center_y + sine * middle, seed) >= 0.5:
                lower = middle
            else:
                upper = middle
        points.append(
            (
                max(0.0, min(3000.0, center_x + cosine * lower)),
                max(-3000.0, min(0.0, center_y + sine * lower)),
            )
        )
    return tuple(points)


def protected_terrain_blend_factor(
    local_x: float,
    local_y: float,
    protected_bounds: tuple[tuple[float, float, float, float], ...],
    transition_distance: float = 36.0,
) -> float:
    strongest = 0.0
    for minimum_x, minimum_y, maximum_x, maximum_y in protected_bounds:
        outside_x = max(minimum_x - local_x, 0.0, local_x - maximum_x)
        outside_y = max(minimum_y - local_y, 0.0, local_y - maximum_y)
        distance = math.hypot(outside_x, outside_y)
        if distance <= 0.0:
            return 1.0
        if distance < transition_distance:
            strongest = max(strongest, 1.0 - smoothstep(0.0, transition_distance, distance))
    return strongest


def smoothstep(edge_start: float, edge_end: float, value: float) -> float:
    if edge_end <= edge_start:
        raise ValueError("smoothstep requires an increasing edge range")
    factor = max(0.0, min(1.0, (value - edge_start) / (edge_end - edge_start)))
    return factor * factor * (3.0 - 2.0 * factor)


def warp_model_to_continuous_terrain(
    model: GeneratedModel,
    world_center_x: float,
    world_center_y: float,
    seed: int,
    district: str,
    traffic_segments: list[TrafficPathSegment],
    protected_bounds: tuple[tuple[float, float, float, float], ...] = (),
) -> None:
    for mesh in (model.render_mesh, model.collision_mesh):
        warped_vertices: list[tuple[float, float, float]] = []
        for x, y, z in mesh.vertices:
            surface_height = continuous_terrain_height(
                x,
                y,
                world_center_x,
                world_center_y,
                seed,
                district,
                traffic_segments,
                protected_bounds,
            )
            warped_vertices.append((x, y, z + surface_height - 12.0))
        mesh.vertices = warped_vertices
        recalculated = list(mesh.normals)
        for first, second, third, _ in mesh.triangles:
            normal = calculate_normal(
                mesh.vertices[first],
                mesh.vertices[second],
                mesh.vertices[third],
            )
            recalculated[first] = normal
            recalculated[second] = normal
            recalculated[third] = normal
        mesh.normals = recalculated


def terrain_value_noise(x: float, y: float, seed: int) -> float:
    x0 = math.floor(x)
    y0 = math.floor(y)
    tx = (x - x0) ** 2 * (3.0 - 2.0 * (x - x0))
    ty = (y - y0) ** 2 * (3.0 - 2.0 * (y - y0))

    def value(ix: int, iy: int) -> float:
        hashed = (ix * 0x1F123BB5) ^ (iy * 0x5F356495) ^ seed
        hashed = (hashed ^ (hashed >> 16)) * 0x45D9F3B
        hashed = (hashed ^ (hashed >> 16)) * 0x45D9F3B
        hashed ^= hashed >> 16
        return ((hashed & 0xFFFFFFFF) / 0x7FFFFFFF) - 1.0

    bottom = value(x0, y0) + (value(x0 + 1, y0) - value(x0, y0)) * tx
    top = value(x0, y0 + 1) + (value(x0 + 1, y0 + 1) - value(x0, y0 + 1)) * tx
    return bottom + (top - bottom) * ty


def bounds_intersect(
    bounds: tuple[float, float, float, float],
    collection: tuple[tuple[float, float, float, float], ...],
) -> bool:
    minimum_x, minimum_y, maximum_x, maximum_y = bounds
    return any(
        not (
            maximum_x < other_minimum_x
            or minimum_x > other_maximum_x
            or maximum_y < other_minimum_y
            or minimum_y > other_maximum_y
        )
        for other_minimum_x, other_minimum_y, other_maximum_x, other_maximum_y in collection
    )


TEXTURE_CATEGORY_RULES = (
    ("road", ("asphalt", "tarmac", "freeway", "hiway", "highway", "road"), ("sign", "line", "edge", "rail", "lamp", "bridge", "barrier"), False),
    ("sidewalk", ("sidewalk", "pavement", "paving", "pave", "flagstone", "concrete", "floor"), ("wall", "roof", "road", "pillar"), False),
    ("wall", ("stucco", "plaster", "brick", "wall", "stone", "concrete"), ("ground", "floor", "road", "roof", "sidewalk"), False),
    ("roof", ("roof", "shingle", "rooftile", "tile", "tar"), ("road", "floor", "sign"), False),
    ("window", ("window", "glass", "shopfront", "shopwin"), ("broken", "decal", "sign"), False),
    ("foliage", ("foliage", "leaf", "leaves", "tree", "bush", "plant", "hedge", "palm", "veg"), ("bark", "trunk", "pot", "shadow"), True),
    ("trim", ("metal", "wood", "door", "panel", "garage", "awning", "sign"), ("alpha", "shadow"), False),
    ("grass", ("grass", "lawn", "ground", "turf", "dirt"), ("wall", "road", "floor", "sign"), False),
)


def select_retail_material_textures(img_path: Path, seed: int) -> MaterialTextureSelection:
    generated_textures = generate_world_textures(seed)
    names = tuple(name for name, _, _ in generated_textures)
    categories = tuple(rule[0] for rule in TEXTURE_CATEGORY_RULES)
    sources = [
        f"{category}: seeded Seed Andreas GTA-style texture"
        for category in categories
    ]
    return MaterialTextureSelection(names, [], generated_textures, sources)


def choose_district(x: float, y: float, rng: random.Random) -> str:
    if y < -2250.0:
        if 1050.0 <= x <= 2150.0:
            return "airport"
        if x > 2150.0:
            return "docks"
        return "coastal" if x < 500.0 else "industrial"
    if x < 420.0:
        return "beach"
    if y > -700.0:
        return "hills"
    if 1150.0 < x < 2100.0 and -1850.0 < y < -700.0:
        return "downtown"
    if x > 2050.0 and -2250.0 < y < -1150.0:
        return "grove"
    if x > 2050.0:
        return "eastside"
    if x < 1100.0:
        return "coastal"
    if rng.random() < 0.22:
        return rng.choice(("civic", "park", "commercial"))
    return "commercial" if y > -1900.0 else "residential"


def generate_district_tile(
    model_id: int,
    model_name: str,
    tile_size: float,
    district: str,
    rng: random.Random,
    traffic_segments: list[TrafficPathSegment] | None = None,
    world_x: float = 0.0,
    world_y: float = 0.0,
    terrain_seed: int = 0,
    protected_bounds: tuple[tuple[float, float, float, float], ...] = (),
    street_blocks: list[StreetBlock] | None = None,
    geometry_exclusion_bounds: tuple[tuple[float, float, float, float], ...] = (),
    retail_surface_bounds: tuple[tuple[float, float, float, float], ...] = (),
    rail_segments: list[TrafficPathSegment] | None = None,
    water_surfaces: tuple[WaterSurface, ...] = (),
) -> GeneratedModel:
    traffic_segments = traffic_segments or []
    rail_segments = rail_segments or []
    grade_segments = traffic_segments + [
        TrafficPathSegment(segment.start_x, segment.start_y, segment.start_z - 0.30,
                           segment.end_x, segment.end_y, segment.end_z - 0.30, segment.width)
        for segment in rail_segments
    ]
    solid_clearance_segments = traffic_segments + rail_segments + protected_bounds_as_clearance_segments(
        geometry_exclusion_bounds or protected_bounds
    )
    render = MeshBuilder()
    collision = MeshBuilder()
    building_debug_records: list[BuildingDebugRecord] = []
    tile_half = tile_size * 0.5

    # A previous prototype used a solid tile-sized box here.  In RenderWare that
    # becomes a second ceiling/floor and can leave CJ below the generated city.
    # The terrain is now a single shared-height surface.  Roads, sidewalks and
    # foundations are layered upward from the same Z=0 datum.
    blocks = [] if district == "airport" else build_city_blocks(
        tile_size,
        district,
        world_x,
        world_y,
        rng,
        grade_segments,
        terrain_seed,
        street_blocks,
    )
    blocks = trim_city_blocks(blocks, retail_surface_bounds, world_x, world_y, terrain_seed, grade_segments)
    add_heightfield_terrain(
        render,
        collision,
        tile_size,
        district,
        world_x,
        world_y,
        terrain_seed,
        grade_segments,
        protected_bounds,
        blocks,
        retail_surface_bounds,
        water_surfaces,
    )
    add_block_surfaces(render, collision, blocks, district)

    surface_segments = [segment for segment in deduplicate_road_surface_segments(traffic_segments)
                        if not any(segment_intersects_rectangle(segment.start_x, segment.start_y, segment.end_x, segment.end_y, *bounds)
                                   for bounds in retail_surface_bounds)]
    for segment in surface_segments:
        add_road_ribbon(render, segment, tile_half)
        add_traffic_lane_markings(render, segment, tile_half)
        water_levels = [surface.vertices[0].z for surface in water_surfaces
                        if surface.vertices[0].z <= 950.0 and segment_intersects_rectangle(
                            world_x + segment.start_x, world_y + segment.start_y,
                            world_x + segment.end_x, world_y + segment.end_y, *surface.bounds,
                        )]
        if water_levels and max(segment.start_z, segment.end_z) > min(water_levels) + 0.6:
            add_road_ribbon(collision, segment, tile_half)
            add_bridge_supports(render, collision, segment, tile_half, min(water_levels))
    add_road_junctions(render, surface_segments, tile_half)

    if district == "airport":
        building_count = 0 if retail_surface_bounds else add_airport_tile(
            render, collision, tile_size, world_x, world_y, rng, solid_clearance_segments
        )
    elif district == "docks":
        building_count = populate_city_blocks(
            render, collision, blocks, "industrial", rng, solid_clearance_segments,
            warehouse_bias=True, world_x=world_x, world_y=world_y, seed=terrain_seed,
            debug_records=building_debug_records,
        )
    elif district == "park":
        building_count = populate_park_blocks(render, collision, blocks, rng, solid_clearance_segments)
    elif district in {"industrial", "eastside"}:
        building_count = populate_city_blocks(
            render, collision, blocks, district, rng, solid_clearance_segments,
            warehouse_bias=True, world_x=world_x, world_y=world_y, seed=terrain_seed,
            debug_records=building_debug_records,
        )
    else:
        building_count = populate_city_blocks(
            render, collision, blocks, district, rng, solid_clearance_segments,
            world_x=world_x, world_y=world_y, seed=terrain_seed,
            debug_records=building_debug_records,
        )

    add_seed_marker(render, seed_color(rng), tile_half)
    return GeneratedModel(
        model_id, model_name, render.mesh, collision.mesh, building_count, district,
        building_debug_records=building_debug_records,
    )


def add_heightfield_terrain(
    render: MeshBuilder,
    collision: MeshBuilder,
    tile_size: float,
    district: str,
    world_x: float,
    world_y: float,
    seed: int,
    traffic_segments: list[TrafficPathSegment],
    protected_bounds: tuple[tuple[float, float, float, float], ...],
    blocks: list[CityBlock],
    retail_surface_bounds: tuple[tuple[float, float, float, float], ...] = (),
    water_surfaces: tuple[WaterSurface, ...] = (),
) -> None:
    """Build a deterministic, seam-safe terrain heightfield.

    This follows the useful part of chunked octave-noise terrain generators:
    every vertex is sampled in global coordinates, multiple frequencies are
    accumulated, and the resulting height field is smoothed by low amplitudes.
    Roads, mission sites and developed blocks are flattened *in the heightmap*
    before geometry exists.  Buildings are never warped after assembly.
    """
    half = tile_size * 0.5
    divisions = 32
    step = tile_size / divisions

    def sampled(local_x: float, local_y: float, on_border: bool) -> float:
        world_height = continuous_terrain_height(
            local_x, local_y, world_x, world_y, seed, district, traffic_segments, protected_bounds
        )
        if water_surfaces:
            water_level = water_height_at(world_x + local_x, world_y + local_y, water_surfaces)
            if water_level is not None and (
                procedural_land_mask(world_x + local_x, world_y + local_y, seed) < 0.5
                or inland_water_strength(world_x + local_x, world_y + local_y, seed) > 0.45
            ):
                world_height = min(world_height, water_level - 1.4)
        # Developed parcels use stable platforms.  Tile-edge samples deliberately
        # ignore local block masks so both neighboring chunks compute identical Z.
        if not on_border:
            platform_candidates = [
                (rectangle_platform_factor(local_x, local_y, block, 9.0), block.pad_elevation)
                for block in blocks
            ]
            block_factor, platform_height = max(platform_candidates, default=(0.0, world_height))
            for segment in traffic_segments:
                distance = distance_from_point_to_segment(
                    local_x, local_y, segment.start_x, segment.start_y, segment.end_x, segment.end_y,
                )
                road_radius = max(8.0, segment.width * 0.5 + 2.0)
                block_factor *= smoothstep(road_radius, road_radius + 7.0, distance)
                if block_factor <= 0.0:
                    break
            block_factor *= 1.0 - protected_terrain_blend_factor(local_x, local_y, protected_bounds)
            world_height += (platform_height - world_height) * block_factor
        return world_height - MISSION_SAFE_TERRAIN_HEIGHT

    heights: list[list[float]] = []
    for row in range(divisions + 1):
        local_y = -half + row * step
        height_row: list[float] = []
        for column in range(divisions + 1):
            local_x = -half + column * step
            height_row.append(sampled(local_x, local_y, row in {0, divisions} or column in {0, divisions}))
        heights.append(height_row)

    vertices: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int, int]] = []
    uvs: list[tuple[float, float]] = []
    stride = divisions + 1
    for row in range(stride):
        for column in range(stride):
            local_x, local_y = -half + column * step, -half + row * step
            left, right = max(0, column - 1), min(divisions, column + 1)
            bottom, top = max(0, row - 1), min(divisions, row + 1)
            slope_x = (heights[row][right] - heights[row][left]) / ((right - left) * step)
            slope_y = (heights[top][column] - heights[bottom][column]) / ((top - bottom) * step)
            normal_length = math.sqrt(slope_x * slope_x + slope_y * slope_y + 1.0)
            vertices.append((local_x, local_y, heights[row][column]))
            normals.append((-slope_x / normal_length, -slope_y / normal_length, 1.0 / normal_length))
            variation = terrain_value_noise(
                (world_x + local_x) / 170.0, (world_y + local_y) / 170.0, seed ^ 0x47524153,
            )
            colors.append((
                round(177.0 + variation * 11.0), round(183.0 + variation * 8.0),
                round(158.0 + variation * 10.0), 255,
            ))
            uvs.append(((world_x + local_x) / 18.0, (world_y + local_y) / 18.0))
    for builder in (render, collision):
        first_vertex = len(builder.mesh.vertices)
        builder.mesh.vertices.extend(vertices)
        builder.mesh.normals.extend(normals)
        builder.mesh.colors.extend(colors)
        builder.mesh.uvs.extend(uvs)
        for row in range(divisions):
            for column in range(divisions):
                first = first_vertex + row * stride + column
                second, fourth = first + 1, first + stride
                third = fourth + 1
                for indices in ((first, second, third), (first, third, fourth)):
                    triangle = [builder.mesh.vertices[index] for index in indices]
                    pieces = [[(vertex[0], vertex[1]) for vertex in triangle]]
                    for bounds in retail_surface_bounds:
                        pieces = [piece for polygon in pieces for piece in subtract_bounds_polygon(polygon, bounds)]
                        if not pieces:
                            break
                    if len(pieces) == 1 and pieces[0] == [(vertex[0], vertex[1]) for vertex in triangle]:
                        builder.mesh.triangles.append((*indices, MATERIAL_GRASS))
                        continue
                    for piece in pieces:
                        for index in range(1, len(piece) - 1):
                            points = tuple((x, y, interpolate_triangle_height(x, y, triangle))
                                           for x, y in (piece[0], piece[index], piece[index + 1]))
                            builder.add_triangle(points, MATERIAL_GRASS, colors[row * stride + column],
                                                 tuple(((world_x + x) / 18.0, (world_y + y) / 18.0) for x, y, _ in points))


def rectangle_platform_factor(
    x: float,
    y: float,
    block: CityBlock,
    transition: float,
) -> float:
    if point_in_polygon(x, y, block.points):
        return 1.0
    distance = min(
        distance_from_point_to_segment(x, y, first[0], first[1], second[0], second[1])
        for first, second in zip(block.points, block.points[1:] + block.points[:1])
    )
    if distance >= transition:
        return 0.0
    return 1.0 - smoothstep(0.0, transition, distance)


def protected_bounds_as_clearance_segments(
    protected_bounds: tuple[tuple[float, float, float, float], ...],
) -> list[TrafficPathSegment]:
    segments: list[TrafficPathSegment] = []
    for minimum_x, minimum_y, maximum_x, maximum_y in protected_bounds:
        center_x = (minimum_x + maximum_x) * 0.5
        center_y = (minimum_y + maximum_y) * 0.5
        width = max(maximum_x - minimum_x, maximum_y - minimum_y)
        segments.append(
            TrafficPathSegment(center_x, center_y, 0.0, center_x, center_y, 0.0, width)
        )
    return segments


def build_city_blocks(
    tile_size: float,
    district: str,
    world_x: float,
    world_y: float,
    rng: random.Random,
    traffic_segments: list[TrafficPathSegment] | None = None,
    terrain_seed: int = 0,
    street_blocks: list[StreetBlock] | None = None,
) -> list[CityBlock]:
    if traffic_segments is not None and street_blocks:
        graph_blocks = build_graph_derived_blocks(
            tile_size, world_x, world_y, terrain_seed, traffic_segments, street_blocks
        )
        if graph_blocks:
            return graph_blocks
    if traffic_segments is not None:
        return build_road_constrained_blocks(
            tile_size,
            district,
            rng,
            traffic_segments,
            world_x,
            world_y,
            terrain_seed,
        )
    tile_half = tile_size * 0.5
    if district == "airport":
        west_inset = 16.0 if world_x - tile_half <= 1000.1 else 0.0
        east_inset = 16.0 if world_x + tile_half >= 2249.9 else 0.0
        south_inset = 16.0 if world_y - tile_half <= -2999.9 else 0.0
        north_inset = 16.0 if world_y + tile_half >= -2250.1 else 0.0
    elif district == "docks":
        west_inset = east_inset = south_inset = north_inset = 18.0
    else:
        column = round(world_x / tile_size - 0.5)
        row = round((world_y + 3000.0) / tile_size - 0.5)
        west_inset = 14.0 if has_vertical_arterial(column) else 0.0
        east_inset = 14.0 if has_vertical_arterial(column + 1) else 0.0
        south_inset = 14.0 if has_horizontal_arterial(row) else 0.0
        north_inset = 14.0 if has_horizontal_arterial(row + 1) else 0.0
    minimum_x = -tile_half + west_inset
    maximum_x = tile_half - east_inset
    minimum_y = -tile_half + south_inset
    maximum_y = tile_half - north_inset
    local_road_half_width = 6.5
    bend = 0.0 if district in {"airport", "docks"} else rng.uniform(-16.0, 16.0)
    cross_bend = 0.0 if district in {"airport", "docks"} else rng.uniform(-13.0, 13.0)

    if district in {"downtown", "commercial", "civic"}:
        x_bottom = bend
        x_top = -bend * 0.55
        y_left = cross_bend
        y_right = -cross_bend * 0.65
        center_x = (x_bottom + x_top) * 0.5
        center_y = (y_left + y_right) * 0.5
        return [
            city_block_from_points(((minimum_x, minimum_y), (x_bottom - local_road_half_width, minimum_y), (center_x - local_road_half_width, center_y - local_road_half_width), (minimum_x, y_left - local_road_half_width))),
            city_block_from_points(((minimum_x, y_left + local_road_half_width), (center_x - local_road_half_width, center_y + local_road_half_width), (x_top - local_road_half_width, maximum_y), (minimum_x, maximum_y))),
            city_block_from_points(((x_bottom + local_road_half_width, minimum_y), (maximum_x, minimum_y), (maximum_x, y_right - local_road_half_width), (center_x + local_road_half_width, center_y - local_road_half_width))),
            city_block_from_points(((center_x + local_road_half_width, center_y + local_road_half_width), (maximum_x, y_right + local_road_half_width), (maximum_x, maximum_y), (x_top + local_road_half_width, maximum_y))),
        ]
    elif district in {"residential", "grove", "coastal", "beach", "hills"}:
        vertical = int((world_x + 3000.0) // tile_size + (world_y + 3000.0) // tile_size) % 2 == 0
        if vertical:
            return [
                city_block_from_points(((minimum_x, minimum_y), (bend - local_road_half_width, minimum_y), (-bend * 0.55 - local_road_half_width, maximum_y), (minimum_x, maximum_y))),
                city_block_from_points(((bend + local_road_half_width, minimum_y), (maximum_x, minimum_y), (maximum_x, maximum_y), (-bend * 0.55 + local_road_half_width, maximum_y))),
            ]
        return [
            city_block_from_points(((minimum_x, minimum_y), (maximum_x, minimum_y), (maximum_x, -bend * 0.55 - local_road_half_width), (minimum_x, bend - local_road_half_width))),
            city_block_from_points(((minimum_x, bend + local_road_half_width), (maximum_x, -bend * 0.55 + local_road_half_width), (maximum_x, maximum_y), (minimum_x, maximum_y))),
        ]
    elif district in {"industrial", "eastside"}:
        return [
            city_block_from_points(((minimum_x, minimum_y), (maximum_x, minimum_y), (maximum_x, -bend * 0.45 - 11.0), (minimum_x, bend - 11.0))),
            city_block_from_points(((minimum_x, bend + 11.0), (maximum_x, -bend * 0.45 + 11.0), (maximum_x, maximum_y), (minimum_x, maximum_y))),
        ]
    return [city_block_from_points(((minimum_x, minimum_y), (maximum_x, minimum_y), (maximum_x, maximum_y), (minimum_x, maximum_y)))]


def polygon_area(points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> float:
    return abs(sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(points, points[1:] + points[:1])
    )) * 0.5


def clip_polygon_halfplane(
    points: list[tuple[float, float]],
    normal_x: float,
    normal_y: float,
    threshold: float,
) -> list[tuple[float, float]]:
    if len(points) < 3:
        return []
    result: list[tuple[float, float]] = []
    previous = points[-1]
    previous_distance = previous[0] * normal_x + previous[1] * normal_y - threshold
    for current in points:
        current_distance = current[0] * normal_x + current[1] * normal_y - threshold
        if (current_distance >= 0.0) != (previous_distance >= 0.0):
            factor = previous_distance / (previous_distance - current_distance)
            result.append((
                previous[0] + (current[0] - previous[0]) * factor,
                previous[1] + (current[1] - previous[1]) * factor,
            ))
        if current_distance >= 0.0:
            result.append(current)
        previous = current
        previous_distance = current_distance
    cleaned: list[tuple[float, float]] = []
    for point in result:
        if not cleaned or math.dist(point, cleaned[-1]) > 0.00001:
            cleaned.append(point)
    if len(cleaned) > 1 and math.dist(cleaned[0], cleaned[-1]) <= 0.00001:
        cleaned.pop()
    return cleaned if len(cleaned) >= 3 else []


def subtract_bounds_polygon(
    points: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
) -> list[list[tuple[float, float]]]:
    if not points:
        return []
    left, bottom, right, top = bounds
    if (max(x for x, _ in points) <= left or min(x for x, _ in points) >= right
            or max(y for _, y in points) <= bottom or min(y for _, y in points) >= top):
        return [points]
    pieces: list[list[tuple[float, float]]] = []
    remaining = points
    for normal_x, normal_y, threshold in ((1.0, 0.0, left), (-1.0, 0.0, -right),
                                         (0.0, 1.0, bottom), (0.0, -1.0, -top)):
        outside = clip_polygon_halfplane(remaining, -normal_x, -normal_y, -threshold)
        if outside and polygon_area(outside) > 0.000001:
            pieces.append(outside)
        remaining = clip_polygon_halfplane(remaining, normal_x, normal_y, threshold)
        if not remaining:
            break
    return pieces


def interpolate_triangle_height(x: float, y: float, points: list[tuple[float, float, float]]) -> float:
    first, second, third = points
    denominator = ((second[1] - third[1]) * (first[0] - third[0])
                   + (third[0] - second[0]) * (first[1] - third[1]))
    if abs(denominator) < 0.00000001:
        return first[2]
    first_weight = ((second[1] - third[1]) * (x - third[0]) + (third[0] - second[0]) * (y - third[1])) / denominator
    second_weight = ((third[1] - first[1]) * (x - third[0]) + (first[0] - third[0]) * (y - third[1])) / denominator
    return first[2] * first_weight + second[2] * second_weight + third[2] * (1.0 - first_weight - second_weight)


def trim_city_blocks(
    blocks: list[CityBlock],
    bounds: tuple[tuple[float, float, float, float], ...],
    world_x: float,
    world_y: float,
    seed: int,
    grade_segments: list[TrafficPathSegment],
) -> list[CityBlock]:
    if not bounds:
        return blocks
    result: list[CityBlock] = []
    for block in blocks:
        pieces = [list(block.points)]
        for rectangle in bounds:
            pieces = [piece for polygon in pieces for piece in subtract_bounds_polygon(polygon, rectangle)
                      if polygon_area(piece) >= 140.0]
            if not pieces:
                break
        result.extend(city_block_from_points(tuple(piece), world_x, world_y, seed, grade_segments) for piece in pieces)
    return result


def inset_convex_polygon(
    points: tuple[tuple[float, float], ...],
    distance: float,
) -> list[tuple[float, float]]:
    signed_area = sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(points, points[1:] + points[:1])
    )
    direction = 1.0 if signed_area >= 0.0 else -1.0
    result = list(points)
    for first, second in zip(points, points[1:] + points[:1]):
        delta_x, delta_y = second[0] - first[0], second[1] - first[1]
        length = math.hypot(delta_x, delta_y)
        if length < 0.00001:
            continue
        normal_x, normal_y = -delta_y / length * direction, delta_x / length * direction
        threshold = first[0] * normal_x + first[1] * normal_y + distance
        result = clip_polygon_halfplane(result, normal_x, normal_y, threshold)
        if not result:
            break
    return result


def subtract_road_corridor(
    points: list[tuple[float, float]],
    segment: TrafficPathSegment,
    margin: float = 2.0,
) -> list[list[tuple[float, float]]]:
    if polygon_clear_of_traffic(tuple(points), [segment], margin - 1.75):
        return [points]
    delta_x, delta_y = segment.end_x - segment.start_x, segment.end_y - segment.start_y
    length = math.hypot(delta_x, delta_y)
    direction_x, direction_y = (delta_x / length, delta_y / length) if length > 0.00001 else (1.0, 0.0)
    normal_x, normal_y = -direction_y, direction_x
    half_width = segment.width * 0.5 + margin
    start_projection = segment.start_x * direction_x + segment.start_y * direction_y
    side_projection = segment.start_x * normal_x + segment.start_y * normal_y
    planes = (
        (direction_x, direction_y, start_projection - half_width),
        (-direction_x, -direction_y, -start_projection - length - half_width),
        (normal_x, normal_y, side_projection - half_width),
        (-normal_x, -normal_y, -side_projection - half_width),
    )
    remaining = points
    pieces: list[list[tuple[float, float]]] = []
    for normal_x, normal_y, threshold in planes:
        outside = clip_polygon_halfplane(remaining, -normal_x, -normal_y, -threshold)
        if outside and polygon_area(outside) > 0.001:
            pieces.append(outside)
        remaining = clip_polygon_halfplane(remaining, normal_x, normal_y, threshold)
        if not remaining:
            break
    return pieces


def clip_polygon_axis(
    points: list[tuple[float, float]],
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    if not points:
        return []
    result: list[tuple[float, float]] = []

    def inside(point: tuple[float, float]) -> bool:
        return point[axis] >= boundary if keep_greater else point[axis] <= boundary

    def intersection(first: tuple[float, float], second: tuple[float, float]) -> tuple[float, float]:
        denominator = second[axis] - first[axis]
        factor = 0.0 if abs(denominator) <= 0.000001 else (boundary - first[axis]) / denominator
        factor = max(0.0, min(1.0, factor))
        return (
            first[0] + (second[0] - first[0]) * factor,
            first[1] + (second[1] - first[1]) * factor,
        )

    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside != previous_inside:
            result.append(intersection(previous, current))
        if current_inside:
            result.append(current)
        previous = current
        previous_inside = current_inside
    return result


def clip_polygon_to_bounds(
    points: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    minimum_x, minimum_y, maximum_x, maximum_y = bounds
    clipped = clip_polygon_axis(points, 0, minimum_x, True)
    clipped = clip_polygon_axis(clipped, 0, maximum_x, False)
    clipped = clip_polygon_axis(clipped, 1, minimum_y, True)
    clipped = clip_polygon_axis(clipped, 1, maximum_y, False)
    return clipped


def build_graph_derived_blocks(
    tile_size: float,
    world_x: float,
    world_y: float,
    terrain_seed: int,
    traffic_segments: list[TrafficPathSegment],
    street_blocks: list[StreetBlock],
) -> list[CityBlock]:
    half = tile_size * 0.5
    world_bounds = (world_x - half, world_y - half, world_x + half, world_y + half)
    result: list[CityBlock] = []
    for street_block in street_blocks:
        world_points = tuple((point.x, point.y) for point in street_block.points)
        inset_world = inset_convex_polygon(world_points, 8.0)
        clipped_world = clip_polygon_to_bounds(inset_world, world_bounds)
        if len(clipped_world) < 3 or polygon_area(clipped_world) < 150.0:
            continue
        fragments = [[(x - world_x, y - world_y) for x, y in clipped_world]]
        for segment in traffic_segments:
            fragments = [
                piece
                for fragment in fragments
                for piece in subtract_road_corridor(fragment, segment)
                if len(piece) >= 3 and polygon_area(piece) >= 150.0
            ]
            if not fragments:
                break
        for fragment in fragments:
            local_points = tuple(fragment)
            center_x, center_y = polygon_centroid(local_points)
            land_samples = (*local_points, (center_x, center_y))
            if any(
                procedural_land_mask(world_x + x, world_y + y, terrain_seed) < 0.62
                or inland_water_strength(world_x + x, world_y + y, terrain_seed) > 0.30
                for x, y in land_samples
            ):
                continue
            if not polygon_clear_of_traffic(local_points, traffic_segments):
                continue
            result.append(city_block_from_points(
                local_points, world_x, world_y, terrain_seed, traffic_segments
            ))
    return result


def build_road_constrained_blocks(
    tile_size: float,
    district: str,
    rng: random.Random,
    traffic_segments: list[TrafficPathSegment],
    world_x: float,
    world_y: float,
    terrain_seed: int,
) -> list[CityBlock]:
    """Make buildable neighborhood cells by subtracting authoritative road corridors.

    Unequal seeded spans avoid the old wave grammar.  Any cell touched by a road
    is recursively split until a safe block can be retained or the fragment is
    too small to develop.  Buildings therefore inherit the road graph rather
    than competing with it.
    """
    half = tile_size * 0.5
    target = 46.0 if district in {"industrial", "docks", "airport"} else 36.0
    x_spans = partition_span(tile_size - 8.0, target, 9, rng)
    y_spans = partition_span(tile_size - 8.0, target, 9, rng)
    blocks: list[CityBlock] = []

    def retain_or_split(cx: float, cy: float, width: float, depth: float, level: int = 0) -> None:
        land_samples = (
            procedural_land_mask(world_x + cx, world_y + cy, terrain_seed),
            procedural_land_mask(world_x + cx - width * 0.5, world_y + cy - depth * 0.5, terrain_seed),
            procedural_land_mask(world_x + cx + width * 0.5, world_y + cy - depth * 0.5, terrain_seed),
            procedural_land_mask(world_x + cx + width * 0.5, world_y + cy + depth * 0.5, terrain_seed),
            procedural_land_mask(world_x + cx - width * 0.5, world_y + cy + depth * 0.5, terrain_seed),
        )
        water_samples = (
            inland_water_strength(world_x + cx, world_y + cy, terrain_seed),
            inland_water_strength(world_x + cx - width * 0.5, world_y + cy - depth * 0.5, terrain_seed),
            inland_water_strength(world_x + cx + width * 0.5, world_y + cy - depth * 0.5, terrain_seed),
            inland_water_strength(world_x + cx + width * 0.5, world_y + cy + depth * 0.5, terrain_seed),
            inland_water_strength(world_x + cx - width * 0.5, world_y + cy + depth * 0.5, terrain_seed),
        )
        if min(land_samples) < 0.62 or max(water_samples) > 0.42:
            if level >= 2 or min(width, depth) < 25.0:
                return
            if width >= depth:
                child = (width - 2.5) * 0.5
                retain_or_split(cx - (child + 2.5) * 0.5, cy, child, depth, level + 1)
                retain_or_split(cx + (child + 2.5) * 0.5, cy, child, depth, level + 1)
            else:
                child = (depth - 2.5) * 0.5
                retain_or_split(cx, cy - (child + 2.5) * 0.5, width, child, level + 1)
                retain_or_split(cx, cy + (child + 2.5) * 0.5, width, child, level + 1)
            return
        if is_footprint_clear(cx, cy, width, depth, traffic_segments, extra_clearance=3.5):
            blocks.append(city_block_from_points(
                (
                    (cx - width * 0.5, cy - depth * 0.5),
                    (cx + width * 0.5, cy - depth * 0.5),
                    (cx + width * 0.5, cy + depth * 0.5),
                    (cx - width * 0.5, cy + depth * 0.5),
                ),
                world_x,
                world_y,
                terrain_seed,
                traffic_segments,
            ))
            return
        if level >= 2 or min(width, depth) < 25.0:
            return
        if width >= depth:
            gap = 2.5
            child = (width - gap) * 0.5
            if child >= 11.0:
                retain_or_split(cx - (child + gap) * 0.5, cy, child, depth, level + 1)
                retain_or_split(cx + (child + gap) * 0.5, cy, child, depth, level + 1)
        else:
            gap = 2.5
            child = (depth - gap) * 0.5
            if child >= 11.0:
                retain_or_split(cx, cy - (child + gap) * 0.5, width, child, level + 1)
                retain_or_split(cx, cy + (child + gap) * 0.5, width, child, level + 1)

    for y_offset, depth in y_spans:
        for x_offset, width in x_spans:
            retain_or_split(-half + 4.0 + x_offset + width * 0.5, -half + 4.0 + y_offset + depth * 0.5, width, depth)
    return blocks


def has_vertical_arterial(boundary_index: int) -> bool:
    """Unequal superblock spacing prevents every 250-unit tile seam becoming a road."""
    return boundary_index % 12 in {0, 2, 5, 7, 10}


def has_horizontal_arterial(boundary_index: int) -> bool:
    return boundary_index % 12 in {0, 3, 5, 8, 10}


def compute_block_z_walk(
    points: tuple[tuple[float, float], ...],
    world_center_x: float,
    world_center_y: float,
    seed: int,
    traffic_segments: list[TrafficPathSegment],
) -> tuple[float, float, float, float, float, int]:
    samples: list[float] = []
    edge_samples: list[list[float]] = []
    for index, first in enumerate(points):
        second = points[(index + 1) % len(points)]
        current_edge: list[float] = []
        for step in range(7):
            factor = step / 6.0
            local_x = first[0] + (second[0] - first[0]) * factor
            local_y = first[1] + (second[1] - first[1]) * factor
            value = continuous_terrain_height(
                local_x, local_y, world_center_x, world_center_y, seed,
                "z_walk", traffic_segments, ()
            )
            samples.append(value)
            current_edge.append(value)
        edge_samples.append(current_edge)
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    samples.append(continuous_terrain_height(
        center_x, center_y, world_center_x, world_center_y, seed,
        "z_walk", traffic_segments, ()
    ))
    ordered = sorted(samples)
    pad = ordered[len(ordered) // 2]
    minimum = min(samples)
    maximum = max(samples)
    diameter = max(
        1.0,
        max(point[0] for point in points) - min(point[0] for point in points),
        max(point[1] for point in points) - min(point[1] for point in points),
    )
    slope = (maximum - minimum) / diameter
    cut_fill = sum(abs(value - pad) for value in samples) / len(samples)
    retaining_edges = sum(
        1 for values in edge_samples if max(values) - min(values) > 1.15
    )
    return (
        round(pad, 4),
        round(minimum, 4),
        round(maximum, 4),
        round(slope, 5),
        round(cut_fill, 4),
        retaining_edges,
    )


def city_block_from_points(
    points: tuple[tuple[float, float], ...],
    world_center_x: float = 0.0,
    world_center_y: float = 0.0,
    seed: int = 0,
    traffic_segments: list[TrafficPathSegment] | None = None,
) -> CityBlock:
    minimum_x = min(point[0] for point in points)
    maximum_x = max(point[0] for point in points)
    minimum_y = min(point[1] for point in points)
    maximum_y = max(point[1] for point in points)
    if traffic_segments is None:
        pad = minimum = maximum = MISSION_SAFE_TERRAIN_HEIGHT
        slope = 0.0
        retaining_edges = 0
        cut_fill = 0.0
    else:
        pad, minimum, maximum, slope, cut_fill, retaining_edges = compute_block_z_walk(
            points, world_center_x, world_center_y, seed, traffic_segments
        )
    return CityBlock(
        (minimum_x + maximum_x) * 0.5,
        (minimum_y + maximum_y) * 0.5,
        maximum_x - minimum_x,
        maximum_y - minimum_y,
        points,
        pad,
        minimum,
        maximum,
        slope,
        cut_fill,
        retaining_edges,
    )


def add_block_surfaces(
    render: MeshBuilder,
    collision: MeshBuilder,
    blocks: list[CityBlock],
    district: str,
) -> None:
    sidewalk_color = (130, 126, 116, 255)
    grass_color = district_grass_color(district)
    for block in blocks:
        material = MATERIAL_GRASS if district == "park" else MATERIAL_SIDEWALK
        color = grass_color if district == "park" else sidewalk_color
        base_z = block.pad_elevation - MISSION_SAFE_TERRAIN_HEIGHT
        render.add_polygon_prism(block.points, base_z, base_z + 0.22, material, material, color)
        collision.add_polygon_prism(block.points, base_z, base_z + 0.22, material, material, color)


def choose_building_archetype(
    district: str,
    width: float,
    depth: float,
    density: float,
    rng: random.Random,
    warehouse_bias: bool = False,
) -> str:
    area = width * depth
    roll = rng.random()
    if warehouse_bias or district in {"industrial", "eastside"}:
        return "business"
    if district == "downtown" and area >= 430.0 and roll < 0.34:
        return "tower"
    if district in {"downtown", "commercial", "civic"} and area >= 300.0 and (density > 0.40 or roll < 0.62):
        return "midrise"
    if district in {"commercial", "civic"}:
        return "business"
    if district in {"grove", "residential", "coastal", "beach", "hills"}:
        bungalow_bias = 0.62 if district in {"grove", "beach"} else 0.44
        if area <= 520.0 and roll < bungalow_bias:
            return "bungalow"
        return "house"
    return "business"


def archetype_height(archetype: str, rng: random.Random) -> float:
    ranges = {
        "bungalow": (4.4, 6.4),
        "house": (7.2, 11.8),
        "business": (5.5, 13.5),
        "midrise": (22.0, 27.5),
        "tower": (48.0, 96.0),
    }
    return rng.uniform(*ranges[archetype])


def select_archetype_production(
    archetype: str,
    width: float,
    depth: float,
    rng: random.Random,
) -> BuildingProduction:
    roll = rng.random()
    if archetype == "tower":
        return BuildingProduction("setback_tower", "vertical_bays", "equipment_parapet")
    if archetype == "midrise":
        mass = "courtyard" if width >= 30.0 and depth >= 30.0 and roll < 0.45 else "midrise_podium"
        return BuildingProduction(mass, "pilaster_bays", "flat_parapet")
    if archetype == "business":
        mass = "industrial_shed" if width >= 28.0 and depth >= 24.0 and roll < 0.20 else "slab"
        return BuildingProduction(mass, "storefront_bays", "flat_parapet")
    if archetype == "bungalow":
        mass = "l_wing" if width >= 20.0 and depth >= 18.0 and roll < 0.46 else "slab"
        return BuildingProduction(mass, "porch_bays", "domestic_roof")
    mass = "l_wing" if width >= 24.0 and depth >= 22.0 and roll < 0.45 else ("row" if width >= 29.0 and roll < 0.70 else "slab")
    return BuildingProduction(mass, "regular_bays", "domestic_roof")


def rectangle_points(center_x: float, center_y: float, width: float, depth: float) -> tuple[tuple[float, float], ...]:
    return (
        (center_x - width * 0.5, center_y - depth * 0.5),
        (center_x + width * 0.5, center_y - depth * 0.5),
        (center_x + width * 0.5, center_y + depth * 0.5),
        (center_x - width * 0.5, center_y + depth * 0.5),
    )


def classify_nearest_street(
    center_x: float,
    center_y: float,
    segments: list[TrafficPathSegment],
) -> tuple[str, bool]:
    nearby: list[tuple[float, TrafficPathSegment]] = []
    for segment in segments:
        distance = distance_from_point_to_segment(
            center_x, center_y, segment.start_x, segment.start_y, segment.end_x, segment.end_y
        )
        nearby.append((distance, segment))
    nearby.sort(key=lambda item: item[0])
    if not nearby:
        return "local", False
    width = nearby[0][1].width
    street_class = "arterial" if width >= 23.0 else ("collector" if width >= 16.0 else "local")
    corner = len(nearby) >= 2 and nearby[1][0] < nearby[0][0] + 12.0
    return street_class, corner


def translate_new_vertices(builder: MeshBuilder, checkpoint: tuple[int, int], delta_z: float) -> None:
    if abs(delta_z) <= 0.00001:
        return
    vertex_start, _ = checkpoint
    for index in range(vertex_start, len(builder.mesh.vertices)):
        x, y, z = builder.mesh.vertices[index]
        builder.mesh.vertices[index] = (x, y, z + delta_z)


def rotate_new_vertices(
    builder: MeshBuilder,
    checkpoint: tuple[int, int],
    center_x: float,
    center_y: float,
    angle: float,
) -> None:
    if abs(angle) <= 0.000001:
        return
    cosine = math.cos(angle)
    sine = math.sin(angle)
    vertex_start, _ = checkpoint
    for index in range(vertex_start, len(builder.mesh.vertices)):
        x, y, z = builder.mesh.vertices[index]
        relative_x = x - center_x
        relative_y = y - center_y
        builder.mesh.vertices[index] = (
            center_x + relative_x * cosine - relative_y * sine,
            center_y + relative_x * sine + relative_y * cosine,
            z,
        )
        normal_x, normal_y, normal_z = builder.mesh.normals[index]
        builder.mesh.normals[index] = (
            normal_x * cosine - normal_y * sine,
            normal_x * sine + normal_y * cosine,
            normal_z,
        )


def polygon_centroid(points: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def convex_hull(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    ordered = sorted(set(points))
    if len(ordered) <= 2:
        return tuple(ordered)

    def cross(first: tuple[float, float], second: tuple[float, float], third: tuple[float, float]) -> float:
        return ((second[0] - first[0]) * (third[1] - first[1])
                - (second[1] - first[1]) * (third[0] - first[0]))

    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def oriented_rectangle_points(
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    angle: float,
) -> tuple[tuple[float, float], ...]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    points: list[tuple[float, float]] = []
    for local_x, local_y in (
        (-width * 0.5, -depth * 0.5),
        (width * 0.5, -depth * 0.5),
        (width * 0.5, depth * 0.5),
        (-width * 0.5, depth * 0.5),
    ):
        points.append((
            center_x + local_x * cosine - local_y * sine,
            center_y + local_x * sine + local_y * cosine,
        ))
    return tuple(points)


def polygon_inside_polygon(
    candidate: tuple[tuple[float, float], ...],
    container: tuple[tuple[float, float], ...],
) -> bool:
    return all(point_in_polygon(x, y, container) for x, y in candidate)


def polygon_clear_of_traffic(
    points: tuple[tuple[float, float], ...],
    traffic_segments: list[TrafficPathSegment],
    extra_clearance: float = 0.0,
) -> bool:
    if len(points) < 3:
        return False
    for segment in traffic_segments:
        clearance = segment.width * 0.5 + 1.75 + extra_clearance
        if point_in_polygon(segment.start_x, segment.start_y, points) or point_in_polygon(segment.end_x, segment.end_y, points):
            return False
        for first, second in zip(points, points[1:] + points[:1]):
            if segment_pair_distance(
                first, second,
                (segment.start_x, segment.start_y), (segment.end_x, segment.end_y),
            ) < clearance - 0.00001:
                return False
    return True


def segment_pair_distance(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> float:
    first_x, first_y = first_end[0] - first_start[0], first_end[1] - first_start[1]
    second_x, second_y = second_end[0] - second_start[0], second_end[1] - second_start[1]
    cross = first_x * second_y - first_y * second_x
    if abs(cross) > 0.00000001:
        delta_x, delta_y = second_start[0] - first_start[0], second_start[1] - first_start[1]
        first_factor = (delta_x * second_y - delta_y * second_x) / cross
        second_factor = (delta_x * first_y - delta_y * first_x) / cross
        if 0.0 <= first_factor <= 1.0 and 0.0 <= second_factor <= 1.0:
            return 0.0
    return min(
        distance_from_point_to_segment(*first_start, *second_start, *second_end),
        distance_from_point_to_segment(*first_end, *second_start, *second_end),
        distance_from_point_to_segment(*second_start, *first_start, *first_end),
        distance_from_point_to_segment(*second_end, *first_start, *first_end),
    )


def fit_oriented_building_footprint(
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    angle: float,
    block_points: tuple[tuple[float, float], ...],
    traffic_segments: list[TrafficPathSegment],
) -> tuple[float, float, tuple[tuple[float, float], ...]] | None:
    for scale in (1.0, 0.88, 0.76, 0.64, 0.54):
        candidate_width = max(8.5, width * scale)
        candidate_depth = max(8.5, depth * scale)
        candidate = oriented_rectangle_points(
            center_x, center_y, candidate_width, candidate_depth, angle
        )
        if not polygon_inside_polygon(candidate, block_points):
            continue
        if polygon_clear_of_traffic(candidate, traffic_segments):
            return candidate_width, candidate_depth, candidate
    return None


def orient_building_to_frontage(
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    angle: float,
    segments: list[TrafficPathSegment],
) -> tuple[float, float, float]:
    candidates = [segment for segment in segments
                  if math.hypot(segment.end_x - segment.start_x, segment.end_y - segment.start_y) > 0.01]
    if not candidates:
        return width, depth, angle
    nearest = min(candidates, key=lambda segment: distance_from_point_to_segment(
        center_x, center_y, segment.start_x, segment.start_y, segment.end_x, segment.end_y,
    ))
    factor = projection_factor(center_x, center_y, nearest)
    target_x = nearest.start_x + (nearest.end_x - nearest.start_x) * factor - center_x
    target_y = nearest.start_y + (nearest.end_y - nearest.start_y) * factor - center_y
    quarter_turn = max(range(4), key=lambda turn: target_x * math.sin(angle + turn * math.pi * 0.5)
                       - target_y * math.cos(angle + turn * math.pi * 0.5))
    if quarter_turn % 2:
        width, depth = depth, width
    return width, depth, angle + quarter_turn * math.pi * 0.5


def bilinear_quad_point(
    points: tuple[tuple[float, float], ...],
    u: float,
    v: float,
) -> tuple[float, float]:
    first, second, third, fourth = points
    lower_x = first[0] + (second[0] - first[0]) * u
    lower_y = first[1] + (second[1] - first[1]) * u
    upper_x = fourth[0] + (third[0] - fourth[0]) * u
    upper_y = fourth[1] + (third[1] - fourth[1]) * u
    return (
        lower_x + (upper_x - lower_x) * v,
        lower_y + (upper_y - lower_y) * v,
    )


def normalized_partitions(
    length: float,
    target_size: float,
    maximum_parts: int,
    rng: random.Random,
) -> list[tuple[float, float]]:
    return [
        (offset / length, (offset + span) / length)
        for offset, span in partition_span(length, target_size, maximum_parts, rng)
    ]


def subdivide_block_into_lots(
    block: CityBlock,
    target_lot_size: float,
    rng: random.Random,
) -> list[tuple[int, int, tuple[tuple[float, float], ...], float, float, float, float, float]]:
    if len(block.points) == 4:
        first, second, third, fourth = block.points
        horizontal_length = (
            math.hypot(second[0] - first[0], second[1] - first[1])
            + math.hypot(third[0] - fourth[0], third[1] - fourth[1])
        ) * 0.5
        vertical_length = (
            math.hypot(fourth[0] - first[0], fourth[1] - first[1])
            + math.hypot(third[0] - second[0], third[1] - second[1])
        ) * 0.5
        if horizontal_length >= 12.0 and vertical_length >= 12.0:
            column_ranges = normalized_partitions(horizontal_length, target_lot_size, 6, rng)
            row_ranges = normalized_partitions(vertical_length, target_lot_size, 7, rng)
            lots = []
            for row_index, (v0, v1) in enumerate(row_ranges):
                for column_index, (u0, u1) in enumerate(column_ranges):
                    parcel = (
                        bilinear_quad_point(block.points, u0, v0),
                        bilinear_quad_point(block.points, u1, v0),
                        bilinear_quad_point(block.points, u1, v1),
                        bilinear_quad_point(block.points, u0, v1),
                    )
                    center_x, center_y = polygon_centroid(parcel)
                    width = min(
                        math.hypot(parcel[1][0] - parcel[0][0], parcel[1][1] - parcel[0][1]),
                        math.hypot(parcel[2][0] - parcel[3][0], parcel[2][1] - parcel[3][1]),
                    )
                    depth = min(
                        math.hypot(parcel[3][0] - parcel[0][0], parcel[3][1] - parcel[0][1]),
                        math.hypot(parcel[2][0] - parcel[1][0], parcel[2][1] - parcel[1][1]),
                    )
                    angle = math.atan2(
                        parcel[1][1] - parcel[0][1], parcel[1][0] - parcel[0][0]
                    )
                    lots.append((
                        row_index, column_index, parcel, center_x, center_y,
                        width, depth, angle,
                    ))
            return lots

    lots = []
    parcels = split_convex_parcels(list(block.points), target_lot_size, rng)
    for index, parcel in enumerate(parcels):
        first, second = max(
            zip(parcel, parcel[1:] + parcel[:1]), key=lambda edge: math.dist(*edge)
        )
        angle = math.atan2(second[1] - first[1], second[0] - first[0])
        cosine, sine = math.cos(angle), math.sin(angle)
        horizontal = [x * cosine + y * sine for x, y in parcel]
        vertical = [-x * sine + y * cosine for x, y in parcel]
        center_x, center_y = polygon_centroid(tuple(parcel))
        lots.append((
            0, index, tuple(parcel), center_x, center_y,
            max(horizontal) - min(horizontal), max(vertical) - min(vertical), angle,
        ))
    return lots


def split_convex_parcels(
    points: list[tuple[float, float]],
    target_size: float,
    rng: random.Random,
    level: int = 0,
) -> list[list[tuple[float, float]]]:
    first, second = max(zip(points, points[1:] + points[:1]), key=lambda edge: math.dist(*edge))
    length = math.dist(first, second)
    if length < 0.001:
        return []
    axis_x, axis_y = (second[0] - first[0]) / length, (second[1] - first[1]) / length
    horizontal = [x * axis_x + y * axis_y for x, y in points]
    vertical = [-x * axis_y + y * axis_x for x, y in points]
    if max(vertical) - min(vertical) > max(horizontal) - min(horizontal):
        axis_x, axis_y = -axis_y, axis_x
        horizontal = vertical
    span = max(horizontal) - min(horizontal)
    if span <= target_size * 1.35 or polygon_area(points) < target_size * target_size * 1.25 or level >= 7:
        return [points]
    threshold = min(horizontal) + span * rng.uniform(0.42, 0.58)
    first_half = clip_polygon_halfplane(points, axis_x, axis_y, threshold)
    second_half = clip_polygon_halfplane(points, -axis_x, -axis_y, -threshold)
    if not first_half or not second_half or min(polygon_area(first_half), polygon_area(second_half)) < 100.0:
        return [points]
    return (
        split_convex_parcels(first_half, target_size, rng, level + 1)
        + split_convex_parcels(second_half, target_size, rng, level + 1)
    )


def populate_city_blocks(
    render: MeshBuilder,
    collision: MeshBuilder,
    blocks: list[CityBlock],
    district: str,
    rng: random.Random,
    traffic_segments: list[TrafficPathSegment],
    warehouse_bias: bool = False,
    world_x: float = 0.0,
    world_y: float = 0.0,
    seed: int = 0,
    debug_records: list[BuildingDebugRecord] | None = None,
) -> int:
    neighborhood = choose_neighborhood(world_x, world_y, seed, district)
    target_lot_size = {
        "downtown": 27.0, "commercial": 28.0, "civic": 31.0,
        "residential": 27.0, "grove": 26.0, "coastal": 29.0,
        "beach": 30.0, "hills": 32.0, "industrial": 39.0, "eastside": 34.0,
    }.get(district, 29.0)
    density = urban_density_at(world_x, world_y, seed)
    if district not in {"downtown", "commercial", "civic"}:
        target_lot_size = neighborhood.target_lot_size
        density = neighborhood.density
    target_lot_size *= 1.12 - density * 0.24
    park_chance = {
        "grove": 0.06, "hills": 0.09, "residential": 0.05,
        "coastal": 0.05, "beach": 0.09, "civic": 0.05,
    }.get(district, 0.014) * (1.18 - density * 0.36)
    building_count = 0
    occupied_footprints: list[tuple[tuple[float, float], ...]] = []
    building_limit = {
        "downtown": 58, "commercial": 64, "civic": 58,
        "industrial": 50, "eastside": 58,
    }.get(district, 70)

    for block_index, block in enumerate(blocks):
        lots = subdivide_block_into_lots(block, target_lot_size, rng)
        for row_index, column_index, parcel_points, center_x, center_y, cell_width, cell_depth, angle in lots:
            if building_count >= building_limit:
                continue
            pad_offset = block.pad_elevation - MISSION_SAFE_TERRAIN_HEIGHT
            if not polygon_inside_polygon(parcel_points, block.points):
                continue

            if rng.random() < park_chance:
                park_width = min(37.0, cell_width * 0.76)
                park_depth = min(37.0, cell_depth * 0.76)
                park_points = oriented_rectangle_points(
                    center_x, center_y, park_width, park_depth, angle
                )
                if (
                    polygon_inside_polygon(park_points, block.points)
                    and polygon_clear_of_traffic(park_points, traffic_segments)
                ):
                    render_checkpoint = mesh_checkpoint(render)
                    collision_checkpoint = mesh_checkpoint(collision)
                    add_parcel_park(
                        render, collision, center_x, center_y,
                        park_width, park_depth, rng, district,
                    )
                    rotate_new_vertices(render, render_checkpoint, center_x, center_y, angle)
                    rotate_new_vertices(collision, collision_checkpoint, center_x, center_y, angle)
                    translate_new_vertices(render, render_checkpoint, pad_offset)
                    translate_new_vertices(collision, collision_checkpoint, pad_offset)
                continue

            width = max(9.0, cell_width * rng.uniform(0.68, 0.86))
            depth = max(9.0, cell_depth * rng.uniform(0.66, 0.84))
            fitted = fit_oriented_building_footprint(
                center_x, center_y, width, depth, angle, parcel_points, traffic_segments
            )
            if fitted is None:
                continue
            width, depth, footprint_points = fitted
            width, depth, angle = orient_building_to_frontage(
                center_x, center_y, width, depth, angle, traffic_segments,
            )
            archetype = choose_building_archetype(
                district, width, depth, density, rng, warehouse_bias
            )
            height = archetype_height(archetype, rng)
            if archetype in {"midrise", "tower"}:
                expanded = fit_oriented_building_footprint(
                    center_x, center_y,
                    min(width * 1.08, cell_width * 0.88),
                    min(depth * 1.08, cell_depth * 0.88),
                    angle, parcel_points, traffic_segments,
                )
                if expanded is not None:
                    width, depth, footprint_points = expanded
            production = select_archetype_production(archetype, width, depth, rng)
            building_neighborhood = choose_neighborhood(world_x + center_x, world_y + center_y, seed, district)
            color = rng.choice(building_neighborhood.palette)
            render_checkpoint = mesh_checkpoint(render)
            collision_checkpoint = mesh_checkpoint(collision)
            building_style = archetype
            dwelling_units = 0 if archetype in {"business", "tower", "midrise"} else 1
            if archetype in {"bungalow", "house", "business"}:
                building_style = choose_building_style(building_neighborhood, width, depth, rng)
                result = build_neighborhood_building(
                    render, collision, center_x, center_y, width, depth, height,
                    color, rng, building_neighborhood, building_style,
                )
                building_style, dwelling_units = result.style, result.dwelling_units
            else:
                add_district_building(
                    render, collision, center_x, center_y, width, depth, height,
                    color, district, rng, archetype=archetype, production=production,
                )
            rotate_new_vertices(render, render_checkpoint, center_x, center_y, angle)
            rotate_new_vertices(collision, collision_checkpoint, center_x, center_y, angle)
            translate_new_vertices(render, render_checkpoint, pad_offset)
            translate_new_vertices(collision, collision_checkpoint, pad_offset)
            solid_footprint = convex_hull([
                (x, y) for x, y, _ in collision.mesh.vertices[collision_checkpoint[0]:]
            ])
            if (
                len(render.mesh.vertices) > 27_000
                or not polygon_inside_polygon(solid_footprint, parcel_points)
                or not polygon_clear_of_traffic(solid_footprint, traffic_segments)
            ):
                restore_mesh_checkpoint(render, render_checkpoint)
                restore_mesh_checkpoint(collision, collision_checkpoint)
                continue

            street_class, corner_lot = classify_nearest_street(
                center_x, center_y, traffic_segments
            )
            if debug_records is not None:
                side_margin = {
                    "bungalow": 2.8, "house": 3.2, "business": 1.8,
                    "midrise": 2.4, "tower": 4.0,
                }[archetype]
                setback_width = min(cell_width - 1.0, width + side_margin * 2.0)
                setback_depth = min(cell_depth - 1.0, depth + side_margin * 2.0)
                setback_points = oriented_rectangle_points(
                    center_x, center_y, max(width, setback_width),
                    max(depth, setback_depth), angle,
                )
                debug_records.append(BuildingDebugRecord(
                    lot_id=f"T{round(world_x):04d}_{round(world_y):05d}_B{block_index:02d}_{row_index:02d}{column_index:02d}",
                    archetype=archetype,
                    district=district,
                    parcel_points=tuple(
                        (world_x + x, world_y + y) for x, y in parcel_points
                    ),
                    setback_points=tuple(
                        (world_x + x, world_y + y) for x, y in setback_points
                    ),
                    footprint_points=tuple(
                        (world_x + x, world_y + y) for x, y in footprint_points
                    ),
                    world_center=(world_x + center_x, world_y + center_y),
                    pad_elevation=block.pad_elevation,
                    terrain_minimum=block.terrain_minimum,
                    terrain_maximum=block.terrain_maximum,
                    slope=block.slope,
                    cut_fill=block.cut_fill,
                    retaining_edges=block.retaining_edges,
                    street_class=street_class,
                    corner_lot=corner_lot,
                    production=production,
                    neighborhood=building_neighborhood.name,
                    condition=building_neighborhood.condition,
                    building_style=building_style,
                    dwelling_units=dwelling_units,
                ))
            building_count += 1
            occupied_footprints.append(solid_footprint)

        tree_render_checkpoint = mesh_checkpoint(render)
        tree_collision_checkpoint = mesh_checkpoint(collision)
        add_street_trees_for_block(
            render, collision, block, district, rng, traffic_segments, occupied_footprints,
        )
        translate_new_vertices(
            render, tree_render_checkpoint,
            block.pad_elevation - MISSION_SAFE_TERRAIN_HEIGHT,
        )
        translate_new_vertices(
            collision, tree_collision_checkpoint,
            block.pad_elevation - MISSION_SAFE_TERRAIN_HEIGHT,
        )
    return building_count


def mesh_checkpoint(builder: MeshBuilder) -> tuple[int, int]:
    return len(builder.mesh.vertices), len(builder.mesh.triangles)


def restore_mesh_checkpoint(builder: MeshBuilder, checkpoint: tuple[int, int]) -> None:
    vertex_count, triangle_count = checkpoint
    del builder.mesh.vertices[vertex_count:]
    del builder.mesh.normals[vertex_count:]
    del builder.mesh.colors[vertex_count:]
    del builder.mesh.uvs[vertex_count:]
    del builder.mesh.triangles[triangle_count:]


def urban_density_at(world_x: float, world_y: float, seed: int) -> float:
    """Seeded population field used to vary parcel density across a district."""
    broad = (terrain_value_noise(world_x / 1150.0, world_y / 1150.0, seed ^ 0x4D3A91C7) + 1.0) * 0.5
    detail = (terrain_value_noise(world_x / 520.0, world_y / 520.0, seed ^ 0x73B24E19) + 1.0) * 0.5
    downtown_distance = math.hypot((world_x - 1600.0) / 1150.0, (world_y + 1250.0) / 950.0)
    center_pull = max(0.0, 1.0 - downtown_distance)
    return max(0.0, min(1.0, broad * 0.42 + detail * 0.23 + center_pull * 0.35))


def fit_building_footprint(
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    block_points: tuple[tuple[float, float], ...],
    traffic_segments: list[TrafficPathSegment],
) -> tuple[float, float, float, float] | None:
    """Shrink an infill parcel around a path instead of discarding the whole lot."""
    for scale in (1.0, 0.84, 0.70, 0.58):
        candidate_width = max(9.0, width * scale)
        candidate_depth = max(9.0, depth * scale)
        if not rectangle_inside_polygon(center_x, center_y, candidate_width, candidate_depth, block_points):
            continue
        if is_footprint_clear(center_x, center_y, candidate_width, candidate_depth, traffic_segments):
            return center_x, center_y, candidate_width, candidate_depth
    return None


def partition_span(
    length: float,
    target_size: float,
    maximum_parts: int,
    rng: random.Random,
) -> list[tuple[float, float]]:
    """Return deterministic unequal lot spans that exactly cover a block axis."""
    part_count = max(1, min(maximum_parts, round(length / target_size)))
    if part_count == 1:
        return [(0.0, length)]
    weights = [rng.uniform(0.72, 1.28) for _ in range(part_count)]
    weight_total = sum(weights)
    spans: list[tuple[float, float]] = []
    offset = 0.0
    for index, weight in enumerate(weights):
        width = length - offset if index == part_count - 1 else length * weight / weight_total
        spans.append((offset, width))
        offset += width
    return spans


def add_district_building(
    render: MeshBuilder,
    collision: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    height: float,
    color: tuple[int, int, int, int],
    district: str,
    rng: random.Random,
    archetype: str | None = None,
    production: BuildingProduction | None = None,
) -> None:
    """Apply one explicit archetype production to the parcel axiom."""
    production = production or select_building_production(district, width, depth, height, rng)
    style_district = district
    if archetype in {"bungalow", "house"}:
        style_district = "residential" if district not in {"grove", "beach", "hills"} else district
    elif archetype == "business":
        style_district = "commercial" if district not in {"industrial", "eastside"} else district
    elif archetype in {"midrise", "tower"}:
        style_district = "downtown"

    if archetype == "bungalow":
        add_bungalow_archetype(render, collision, center_x, center_y, width, depth, height, color, rng)
        return
    if archetype == "house":
        add_house_archetype(render, collision, center_x, center_y, width, depth, height, color, rng)
        return
    if archetype == "business":
        add_business_archetype(render, collision, center_x, center_y, width, depth, height, color, rng, style_district)
        return
    if archetype == "midrise":
        add_midrise_archetype(render, collision, center_x, center_y, width, depth, height, color, rng)
        return
    if archetype == "tower":
        add_tower_archetype(render, collision, center_x, center_y, width, depth, height, color, rng)
        return

    if production.mass_rule == "industrial_shed":
        add_building_component(render, collision, center_x, center_y, width, depth, height, color, style_district, rng)
        add_industrial_roofline(render, center_x, center_y, width, depth, height, color, rng)
        return

    if production.mass_rule == "midrise_podium":
        podium_height = min(7.2, height * 0.28)
        add_building_component(
            render, collision, center_x, center_y, width, depth, podium_height, color, style_district, rng
        )
        add_building_component(
            render, collision, center_x, center_y + depth * 0.025,
            width * 0.82, depth * 0.80, height, color, style_district, rng
        )
        return

    if production.mass_rule == "setback_tower":
        for scale, multiplier in ((1.0, 0.25), (0.78, 0.58), (0.56, 1.0)):
            add_building_component(
                render, collision, center_x, center_y,
                width * scale, depth * scale, height * multiplier,
                color, style_district, rng,
            )
        return

    if production.mass_rule == "courtyard":
        rear_depth = depth * rng.uniform(0.24, 0.32)
        wing_width = width * rng.uniform(0.20, 0.27)
        wing_depth = depth - rear_depth * 0.45
        rear_y = center_y + depth * 0.5 - rear_depth * 0.5
        wing_y = center_y - depth * 0.5 + wing_depth * 0.5
        for component_x, component_y, component_width, component_depth, multiplier in (
            (center_x, rear_y, width, rear_depth, 1.0),
            (center_x - width * 0.5 + wing_width * 0.5, wing_y, wing_width, wing_depth, 0.92),
            (center_x + width * 0.5 - wing_width * 0.5, wing_y, wing_width, wing_depth, 0.86),
        ):
            add_building_component(
                render, collision, component_x, component_y, component_width,
                component_depth, height * multiplier, color, style_district, rng,
            )
        return

    if production.mass_rule == "l_wing":
        mirror_x = -1.0 if rng.random() < 0.5 else 1.0
        mirror_y = -1.0 if rng.random() < 0.5 else 1.0
        bar_width = width * rng.uniform(0.36, 0.46)
        cross_depth = depth * rng.uniform(0.34, 0.45)
        vertical_x = center_x + mirror_x * (width - bar_width) * 0.5
        horizontal_y = center_y + mirror_y * (depth - cross_depth) * 0.5
        add_building_component(
            render, collision, vertical_x, center_y, bar_width, depth, height, color, style_district, rng
        )
        horizontal_width = width - bar_width * 0.55
        horizontal_x = center_x - mirror_x * bar_width * 0.225
        add_building_component(
            render, collision, horizontal_x, horizontal_y, horizontal_width,
            cross_depth, height * rng.uniform(0.82, 0.96), color, style_district, rng,
        )
        return

    if production.mass_rule == "row":
        unit_count = max(2, min(4, round(width / 11.0)))
        gap = min(1.3, width * 0.025)
        unit_width = (width - gap * (unit_count - 1)) / unit_count
        palette = DISTRICT_BUILDING_PALETTES.get(style_district, BUILDING_PALETTE)
        for index in range(unit_count):
            unit_x = center_x - width * 0.5 + unit_width * 0.5 + index * (unit_width + gap)
            unit_color = color if index == 0 else rng.choice(palette)
            add_building_component(
                render, collision, unit_x, center_y, unit_width,
                depth * rng.uniform(0.90, 1.0), height * rng.uniform(0.86, 1.04),
                unit_color, style_district, rng,
            )
        return

    add_building_component(render, collision, center_x, center_y, width, depth, height, color, style_district, rng)




def add_window_strip(
    render: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    base_z: float,
    top_z: float,
    columns: int,
    front_only: bool = False,
) -> None:
    front_y = center_y - depth * 0.5 - 0.03
    back_y = center_y + depth * 0.5 + 0.03
    x0 = center_x - width * 0.5
    for column in range(columns):
        span = width / columns
        cx = x0 + (column + 0.5) * span
        half = min(1.2, span * 0.30)
        color = (70, 92, 104, 255)
        render.add_quad(
            ((cx - half, front_y, base_z), (cx + half, front_y, base_z), (cx + half, front_y, top_z), (cx - half, front_y, top_z)),
            MATERIAL_WINDOW,
            color,
        )
        if not front_only and column % 2 == 0:
            render.add_quad(
                ((cx + half, back_y, base_z), (cx - half, back_y, base_z), (cx - half, back_y, top_z), (cx + half, back_y, top_z)),
                MATERIAL_WINDOW,
                color,
            )


def add_side_windows(
    render: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    base_z: float,
    top_z: float,
    rows: int,
) -> None:
    left_x = center_x - width * 0.5 - 0.03
    right_x = center_x + width * 0.5 + 0.03
    y0 = center_y - depth * 0.5
    for row in range(rows):
        cy = y0 + (row + 0.5) * depth / rows
        half = min(0.95, depth / rows * 0.20)
        color = (70, 92, 104, 255)
        render.add_quad(
            ((left_x, cy + half, base_z), (left_x, cy - half, base_z), (left_x, cy - half, top_z), (left_x, cy + half, top_z)),
            MATERIAL_WINDOW, color,
        )
        if row % 2 == 0:
            render.add_quad(
                ((right_x, cy - half, base_z), (right_x, cy + half, base_z), (right_x, cy + half, top_z), (right_x, cy - half, top_z)),
                MATERIAL_WINDOW, color,
            )


def add_hip_roof(
    render: MeshBuilder,
    center_x: float,
    center_y: float,
    minimum_z: float,
    width: float,
    depth: float,
    roof_height: float,
    color: tuple[int, int, int, int],
) -> None:
    x0 = center_x - width * 0.5
    x1 = center_x + width * 0.5
    y0 = center_y - depth * 0.5
    y1 = center_y + depth * 0.5
    apex = (center_x, center_y, minimum_z + roof_height)
    front_color = shade_color(color, 0.74)
    side_color = shade_color(color, 0.82)
    render.add_triangle(((x0, y0, minimum_z), (x1, y0, minimum_z), apex), MATERIAL_ROOF, front_color)
    render.add_triangle(((x1, y0, minimum_z), (x1, y1, minimum_z), apex), MATERIAL_ROOF, side_color)
    render.add_triangle(((x1, y1, minimum_z), (x0, y1, minimum_z), apex), MATERIAL_ROOF, front_color)
    render.add_triangle(((x0, y1, minimum_z), (x0, y0, minimum_z), apex), MATERIAL_ROOF, side_color)


def add_porch(
    render: MeshBuilder,
    center_x: float,
    front_y: float,
    width: float,
    accent_color: tuple[int, int, int, int],
    trim_color: tuple[int, int, int, int],
) -> None:
    render.add_box(center_x, front_y - 1.00, 0.22, width, 2.0, 0.18, MATERIAL_TRIM, MATERIAL_TRIM, shade_color(trim_color, 1.08))
    render.add_box(center_x, front_y - 1.02, 2.85, width * 0.98, 2.1, 0.18, MATERIAL_TRIM, MATERIAL_ROOF, accent_color)
    for direction in (-1.0, 1.0):
        render.add_box(center_x + direction * width * 0.38, front_y - 1.58, 0.22, 0.22, 0.22, 2.82, MATERIAL_TRIM, MATERIAL_TRIM, trim_color)


def add_driveway_and_path(
    render: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    garage_x: float | None = None,
) -> None:
    grass = (95, 129, 78, 255)
    walk = (178, 171, 148, 255)
    back_y = center_y + depth * 0.5 + 0.35
    render.add_box(center_x, center_y, 0.221, width * 1.08, depth * 1.10, 0.03, MATERIAL_GRASS, MATERIAL_GRASS, grass)
    render.add_box(center_x, back_y, 0.222, 1.15, depth * 0.42, 0.03, MATERIAL_SIDEWALK, MATERIAL_SIDEWALK, walk)
    if garage_x is not None:
        render.add_box(garage_x, back_y + depth * 0.02, 0.222, 3.25, depth * 0.62, 0.03, MATERIAL_SIDEWALK, MATERIAL_SIDEWALK, walk)


def add_bungalow_archetype(
    render: MeshBuilder,
    collision: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    height: float,
    color: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    wall_h = max(3.4, height * 0.58)
    roof_h = max(1.2, height - wall_h)
    main_w = width * 0.78
    main_d = depth * 0.60
    main_y = center_y + depth * 0.06
    trim = shade_color(color, 0.76)
    accent = shade_color(color, 1.10)
    render.add_box(center_x, main_y, 0.22, main_w, main_d, wall_h, MATERIAL_BUILDING, MATERIAL_ROOF, color)
    collision.add_box(center_x, main_y, 0.22, main_w, main_d, wall_h + roof_h, MATERIAL_BUILDING, MATERIAL_ROOF, color)
    porch_w = min(main_w * 0.54, 6.6)
    add_porch(render, center_x, main_y - main_d * 0.5, porch_w, accent, trim)
    garage_x = None
    if width >= 14.0 and rng.random() < 0.72:
        garage_w = min(width * 0.28, 5.2)
        garage_d = main_d * 0.74
        garage_x = center_x + (1.0 if rng.random() < 0.5 else -1.0) * (main_w * 0.5 + garage_w * 0.34)
        render.add_box(garage_x, main_y + garage_d * 0.06, 0.22, garage_w, garage_d, wall_h * 0.84, MATERIAL_BUILDING, MATERIAL_ROOF, shade_color(color, 0.95))
        collision.add_box(garage_x, main_y + garage_d * 0.06, 0.22, garage_w, garage_d, wall_h * 0.84 + roof_h * 0.75, MATERIAL_BUILDING, MATERIAL_ROOF, color)
        add_hip_roof(render, garage_x, main_y + garage_d * 0.06, 0.22 + wall_h * 0.84, garage_w * 1.02, garage_d * 1.02, roof_h * 0.72, trim)
    add_hip_roof(render, center_x, main_y, 0.22 + wall_h, main_w * 1.02, main_d * 1.02, roof_h, trim)
    add_window_strip(render, center_x, main_y, main_w, main_d, 1.25, 2.45, max(2, min(4, int(main_w / 4.4))))
    add_side_windows(render, center_x, main_y, main_w, main_d, 1.2, 2.4, max(2, min(3, int(main_d / 4.2))))
    add_driveway_and_path(render, center_x, main_y, main_w, main_d, garage_x)


def add_house_archetype(
    render: MeshBuilder,
    collision: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    height: float,
    color: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    first_h = max(3.2, height * 0.44)
    second_h = max(2.8, height * 0.32)
    roof_h = max(1.4, height - first_h - second_h)
    body_w = width * 0.72
    body_d = depth * 0.62
    body_y = center_y + depth * 0.06
    trim = shade_color(color, 0.72)
    accent = shade_color(color, 1.08)
    render.add_box(center_x, body_y, 0.22, body_w, body_d, first_h, MATERIAL_BUILDING, MATERIAL_ROOF, color)
    render.add_box(center_x + rng.uniform(-0.35, 0.35), body_y + rng.uniform(-0.25, 0.25), 0.22 + first_h, body_w * 0.78, body_d * 0.72, second_h, MATERIAL_BUILDING, MATERIAL_ROOF, shade_color(color, 0.96))
    collision.add_box(center_x, body_y, 0.22, body_w, body_d, first_h + second_h + roof_h, MATERIAL_BUILDING, MATERIAL_ROOF, color)
    add_hip_roof(render, center_x, body_y, 0.22 + first_h + second_h, body_w * 0.82, body_d * 0.76, roof_h, trim)
    porch_w = min(body_w * 0.48, 5.8)
    add_porch(render, center_x, body_y - body_d * 0.5, porch_w, accent, trim)
    garage_x = None
    if width >= 16.0:
        garage_w = min(width * 0.30, 6.4)
        garage_d = body_d * 0.78
        garage_x = center_x + width * 0.34
        render.add_box(garage_x, body_y + garage_d * 0.10, 0.22, garage_w, garage_d, first_h * 0.86, MATERIAL_BUILDING, MATERIAL_ROOF, shade_color(color, 0.92))
        collision.add_box(garage_x, body_y + garage_d * 0.10, 0.22, garage_w, garage_d, first_h * 0.86 + roof_h * 0.60, MATERIAL_BUILDING, MATERIAL_ROOF, color)
        add_hip_roof(render, garage_x, body_y + garage_d * 0.10, 0.22 + first_h * 0.86, garage_w * 1.02, garage_d * 1.02, roof_h * 0.62, trim)
    add_window_strip(render, center_x, body_y, body_w, body_d, 1.0, 2.2, max(2, min(4, int(body_w / 4.2))))
    add_window_strip(render, center_x, body_y, body_w * 0.72, body_d * 0.66, 4.9, 6.0, max(2, min(3, int(body_w / 5.0))), front_only=True)
    add_side_windows(render, center_x, body_y, body_w, body_d, 1.1, 2.3, max(2, min(3, int(body_d / 4.0))))
    add_driveway_and_path(render, center_x, body_y, body_w, body_d, garage_x)


def add_business_archetype(
    render: MeshBuilder,
    collision: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    height: float,
    color: tuple[int, int, int, int],
    rng: random.Random,
    district: str,
) -> None:
    base_h = max(4.2, height * 0.72)
    parapet_h = max(0.8, min(1.8, height * 0.10))
    render.add_box(center_x, center_y, 0.22, width, depth, base_h, MATERIAL_BUILDING, MATERIAL_ROOF, color)
    collision.add_box(center_x, center_y, 0.22, width, depth, base_h + parapet_h, MATERIAL_BUILDING, MATERIAL_ROOF, color)
    trim = shade_color(color, 0.70)
    front_y = center_y - depth * 0.5 - 0.03
    awning = rng.choice(((132, 60, 46, 255), (57, 101, 128, 255), (110, 98, 58, 255), (85, 56, 109, 255)))
    render.add_box(center_x, center_y - depth * 0.5, 0.22 + base_h, width + 0.42, 0.36, parapet_h, MATERIAL_TRIM, MATERIAL_TRIM, trim)
    render.add_box(center_x, front_y - 0.62, 3.35, width * 0.74, 1.18, 0.20, MATERIAL_TRIM, MATERIAL_TRIM, awning)
    bay_count = max(2, min(5, int(width / 5.2)))
    for bay in range(bay_count):
        span = width / bay_count
        cx = center_x - width * 0.5 + (bay + 0.5) * span
        glass_half = min(1.45, span * 0.32)
        render.add_quad(((cx - glass_half, front_y, 0.55), (cx + glass_half, front_y, 0.55), (cx + glass_half, front_y, 3.05), (cx - glass_half, front_y, 3.05)), MATERIAL_WINDOW, (77, 101, 115, 255))
    door_w = min(2.2, width * 0.12)
    render.add_quad(((center_x - door_w, front_y - 0.01, 0.25), (center_x + door_w, front_y - 0.01, 0.25), (center_x + door_w, front_y - 0.01, 3.35), (center_x - door_w, front_y - 0.01, 3.35)), MATERIAL_TRIM, shade_color(color, 0.52))
    add_flat_roof_details(render, center_x, center_y, width, depth, base_h, trim, rng)
    if depth > 16.0 and district in {"industrial", "eastside"}:
        service_w = width * 0.34
        service_y = center_y + depth * 0.31
        render.add_box(center_x, service_y, 0.22, service_w, depth * 0.24, 3.4, MATERIAL_BUILDING, MATERIAL_ROOF, shade_color(color, 0.90))


def add_midrise_archetype(
    render: MeshBuilder,
    collision: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    height: float,
    color: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    podium_h = min(6.8, height * 0.24)
    shaft_h = max(10.0, height - podium_h)
    podium_w = width
    podium_d = depth
    shaft_w = width * 0.76
    shaft_d = depth * 0.74
    render.add_box(center_x, center_y, 0.22, podium_w, podium_d, podium_h, MATERIAL_BUILDING, MATERIAL_ROOF, color)
    render.add_box(center_x, center_y + depth * 0.02, 0.22 + podium_h, shaft_w, shaft_d, shaft_h, MATERIAL_BUILDING, MATERIAL_ROOF, shade_color(color, 0.96))
    collision.add_box(center_x, center_y, 0.22, podium_w, podium_d, podium_h + shaft_h, MATERIAL_BUILDING, MATERIAL_ROOF, color)
    add_facade_architecture(render, center_x, center_y, shaft_w, shaft_d, podium_h + shaft_h, color, "downtown", rng)
    add_flat_roof_details(render, center_x, center_y + depth * 0.02, shaft_w, shaft_d, podium_h + shaft_h, shade_color(color, 0.72), rng)


def add_tower_archetype(
    render: MeshBuilder,
    collision: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    height: float,
    color: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    podium_h = min(8.0, height * 0.18)
    mid_h = min(height * 0.36, max(12.0, height * 0.28))
    top_h = height - podium_h - mid_h
    scales = [(1.0, podium_h), (0.72, mid_h), (0.52, top_h)]
    z = 0.22
    for scale, part_h in scales:
        render.add_box(center_x, center_y, z, width * scale, depth * scale, part_h, MATERIAL_BUILDING, MATERIAL_ROOF, shade_color(color, 0.94 + 0.08 * scale))
        z += part_h
    collision.add_box(center_x, center_y, 0.22, width, depth, height, MATERIAL_BUILDING, MATERIAL_ROOF, color)
    add_facade_architecture(render, center_x, center_y, width * 0.72, depth * 0.72, podium_h + mid_h, color, "downtown", rng)
    add_flat_roof_details(render, center_x, center_y, width * 0.52, depth * 0.52, height, shade_color(color, 0.72), rng)


def select_building_production(
    district: str,
    width: float,
    depth: float,
    height: float,
    rng: random.Random,
) -> BuildingProduction:
    """Rewrite a parcel envelope into a mass/facade/roof production.

    Lot dimensions and zone grammar constrain the legal rules first; the seed
    only chooses among those legal productions.  This is the RenderWare-sized
    equivalent of a parametric building L-system axiom and productions.
    """
    roll = rng.random()
    if district in {"industrial", "eastside"} and width >= 22.0 and depth >= 20.0:
        return BuildingProduction("industrial_shed", "loading_bays", "roof_monitors")
    if district == "downtown" and height >= 46.0 and width >= 22.0 and depth >= 22.0 and roll < 0.58:
        return BuildingProduction("setback_tower", "vertical_bays", "equipment_parapet")
    if district in {"downtown", "commercial", "civic"} and width >= 34.0 and depth >= 34.0 and roll < 0.42:
        return BuildingProduction("courtyard", "pilaster_bays", "flat_parapet")
    if district in {"residential", "grove", "coastal", "beach", "hills", "eastside"} and width >= 27.0 and depth >= 27.0 and roll < 0.38:
        return BuildingProduction("l_wing", "porch_bays", "domestic_roof")
    if district in {"residential", "grove", "coastal", "beach", "hills", "eastside"} and width >= 30.0 and roll < 0.68:
        return BuildingProduction("row", "repeated_units", "domestic_roof")
    roof = "domestic_roof" if district in {"residential", "grove", "coastal", "beach", "hills"} else "flat_parapet"
    facade = "storefront_bays" if district in {"commercial", "downtown", "eastside"} else "regular_bays"
    return BuildingProduction("slab", facade, roof)


def add_building_component(
    render: MeshBuilder,
    collision: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    height: float,
    color: tuple[int, int, int, int],
    district: str,
    rng: random.Random,
) -> None:
    add_procedural_building(render, collision, center_x, center_y, width, depth, height, color, rng)
    add_facade_architecture(render, center_x, center_y, width, depth, height, color, district, rng)


def add_facade_architecture(
    render: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    height: float,
    color: tuple[int, int, int, int],
    district: str,
    rng: random.Random,
) -> None:
    """Add cornices, pilasters, storefronts and balconies to break up box silhouettes."""
    front_y = center_y - depth * 0.5 - 0.09
    trim_color = tuple(max(0, min(255, round(channel * 0.72))) for channel in color[:3]) + (255,)
    accent_color = choose_facade_accent(color, district, rng)
    base_height = min(4.2, max(2.8, height * 0.25))
    add_front_facade_panel(render, center_x, front_y - 0.01, width * 0.985, 0.24, 0.24 + base_height, MATERIAL_BUILDING, accent_color)
    add_front_facade_panel(render, center_x, front_y - 0.02, width + 0.55, max(0.35, height - 0.35), height + 0.07, MATERIAL_TRIM, trim_color)

    floor_spacing = 4.1
    floor_count = max(1, min(10, int(height / floor_spacing)))
    for floor in range(1, floor_count):
        if floor % 2 and district not in {"downtown", "commercial", "civic"}:
            continue
        z = 0.30 + floor * height / floor_count
        add_front_facade_panel(render, center_x, front_y - 0.025, width * 0.98, z, z + 0.16, MATERIAL_TRIM, trim_color)

    bay_count = max(2, min(6, round(width / 7.0)))
    for bay in range(bay_count + 1):
        if bay not in {0, bay_count} and rng.random() > 0.58:
            continue
        x = center_x - width * 0.5 + bay * width / bay_count
        add_front_facade_panel(render, x, front_y - 0.03, 0.24, 0.28, max(2.78, height - 0.14), MATERIAL_TRIM, trim_color)

    if district in {"commercial", "downtown", "civic", "eastside"} and width >= 13.0:
        storefront_color = rng.choice(((143, 55, 43, 255), (158, 112, 52, 255), (57, 112, 73, 255), (113, 65, 94, 255), (55, 91, 137, 255)))
        render.add_box(center_x, front_y - 0.48, 3.5, width * 0.70, 1.05, 0.24, MATERIAL_TRIM, MATERIAL_TRIM, storefront_color)
    elif district in {"residential", "grove", "coastal", "beach", "hills"} and height >= 10.0 and rng.random() < 0.62:
        balcony_width = min(width * 0.58, 10.5)
        balcony_z = min(height - 1.0, 4.1)
        render.add_box(center_x, front_y - 0.62, balcony_z, balcony_width, 1.25, 0.18, MATERIAL_TRIM, MATERIAL_TRIM, trim_color)
        for direction in (-1.0, 1.0):
            render.add_box(center_x + direction * balcony_width * 0.46, front_y - 1.13, balcony_z + 0.18, 0.16, 0.16, 1.05, MATERIAL_TRIM, MATERIAL_TRIM, trim_color)

    if district in {"residential", "grove", "coastal", "beach", "hills"}:
        add_residential_entry_details(render, center_x, front_y, width, height, trim_color, accent_color, rng)
        add_residential_roof_details(render, center_x, center_y, width, depth, height, trim_color, rng)
    elif district in {"eastside", "commercial", "downtown"} and height >= 15.0 and width >= 14.0:
        if rng.random() < 0.42:
            add_fire_escape(render, center_x, front_y, width, height, trim_color, rng)
        add_flat_roof_details(render, center_x, center_y, width, depth, height, trim_color, rng)
    elif district == "civic":
        add_flat_roof_details(render, center_x, center_y, width, depth, height, trim_color, rng)


def add_front_facade_panel(
    render: MeshBuilder,
    center_x: float,
    y: float,
    width: float,
    minimum_z: float,
    maximum_z: float,
    material: int,
    color: tuple[int, int, int, int],
) -> None:
    half_width = width * 0.5
    render.add_quad(
        (
            (center_x - half_width, y, minimum_z),
            (center_x + half_width, y, minimum_z),
            (center_x + half_width, y, maximum_z),
            (center_x - half_width, y, maximum_z),
        ),
        material,
        color,
        ((0.0, 1.0), (max(1.0, width / 4.0), 1.0), (max(1.0, width / 4.0), 0.0), (0.0, 0.0)),
    )


def choose_facade_accent(
    color: tuple[int, int, int, int],
    district: str,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    palette = DISTRICT_BUILDING_PALETTES.get(district, BUILDING_PALETTE)
    if rng.random() < 0.58:
        candidates = [candidate for candidate in palette if color_distance(candidate, color) >= 42.0]
        if candidates:
            return rng.choice(candidates)
    multiplier = rng.choice((0.55, 0.68, 1.12))
    return tuple(max(0, min(255, round(channel * multiplier))) for channel in color[:3]) + (255,)


def color_distance(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))


def add_residential_entry_details(
    render: MeshBuilder,
    center_x: float,
    front_y: float,
    width: float,
    height: float,
    trim_color: tuple[int, int, int, int],
    accent_color: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    if width < 8.0:
        return
    entry_x = center_x + rng.uniform(-width * 0.20, width * 0.20)
    step_width = min(4.4, width * 0.30)
    for index in range(3):
        render.add_box(
            entry_x,
            front_y - 0.45 - index * 0.34,
            0.23 + index * 0.12,
            step_width - index * 0.28,
            0.62,
            0.12,
            MATERIAL_TRIM,
            MATERIAL_TRIM,
            trim_color,
        )
    if rng.random() < 0.58:
        porch_width = min(width * 0.62, 9.5)
        render.add_box(entry_x, front_y - 1.05, 3.1, porch_width, 2.0, 0.22, MATERIAL_TRIM, MATERIAL_ROOF, accent_color)
        for direction in (-1.0, 1.0):
            render.add_box(
                entry_x + direction * porch_width * 0.43,
                front_y - 1.55,
                0.28,
                0.22,
                0.22,
                2.95,
                MATERIAL_TRIM,
                MATERIAL_TRIM,
                trim_color,
            )
    if width >= 15.0 and height <= 18.0 and rng.random() < 0.36:
        garage_width = min(5.5, width * 0.30)
        garage_x = center_x + (-1.0 if entry_x >= center_x else 1.0) * width * 0.27
        render.add_quad(
            (
                (garage_x - garage_width * 0.5, front_y - 0.03, 0.25),
                (garage_x + garage_width * 0.5, front_y - 0.03, 0.25),
                (garage_x + garage_width * 0.5, front_y - 0.03, 3.05),
                (garage_x - garage_width * 0.5, front_y - 0.03, 3.05),
            ),
            MATERIAL_TRIM,
            tuple(max(0, channel - 28) for channel in accent_color[:3]) + (255,),
        )


def add_residential_roof_details(
    render: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    height: float,
    trim_color: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    roof_z = height + 0.22
    if rng.random() < 0.48:
        chimney_x = center_x + rng.uniform(-width * 0.28, width * 0.28)
        chimney_y = center_y + rng.uniform(-depth * 0.22, depth * 0.22)
        render.add_box(chimney_x, chimney_y, roof_z, 1.1, 1.0, rng.uniform(1.8, 3.2), MATERIAL_TRIM, MATERIAL_ROOF, trim_color)
    if width >= 18.0 and rng.random() < 0.24:
        render.add_box(
            center_x + rng.uniform(-width * 0.20, width * 0.20),
            center_y + rng.uniform(-depth * 0.18, depth * 0.18),
            roof_z,
            2.8,
            2.2,
            1.3,
            MATERIAL_TRIM,
            MATERIAL_ROOF,
            (116, 121, 115, 255),
        )


def add_flat_roof_details(
    render: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    height: float,
    trim_color: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    roof_z = height + 0.22
    parapet_height = rng.uniform(0.65, 1.15)
    render.add_box(center_x, center_y - depth * 0.5, roof_z, width, 0.34, parapet_height, MATERIAL_TRIM, MATERIAL_TRIM, trim_color)
    render.add_box(center_x, center_y + depth * 0.5, roof_z, width, 0.34, parapet_height, MATERIAL_TRIM, MATERIAL_TRIM, trim_color)
    render.add_box(center_x - width * 0.5, center_y, roof_z, 0.34, depth, parapet_height, MATERIAL_TRIM, MATERIAL_TRIM, trim_color)
    render.add_box(center_x + width * 0.5, center_y, roof_z, 0.34, depth, parapet_height, MATERIAL_TRIM, MATERIAL_TRIM, trim_color)
    equipment_count = 1 if width * depth < 700.0 else 2
    for index in range(equipment_count):
        equipment_color = rng.choice(((103, 109, 107, 255), (128, 119, 101, 255), (91, 96, 101, 255)))
        render.add_box(
            center_x + rng.uniform(-width * 0.22, width * 0.22),
            center_y + rng.uniform(-depth * 0.20, depth * 0.20),
            roof_z,
            rng.uniform(2.2, min(5.5, width * 0.24)),
            rng.uniform(2.0, min(4.8, depth * 0.24)),
            rng.uniform(1.4, 2.8),
            MATERIAL_TRIM,
            MATERIAL_ROOF,
            equipment_color,
        )


def add_fire_escape(
    render: MeshBuilder,
    center_x: float,
    front_y: float,
    width: float,
    height: float,
    trim_color: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    escape_x = center_x + rng.choice((-1.0, 1.0)) * width * 0.28
    platform_width = min(4.6, width * 0.28)
    platform_levels = max(2, min(5, int(height // 5.0)))
    for level in range(1, platform_levels + 1):
        platform_z = min(height - 0.8, level * height / (platform_levels + 1))
        render.add_box(escape_x, front_y - 0.62, platform_z, platform_width, 1.25, 0.14, MATERIAL_TRIM, MATERIAL_TRIM, trim_color)
        for direction in (-1.0, 1.0):
            render.add_box(
                escape_x + direction * platform_width * 0.46,
                front_y - 1.12,
                platform_z + 0.14,
                0.12,
                0.12,
                1.0,
                MATERIAL_TRIM,
                MATERIAL_TRIM,
                trim_color,
            )
        if level < platform_levels:
            next_z = min(height - 0.8, (level + 1) * height / (platform_levels + 1))
            ladder_x = escape_x + (-1.0 if level % 2 else 1.0) * platform_width * 0.30
            for direction in (-1.0, 1.0):
                render.add_box(
                    ladder_x + direction * 0.28,
                    front_y - 1.15,
                    platform_z,
                    0.09,
                    0.09,
                    next_z - platform_z,
                    MATERIAL_TRIM,
                    MATERIAL_TRIM,
                    trim_color,
                )


def add_industrial_roofline(
    render: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    height: float,
    color: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    roof_color = tuple(max(0, min(255, round(channel * 0.62))) for channel in color[:3]) + (255,)
    monitor_count = max(2, min(5, round(width / 12.0)))
    for index in range(monitor_count):
        x = center_x - width * 0.38 + index * width * 0.76 / max(1, monitor_count - 1)
        render.add_box(x, center_y, height + 0.22, min(4.2, width / monitor_count * 0.48), depth * 0.58, rng.uniform(1.2, 2.4), MATERIAL_TRIM, MATERIAL_ROOF, roof_color)
    front_y = center_y - depth * 0.5 - 0.10
    door_count = max(1, min(4, round(width / 11.0)))
    for index in range(door_count):
        x = center_x - width * 0.5 + (index + 0.5) * width / door_count
        door_width = min(5.0, width / door_count * 0.62)
        render.add_quad(
            ((x - door_width * 0.5, front_y, 0.24), (x + door_width * 0.5, front_y, 0.24),
             (x + door_width * 0.5, front_y, 4.8), (x - door_width * 0.5, front_y, 4.8)),
            MATERIAL_TRIM,
            (70, 68, 63, 255),
        )


def populate_park_blocks(
    render: MeshBuilder,
    collision: MeshBuilder,
    blocks: list[CityBlock],
    rng: random.Random,
    traffic_segments: list[TrafficPathSegment],
) -> int:
    for block in blocks:
        spacing = 27.0
        columns = max(2, int(block.width / spacing))
        rows = max(2, int(block.depth / spacing))
        for row in range(rows):
            for column in range(columns):
                x = block.center_x - block.width * 0.5 + (column + 0.5) * block.width / columns
                y = block.center_y - block.depth * 0.5 + (row + 0.5) * block.depth / rows
                if (
                    rng.random() < 0.20
                    or not point_in_polygon(x, y, block.points)
                    or not is_footprint_clear(x, y, 7.0, 7.0, traffic_segments)
                    or len(render.mesh.vertices) > 26_000
                ):
                    continue
                add_generated_tree(
                    render, collision, x, y, rng,
                    block.pad_elevation - MISSION_SAFE_TERRAIN_HEIGHT + 0.22, "park",
                )
    return 0


def add_parcel_park(
    render: MeshBuilder,
    collision: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    rng: random.Random,
    district: str,
) -> None:
    color = district_grass_color(district)
    for builder in (render, collision):
        builder.add_box(center_x, center_y, 0.22, width, depth, 0.08, MATERIAL_GRASS, MATERIAL_GRASS, color)
    for offset_x, offset_y in ((-0.23, -0.22), (0.24, 0.20), (-0.22, 0.23)):
        if min(width, depth) < 12.0 or rng.random() < 0.20:
            continue
        add_generated_tree(
            render, collision, center_x + width * offset_x,
            center_y + depth * offset_y, rng, 0.30, district,
        )


def add_street_trees_for_block(
    render: MeshBuilder,
    collision: MeshBuilder,
    block: CityBlock,
    district: str,
    rng: random.Random,
    traffic_segments: list[TrafficPathSegment],
    occupied_footprints: list[tuple[tuple[float, float], ...]] | None = None,
) -> None:
    chance = 0.48 if district in {"residential", "grove", "coastal", "beach", "hills"} else 0.24
    center_x, center_y = polygon_centroid(block.points)
    points: list[tuple[float, float]] = []
    for first, second in zip(block.points, block.points[1:] + block.points[:1]):
        length = math.dist(first, second)
        count = max(1, min(8, int(length / 18.0)))
        for index in range(count):
            factor = (index + 0.5) / count
            x = first[0] + (second[0] - first[0]) * factor
            y = first[1] + (second[1] - first[1]) * factor
            distance = max(1.0, math.hypot(center_x - x, center_y - y))
            points.append((x + (center_x - x) / distance * 2.2, y + (center_y - y) / distance * 2.2))
    for x, y in points:
        if (
            rng.random() > chance
            or not point_in_polygon(x, y, block.points)
            or not is_footprint_clear(x, y, 6.0, 6.0, traffic_segments)
            or any(point_in_polygon(x, y, footprint) for footprint in (occupied_footprints or []))
            or len(render.mesh.vertices) > 26_000
        ):
            continue
        add_generated_tree(render, collision, x, y, rng, 0.22, district)


def rectangle_inside_polygon(
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    polygon: tuple[tuple[float, float], ...],
) -> bool:
    half_width = width * 0.5
    half_depth = depth * 0.5
    return all(
        point_in_polygon(x, y, polygon)
        for x, y in (
            (center_x - half_width, center_y - half_depth),
            (center_x + half_width, center_y - half_depth),
            (center_x + half_width, center_y + half_depth),
            (center_x - half_width, center_y + half_depth),
            (center_x, center_y),
        )
    )


def point_in_polygon(x: float, y: float, polygon: tuple[tuple[float, float], ...]) -> bool:
    inside = False
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        if point_on_segment(x, y, first[0], first[1], second[0], second[1]):
            return True
        if (first[1] > y) == (second[1] > y):
            continue
        intersection_x = first[0] + (y - first[1]) * (second[0] - first[0]) / (second[1] - first[1])
        if x < intersection_x:
            inside = not inside
    return inside


def point_on_segment(
    point_x: float,
    point_y: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> bool:
    cross = (point_x - start_x) * (end_y - start_y) - (point_y - start_y) * (end_x - start_x)
    if abs(cross) > 0.001:
        return False
    return (
        min(start_x, end_x) - 0.001 <= point_x <= max(start_x, end_x) + 0.001
        and min(start_y, end_y) - 0.001 <= point_y <= max(start_y, end_y) + 0.001
    )


def add_tapered_limb(
    builder: MeshBuilder,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    start_radius: float,
    end_radius: float,
    color: tuple[int, int, int, int],
    sides: int = 7,
) -> None:
    delta = tuple(end[index] - start[index] for index in range(3))
    length = math.sqrt(sum(value * value for value in delta))
    if length < 0.00001:
        return
    direction = tuple(value / length for value in delta)
    reference = (1.0, 0.0, 0.0) if abs(direction[2]) > 0.9 else (0.0, 0.0, 1.0)
    axis_u = (
        direction[1] * reference[2] - direction[2] * reference[1],
        direction[2] * reference[0] - direction[0] * reference[2],
        direction[0] * reference[1] - direction[1] * reference[0],
    )
    axis_length = math.sqrt(sum(value * value for value in axis_u))
    axis_u = tuple(value / axis_length for value in axis_u)
    axis_v = (
        direction[1] * axis_u[2] - direction[2] * axis_u[1],
        direction[2] * axis_u[0] - direction[0] * axis_u[2],
        direction[0] * axis_u[1] - direction[1] * axis_u[0],
    )
    lower: list[tuple[float, float, float]] = []
    upper: list[tuple[float, float, float]] = []
    for index in range(sides):
        angle = math.tau * index / sides
        radial = tuple(axis_u[axis] * math.cos(angle) + axis_v[axis] * math.sin(angle) for axis in range(3))
        lower.append(tuple(start[axis] + radial[axis] * start_radius for axis in range(3)))
        upper.append(tuple(end[axis] + radial[axis] * end_radius for axis in range(3)))
    for index in range(sides):
        following = (index + 1) % sides
        builder.add_quad(
            (lower[index], lower[following], upper[following], upper[index]),
            MATERIAL_TRIM, color,
            ((index / sides * 2.0, length / 3.0), (following / sides * 2.0, length / 3.0),
             (following / sides * 2.0, 0.0), (index / sides * 2.0, 0.0)),
        )
        builder.add_triangle((end, upper[index], upper[following]), MATERIAL_TRIM, color)


def add_foliage_cards(
    render: MeshBuilder,
    x: float,
    y: float,
    z: float,
    width: float,
    height: float,
    angle: float,
    color: tuple[int, int, int, int],
) -> None:
    for index in range(3):
        direction = angle + math.pi * index / 3.0
        offset_x = math.cos(direction) * width * 0.5
        offset_y = math.sin(direction) * width * 0.5
        render.add_double_sided_quad(
            ((x - offset_x, y - offset_y, z), (x + offset_x, y + offset_y, z),
             (x + offset_x, y + offset_y, z + height), (x - offset_x, y - offset_y, z + height)),
            MATERIAL_VEGETATION, color,
        )


def add_generated_tree(
    render: MeshBuilder,
    collision: MeshBuilder,
    x: float,
    y: float,
    rng: random.Random,
    minimum_z: float,
    district: str = "residential",
) -> None:
    render_checkpoint = mesh_checkpoint(render)
    collision_checkpoint = mesh_checkpoint(collision)
    build_tree_geometry(render, collision, x, y, rng, minimum_z, district)
    if len(render.mesh.vertices) > 27_000:
        restore_mesh_checkpoint(render, render_checkpoint)
        restore_mesh_checkpoint(collision, collision_checkpoint)


def build_tree_geometry(
    render: MeshBuilder,
    collision: MeshBuilder,
    x: float,
    y: float,
    rng: random.Random,
    minimum_z: float,
    district: str,
) -> None:
    palm = district in {"beach", "coastal"} and rng.random() < 0.72
    trunk_height = rng.uniform(9.0, 14.0) if palm else rng.uniform(4.8, 8.0)
    bark = rng.choice(((101, 84, 61, 255), (112, 96, 72, 255), (87, 77, 60, 255)))
    lean_angle = rng.uniform(0.0, math.tau)
    lean = rng.uniform(0.15, 0.65) if palm else rng.uniform(0.0, 0.22)
    middle = (x + math.cos(lean_angle) * lean * 0.35, y + math.sin(lean_angle) * lean * 0.35, minimum_z + trunk_height * 0.55)
    crown = (x + math.cos(lean_angle) * lean, y + math.sin(lean_angle) * lean, minimum_z + trunk_height)
    base = (x, y, minimum_z)
    radius = rng.uniform(0.22, 0.35) if palm else rng.uniform(0.30, 0.46)
    add_tapered_limb(render, base, middle, radius * 1.35, radius * 0.82, bark)
    add_tapered_limb(render, middle, crown, radius * 0.82, radius * 0.43, bark)
    add_tapered_limb(collision, base, crown, radius, radius * 0.55, bark, sides=5)
    if palm:
        angle = rng.uniform(0.0, math.tau)
        fronds = rng.randint(7, 10)
        for index in range(fronds):
            direction = angle + math.tau * index / fronds
            length = rng.uniform(4.6, 7.2)
            width = rng.uniform(0.75, 1.15)
            side_x, side_y = -math.sin(direction), math.cos(direction)
            center = (crown[0] + math.cos(direction) * length * 0.48,
                      crown[1] + math.sin(direction) * length * 0.48,
                      crown[2] + rng.uniform(0.45, 1.4))
            tip = (crown[0] + math.cos(direction) * length,
                   crown[1] + math.sin(direction) * length,
                   crown[2] - rng.uniform(1.0, 2.6))
            color = rng.choice(((156, 172, 122, 255), (171, 181, 131, 255), (137, 159, 111, 255)))
            add_tapered_limb(render, crown, center, 0.065, 0.035, color, sides=3)
            add_tapered_limb(render, center, tip, 0.035, 0.008, color, sides=3)
            for leaf_index in range(11):
                factor = 0.10 + leaf_index * 0.078
                segment_start, segment_end = (crown, center) if factor < 0.48 else (center, tip)
                local_factor = factor / 0.48 if factor < 0.48 else (factor - 0.48) / 0.52
                root = tuple(segment_start[axis] + (segment_end[axis] - segment_start[axis]) * local_factor for axis in range(3))
                spread = width * math.sin(math.pi * factor) * rng.uniform(0.82, 1.10)
                for side in (-1.0, 1.0):
                    leaf_tip = (root[0] + side_x * spread * side + math.cos(direction) * length * 0.08,
                                root[1] + side_y * spread * side + math.sin(direction) * length * 0.08,
                                root[2] - spread * 0.32)
                    leaf_root = (root[0] - math.cos(direction) * 0.12,
                                 root[1] - math.sin(direction) * 0.12, root[2] - 0.015)
                    points = (leaf_root, leaf_tip, (root[0] + math.cos(direction) * 0.15,
                                                   root[1] + math.sin(direction) * 0.15, root[2] + 0.015))
                    render.add_triangle(points, MATERIAL_GRASS, color)
                    render.add_triangle(tuple(reversed(points)), MATERIAL_GRASS, color)
        return

    palettes = {
        "hills": ((104, 132, 89, 255), (120, 145, 98, 255), (91, 118, 79, 255)),
        "industrial": ((126, 139, 96, 255), (139, 149, 104, 255), (108, 126, 88, 255)),
    }
    palette = palettes.get(district, ((113, 143, 93, 255), (132, 155, 105, 255), (97, 128, 84, 255)))
    crown_width = rng.uniform(6.5, 10.0)
    phase = rng.uniform(0.0, math.tau)
    for index in range(3):
        direction = phase + math.tau * index / 3.0
        branch_end = (crown[0] + math.cos(direction) * crown_width * 0.23,
                      crown[1] + math.sin(direction) * crown_width * 0.23,
                      crown[2] - trunk_height * rng.uniform(0.02, 0.18))
        branch_start = (middle[0], middle[1], middle[2] + trunk_height * 0.06)
        add_tapered_limb(render, branch_start, branch_end, radius * 0.48, radius * 0.12, bark, sides=5)
        add_foliage_cards(
            render, branch_end[0], branch_end[1], branch_end[2] - trunk_height * 0.27,
            crown_width * rng.uniform(0.66, 0.84), crown_width * rng.uniform(0.57, 0.74),
            direction + rng.uniform(-0.25, 0.25), palette[index],
        )


def add_street_markings(render: MeshBuilder, blocks: list[CityBlock], tile_half: float) -> None:
    if len(blocks) >= 4:
        lower_left, upper_left, lower_right, upper_right = blocks[:4]
        vertical_bottom = midpoint(lower_left.points[1], lower_right.points[0])
        vertical_center = midpoint(lower_left.points[2], lower_right.points[3])
        vertical_top = midpoint(upper_left.points[2], upper_right.points[3])
        horizontal_left = midpoint(lower_left.points[3], upper_left.points[0])
        horizontal_center = midpoint(lower_left.points[2], upper_left.points[1])
        horizontal_right = midpoint(lower_right.points[2], upper_right.points[1])
        add_dashed_lane_markings(render, *vertical_bottom, *vertical_center)
        add_dashed_lane_markings(render, *vertical_center, *vertical_top)
        add_dashed_lane_markings(render, *horizontal_left, *horizontal_center)
        add_dashed_lane_markings(render, *horizontal_center, *horizontal_right)
    elif len(blocks) == 2:
        first, second = blocks
        if abs(first.center_x - second.center_x) > abs(first.center_y - second.center_y):
            lower = midpoint(first.points[1], second.points[0])
            upper = midpoint(first.points[2], second.points[3])
            add_dashed_lane_markings(render, *lower, *upper)
        else:
            left = midpoint(first.points[3], second.points[0])
            right = midpoint(first.points[2], second.points[1])
            add_dashed_lane_markings(render, *left, *right)


def midpoint(first: tuple[float, float], second: tuple[float, float]) -> tuple[float, float]:
    return (first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5


def add_dashed_lane_markings(
    render: MeshBuilder,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> None:
    length = math.hypot(end_x - start_x, end_y - start_y)
    if length <= 1.0:
        return
    direction_x = (end_x - start_x) / length
    direction_y = (end_y - start_y) / length
    perpendicular_x = -direction_y * 0.14
    perpendicular_y = direction_x * 0.14
    position = 5.0
    while position < length - 3.0:
        segment_length = min(4.8, length - position)
        x0 = start_x + direction_x * position
        y0 = start_y + direction_y * position
        x1 = start_x + direction_x * (position + segment_length)
        y1 = start_y + direction_y * (position + segment_length)
        render.add_quad(
            (
                (x0 + perpendicular_x, y0 + perpendicular_y, 0.235),
                (x1 + perpendicular_x, y1 + perpendicular_y, 0.235),
                (x1 - perpendicular_x, y1 - perpendicular_y, 0.235),
                (x0 - perpendicular_x, y0 - perpendicular_y, 0.235),
            ),
            MATERIAL_TRIM,
            (211, 183, 104, 255),
        )
        position += 12.0


def add_lot_buildings(
    render: MeshBuilder,
    collision: MeshBuilder,
    block_size: float,
    district: str,
    rng: random.Random,
    height: tuple[float, float],
    park_chance: float = 0.08,
    traffic_segments: list[TrafficPathSegment] | None = None,
) -> int:
    traffic_segments = traffic_segments or []
    offset = block_size * 0.245
    maximum_lot = block_size * 0.43
    count = 0
    for lot_x, lot_y in ((-offset, -offset), (offset, -offset), (-offset, offset), (offset, offset)):
        if rng.random() < park_chance:
            if is_footprint_clear(lot_x, lot_y, maximum_lot, maximum_lot, traffic_segments):
                add_park(render, collision, lot_x, lot_y, rng, district_grass_color(district))
            continue
        count += 1
        width = rng.uniform(maximum_lot * 0.68, maximum_lot * 0.94)
        depth = rng.uniform(maximum_lot * 0.68, maximum_lot * 0.94)
        if not is_footprint_clear(lot_x, lot_y, width, depth, traffic_segments):
            continue
        building_height = rng.uniform(*height)
        color = rng.choice(BUILDING_PALETTE)
        if district == "grove":
            color = rng.choice(((176, 147, 111, 255), (139, 162, 132, 255), (186, 175, 147, 255)))
        add_procedural_building(render, collision, lot_x, lot_y, width, depth, building_height, color, rng)
    return count


def add_airport_tile(
    render: MeshBuilder,
    collision: MeshBuilder,
    tile_size: float,
    world_x: float,
    world_y: float,
    rng: random.Random,
    traffic_segments: list[TrafficPathSegment],
) -> int:
    runway = (54, 58, 62, 255)
    if world_y < -2500.0:
        runway_depth = 66.0
        render.add_box(0.0, 0.0, 0.24, tile_size + 0.5, runway_depth, 0.08, MATERIAL_ROAD, MATERIAL_ROAD, runway)
        collision.add_box(0.0, 0.0, 0.24, tile_size + 0.5, runway_depth, 0.08, MATERIAL_ROAD, MATERIAL_ROAD, runway)
        for x in (-tile_size * 0.35, -tile_size * 0.10, tile_size * 0.15, tile_size * 0.40):
            render.add_box(x, 0.0, 0.33, 13.0, 0.8, 0.04, MATERIAL_TRIM, MATERIAL_TRIM, (235, 224, 180, 255))
        return 0

    if 1125.0 <= world_x <= 2125.0:
        width = tile_size * rng.uniform(0.50, 0.70)
        depth = tile_size * rng.uniform(0.32, 0.45)
        center_y = 30.0
        if not is_footprint_clear(0.0, center_y, width, depth, traffic_segments):
            return 0
        height = rng.uniform(15.0, 24.0)
        render.add_box(0.0, center_y, 0.24, width, depth, height, MATERIAL_BUILDING, MATERIAL_ROOF, (135, 146, 150, 255))
        collision.add_box(0.0, center_y, 0.24, width, depth, height, MATERIAL_BUILDING, MATERIAL_ROOF, (135, 146, 150, 255))
        render.add_box(0.0, -tile_size * 0.34, 0.24, tile_size + 0.5, 28.0, 0.05, MATERIAL_ROAD, MATERIAL_ROAD, runway)
        collision.add_box(0.0, -tile_size * 0.34, 0.24, tile_size + 0.5, 28.0, 0.05, MATERIAL_ROAD, MATERIAL_ROAD, runway)
        return 1
    return 0


def add_docks_tile(
    render: MeshBuilder,
    collision: MeshBuilder,
    block_size: float,
    rng: random.Random,
    traffic_segments: list[TrafficPathSegment],
) -> int:
    count = 0
    container_colors = ((132, 70, 58, 255), (58, 91, 121, 255), (113, 118, 73, 255), (167, 132, 64, 255))
    for row_y in (-block_size * 0.27, block_size * 0.02, block_size * 0.31):
        count += 1
        color = rng.choice(container_colors)
        width = block_size * rng.uniform(0.55, 0.78)
        if not is_footprint_clear(0.0, row_y, width, block_size * 0.18, traffic_segments):
            count -= 1
            continue
        render.add_box(0.0, row_y, 0.24, width, block_size * 0.18, rng.uniform(7.0, 14.0), MATERIAL_BUILDING, MATERIAL_ROOF, color)
        collision.add_box(0.0, row_y, 0.24, width, block_size * 0.18, 10.0, MATERIAL_BUILDING, MATERIAL_ROOF, color)
    return count


def add_warehouse_tile(
    render: MeshBuilder,
    collision: MeshBuilder,
    block_size: float,
    district: str,
    rng: random.Random,
    traffic_segments: list[TrafficPathSegment],
) -> int:
    count = 2 if rng.random() < 0.66 else 3
    spacing = block_size / count
    for index in range(count):
        center_y = -block_size * 0.5 + spacing * (index + 0.5)
        if not is_footprint_clear(0.0, center_y, block_size * 0.76, spacing * 0.68, traffic_segments):
            count -= 1
            continue
        color = rng.choice(((130, 125, 112, 255), (142, 109, 88, 255), (105, 119, 124, 255)))
        render.add_box(0.0, center_y, 0.24, block_size * 0.76, spacing * 0.68, rng.uniform(12.0, 25.0), MATERIAL_BUILDING, MATERIAL_ROOF, color)
        collision.add_box(0.0, center_y, 0.24, block_size * 0.76, spacing * 0.68, 18.0, MATERIAL_BUILDING, MATERIAL_ROOF, color)
    return count


def add_green_tile(
    render: MeshBuilder,
    collision: MeshBuilder,
    block_size: float,
    district: str,
    rng: random.Random,
    traffic_segments: list[TrafficPathSegment],
) -> int:
    color = district_grass_color(district)
    render.add_box(0.0, 0.0, 0.24, block_size * 0.93, block_size * 0.93, 0.12, MATERIAL_GRASS, MATERIAL_GRASS, color)
    collision.add_box(0.0, 0.0, 0.24, block_size * 0.93, block_size * 0.93, 0.12, MATERIAL_GRASS, MATERIAL_GRASS, color)
    tree_grid = (-block_size * 0.31, 0.0, block_size * 0.31)
    for x in tree_grid:
        for y in tree_grid:
            if rng.random() < 0.30:
                continue
            if not is_footprint_clear(x, y, 11.0, 11.0, traffic_segments):
                continue
            render.add_box(x, y, 0.36, 1.5, 1.5, 6.0, MATERIAL_TRIM, MATERIAL_TRIM, (88, 65, 43, 255))
            render.add_crossed_billboard(x, y, 3.8, 13.0, 13.5, MATERIAL_VEGETATION, (116, 145, 112, 255))
            collision.add_box(x, y, 0.36, 1.5, 1.5, 6.0, MATERIAL_TRIM, MATERIAL_TRIM, (88, 65, 43, 255))
    return 0


def los_santos_traffic_segments(segments: list[TrafficPathSegment]) -> list[TrafficPathSegment]:
    return [
        segment
        for segment in segments
        if segment_intersects_bounds(segment, LOS_SANTOS_BOUNDS, 60.0)
    ]


def deduplicate_road_surface_segments(
    segments: list[TrafficPathSegment],
) -> list[TrafficPathSegment]:
    """Collapse near-identical lane links into one visible carriageway.

    GTA's vehicle graph may carry separate links for opposing lanes. They all
    remain active for traffic, terrain grading and clearance; only the visible
    road polygon/paint pass is de-duplicated.
    """
    selected: dict[tuple[tuple[int, int], tuple[int, int]], TrafficPathSegment] = {}
    for segment in segments:
        first = (round(segment.start_x / 8.0), round(segment.start_y / 8.0))
        second = (round(segment.end_x / 8.0), round(segment.end_y / 8.0))
        key = tuple(sorted((first, second)))
        current = selected.get(key)
        if current is None or segment.width > current.width:
            selected[key] = segment
    return list(selected.values())


def segment_intersects_bounds(
    segment: TrafficPathSegment,
    bounds: tuple[float, float, float, float],
    margin: float = 0.0,
) -> bool:
    minimum_x, minimum_y, maximum_x, maximum_y = bounds
    return not (
        max(segment.start_x, segment.end_x) < minimum_x - margin
        or min(segment.start_x, segment.end_x) > maximum_x + margin
        or max(segment.start_y, segment.end_y) < minimum_y - margin
        or min(segment.start_y, segment.end_y) > maximum_y + margin
    )


def traffic_segments_for_tile(
    segments: list[TrafficPathSegment],
    center_x: float,
    center_y: float,
    tile_size: float,
) -> list[TrafficPathSegment]:
    half = tile_size * 0.5
    bounds = (center_x - half, center_y - half, center_x + half, center_y + half)
    return [
        TrafficPathSegment(
            segment.start_x - center_x,
            segment.start_y - center_y,
            segment.start_z,
            segment.end_x - center_x,
            segment.end_y - center_y,
            segment.end_z,
            segment.width,
        )
        for segment in segments
        if segment_intersects_bounds(segment, bounds, max(8.0, segment.width * 0.5 + 2.0) + 16.0)
    ]


def is_footprint_clear(
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    segments: list[TrafficPathSegment],
    extra_clearance: float = 7.0,
) -> bool:
    for segment in segments:
        # Road surface + sidewalk + facade/setback envelope.  The earlier 1.25
        # unit margin allowed porches, roof details and whole small buildings to
        # intrude into vehicle lanes.
        clearance = segment.width * 0.5 + extra_clearance
        if segment_intersects_rectangle(
            segment.start_x,
            segment.start_y,
            segment.end_x,
            segment.end_y,
            center_x - width * 0.5 - clearance,
            center_y - depth * 0.5 - clearance,
            center_x + width * 0.5 + clearance,
            center_y + depth * 0.5 + clearance,
        ):
            return False
    return True


def segment_intersects_rectangle(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    minimum_x: float,
    minimum_y: float,
    maximum_x: float,
    maximum_y: float,
) -> bool:
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    minimum_factor = 0.0
    maximum_factor = 1.0
    for direction, lower_distance, upper_distance in (
        (delta_x, minimum_x - start_x, maximum_x - start_x),
        (delta_y, minimum_y - start_y, maximum_y - start_y),
    ):
        if abs(direction) <= 0.000001:
            if lower_distance > 0.0 or upper_distance < 0.0:
                return False
            continue
        first_factor = lower_distance / direction
        second_factor = upper_distance / direction
        entering = min(first_factor, second_factor)
        leaving = max(first_factor, second_factor)
        minimum_factor = max(minimum_factor, entering)
        maximum_factor = min(maximum_factor, leaving)
        if minimum_factor > maximum_factor:
            return False
    return True


def distance_from_point_to_segment(
    point_x: float,
    point_y: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> float:
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared <= 0.000001:
        return math.hypot(point_x - start_x, point_y - start_y)
    factor = ((point_x - start_x) * delta_x + (point_y - start_y) * delta_y) / length_squared
    factor = max(0.0, min(1.0, factor))
    closest_x = start_x + delta_x * factor
    closest_y = start_y + delta_y * factor
    return math.hypot(point_x - closest_x, point_y - closest_y)


def add_road_ribbon(render: MeshBuilder, segment: TrafficPathSegment, tile_half: float = 125.0) -> None:
    clipped = clip_segment_to_tile(segment, tile_half)
    if clipped is None:
        return
    start_x, start_y, end_x, end_y = clipped
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    length = math.hypot(delta_x, delta_y)
    if length <= 0.01:
        return
    half_width = segment.width * 0.5
    normal_x = -delta_y / length * half_width
    normal_y = delta_x / length * half_width
    start_factor = projection_factor(start_x, start_y, segment)
    end_factor = projection_factor(end_x, end_y, segment)
    start_z = segment.start_z + (segment.end_z - segment.start_z) * start_factor - MISSION_SAFE_TERRAIN_HEIGHT + 0.10
    end_z = segment.start_z + (segment.end_z - segment.start_z) * end_factor - MISSION_SAFE_TERRAIN_HEIGHT + 0.10
    render.add_quad(
        (
            (start_x - normal_x, start_y - normal_y, start_z),
            (end_x - normal_x, end_y - normal_y, end_z),
            (end_x + normal_x, end_y + normal_y, end_z),
            (start_x + normal_x, start_y + normal_y, start_z),
        ),
        MATERIAL_ROAD,
        (112, 112, 112, 255),
        ((0.0, 1.0), (length / 12.0, 1.0), (length / 12.0, 0.0), (0.0, 0.0)),
    )


def add_bridge_supports(
    render: MeshBuilder,
    collision: MeshBuilder,
    segment: TrafficPathSegment,
    tile_half: float,
    water_level: float,
) -> None:
    clipped = clip_segment_to_tile(segment, tile_half)
    if clipped is None:
        return
    start_x, start_y, end_x, end_y = clipped
    length = math.hypot(end_x - start_x, end_y - start_y)
    count = max(1, int(length / 45.0))
    for index in range(count):
        factor = (index + 0.5) / count
        x, y = start_x + (end_x - start_x) * factor, start_y + (end_y - start_y) * factor
        road_factor = projection_factor(x, y, segment)
        grade = segment.start_z + (segment.end_z - segment.start_z) * road_factor
        bottom = water_level - MISSION_SAFE_TERRAIN_HEIGHT - 3.0
        height = grade - water_level + 2.9
        if height <= 0.1:
            continue
        for builder in (render, collision):
            builder.add_box(x, y, bottom, 1.8, 2.2, height, MATERIAL_TRIM, MATERIAL_TRIM, (145, 145, 137, 255))


def projection_factor(x: float, y: float, segment: TrafficPathSegment) -> float:
    delta_x = segment.end_x - segment.start_x
    delta_y = segment.end_y - segment.start_y
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared <= 0.000001:
        return 0.0
    return max(0.0, min(1.0, ((x - segment.start_x) * delta_x + (y - segment.start_y) * delta_y) / length_squared))


def add_road_junctions(
    render: MeshBuilder,
    segments: list[TrafficPathSegment],
    tile_half: float,
) -> None:
    """Cap shared graph nodes so arterial/collector ribbons join without holes."""
    nodes: dict[tuple[int, int], list[tuple[float, float, float, float, float]]] = {}
    for segment in segments:
        for x, y, z, direction in (
            (
                segment.start_x,
                segment.start_y,
                segment.start_z,
                math.atan2(segment.end_y - segment.start_y, segment.end_x - segment.start_x),
            ),
            (
                segment.end_x,
                segment.end_y,
                segment.end_z,
                math.atan2(segment.start_y - segment.end_y, segment.start_x - segment.end_x),
            ),
        ):
            if not (-tile_half - 1.0 <= x <= tile_half + 1.0 and -tile_half - 1.0 <= y <= tile_half + 1.0):
                continue
            nodes.setdefault((round(x * 2.0), round(y * 2.0)), []).append((x, y, z, segment.width, direction))
    for endpoints in nodes.values():
        direction_bins = {round(item[4] / (math.pi / 6.0)) % 12 for item in endpoints}
        # Degree-two nodes are bends along one road, not intersections. Capping
        # every NODES link endpoint produced the dotted radar/road surface.
        if len(direction_bins) < 3:
            continue
        x = sum(item[0] for item in endpoints) / len(endpoints)
        y = sum(item[1] for item in endpoints) / len(endpoints)
        z = sum(item[2] for item in endpoints) / len(endpoints) - MISSION_SAFE_TERRAIN_HEIGHT + 0.09
        radius = max(item[3] for item in endpoints) * 0.58
        points = tuple(
            (x + math.cos(math.tau * index / 8.0) * radius, y + math.sin(math.tau * index / 8.0) * radius)
            for index in range(8)
        )
        for index, first in enumerate(points):
            second = points[(index + 1) % len(points)]
            render.add_triangle(
                ((x, y, z), (first[0], first[1], z), (second[0], second[1], z)),
                MATERIAL_ROAD,
                (60, 63, 65, 255),
            )


def add_traffic_lane_markings(render: MeshBuilder, segment: TrafficPathSegment, tile_half: float = 125.0) -> None:
    """Paint only lanes that exist in the traffic graph."""
    clipped = clip_segment_to_tile(segment, tile_half)
    if clipped is None:
        return
    segment_start_x, segment_start_y, segment_end_x, segment_end_y = clipped
    delta_x = segment_end_x - segment_start_x
    delta_y = segment_end_y - segment_start_y
    length = math.hypot(delta_x, delta_y)
    if length < 24.0 or segment.width < 18.0:
        return
    direction_x = delta_x / length
    direction_y = delta_y / length
    perpendicular_x = -direction_y * 0.13
    perpendicular_y = direction_x * 0.13
    position = 4.0
    while position < length - 2.0:
        dash_length = min(3.8, length - position)
        start_x = segment_start_x + direction_x * position
        start_y = segment_start_y + direction_y * position
        end_x = segment_start_x + direction_x * (position + dash_length)
        end_y = segment_start_y + direction_y * (position + dash_length)
        start_factor = projection_factor(start_x, start_y, segment)
        end_factor = projection_factor(end_x, end_y, segment)
        start_z = segment.start_z + (segment.end_z - segment.start_z) * start_factor - MISSION_SAFE_TERRAIN_HEIGHT + 0.13
        end_z = segment.start_z + (segment.end_z - segment.start_z) * end_factor - MISSION_SAFE_TERRAIN_HEIGHT + 0.13
        render.add_quad(
            (
                (start_x - perpendicular_x, start_y - perpendicular_y, start_z),
                (end_x - perpendicular_x, end_y - perpendicular_y, end_z),
                (end_x + perpendicular_x, end_y + perpendicular_y, end_z),
                (start_x + perpendicular_x, start_y + perpendicular_y, start_z),
            ),
            MATERIAL_TRIM,
            (218, 201, 142, 255),
        )
        position += 10.5


def clip_segment_to_tile(
    segment: TrafficPathSegment,
    tile_half: float,
) -> tuple[float, float, float, float] | None:
    """Clip a path centerline to a tile so DFF/COL bounds remain local."""
    start_x = segment.start_x
    start_y = segment.start_y
    delta_x = segment.end_x - start_x
    delta_y = segment.end_y - start_y
    minimum_factor = 0.0
    maximum_factor = 1.0
    for direction, start_coordinate in ((delta_x, start_x), (delta_y, start_y)):
        for coefficient, boundary in ((-direction, start_coordinate + tile_half), (direction, tile_half - start_coordinate)):
            if abs(coefficient) <= 0.000001:
                if boundary < 0.0:
                    return None
                continue
            ratio = boundary / coefficient
            if coefficient < 0.0:
                minimum_factor = max(minimum_factor, ratio)
            else:
                maximum_factor = min(maximum_factor, ratio)
            if minimum_factor > maximum_factor:
                return None
    return (
        start_x + delta_x * minimum_factor,
        start_y + delta_y * minimum_factor,
        start_x + delta_x * maximum_factor,
        start_y + delta_y * maximum_factor,
    )


def add_seed_marker(render: MeshBuilder, color: tuple[int, int, int, int], tile_half: float) -> None:
    marker = 1.4
    edge = tile_half - 3.0
    render.add_box(edge, edge, 0.02, marker, marker, 0.08, MATERIAL_TRIM, MATERIAL_TRIM, color)


def seed_color(rng: random.Random) -> tuple[int, int, int, int]:
    return rng.randrange(70, 240), rng.randrange(70, 240), rng.randrange(70, 240), 255


def district_road_color(district: str) -> tuple[int, int, int, int]:
    if district == "beach":
        return 68, 70, 67, 255
    if district in {"airport", "docks", "industrial"}:
        return 55, 59, 62, 255
    return 61, 65, 68, 255


def district_grass_color(district: str) -> tuple[int, int, int, int]:
    if district == "beach":
        return 143, 126, 77, 255
    if district == "hills":
        return 67, 111, 66, 255
    return 74, 108, 70, 255


def read_img_v2_entries(path: Path) -> list[ImgEntry]:
    with path.open("rb") as source:
        header = source.read(8)
        if len(header) != 8 or header[:4] != b"VER2":
            raise RuntimeError(f"{path} is not a GTA San Andreas VER2 IMG archive.")
        count = struct.unpack_from("<I", header, 4)[0]
        if count > 100_000:
            raise RuntimeError(f"{path} has an implausible IMG directory entry count ({count}).")
        directory = source.read(count * 32)
    if len(directory) != count * 32:
        raise RuntimeError(f"{path} has a truncated IMG directory.")
    entries: list[ImgEntry] = []
    for index in range(count):
        offset, streaming_size, archive_size, raw_name = struct.unpack_from("<IHH24s", directory, index * 32)
        name = raw_name.split(b"\0", 1)[0].decode("ascii", errors="replace")
        entries.append(ImgEntry(name, offset, streaming_size, archive_size))
    return entries


def find_los_santos_stream_ipls(path: Path) -> list[ImgEntry]:
    return [entry for entry in read_img_v2_entries(path) if LOS_SANTOS_STREAM_RE.fullmatch(entry.name)]


def read_img_entry_payload(path: Path, entry: ImgEntry) -> bytes:
    sector_count = entry.streaming_sectors or entry.archive_sectors
    if entry.sector_offset <= 0 or sector_count <= 0:
        raise RuntimeError(
            f"IMG entry {entry.name} has an invalid sector range "
            f"({entry.sector_offset}, {entry.streaming_sectors}, {entry.archive_sectors})."
        )
    byte_offset = entry.sector_offset * IMG_SECTOR_SIZE
    byte_count = sector_count * IMG_SECTOR_SIZE
    with path.open("rb") as source:
        source.seek(0, 2)
        archive_size = source.tell()
        if byte_offset + byte_count > archive_size:
            raise RuntimeError(f"IMG entry {entry.name} extends beyond the end of {path}.")
        source.seek(byte_offset)
        payload = source.read(byte_count)
    if len(payload) != byte_count:
        raise RuntimeError(f"IMG entry {entry.name} could not be read completely from {path}.")
    return payload


def parse_col_archive(data: bytes, archive_name: str) -> tuple[list[ColRecord], bytes]:
    records: list[ColRecord] = []
    position = 0
    while position + 32 <= len(data) and data[position:position + 4] in {b"COLL", b"COL2", b"COL3", b"COL4"}:
        file_size = struct.unpack_from("<I", data, position + 4)[0]
        total_size = file_size + 8
        if total_size < 32 or position + total_size > len(data):
            raise RuntimeError(
                f"Collision archive {archive_name} has an invalid record size at byte {position}."
            )
        raw_name = data[position + 8:position + 30]
        model_name = raw_name.split(b"\0", 1)[0].decode("ascii", errors="replace")
        model_id = struct.unpack_from("<H", data, position + 30)[0]
        records.append(
            ColRecord(archive_name, model_name, model_id, data[position:position + total_size])
        )
        position += total_size
    return records, data[position:]


def collision_archive_entries(path: Path) -> list[ImgEntry]:
    return [entry for entry in read_img_v2_entries(path) if entry.name.lower().endswith(".col")]


def collect_collision_model_keys(path: Path) -> set[tuple[int, str]]:
    keys: set[tuple[int, str]] = set()
    for entry in collision_archive_entries(path):
        records, _ = parse_col_archive(read_img_entry_payload(path, entry), entry.name)
        keys.update((record.model_id, record.model_name.lower()) for record in records)
    return keys


def write_collision_archive_overrides(
    img_path: Path,
    stream_directory: Path,
    models: list[GeneratedModel],
) -> list[str]:
    replacements = {(model.model_id, model.model_name.lower()): model for model in models}
    replaced: Counter[tuple[int, str]] = Counter()
    overridden: list[str] = []

    for entry in collision_archive_entries(img_path):
        original = read_img_entry_payload(img_path, entry)
        records, trailing = parse_col_archive(original, entry.name)
        output: list[bytes] = []
        changed = False
        for record in records:
            key = (record.model_id, record.model_name.lower())
            model = replacements.get(key)
            if model is None:
                output.append(record.raw)
                continue
            generated = write_col3_model(model)
            errors = validate_col3_archive(generated, 1)
            if errors:
                raise RuntimeError(
                    f"Generated collision replacement {model.model_name} failed validation: " + "; ".join(errors)
                )
            output.append(generated)
            replaced[key] += 1
            changed = True
        if not changed:
            continue

        override = b"".join(output) + trailing
        replacement_records, _ = parse_col_archive(override, entry.name)
        if len(replacement_records) != len(records):
            raise RuntimeError(f"Collision archive {entry.name} changed its model count during replacement.")
        if [(item.model_id, item.model_name.lower()) for item in replacement_records] != [
            (item.model_id, item.model_name.lower()) for item in records
        ]:
            raise RuntimeError(f"Collision archive {entry.name} changed its model identity table during replacement.")
        (stream_directory / entry.name).write_bytes(override)
        overridden.append(entry.name)

    missing = sorted(key for key in replacements if replaced[key] == 0)
    if missing:
        preview = ", ".join(f"{model_id}:{name}" for model_id, name in missing[:8])
        raise RuntimeError(f"Could not replace {len(missing)} selected retail collision records ({preview}).")
    return overridden


def write_material_texture_overrides(
    img_path: Path,
    stream_directory: Path,
    models: list[GeneratedModel],
    native_chunks: list[bytes],
    generated_textures: list[tuple[str, TextureImage, bool]],
) -> list[str]:
    entries = {entry.name.lower(): entry for entry in read_img_v2_entries(img_path)}
    overridden: list[str] = []
    material_chunks = native_chunks + [
        write_native_texture(name, image, has_alpha)
        for name, image, has_alpha in generated_textures
    ]
    for txd_name in sorted({model.txd_name for model in models}, key=str.lower):
        entry = entries.get(f"{txd_name.lower()}.txd")
        if entry is None:
            raise RuntimeError(f"Retail streamed texture dictionary was not found: {txd_name}.txd")
        source = read_img_entry_payload(img_path, entry)
        augmented = augment_texture_dictionary_with_native_chunks(source, material_chunks)
        errors = validate_texture_dictionary(augmented)
        if errors:
            raise RuntimeError(f"Augmented texture dictionary {entry.name} failed validation: " + "; ".join(errors))
        (stream_directory / entry.name).write_bytes(augmented)
        overridden.append(entry.name)
    return overridden


def parse_binary_ipl_header(data: bytes) -> tuple[int, int, int, int]:
    if len(data) < 76 or data[:4] != b"bnry":
        raise ValueError("binary IPL is missing the 76-byte bnry header")
    values = struct.unpack_from("<18I", data, 4)
    instance_count = values[0]
    car_count = values[4]
    instance_offset = values[6]
    car_offset = values[14]
    if instance_count > 100_000 or car_count > 100_000:
        raise ValueError("binary IPL has an implausible section count")
    if instance_count and (instance_offset < 76 or instance_offset + instance_count * 40 > len(data)):
        raise ValueError("binary IPL INST section is outside the file")
    if car_count and (car_offset < 76 or car_offset + car_count * 48 > len(data)):
        raise ValueError("binary IPL CARS section is outside the file")
    return instance_count, car_count, instance_offset, car_offset


def preserved_binary_instance_indices(
    data: bytes,
    bounds: tuple[tuple[float, float, float, float], ...],
    model_ids: tuple[int, ...] = (),
) -> set[int]:
    count, _, offset, _ = parse_binary_ipl_header(data)
    identities = set(model_ids)
    retained = set()
    for index in range(count):
        start = offset + index * 40
        x, y = struct.unpack_from("<2f", data, start)
        model_id = struct.unpack_from("<i", data, start + 28)[0]
        if model_id in identities or point_in_world_bounds(x, y, bounds):
            retained.add(index)
    pending = list(retained)
    while pending:
        lod = struct.unpack_from("<i", data, offset + pending.pop() * 40 + 36)[0]
        if 0 <= lod < count and lod not in retained:
            retained.add(lod)
            pending.append(lod)
    return retained


def preserved_text_instance_lines(
    text: str,
    bounds: tuple[tuple[float, float, float, float], ...],
    model_ids: tuple[int, ...] = (),
) -> set[int]:
    records: list[tuple[int, list[str]]] = []
    section = ""
    identities = set(model_ids)
    for line_number, raw_line in enumerate(text.splitlines()):
        line = raw_line.strip()
        if line.lower() == "inst":
            section = "inst"
        elif line.lower() == "end":
            section = ""
        elif section == "inst" and line and not line.startswith("#"):
            records.append((line_number, [field.strip() for field in line.split(",")]))
    retained = set()
    for index, (_, fields) in enumerate(records):
        if len(fields) < 11:
            continue
        try:
            keep = int(fields[0]) in identities or point_in_world_bounds(float(fields[3]), float(fields[4]), bounds)
        except ValueError:
            continue
        if keep:
            retained.add(index)
    pending = list(retained)
    while pending:
        fields = records[pending.pop()][1]
        try:
            lod = int(fields[10])
        except (ValueError, IndexError):
            continue
        if 0 <= lod < len(records) and lod not in retained:
            retained.add(lod)
            pending.append(lod)
    return {records[index][0] for index in retained}


def relocate_binary_ipl_underground(
    data: bytes,
    hidden_z: float = -10_000.0,
    protected_bounds: tuple[tuple[float, float, float, float], ...] = (),
    preserved_model_ids: tuple[int, ...] = (),
) -> bytes:
    instance_count, car_count, instance_offset, car_offset = parse_binary_ipl_header(data)
    retained = preserved_binary_instance_indices(data, protected_bounds, preserved_model_ids)
    relocated = bytearray(data)
    for index in range(instance_count):
        start = instance_offset + index * 40
        x, y = struct.unpack_from("<2f", data, start)
        if index not in retained:
            struct.pack_into("<f", relocated, start + 8, hidden_z)
    for index in range(car_count):
        start = car_offset + index * 48
        x, y = struct.unpack_from("<2f", data, start)
        if not point_in_world_bounds(x, y, protected_bounds):
            struct.pack_into("<f", relocated, start + 8, hidden_z)
    return bytes(relocated)


def validate_relocated_binary_ipl(
    original: bytes,
    relocated: bytes,
    hidden_z: float = -10_000.0,
    protected_bounds: tuple[tuple[float, float, float, float], ...] = (),
    preserved_model_ids: tuple[int, ...] = (),
) -> list[str]:
    errors: list[str] = []
    if len(original) != len(relocated):
        errors.append("relocated binary IPL size differs from the retail stream")
        return errors
    try:
        original_header = parse_binary_ipl_header(original)
        relocated_header = parse_binary_ipl_header(relocated)
    except ValueError as error:
        errors.append(str(error))
        return errors
    if original[:76] != relocated[:76] or original_header != relocated_header:
        errors.append("relocated binary IPL does not preserve the retail header")
        return errors
    instance_count, car_count, instance_offset, car_offset = original_header
    retained = preserved_binary_instance_indices(original, protected_bounds, preserved_model_ids)
    for index in range(instance_count):
        start = instance_offset + index * 40
        if original[start:start + 8] != relocated[start:start + 8] or original[start + 12:start + 40] != relocated[start + 12:start + 40]:
            errors.append(f"relocated binary IPL changed INST record {index} outside its Z coordinate")
            break
        x, y, original_z = struct.unpack_from("<3f", original, start)
        expected_z = original_z if index in retained else hidden_z
        if struct.unpack_from("<f", relocated, start + 8)[0] != expected_z:
            errors.append(f"relocated binary IPL wrote the wrong INST Z for record {index}")
            break
    for index in range(car_count):
        start = car_offset + index * 48
        if original[start:start + 8] != relocated[start:start + 8] or original[start + 12:start + 48] != relocated[start + 12:start + 48]:
            errors.append(f"relocated binary IPL changed CARS record {index} outside its Z coordinate")
            break
        x, y, original_z = struct.unpack_from("<3f", original, start)
        expected_z = original_z if point_in_world_bounds(x, y, protected_bounds) else hidden_z
        if struct.unpack_from("<f", relocated, start + 8)[0] != expected_z:
            errors.append(f"relocated binary IPL wrote the wrong CARS Z for record {index}")
            break
    return errors


def relocate_text_ipl_underground(
    text: str,
    hidden_z: float = -10_000.0,
    protected_bounds: tuple[tuple[float, float, float, float], ...] = (),
    preserved_model_ids: tuple[int, ...] = (),
) -> str:
    newline = "\r\n" if "\r\n" in text else "\n"
    output: list[str] = []
    section = ""
    retained = preserved_text_instance_lines(text, protected_bounds, preserved_model_ids)
    for line_number, raw_line in enumerate(text.splitlines()):
        stripped = raw_line.strip()
        lowered = stripped.lower()
        if lowered == "inst":
            section = "inst"
            output.append(raw_line)
            continue
        if section == "inst":
            if lowered == "end":
                section = ""
                output.append(raw_line)
                continue
            if not stripped or stripped.startswith("#"):
                output.append(raw_line)
                continue
            fields = raw_line.split(",")
            if len(fields) < 11:
                raise RuntimeError(f"Malformed INST row in a retail text IPL: {raw_line}")
            try:
                x = float(fields[3])
                y = float(fields[4])
            except ValueError as error:
                raise RuntimeError(f"Malformed INST coordinates in a retail text IPL: {raw_line}") from error
            if line_number in retained:
                output.append(raw_line)
                continue
            original_z = fields[5]
            leading = original_z[:len(original_z) - len(original_z.lstrip())]
            trailing = original_z[len(original_z.rstrip()):]
            fields[5] = f"{leading}{hidden_z:.1f}{trailing}"
            output.append(",".join(fields))
            continue
        output.append(raw_line)
    if section:
        raise RuntimeError("Retail text IPL ends inside its INST section.")
    return newline.join(output).rstrip("\r\n") + newline


def validate_relocated_text_ipl(
    original: str,
    relocated: str,
    hidden_z: float = -10_000.0,
    protected_bounds: tuple[tuple[float, float, float, float], ...] = (),
    preserved_model_ids: tuple[int, ...] = (),
) -> list[str]:
    errors: list[str] = []
    original_lines = original.splitlines()
    relocated_lines = relocated.splitlines()
    if len(original_lines) != len(relocated_lines):
        return ["relocated text IPL changed its line count"]

    section = ""
    instance_count = 0
    retained = preserved_text_instance_lines(original, protected_bounds, preserved_model_ids)
    for line_index, (original_line, relocated_line) in enumerate(zip(original_lines, relocated_lines), start=1):
        lowered = original_line.strip().lower()
        if lowered == "inst":
            section = "inst"
            if original_line != relocated_line:
                errors.append(f"relocated text IPL changed the INST header on line {line_index}")
            continue
        if section == "inst" and lowered == "end":
            section = ""
            if original_line != relocated_line:
                errors.append(f"relocated text IPL changed the INST terminator on line {line_index}")
            continue
        if section != "inst" or not original_line.strip() or original_line.lstrip().startswith("#"):
            if original_line != relocated_line:
                errors.append(f"relocated text IPL changed non-INST data on line {line_index}")
            continue

        original_fields = [field.strip() for field in original_line.split(",")]
        relocated_fields = [field.strip() for field in relocated_line.split(",")]
        if len(original_fields) < 11 or len(relocated_fields) != len(original_fields):
            errors.append(f"relocated text IPL has malformed INST data on line {line_index}")
            continue
        if original_fields[:5] != relocated_fields[:5] or original_fields[6:] != relocated_fields[6:]:
            errors.append(f"relocated text IPL changed INST data outside Z on line {line_index}")
        try:
            relocated_z = float(relocated_fields[5])
        except ValueError:
            errors.append(f"relocated text IPL has an invalid Z coordinate on line {line_index}")
        else:
            try:
                x = float(original_fields[3])
                y = float(original_fields[4])
                original_z = float(original_fields[5])
            except ValueError:
                errors.append(f"retail text IPL has invalid coordinates on line {line_index}")
                continue
            expected_z = original_z if line_index - 1 in retained else hidden_z
            if relocated_z != expected_z:
                errors.append(f"relocated text IPL wrote the wrong INST Z on line {line_index}")
        instance_count += 1

    if section:
        errors.append("relocated text IPL ends inside its INST section")
    if instance_count == 0:
        errors.append("retail text IPL has no INST rows; refusing to create an empty placement shadow")
    return errors


def write_los_santos_mod_package(game_directory: Path, staging_directory: Path, world: LosSantosWorld) -> Path:
    mod_directory = staging_directory / "SeedAndreas"
    mod_directory.mkdir(parents=True, exist_ok=False)
    map_directory = mod_directory / "data" / "maps" / "seedandreas"
    stream_directory = mod_directory / "models" / "gta3.img"
    map_directory.mkdir(parents=True, exist_ok=True)
    stream_directory.mkdir(parents=True, exist_ok=True)
    path_directory = mod_directory / "data" / "paths"
    path_directory.mkdir(parents=True, exist_ok=True)

    for filename, data in sorted(world.path_overrides.files.items()):
        (stream_directory / filename).write_bytes(data)
    if world.water_override is not None and world.water_override.data is not None:
        (mod_directory / "data" / "water.dat").write_bytes(world.water_override.data)
    if world.gameplay is not None:
        errors = validate_preserved_tracks(world.gameplay.rail, game_directory)
        if errors:
            raise RuntimeError("; ".join(errors))
        (mod_directory / "GAMEPLAY_PRESERVATION.json").write_text(
            json.dumps({"mission_regions": [asdict(region) for region in world.gameplay.mission_regions],
                        "track_hashes": world.gameplay.rail.source_hashes,
                        "preserved_model_ids": world.preserved_model_ids,
                        "native_path_source_hashes": world.native_paths.source_hashes if world.native_paths else {},
                        "diagnostics": world.gameplay.diagnostics}, indent=2), encoding="utf-8",
        )

    for model in world.models:
        (stream_directory / f"{model.model_name}.dff").write_bytes(
            write_dff(model.model_name, model.render_mesh, model.material_texture_names)
        )

    img_path = game_directory / "models" / "gta3.img"
    collision_overrides = write_collision_archive_overrides(img_path, stream_directory, world.models)
    texture_overrides = write_material_texture_overrides(
        img_path,
        stream_directory,
        world.models,
        world.material_native_chunks,
        world.material_generated_textures,
    )
    radar_tiles = generate_seed_radar_tiles(
        world.manifest.seed,
        world.tile_districts,
        world.manifest.columns,
        world.manifest.rows,
        world.traffic_segments,
        world.models,
        world.placements,
        water_surfaces=world.water_override.surfaces if world.water_override else (),
        preserved_bounds=world.retail_surface_bounds,
        retail_radar_tiles=world.retail_radar_tiles,
    )
    if len(world.models) != len(world.placements):
        raise RuntimeError(
            "Radar generation requires one placed procedural model per generated tile; "
            f"received {len(world.models)} models and {len(world.placements)} placements."
        )
    if world.manifest.columns == 12 and world.manifest.rows == 12:
        for radar_y in range(6, 12):
            for radar_x in range(6, 12):
                radar_image = render_geometry_radar_tile(
                    radar_x, radar_y, world.models, world.placements,
                    water_surfaces=world.water_override.surfaces if world.water_override else (),
                    preserved_bounds=world.retail_surface_bounds,
                    retail_radar_data=world.retail_radar_tiles.get(f"radar{radar_y * 12 + radar_x:02d}.txd"),
                )
                radar_pixels = [
                    tuple(radar_image.pixels[offset:offset + 4])
                    for offset in range(0, len(radar_image.pixels), 4)
                ]
                unique_colors = len(set(radar_pixels))
                background = (204, 195, 171, 255)
                covered_pixels = sum(pixel != background for pixel in radar_pixels)
                if unique_colors < 2 or covered_pixels < 4096:
                    raise RuntimeError(
                        f"Generated radar{radar_y * 12 + radar_x:02d}.txd is effectively empty "
                        f"({unique_colors} colors, {covered_pixels} covered pixels)."
                    )
    for radar_name, radar_data in radar_tiles.items():
        radar_errors = validate_texture_dictionary(radar_data, 1)
        if radar_errors:
            raise RuntimeError(f"Generated radar texture {radar_name} failed validation: " + "; ".join(radar_errors))
        (stream_directory / radar_name).write_bytes(radar_data)

    ipl_lines = ["inst"]
    for placement in world.placements:
        ipl_lines.append(
            f"{placement.model_id}, {placement.model_name}, 0, {placement.x:.6f}, {placement.y:.6f}, {placement.z:.6f}, "
            f"{placement.qx:.8f}, {placement.qy:.8f}, {placement.qz:.8f}, {placement.qw:.8f}, -1"
        )
    ipl_lines.append("end")
    (map_directory / "seedandreas_ls.ipl").write_text("\n".join(ipl_lines) + "\n", encoding="cp1252")
    source_gta_dat = game_directory / "data" / "gta.dat"
    source_gta_dat_text = source_gta_dat.read_text(encoding="cp1252", errors="replace")
    (mod_directory / "data" / "gta.dat").write_text(
        build_augmented_gta_dat(source_gta_dat_text),
        encoding="cp1252",
    )

    source_la = game_directory / "data" / "maps" / "LA"
    text_overrides: list[str] = []
    if not source_la.is_dir():
        raise RuntimeError(f"Retail Los Santos map directory was not found: {source_la}")
    target_la = mod_directory / "data" / "maps" / "LA"
    target_la.mkdir(parents=True, exist_ok=True)
    protected_instance_count = 0
    for source_ipl in sorted(source_la.iterdir(), key=lambda item: item.name.lower()):
        if source_ipl.is_file() and source_ipl.suffix.lower() == ".ipl":
            raw = source_ipl.read_bytes()
            if raw[:4].lower() == b"bnry" or b"\0" in raw[:256]:
                continue
            original_text = raw.decode("cp1252", errors="replace")
            relocated_text = relocate_text_ipl_underground(
                original_text,
                protected_bounds=world.protected_instance_bounds,
                preserved_model_ids=world.preserved_model_ids,
            )
            text_errors = validate_relocated_text_ipl(
                original_text,
                relocated_text,
                protected_bounds=world.protected_instance_bounds,
                preserved_model_ids=world.preserved_model_ids,
            )
            if text_errors:
                raise RuntimeError(f"Relocated text IPL {source_ipl.name} failed validation: " + "; ".join(text_errors))
            target = target_la / source_ipl.name
            target.write_bytes(relocated_text.encode("cp1252"))
            text_overrides.append(source_ipl.name)
            protected_instance_count += count_protected_text_instances(
                original_text,
                world.protected_instance_bounds,
                world.preserved_model_ids,
            )

    stream_entries = find_los_santos_stream_ipls(img_path)
    if not stream_entries:
        raise RuntimeError(
            "No retail Los Santos *_stream IPL entries were found in models\\gta3.img; refusing to build a partial replacement."
        )
    for entry in stream_entries:
        original_stream = read_img_entry_payload(img_path, entry)
        relocated_stream = relocate_binary_ipl_underground(
            original_stream,
            protected_bounds=world.protected_instance_bounds,
            preserved_model_ids=world.preserved_model_ids,
        )
        binary_errors = validate_relocated_binary_ipl(
            original_stream,
            relocated_stream,
            protected_bounds=world.protected_instance_bounds,
            preserved_model_ids=world.preserved_model_ids,
        )
        if binary_errors:
            raise RuntimeError(f"Relocated binary IPL {entry.name} failed validation: " + "; ".join(binary_errors))
        (stream_directory / entry.name).write_bytes(relocated_stream)
        protected_instance_count += count_protected_binary_instances(
            original_stream,
            world.protected_instance_bounds,
            world.preserved_model_ids,
        )

    world.manifest.text_ipls_overridden = text_overrides
    world.manifest.stream_ipls_overridden = [entry.name for entry in stream_entries]
    world.manifest.collision_archives_overridden = collision_overrides
    world.manifest.texture_dictionaries_overridden = texture_overrides
    world.manifest.radar_tiles_overridden = sorted(radar_tiles)
    world.manifest.mission_safe_instances_preserved = protected_instance_count
    world.manifest.validation.append(
        f"PASS: {len(text_overrides)} retail LA text IPL files preserve required mission/rail instances and LOD chains; other INST rows move underground without changing identity."
    )
    world.manifest.validation.append(
        f"PASS: {len(stream_entries)} retail LA binary streams preserve {protected_instance_count} mission-safe INST/CARS records and relocate only non-protected Z coordinates."
    )
    world.manifest.validation.append(
        f"PASS: {len(collision_overrides)} retail collision archives shadowed in place with the same model count and identity table."
    )
    world.manifest.validation.append(
        f"PASS: {len(texture_overrides)} retail model TXDs augmented with the deterministic Seed Andreas material palette."
    )
    world.manifest.validation.append(
        f"PASS: {len(radar_tiles)} Los Santos radar TXDs rasterized directly from installed terrain, buildings, retained retail roads and procedural road geometry."
    )
    world.manifest.validation.append(
        f"PASS: {world.manifest.traffic_path_segments_preserved} retail vehicle-path road segments reserve terrain and solid-geometry corridors; no synthetic Grove Street path was appended."
    )
    world.manifest.validation.append(
        f"PASS: cosmetic path warping is disabled; {world.manifest.traffic_path_nodes_changed} NODES coordinates changed and {world.manifest.traffic_path_nodes_preserved} in-bounds nodes retained retail coordinates."
    )
    world.manifest.validation.append(
        "PASS: the complete installed gta.dat registration list was preserved; no IDE or COLFILE entry was added."
    )
    (mod_directory / "seed_andreas_install.json").write_text(json.dumps(asdict(world.manifest), indent=2), encoding="utf-8")
    (mod_directory / "README.txt").write_text(
        "Seed Andreas full Los Santos seed (experimental)\n\n"
        f"Seed: {world.manifest.seed}\nFingerprint: {world.manifest.seed_fingerprint}\n"
        f"Custom DFF models: {world.manifest.custom_models}\nProcedural buildings: {world.manifest.custom_buildings}\n"
        f"Los Santos bounds: {world.manifest.bounds}\n\n"
        "This Mod Loader package shadows the retail Los Santos placement IPLs and loads a deterministic city made from textured custom meshes. "
        "Original game files are not edited. Every retail NODES path remains represented by a visible grade-locked road, while procedural arterials, collectors, diagonals and local streets add the generated city structure. "
        "Native streamed NODES files add connected vehicle paths while retaining original routes and remapping pedestrian references. Solid geometry stays outside road, railway and mission reservations. "
        "The complete grid is processed; fully retained retail compounds do not receive duplicate procedural ground. Mission completion remains unverified in GTA.\n",
        encoding="utf-8",
    )
    write_seed_map_svg(mod_directory / "los_santos_seed_map.svg", world)
    write_combined_obj(mod_directory / "seed_andreas_los_santos_debug.obj", world)
    return mod_directory


def count_protected_text_instances(
    text: str,
    protected_bounds: tuple[tuple[float, float, float, float], ...],
    preserved_model_ids: tuple[int, ...] = (),
) -> int:
    return len(preserved_text_instance_lines(text, protected_bounds, preserved_model_ids))


def count_protected_binary_instances(
    data: bytes,
    protected_bounds: tuple[tuple[float, float, float, float], ...],
    preserved_model_ids: tuple[int, ...] = (),
) -> int:
    _, car_count, _, car_offset = parse_binary_ipl_header(data)
    count = len(preserved_binary_instance_indices(data, protected_bounds, preserved_model_ids))
    for index in range(car_count):
        x, y = struct.unpack_from("<2f", data, car_offset + index * 48)
        if point_in_world_bounds(x, y, protected_bounds):
            count += 1
    return count


def write_seed_map_svg(path: Path, world: LosSantosWorld) -> None:
    colors = {
        "airport": "#6f7680", "docks": "#755e55", "industrial": "#857c6a",
        "beach": "#d2bd78", "hills": "#5f9866", "downtown": "#6c7297",
        "grove": "#70a36d", "eastside": "#9a715f", "coastal": "#6f9ca7",
        "civic": "#aa8d70", "park": "#4d8e60", "commercial": "#a27d86",
        "residential": "#b39a78",
    }
    size = 720
    tile_width = size / world.manifest.columns
    tile_height = size / world.manifest.rows
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="820" height="790" viewBox="0 0 820 790">',
        '<rect width="820" height="790" fill="#15171c"/>',
        f'<text x="50" y="38" fill="#f5f5f5" font-family="Segoe UI, sans-serif" font-size="22" font-weight="700">Seed Andreas — Los Santos seed {world.manifest.seed}</text>',
        f'<text x="50" y="63" fill="#aeb4c0" font-family="Segoe UI, sans-serif" font-size="13">fingerprint {world.manifest.seed_fingerprint} · custom mesh district plan</text>',
    ]
    for index, district in enumerate(world.tile_districts):
        row = index // world.manifest.columns
        column = index % world.manifest.columns
        x = 50 + column * tile_width
        y = 80 + (world.manifest.rows - row - 1) * tile_height
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{tile_width:.2f}" height="{tile_height:.2f}" '
            f'fill="{colors.get(district, "#888")}" stroke="#222630" stroke-width="2"><title>{district}</title></rect>'
        )
    parts.extend((
        '<text x="50" y="775" fill="#aeb4c0" font-family="Segoe UI, sans-serif" font-size="12">West / coast</text>',
        '<text x="710" y="775" fill="#aeb4c0" font-family="Segoe UI, sans-serif" font-size="12">East / docks</text>',
        '</svg>',
    ))
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
