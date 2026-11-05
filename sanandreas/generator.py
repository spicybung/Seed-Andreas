from __future__ import annotations

import math
import random

from .contracts import (
    ANCHOR_CONTRACTS,
    CHECKPOINT_CONTRACTS,
    CULL_CONTRACTS,
    ENTRANCE_CONTRACTS,
    MISSION_CONNECTIONS,
)
from .model import (
    Building,
    Checkpoint,
    City,
    CullZone,
    Entrance,
    GeneratedWorld,
    MissionAnchor,
    RoadEdge,
    RoadNode,
    Vec3,
)
from .street_graph import generate_hierarchical_street_graph
from .voice import generate_voice_variations


GENERATOR_VERSION = "0.10.0"


CITY_TEMPLATES = (
    City("ls", "Los Santos", Vec3(1750.0, -1700.0, 13.0), 880.0, 720.0, "coastal_sprawl"),
    City("sf", "San Fierro", Vec3(-1950.0, 450.0, 28.0), 720.0, 680.0, "steep_industrial_bay"),
    City("lv", "Las Venturas", Vec3(1750.0, 1450.0, 11.0), 900.0, 760.0, "desert_strip"),
)


MODEL_VARIANTS = (
    (19000, "sai_residential_01", "residential"),
    (19001, "sai_residential_02", "residential"),
    (19002, "sai_shopfront_01", "commercial"),
    (19003, "sai_office_01", "commercial"),
    (19004, "sai_warehouse_01", "industrial"),
    (19005, "sai_casino_block_01", "entertainment"),
)


def generate_world(seed: int, profile: str = "story_safe", voice_chance: float = 0.28) -> GeneratedWorld:
    random_source = random.Random(seed)
    world = GeneratedWorld(GENERATOR_VERSION, seed, profile)
    world.cities = [City(**city.__dict__) for city in CITY_TEMPLATES]
    world.anchors = generate_anchors(random_source, preserve_script_positions=profile == "story_safe")
    world.road_nodes, world.road_edges = generate_roads(random_source, world.cities, world.anchors)
    world.buildings = generate_buildings(random_source, world)
    world.entrances = generate_entrances(world.anchors)
    world.checkpoints = generate_checkpoints(world.anchors)
    world.cull_zones = generate_cull_zones(world.anchors)
    world.voices = generate_voice_variations(seed, voice_chance)
    return world


def generate_anchors(random_source: random.Random, preserve_script_positions: bool = True) -> list[MissionAnchor]:
    anchors: list[MissionAnchor] = []
    for contract in ANCHOR_CONTRACTS:
        generated = contract.original_position
        if contract.policy == "relocatable" and not preserve_script_positions:
            angle = random_source.uniform(0.0, math.tau)
            distance = random_source.uniform(contract.relocation_radius * 0.35, contract.relocation_radius)
            generated = generated.moved(math.cos(angle) * distance, math.sin(angle) * distance)
        anchors.append(
            MissionAnchor(
                anchor_id=contract.anchor_id,
                mission=contract.mission,
                city_id=contract.city_id,
                original_position=contract.original_position,
                generated_position=generated,
                protected_radius=contract.protected_radius,
                policy=contract.policy,
                required_tags=list(contract.required_tags),
            )
        )
    return anchors


def generate_roads(
    random_source: random.Random,
    cities: list[City],
    anchors: list[MissionAnchor],
) -> tuple[list[RoadNode], list[RoadEdge]]:
    nodes: list[RoadNode] = []
    edges: list[RoadEdge] = []
    nodes_by_city: dict[str, list[RoadNode]] = {}

    for city in cities:
        city_nodes: list[RoadNode] = []
        graph = generate_hierarchical_street_graph(
            random_source.getrandbits(63),
            city.center.x - city.half_width * 0.78,
            city.center.y - city.half_height * 0.78,
            city.center.x + city.half_width * 0.78,
            city.center.y + city.half_height * 0.78,
            neighborhood_count=7,
        )
        for node_index, graph_node in enumerate(graph.nodes):
            node = RoadNode(
                f"{city.city_id}_n_{node_index:04d}",
                city.city_id,
                Vec3(graph_node.x, graph_node.y, city.center.z),
            )
            nodes.append(node)
            city_nodes.append(node)

        road_types = {"local": "street", "collector": "collector", "arterial": "arterial"}
        for graph_edge in graph.segments:
            edges.append(
                make_edge(
                    city_nodes[graph_edge.start_node],
                    city_nodes[graph_edge.end_node],
                    len(edges),
                    road_types[graph_edge.street_class],
                )
            )
        nodes_by_city[city.city_id] = city_nodes

    anchor_nodes: dict[str, RoadNode] = {}
    for anchor in anchors:
        anchor_node = RoadNode(
            f"anchor_{anchor.anchor_id}",
            anchor.city_id,
            anchor.generated_position,
            "mission_anchor",
            anchor.anchor_id,
        )
        nodes.append(anchor_node)
        anchor_nodes[anchor.anchor_id] = anchor_node
        nearest = min(nodes_by_city[anchor.city_id], key=lambda node: node.position.distance_2d(anchor.generated_position))
        edges.append(make_edge(anchor_node, nearest, len(edges), "mission_access", True))

    for start_id, end_id, road_type in MISSION_CONNECTIONS:
        start = anchor_nodes[start_id]
        end = anchor_nodes[end_id]
        edges.append(make_edge(start, end, len(edges), road_type, True))

    city_center_nodes = {
        city_id: min(city_nodes, key=lambda node: node.position.distance_2d(next(city.center for city in cities if city.city_id == city_id)))
        for city_id, city_nodes in nodes_by_city.items()
    }
    edges.append(make_edge(city_center_nodes["ls"], city_center_nodes["sf"], len(edges), "intercity_highway", True))
    edges.append(make_edge(city_center_nodes["sf"], city_center_nodes["lv"], len(edges), "intercity_highway", True))
    return nodes, edges


def make_edge(start: RoadNode, end: RoadNode, index: int, road_type: str, mission_critical: bool = False) -> RoadEdge:
    width_by_type = {
        "street": (14.0, 2),
        "collector": (18.0, 2),
        "arterial": (22.0, 4),
        "mission_access": (18.0, 2),
        "mission_corridor": (20.0, 4),
        "highway": (28.0, 4),
        "intercity_highway": (30.0, 4),
    }
    width, lanes = width_by_type[road_type]
    return RoadEdge(f"road_{index:04d}", start.node_id, end.node_id, road_type, width, lanes, mission_critical)


def generate_buildings(random_source: random.Random, world: GeneratedWorld) -> list[Building]:
    buildings: list[Building] = []
    node_lookup = {node.node_id: node for node in world.road_nodes}
    protected = [(anchor.generated_position, anchor.protected_radius) for anchor in world.anchors]
    candidates = [edge for edge in world.road_edges if edge.road_type in {"street", "collector", "arterial"}]

    for edge in candidates:
        start = node_lookup[edge.start_node].position
        end = node_lookup[edge.end_node].position
        dx = end.x - start.x
        dy = end.y - start.y
        length = math.hypot(dx, dy)
        if length < 60.0:
            continue
        normal_x = -dy / length
        normal_y = dx / length
        yaw = math.degrees(math.atan2(dy, dx))
        city_id = node_lookup[edge.start_node].city_id

        for side in (-1.0, 1.0):
            along = random_source.uniform(0.32, 0.68)
            setback = edge.width * 0.5 + random_source.uniform(26.0, 42.0)
            position = Vec3(
                start.x + dx * along + normal_x * setback * side,
                start.y + dy * along + normal_y * setback * side,
                start.z,
            )
            if any(position.distance_2d(center) < radius + 22.0 for center, radius in protected):
                continue
            width = random_source.uniform(28.0, min(72.0, length * 0.45))
            depth = random_source.uniform(22.0, 48.0)
            height = choose_height(random_source, city_id)
            model_id, model_name, district = choose_model(random_source, city_id)
            buildings.append(
                Building(
                    building_id=f"building_{len(buildings):04d}",
                    city_id=city_id,
                    model_name=model_name,
                    model_id=model_id,
                    position=position,
                    width=round(width, 3),
                    depth=round(depth, 3),
                    height=round(height, 3),
                    yaw_degrees=round(yaw, 3),
                    district=district,
                )
            )
    return buildings


def choose_height(random_source: random.Random, city_id: str) -> float:
    ranges = {"ls": (8.0, 34.0), "sf": (10.0, 52.0), "lv": (9.0, 46.0)}
    low, high = ranges[city_id]
    return random_source.uniform(low, high)


def choose_model(random_source: random.Random, city_id: str) -> tuple[int, str, str]:
    choices = list(MODEL_VARIANTS)
    if city_id == "ls":
        weights = (5, 4, 3, 2, 2, 1)
    elif city_id == "sf":
        weights = (2, 2, 3, 3, 5, 1)
    else:
        weights = (2, 2, 4, 3, 2, 5)
    return random_source.choices(choices, weights=weights, k=1)[0]


def generate_entrances(anchors: list[MissionAnchor]) -> list[Entrance]:
    lookup = {anchor.anchor_id: anchor for anchor in anchors}
    entries: list[Entrance] = []
    for contract in ENTRANCE_CONTRACTS:
        anchor = lookup[contract.anchor_id]
        entries.append(
            Entrance(
                contract.entrance_id,
                contract.anchor_id,
                anchor.generated_position.moved(contract.exterior_offset.x, contract.exterior_offset.y, contract.exterior_offset.z),
                contract.interior_position,
                contract.interior_id,
                contract.facing_degrees,
                contract.name,
            )
        )
    return entries


def generate_checkpoints(anchors: list[MissionAnchor]) -> list[Checkpoint]:
    lookup = {anchor.anchor_id: anchor for anchor in anchors}
    checkpoints: list[Checkpoint] = []
    for contract in CHECKPOINT_CONTRACTS:
        anchor = lookup[contract.anchor_id]
        checkpoints.append(
            Checkpoint(
                contract.checkpoint_id,
                contract.anchor_id,
                contract.mission,
                anchor.generated_position.moved(contract.offset.x, contract.offset.y, contract.offset.z),
                contract.radius,
                contract.vehicle_class,
                contract.maximum_road_distance,
            )
        )
    return checkpoints


def generate_cull_zones(anchors: list[MissionAnchor]) -> list[CullZone]:
    lookup = {anchor.anchor_id: anchor for anchor in anchors}
    zones: list[CullZone] = []
    for contract in CULL_CONTRACTS:
        anchor = lookup[contract.anchor_id]
        zones.append(
            CullZone(
                contract.cull_id,
                contract.anchor_id,
                anchor.generated_position.moved(contract.offset.x, contract.offset.y, contract.offset.z),
                contract.size,
                contract.purpose,
                contract.flags,
            )
        )
    return zones
