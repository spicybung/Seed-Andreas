from __future__ import annotations

from collections import defaultdict, deque
from math import hypot

from .model import GeneratedWorld, RoadEdge, ValidationIssue, Vec3


def validate_world(world: GeneratedWorld) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    node_lookup = {node.node_id: node for node in world.road_nodes}
    anchor_lookup = {anchor.anchor_id: anchor for anchor in world.anchors}

    issues.extend(validate_references(world))
    issues.extend(validate_road_connectivity(world))
    issues.extend(validate_checkpoints(world, node_lookup))
    issues.extend(validate_entrances(world, node_lookup))
    issues.extend(validate_cull_zones(world, anchor_lookup))
    issues.extend(validate_buildings(world, node_lookup))
    issues.extend(validate_deterministic_ids(world))
    return issues


def validate_references(world: GeneratedWorld) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    node_ids = {node.node_id for node in world.road_nodes}
    anchor_ids = {anchor.anchor_id for anchor in world.anchors}

    for edge in world.road_edges:
        if edge.start_node not in node_ids or edge.end_node not in node_ids:
            issues.append(error("BROKEN_ROAD_REFERENCE", edge.edge_id, "Road endpoint does not exist."))

    for entrance in world.entrances:
        if entrance.anchor_id not in anchor_ids:
            issues.append(error("BROKEN_ENEX_ANCHOR", entrance.entrance_id, "ENEX anchor does not exist."))

    for checkpoint in world.checkpoints:
        if checkpoint.anchor_id not in anchor_ids:
            issues.append(error("BROKEN_CHECKPOINT_ANCHOR", checkpoint.checkpoint_id, "Checkpoint anchor does not exist."))

    for cull in world.cull_zones:
        if cull.anchor_id not in anchor_ids:
            issues.append(error("BROKEN_CULL_ANCHOR", cull.cull_id, "CULL anchor does not exist."))
    return issues


def validate_road_connectivity(world: GeneratedWorld) -> list[ValidationIssue]:
    if not world.road_nodes:
        return [error("EMPTY_ROAD_GRAPH", "world", "No road nodes were generated.")]

    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in world.road_edges:
        adjacency[edge.start_node].add(edge.end_node)
        adjacency[edge.end_node].add(edge.start_node)

    visited: set[str] = set()
    queue = deque([world.road_nodes[0].node_id])
    while queue:
        node_id = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)
        queue.extend(adjacency[node_id] - visited)

    missing = [node.node_id for node in world.road_nodes if node.node_id not in visited]
    if missing:
        return [error("DISCONNECTED_ROAD_GRAPH", "world", f"{len(missing)} road nodes are disconnected.")]
    return []


def validate_checkpoints(world: GeneratedWorld, node_lookup: dict[str, object]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for checkpoint in world.checkpoints:
        if checkpoint.vehicle_class == "aircraft":
            continue
        distance = nearest_road_distance(checkpoint.position, world.road_edges, node_lookup)
        if distance > checkpoint.maximum_road_distance:
            issues.append(
                error(
                    "CHECKPOINT_OFF_ROAD",
                    checkpoint.checkpoint_id,
                    f"Checkpoint is {distance:.1f} metres from a road; maximum is {checkpoint.maximum_road_distance:.1f}.",
                )
            )
    return issues


def validate_entrances(world: GeneratedWorld, node_lookup: dict[str, object]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for entrance in world.entrances:
        road_distance = nearest_road_distance(entrance.exterior, world.road_edges, node_lookup)
        if road_distance > 130.0:
            issues.append(error("ENEX_UNREACHABLE", entrance.entrance_id, f"Exterior is {road_distance:.1f} metres from the road network."))
        if entrance.interior_id < 0:
            issues.append(error("ENEX_BAD_INTERIOR", entrance.entrance_id, "Interior ID is negative."))
    return issues


def validate_cull_zones(world: GeneratedWorld, anchor_lookup: dict[str, object]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for cull in world.cull_zones:
        if cull.size.x <= 0.0 or cull.size.y <= 0.0 or cull.size.z <= 0.0:
            issues.append(error("CULL_EMPTY_VOLUME", cull.cull_id, "CULL dimensions must all be positive."))
            continue
        anchor = anchor_lookup.get(cull.anchor_id)
        if anchor is not None and cull.center.distance_2d(anchor.generated_position) > anchor.protected_radius:
            issues.append(error("CULL_LEFT_ANCHOR", cull.cull_id, "CULL no longer overlaps its protected mission anchor."))
    return issues


def validate_buildings(world: GeneratedWorld, node_lookup: dict[str, object]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for building in world.buildings:
        if building.width <= 0.0 or building.depth <= 0.0 or building.height <= 0.0:
            issues.append(error("INVALID_BUILDING_SIZE", building.building_id, "Building dimensions must be positive."))
        for anchor in world.anchors:
            minimum_distance = anchor.protected_radius + min(building.width, building.depth) * 0.35
            if building.position.distance_2d(anchor.generated_position) < minimum_distance:
                issues.append(error("BUILDING_IN_PROTECTED_REGION", building.building_id, f"Building overlaps mission anchor {anchor.anchor_id}."))
                break
    return issues


def validate_deterministic_ids(world: GeneratedWorld) -> list[ValidationIssue]:
    groups = {
        "anchor": [item.anchor_id for item in world.anchors],
        "road node": [item.node_id for item in world.road_nodes],
        "road": [item.edge_id for item in world.road_edges],
        "building": [item.building_id for item in world.buildings],
        "ENEX": [item.entrance_id for item in world.entrances],
        "checkpoint": [item.checkpoint_id for item in world.checkpoints],
        "CULL": [item.cull_id for item in world.cull_zones],
    }
    issues: list[ValidationIssue] = []
    for label, identifiers in groups.items():
        if len(identifiers) != len(set(identifiers)):
            issues.append(error("DUPLICATE_STABLE_ID", label, f"Duplicate {label} identifier detected."))
    return issues


def nearest_road_distance(position: Vec3, edges: list[RoadEdge], node_lookup: dict[str, object]) -> float:
    nearest = float("inf")
    for edge in edges:
        start = node_lookup.get(edge.start_node)
        end = node_lookup.get(edge.end_node)
        if start is None or end is None:
            continue
        centerline_distance = point_segment_distance(position, start.position, end.position)
        surface_distance = max(0.0, centerline_distance - edge.width * 0.5)
        nearest = min(nearest, surface_distance)
    return nearest


def point_segment_distance(point: Vec3, start: Vec3, end: Vec3) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    length_squared = dx * dx + dy * dy
    if length_squared <= 0.000001:
        return point.distance_2d(start)
    projection = ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_squared
    projection = max(0.0, min(1.0, projection))
    closest_x = start.x + projection * dx
    closest_y = start.y + projection * dy
    return hypot(point.x - closest_x, point.y - closest_y)


def error(code: str, subject: str, message: str) -> ValidationIssue:
    return ValidationIssue("error", code, subject, message)
