from __future__ import annotations

from dataclasses import dataclass, field
import math
import random


@dataclass(frozen=True)
class StreetPoint:
    x: float
    y: float


@dataclass(frozen=True)
class StreetSegment:
    start_node: int
    end_node: int
    street_class: str


@dataclass(frozen=True)
class StreetBlock:
    points: tuple[StreetPoint, ...]
    district_index: int


@dataclass(frozen=True)
class WaterBody:
    points: tuple[StreetPoint, ...]
    water_type: str


@dataclass
class StreetGraph:
    nodes: list[StreetPoint] = field(default_factory=list)
    segments: list[StreetSegment] = field(default_factory=list)
    blocks: list[StreetBlock] = field(default_factory=list)
    water_bodies: list[WaterBody] = field(default_factory=list)


class StreetGraphBuilder:
    def __init__(self) -> None:
        self.graph = StreetGraph()
        self.node_lookup: dict[tuple[int, int], int] = {}
        self.edge_lookup: set[tuple[int, int]] = set()

    def add_node(self, point: StreetPoint) -> int:
        key = (round(point.x * 1000.0), round(point.y * 1000.0))
        existing = self.node_lookup.get(key)
        if existing is not None:
            return existing
        index = len(self.graph.nodes)
        self.graph.nodes.append(point)
        self.node_lookup[key] = index
        return index

    def add_segment(self, start: StreetPoint, end: StreetPoint, street_class: str) -> tuple[int, int]:
        if distance(start, end) < 1.0:
            node = self.add_node(start)
            return node, node
        start_node = self.add_node(start)
        end_node = self.add_node(end)
        key = tuple(sorted((start_node, end_node)))
        if key not in self.edge_lookup:
            self.graph.segments.append(StreetSegment(start_node, end_node, street_class))
            self.edge_lookup.add(key)
        return start_node, end_node


def generate_hierarchical_street_graph(
    seed: int,
    minimum_x: float,
    minimum_y: float,
    maximum_x: float,
    maximum_y: float,
    neighborhood_count: int = 8,
) -> StreetGraph:
    """Generate a connected Los-Angeles-like arterial fabric.

    Long boulevards and avenues are established first, a small number of
    diagonals cut across them, collectors attach rotated neighborhood patches,
    and only then are local streets and blocks emitted.  No block is retained
    without all four bounding local street edges.
    """
    if maximum_x <= minimum_x or maximum_y <= minimum_y:
        raise ValueError("Street graph bounds must have positive width and height.")
    if neighborhood_count < 3:
        raise ValueError("At least three neighborhoods are required.")

    random_source = random.Random(seed ^ 0x53454544524F4144)
    builder = StreetGraphBuilder()
    width = maximum_x - minimum_x
    height = maximum_y - minimum_y
    margin = min(width, height) * 0.085
    inner_minimum_x = minimum_x + margin
    inner_maximum_x = maximum_x - margin
    inner_minimum_y = minimum_y + margin
    inner_maximum_y = maximum_y - margin
    inner_width = inner_maximum_x - inner_minimum_x
    inner_height = inner_maximum_y - inner_minimum_y

    water_bodies = generate_seeded_water_bodies(seed, minimum_x, minimum_y, maximum_x, maximum_y)
    builder.graph.water_bodies.extend(water_bodies)

    def add_route(points: list[StreetPoint], street_class: str, subdivisions: int = 1) -> list[StreetPoint]:
        sampled: list[StreetPoint] = [points[0]]
        for first, second in zip(points, points[1:]):
            for step in range(1, subdivisions + 1):
                sampled.append(interpolate(first, second, step / subdivisions))
        for first, second in zip(sampled, sampled[1:]):
            if street_class != "arterial" and segment_intersects_any_water(first, second, water_bodies):
                continue
            builder.add_segment(first, second, street_class)
        return sampled

    # Unequal full-city arterial rows and columns form the underlying American
    # boulevard structure.  Corridor-level drift keeps them straight for long
    # runs without producing a perfect square lattice.
    arterial_column_count = 5
    arterial_row_count = 4
    column_positions = bounded_irregular_positions(
        random_source, inner_minimum_x, inner_maximum_x, arterial_column_count, 0.24
    )
    row_positions = bounded_irregular_positions(
        random_source, inner_minimum_y, inner_maximum_y, arterial_row_count, 0.22
    )
    row_shift_x = bounded_random_walk(random_source, arterial_row_count, inner_width * 0.012)
    column_shift_y = bounded_random_walk(random_source, arterial_column_count, inner_height * 0.012)

    arterial_grid: list[list[StreetPoint]] = []
    for row_index, y in enumerate(row_positions):
        row: list[StreetPoint] = []
        for column_index, x in enumerate(column_positions):
            row.append(StreetPoint(x + row_shift_x[row_index], y + column_shift_y[column_index]))
        arterial_grid.append(row)

    arterial_nodes: list[StreetPoint] = []
    for row in arterial_grid:
        for first, second in zip(row, row[1:]):
            # A single gentle bend per superblock is enough to escape sterile
            # axis alignment while retaining legible long boulevards.
            route = build_bent_route(first, second, random_source, inner_height * 0.018)
            arterial_nodes.extend(add_route(route, "arterial", subdivisions=2))
    for column_index in range(arterial_column_count):
        column = [arterial_grid[row_index][column_index] for row_index in range(arterial_row_count)]
        for first, second in zip(column, column[1:]):
            route = build_bent_route(first, second, random_source, inner_width * 0.016)
            arterial_nodes.extend(add_route(route, "arterial", subdivisions=2))

    # Two continuous diagonals recover the diamond-grid character and provide
    # natural cross-city shortcuts similar to older LA street alignments.
    diagonal_routes = (
        [arterial_grid[0][0], arterial_grid[1][1], arterial_grid[2][3], arterial_grid[3][4]],
        [arterial_grid[3][0], arterial_grid[2][1], arterial_grid[1][3], arterial_grid[0][4]],
    )
    for route in diagonal_routes:
        arterial_nodes.extend(add_route(route, "arterial", subdivisions=3))

    patch_candidates: list[tuple[int, int, StreetPoint, float, float]] = []
    for row_index in range(arterial_row_count - 1):
        for column_index in range(arterial_column_count - 1):
            corners = (
                arterial_grid[row_index][column_index],
                arterial_grid[row_index][column_index + 1],
                arterial_grid[row_index + 1][column_index + 1],
                arterial_grid[row_index + 1][column_index],
            )
            center = StreetPoint(
                sum(point.x for point in corners) / 4.0 + random_source.uniform(-inner_width * 0.022, inner_width * 0.022),
                sum(point.y for point in corners) / 4.0 + random_source.uniform(-inner_height * 0.022, inner_height * 0.022),
            )
            cell_width = min(distance(corners[0], corners[1]), distance(corners[3], corners[2]))
            cell_height = min(distance(corners[0], corners[3]), distance(corners[1], corners[2]))
            patch_candidates.append((row_index, column_index, center, cell_width, cell_height))

    # Prefer land cells while keeping deterministic geographic spread.
    selected_candidates = [candidate for candidate in patch_candidates if not point_inside_any_water(candidate[2], water_bodies, padding=min(width, height) * 0.025)]
    if len(selected_candidates) < neighborhood_count:
        selected_candidates.extend(candidate for candidate in patch_candidates if candidate not in selected_candidates)
    selected_candidates = selected_candidates[:neighborhood_count]

    patch_connections: list[tuple[StreetPoint, StreetPoint]] = []
    patch_centers: list[StreetPoint] = []
    for district_index, (row_index, column_index, patch_center, cell_width, cell_height) in enumerate(selected_candidates):
        patch_centers.append(patch_center)
        base_heading = math.atan2(
            arterial_grid[row_index][column_index + 1].y - arterial_grid[row_index][column_index].y,
            arterial_grid[row_index][column_index + 1].x - arterial_grid[row_index][column_index].x,
        )
        # Alternating rotations create the desired diamond neighborhoods while
        # preserving the surrounding arterial frame.
        heading = base_heading + (0.22 if district_index % 2 == 0 else -0.27) + random_source.uniform(-0.09, 0.09)
        patch_width = max(70.0, cell_width * random_source.uniform(0.54, 0.68))
        patch_height = max(70.0, cell_height * random_source.uniform(0.50, 0.66))
        patch_points = add_neighborhood_patch(
            builder,
            random_source,
            patch_center,
            heading,
            patch_width,
            patch_height,
            6,
            5,
            district_index,
            (inner_minimum_x, inner_minimum_y, inner_maximum_x, inner_maximum_y),
            omit_probability=0.0,
        )
        if not patch_points:
            continue

        surrounding_arterials = (
            arterial_grid[row_index][column_index],
            arterial_grid[row_index][column_index + 1],
            arterial_grid[row_index + 1][column_index + 1],
            arterial_grid[row_index + 1][column_index],
        )
        nearest_pairs: list[tuple[float, StreetPoint, StreetPoint]] = []
        for arterial_point in surrounding_arterials:
            patch_point = min(patch_points, key=lambda point: distance(point, arterial_point))
            nearest_pairs.append((distance(patch_point, arterial_point), arterial_point, patch_point))
        nearest_pairs.sort(key=lambda item: item[0])
        # Two separated collector connections stop neighborhood grids from
        # becoming isolated islands and create through-routes between them.
        chosen = [nearest_pairs[0]]
        for candidate in nearest_pairs[1:]:
            if distance(candidate[1], chosen[0][1]) > min(cell_width, cell_height) * 0.55:
                chosen.append(candidate)
                break
        for _, arterial_point, patch_point in chosen:
            route = build_bent_route(arterial_point, patch_point, random_source, min(width, height) * 0.018)
            add_route(route, "collector", subdivisions=2)
            patch_connections.append((arterial_point, patch_point))

    # Collector links between adjacent neighborhood patches provide the missing
    # roads between grids while still respecting inland water constraints.
    for index, first_center in enumerate(patch_centers):
        neighbors = sorted(
            (candidate for candidate in patch_centers[index + 1:]),
            key=lambda point: distance(first_center, point),
        )
        for second_center in neighbors[:2]:
            if distance(first_center, second_center) > min(width, height) * 0.48:
                continue
            if segment_intersects_any_water(first_center, second_center, water_bodies):
                continue
            first_node = min(builder.graph.nodes, key=lambda point: distance(point, first_center))
            second_node = min(builder.graph.nodes, key=lambda point: distance(point, second_center))
            add_route(
                build_bent_route(first_node, second_node, random_source, min(width, height) * 0.012),
                "collector",
                subdivisions=2,
            )

    return finalize_street_graph(builder.graph)

def generate_seeded_water_bodies(
    seed: int,
    minimum_x: float,
    minimum_y: float,
    maximum_x: float,
    maximum_y: float,
) -> list[WaterBody]:
    random_source = random.Random(seed ^ 0x5741544552424F44)
    return generate_water_bodies(
        random_source, minimum_x, minimum_y, maximum_x, maximum_y
    )


def generate_water_bodies(
    random_source: random.Random,
    minimum_x: float,
    minimum_y: float,
    maximum_x: float,
    maximum_y: float,
) -> list[WaterBody]:
    width = maximum_x - minimum_x
    height = maximum_y - minimum_y
    water_bodies: list[WaterBody] = []
    count = 2 + random_source.randrange(0, 3)
    for index in range(count):
        center_x = minimum_x + width * random_source.uniform(0.17, 0.83)
        center_y = minimum_y + height * random_source.uniform(0.16, 0.84)
        radius_x = width * random_source.uniform(0.035, 0.075)
        radius_y = height * random_source.uniform(0.035, 0.080)
        rotation = random_source.uniform(-math.pi, math.pi)
        points: list[StreetPoint] = []
        for point_index in range(18):
            angle = math.tau * point_index / 18.0
            wobble = random_source.uniform(0.88, 1.12)
            local_x = math.cos(angle) * radius_x * wobble
            local_y = math.sin(angle) * radius_y * wobble
            points.append(
                StreetPoint(
                    center_x + local_x * math.cos(rotation) - local_y * math.sin(rotation),
                    center_y + local_x * math.sin(rotation) + local_y * math.cos(rotation),
                )
            )
        water_bodies.append(WaterBody(tuple(points), "pond" if index else "lake"))
    return water_bodies


def point_inside_any_water(
    point: StreetPoint,
    water_bodies: list[WaterBody],
    padding: float = 0.0,
) -> bool:
    for water_body in water_bodies:
        if point_inside_polygon(point, water_body.points):
            return True
        if padding > 0.0:
            for start, end in zip(water_body.points, water_body.points[1:] + water_body.points[:1]):
                if point_segment_distance(point, start, end) <= padding:
                    return True
    return False


def segment_intersects_any_water(
    start: StreetPoint,
    end: StreetPoint,
    water_bodies: list[WaterBody],
) -> bool:
    if point_inside_any_water(start, water_bodies) or point_inside_any_water(end, water_bodies):
        return True
    for water_body in water_bodies:
        for first, second in zip(water_body.points, water_body.points[1:] + water_body.points[:1]):
            if segment_intersection_parameters(start, end, first, second):
                return True
    return False


def point_segment_distance(point: StreetPoint, start: StreetPoint, end: StreetPoint) -> float:
    delta_x = end.x - start.x
    delta_y = end.y - start.y
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared <= 1.0e-12:
        return distance(point, start)
    factor = ((point.x - start.x) * delta_x + (point.y - start.y) * delta_y) / length_squared
    return distance(point, interpolate(start, end, max(0.0, min(1.0, factor))))


def point_inside_polygon(point: StreetPoint, polygon: tuple[StreetPoint, ...]) -> bool:
    inside = False
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        if point_segment_distance(point, start, end) <= 1.0e-7:
            return True
        if (start.y > point.y) != (end.y > point.y):
            crossing_x = start.x + (point.y - start.y) * (end.x - start.x) / (end.y - start.y)
            if point.x < crossing_x:
                inside = not inside
    return inside


def segment_intersection_parameters(
    first_start: StreetPoint,
    first_end: StreetPoint,
    second_start: StreetPoint,
    second_end: StreetPoint,
) -> list[tuple[float, float]]:
    first_x = first_end.x - first_start.x
    first_y = first_end.y - first_start.y
    second_x = second_end.x - second_start.x
    second_y = second_end.y - second_start.y
    offset_x = second_start.x - first_start.x
    offset_y = second_start.y - first_start.y
    denominator = first_x * second_y - first_y * second_x
    tolerance = 1.0e-8
    if abs(denominator) > tolerance:
        first_factor = (offset_x * second_y - offset_y * second_x) / denominator
        second_factor = (offset_x * first_y - offset_y * first_x) / denominator
        if -tolerance <= first_factor <= 1.0 + tolerance and -tolerance <= second_factor <= 1.0 + tolerance:
            return [(max(0.0, min(1.0, first_factor)), max(0.0, min(1.0, second_factor)))]
        return []
    if abs(offset_x * first_y - offset_y * first_x) > tolerance:
        return []
    first_length_squared = first_x * first_x + first_y * first_y
    second_length_squared = second_x * second_x + second_y * second_y
    if first_length_squared <= tolerance or second_length_squared <= tolerance:
        return []
    result: list[tuple[float, float]] = []
    for point in (first_start, first_end, second_start, second_end):
        first_factor = ((point.x - first_start.x) * first_x + (point.y - first_start.y) * first_y) / first_length_squared
        second_factor = ((point.x - second_start.x) * second_x + (point.y - second_start.y) * second_y) / second_length_squared
        if -tolerance <= first_factor <= 1.0 + tolerance and -tolerance <= second_factor <= 1.0 + tolerance:
            result.append((max(0.0, min(1.0, first_factor)), max(0.0, min(1.0, second_factor))))
    return result


def finalize_street_graph(graph: StreetGraph) -> StreetGraph:
    parameters = [{0.0, 1.0} for _ in graph.segments]
    for first_index, first_segment in enumerate(graph.segments):
        first_start = graph.nodes[first_segment.start_node]
        first_end = graph.nodes[first_segment.end_node]
        for second_index in range(first_index + 1, len(graph.segments)):
            second_segment = graph.segments[second_index]
            second_start = graph.nodes[second_segment.start_node]
            second_end = graph.nodes[second_segment.end_node]
            if (max(first_start.x, first_end.x) < min(second_start.x, second_end.x)
                    or max(second_start.x, second_end.x) < min(first_start.x, first_end.x)
                    or max(first_start.y, first_end.y) < min(second_start.y, second_end.y)
                    or max(second_start.y, second_end.y) < min(first_start.y, first_end.y)):
                continue
            for first_factor, second_factor in segment_intersection_parameters(first_start, first_end, second_start, second_end):
                parameters[first_index].add(first_factor)
                parameters[second_index].add(second_factor)

    builder = StreetGraphBuilder()
    builder.graph.water_bodies = list(graph.water_bodies)
    priorities = {"arterial": 0, "collector": 1, "local": 2}
    for segment_index in sorted(range(len(graph.segments)), key=lambda index: priorities.get(graph.segments[index].street_class, 3)):
        segment = graph.segments[segment_index]
        start = graph.nodes[segment.start_node]
        end = graph.nodes[segment.end_node]
        points = [interpolate(start, end, factor) for factor in sorted(parameters[segment_index])]
        for first, second in zip(points, points[1:]):
            if distance(first, second) > 0.001:
                first_node = builder.add_node(first)
                second_node = builder.add_node(second)
                key = tuple(sorted((first_node, second_node)))
                if first_node != second_node and key not in builder.edge_lookup:
                    builder.graph.segments.append(StreetSegment(first_node, second_node, segment.street_class))
                    builder.edge_lookup.add(key)

    adjacency: dict[int, set[int]] = {index: set() for index in range(len(builder.graph.nodes))}
    for segment in builder.graph.segments:
        adjacency[segment.start_node].add(segment.end_node)
        adjacency[segment.end_node].add(segment.start_node)
    unvisited = set(adjacency)
    components: list[set[int]] = []
    while unvisited:
        component = {min(unvisited)}
        pending = list(component)
        unvisited.difference_update(component)
        while pending:
            current = pending.pop()
            neighbors = adjacency[current] & unvisited
            component.update(neighbors)
            pending.extend(neighbors)
            unvisited.difference_update(neighbors)
        components.append(component)
    if not components:
        return builder.graph
    connected = max(components, key=len)
    result = StreetGraphBuilder()
    result.graph.water_bodies = list(graph.water_bodies)
    for segment in builder.graph.segments:
        if segment.start_node in connected:
            start_node = result.add_node(builder.graph.nodes[segment.start_node])
            end_node = result.add_node(builder.graph.nodes[segment.end_node])
            result.graph.segments.append(StreetSegment(start_node, end_node, segment.street_class))
    for block in graph.blocks:
        if any(point_inside_any_water(point, graph.water_bodies) for point in block.points):
            continue
        if any(point_inside_polygon(point, block.points) for water in graph.water_bodies for point in water.points):
            continue
        if any(segment_intersects_any_water(start, end, graph.water_bodies)
               for start, end in zip(block.points, block.points[1:] + block.points[:1])):
            continue
        if all(builder.node_lookup.get((round(point.x * 1000.0), round(point.y * 1000.0))) in connected for point in block.points):
            result.graph.blocks.append(block)
    return result.graph

def bounded_irregular_positions(
    random_source: random.Random,
    minimum: float,
    maximum: float,
    count: int,
    variation: float,
) -> list[float]:
    """Partition an axis into unequal but ordered American-style block spans."""
    if count < 2:
        return [minimum]
    span_count = count - 1
    weights = [random_source.uniform(1.0 - variation, 1.0 + variation) for _ in range(span_count)]
    total = sum(weights)
    positions = [minimum]
    cursor = minimum
    for index, weight in enumerate(weights):
        cursor = maximum if index == span_count - 1 else cursor + (maximum - minimum) * weight / total
        positions.append(cursor)
    return positions


def bounded_random_walk(
    random_source: random.Random,
    count: int,
    maximum_step: float,
) -> list[float]:
    """Return a centered, gently drifting corridor alignment."""
    offsets = [0.0]
    for _ in range(1, count):
        offsets.append(offsets[-1] + random_source.uniform(-maximum_step, maximum_step))
    average = sum(offsets) / len(offsets)
    limit = maximum_step * 2.25
    return [max(-limit, min(limit, value - average)) for value in offsets]


def build_bent_route(
    start: StreetPoint,
    end: StreetPoint,
    random_source: random.Random,
    maximum_offset: float,
) -> list[StreetPoint]:
    delta_x = end.x - start.x
    delta_y = end.y - start.y
    length = math.hypot(delta_x, delta_y)
    if length <= 0.0001:
        return [start, end]
    normal_x = -delta_y / length
    normal_y = delta_x / length
    first_offset = random_source.uniform(-maximum_offset, maximum_offset)
    second_offset = random_source.uniform(-maximum_offset, maximum_offset)
    return [
        start,
        StreetPoint(start.x + delta_x * 0.34 + normal_x * first_offset, start.y + delta_y * 0.34 + normal_y * first_offset),
        StreetPoint(start.x + delta_x * 0.68 + normal_x * second_offset, start.y + delta_y * 0.68 + normal_y * second_offset),
        end,
    ]


def add_neighborhood_patch(
    builder: StreetGraphBuilder,
    random_source: random.Random,
    center: StreetPoint,
    heading: float,
    patch_width: float,
    patch_height: float,
    columns: int,
    rows: int,
    district_index: int,
    bounds: tuple[float, float, float, float],
    omit_probability: float = 0.0,
) -> list[StreetPoint]:
    minimum_x, minimum_y, maximum_x, maximum_y = bounds
    column_positions = irregular_axis_positions(random_source, patch_width, columns)
    row_positions = irregular_axis_positions(random_source, patch_height, rows)
    grid: list[list[StreetPoint | None]] = []
    patch_points: list[StreetPoint] = []

    for local_y in row_positions:
        grid_row: list[StreetPoint | None] = []
        for local_x in column_positions:
            point = rotate_local_point(center, local_x, local_y, heading)
            if minimum_x <= point.x <= maximum_x and minimum_y <= point.y <= maximum_y:
                grid_row.append(point)
                patch_points.append(point)
            else:
                grid_row.append(None)
        grid.append(grid_row)

    for row_index, grid_row in enumerate(grid):
        for column_index, point in enumerate(grid_row):
            if point is None:
                continue
            if column_index + 1 < len(grid_row) and grid_row[column_index + 1] is not None:
                if omit_probability <= 0.0 or random_source.random() > omit_probability or row_index in (0, len(grid) - 1):
                    if not segment_intersects_any_water(point, grid_row[column_index + 1], builder.graph.water_bodies):
                        builder.add_segment(point, grid_row[column_index + 1], "local")
            if row_index + 1 < len(grid) and grid[row_index + 1][column_index] is not None:
                if omit_probability <= 0.0 or random_source.random() > omit_probability or column_index in (0, len(grid_row) - 1):
                    if not segment_intersects_any_water(point, grid[row_index + 1][column_index], builder.graph.water_bodies):
                        builder.add_segment(point, grid[row_index + 1][column_index], "local")

    for row_index in range(len(grid) - 1):
        for column_index in range(len(grid[row_index]) - 1):
            corners = (
                grid[row_index][column_index],
                grid[row_index][column_index + 1],
                grid[row_index + 1][column_index + 1],
                grid[row_index + 1][column_index],
            )
            if all(corner is not None for corner in corners):
                points = tuple(corner for corner in corners if corner is not None)
                boundary_edges = [
                    tuple(sorted((builder.node_lookup.get((round(start.x * 1000.0), round(start.y * 1000.0)), -1),
                                  builder.node_lookup.get((round(end.x * 1000.0), round(end.y * 1000.0)), -1))))
                    for start, end in zip(points, points[1:] + points[:1])
                ]
                if all(edge in builder.edge_lookup for edge in boundary_edges):
                    builder.graph.blocks.append(StreetBlock(points, district_index))
    return patch_points


def irregular_axis_positions(random_source: random.Random, span: float, count: int) -> list[float]:
    positions = []
    for index in range(count):
        ratio = index / (count - 1)
        position = -span * 0.5 + span * ratio
        if index not in (0, count - 1):
            position += random_source.uniform(-span * 0.035, span * 0.035)
        positions.append(position)
    return sorted(positions)


def rotate_local_point(center: StreetPoint, local_x: float, local_y: float, heading: float) -> StreetPoint:
    cosine = math.cos(heading)
    sine = math.sin(heading)
    return StreetPoint(
        center.x + local_x * cosine - local_y * sine,
        center.y + local_x * sine + local_y * cosine,
    )


def clamp_point(
    point: StreetPoint,
    minimum_x: float,
    minimum_y: float,
    maximum_x: float,
    maximum_y: float,
) -> StreetPoint:
    return StreetPoint(
        max(minimum_x, min(maximum_x, point.x)),
        max(minimum_y, min(maximum_y, point.y)),
    )


def interpolate(start: StreetPoint, end: StreetPoint, factor: float) -> StreetPoint:
    return StreetPoint(
        start.x + (end.x - start.x) * factor,
        start.y + (end.y - start.y) * factor,
    )


def distance(first: StreetPoint, second: StreetPoint) -> float:
    return math.hypot(second.x - first.x, second.y - first.y)
