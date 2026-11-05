from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


NATIVE_QUAD_LIMIT = 301
NATIVE_TRIANGLE_LIMIT = 6
NATIVE_VERTEX_LIMIT = 1021
WATER_SOURCE_URL = "https://github.com/gta-reversed/gta-reversed/blob/master/source/game_sa/WaterLevel.cpp"
WATER_HEADER_URL = "https://github.com/gta-reversed/gta-reversed/blob/master/source/game_sa/WaterLevel.h"


@dataclass(frozen=True)
class WaterVertex:
    x: float
    y: float
    z: float
    flow_x: float = 0.0
    flow_y: float = 0.0
    big_waves: float = 0.0
    small_waves: float = 0.0


@dataclass(frozen=True)
class WaterSurface:
    vertices: tuple[WaterVertex, ...]
    flags: int = 1
    kind: str = "retail"
    original_line: str | None = None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            min(vertex.x for vertex in self.vertices),
            min(vertex.y for vertex in self.vertices),
            max(vertex.x for vertex in self.vertices),
            max(vertex.y for vertex in self.vertices),
        )


@dataclass(frozen=True)
class WaterDocument:
    surfaces: tuple[WaterSurface, ...]
    comments: tuple[str, ...] = ()


@dataclass(frozen=True)
class WaterOverride:
    data: bytes | None
    surfaces: tuple[WaterSurface, ...]
    diagnostics: dict[str, object]


class WaterFormatError(ValueError):
    pass


def parse_water_dat(data: bytes | str) -> WaterDocument:
    text = data.decode("cp1252") if isinstance(data, bytes) else data
    surfaces: list[WaterSurface] = []
    comments: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(keepends=True), 1):
        line = raw_line.strip()
        if not line or line.startswith(("#", ";", "//")):
            comments.append(raw_line)
            continue
        line = line.split("#", 1)[0].split(";", 1)[0].replace(",", " ").strip()
        fields = line.split()
        try:
            first = float(fields[0])
        except (ValueError, IndexError):
            comments.append(raw_line)
            continue
        if not math.isfinite(first):
            raise WaterFormatError(f"Water line {line_number} contains a non-finite coordinate.")
        if len(fields) not in {21, 22, 28, 29}:
            raise WaterFormatError(f"Water line {line_number} has {len(fields)} fields; expected three or four seven-field vertices and optional flags.")
        vertex_count = 3 if len(fields) in {21, 22} else 4
        try:
            values = [float(value) for value in fields[:vertex_count * 7]]
            flags = int(fields[-1], 10) if len(fields) == vertex_count * 7 + 1 else 0
        except ValueError as error:
            raise WaterFormatError(f"Water line {line_number} contains an invalid number.") from error
        if not all(math.isfinite(value) for value in values):
            raise WaterFormatError(f"Water line {line_number} contains a non-finite value.")
        if not 0 <= flags <= 0xFFFFFFFF:
            raise WaterFormatError(f"Water line {line_number} contains invalid flags.")
        vertices = tuple(
            WaterVertex(float(int(values[index * 7])), float(int(values[index * 7 + 1])), *values[index * 7 + 2:(index + 1) * 7])
            for index in range(vertex_count)
        )
        if any(not -2.0 <= value < 2.0 for vertex in vertices for value in (vertex.flow_x, vertex.flow_y)):
            raise WaterFormatError(f"Water line {line_number} has flow outside the native signed 1/64 range.")
        surface = WaterSurface(vertices, flags, original_line=raw_line)
        if polygon_area(surface_polygon(surface)) <= 0.01:
            raise WaterFormatError(f"Water line {line_number} has a degenerate polygon.")
        surfaces.append(surface)
    return WaterDocument(tuple(surfaces), tuple(comments))


def water_capacity(surfaces: Sequence[WaterSurface]) -> dict[str, int]:
    coordinates = {(int(vertex.x), int(vertex.y), vertex.z) for surface in surfaces for vertex in surface.vertices}
    return {
        "quads": sum(len(surface.vertices) == 4 for surface in surfaces),
        "triangles": sum(len(surface.vertices) == 3 for surface in surfaces),
        "vertices": len(coordinates),
    }


def validate_water_surface(surface: WaterSurface) -> None:
    if len(surface.vertices) not in {3, 4}:
        raise WaterFormatError("A native water surface requires three or four vertices.")
    if not 0 <= surface.flags <= 0xFFFFFFFF:
        raise WaterFormatError("A native water surface has invalid flags.")
    if any(not math.isfinite(value) for vertex in surface.vertices for value in (vertex.x, vertex.y, vertex.z, vertex.flow_x, vertex.flow_y, vertex.big_waves, vertex.small_waves)):
        raise WaterFormatError("A native water surface has a non-finite vertex field.")
    if any(not -2.0 <= value < 2.0 for vertex in surface.vertices for value in (vertex.flow_x, vertex.flow_y)):
        raise WaterFormatError("A native water surface flow exceeds the signed 1/64 range.")
    points = tuple((int(vertex.x), int(vertex.y)) for vertex in surface.vertices)
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    ordered = tuple(sorted(points, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x)))
    if len(set(points)) != len(points) or polygon_area(ordered) <= 0.0:
        raise WaterFormatError("A water surface collapses after the native integer-coordinate conversion.")
    if surface.original_line is None:
        if len(surface.vertices) == 4 and set(points) != set(rectangle_polygon(surface.bounds)):
            raise WaterFormatError("Generated native water quads must be axis-aligned rectangles with integer corners.")
        if len(surface.vertices) == 3 and (len({point[0] for point in points}) != 2 or len({point[1] for point in points}) != 2):
            raise WaterFormatError("Generated native water triangles must have the axis-sharing right-triangle shape used by the game.")


def validate_water_capacity(surfaces: Sequence[WaterSurface]) -> dict[str, int]:
    for surface in surfaces:
        validate_water_surface(surface)
    counts = water_capacity(surfaces)
    for key, limit in (("quads", NATIVE_QUAD_LIMIT), ("triangles", NATIVE_TRIANGLE_LIMIT), ("vertices", NATIVE_VERTEX_LIMIT)):
        if counts[key] > limit:
            raise WaterFormatError(f"Native water {key} capacity exceeded: {counts[key]}/{limit}.")
    return counts


def write_water_dat(surfaces: Sequence[WaterSurface], comments: Sequence[str] = ()) -> bytes:
    validate_water_capacity(surfaces)
    lines = [line if line.endswith(("\n", "\r")) else line + "\n" for line in comments]
    for surface in surfaces:
        if surface.original_line is not None:
            line = surface.original_line
        else:
            fields: list[str] = []
            for vertex in surface.vertices:
                values = (vertex.x, vertex.y, vertex.z, vertex.flow_x, vertex.flow_y, vertex.big_waves, vertex.small_waves)
                if not all(math.isfinite(value) for value in values):
                    raise WaterFormatError("Generated water has a non-finite vertex field.")
                fields.extend(format_number(value) for value in values)
            fields.append(str(surface.flags))
            line = " ".join(fields)
        lines.append(line if line.endswith(("\n", "\r")) else line + "\n")
    return "".join(lines).encode("cp1252")


def format_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") if value != 0.0 else "0"


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    return abs(sum(first[0] * second[1] - second[0] * first[1] for first, second in zip(points, (*points[1:], points[0])))) * 0.5


def is_exterior_surface(surface: WaterSurface) -> bool:
    if len(surface.vertices) == 4:
        first = min(surface.vertices, key=lambda vertex: (int(vertex.y), int(vertex.x)))
    else:
        first, second, third = surface.vertices
        if int(first.y) == int(second.y):
            first = min((first, second), key=lambda vertex: int(vertex.x))
        elif int(first.y) == int(third.y):
            first = min((first, third), key=lambda vertex: int(vertex.x))
        else:
            first = min((second, third), key=lambda vertex: int(vertex.x))
    return first.z <= 950.0


def surface_polygon(surface: WaterSurface) -> tuple[tuple[float, float], ...]:
    points = tuple((vertex.x, vertex.y) for vertex in surface.vertices)
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    return tuple(sorted(points, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x)))


def point_on_segment(point: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> bool:
    cross = (point[0] - first[0]) * (second[1] - first[1]) - (point[1] - first[1]) * (second[0] - first[0])
    return abs(cross) <= 0.000001 and min(first[0], second[0]) - 0.000001 <= point[0] <= max(first[0], second[0]) + 0.000001 and min(first[1], second[1]) - 0.000001 <= point[1] <= max(first[1], second[1]) + 0.000001


def point_in_polygon(x: float, y: float, points: Sequence[tuple[float, float]]) -> bool:
    inside = False
    for first, second in zip(points, (*points[1:], points[0])):
        if point_on_segment((x, y), first, second):
            return True
        if (first[1] > y) != (second[1] > y):
            intersection = first[0] + (y - first[1]) * (second[0] - first[0]) / (second[1] - first[1])
            if x < intersection:
                inside = not inside
    return inside


def proper_segments_cross(first: tuple[float, float], second: tuple[float, float], third: tuple[float, float], fourth: tuple[float, float]) -> bool:
    first_x, first_y = second[0] - first[0], second[1] - first[1]
    second_x, second_y = fourth[0] - third[0], fourth[1] - third[1]
    determinant = first_x * second_y - first_y * second_x
    if abs(determinant) < 0.000001:
        return False
    delta_x, delta_y = third[0] - first[0], third[1] - first[1]
    first_factor = (delta_x * second_y - delta_y * second_x) / determinant
    second_factor = (delta_x * first_y - delta_y * first_x) / determinant
    return 0.000001 < first_factor < 0.999999 and 0.000001 < second_factor < 0.999999


def rectangle_polygon(bounds: tuple[float, float, float, float]) -> tuple[tuple[float, float], ...]:
    minimum_x, minimum_y, maximum_x, maximum_y = bounds
    return ((minimum_x, minimum_y), (maximum_x, minimum_y), (maximum_x, maximum_y), (minimum_x, maximum_y))


def rectangles_overlap(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
    return first[0] < second[2] and first[2] > second[0] and first[1] < second[3] and first[3] > second[1]


def rectangle_touches_polygon(bounds: tuple[float, float, float, float], points: Sequence[tuple[float, float]]) -> bool:
    corners = rectangle_polygon(bounds)
    if any(point_in_polygon(x, y, points) for x, y in corners):
        return True
    if any(bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3] for x, y in points):
        return True
    return any(
        proper_segments_cross(first, second, third, fourth)
        for first, second in zip(corners, (*corners[1:], corners[0]))
        for third, fourth in zip(points, (*points[1:], points[0]))
    )


def rectangle_inside_polygon(bounds: tuple[float, float, float, float], points: Sequence[tuple[float, float]]) -> bool:
    corners = rectangle_polygon(bounds)
    if not all(point_in_polygon(x, y, points) for x, y in corners):
        return False
    return not any(
        proper_segments_cross(first, second, third, fourth)
        for first, second in zip(corners, (*corners[1:], corners[0]))
        for third, fourth in zip(points, (*points[1:], points[0]))
    )


def subtract_rectangle(bounds: tuple[float, float, float, float], removed: tuple[float, float, float, float]) -> list[tuple[float, float, float, float]]:
    if not rectangles_overlap(bounds, removed):
        return [bounds]
    minimum_x, minimum_y, maximum_x, maximum_y = bounds
    left, bottom = max(minimum_x, removed[0]), max(minimum_y, removed[1])
    right, top = min(maximum_x, removed[2]), min(maximum_y, removed[3])
    pieces = ((minimum_x, minimum_y, left, maximum_y), (right, minimum_y, maximum_x, maximum_y), (left, minimum_y, right, bottom), (left, top, right, maximum_y))
    return [piece for piece in pieces if piece[2] - piece[0] > 0.001 and piece[3] - piece[1] > 0.001]


def interpolate_vertex(surface: WaterSurface, x: float, y: float) -> WaterVertex:
    minimum_x, minimum_y, maximum_x, maximum_y = surface.bounds
    width, depth = maximum_x - minimum_x, maximum_y - minimum_y
    if width <= 0.0 or depth <= 0.0 or len(surface.vertices) != 4:
        raise WaterFormatError("Clipping requires a nondegenerate axis-aligned water quad.")
    expected_corners = set(rectangle_polygon(surface.bounds))
    if {(vertex.x, vertex.y) for vertex in surface.vertices} != expected_corners:
        raise WaterFormatError("Retail water quad crossing the replacement boundary is not axis aligned.")
    by_corner = {(int(vertex.x == maximum_x), int(vertex.y == maximum_y)): vertex for vertex in surface.vertices}
    if len(by_corner) != 4:
        raise WaterFormatError("Retail water quad crossing the replacement boundary is not axis aligned.")
    u, v = (x - minimum_x) / width, (y - minimum_y) / depth
    weights = (((0, 0), (1.0 - u) * (1.0 - v)), ((1, 0), u * (1.0 - v)), ((1, 1), u * v), ((0, 1), (1.0 - u) * v))
    values = [sum(getattr(by_corner[corner], field) * weight for corner, weight in weights) for field in ("z", "flow_x", "flow_y", "big_waves", "small_waves")]
    return WaterVertex(x, y, *values)


def clipped_surface(surface: WaterSurface, bounds: tuple[float, float, float, float]) -> WaterSurface:
    vertices = tuple(interpolate_vertex(surface, x, y) for x, y in rectangle_polygon(bounds))
    return WaterSurface(vertices, surface.flags, surface.kind)


def intersect_rectangles(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    result = (max(first[0], second[0]), max(first[1], second[1]), min(first[2], second[2]), min(first[3], second[3]))
    return result if result[2] > result[0] and result[3] > result[1] else None


def protected_water_rectangles(
    surface_bounds: tuple[float, float, float, float],
    replacement_bounds: tuple[float, float, float, float],
    preserved_bounds: Sequence[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    inside = intersect_rectangles(surface_bounds, replacement_bounds)
    if inside is None:
        return []
    retained: list[tuple[float, float, float, float]] = []
    for protected in preserved_bounds:
        # Native water X/Y are integers. Outward rounding retains the entire
        # protected waterfront instead of shaving off fractional border strips.
        native_bounds = (math.floor(protected[0]), math.floor(protected[1]), math.ceil(protected[2]), math.ceil(protected[3]))
        candidate = intersect_rectangles(inside, native_bounds)
        if candidate is None:
            continue
        fragments = [candidate]
        for previous in retained:
            fragments = [piece for fragment in fragments for piece in subtract_rectangle(fragment, previous)]
        retained.extend(fragments)
    return retained


def preserve_retail_water(
    document: WaterDocument,
    bounds: tuple[float, float, float, float],
    preserved_bounds: Sequence[tuple[float, float, float, float]] = (),
) -> list[WaterSurface]:
    retained: list[WaterSurface] = []
    for surface in document.surfaces:
        if not is_exterior_surface(surface) or not rectangles_overlap(surface.bounds, bounds):
            retained.append(surface)
            continue
        protected_pieces = protected_water_rectangles(surface.bounds, bounds, preserved_bounds)
        if len(surface.vertices) == 3:
            fully_inside = all(bounds[0] <= vertex.x <= bounds[2] and bounds[1] <= vertex.y <= bounds[3] for vertex in surface.vertices)
            if not fully_inside:
                raise WaterFormatError("A retail water triangle crosses the replacement boundary; preserving it requires native triangle capacity and exact right-triangle clipping.")
            if protected_pieces:
                if any(all(piece[0] <= vertex.x <= piece[2] and piece[1] <= vertex.y <= piece[3] for vertex in surface.vertices) for piece in protected_pieces):
                    retained.append(surface)
                else:
                    raise WaterFormatError("A protected waterfront partly intersects a retail water triangle; exact native right-triangle clipping is required before replacement.")
            continue
        for piece in subtract_rectangle(surface.bounds, bounds):
            retained.append(clipped_surface(surface, piece))
        for piece in protected_pieces:
            retained.append(surface if piece == surface.bounds else clipped_surface(surface, piece))
    return retained


def normalized_polygon(points: Iterable[object]) -> tuple[tuple[float, float], ...]:
    result = tuple((float(point.x), float(point.y)) if hasattr(point, "x") else (float(point[0]), float(point[1])) for point in points)
    if len(result) < 3 or not all(math.isfinite(value) for point in result for value in point):
        raise WaterFormatError("A generated water/land outline requires at least three finite points.")
    return result


def merged_grid_rectangles(rows: list[list[bool]], x_edges: list[int], y_edges: list[int]) -> list[tuple[float, float, float, float]]:
    active: dict[tuple[int, int], tuple[int, int]] = {}
    rectangles: list[tuple[float, float, float, float]] = []
    for row_index, row in enumerate(rows):
        spans: list[tuple[int, int]] = []
        column = 0
        while column < len(row):
            if not row[column]:
                column += 1
                continue
            first = column
            while column < len(row) and row[column]:
                column += 1
            spans.append((first, column))
        next_active: dict[tuple[int, int], tuple[int, int]] = {}
        for span in spans:
            first_row, _ = active.pop(span, (row_index, row_index))
            next_active[span] = (first_row, row_index + 1)
        for (first_column, last_column), (first_row, last_row) in active.items():
            rectangles.append((x_edges[first_column], y_edges[first_row], x_edges[last_column], y_edges[last_row]))
        active = next_active
    for (first_column, last_column), (first_row, last_row) in active.items():
        rectangles.append((x_edges[first_column], y_edges[first_row], x_edges[last_column], y_edges[last_row]))
    return rectangles


def grid_edges(minimum: float, maximum: float, step: int) -> list[int]:
    first, last = math.ceil(minimum), math.floor(maximum)
    if last <= first:
        raise WaterFormatError("Water replacement bounds contain no integer-coordinate area.")
    return [*range(first, last, step), last]


def generated_water_rectangles(
    land_outline: Sequence[tuple[float, float]],
    inland_polygons: Sequence[Sequence[tuple[float, float]]],
    bounds: tuple[float, float, float, float],
    exclusion_bounds: Sequence[tuple[float, float, float, float]],
    step: int,
) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float, float]]]:
    x_edges = grid_edges(bounds[0], bounds[2], step)
    y_edges = grid_edges(bounds[1], bounds[3], step)
    ocean_rows: list[list[bool]] = []
    lake_rows: list[list[bool]] = []
    for first_y, second_y in zip(y_edges, y_edges[1:]):
        ocean_row, lake_row = [], []
        for first_x, second_x in zip(x_edges, x_edges[1:]):
            cell = (first_x, first_y, second_x, second_y)
            excluded = any(rectangles_overlap(cell, excluded) for excluded in exclusion_bounds)
            # Shore quads extend under the bank; the terrain crosses sea level
            # at the land boundary, preventing an uncovered seabed fringe.
            ocean_row.append(not excluded and not rectangle_inside_polygon(cell, land_outline))
            lake_row.append(not excluded and any(rectangle_inside_polygon(cell, polygon) for polygon in inland_polygons))
        ocean_rows.append(ocean_row)
        lake_rows.append(lake_row)
    return merged_grid_rectangles(ocean_rows, x_edges, y_edges), merged_grid_rectangles(lake_rows, x_edges, y_edges)


def lakes_outside_protection(
    lakes: Sequence[Sequence[tuple[float, float]]],
    bounds: tuple[float, float, float, float],
    exclusion_bounds: Sequence[tuple[float, float, float, float]],
) -> tuple[list[Sequence[tuple[float, float]]], int]:
    active: list[Sequence[tuple[float, float]]] = []
    protected_count = 0
    for lake in lakes:
        lake_bounds = (
            max(bounds[0], min(point[0] for point in lake)),
            max(bounds[1], min(point[1] for point in lake)),
            min(bounds[2], max(point[0] for point in lake)),
            min(bounds[3], max(point[1] for point in lake)),
        )
        if lake_bounds[2] - lake_bounds[0] < 2.0 or lake_bounds[3] - lake_bounds[1] < 2.0:
            continue
        step = max(1, min(24, int(min(lake_bounds[2] - lake_bounds[0], lake_bounds[3] - lake_bounds[1]) / 6.0)))
        x_edges = grid_edges(lake_bounds[0], lake_bounds[2], step)
        y_edges = grid_edges(lake_bounds[1], lake_bounds[3], step)
        has_water = False
        has_unprotected_water = False
        for first_y, second_y in zip(y_edges, y_edges[1:]):
            for first_x, second_x in zip(x_edges, x_edges[1:]):
                cell = (first_x, first_y, second_x, second_y)
                if not rectangle_inside_polygon(cell, lake):
                    continue
                has_water = True
                if not any(rectangles_overlap(cell, excluded) for excluded in exclusion_bounds):
                    has_unprotected_water = True
                    break
            if has_unprotected_water:
                break
        if has_unprotected_water:
            active.append(lake)
        elif has_water:
            protected_count += 1
    return active, protected_count


def flat_water_surface(bounds: tuple[float, float, float, float], z: float, kind: str, big_waves: float, small_waves: float) -> WaterSurface:
    return WaterSurface(tuple(WaterVertex(x, y, z, 0.0, 0.0, big_waves, small_waves) for x, y in rectangle_polygon(bounds)), 1, kind)


def every_lake_has_water(
    rectangles: Sequence[tuple[float, float, float, float]],
    lakes: Sequence[Sequence[tuple[float, float]]],
) -> bool:
    for lake in lakes:
        lake_bounds = (
            min(point[0] for point in lake), min(point[1] for point in lake),
            max(point[0] for point in lake), max(point[1] for point in lake),
        )
        if not any(
            rectangles_overlap(rectangle, lake_bounds)
            and point_in_polygon((rectangle[0] + rectangle[2]) * 0.5, (rectangle[1] + rectangle[3]) * 0.5, lake)
            for rectangle in rectangles
        ):
            return False
    return True


def find_water_file(game_directory: Path) -> Path | None:
    data_directory = game_directory / "data"
    if not data_directory.is_dir():
        return None
    return next((path for path in data_directory.iterdir() if path.is_file() and path.name.lower() == "water.dat"), None)


def build_water_override(
    game_directory: Path,
    land_outline: Iterable[object],
    inland_water_bodies: Iterable[object],
    bounds: tuple[float, float, float, float],
    exclusion_bounds: Sequence[tuple[float, float, float, float]] = (),
    sea_level: float = 0.0,
    lake_level: float = 6.25,
    preserved_water_bounds: Sequence[tuple[float, float, float, float]] | None = None,
) -> WaterOverride:
    source = find_water_file(game_directory)
    if source is None:
        return WaterOverride(None, (), {"status": "missing_water_dat", "native_override_written": False, "reason": "No installed data/water.dat was found; a replacement cannot preserve water in the other cities."})
    if not all(math.isfinite(value) for value in (*bounds, sea_level, lake_level)) or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise WaterFormatError("Water replacement bounds and heights must be finite and nondegenerate.")
    document = parse_water_dat(source.read_bytes())
    original_counts = validate_water_capacity(document.surfaces)
    land = normalized_polygon(land_outline)
    all_lakes = [normalized_polygon(getattr(body, "points", body)) for body in inland_water_bodies]
    lakes, protected_lakes = lakes_outside_protection(all_lakes, bounds, exclusion_bounds)
    source_preservation = exclusion_bounds if preserved_water_bounds is None else preserved_water_bounds
    retained = preserve_retail_water(document, bounds, source_preservation)
    if water_capacity(retained)["quads"] >= NATIVE_QUAD_LIMIT:
        raise WaterFormatError("Retained retail water leaves no free native quad slots for the generated coast and lakes.")
    last_error = "No valid water tessellation was found."
    resolutions = (24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768)
    ocean_options: list[tuple[int, list[tuple[float, float, float, float]]]] = []
    lake_options: list[tuple[int, list[tuple[float, float, float, float]]]] = []
    for step in resolutions:
        ocean_rectangles, lake_rectangles = generated_water_rectangles(land, lakes, bounds, exclusion_bounds, step)
        ocean_options.append((step, ocean_rectangles))
        if every_lake_has_water(lake_rectangles, lakes):
            lake_options.append((step, lake_rectangles))
    if not lake_options:
        raise WaterFormatError("No native water resolution preserves every unprotected inland water body.")
    ocean_required = bool(ocean_options[0][1])
    combinations = sorted(
        ((ocean_step, lake_step, ocean_rectangles, lake_rectangles)
         for ocean_step, ocean_rectangles in ocean_options
         for lake_step, lake_rectangles in lake_options
         if ocean_rectangles or not ocean_required),
        key=lambda option: (
            math.log2(option[0] / 24.0) + 1.5 * math.log2(option[1] / 24.0),
            option[0], option[1],
        ),
    )
    for ocean_step, lake_step, ocean_rectangles, lake_rectangles in combinations:
        generated = [flat_water_surface(rectangle, sea_level, "ocean", 1.0, 1.0) for rectangle in ocean_rectangles]
        generated.extend(flat_water_surface(rectangle, lake_level, "lake", 0.12, 0.25) for rectangle in lake_rectangles)
        surfaces = (*retained, *generated)
        try:
            counts = validate_water_capacity(surfaces)
        except WaterFormatError as error:
            last_error = str(error)
            continue
        data = write_water_dat(surfaces, document.comments)
        parsed = parse_water_dat(data)
        validate_water_capacity(parsed.surfaces)
        diagnostics: dict[str, object] = {
            "status": "native_water_generated", "native_override_written": True,
            "source": str(source), "source_counts": original_counts,
            "retained_surfaces": len(retained), "generated_ocean_quads": len(ocean_rectangles),
            "retail_water_preservation_regions": len(source_preservation),
            "generated_lake_quads": len(lake_rectangles), "grid_step": ocean_step,
            "sea_grid_step": ocean_step, "lake_grid_step": lake_step,
            "resolution_mode": "independent_sea_and_inland_budget",
            "inland_bodies_requested": len(all_lakes), "inland_bodies_excluded_for_gameplay": protected_lakes,
            "inland_bodies_generated": len(lakes),
            "sea_level": sea_level, "lake_level": lake_level,
            "counts": counts,
            "limits": {"quads": NATIVE_QUAD_LIMIT, "triangles": NATIVE_TRIANGLE_LIMIT, "vertices": NATIVE_VERTEX_LIMIT},
            "sources": [WATER_SOURCE_URL, WATER_HEADER_URL],
        }
        return WaterOverride(data, tuple(surfaces), diagnostics)
    raise WaterFormatError("Generated water cannot fit the stock native capacities while preserving every unprotected inland water body: " + last_error)


def water_surface_polygons(
    surfaces: Sequence[WaterSurface],
    generated_only: bool = False,
    include_interiors: bool = False,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    return tuple(
        surface_polygon(surface) for surface in surfaces
        if (not generated_only or surface.kind != "retail") and (include_interiors or is_exterior_surface(surface))
    )


def water_height_at(
    x: float,
    y: float,
    surfaces: Sequence[WaterSurface],
    include_interiors: bool = False,
) -> float | None:
    heights: list[float] = []
    for surface in surfaces:
        if not include_interiors and not is_exterior_surface(surface):
            continue
        minimum_x, minimum_y, maximum_x, maximum_y = surface.bounds
        if not (minimum_x <= x <= maximum_x and minimum_y <= y <= maximum_y):
            continue
        if not point_in_polygon(x, y, surface_polygon(surface)):
            continue
        if len(surface.vertices) == 4:
            heights.append(interpolate_vertex(surface, x, y).z)
        else:
            first, second, third = surface.vertices
            denominator = (second.y - third.y) * (first.x - third.x) + (third.x - second.x) * (first.y - third.y)
            if abs(denominator) < 0.00001:
                continue
            first_factor = ((second.y - third.y) * (x - third.x) + (third.x - second.x) * (y - third.y)) / denominator
            second_factor = ((third.y - first.y) * (x - third.x) + (first.x - third.x) * (y - third.y)) / denominator
            heights.append(first.z * first_factor + second.z * second_factor + third.z * (1.0 - first_factor - second_factor))
    return max(heights) if heights else None
