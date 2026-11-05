from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .custom_world import (
    MATERIAL_BUILDING,
    MATERIAL_GRASS,
    MATERIAL_ROOF,
    MATERIAL_SIDEWALK,
    MATERIAL_TRIM,
    MATERIAL_WINDOW,
    MeshBuilder,
    shade_color,
)


Color = tuple[int, int, int, int]


@dataclass(frozen=True)
class NeighborhoodProfile:
    name: str
    condition: str
    wealth: float
    density: float
    target_lot_size: float
    palette: tuple[Color, ...]


@dataclass(frozen=True)
class BuildingResult:
    style: str
    condition: str
    dwelling_units: int


@dataclass(frozen=True)
class TownPlan:
    town_id: str
    center_x: float
    center_y: float
    main_street_angle: float
    center_radius: float
    suburban_radius: float
    rural_radius: float


NEIGHBORHOOD_PALETTES: dict[str, tuple[Color, ...]] = {
    "slum": ((153, 137, 109, 255), (137, 146, 119, 255), (169, 131, 102, 255), (147, 144, 128, 255)),
    "ghetto": ((191, 156, 113, 255), (155, 174, 137, 255), (185, 132, 110, 255), (201, 183, 149, 255)),
    "residential": ((211, 192, 151, 255), (173, 185, 157, 255), (198, 177, 158, 255), (188, 193, 186, 255)),
    "middle_class": ((230, 217, 185, 255), (207, 188, 154, 255), (184, 198, 179, 255), (225, 211, 199, 255)),
    "high_class": ((237, 225, 201, 255), (221, 207, 181, 255), (230, 221, 209, 255), (208, 216, 198, 255)),
    "commercial": ((195, 167, 124, 255), (186, 145, 119, 255), (193, 192, 174, 255), (170, 186, 172, 255)),
    "industrial": ((155, 147, 128, 255), (137, 151, 151, 255), (171, 151, 123, 255), (151, 137, 129, 255)),
}


def neighborhood_hash(x: int, y: int, seed: int) -> float:
    value = (x * 0x1F123BB5) ^ (y * 0x5F356495) ^ seed
    value = (value ^ (value >> 16)) * 0x45D9F3B
    value = (value ^ (value >> 16)) * 0x45D9F3B
    value ^= value >> 16
    return (value & 0xFFFFFFFF) / 0xFFFFFFFF


def neighborhood_field(x: float, y: float, seed: int, scale: float) -> float:
    grid_x, grid_y = x / scale, y / scale
    first_x, first_y = math.floor(grid_x), math.floor(grid_y)
    factor_x, factor_y = grid_x - first_x, grid_y - first_y
    factor_x = factor_x * factor_x * (3.0 - 2.0 * factor_x)
    factor_y = factor_y * factor_y * (3.0 - 2.0 * factor_y)
    lower = (
        neighborhood_hash(first_x, first_y, seed) * (1.0 - factor_x)
        + neighborhood_hash(first_x + 1, first_y, seed) * factor_x
    )
    upper = (
        neighborhood_hash(first_x, first_y + 1, seed) * (1.0 - factor_x)
        + neighborhood_hash(first_x + 1, first_y + 1, seed) * factor_x
    )
    return lower * (1.0 - factor_y) + upper * factor_y


def generate_town_plans(seed: int) -> tuple[TownPlan, ...]:
    rng = random.Random(seed ^ 0x544F574E504C414E)
    plans: list[TownPlan] = []
    for town_id, bounds, angle in (
        ("northwest", (530.0, -700.0, 950.0, -440.0), math.radians(8.0)),
        ("northeast", (2340.0, -740.0, 2700.0, -450.0), math.radians(-18.0)),
    ):
        minimum_x, minimum_y, maximum_x, maximum_y = bounds
        center_radius = rng.uniform(120.0, 165.0)
        suburban_radius = center_radius + rng.uniform(175.0, 220.0)
        plans.append(TownPlan(
            town_id=town_id,
            center_x=round(rng.uniform(minimum_x, maximum_x), 3),
            center_y=round(rng.uniform(minimum_y, maximum_y), 3),
            main_street_angle=angle + rng.uniform(-0.16, 0.16),
            center_radius=round(center_radius, 3),
            suburban_radius=round(suburban_radius, 3),
            rural_radius=round(suburban_radius + rng.uniform(145.0, 195.0), 3),
        ))
    return tuple(plans)


def town_plan_at(world_x: float, world_y: float, seed: int) -> TownPlan | None:
    if not all(math.isfinite(value) for value in (world_x, world_y)):
        raise ValueError("Settlement coordinates must be finite.")
    plans = generate_town_plans(seed)
    nearest = min(plans, key=lambda plan: math.hypot(world_x - plan.center_x, world_y - plan.center_y) / plan.rural_radius)
    distance = math.hypot(world_x - nearest.center_x, world_y - nearest.center_y)
    return nearest if distance <= nearest.rural_radius else None


def choose_settlement(world_x: float, world_y: float, seed: int) -> str:
    plan = town_plan_at(world_x, world_y, seed)
    if plan is not None:
        distance = math.hypot(world_x - plan.center_x, world_y - plan.center_y)
        if distance <= plan.center_radius:
            return "town"
        if distance <= plan.suburban_radius:
            return "suburban"
        return "rural"
    northern_fringe = world_y > -390.0
    eastern_fringe = world_x > 2760.0 and world_y > -1500.0
    if northern_fringe or eastern_fringe:
        return "rural"
    central_city = 1050.0 < world_x < 2220.0 and -2200.0 < world_y < -900.0
    return "urban" if central_city or world_y < -1550.0 else "suburban"


def town_frontage_strength(world_x: float, world_y: float, plan: TownPlan) -> float:
    if not all(math.isfinite(value) for value in (world_x, world_y)):
        raise ValueError("Town frontage coordinates must be finite.")
    delta_x, delta_y = world_x - plan.center_x, world_y - plan.center_y
    cosine, sine = math.cos(plan.main_street_angle), math.sin(plan.main_street_angle)
    along = delta_x * cosine + delta_y * sine
    across = -delta_x * sine + delta_y * cosine
    along_factor = max(0.0, 1.0 - (abs(along) / (plan.center_radius * 1.15)) ** 2)
    across_factor = max(0.0, 1.0 - (abs(across) / 62.0) ** 2)
    return along_factor * across_factor


def town_main_street_endpoints(plan: TownPlan) -> tuple[tuple[float, float], tuple[float, float]]:
    span = plan.center_radius * 1.25
    delta_x, delta_y = math.cos(plan.main_street_angle) * span, math.sin(plan.main_street_angle) * span
    return ((plan.center_x - delta_x, plan.center_y - delta_y),
            (plan.center_x + delta_x, plan.center_y + delta_y))


def choose_settlement_district(world_x: float, world_y: float, seed: int, default_district: str) -> str:
    if default_district in {"airport", "docks", "industrial", "downtown", "park"}:
        return default_district
    plan = town_plan_at(world_x, world_y, seed)
    if plan is None:
        return default_district
    settlement = choose_settlement(world_x, world_y, seed)
    return settlement if settlement in {"town", "suburban", "rural"} else default_district


def choose_neighborhood(world_x: float, world_y: float, seed: int, district: str) -> NeighborhoodProfile:
    if not all(math.isfinite(value) for value in (world_x, world_y)):
        raise ValueError("Neighborhood coordinates must be finite.")
    broad = neighborhood_field(world_x, world_y, seed ^ 0x5745414C, 430.0)
    detail = neighborhood_field(world_x, world_y, seed ^ 0x50415243, 165.0)
    wealth = broad * 0.78 + detail * 0.22
    if district in {"grove", "eastside"}:
        wealth = max(0.0, wealth - 0.18)
    elif district == "hills":
        wealth = min(1.0, wealth + 0.20)
    elif district in {"beach", "coastal"}:
        wealth = min(1.0, wealth + 0.06)
    elif district == "suburban":
        wealth = min(1.0, wealth + 0.09)
    elif district == "rural":
        wealth = max(0.0, wealth - 0.07)
    town_plan = town_plan_at(world_x, world_y, seed) if district == "town" else None
    if district in {"industrial", "docks"}:
        name = "industrial"
    elif district in {"downtown", "commercial", "civic"}:
        name = "commercial"
    elif district == "town" and (
        town_plan is not None
        and town_frontage_strength(world_x, world_y, town_plan) > 0.28
        and neighborhood_hash(math.floor(world_x / 42.0), math.floor(world_y / 42.0), seed ^ 0x53484F50) < 0.82
    ):
        name = "commercial"
    elif wealth < 0.24:
        name = "slum"
    elif wealth < 0.40:
        name = "ghetto"
    elif wealth < 0.57:
        name = "residential"
    elif wealth < 0.74:
        name = "middle_class"
    else:
        name = "high_class"

    maintenance = neighborhood_field(world_x, world_y, seed ^ 0x53544154, 145.0)
    maintenance = maintenance * 0.62 + wealth * 0.38
    if maintenance < 0.27:
        condition = "neglected"
    elif maintenance < 0.47:
        condition = "weathered"
    elif maintenance < 0.78:
        condition = "maintained"
    else:
        condition = "restored"
    density_by_name = {
        "slum": 0.94, "ghetto": 0.88, "residential": 0.74,
        "middle_class": 0.62, "high_class": 0.38,
        "commercial": 0.89, "industrial": 0.70,
    }
    lot_sizes = {
        "slum": 19.0, "ghetto": 23.0, "residential": 27.0,
        "middle_class": 34.0, "high_class": 48.0,
        "commercial": 30.0, "industrial": 43.0,
    }
    density = density_by_name[name]
    lot_size = lot_sizes[name]
    if district == "town":
        density = min(0.94, density * 1.08)
        lot_size = 23.0 if name == "commercial" else min(lot_size, 29.0)
    elif district == "suburban":
        density *= 0.86
        lot_size *= 1.10
    elif district == "rural":
        density *= 0.50
        lot_size = max(48.0, lot_size * 1.30)
    return NeighborhoodProfile(
        name=name,
        condition=condition,
        wealth=round(wealth, 6),
        density=round(density, 6),
        target_lot_size=round(lot_size, 3),
        palette=NEIGHBORHOOD_PALETTES[name],
    )


def choose_building_style(
    profile: NeighborhoodProfile,
    width: float,
    depth: float,
    rng: random.Random,
) -> str:
    if profile.name == "industrial":
        return "industrial"
    if profile.name == "commercial":
        return "commercial"
    if profile.name == "slum":
        return "shack" if rng.random() < 0.65 else "worn_house"
    if profile.name == "ghetto":
        if width >= 17.0 and depth >= 16.0 and rng.random() < 0.38:
            return "duplex"
        return "worn_house" if rng.random() < 0.72 else "residential"
    if profile.name == "high_class":
        return "mansion" if width >= 24.0 and depth >= 24.0 else "middle_class"
    if profile.name == "middle_class":
        if width >= 18.0 and depth >= 17.0 and rng.random() < 0.24:
            return "duplex"
        return "middle_class"
    if width >= 18.0 and depth >= 17.0 and rng.random() < 0.28:
        return "duplex"
    return "residential"


def add_solid_box(
    render: MeshBuilder,
    collision: MeshBuilder,
    center_x: float,
    center_y: float,
    minimum_z: float,
    width: float,
    depth: float,
    height: float,
    color: Color,
    side_material: int = MATERIAL_BUILDING,
    top_material: int = MATERIAL_ROOF,
) -> None:
    for builder in (render, collision):
        builder.add_box(center_x, center_y, minimum_z, width, depth, height, side_material, top_material, color)


def add_pitched_roof(
    render: MeshBuilder,
    center_x: float,
    center_y: float,
    minimum_z: float,
    width: float,
    depth: float,
    ridge_height: float,
    color: Color,
    hipped: bool = False,
    collision: MeshBuilder | None = None,
) -> None:
    x0, x1 = center_x - width * 0.5, center_x + width * 0.5
    y0, y1 = center_y - depth * 0.5, center_y + depth * 0.5
    inset = min(depth * 0.24, width * 0.24) if hipped else 0.0
    first_ridge = (center_x, y0 + inset, minimum_z + ridge_height)
    second_ridge = (center_x, y1 - inset, minimum_z + ridge_height)
    repeat_length = 3.0
    slope_repeat = math.hypot(width * 0.5, ridge_height) / repeat_length
    depth_repeat = depth / repeat_length
    inset_repeat = inset / repeat_length
    left_uvs = ((0.0, 0.0), (inset_repeat, slope_repeat),
                (depth_repeat - inset_repeat, slope_repeat), (depth_repeat, 0.0))
    right_uvs = ((inset_repeat, slope_repeat), (0.0, 0.0),
                 (depth_repeat, 0.0), (depth_repeat - inset_repeat, slope_repeat))
    end_uvs = ((0.0, 0.0), (width / repeat_length, 0.0),
               (width * 0.5 / repeat_length, math.hypot(inset, ridge_height) / repeat_length))
    builders = (render,) if collision is None else (render, collision)
    for builder in builders:
        builder.add_quad(((x0, y0, minimum_z), first_ridge, second_ridge, (x0, y1, minimum_z)), MATERIAL_ROOF, color, left_uvs)
        builder.add_quad((first_ridge, (x1, y0, minimum_z), (x1, y1, minimum_z), second_ridge), MATERIAL_ROOF, shade_color(color, 0.86), right_uvs)
        builder.add_triangle(((x0, y0, minimum_z), (x1, y0, minimum_z), first_ridge), MATERIAL_ROOF, shade_color(color, 0.92), end_uvs)
        builder.add_triangle(((x1, y1, minimum_z), (x0, y1, minimum_z), second_ridge), MATERIAL_ROOF, color, end_uvs)


def add_front_door(
    render: MeshBuilder,
    x: float,
    front_y: float,
    floor_z: float,
    width: float,
    height: float,
    color: Color,
) -> None:
    render.add_quad(
        ((x - width * 0.5, front_y, floor_z), (x + width * 0.5, front_y, floor_z),
         (x + width * 0.5, front_y, floor_z + height), (x - width * 0.5, front_y, floor_z + height)),
        MATERIAL_TRIM, color,
    )
    render.add_box(x + width * 0.30, front_y - 0.025, floor_z + height * 0.47, 0.10, 0.045, 0.10, MATERIAL_TRIM, MATERIAL_TRIM, (190, 174, 124, 255))


def add_facade_windows(
    render: MeshBuilder,
    center_x: float,
    front_y: float,
    width: float,
    minimum_z: float,
    height: float,
    bays: int,
    condition: str,
    rng: random.Random,
) -> None:
    for index in range(bays):
        x = center_x - width * 0.5 + width * (index + 0.5) / bays
        window_width = min(2.5, width / bays * 0.60)
        window_height = min(1.45, max(0.9, height * 0.42))
        z = minimum_z + min(1.05, height * 0.27)
        render.add_quad(
            ((x - window_width * 0.5, front_y, z), (x + window_width * 0.5, front_y, z),
             (x + window_width * 0.5, front_y, z + window_height), (x - window_width * 0.5, front_y, z + window_height)),
            MATERIAL_WINDOW, (168, 181, 178, 255),
        )
        if condition == "neglected" and rng.random() < 0.65:
            for board_index in range(2):
                render.add_box(x, front_y - 0.035, z + window_height * (0.20 + board_index * 0.49), window_width * 1.03, 0.04, 0.20, MATERIAL_TRIM, MATERIAL_TRIM, (129, 103, 74, 255))
        elif condition == "restored":
            for side in (-1.0, 1.0):
                render.add_box(x + side * (window_width * 0.5 + 0.12), front_y - 0.02, z - 0.06, 0.16, 0.05, window_height + 0.12, MATERIAL_TRIM, MATERIAL_TRIM, (216, 211, 192, 255))


def add_side_wall_windows(
    render: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    minimum_z: float,
    storey_height: float,
    condition: str,
    rng: random.Random,
) -> None:
    bays = max(1, min(4, int(depth / 3.5)))
    bay_span = depth * 0.78
    window_width = min(1.7, bay_span / bays * 0.58)
    window_height = min(1.45, max(0.9, storey_height * 0.42))
    z = minimum_z + min(1.05, storey_height * 0.27)
    for side in (-1.0, 1.0):
        wall_x = center_x + side * (width * 0.5 + 0.012)
        for index in range(bays):
            y = center_y - bay_span * 0.5 + bay_span * (index + 0.5) / bays
            y0, y1 = y - window_width * 0.5, y + window_width * 0.5
            if side < 0.0:
                points = ((wall_x, y0, z), (wall_x, y0, z + window_height),
                          (wall_x, y1, z + window_height), (wall_x, y1, z))
                uvs = ((0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
            else:
                points = ((wall_x, y0, z), (wall_x, y1, z),
                          (wall_x, y1, z + window_height), (wall_x, y0, z + window_height))
                uvs = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))
            render.add_quad(points, MATERIAL_WINDOW, (168, 181, 178, 255), uvs)
            if condition == "neglected" and rng.random() < 0.45:
                for board in (0.23, 0.72):
                    render.add_box(wall_x + side * 0.025, y, z + window_height * board,
                                   0.04, window_width * 1.04, 0.18,
                                   MATERIAL_TRIM, MATERIAL_TRIM, (129, 103, 74, 255))


def add_condition_details(
    render: MeshBuilder,
    center_x: float,
    front_y: float,
    width: float,
    wall_height: float,
    condition: str,
    rng: random.Random,
) -> None:
    if condition not in {"neglected", "weathered"}:
        return
    count = 5 if condition == "neglected" else 2
    for _ in range(count):
        patch_width = min(width * 0.19, rng.uniform(0.6, 2.0))
        patch_height = min(wall_height * 0.30, rng.uniform(0.4, 1.0))
        x = center_x + rng.uniform(-0.35, 0.35) * width
        z = 0.30 + rng.uniform(0.0, max(0.1, wall_height - patch_height - 0.45))
        color = rng.choice(((121, 115, 99, 255), (163, 151, 129, 255), (143, 125, 105, 255)))
        render.add_quad(
            ((x - patch_width * 0.5, front_y, z), (x + patch_width * 0.5, front_y, z + patch_height * 0.12),
             (x + patch_width * 0.40, front_y, z + patch_height), (x - patch_width * 0.40, front_y, z + patch_height * 0.82)),
            MATERIAL_BUILDING, color,
        )


def add_front_walk(render: MeshBuilder, center_x: float, center_y: float, width: float, depth: float, front_y: float) -> None:
    end_y = center_y - depth * 0.46
    length = max(0.05, front_y - end_y)
    render.add_box(center_x, (front_y + end_y) * 0.5, 0.225, min(1.35, width * 0.10), length, 0.025, MATERIAL_SIDEWALK, MATERIAL_SIDEWALK, (183, 178, 162, 255))


def add_house_mass(
    render: MeshBuilder,
    collision: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    height: float,
    color: Color,
    rng: random.Random,
    condition: str,
    two_storey: bool = False,
    hipped: bool = False,
) -> tuple[float, float, float]:
    floor_z = 0.22
    wall_height = max(5.8, min(height * 0.76, 6.8)) if two_storey else max(2.7, min(height * 0.72, 3.8))
    roof_height = min(2.0, max(0.85, height - wall_height))
    add_solid_box(render, collision, center_x, center_y, floor_z, width, depth, wall_height, color)
    add_pitched_roof(render, center_x, center_y, floor_z + wall_height, width * 1.025, depth * 1.025, roof_height, shade_color(color, 0.62), hipped, collision)
    front_y = center_y - depth * 0.5 - 0.012
    bays = max(2, min(5, int(width / 3.1)))
    storeys = 2 if two_storey else 1
    storey_height = wall_height / storeys
    for storey in range(storeys):
        storey_z = floor_z + storey * storey_height
        add_facade_windows(render, center_x, front_y, width * 0.84, storey_z, storey_height, bays, condition, rng)
        add_side_wall_windows(render, center_x, center_y, width, depth, storey_z, storey_height, condition, rng)
    if two_storey:
        render.add_box(center_x, front_y - 0.025, floor_z + storey_height - 0.08,
                       width, 0.06, 0.16, MATERIAL_TRIM, MATERIAL_TRIM, shade_color(color, 1.06))
    add_front_door(render, center_x, front_y - 0.025, floor_z, min(1.15, width * 0.12), 2.2, shade_color(color, 0.50))
    add_condition_details(render, center_x, front_y - 0.045, width, wall_height, condition, rng)
    return front_y, wall_height, roof_height


def build_shack(
    render: MeshBuilder, collision: MeshBuilder,
    center_x: float, center_y: float, width: float, depth: float, height: float,
    color: Color, rng: random.Random, condition: str,
) -> int:
    main_width, main_depth = min(width * 0.67, 10.0), min(depth * 0.53, 8.0)
    wall_height = rng.uniform(2.65, 3.25)
    main_x = center_x - min(width * 0.06, 0.8)
    main_y = center_y + min(depth * 0.08, 1.2)
    add_solid_box(render, collision, main_x, main_y, 0.22, main_width, main_depth, wall_height, color)
    roof_z = 0.22 + wall_height
    x0, x1 = main_x - main_width * 0.55, main_x + main_width * 0.55
    y0, y1 = main_y - main_depth * 0.55, main_y + main_depth * 0.55
    roof_points = ((x0, y0, roof_z + 0.30), (x1, y0, roof_z - 0.10), (x1, y1, roof_z - 0.10), (x0, y1, roof_z + 0.30))
    roof_uvs = ((0.0, 0.0), ((x1 - x0) / 3.0, 0.0),
                ((x1 - x0) / 3.0, (y1 - y0) / 3.0), (0.0, (y1 - y0) / 3.0))
    render.add_double_sided_quad(roof_points, MATERIAL_ROOF, (126, 121, 105, 255), roof_uvs)
    collision.add_quad(roof_points, MATERIAL_ROOF, (126, 121, 105, 255), roof_uvs)
    for index in range(6):
        x = x0 + (x1 - x0) * (index + 0.5) / 6.0
        sheet_height = roof_z + 0.30 - (x - x0) / (x1 - x0) * 0.40
        render.add_box(x, main_y, sheet_height + 0.015, 0.10, main_depth * 1.10, 0.07, MATERIAL_TRIM, MATERIAL_TRIM, (119, 112, 92, 255))
    shed_width, shed_depth = min(width * 0.23, 3.2), min(depth * 0.35, 5.0)
    shed_x = main_x + main_width * 0.5 + shed_width * 0.5 - 0.20
    shed_y = main_y + main_depth * 0.14
    add_solid_box(render, collision, shed_x, shed_y, 0.22, shed_width, shed_depth, wall_height * 0.71, shade_color(color, 0.82))
    front_y = main_y - main_depth * 0.5 - 0.015
    add_front_door(render, main_x - main_width * 0.09, front_y, 0.22, 0.95, 1.95, (101, 85, 64, 255))
    add_facade_windows(render, main_x, front_y - 0.01, main_width * 0.78, 0.22, wall_height, 2, condition, rng)
    add_side_wall_windows(render, main_x, main_y, main_width, main_depth, 0.22, wall_height, condition, rng)
    add_condition_details(render, main_x, front_y - 0.03, main_width, wall_height, condition, rng)
    return 1


def build_residential_house(
    render: MeshBuilder, collision: MeshBuilder,
    center_x: float, center_y: float, width: float, depth: float, height: float,
    color: Color, rng: random.Random, condition: str, middle_class: bool = False,
) -> int:
    main_width = min(width * 0.60, 18.0) if middle_class else min(width * 0.73, 14.0)
    main_depth = min(depth * 0.59, 14.0) if middle_class else min(depth * 0.61, 12.0)
    main_x = center_x - min(width * 0.10, 3.0) if middle_class else center_x - min(width * 0.035, 1.2)
    main_y = center_y + min(depth * 0.05, 1.6)
    front_y, wall_height, roof_height = add_house_mass(
        render, collision, main_x, main_y, main_width, main_depth, height,
        color, rng, condition, two_storey=middle_class, hipped=middle_class,
    )
    porch_width = min(6.0, main_width * 0.50)
    porch_depth = min(2.0, main_depth * 0.18)
    porch_y = front_y - porch_depth * 0.5
    add_solid_box(render, collision, main_x, porch_y, 0.22, porch_width, porch_depth, 0.18, shade_color(color, 0.90), MATERIAL_SIDEWALK, MATERIAL_SIDEWALK)
    render.add_box(main_x, porch_y, 2.58, porch_width, porch_depth * 1.08, 0.16, MATERIAL_TRIM, MATERIAL_ROOF, shade_color(color, 0.76))
    for side in (-1.0, 1.0):
        render.add_box(main_x + side * porch_width * 0.42, porch_y - porch_depth * 0.35, 0.40, 0.16, 0.16, 2.18, MATERIAL_TRIM, MATERIAL_TRIM, (212, 207, 185, 255))
    if middle_class:
        garage_width, garage_depth = min(width * 0.26, 6.4), min(depth * 0.40, 7.5)
        garage_x = main_x + main_width * 0.5 + garage_width * 0.5 - 0.15
        garage_y = main_y - main_depth * 0.5 + garage_depth * 0.5 + 0.30
        add_solid_box(render, collision, garage_x, garage_y, 0.22, garage_width, garage_depth, 2.8, shade_color(color, 0.94))
        add_pitched_roof(render, garage_x, garage_y, 3.02, garage_width * 1.025, garage_depth * 1.025, min(1.2, roof_height), shade_color(color, 0.60), True, collision)
        garage_front = garage_y - garage_depth * 0.5 - 0.015
        render.add_quad(((garage_x - garage_width * 0.38, garage_front, 0.28), (garage_x + garage_width * 0.38, garage_front, 0.28), (garage_x + garage_width * 0.38, garage_front, 2.55), (garage_x - garage_width * 0.38, garage_front, 2.55)), MATERIAL_TRIM, (175, 173, 157, 255))
        driveway_end = center_y - depth * 0.46
        driveway_depth = max(0.05, garage_front - driveway_end)
        render.add_box(garage_x, (garage_front + driveway_end) * 0.5, 0.225, garage_width * 0.88, driveway_depth, 0.03, MATERIAL_SIDEWALK, MATERIAL_SIDEWALK, (183, 178, 160, 255))
    else:
        chimney_x = main_x + main_width * 0.22
        add_solid_box(render, collision, chimney_x, main_y + main_depth * 0.15, 0.22 + wall_height, min(1.0, main_width * 0.12), min(1.0, main_depth * 0.12), roof_height * 1.05, shade_color(color, 0.59))
    add_front_walk(render, main_x, center_y, width, depth, front_y)
    return 1



def build_worn_house(
    render: MeshBuilder, collision: MeshBuilder,
    center_x: float, center_y: float, width: float, depth: float, height: float,
    color: Color, rng: random.Random, condition: str,
) -> int:
    build_residential_house(render, collision, center_x, center_y, width, depth, min(height, 6.1), color, rng, condition)
    main_width, main_depth = min(width * 0.73, 14.0), min(depth * 0.61, 12.0)
    main_x = center_x - min(width * 0.035, 1.2)
    main_y = center_y + min(depth * 0.05, 1.6)
    extension_width, extension_depth = min(width * 0.34, 5.6), min(depth * 0.23, 3.5)
    rear_x = main_x - main_width * 0.24
    rear_y = min(main_y + main_depth * 0.5 + extension_depth * 0.5 - 0.20,
                 center_y + depth * 0.5 - extension_depth * 0.5 - 0.10)
    add_solid_box(render, collision, rear_x, rear_y, 0.22, extension_width, extension_depth, 2.35, shade_color(color, 0.81))
    add_solid_box(render, collision, rear_x, rear_y, 2.57, extension_width * 1.02, extension_depth * 1.03, 0.12, (122, 119, 105, 255), MATERIAL_ROOF, MATERIAL_ROOF)
    for index in range(3):
        render.add_box(rear_x + (index - 1) * extension_width * 0.25, rear_y, 2.69, extension_width * 0.19, extension_depth * 0.82, 0.025, MATERIAL_ROOF, MATERIAL_ROOF, (144, 136, 112, 255))
    return 1


def build_duplex(
    render: MeshBuilder, collision: MeshBuilder,
    center_x: float, center_y: float, width: float, depth: float, height: float,
    color: Color, rng: random.Random, condition: str,
) -> int:
    unit_width, unit_depth = min(width * 0.43, 9.5), min(depth * 0.62, 12.0)
    unit_y = center_y + min(depth * 0.04, 1.2)
    for index, side in enumerate((-1.0, 1.0)):
        unit_x = center_x + side * unit_width * 0.5
        unit_color = color if index == 0 else shade_color(color, 0.92)
        front_y, _, _ = add_house_mass(render, collision, unit_x, unit_y, unit_width, unit_depth, min(7.0, height), unit_color, rng, condition)
        add_solid_box(render, collision, unit_x, front_y - depth * 0.035, 0.22, unit_width * 0.47, depth * 0.07, 0.18, (176, 170, 151, 255), MATERIAL_SIDEWALK, MATERIAL_SIDEWALK)
        add_front_walk(render, unit_x, center_y, unit_width, depth, front_y)
    render.add_box(center_x, unit_y, 0.22, 0.16, depth * 0.72, 1.05, MATERIAL_TRIM, MATERIAL_TRIM, shade_color(color, 0.70))
    return 2


def build_mansion(
    render: MeshBuilder, collision: MeshBuilder,
    center_x: float, center_y: float, width: float, depth: float, height: float,
    color: Color, rng: random.Random, condition: str,
) -> int:
    main_width, main_depth = width * 0.48, depth * 0.42
    main_y = center_y + depth * 0.14
    storey_height = max(6.0, min(8.5, height))
    add_house_mass(render, collision, center_x, main_y, main_width, main_depth, storey_height + 1.8, color, rng, condition, True, True)
    for side in (-1.0, 1.0):
        wing_width, wing_depth = width * 0.18, depth * 0.48
        wing_x, wing_y = center_x + side * width * 0.31, center_y - depth * 0.015
        add_house_mass(render, collision, wing_x, wing_y, wing_width, wing_depth, storey_height * 0.67, shade_color(color, 0.96), rng, condition, False, True)
    entrance_y = main_y - main_depth * 0.5 - depth * 0.07
    add_solid_box(render, collision, center_x, entrance_y, 0.22, width * 0.25, depth * 0.14, 0.22, (210, 203, 181, 255), MATERIAL_SIDEWALK, MATERIAL_SIDEWALK)
    render.add_box(center_x, entrance_y, 3.15, width * 0.26, depth * 0.15, 0.24, MATERIAL_TRIM, MATERIAL_ROOF, shade_color(color, 0.85))
    for side in (-1.0, 1.0):
        add_solid_box(render, collision, center_x + side * width * 0.095, entrance_y - depth * 0.045, 0.44, 0.30, 0.30, 2.71, (224, 217, 196, 255), MATERIAL_TRIM, MATERIAL_TRIM)
        add_solid_box(render, collision, center_x + side * width * 0.465, center_y, 0.22, width * 0.025, depth * 0.91, 1.05, shade_color(color, 0.88))
        render.add_box(center_x + side * width * 0.28, center_y - depth * 0.30, 0.225, width * 0.20, depth * 0.19, 0.045, MATERIAL_GRASS, MATERIAL_GRASS, (161, 181, 136, 255))
        add_solid_box(render, collision, center_x + side * width * 0.31, center_y - depth * 0.45, 0.22, width * 0.30, depth * 0.025, 1.05, shade_color(color, 0.88))
    add_solid_box(render, collision, center_x, center_y + depth * 0.46, 0.22, width * 0.95, depth * 0.025, 1.05, shade_color(color, 0.88))
    add_front_walk(render, center_x, center_y, width, depth, entrance_y)
    return 1


def build_commercial_strip(
    render: MeshBuilder, collision: MeshBuilder,
    center_x: float, center_y: float, width: float, depth: float, height: float,
    color: Color, rng: random.Random, condition: str,
) -> int:
    shops = max(1, min(3, int(width / 7.0)))
    shop_width = width * 0.86 / shops
    main_depth = depth * 0.64
    main_y = center_y + depth * 0.08
    wall_height = min(8.0, max(3.8, height * 0.70))
    for index in range(shops):
        shop_x = center_x - width * 0.43 + shop_width * (index + 0.5)
        shop_color = shade_color(color, 0.90 + index * 0.05)
        add_solid_box(render, collision, shop_x, main_y, 0.22, shop_width * 0.98, main_depth, wall_height, shop_color)
        front_y = main_y - main_depth * 0.5 - 0.015
        add_facade_windows(render, shop_x, front_y, shop_width * 0.84, 0.05, 4.8, 2, condition, rng)
        add_side_wall_windows(render, shop_x, main_y, shop_width * 0.98, main_depth,
                              0.22, min(4.5, wall_height), condition, rng)
        add_front_door(render, shop_x + shop_width * 0.22, front_y - 0.015, 0.22, min(1.25, shop_width * 0.17), 2.5, shade_color(shop_color, 0.43))
        awning_color = rng.choice(((142, 86, 68, 255), (87, 126, 111, 255), (112, 126, 151, 255)))
        render.add_box(shop_x, front_y - depth * 0.045, 3.0, shop_width * 0.93, depth * 0.09, 0.16, MATERIAL_TRIM, MATERIAL_TRIM, awning_color)
        render.add_box(shop_x, front_y - 0.03, 3.60, shop_width * 0.82, 0.09, 0.62, MATERIAL_TRIM, MATERIAL_TRIM, shade_color(shop_color, 0.59))
        add_condition_details(render, shop_x, front_y - 0.04, shop_width * 0.90, wall_height, condition, rng)
        render.add_box(shop_x, main_y + main_depth * 0.13, wall_height + 0.22, shop_width * 0.26, main_depth * 0.17, 0.72, MATERIAL_TRIM, MATERIAL_TRIM, (138, 142, 137, 255))
    return 0


def build_industrial_yard(
    render: MeshBuilder, collision: MeshBuilder,
    center_x: float, center_y: float, width: float, depth: float, height: float,
    color: Color, rng: random.Random, condition: str,
) -> int:
    main_width, main_depth = width * 0.82, depth * 0.67
    main_y = center_y + depth * 0.07
    wall_height = max(5.0, min(12.0, height * 0.78))
    add_solid_box(render, collision, center_x, main_y, 0.22, main_width, main_depth, wall_height, color)
    front_y = main_y - main_depth * 0.5 - 0.015
    add_side_wall_windows(render, center_x, main_y, main_width, main_depth,
                          0.22 + wall_height - 2.8, 2.8, condition, rng)
    bays = max(1, min(4, int(main_width / 5.0)))
    for index in range(bays):
        bay_x = center_x - main_width * 0.5 + main_width * (index + 0.5) / bays
        door_width = main_width / bays * 0.68
        render.add_quad(((bay_x - door_width * 0.5, front_y, 0.35), (bay_x + door_width * 0.5, front_y, 0.35), (bay_x + door_width * 0.5, front_y, 3.8), (bay_x - door_width * 0.5, front_y, 3.8)), MATERIAL_TRIM, (124, 131, 127, 255))
        add_solid_box(render, collision, bay_x, front_y - depth * 0.035, 0.22, door_width * 1.12, depth * 0.07, 0.43, (140, 140, 128, 255), MATERIAL_SIDEWALK, MATERIAL_SIDEWALK)
    for index in range(3):
        monitor_x = center_x + (index - 1) * main_width * 0.24
        monitor_width, monitor_depth = main_width * 0.17, main_depth * 0.68
        add_solid_box(render, collision, monitor_x, main_y, wall_height + 0.22, monitor_width, monitor_depth, 1.15, shade_color(color, 0.72), MATERIAL_TRIM, MATERIAL_ROOF)
        add_pitched_roof(render, monitor_x, main_y, wall_height + 1.37, monitor_width, monitor_depth, 0.55, shade_color(color, 0.59), collision=collision)
    add_condition_details(render, center_x, front_y - 0.03, main_width, wall_height, condition, rng)
    return 0


def build_neighborhood_building(
    render: MeshBuilder,
    collision: MeshBuilder,
    center_x: float,
    center_y: float,
    width: float,
    depth: float,
    height: float,
    color: Color,
    rng: random.Random,
    profile: NeighborhoodProfile,
    style: str | None = None,
) -> BuildingResult:
    if not all(math.isfinite(value) for value in (center_x, center_y, width, depth, height)):
        raise ValueError("Building dimensions and coordinates must be finite.")
    if min(width, depth) < 6.0 or height <= 0.0:
        raise ValueError("A neighborhood building requires an envelope of at least six units and positive height.")
    style = style or choose_building_style(profile, width, depth, rng)
    condition = profile.condition
    if style == "shack":
        units = build_shack(render, collision, center_x, center_y, width, depth, height, color, rng, condition)
    elif style == "worn_house":
        units = build_worn_house(render, collision, center_x, center_y, width, depth, height, color, rng, condition)
    elif style == "residential":
        units = build_residential_house(render, collision, center_x, center_y, width, depth, min(height, 7.1), color, rng, condition)
    elif style == "middle_class":
        units = build_residential_house(render, collision, center_x, center_y, width, depth, max(7.0, min(height, 9.8)), color, rng, condition, True)
    elif style == "duplex":
        units = build_duplex(render, collision, center_x, center_y, width, depth, height, color, rng, condition)
    elif style == "mansion":
        units = build_mansion(render, collision, center_x, center_y, width, depth, height, color, rng, condition)
    elif style == "commercial":
        units = build_commercial_strip(render, collision, center_x, center_y, width, depth, height, color, rng, condition)
    elif style == "industrial":
        units = build_industrial_yard(render, collision, center_x, center_y, width, depth, height, color, rng, condition)
    else:
        raise ValueError(f"Unknown neighborhood building style: {style}")
    return BuildingResult(style, condition, units)
