from __future__ import annotations

import math
import random
import struct
from dataclasses import dataclass
from typing import Mapping, Sequence


RW_STRUCT = 0x01
RW_EXTENSION = 0x03
RW_TEXTURE_NATIVE = 0x15
RW_TEX_DICTIONARY = 0x16
RW_LIBRARY_ID = 0x1803FFFF

MATERIAL_TEXTURE_NAMES = (
    "sai_road",
    "sai_walk",
    "sai_wall",
    "sai_roof",
    "sai_window",
    "sai_leaf",
    "sai_trim",
    "sai_grass",
)

DISTRICT_RADAR_COLORS = {
    "airport": (101, 107, 113, 255),
    "docks": (111, 91, 76, 255),
    "industrial": (125, 116, 95, 255),
    "beach": (194, 168, 102, 255),
    "hills": (79, 125, 77, 255),
    "downtown": (102, 105, 126, 255),
    "grove": (86, 137, 76, 255),
    "eastside": (135, 94, 75, 255),
    "coastal": (88, 127, 135, 255),
    "civic": (146, 119, 91, 255),
    "park": (65, 123, 73, 255),
    "commercial": (139, 104, 112, 255),
    "residential": (151, 127, 95, 255),
}


@dataclass
class TextureImage:
    width: int
    height: int
    pixels: bytearray

    @classmethod
    def solid(cls, width: int, height: int, color: tuple[int, int, int, int]) -> "TextureImage":
        return cls(width, height, bytearray(color * (width * height)))

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int, int]) -> None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        offset = (y * self.width + x) * 4
        self.pixels[offset:offset + 4] = bytes(color)

    def get_pixel(self, x: int, y: int) -> tuple[int, int, int, int]:
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        offset = (y * self.width + x) * 4
        return tuple(self.pixels[offset:offset + 4])


def generate_world_textures(seed: int) -> list[tuple[str, TextureImage, bool]]:
    random_source = random.Random(seed ^ 0x534149545844)
    return [
        ("sai_road", generate_asphalt_texture(random_source), False),
        ("sai_walk", generate_sidewalk_texture(random_source), False),
        ("sai_wall", generate_wall_texture(random_source), False),
        ("sai_roof", generate_roof_texture(random_source), False),
        ("sai_window", generate_window_texture(random_source), False),
        ("sai_leaf", generate_leaf_texture(random_source), True),
        ("sai_trim", generate_trim_texture(random_source), False),
        ("sai_grass", generate_grass_texture(random_source), False),
    ]


def generate_asphalt_texture(random_source: random.Random) -> TextureImage:
    size = 256
    broad = tileable_noise_field(random_source, size, 4)
    wear = tileable_noise_field(random_source, size, 17)
    image = TextureImage.solid(size, size, (66, 67, 66, 255))
    for y in range(size):
        for x in range(size):
            index = y * size + x
            grain = random_source.randrange(-6, 7)
            aggregate = random_source.randrange(4, 11) if random_source.random() < 0.09 else 0
            base = round(66 + broad[index] * 9.0 + wear[index] * 4.0 + grain + aggregate)
            image.set_pixel(x, y, (base, base + 1, base, 255))
    for _ in range(4):
        x = random_source.randrange(size)
        y = random_source.randrange(size)
        points = [(x, y)]
        for _ in range(random_source.randrange(5, 11)):
            x += random_source.randrange(-9, 10)
            y += random_source.randrange(3, 15)
            points.append((x, y))
        for start, end in zip(points, points[1:]):
            draw_texture_line(image, start[0], start[1], end[0], end[1], (44, 45, 43, 255), wrap=True)
    return image


def generate_sidewalk_texture(random_source: random.Random) -> TextureImage:
    size = 256
    weather = tileable_noise_field(random_source, size, 5)
    surface = tileable_noise_field(random_source, size, 33)
    slab_tones = [random_source.uniform(-3.0, 3.0) for _ in range(4)]
    image = TextureImage.solid(size, size, (151, 148, 139, 255))
    for y in range(size):
        for x in range(size):
            index = y * size + x
            joint_distance = min(x % 128, y % 128, 128 - x % 128, 128 - y % 128)
            joint_shade = 27.0 if joint_distance == 0 else max(0.0, 7.0 - joint_distance * 1.6)
            grain = random_source.randrange(-5, 6)
            base = round(148 + weather[index] * 8.0 + surface[index] * 3.0 + slab_tones[(y // 128) * 2 + x // 128] + grain - joint_shade)
            image.set_pixel(x, y, (base + 3, base, base - 8, 255))
    return image


def tileable_noise_field(random_source: random.Random, size: int, cells: int) -> list[float]:
    points = [[random_source.uniform(-1.0, 1.0) for _ in range(cells)] for _ in range(cells)]
    coordinates = []
    for index in range(size):
        position = index * cells / size
        first = math.floor(position)
        fraction = position - first
        coordinates.append((first % cells, (first + 1) % cells, fraction * fraction * (3.0 - 2.0 * fraction)))
    result: list[float] = []
    for first_y, second_y, blend_y in coordinates:
        for first_x, second_x, blend_x in coordinates:
            top = points[first_y][first_x] * (1.0 - blend_x) + points[first_y][second_x] * blend_x
            bottom = points[second_y][first_x] * (1.0 - blend_x) + points[second_y][second_x] * blend_x
            result.append(top * (1.0 - blend_y) + bottom * blend_y)
    return result


def draw_texture_line(
    image: TextureImage,
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    color: tuple[int, int, int, int],
    wrap: bool = False,
) -> None:
    steps = max(1, abs(end_x - start_x), abs(end_y - start_y))
    for step in range(steps + 1):
        factor = step / steps
        x = round(start_x + (end_x - start_x) * factor)
        y = round(start_y + (end_y - start_y) * factor)
        if wrap:
            x %= image.width
            y %= image.height
        image.set_pixel(x, y, color)


def generate_wall_texture(random_source: random.Random) -> TextureImage:
    size = 256
    weather = tileable_noise_field(random_source, size, 5)
    plaster = tileable_noise_field(random_source, size, 41)
    image = TextureImage.solid(size, size, (198, 197, 195, 255))
    for y in range(size):
        for x in range(size):
            index = y * size + x
            grain = random_source.randrange(-5, 6)
            pore = random_source.randrange(5, 12) if random_source.random() < 0.025 else 0
            base = round(196 + weather[index] * 7.0 + plaster[index] * 3.0 + grain - pore)
            image.set_pixel(x, y, (base + 2, base + 1, base - 1, 255))
    return image


def generate_roof_texture(random_source: random.Random) -> TextureImage:
    size = 128
    image = TextureImage.solid(size, size, (87, 84, 79, 255))
    for y in range(size):
        for x in range(size):
            noise = random_source.randrange(-16, 17)
            base = 87 + noise
            if (x - y) % 47 == 0:
                base -= 18
            image.set_pixel(x, y, (base, base - 2, base - 5, 255))
    for coordinate in (0, 63, 64, 127):
        for other in range(size):
            image.set_pixel(coordinate, other, (67, 66, 63, 255))
            image.set_pixel(other, coordinate, (67, 66, 63, 255))
    return image


def generate_window_texture(random_source: random.Random) -> TextureImage:
    image = TextureImage.solid(64, 64, (46, 65, 76, 255))
    for y in range(64):
        for x in range(64):
            reflection = max(0, 20 - abs((x - y) % 40 - 20))
            noise = random_source.randrange(-4, 5)
            image.set_pixel(x, y, (45 + reflection + noise, 63 + reflection + noise, 74 + reflection + noise, 255))
    for coordinate in range(64):
        for offset in (0, 1, 31, 32, 62, 63):
            image.set_pixel(offset, coordinate, (28, 30, 31, 255))
            image.set_pixel(coordinate, offset, (28, 30, 31, 255))
    return image


def generate_leaf_texture(random_source: random.Random) -> TextureImage:
    image = TextureImage.solid(128, 128, (43, 73, 38, 0))
    clusters = (
        (37.0, 48.0, 27.0, 26.0),
        (63.0, 32.0, 27.0, 23.0),
        (88.0, 48.0, 27.0, 28.0),
        (43.0, 77.0, 27.0, 26.0),
        (84.0, 80.0, 27.0, 24.0),
        (65.0, 62.0, 30.0, 29.0),
    )
    for center_x, center_y, radius_x, radius_y in clusters:
        draw_texture_line(image, 65, 108, round(center_x), round(center_y), (66, 59, 37, 220))
        for _ in range(3):
            angle = random_source.uniform(0.0, math.tau)
            end_x = center_x + math.cos(angle) * radius_x * 0.75
            end_y = center_y + math.sin(angle) * radius_y * 0.75
            draw_texture_line(image, round(center_x), round(center_y), round(end_x), round(end_y), (67, 61, 39, 220))
    for center_x, center_y, radius_x, radius_y in clusters:
        for _ in range(95):
            angle = random_source.uniform(0.0, math.tau)
            radius = math.sqrt(random_source.random())
            leaf_x = center_x + math.cos(angle) * radius_x * radius
            leaf_y = center_y + math.sin(angle) * radius_y * radius
            green = random_source.randrange(69, 112)
            draw_foliage_leaf(
                image,
                leaf_x,
                leaf_y,
                random_source.uniform(3.0, 5.8),
                random_source.uniform(1.6, 3.0),
                random_source.uniform(0.0, math.tau),
                (round(green * 0.55), green, round(green * 0.44)),
            )
    return image


def draw_foliage_leaf(
    image: TextureImage,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    angle: float,
    color: tuple[int, int, int],
) -> None:
    cosine, sine = math.cos(angle), math.sin(angle)
    extent = math.ceil(max(radius_x, radius_y))
    for y in range(max(3, math.floor(center_y) - extent), min(image.height - 3, math.ceil(center_y) + extent + 1)):
        for x in range(max(3, math.floor(center_x) - extent), min(image.width - 3, math.ceil(center_x) + extent + 1)):
            offset_x, offset_y = x + 0.5 - center_x, y + 0.5 - center_y
            local_x = offset_x * cosine + offset_y * sine
            local_y = offset_y * cosine - offset_x * sine
            distance = (local_x / radius_x) ** 2 + (local_y / radius_y) ** 2
            if distance >= 1.0:
                continue
            alpha = min(1.0, (1.0 - distance) * 5.0)
            highlight = round((1.0 - distance) * 12.0 + (4.0 if local_y < 0.0 else -3.0))
            source = tuple(max(0, min(255, component + highlight)) for component in color)
            previous = image.get_pixel(x, y)
            previous_alpha = previous[3] / 255.0
            combined_alpha = alpha + previous_alpha * (1.0 - alpha)
            blended = tuple(
                round((source[channel] * alpha + previous[channel] * previous_alpha * (1.0 - alpha)) / combined_alpha)
                for channel in range(3)
            )
            image.set_pixel(x, y, blended + (round(combined_alpha * 255.0),))


def generate_trim_texture(random_source: random.Random) -> TextureImage:
    size = 128
    image = TextureImage.solid(size, size, (112, 104, 92, 255))
    for y in range(size):
        for x in range(size):
            grain = round(9.0 * math.sin((x + y * 0.18) * 0.42)) + random_source.randrange(-5, 6)
            image.set_pixel(x, y, (113 + grain, 104 + grain, 91 + grain, 255))
    return image


def generate_grass_texture(random_source: random.Random) -> TextureImage:
    size = 256
    patches = tileable_noise_field(random_source, size, 6)
    growth = tileable_noise_field(random_source, size, 29)
    image = TextureImage.solid(size, size, (68, 88, 50, 255))
    for y in range(size):
        for x in range(size):
            index = y * size + x
            grain = random_source.randrange(-6, 7)
            patch = patches[index]
            dry = max(0.0, patch - 0.12)
            blade = random_source.randrange(3, 11) if random_source.random() < 0.12 else 0
            detail = growth[index] * 6.0
            image.set_pixel(x, y, (
                round(65 + dry * 28.0 + detail + grain * 0.6),
                round(86 - dry * 4.0 + detail + grain + blade),
                round(48 + dry * 10.0 + detail * 0.5 + grain * 0.4),
                255,
            ))
    return image


def generate_seed_radar_tiles(
    seed: int,
    tile_districts: list[str],
    columns: int,
    rows: int,
    traffic_segments: list[object] | None = None,
    models: list[object] | None = None,
    placements: list[object] | None = None,
    water_surfaces: Sequence[object] = (),
    preserved_bounds: Sequence[tuple[float, float, float, float]] = (),
    retail_radar_tiles: Mapping[str, bytes] | None = None,
) -> dict[str, bytes]:
    if columns <= 0 or rows <= 0 or len(tile_districts) != columns * rows:
        raise RuntimeError("Seed radar generation requires one district per map tile.")
    output: dict[str, bytes] = {}
    for radar_y in range(6, 12):
        for radar_x in range(6, 12):
            radar_index = radar_y * 12 + radar_x
            texture_name = f"radar{radar_index:02d}"
            source_data = None
            if retail_radar_tiles is not None:
                source_data = retail_radar_tiles.get(f"{texture_name}.txd", retail_radar_tiles.get(texture_name))
            if models is not None and placements is not None:
                image = render_geometry_radar_tile(
                    radar_x, radar_y, models, placements, water_surfaces, preserved_bounds, source_data,
                )
                # Road polygons already come from the same NODES graph. Drawing
                # every centreline again made the radar a doubled, thick tangle.
            else:
                image = TextureImage.solid(256, 256, (42, 55, 50, 255))
                for sub_y in range(2):
                    for sub_x in range(2):
                        source_column = (radar_x - 6) * 2 + sub_x
                        source_row = (11 - radar_y) * 2 + (1 - sub_y)
                        world_column = min(columns - 1, source_column * columns // 12)
                        world_row = min(rows - 1, source_row * rows // 12)
                        district = tile_districts[world_row * columns + world_column]
                        draw_radar_district_cell(image, sub_x * 128, sub_y * 128, district, seed, world_column, world_row)
                for segment in traffic_segments or []:
                    draw_radar_road_segment(image, radar_x, radar_y, segment)
                if water_surfaces:
                    overlay_native_water(image, radar_x, radar_y, water_surfaces)
                if radar_bounds_overlap(radar_x, radar_y, preserved_bounds):
                    if source_data is None:
                        raise RuntimeError(f"Preserved gameplay terrain requires its installed {texture_name}.txd radar texture.")
                    image = composite_preserved_radar(image, source_data, radar_x, radar_y, preserved_bounds)
            output[f"{texture_name}.txd"] = write_texture_dictionary(
                [(texture_name, image, False)], address_mode="clamp"
            )
    return output


def render_geometry_radar_tile(
    radar_x: int,
    radar_y: int,
    models: list[object],
    placements: list[object],
    water_surfaces: Sequence[object] = (),
    preserved_bounds: Sequence[tuple[float, float, float, float]] = (),
    retail_radar_data: bytes | None = None,
) -> TextureImage:
    if len(models) != len(placements):
        raise RuntimeError("Radar rendering requires one placement per generated model.")
    water_color = (49, 91, 108, 255)
    image = TextureImage.solid(256, 256, water_color)
    depth_buffer = [-100_000.0] * (256 * 256)
    world_left = -3000.0 + radar_x * 500.0
    world_top = 3000.0 - radar_y * 500.0
    world_right = world_left + 500.0
    world_bottom = world_top - 500.0

    for model, placement in zip(models, placements):
        placement_x = float(getattr(placement, "x"))
        placement_y = float(getattr(placement, "y"))
        placement_z = float(getattr(placement, "z"))
        if not (world_left - 220.0 <= placement_x <= world_right + 220.0):
            continue
        if not (world_bottom - 220.0 <= placement_y <= world_top + 220.0):
            continue
        mesh = getattr(model, "render_mesh")
        district = str(getattr(model, "style", "residential"))
        for first, second, third, material in mesh.triangles:
            vertices = (mesh.vertices[first], mesh.vertices[second], mesh.vertices[third])
            normal_z = (mesh.normals[first][2] + mesh.normals[second][2] + mesh.normals[third][2]) / 3.0
            world_vertices = tuple(
                (vertex[0] + placement_x, vertex[1] + placement_y, vertex[2] + placement_z)
                for vertex in vertices
            )
            if material == 5:
                # Individual canopies turn the 36-tile radar into a field of
                # dots. Parks remain visible through their grass/parcel shape.
                continue
            if normal_z <= 0.20 or material not in {0, 1, 2, 3, 6, 7}:
                continue
            pixel_vertices = tuple(
                (
                    (vertex[0] - world_left) * 256.0 / 500.0,
                    (world_top - vertex[1]) * 256.0 / 500.0,
                    vertex[2],
                )
                for vertex in world_vertices
            )
            if (
                max(vertex[0] for vertex in pixel_vertices) < 0.0
                or min(vertex[0] for vertex in pixel_vertices) > 255.0
                or max(vertex[1] for vertex in pixel_vertices) < 0.0
                or min(vertex[1] for vertex in pixel_vertices) > 255.0
            ):
                continue
            color = radar_material_color(material, district)
            rasterize_radar_triangle(
                image,
                depth_buffer,
                pixel_vertices,
                color,
            )
    if water_surfaces:
        overlay_native_water(image, radar_x, radar_y, water_surfaces, depth_buffer)
    if radar_bounds_overlap(radar_x, radar_y, preserved_bounds):
        if retail_radar_data is None:
            raise RuntimeError(f"Preserved gameplay terrain requires installed radar{radar_y * 12 + radar_x:02d}.txd pixels.")
        image = composite_preserved_radar(image, retail_radar_data, radar_x, radar_y, preserved_bounds)
    return image


def radar_bounds_overlap(
    radar_x: int,
    radar_y: int,
    protected_bounds: Sequence[tuple[float, float, float, float]],
) -> bool:
    world_left = -3000.0 + radar_x * 500.0
    world_top = 3000.0 - radar_y * 500.0
    return any(
        minimum_x < world_left + 500.0 and maximum_x > world_left
        and minimum_y < world_top and maximum_y > world_top - 500.0
        for minimum_x, minimum_y, maximum_x, maximum_y in protected_bounds
    )


def composite_preserved_radar(
    image: TextureImage,
    retail_data: bytes,
    radar_x: int,
    radar_y: int,
    protected_bounds: Sequence[tuple[float, float, float, float]],
) -> TextureImage:
    if image.width <= 0 or image.height <= 0 or len(image.pixels) != image.width * image.height * 4:
        raise ValueError("Radar compositing requires a complete RGBA image.")
    if any(
        len(bounds) != 4 or not all(math.isfinite(value) for value in bounds)
        or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]
        for bounds in protected_bounds
    ):
        raise ValueError("Preserved radar bounds must be finite, nondegenerate world rectangles.")
    result = TextureImage(image.width, image.height, bytearray(image.pixels))
    if not radar_bounds_overlap(radar_x, radar_y, protected_bounds):
        return result
    retail = decode_texture_dictionary_texture(retail_data, f"radar{radar_y * 12 + radar_x:02d}")
    world_left = -3000.0 + radar_x * 500.0
    world_top = 3000.0 - radar_y * 500.0
    for minimum_x, minimum_y, maximum_x, maximum_y in protected_bounds:
        left = max(0, math.ceil((minimum_x - world_left) * image.width / 500.0 - 0.5))
        right = min(image.width, math.ceil((maximum_x - world_left) * image.width / 500.0 - 0.5))
        top = max(0, math.floor((world_top - maximum_y) * image.height / 500.0 - 0.5) + 1)
        bottom = min(image.height, math.floor((world_top - minimum_y) * image.height / 500.0 - 0.5) + 1)
        for y in range(top, bottom):
            source_y = min(retail.height - 1, (2 * y + 1) * retail.height // (2 * image.height))
            for x in range(left, right):
                source_x = min(retail.width - 1, (2 * x + 1) * retail.width // (2 * image.width))
                result.set_pixel(x, y, retail.get_pixel(source_x, source_y))
    return result


def overlay_native_water(
    image: TextureImage,
    radar_x: int,
    radar_y: int,
    surfaces: Sequence[object],
    depth_buffer: list[float] | None = None,
) -> TextureImage:
    if image.width <= 0 or image.height <= 0 or len(image.pixels) != image.width * image.height * 4:
        raise ValueError("Water radar rendering requires a complete RGBA image.")
    if depth_buffer is None:
        depth_buffer = [-math.inf] * (image.width * image.height)
    elif len(depth_buffer) != image.width * image.height:
        raise ValueError("Radar water depth buffer does not match the image dimensions.")
    world_left = -3000.0 + radar_x * 500.0
    world_top = 3000.0 - radar_y * 500.0
    for surface in surfaces:
        vertices = tuple(getattr(surface, "vertices"))
        if len(vertices) not in {3, 4}:
            raise ValueError("Native radar water requires triangle or quad surfaces.")
        if not all(math.isfinite(float(getattr(vertex, axis))) for vertex in vertices for axis in ("x", "y", "z")):
            raise ValueError("Native radar water contains a non-finite coordinate.")
        if min(float(vertex.z) for vertex in vertices) > 950.0:
            continue
        if (
            max(float(vertex.x) for vertex in vertices) < world_left
            or min(float(vertex.x) for vertex in vertices) > world_left + 500.0
            or max(float(vertex.y) for vertex in vertices) < world_top - 500.0
            or min(float(vertex.y) for vertex in vertices) > world_top
        ):
            continue
        center_x = sum(float(vertex.x) for vertex in vertices) / len(vertices)
        center_y = sum(float(vertex.y) for vertex in vertices) / len(vertices)
        vertices = tuple(sorted(vertices, key=lambda vertex: math.atan2(float(vertex.y) - center_y, float(vertex.x) - center_x)))
        projected = tuple(
            (
                (float(vertex.x) - world_left) * image.width / 500.0,
                (world_top - float(vertex.y)) * image.height / 500.0,
                float(vertex.z),
            )
            for vertex in vertices
        )
        for index in range(1, len(projected) - 1):
            rasterize_radar_triangle(image, depth_buffer, (projected[0], projected[index], projected[index + 1]), (49, 91, 108, 255))
    return image


def radar_material_color(material: int, district: str) -> tuple[int, int, int, int]:
    if material == 0:
        return 58, 61, 61, 255
    if material == 1:
        return 174, 164, 145, 255
    if material == 7:
        return 91, 119, 72, 255
    if material == 6:
        return 198, 184, 139, 255
    roof_colors = {
        "airport": (119, 126, 127, 255),
        "docks": (118, 91, 73, 255),
        "industrial": (119, 111, 93, 255),
        "beach": (171, 143, 101, 255),
        "hills": (143, 124, 94, 255),
        "downtown": (112, 111, 121, 255),
        "grove": (137, 115, 88, 255),
        "eastside": (137, 98, 78, 255),
        "coastal": (132, 116, 91, 255),
        "civic": (145, 125, 101, 255),
        "commercial": (136, 108, 105, 255),
        "residential": (146, 124, 93, 255),
    }
    return roof_colors.get(district, (132, 113, 91, 255))


def rasterize_radar_triangle(
    image: TextureImage,
    depth_buffer: list[float],
    vertices: tuple[tuple[float, float, float], ...],
    color: tuple[int, int, int, int],
) -> None:
    minimum_x = max(0, math.floor(min(vertex[0] for vertex in vertices)))
    maximum_x = min(image.width - 1, math.ceil(max(vertex[0] for vertex in vertices)))
    minimum_y = max(0, math.floor(min(vertex[1] for vertex in vertices)))
    maximum_y = min(image.height - 1, math.ceil(max(vertex[1] for vertex in vertices)))
    first, second, third = vertices
    denominator = (
        (second[1] - third[1]) * (first[0] - third[0])
        + (third[0] - second[0]) * (first[1] - third[1])
    )
    if abs(denominator) <= 0.000001:
        return
    for y in range(minimum_y, maximum_y + 1):
        sample_y = y + 0.5
        for x in range(minimum_x, maximum_x + 1):
            sample_x = x + 0.5
            first_weight = (
                (second[1] - third[1]) * (sample_x - third[0])
                + (third[0] - second[0]) * (sample_y - third[1])
            ) / denominator
            second_weight = (
                (third[1] - first[1]) * (sample_x - third[0])
                + (first[0] - third[0]) * (sample_y - third[1])
            ) / denominator
            third_weight = 1.0 - first_weight - second_weight
            if first_weight < -0.0001 or second_weight < -0.0001 or third_weight < -0.0001:
                continue
            depth = first_weight * first[2] + second_weight * second[2] + third_weight * third[2]
            depth_index = y * image.width + x
            if depth + 0.0001 < depth_buffer[depth_index]:
                continue
            depth_buffer[depth_index] = depth
            image.set_pixel(x, y, color)


def draw_radar_road_segment(image: TextureImage, radar_x: int, radar_y: int, segment: object) -> None:
    world_left = -3000.0 + radar_x * 500.0
    world_top = 3000.0 - radar_y * 500.0
    world_right = world_left + 500.0
    world_bottom = world_top - 500.0
    start_x = float(getattr(segment, "start_x"))
    start_y = float(getattr(segment, "start_y"))
    end_x = float(getattr(segment, "end_x"))
    end_y = float(getattr(segment, "end_y"))
    if (
        max(start_x, end_x) < world_left
        or min(start_x, end_x) > world_right
        or max(start_y, end_y) < world_bottom
        or min(start_y, end_y) > world_top
    ):
        return
    pixel_start_x = (start_x - world_left) * 256.0 / 500.0
    pixel_start_y = (world_top - start_y) * 256.0 / 500.0
    pixel_end_x = (end_x - world_left) * 256.0 / 500.0
    pixel_end_y = (world_top - end_y) * 256.0 / 500.0
    steps = max(1, math.ceil(max(abs(pixel_end_x - pixel_start_x), abs(pixel_end_y - pixel_start_y))))
    radius = max(1, min(7, round(float(getattr(segment, "width", 16.0)) * 256.0 / 1000.0)))
    points = [
        (
            round(pixel_start_x + (pixel_end_x - pixel_start_x) * step / steps),
            round(pixel_start_y + (pixel_end_y - pixel_start_y) * step / steps),
        )
        for step in range(steps + 1)
    ]
    for x, y in points:
        draw_disc(image, x, y, radius + 2, (178, 168, 149, 255))
    for x, y in points:
        draw_disc(image, x, y, radius + 1, (92, 94, 94, 255))
    for x, y in points:
        draw_disc(image, x, y, radius, (58, 61, 62, 255))


def draw_radar_district_cell(
    image: TextureImage,
    origin_x: int,
    origin_y: int,
    district: str,
    seed: int,
    world_column: int,
    world_row: int,
) -> None:
    road = (52, 58, 60, 255)
    sidewalk = (133, 126, 111, 255)
    district_color = DISTRICT_RADAR_COLORS.get(district, (125, 120, 107, 255))
    fill_rectangle(image, origin_x, origin_y, origin_x + 127, origin_y + 127, road)
    fill_rectangle(image, origin_x + 7, origin_y + 7, origin_x + 120, origin_y + 120, sidewalk)
    fill_rectangle(image, origin_x + 11, origin_y + 11, origin_x + 116, origin_y + 116, district_color)

    random_source = random.Random(seed ^ (world_column * 0x45D9F3B) ^ (world_row * 0x119DE1F3))
    if district in {"beach", "park"}:
        for _ in range(18):
            x = random_source.randrange(origin_x + 16, origin_x + 112)
            y = random_source.randrange(origin_y + 16, origin_y + 112)
            draw_disc(image, x, y, random_source.randrange(2, 5), (42, 88, 48, 255))
        return
    if district == "airport":
        fill_rectangle(image, origin_x + 17, origin_y + 48, origin_x + 110, origin_y + 79, (65, 69, 71, 255))
        fill_rectangle(image, origin_x + 21, origin_y + 62, origin_x + 106, origin_y + 65, (204, 194, 153, 255))
        return

    building_color = darken(district_color, 0.58)
    for lot_y in range(2):
        for lot_x in range(2):
            left = origin_x + 16 + lot_x * 52 + random_source.randrange(0, 6)
            top = origin_y + 16 + lot_y * 52 + random_source.randrange(0, 6)
            width = random_source.randrange(28, 43)
            height = random_source.randrange(27, 43)
            fill_rectangle(image, left, top, left + width, top + height, building_color)


def fill_rectangle(
    image: TextureImage,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: tuple[int, int, int, int],
) -> None:
    for y in range(max(0, top), min(image.height, bottom + 1)):
        for x in range(max(0, left), min(image.width, right + 1)):
            image.set_pixel(x, y, color)


def draw_disc(image: TextureImage, center_x: int, center_y: int, radius: int, color: tuple[int, int, int, int]) -> None:
    for y in range(center_y - radius, center_y + radius + 1):
        for x in range(center_x - radius, center_x + radius + 1):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                image.set_pixel(x, y, color)


def darken(color: tuple[int, int, int, int], factor: float) -> tuple[int, int, int, int]:
    return tuple(round(component * factor) for component in color[:3]) + (color[3],)


def write_texture_dictionary(textures: list[tuple[str, TextureImage, bool]], address_mode: str = "wrap") -> bytes:
    if len(textures) > 65535:
        raise ValueError("A TXD cannot contain more than 65535 textures.")
    names = [name.lower() for name, _, _ in textures]
    if len(names) != len(set(names)):
        raise ValueError("A TXD cannot contain duplicate texture names.")
    dictionary_struct = chunk(RW_STRUCT, struct.pack("<HH", len(textures), 2))
    native_chunks = b"".join(write_native_texture(name, image, alpha, address_mode) for name, image, alpha in textures)
    return chunk(RW_TEX_DICTIONARY, dictionary_struct + native_chunks + chunk(RW_EXTENSION, b""))


def augment_texture_dictionary(source: bytes, textures: list[tuple[str, TextureImage, bool]]) -> bytes:
    if len(source) < 24:
        raise RuntimeError("Retail TXD is shorter than a RenderWare dictionary header.")
    root_type, root_size, root_version = struct.unpack_from("<III", source, 0)
    if root_type != RW_TEX_DICTIONARY or root_size + 12 > len(source):
        raise RuntimeError("Retail TXD is not a complete RenderWare texture dictionary.")
    payload = source[12:12 + root_size]
    children = parse_chunks(payload)
    if not children or children[0][0] != RW_STRUCT or children[0][1] < 4:
        raise RuntimeError("Retail TXD has no valid dictionary Struct chunk.")

    existing_names = {
        native_texture_name(payload[start:end]).lower()
        for chunk_type, _, start, end in children
        if chunk_type == RW_TEXTURE_NATIVE
    }
    additions: list[tuple[str, TextureImage, bool]] = []
    for texture in textures:
        name = texture[0].lower()
        if name not in existing_names:
            additions.append(texture)
            existing_names.add(name)
    if not additions:
        return source[:12 + root_size]

    struct_type, struct_size, struct_start, struct_end = children[0]
    struct_payload = bytearray(payload[struct_start + 12:struct_end])
    texture_count = struct.unpack_from("<H", struct_payload, 0)[0]
    if texture_count + len(additions) > 65535:
        raise RuntimeError("Augmented TXD exceeds its 65535-texture dictionary limit.")
    struct.pack_into("<H", struct_payload, 0, texture_count + len(additions))
    rebuilt_children = [chunk(struct_type, bytes(struct_payload), root_version)]
    inserted = False
    for chunk_type, _, start, end in children[1:]:
        if chunk_type == RW_EXTENSION and not inserted:
            rebuilt_children.extend(write_native_texture(name, image, alpha) for name, image, alpha in additions)
            inserted = True
        rebuilt_children.append(payload[start:end])
    if not inserted:
        rebuilt_children.extend(write_native_texture(name, image, alpha) for name, image, alpha in additions)
        rebuilt_children.append(chunk(RW_EXTENSION, b"", root_version))
    result = chunk(RW_TEX_DICTIONARY, b"".join(rebuilt_children), root_version)
    errors = validate_texture_dictionary(result, texture_count + len(additions))
    if errors:
        raise RuntimeError("Augmented retail TXD failed validation: " + "; ".join(errors))
    return result


def extract_native_texture_chunks(source: bytes) -> dict[str, bytes]:
    if len(source) < 24:
        return {}
    root_type, root_size, _ = struct.unpack_from("<III", source, 0)
    if root_type != RW_TEX_DICTIONARY or root_size + 12 > len(source):
        return {}
    payload = source[12:12 + root_size]
    try:
        children = parse_chunks(payload)
    except RuntimeError:
        return {}
    extracted: dict[str, bytes] = {}
    for chunk_type, _, start, end in children:
        if chunk_type != RW_TEXTURE_NATIVE:
            continue
        native = payload[start:end]
        name = native_texture_name(native)
        if name:
            extracted.setdefault(name.lower(), native)
    return extracted


def native_texture_has_alpha(native_chunk: bytes) -> bool:
    native_struct = read_native_texture_struct(native_chunk)
    if native_struct is None:
        return False
    platform = struct.unpack_from("<I", native_struct, 0)[0]
    if platform == 9:
        return bool(native_struct[87] & 1)
    if platform == 8:
        return bool(struct.unpack_from("<I", native_struct, 76)[0])
    return False


def augment_texture_dictionary_with_native_chunks(source: bytes, native_chunks: list[bytes]) -> bytes:
    if len(source) < 24:
        raise RuntimeError("Retail TXD is shorter than a RenderWare dictionary header.")
    root_type, root_size, root_version = struct.unpack_from("<III", source, 0)
    if root_type != RW_TEX_DICTIONARY or root_size + 12 > len(source):
        raise RuntimeError("Retail TXD is not a complete RenderWare texture dictionary.")
    payload = source[12:12 + root_size]
    children = parse_chunks(payload)
    if not children or children[0][0] != RW_STRUCT or children[0][1] < 4:
        raise RuntimeError("Retail TXD has no valid dictionary Struct chunk.")

    existing_names = {
        native_texture_name(payload[start:end]).lower()
        for chunk_type, _, start, end in children
        if chunk_type == RW_TEXTURE_NATIVE
    }
    additions: list[bytes] = []
    for native in native_chunks:
        name = native_texture_name(native).lower()
        if not name or name in existing_names:
            continue
        existing_names.add(name)
        additions.append(native)
    if not additions:
        return source[:12 + root_size]

    struct_type, _, struct_start, struct_end = children[0]
    struct_payload = bytearray(payload[struct_start + 12:struct_end])
    texture_count = struct.unpack_from("<H", struct_payload, 0)[0]
    if texture_count + len(additions) > 65535:
        raise RuntimeError("Augmented TXD exceeds its 65535-texture dictionary limit.")
    struct.pack_into("<H", struct_payload, 0, texture_count + len(additions))
    rebuilt_children = [chunk(struct_type, bytes(struct_payload), root_version)]
    inserted = False
    for chunk_type, _, start, end in children[1:]:
        if chunk_type == RW_EXTENSION and not inserted:
            rebuilt_children.extend(additions)
            inserted = True
        rebuilt_children.append(payload[start:end])
    if not inserted:
        rebuilt_children.extend(additions)
        rebuilt_children.append(chunk(RW_EXTENSION, b"", root_version))
    result = chunk(RW_TEX_DICTIONARY, b"".join(rebuilt_children), root_version)
    errors = validate_texture_dictionary(result, texture_count + len(additions))
    if errors:
        raise RuntimeError("Retail-native TXD augmentation failed validation: " + "; ".join(errors))
    return result


def parse_chunks(payload: bytes) -> list[tuple[int, int, int, int]]:
    chunks: list[tuple[int, int, int, int]] = []
    position = 0
    while position < len(payload):
        if position + 12 > len(payload):
            raise RuntimeError("RenderWare child chunk header is truncated.")
        chunk_type, size, _ = struct.unpack_from("<III", payload, position)
        end = position + 12 + size
        if end > len(payload):
            raise RuntimeError("RenderWare child chunk crosses its parent boundary.")
        chunks.append((chunk_type, size, position, end))
        position = end
    return chunks


def read_native_texture_struct(native_chunk: bytes) -> bytes | None:
    if len(native_chunk) < 24:
        return None
    chunk_type, chunk_size, _ = struct.unpack_from("<III", native_chunk, 0)
    if chunk_type != RW_TEXTURE_NATIVE or chunk_size + 12 > len(native_chunk):
        return None
    struct_type, struct_size, _ = struct.unpack_from("<III", native_chunk, 12)
    if struct_type != RW_STRUCT or struct_size < 88 or 24 + struct_size > 12 + chunk_size:
        return None
    return bytes(native_chunk[24:24 + struct_size])


def native_texture_name(native_chunk: bytes) -> str:
    native_struct = read_native_texture_struct(native_chunk)
    if native_struct is None:
        return ""
    return native_struct[8:40].split(b"\0", 1)[0].decode("ascii", errors="replace")


def validate_native_texture(native_chunk: bytes) -> list[str]:
    native_struct = read_native_texture_struct(native_chunk)
    if native_struct is None:
        return ["native texture has a truncated or invalid Struct chunk"]
    if not native_texture_name(native_chunk):
        return ["native texture has no name"]
    platform = struct.unpack_from("<I", native_struct, 0)[0]
    if platform not in {8, 9}:
        return []
    raster_format = struct.unpack_from("<I", native_struct, 72)[0]
    width, height = struct.unpack_from("<HH", native_struct, 80)
    mip_count = native_struct[85]
    flags = native_struct[87]
    if width == 0 or height == 0 or mip_count == 0:
        return ["native texture has zero dimensions or no mip levels"]
    if mip_count > max(width, height).bit_length():
        return ["native texture has more mip levels than its dimensions allow"]

    block_bytes = 0
    face_count = 1
    if platform == 9:
        if flags & 2:
            face_count = 6
        if flags & 8:
            fourcc = native_struct[76:80]
            if fourcc not in {b"DXT1", b"DXT2", b"DXT3", b"DXT4", b"DXT5"}:
                return ["native texture declares unsupported block compression"]
            block_bytes = 8 if fourcc == b"DXT1" else 16
    elif flags:
        if flags not in {1, 2, 3, 4, 5}:
            return ["native D3D8 texture has an invalid compression type"]
        block_bytes = 8 if flags == 1 else 16

    palette_bytes = 0
    if raster_format & 0x2000:
        palette_bytes = 256 * 4
    elif raster_format & 0x4000:
        palette_bytes = 32 * 4
    position = 88 + palette_bytes
    for face_index in range(face_count):
        mip_width, mip_height = width, height
        for mip_index in range(mip_count):
            if position + 4 > len(native_struct):
                return [f"native texture mip {mip_index} size is truncated"]
            size = struct.unpack_from("<I", native_struct, position)[0]
            position += 4
            if position + size > len(native_struct):
                return [f"native texture mip {mip_index} crosses its Struct boundary"]
            if size == 0:
                return [f"native texture mip {mip_index} has no pixel data"]
            if block_bytes:
                expected = max(1, (mip_width + 3) // 4) * max(1, (mip_height + 3) // 4) * block_bytes
                if size != expected:
                    return [f"native texture mip {mip_index} has {size} compressed bytes, expected {expected}"]
            position += size
            mip_width = max(1, mip_width // 2)
            mip_height = max(1, mip_height // 2)
    return []


def validate_texture_dictionary(data: bytes, expected_textures: int | None = None) -> list[str]:
    errors: list[str] = []
    if len(data) < 28:
        return ["TXD is shorter than its root and dictionary Struct chunks"]
    root_type, root_size, _ = struct.unpack_from("<III", data, 0)
    if root_type != RW_TEX_DICTIONARY:
        return ["TXD root is not a Texture Dictionary chunk"]
    if root_size != len(data) - 12:
        errors.append("TXD root size does not match file length")
        return errors
    try:
        children = parse_chunks(data[12:])
    except RuntimeError as error:
        return [str(error)]
    if not children or children[0][0] != RW_STRUCT:
        return ["TXD does not begin with a dictionary Struct chunk"]
    struct_payload = data[12 + children[0][2] + 12:12 + children[0][3]]
    if len(struct_payload) < 4:
        return ["TXD dictionary Struct chunk is truncated"]
    count = struct.unpack_from("<H", struct_payload, 0)[0]
    native_chunks = [item for item in children if item[0] == RW_TEXTURE_NATIVE]
    if count != len(native_chunks):
        errors.append(f"TXD declares {count} textures but contains {len(native_chunks)} native chunks")
    if expected_textures is not None and count != expected_textures:
        errors.append(f"TXD contains {count} textures, expected {expected_textures}")
    names: set[str] = set()
    for chunk_info in native_chunks:
        _, _, start, end = chunk_info
        native = data[12 + start:12 + end]
        name = native_texture_name(native)
        if name.lower() in names:
            errors.append(f"TXD contains duplicate texture name {name}")
        names.add(name.lower())
        errors.extend(f"TXD texture {name or '<unnamed>'}: {error}" for error in validate_native_texture(native))
    return errors


def decode_texture_dictionary_texture(data: bytes, texture_name: str | None = None) -> TextureImage:
    if len(data) < 12:
        raise ValueError("Retail radar TXD is shorter than its root header.")
    chunk_type, size, _ = struct.unpack_from("<III", data, 0)
    if chunk_type != RW_TEX_DICTIONARY or 12 + size > len(data):
        raise ValueError("Retail radar data is not a complete texture dictionary.")
    dictionary = bytes(data[:12 + size])
    errors = validate_texture_dictionary(dictionary)
    if errors:
        raise ValueError("Retail radar TXD failed validation: " + "; ".join(errors))
    textures = extract_native_texture_chunks(dictionary)
    if not textures:
        raise ValueError("Retail radar TXD contains no native texture.")
    if texture_name is not None and texture_name.lower() in textures:
        return decode_native_texture(textures[texture_name.lower()])
    if len(textures) == 1:
        return decode_native_texture(next(iter(textures.values())))
    if texture_name is None:
        raise ValueError("A texture name is required for a retail TXD containing multiple native textures.")
    raise ValueError(f"Retail radar TXD does not contain texture {texture_name}.")


def decode_native_texture(native_chunk: bytes) -> TextureImage:
    errors = validate_native_texture(native_chunk)
    if errors:
        raise ValueError("Native radar texture failed validation: " + "; ".join(errors))
    native = read_native_texture_struct(native_chunk)
    if native is None:
        raise ValueError("Native radar texture has no raster Struct.")
    platform = struct.unpack_from("<I", native, 0)[0]
    if platform not in {8, 9}:
        raise ValueError(f"Radar decoding supports PC D3D8/D3D9 rasters, not platform {platform}.")
    raster_format = struct.unpack_from("<I", native, 72)[0]
    native_format = struct.unpack_from("<I", native, 76)[0]
    width, height = struct.unpack_from("<HH", native, 80)
    depth, _, raster_type, flags = native[84:88]
    if width * height > 16_777_216:
        raise ValueError("Native radar texture exceeds the safe 16-million-texel decode limit.")
    if platform == 9 and flags & 2:
        raise ValueError("A cube-map raster cannot be used as a two-dimensional radar tile.")
    if (raster_type & 7) not in {1, 4}:
        raise ValueError("Native radar texture does not contain a texture raster.")

    palette_count = 256 if raster_format & 0x2000 else 32 if raster_format & 0x4000 else 0
    palette_end = 88 + palette_count * 4
    position = palette_end
    if position + 4 > len(native):
        raise ValueError("Native radar texture has no first mip-size field.")
    size = struct.unpack_from("<I", native, position)[0]
    pixels = native[position + 4:position + 4 + size]
    if len(pixels) != size:
        raise ValueError("Native radar texture first mip data is truncated.")
    compression = 0
    if platform == 9 and flags & 8:
        compression = {b"DXT1": 1, b"DXT2": 2, b"DXT3": 3, b"DXT4": 4, b"DXT5": 5}.get(native[76:80], 0)
    elif platform == 8:
        compression = flags
    if compression:
        return decode_dxt_texture(pixels, width, height, compression)

    has_alpha = native_texture_has_alpha(native_chunk)
    image = TextureImage.solid(width, height, (0, 0, 0, 255))
    if palette_count:
        if len(pixels) % height or len(pixels) // height < width:
            raise ValueError("PC paletted radar mip does not contain a complete byte-indexed row per image row.")
        stride = len(pixels) // height
        palette = native[88:palette_end]
        valid_entries = 16 if raster_format & 0x4000 else 256
        for y in range(height):
            for x in range(width):
                palette_index = pixels[y * stride + x]
                if palette_index >= valid_entries:
                    raise ValueError("PC paletted radar mip contains an out-of-range palette index.")
                offset = palette_index * 4
                red, green, blue, alpha = palette[offset:offset + 4]
                image.set_pixel(x, y, (red, green, blue, alpha if has_alpha else 255))
        return image

    if platform == 8:
        native_format = {
            0x0100: 25, 0x0200: 23, 0x0300: 26,
            0x0400: 50, 0x0500: 21, 0x0600: 22 if depth == 32 or len(pixels) == width * height * 4 else 20, 0x0A00: 24,
        }.get(raster_format & 0x0F00, 0)
    bytes_per_pixel = {20: 3, 21: 4, 22: 4, 23: 2, 24: 2, 25: 2, 26: 2, 28: 1, 32: 4, 33: 4, 50: 1, 51: 2}.get(native_format)
    if bytes_per_pixel is None:
        raise ValueError(f"Unsupported uncompressed PC radar raster format {native_format} with depth {depth}.")
    if len(pixels) % height or len(pixels) // height < width * bytes_per_pixel:
        raise ValueError("Uncompressed radar mip does not contain complete pixel rows.")
    stride = len(pixels) // height
    for y in range(height):
        for x in range(width):
            offset = y * stride + x * bytes_per_pixel
            alpha = 255
            if native_format in {20, 21, 22}:
                blue, green, red = pixels[offset:offset + 3]
                if native_format == 21 and has_alpha:
                    alpha = pixels[offset + 3]
            elif native_format in {32, 33}:
                red, green, blue = pixels[offset:offset + 3]
                if native_format == 32 and has_alpha:
                    alpha = pixels[offset + 3]
            elif native_format in {23, 24, 25, 26}:
                value = struct.unpack_from("<H", pixels, offset)[0]
                if native_format == 23:
                    red, green, blue = rgb_from_565(value)
                elif native_format in {24, 25}:
                    red, green, blue = tuple(round(((value >> shift) & 31) * 255 / 31) for shift in (10, 5, 0))
                    if native_format == 25 and has_alpha:
                        alpha = 255 if value & 0x8000 else 0
                else:
                    red, green, blue = tuple(((value >> shift) & 15) * 17 for shift in (8, 4, 0))
                    if has_alpha:
                        alpha = ((value >> 12) & 15) * 17
            elif native_format == 28:
                red = green = blue = 255
                alpha = pixels[offset]
            else:
                red = green = blue = pixels[offset]
                if native_format == 51 and has_alpha:
                    alpha = pixels[offset + 1]
            image.set_pixel(x, y, (red, green, blue, alpha))
    return image


def decode_dxt_texture(data: bytes, width: int, height: int, compression: int) -> TextureImage:
    if width <= 0 or height <= 0 or compression not in {1, 2, 3, 4, 5}:
        raise ValueError("DXT radar decoding requires positive dimensions and a DXT1–DXT5 compression type.")
    if width * height > 16_777_216:
        raise ValueError("DXT radar texture exceeds the safe 16-million-texel decode limit.")
    block_size = 8 if compression == 1 else 16
    expected = ((width + 3) // 4) * ((height + 3) // 4) * block_size
    if len(data) != expected:
        raise ValueError(f"DXT radar mip has {len(data)} bytes, expected {expected}.")
    image = TextureImage.solid(width, height, (0, 0, 0, 0))
    position = 0
    for block_y in range(0, height, 4):
        for block_x in range(0, width, 4):
            block = data[position:position + block_size]
            position += block_size
            color_offset = 0 if compression == 1 else 8
            color_0, color_1, indices = struct.unpack_from("<HHI", block, color_offset)
            first, second = rgb_from_565(color_0), rgb_from_565(color_1)
            colors = [first + (255,), second + (255,)]
            if color_0 > color_1 or compression != 1:
                colors.extend((
                    tuple((2 * first[channel] + second[channel]) // 3 for channel in range(3)) + (255,),
                    tuple((first[channel] + 2 * second[channel]) // 3 for channel in range(3)) + (255,),
                ))
            else:
                colors.extend((tuple((first[channel] + second[channel]) // 2 for channel in range(3)) + (255,), (0, 0, 0, 0)))
            alphas: list[int] | None = None
            if compression in {2, 3}:
                alpha_bits = int.from_bytes(block[:8], "little")
                alphas = [((alpha_bits >> (index * 4)) & 15) * 17 for index in range(16)]
            elif compression in {4, 5}:
                alpha_0, alpha_1 = block[:2]
                palette = [alpha_0, alpha_1]
                if alpha_0 > alpha_1:
                    palette.extend(((7 - index) * alpha_0 + index * alpha_1 + 3) // 7 for index in range(1, 7))
                else:
                    palette.extend(((5 - index) * alpha_0 + index * alpha_1 + 2) // 5 for index in range(1, 5))
                    palette.extend((0, 255))
                alpha_bits = int.from_bytes(block[2:8], "little")
                alphas = [palette[(alpha_bits >> (index * 3)) & 7] for index in range(16)]
            for index in range(16):
                x, y = block_x + index % 4, block_y + index // 4
                if x >= width or y >= height:
                    continue
                color = colors[(indices >> (index * 2)) & 3]
                if alphas is not None:
                    color = color[:3] + (alphas[index],)
                if compression in {2, 4} and color[3]:
                    color = tuple(min(255, round(component * 255 / color[3])) for component in color[:3]) + (color[3],)
                image.set_pixel(x, y, color)
    return image


def write_native_texture(name: str, image: TextureImage, has_alpha: bool, address_mode: str = "wrap") -> bytes:
    if image.width <= 0 or image.height <= 0 or image.width & (image.width - 1) or image.height & (image.height - 1):
        raise ValueError("TXD texture dimensions must be positive powers of two.")
    if image.width > 65535 or image.height > 65535:
        raise ValueError("TXD texture dimensions exceed the native raster header.")
    if len(image.pixels) != image.width * image.height * 4:
        raise ValueError("TXD texture pixels must contain exactly one RGBA value per texel.")
    try:
        encoded_name = name.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("TXD texture names must contain ASCII characters.") from error
    if not encoded_name or len(encoded_name) > 31 or b"\0" in encoded_name:
        raise ValueError("TXD texture names must contain 1 to 31 non-NUL ASCII characters.")
    address_modes = {"wrap": 1, "clamp": 3}
    if address_mode not in address_modes:
        raise ValueError("TXD texture address mode must be wrap or clamp.")
    addressing = address_modes[address_mode]
    encoded_mips: list[bytes] = []
    mip = image
    while True:
        encoded = encode_dxt3(mip) if has_alpha else encode_dxt1(mip)
        encoded_mips.append(struct.pack("<I", len(encoded)) + encoded)
        if mip.width == 1 and mip.height == 1:
            break
        mip = downsample(mip)

    raster_format = 0x8300 if has_alpha else 0x8200
    compression_flags = 9 if has_alpha else 8
    native_struct = bytearray()
    native_struct += struct.pack("<II", 9, 6 | (addressing << 8) | (addressing << 12))
    native_struct += encoded_name.ljust(32, b"\0")
    native_struct += b"\0" * 32
    native_struct += struct.pack(
        "<I4sHHBBBB",
        raster_format,
        b"DXT3" if has_alpha else b"DXT1",
        image.width,
        image.height,
        16,
        len(encoded_mips),
        4,
        compression_flags,
    )
    native_struct += b"".join(encoded_mips)
    return chunk(
        RW_TEXTURE_NATIVE,
        chunk(RW_STRUCT, bytes(native_struct)) + chunk(RW_EXTENSION, b""),
    )


def downsample(image: TextureImage) -> TextureImage:
    width = max(1, image.width // 2)
    height = max(1, image.height // 2)
    result = TextureImage.solid(width, height, (0, 0, 0, 0))
    for y in range(height):
        for x in range(width):
            samples = [
                image.get_pixel(x * 2 + offset_x, y * 2 + offset_y)
                for offset_y in range(2)
                for offset_x in range(2)
            ]
            alpha_sum = sum(sample[3] for sample in samples)
            if alpha_sum:
                color = tuple(
                    round(sum(sample[channel] * sample[3] for sample in samples) / alpha_sum)
                    for channel in range(3)
                )
            else:
                color = tuple(round(sum(sample[channel] for sample in samples) / 4) for channel in range(3))
            result.set_pixel(x, y, color + (round(alpha_sum / 4),))
    return result


def encode_dxt1(image: TextureImage) -> bytes:
    output = bytearray()
    for block_y in range(0, image.height, 4):
        for block_x in range(0, image.width, 4):
            pixels = [image.get_pixel(block_x + x, block_y + y) for y in range(4) for x in range(4)]
            output += encode_color_block(pixels)
    return bytes(output)


def encode_dxt3(image: TextureImage) -> bytes:
    output = bytearray()
    for block_y in range(0, image.height, 4):
        for block_x in range(0, image.width, 4):
            pixels = [image.get_pixel(block_x + x, block_y + y) for y in range(4) for x in range(4)]
            alpha_bits = 0
            for index, pixel in enumerate(pixels):
                alpha_bits |= round(pixel[3] / 17) << (index * 4)
            output += struct.pack("<Q", alpha_bits)
            output += encode_color_block(pixels, alpha_weighted=True)
    return bytes(output)


def encode_color_block(pixels: list[tuple[int, int, int, int]], alpha_weighted: bool = False) -> bytes:
    endpoint_pixels = [pixel for pixel in pixels if pixel[3] >= 8] if alpha_weighted else pixels
    if not endpoint_pixels:
        endpoint_pixels = pixels
    darkest = min(endpoint_pixels, key=lambda color: color[0] * 3 + color[1] * 6 + color[2])
    lightest = max(endpoint_pixels, key=lambda color: color[0] * 3 + color[1] * 6 + color[2])
    color_0 = rgb_to_565(lightest)
    color_1 = rgb_to_565(darkest)
    if color_0 <= color_1:
        if color_1 < 0xFFFF:
            color_0 = color_1 + 1
        else:
            color_1 = max(0, color_0 - 1)
    first = rgb_from_565(color_0)
    second = rgb_from_565(color_1)
    palette = (
        first,
        second,
        tuple((2 * first[channel] + second[channel]) // 3 for channel in range(3)),
        tuple((first[channel] + 2 * second[channel]) // 3 for channel in range(3)),
    )
    indices = 0
    for pixel_index, pixel in enumerate(pixels):
        best_index = min(
            range(4),
            key=lambda index: sum((pixel[channel] - palette[index][channel]) ** 2 for channel in range(3)),
        )
        indices |= best_index << (pixel_index * 2)
    return struct.pack("<HHI", color_0, color_1, indices)


def rgb_to_565(color: tuple[int, int, int, int]) -> int:
    return (round(color[0] * 31 / 255) << 11) | (round(color[1] * 63 / 255) << 5) | round(color[2] * 31 / 255)


def rgb_from_565(color: int) -> tuple[int, int, int]:
    return (
        round(((color >> 11) & 31) * 255 / 31),
        round(((color >> 5) & 63) * 255 / 63),
        round((color & 31) * 255 / 31),
    )


def chunk(chunk_type: int, payload: bytes, version: int = RW_LIBRARY_ID) -> bytes:
    return struct.pack("<III", chunk_type, len(payload), version) + payload
