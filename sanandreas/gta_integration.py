from __future__ import annotations

import json
import math
import os
import random
import shutil
import struct
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_GAME_DIRECTORY = Path(r"C:\Program Files (x86)\Rockstar Games\Grand Theft Auto San Andreas")
MOD_NAME = "SeedAndreas"
LEGACY_MOD_PREFIXES = ("sanandrais", "seedandreas")
INSTALL_MARKER_NAMES = ("sanandrais_install.json", "seed_andreas_install.json")


@dataclass(frozen=True)
class IdeModel:
    model_id: int
    model_name: str
    txd_name: str
    section: str
    source_file: str
    flags: int


@dataclass(frozen=True)
class GtaPlacement:
    model_id: int
    model_name: str
    interior: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    lod_index: int
    source_file: str


@dataclass(frozen=True)
class TrafficPathSegment:
    start_x: float
    start_y: float
    start_z: float
    end_x: float
    end_y: float
    end_z: float
    width: float


@dataclass(frozen=True)
class SeededPathOverrides:
    files: dict[str, bytes]
    vehicle_segments: list[TrafficPathSegment]
    changed_nodes: int
    preserved_nodes: int


@dataclass
class GameScan:
    game_directory: str
    models: dict[int, IdeModel]
    placements: list[GtaPlacement]
    ide_files_scanned: int
    ipl_files_scanned: int
    binary_ipl_files_skipped: int
    vehicle_path_segments: list[TrafficPathSegment]


@dataclass
class InstalledPrototype:
    seed: int
    game_directory: str
    mod_directory: str
    source_tile: tuple[int, int]
    source_center: tuple[float, float]
    target_center: tuple[float, float, float]
    rotation_degrees: int
    placements_written: int
    unique_models: int
    source_files: list[str]
    validation: list[str]


def validate_game_directory(game_directory: Path, require_mod_loader: bool = True) -> list[str]:
    required = (
        game_directory / "gta_sa.exe",
        game_directory / "data" / "gta.dat",
        game_directory / "models" / "gta3.img",
    )
    problems = [f"Missing required GTA SA file: {path}" for path in required if not path.is_file()]
    if require_mod_loader:
        if not (game_directory / "modloader.asi").is_file():
            problems.append("modloader.asi was not found in the GTA SA directory.")
        if not (game_directory / "modloader").is_dir():
            problems.append("The modloader directory was not found in the GTA SA directory.")
        elif not (game_directory / "modloader" / ".data").is_dir():
            problems.append(
                "modloader\\.data was not found. This directory contains Mod Loader's file-handler plugins; "
                "restore it before installing Seed Andreas."
            )
    return problems


def build_augmented_gta_dat(
    source_text: str,
    colfile_line: str | None = None,
    ipl_line: str = "IPL DATA\\MAPS\\SEEDANDREAS\\SEEDANDREAS_LS.IPL",
) -> str:
    lines = [
        line
        for line in source_text.splitlines()
        if "SANANDRAIS" not in line.upper() and "SEEDANDREAS" not in line.upper()
    ]
    if not lines:
        raise RuntimeError("The retail data\\gta.dat file is empty.")

    ide_indices = [
        index for index, line in enumerate(lines)
        if line.strip().upper().startswith("IDE ")
    ]
    if not ide_indices:
        raise RuntimeError("The retail data\\gta.dat file contains no IDE entries.")

    if colfile_line is not None:
        col_indices = [
            index for index, line in enumerate(lines)
            if line.strip().upper().startswith("COLFILE ")
        ]
        col_insert_index = col_indices[-1] + 1 if col_indices else ide_indices[-1] + 1
        lines.insert(col_insert_index, colfile_line)

    ipl_indices = [
        index for index, line in enumerate(lines)
        if line.strip().upper().startswith("IPL ")
    ]
    if ipl_indices:
        ipl_insert_index = ipl_indices[-1] + 1
    else:
        exit_indices = [
            index for index, line in enumerate(lines)
            if line.strip().upper() == "EXIT"
        ]
        ipl_insert_index = exit_indices[0] if exit_indices else len(lines)
    lines.insert(ipl_insert_index, ipl_line)

    if colfile_line is not None and lines.index(colfile_line) >= lines.index(ipl_line):
        raise RuntimeError("Could not preserve the required COLFILE-before-IPL order in data\\gta.dat.")
    return "\n".join(lines).rstrip() + "\n"


def scan_game_map(game_directory: Path) -> GameScan:
    problems = validate_game_directory(game_directory, require_mod_loader=False)
    if problems:
        raise RuntimeError("\n".join(problems))

    data_directory = game_directory / "data"
    ide_paths = find_files_case_insensitive(data_directory, ".ide")
    mod_loader_directory = game_directory / "modloader"
    if mod_loader_directory.is_dir():
        excluded_directories = {
            path.name
            for path in mod_loader_directory.iterdir()
            if path.name == ".data" or is_seed_andreas_mod_directory(path)
        }
        ide_paths.extend(
            path
            for path in find_files_case_insensitive(mod_loader_directory, ".ide")
            if path.relative_to(mod_loader_directory).parts[0] not in excluded_directories
        )
    ide_paths = sorted(set(ide_paths), key=lambda path: str(path).lower())
    ipl_paths = sorted(find_files_case_insensitive(data_directory, ".ipl"), key=lambda path: str(path).lower())
    models: dict[int, IdeModel] = {}
    placements: list[GtaPlacement] = []
    binary_skipped = 0

    for ide_path in ide_paths:
        for model in parse_ide(ide_path, game_directory):
            models.setdefault(model.model_id, model)

    for ipl_path in ipl_paths:
        parsed, was_binary = parse_ipl(ipl_path, game_directory)
        if was_binary:
            binary_skipped += 1
        else:
            placements.extend(parsed)

    if not models:
        raise RuntimeError("No IDE object definitions could be read from the GTA SA data directory.")
    if not placements:
        raise RuntimeError("No text IPL placements could be read from the GTA SA data directory.")

    vehicle_path_segments = read_vehicle_path_segments(game_directory / "data" / "paths")

    return GameScan(
        game_directory=str(game_directory),
        models=models,
        placements=placements,
        ide_files_scanned=len(ide_paths),
        ipl_files_scanned=len(ipl_paths),
        binary_ipl_files_skipped=binary_skipped,
        vehicle_path_segments=vehicle_path_segments,
    )


@dataclass(frozen=True)
class PathNodeRecord:
    area_id: int
    node_id: int
    x: float
    y: float
    z: float
    path_width: float
    base_link_id: int
    link_count: int
    water_node: bool = False


def read_vehicle_path_segments(paths_directory: Path) -> list[TrafficPathSegment]:
    nodes: dict[tuple[int, int], PathNodeRecord] = {}
    links_by_area: dict[int, list[tuple[int, int]]] = {}
    sources = read_game_path_files(paths_directory.parent.parent)
    for area_id in range(64):
        if area_id not in sources:
            path = find_path_node_file(paths_directory, area_id)
            if path is not None:
                sources[area_id] = path.read_bytes()

    for area_id in range(64):
        data = sources.get(area_id)
        if data is None:
            continue
        path = paths_directory / f"nodes{area_id}.dat"
        if len(data) < 20:
            raise RuntimeError(f"Vehicle path file is truncated: {path}")
        node_count, vehicle_count, pedestrian_count, car_link_count, address_count = struct.unpack_from(
            "<5I", data, 0
        )
        if node_count != vehicle_count + pedestrian_count or node_count > 100_000:
            raise RuntimeError(f"Vehicle path file has invalid node counts: {path}")
        node_offset = 20
        node_end = node_offset + node_count * 28
        car_link_end = node_end + car_link_count * 14
        stored_address_count = address_count + 16 * 12 if address_count else 0
        address_end = car_link_end + stored_address_count * 4
        if address_end > len(data):
            raise RuntimeError(f"Vehicle path file has truncated node/link arrays: {path}")

        for node_id in range(vehicle_count):
            offset = node_offset + node_id * 28
            compressed_x, compressed_y, compressed_z = struct.unpack_from("<3h", data, offset + 8)
            base_link_id = struct.unpack_from("<h", data, offset + 16)[0]
            path_width = data[offset + 22] / 16.0
            link_count = data[offset + 24] & 0x0F
            nodes[(area_id, node_id)] = PathNodeRecord(
                area_id,
                node_id,
                compressed_x / 8.0,
                compressed_y / 8.0,
                compressed_z / 8.0,
                path_width,
                base_link_id,
                link_count,
                bool(data[offset + 24] & 0x80),
            )

        addresses = [
            struct.unpack_from("<HH", data, car_link_end + index * 4)
            for index in range(stored_address_count)
        ]
        links_by_area[area_id] = addresses

    segments: list[TrafficPathSegment] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for key, node in nodes.items():
        if node.water_node:
            continue
        addresses = links_by_area.get(node.area_id, [])
        if node.base_link_id < 0 or node.base_link_id + node.link_count > len(addresses):
            continue
        for link_index in range(node.base_link_id, node.base_link_id + node.link_count):
            target_key = addresses[link_index]
            target = nodes.get(target_key)
            if target is None or target.water_node:
                continue
            identity = tuple(sorted((key, target_key)))
            if identity in seen:
                continue
            seen.add(identity)
            segments.append(
                TrafficPathSegment(
                    node.x,
                    node.y,
                    node.z,
                    target.x,
                    target.y,
                    target.z,
                    max(16.0, (node.path_width + target.path_width) * 2.0 + 8.0),
                )
            )
    return segments


def read_game_path_files(game_directory: Path) -> dict[int, bytes]:
    sources: dict[int, bytes] = {}
    img_path = game_directory / "models" / "gta3.img"
    if img_path.is_file():
        with img_path.open("rb") as archive:
            header = archive.read(8)
            if len(header) != 8 or header[:4] != b"VER2":
                raise RuntimeError("GTA path loading requires a VER2 gta3.img archive.")
            count = struct.unpack_from("<I", header, 4)[0]
            archive_size = img_path.stat().st_size
            if count > 100000 or 8 + count * 32 > archive_size:
                raise RuntimeError("GTA IMG path directory is truncated or invalid.")
            directory = archive.read(count * 32)
            for index in range(count):
                offset, streaming, allocated, raw_name = struct.unpack_from("<IHH24s", directory, index * 32)
                name = raw_name.split(b"\0", 1)[0].decode("ascii", errors="replace").lower()
                if not name.startswith("nodes") or not name.endswith(".dat"):
                    continue
                try:
                    area_id = int(name[5:-4])
                except ValueError:
                    continue
                if not 0 <= area_id < 64:
                    continue
                length = (streaming or allocated) * 2048
                if offset * 2048 + length > archive_size or length < 20:
                    raise RuntimeError(f"IMG path entry {name} is truncated.")
                archive.seek(offset * 2048)
                sources[area_id] = archive.read(length)
    paths_directory = game_directory / "data" / "paths"
    for area_id in range(64):
        if area_id in sources:
            continue
        path = find_path_node_file(paths_directory, area_id)
        if path is not None:
            sources[area_id] = path.read_bytes()
    return sources


def build_seeded_path_overrides(
    paths_directory: Path,
    seed: int,
    bounds: tuple[float, float, float, float],
    protected_bounds: tuple[tuple[float, float, float, float], ...],
    maximum_displacement: float = 0.0,
) -> SeededPathOverrides:
    if maximum_displacement < 0.0 or maximum_displacement > 12.0:
        raise ValueError("Seeded path displacement must be between 0 and 12 world units.")

    overrides: dict[str, bytes] = {}
    changed_nodes = 0
    preserved_nodes = 0
    minimum_x, minimum_y, maximum_x, maximum_y = bounds

    for area_id in range(64):
        path = find_path_node_file(paths_directory, area_id)
        if path is None:
            continue
        original = path.read_bytes()
        if len(original) < 20:
            raise RuntimeError(f"Path file is truncated: {path}")
        node_count, vehicle_count, pedestrian_count, car_link_count, address_count = struct.unpack_from(
            "<5I", original, 0
        )
        if node_count != vehicle_count + pedestrian_count or node_count > 100_000:
            raise RuntimeError(f"Path file has invalid node counts: {path}")
        node_end = 20 + node_count * 28
        if node_end > len(original):
            raise RuntimeError(f"Path file has a truncated node table: {path}")

        modified = bytearray(original)
        file_changed = False
        for node_id in range(node_count):
            offset = 20 + node_id * 28
            compressed_x, compressed_y, compressed_z = struct.unpack_from("<3h", original, offset + 8)
            x = compressed_x / 8.0
            y = compressed_y / 8.0
            z = compressed_z / 8.0
            if not (minimum_x <= x <= maximum_x and minimum_y <= y <= maximum_y):
                continue
            if point_in_any_bounds(x, y, protected_bounds):
                preserved_nodes += 1
                continue

            seeded_x, seeded_y = seeded_path_xy(x, y, seed)
            delta_x = max(-maximum_displacement, min(maximum_displacement, seeded_x - x))
            delta_y = max(-maximum_displacement, min(maximum_displacement, seeded_y - y))
            target_x = round((x + delta_x) * 8.0)
            target_y = round((y + delta_y) * 8.0)
            target_z = round(z * 8.0)
            if not all(-32768 <= value <= 32767 for value in (target_x, target_y, target_z)):
                preserved_nodes += 1
                continue
            if target_x == compressed_x and target_y == compressed_y:
                preserved_nodes += 1
                continue
            struct.pack_into("<3h", modified, offset + 8, target_x, target_y, target_z)
            changed_nodes += 1
            file_changed = True

        if file_changed:
            modified_bytes = bytes(modified)
            errors = validate_seeded_path_override(original, modified_bytes)
            if errors:
                raise RuntimeError(f"Seeded path override {path.name} failed validation: " + "; ".join(errors))
            overrides[path.name] = modified_bytes

    transformed_segments: list[TrafficPathSegment] = []
    for segment in read_vehicle_path_segments(paths_directory):
        start_x, start_y = transform_path_point(
            segment.start_x,
            segment.start_y,
            seed,
            bounds,
            protected_bounds,
            maximum_displacement,
        )
        end_x, end_y = transform_path_point(
            segment.end_x,
            segment.end_y,
            seed,
            bounds,
            protected_bounds,
            maximum_displacement,
        )
        transformed_segments.append(
            TrafficPathSegment(
                start_x,
                start_y,
                segment.start_z,
                end_x,
                end_y,
                segment.end_z,
                segment.width,
            )
        )
    return SeededPathOverrides(overrides, transformed_segments, changed_nodes, preserved_nodes)


def find_path_node_file(paths_directory: Path, area_id: int) -> Path | None:
    expected = f"nodes{area_id}.dat"
    direct = paths_directory / expected
    if direct.is_file():
        return direct
    if not paths_directory.is_dir():
        return None
    return next(
        (
            candidate
            for candidate in paths_directory.iterdir()
            if candidate.is_file() and candidate.name.lower() == expected
        ),
        None,
    )


def validate_seeded_path_override(original: bytes, modified: bytes) -> list[str]:
    errors: list[str] = []
    if len(original) != len(modified):
        return ["seeded path override changed the file length"]
    if len(original) < 20 or original[:20] != modified[:20]:
        return ["seeded path override changed or truncated the NODES header"]
    node_count = struct.unpack_from("<I", original, 0)[0]
    node_end = 20 + node_count * 28
    if node_end > len(original):
        return ["retail NODES file has a truncated node table"]
    original_masked = bytearray(original)
    modified_masked = bytearray(modified)
    for node_id in range(node_count):
        coordinate_offset = 20 + node_id * 28 + 8
        original_masked[coordinate_offset:coordinate_offset + 6] = b"\0" * 6
        modified_masked[coordinate_offset:coordinate_offset + 6] = b"\0" * 6
    if original_masked != modified_masked:
        errors.append("seeded path override changed bytes outside packed node XYZ coordinates")
    return errors


def transform_path_point(
    x: float,
    y: float,
    seed: int,
    bounds: tuple[float, float, float, float],
    protected_bounds: tuple[tuple[float, float, float, float], ...],
    maximum_displacement: float,
) -> tuple[float, float]:
    minimum_x, minimum_y, maximum_x, maximum_y = bounds
    if not (minimum_x <= x <= maximum_x and minimum_y <= y <= maximum_y):
        return x, y
    if point_in_any_bounds(x, y, protected_bounds):
        return x, y
    seeded_x, seeded_y = seeded_path_xy(x, y, seed)
    return (
        round((x + max(-maximum_displacement, min(maximum_displacement, seeded_x - x))) * 8.0) / 8.0,
        round((y + max(-maximum_displacement, min(maximum_displacement, seeded_y - y))) * 8.0) / 8.0,
    )


def seeded_path_xy(x: float, y: float, seed: int) -> tuple[float, float]:
    frequency = 1.0 / 420.0
    noise_x = interpolated_value_noise(x * frequency, y * frequency, seed ^ 0x51F15EED)
    noise_y = interpolated_value_noise(x * frequency, y * frequency, seed ^ 0x2D68A5B9)
    return x + noise_x * 8.0, y + noise_y * 8.0


def interpolated_value_noise(x: float, y: float, seed: int) -> float:
    x0 = math.floor(x)
    y0 = math.floor(y)
    tx = smoothstep(x - x0)
    ty = smoothstep(y - y0)
    bottom = linear_interpolate(hash_noise(x0, y0, seed), hash_noise(x0 + 1, y0, seed), tx)
    top = linear_interpolate(hash_noise(x0, y0 + 1, seed), hash_noise(x0 + 1, y0 + 1, seed), tx)
    return linear_interpolate(bottom, top, ty)


def hash_noise(x: int, y: int, seed: int) -> float:
    value = (x * 0x1F123BB5) ^ (y * 0x5F356495) ^ seed
    value = (value ^ (value >> 16)) * 0x45D9F3B
    value = (value ^ (value >> 16)) * 0x45D9F3B
    value ^= value >> 16
    return ((value & 0xFFFFFFFF) / 0x7FFFFFFF) - 1.0


def smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def linear_interpolate(first: float, second: float, factor: float) -> float:
    return first + (second - first) * factor


def point_in_any_bounds(
    x: float,
    y: float,
    bounds_collection: tuple[tuple[float, float, float, float], ...],
) -> bool:
    return any(
        minimum_x <= x <= maximum_x and minimum_y <= y <= maximum_y
        for minimum_x, minimum_y, maximum_x, maximum_y in bounds_collection
    )


def find_files_case_insensitive(root: Path, suffix: str) -> list[Path]:
    suffix = suffix.lower()
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == suffix]


def parse_ide(path: Path, game_directory: Path) -> list[IdeModel]:
    try:
        text = path.read_text(encoding="cp1252", errors="replace")
    except OSError:
        return []

    section = ""
    supported_sections = {"objs", "tobj", "anim"}
    models: list[IdeModel] = []
    for raw_line in text.splitlines():
        line = clean_data_line(raw_line)
        if not line:
            continue
        lowered = line.lower()
        if lowered == "end":
            section = ""
            continue
        if lowered in {"objs", "tobj", "anim", "cars", "peds", "hier", "path", "2dfx", "txdp"}:
            section = lowered
            continue
        if section not in supported_sections:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        try:
            model_id = int(fields[0], 10)
        except ValueError:
            continue
        model_name = fields[1]
        txd_name = fields[2]
        if not model_name:
            continue
        try:
            flags = int(fields[-1], 10)
        except ValueError:
            flags = 0
        models.append(IdeModel(model_id, model_name, txd_name, section, relative_source(path, game_directory), flags))
    return models


def parse_ipl(path: Path, game_directory: Path) -> tuple[list[GtaPlacement], bool]:
    try:
        raw = path.read_bytes()
    except OSError:
        return [], False
    if raw[:4].lower() == b"bnry" or b"\x00" in raw[:256]:
        return [], True

    text = raw.decode("cp1252", errors="replace")
    section = ""
    placements: list[GtaPlacement] = []
    source_file = relative_source(path, game_directory)
    for raw_line in text.splitlines():
        line = clean_data_line(raw_line)
        if not line:
            continue
        lowered = line.lower()
        if lowered == "end":
            section = ""
            continue
        if lowered in {"inst", "cull", "pick", "path", "grge", "enex", "cars", "jump", "tcyc", "auzo", "mult", "occl", "zone"}:
            section = lowered
            continue
        if section != "inst":
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 11:
            continue
        try:
            placement = GtaPlacement(
                model_id=int(fields[0], 10),
                model_name=fields[1],
                interior=int(fields[2], 10),
                x=float(fields[3]),
                y=float(fields[4]),
                z=float(fields[5]),
                qx=float(fields[6]),
                qy=float(fields[7]),
                qz=float(fields[8]),
                qw=float(fields[9]),
                lod_index=int(fields[10], 10),
                source_file=source_file,
            )
        except ValueError:
            continue
        placements.append(placement)
    return placements, False


def clean_data_line(raw_line: str) -> str:
    line = raw_line.strip()
    if not line or line.startswith("#") or line.startswith(";"):
        return ""
    for marker in ("#", ";"):
        marker_index = line.find(marker)
        if marker_index >= 0:
            line = line[:marker_index].rstrip()
    return line


def relative_source(path: Path, game_directory: Path) -> str:
    try:
        return str(path.relative_to(game_directory)).replace("\\", "/")
    except ValueError:
        return str(path)


def generate_stock_map_prototype(
    scan: GameScan,
    seed: int,
    target_x: float = 2600.0,
    target_y: float = -2600.0,
    target_ground_z: float = 28.0,
    tile_size: float = 420.0,
    maximum_placements: int = 550,
) -> tuple[list[GtaPlacement], InstalledPrototype]:
    random_source = random.Random(seed)
    usable = [
        placement
        for placement in scan.placements
        if placement.interior == 0
        and placement.model_id in scan.models
        and not placement.model_name.lower().startswith("lod")
        and -3000.0 <= placement.x <= 3000.0
        and -3000.0 <= placement.y <= 3000.0
    ]
    tiles: dict[tuple[int, int], list[GtaPlacement]] = {}
    for placement in usable:
        tile = (math.floor(placement.x / tile_size), math.floor(placement.y / tile_size))
        tiles.setdefault(tile, []).append(placement)

    candidates = [(tile, values) for tile, values in tiles.items() if 55 <= len(values) <= 1400]
    if not candidates:
        candidates = [(tile, values) for tile, values in tiles.items() if len(values) >= 25]
    if not candidates:
        raise RuntimeError("The retail IPL files did not contain a sufficiently populated source tile.")

    candidates.sort(key=lambda item: item[0])
    source_tile, source_placements = random_source.choice(candidates)
    source_center_x = (source_tile[0] + 0.5) * tile_size
    source_center_y = (source_tile[1] + 0.5) * tile_size
    source_placements = sorted(
        source_placements,
        key=lambda placement: (
            (placement.x - source_center_x) ** 2 + (placement.y - source_center_y) ** 2,
            placement.source_file.lower(),
            placement.model_id,
            placement.x,
            placement.y,
            placement.z,
        ),
    )[:maximum_placements]

    surface_candidates = [placement.z for placement in source_placements if -100.0 <= placement.z <= 200.0]
    surface_candidates.sort()
    source_ground_z = surface_candidates[len(surface_candidates) // 2] if surface_candidates else 0.0
    rotation_degrees = random_source.choice((0, 90, 180, 270))
    rotation_radians = math.radians(rotation_degrees)
    rotation_sine = math.sin(rotation_radians)
    rotation_cosine = math.cos(rotation_radians)
    rotation_qz = math.sin(rotation_radians * 0.5)
    rotation_qw = math.cos(rotation_radians * 0.5)

    generated: list[GtaPlacement] = []
    for placement in source_placements:
        local_x = placement.x - source_center_x
        local_y = placement.y - source_center_y
        rotated_x = local_x * rotation_cosine - local_y * rotation_sine
        rotated_y = local_x * rotation_sine + local_y * rotation_cosine
        qx, qy, qz, qw = multiply_quaternion(
            0.0,
            0.0,
            rotation_qz,
            rotation_qw,
            placement.qx,
            placement.qy,
            placement.qz,
            placement.qw,
        )
        generated.append(
            GtaPlacement(
                model_id=placement.model_id,
                model_name=placement.model_name,
                interior=0,
                x=target_x + rotated_x,
                y=target_y + rotated_y,
                z=target_ground_z + (placement.z - source_ground_z),
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw,
                lod_index=-1,
                source_file=placement.source_file,
            )
        )

    validation = validate_stock_prototype(scan, generated)
    errors = [message for message in validation if message.startswith("ERROR:")]
    if errors:
        raise RuntimeError("Generated map failed validation:\n" + "\n".join(errors))

    manifest = InstalledPrototype(
        seed=seed,
        game_directory=scan.game_directory,
        mod_directory="",
        source_tile=source_tile,
        source_center=(source_center_x, source_center_y),
        target_center=(target_x, target_y, target_ground_z),
        rotation_degrees=rotation_degrees,
        placements_written=len(generated),
        unique_models=len({placement.model_id for placement in generated}),
        source_files=sorted({placement.source_file for placement in generated}),
        validation=validation,
    )
    return generated, manifest


def multiply_quaternion(
    ax: float,
    ay: float,
    az: float,
    aw: float,
    bx: float,
    by: float,
    bz: float,
    bw: float,
) -> tuple[float, float, float, float]:
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def validate_stock_prototype(scan: GameScan, placements: list[GtaPlacement]) -> list[str]:
    messages: list[str] = []
    if len(placements) < 25:
        messages.append(f"ERROR: Only {len(placements)} placements were generated.")
    missing_ids = sorted({placement.model_id for placement in placements if placement.model_id not in scan.models})
    if missing_ids:
        messages.append(f"ERROR: {len(missing_ids)} generated model IDs are absent from the loaded IDE definitions.")
    outside_bounds = [placement for placement in placements if abs(placement.x) > 2990.0 or abs(placement.y) > 2990.0]
    if outside_bounds:
        messages.append(f"ERROR: {len(outside_bounds)} placements exceed the conservative GTA SA world bounds.")
    invalid_quaternions = [
        placement
        for placement in placements
        if not 0.80 <= math.sqrt(placement.qx ** 2 + placement.qy ** 2 + placement.qz ** 2 + placement.qw ** 2) <= 1.20
    ]
    if invalid_quaternions:
        messages.append(f"ERROR: {len(invalid_quaternions)} placements have invalid rotation quaternions.")
    if not messages:
        messages.append("PASS: All generated IDs exist in the retail IDE catalogue.")
        messages.append("PASS: All generated placements remain within conservative world bounds.")
        messages.append("PASS: All placement quaternions have valid magnitudes.")
        messages.append("PASS: Every LOD reference is disabled instead of retaining an invalid source-IPL index.")
    return messages


def write_mod_package(staging_directory: Path, placements: list[GtaPlacement], manifest: InstalledPrototype) -> Path:
    mod_directory = staging_directory / MOD_NAME
    map_directory = mod_directory / "data" / "maps" / "seedandreas"
    map_directory.mkdir(parents=True, exist_ok=True)
    gta_dat_directory = mod_directory / "data"
    gta_dat_directory.mkdir(parents=True, exist_ok=True)

    ipl_lines = ["inst"]
    for placement in placements:
        ipl_lines.append(
            f"{placement.model_id}, {placement.model_name}, {placement.interior}, "
            f"{placement.x:.6f}, {placement.y:.6f}, {placement.z:.6f}, "
            f"{placement.qx:.8f}, {placement.qy:.8f}, {placement.qz:.8f}, {placement.qw:.8f}, -1"
        )
    ipl_lines.append("end")
    (map_directory / "seedandreas.ipl").write_text("\n".join(ipl_lines) + "\n", encoding="cp1252")

    (gta_dat_directory / "gta.dat").write_text(
        "IPL DATA\\MAPS\\SEEDANDREAS\\SEEDANDREAS.IPL\n",
        encoding="cp1252",
    )
    (mod_directory / "README.txt").write_text(
        "Seed Andreas playable stock-asset prototype\n\n"
        f"Seed: {manifest.seed}\n"
        f"Test district center: {manifest.target_center[0]:.1f}, {manifest.target_center[1]:.1f}, {manifest.target_center[2]:.1f}\n"
        f"Placements: {manifest.placements_written}\n\n"
        "This Mod Loader package adds a procedurally selected and rotated test district over the southeast ocean. "
        "It uses models and collision already present in the legal PC installation. It does not replace gta3.img, main.scm, "
        "the original IPL files, paths, CULL zones, ENEX entries, checkpoints, saves, or audio. Disable the SeedAndreas folder "
        "from Mod Loader configuration to return to the untouched game.\n",
        encoding="utf-8",
    )
    (mod_directory / "seed_andreas_install.json").write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return mod_directory


def normalize_mod_directory_name(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum())


def is_seed_andreas_mod_directory(path: Path) -> bool:
    if not path.is_dir() or path.name == ".data":
        return False
    normalized_name = normalize_mod_directory_name(path.name)
    if any(normalized_name.startswith(prefix) for prefix in LEGACY_MOD_PREFIXES):
        return True
    try:
        child_names = {child.name.lower() for child in path.iterdir() if child.is_file()}
    except OSError:
        return False
    return any(marker in child_names for marker in INSTALL_MARKER_NAMES)


def find_seed_andreas_mod_directories(game_directory: Path) -> list[Path]:
    modloader_directory = game_directory / "modloader"
    if not modloader_directory.is_dir():
        return []
    return sorted(
        (path for path in modloader_directory.iterdir() if is_seed_andreas_mod_directory(path)),
        key=lambda path: path.name.lower(),
    )


def create_backup_bundle(category: str) -> Path:
    backup_root = user_backup_directory() / category
    backup_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    bundle = backup_root / f"SeedAndreas_{timestamp}"
    bundle.mkdir(parents=True, exist_ok=False)
    return bundle


def validate_installed_mod_package(
    game_directory: Path,
    destination: Path,
    manifest: object,
    check_competing_packages: bool = True,
) -> list[str]:
    errors: list[str] = []
    competing = [
        path
        for path in find_seed_andreas_mod_directories(game_directory)
        if path.resolve() != destination.resolve()
    ]
    if check_competing_packages and competing:
        errors.append(
            "Competing Seed Andreas/San AndrAIs Mod Loader folders remain enabled: "
            + ", ".join(path.name for path in competing)
        )
    if not (destination / "seed_andreas_install.json").is_file():
        errors.append("The installed package has no seed_andreas_install.json marker.")
    gta_dat = destination / "data" / "gta.dat"
    if not gta_dat.is_file():
        errors.append("The installed package has no data\\gta.dat override.")
    else:
        try:
            gta_dat_text = gta_dat.read_text(encoding="cp1252").upper()
        except OSError as error:
            errors.append(f"The installed data\\gta.dat could not be read: {error}")
        else:
            if "SEEDANDREAS" not in gta_dat_text:
                errors.append("The installed data\\gta.dat does not register the Seed Andreas IPL.")

    expected_radar_names = list(getattr(manifest, "radar_tiles_overridden", []))
    if expected_radar_names:
        if len(expected_radar_names) != 36:
            errors.append(f"The manifest registers {len(expected_radar_names)} radar overrides instead of 36.")
        stream_directory = destination / "models" / "gta3.img"
        missing_radar = [name for name in expected_radar_names if not (stream_directory / name).is_file()]
        if missing_radar:
            errors.append("Missing installed radar overrides: " + ", ".join(missing_radar))
        full_world_ipl = destination / "data" / "maps" / "seedandreas" / "seedandreas_ls.ipl"
        if not full_world_ipl.is_file():
            errors.append("The full Los Santos placement file seedandreas_ls.ipl is missing.")
    return errors


def install_mod_package(game_directory: Path, staged_mod_directory: Path, manifest: InstalledPrototype) -> tuple[Path, Path | None]:
    problems = validate_game_directory(game_directory, require_mod_loader=True)
    if problems:
        raise RuntimeError("\n".join(problems))
    if not staged_mod_directory.is_dir():
        raise RuntimeError(f"The generated package directory does not exist: {staged_mod_directory}")
    destination = game_directory / "modloader" / MOD_NAME
    # Prepare on the game's volume, outside Mod Loader's active search tree.
    with tempfile.TemporaryDirectory(prefix=".SeedAndreas-install-", dir=game_directory, ignore_cleanup_errors=True) as temporary_directory:
        prepared_mod = Path(temporary_directory) / MOD_NAME
        shutil.copytree(staged_mod_directory, prepared_mod)
        prepared_manifest = asdict(manifest)
        prepared_manifest["mod_directory"] = str(destination)
        (prepared_mod / "seed_andreas_install.json").write_text(
            json.dumps(prepared_manifest, indent=2),
            encoding="utf-8",
        )
        install_errors = validate_installed_mod_package(
            game_directory, prepared_mod, manifest, check_competing_packages=False
        )
        if install_errors:
            raise RuntimeError("Generated Seed Andreas package failed verification:\n" + "\n".join(install_errors))

        conflicts = find_seed_andreas_mod_directories(game_directory)
        expected_radar_names = list(getattr(manifest, "radar_tiles_overridden", []))
        diagnostics = [
            "Seed Andreas install verification",
            f"Destination: {destination}",
            f"Legacy packages moved: {len(conflicts)}",
            f"Radar overrides verified: {len(expected_radar_names)}",
            "Competing packages remaining: 0",
            "Replacement copied and verified before existing packages were moved.",
        ]
        diagnostics.extend(f"Moved legacy package: {source.name}" for source in conflicts)
        (prepared_mod / "INSTALL_DIAGNOSTICS.txt").write_text("\n".join(diagnostics) + "\n", encoding="utf-8")

        backup: Path | None = create_backup_bundle("Replaced") if conflicts else None
        moved_directories: list[tuple[Path, Path]] = []
        previous_packages = Path(temporary_directory) / "previous"
        previous_packages.mkdir()
        activated = False
        try:
            if backup is not None:
                # Finish cross-volume backup copies before removing any active package.
                for source in conflicts:
                    shutil.copytree(source, backup / source.name)
                for source in conflicts:
                    target = previous_packages / source.name
                    source.rename(target)
                    moved_directories.append((source, target))
            prepared_mod.rename(destination)
            activated = True
            install_errors = validate_installed_mod_package(game_directory, destination, manifest)
            if install_errors:
                raise RuntimeError("Installed Seed Andreas package failed verification:\n" + "\n".join(install_errors))
        except Exception as installation_error:
            rollback_errors: list[str] = []
            if activated:
                try:
                    shutil.rmtree(destination)
                except OSError as error:
                    rollback_errors.append(f"Could not remove the new package at {destination}: {error}")
            rollback_errors.extend(restore_moved_packages(moved_directories))
            if backup is not None and not rollback_errors:
                try:
                    shutil.rmtree(backup)
                except OSError:
                    pass
            if rollback_errors:
                raise RuntimeError(
                    f"Installation failed: {installation_error}\n"
                    "The previous map could not be fully restored:\n" + "\n".join(rollback_errors)
                    + f"\nRecoverable copies of the previous packages are at {backup}."
                ) from installation_error
            raise
    manifest.mod_directory = str(destination)
    return destination, backup


def restore_moved_packages(moved_directories: list[tuple[Path, Path]]) -> list[str]:
    errors: list[str] = []
    for source, target in reversed(moved_directories):
        if not target.exists():
            continue
        if source.exists():
            errors.append(f"Restore blocked by {source}; the previous package remains at {target}.")
            continue
        try:
            target.rename(source)
        except OSError as error:
            errors.append(f"Could not restore {source}; the previous package remains at {target}: {error}")
    return errors


def uninstall_mod_package(game_directory: Path) -> Path | None:
    installed_directories = find_seed_andreas_mod_directories(game_directory)
    if not installed_directories:
        return None
    bundle = create_backup_bundle("Uninstalled")
    with tempfile.TemporaryDirectory(prefix=".SeedAndreas-remove-", dir=game_directory, ignore_cleanup_errors=True) as temporary_directory:
        moved_directories: list[tuple[Path, Path]] = []
        try:
            for installed_directory in installed_directories:
                shutil.copytree(installed_directory, bundle / installed_directory.name)
            for installed_directory in installed_directories:
                target = Path(temporary_directory) / installed_directory.name
                installed_directory.rename(target)
                moved_directories.append((installed_directory, target))
        except Exception as uninstall_error:
            rollback_errors = restore_moved_packages(moved_directories)
            if not rollback_errors:
                try:
                    shutil.rmtree(bundle)
                except OSError:
                    pass
            if rollback_errors:
                raise RuntimeError(
                    f"Removal failed: {uninstall_error}\n"
                    "The previous map could not be fully restored:\n" + "\n".join(rollback_errors)
                    + f"\nRecoverable copies of the previous packages are at {bundle}."
                ) from uninstall_error
            raise
    return bundle


def user_backup_directory() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / MOD_NAME / "Backups"
    return Path.home() / MOD_NAME / "Backups"


def launch_game(game_directory: Path) -> None:
    game_directory = game_directory.expanduser().resolve()
    destination = game_directory / "modloader" / MOD_NAME
    competing = [
        path
        for path in find_seed_andreas_mod_directories(game_directory)
        if path.resolve() != destination.resolve()
    ]
    if competing:
        raise RuntimeError(
            "GTA was not launched because competing Seed Andreas/San AndrAIs packages are still enabled: "
            + ", ".join(path.name for path in competing)
            + ". Reinstall the current seed to move them into the recoverable backup automatically."
        )
    executable = game_directory / "gta_sa.exe"
    if not executable.is_file():
        raise RuntimeError(f"gta_sa.exe was not found at {executable}")
    subprocess.Popen([str(executable)], cwd=str(game_directory))
