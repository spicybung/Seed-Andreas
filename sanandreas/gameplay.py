from __future__ import annotations

import hashlib
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .gta_integration import GameScan, TrafficPathSegment


WORLD_BOUNDS = (0.0, -3000.0, 3000.0, 0.0)
TRACK_FILENAMES = ("tracks.dat", "tracks3.dat", "tracks2.dat", "tracks4.dat")
RAIL_MODEL_PATTERN = re.compile(r"(?:rail|track|train|tram|tunnel|tunl|subway|stn_|_stn|station)", re.IGNORECASE)
RAIL_STRUCTURE_PATTERN = re.compile(r"(?:bridge|bridg|brg|viaduct|underpass)", re.IGNORECASE)


@dataclass(frozen=True)
class TrackPoint:
    x: float
    y: float
    z: float
    metadata: tuple[str, ...] = ()


@dataclass(frozen=True)
class RailTrack:
    filename: str
    track_id: int
    points: tuple[TrackPoint, ...]
    original_bytes: bytes
    sha256: str


@dataclass(frozen=True)
class RailPlan:
    tracks: tuple[RailTrack, ...]
    segments: tuple[TrafficPathSegment, ...]
    protected_bounds: tuple[tuple[float, float, float, float], ...]
    protected_model_ids: frozenset[int]
    source_hashes: dict[str, str]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class MissionRegion:
    name: str
    bounds: tuple[float, float, float, float]
    reason: str
    preserve_retail_surface: bool = True


@dataclass(frozen=True)
class VehicleRecordingFrame:
    timestamp: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class RecordingPlan:
    segments: tuple[TrafficPathSegment, ...] = ()
    protected_bounds: tuple[tuple[float, float, float, float], ...] = ()
    source_hashes: dict[str, str] = field(default_factory=dict)
    recordings: int = 0
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class GameplayPlan:
    rail: RailPlan
    mission_regions: tuple[MissionRegion, ...]
    protected_bounds: tuple[tuple[float, float, float, float], ...]
    protected_model_ids: frozenset[int]
    diagnostics: tuple[str, ...]
    recordings: RecordingPlan = field(default_factory=RecordingPlan)


def parse_track_file(data: bytes, filename: str = "tracks.dat", track_id: int = 0) -> RailTrack:
    if not data or b"\0" in data:
        raise ValueError(f"{filename}: train paths must be a nonempty text file.")
    lines = []
    for line_number, raw in enumerate(data.decode("ascii", errors="strict").splitlines(), 1):
        line = raw.partition("#")[0].strip()
        if line:
            lines.append((line_number, line))
    if not lines:
        raise ValueError(f"{filename}: no train-path count was found.")
    try:
        count = int(lines[0][1], 10)
    except ValueError as error:
        raise ValueError(f"{filename}: the first data line must contain the point count.") from error
    if count < 2 or count > 32767:
        raise ValueError(f"{filename}: train-path point count {count} is outside the supported range.")
    if len(lines) - 1 != count:
        raise ValueError(f"{filename}: declares {count} train-path points but contains {len(lines) - 1}.")
    points = []
    for line_number, line in lines[1:]:
        fields = line.replace(",", " ").split()
        if len(fields) < 3:
            raise ValueError(f"{filename}:{line_number}: a train-path point needs X, Y and Z.")
        try:
            x, y, z = (float(value) for value in fields[:3])
        except ValueError as error:
            raise ValueError(f"{filename}:{line_number}: invalid train-path coordinate.") from error
        if not all(math.isfinite(value) and -4096.0 <= value <= 4095.875 for value in (x, y, z)):
            raise ValueError(f"{filename}:{line_number}: coordinate cannot fit the native 1/8-unit signed train position.")
        native_x, native_y, native_z = (math.trunc(value * 8.0) / 8.0 for value in (x, y, z))
        points.append(TrackPoint(native_x, native_y, native_z, tuple(fields[3:])))
    return RailTrack(filename, track_id, tuple(points), data, hashlib.sha256(data).hexdigest())


def find_case_insensitive(directory: Path, filename: str) -> Path | None:
    if not directory.is_dir():
        return None
    return next((path for path in directory.iterdir() if path.is_file() and path.name.lower() == filename.lower()), None)


def segment_reaches_bounds(segment: TrafficPathSegment, bounds: tuple[float, float, float, float], margin: float = 0.0) -> bool:
    minimum_x, minimum_y, maximum_x, maximum_y = bounds
    enter, leave = 0.0, 1.0
    for start, delta, lower, upper in (
        (segment.start_x, segment.end_x - segment.start_x, minimum_x - margin, maximum_x + margin),
        (segment.start_y, segment.end_y - segment.start_y, minimum_y - margin, maximum_y + margin),
    ):
        if abs(delta) <= 0.0000001:
            if not lower <= start <= upper:
                return False
            continue
        first, second = (lower - start) / delta, (upper - start) / delta
        enter, leave = max(enter, min(first, second)), min(leave, max(first, second))
        if enter > leave:
            return False
    return True


def rail_corridor_bounds(segments: Iterable[TrafficPathSegment], maximum_piece_length: float = 35.0) -> tuple[tuple[float, float, float, float], ...]:
    if maximum_piece_length <= 0.0:
        raise ValueError("Rail corridor segment length must be positive.")
    bounds = []
    for segment in segments:
        distance = math.hypot(segment.end_x - segment.start_x, segment.end_y - segment.start_y)
        pieces = max(1, math.ceil(distance / maximum_piece_length))
        margin = segment.width * 0.5
        for index in range(pieces):
            first_factor, second_factor = index / pieces, (index + 1) / pieces
            x0 = segment.start_x + (segment.end_x - segment.start_x) * first_factor
            y0 = segment.start_y + (segment.end_y - segment.start_y) * first_factor
            x1 = segment.start_x + (segment.end_x - segment.start_x) * second_factor
            y1 = segment.start_y + (segment.end_y - segment.start_y) * second_factor
            bounds.append((min(x0, x1) - margin, min(y0, y1) - margin, max(x0, x1) + margin, max(y0, y1) + margin))
    return tuple(dict.fromkeys(bounds))


def point_in_bounds(x: float, y: float, bounds: Iterable[tuple[float, float, float, float]]) -> bool:
    return any(left <= x <= right and bottom <= y <= top for left, bottom, right, top in bounds)


def find_child_directory(parent: Path, name: str) -> Path:
    if parent.is_dir():
        for candidate in parent.iterdir():
            if candidate.is_dir() and candidate.name.lower() == name.lower():
                return candidate
    return parent / name


def collect_rail_plan(game_directory: Path, scan: GameScan | None = None, bounds: tuple[float, float, float, float] = WORLD_BOUNDS) -> RailPlan:
    data_directory = find_child_directory(game_directory, "data")
    paths_directory = find_child_directory(data_directory, "paths")
    tracks = []
    segments = []
    diagnostics = []
    for track_id, filename in enumerate(TRACK_FILENAMES):
        path = find_case_insensitive(paths_directory, filename)
        if path is None:
            diagnostics.append(f"Train path {filename} was not present in this installation.")
            continue
        track = parse_track_file(path.read_bytes(), path.name, track_id)
        tracks.append(track)
        pairs = list(zip(track.points, track.points[1:]))
        closing_distance = math.dist((track.points[-1].x, track.points[-1].y, track.points[-1].z), (track.points[0].x, track.points[0].y, track.points[0].z))
        if closing_distance <= 100.0:
            pairs.append((track.points[-1], track.points[0]))
        else:
            diagnostics.append(f"{path.name}: distant final-to-first mission-track wrap is retained in the source; no artificial cross-city rail corridor was invented.")
        for first, second in pairs:
            if math.hypot(second.x - first.x, second.y - first.y) <= 0.0001:
                continue
            segment = TrafficPathSegment(first.x, first.y, first.z, second.x, second.y, second.z, 24.0)
            if segment_reaches_bounds(segment, bounds, segment.width * 0.5):
                segments.append(segment)
    corridor_bounds = rail_corridor_bounds(segments)
    model_ids = set()
    if scan is not None:
        for model_id, model in scan.models.items():
            if RAIL_MODEL_PATTERN.search(model.model_name):
                model_ids.add(model_id)
        for placement in scan.placements:
            if point_in_bounds(placement.x, placement.y, corridor_bounds):
                model_ids.add(placement.model_id)
                continue
            model = scan.models.get(placement.model_id)
            if model is not None and RAIL_STRUCTURE_PATTERN.search(model.model_name):
                if any(math.hypot(placement.x - (segment.start_x + segment.end_x) * 0.5, placement.y - (segment.start_y + segment.end_y) * 0.5) <= 180.0 for segment in segments):
                    model_ids.add(placement.model_id)
    diagnostics.append(f"Preserved {len(tracks)} original train-path files, {len(segments)} local rail segments and {len(model_ids)} railway/static model identities.")
    return RailPlan(tuple(tracks), tuple(segments), corridor_bounds, frozenset(model_ids), {track.filename: track.sha256 for track in tracks}, tuple(diagnostics))


def los_santos_mission_regions() -> tuple[MissionRegion, ...]:
    return (
        MissionRegion("Grove Street compound", (2380.0, -1795.0, 2595.0, -1575.0), "Johnson, Sweet and Ryder houses; opening and return-to-Los-Santos mission staging."),
        MissionRegion("Big Smoke house", (2390.0, -1340.0, 2580.0, -1190.0), "Smoke mission starts, driveway, surrounding cutscene and vehicle access."),
        MissionRegion("OG Loc neighborhood", (2330.0, -1240.0, 2475.0, -1060.0), "OG Loc house and nearby mission staging."),
        MissionRegion("Cesar and El Corona", (1730.0, -2150.0, 1940.0, -1910.0), "Cesar staging, lowrider gathering and approaches."),
        MissionRegion("Emmet yard", (2330.0, -2050.0, 2470.0, -1910.0), "Nines and AKs shooting-yard collision and access."),
        MissionRegion("Unity Station", (1620.0, -1990.0, 1910.0, -1770.0), "Wrong Side of the Tracks start, platform, station access and train corridor."),
        MissionRegion("Jefferson Motel", (2110.0, -1250.0, 2340.0, -1050.0), "Reuniting the Families motel, roof, balconies and escape approaches."),
        MissionRegion("Burning Desire house", (2260.0, -1245.0, 2440.0, -1100.0), "Fixed exterior/interior connection and emergency access."),
        MissionRegion("Madd Dogg estate", (1180.0, -920.0, 1425.0, -680.0), "Madd Dogg mansion entrance, grounds, roof and later mission staging."),
        MissionRegion("Vinewood cemetery", (740.0, -1265.0, 1080.0, -1000.0), "Los Sepulcros cemetery walls, graves and exits."),
        MissionRegion("Verona conference compound", (1030.0, -1535.0, 1265.0, -1280.0), "Just Business gunfight staging and surrounding access."),
        MissionRegion("Idlewood restaurant", (2320.0, -1900.0, 2500.0, -1750.0), "Drive-Thru restaurant, driveways and scripted vehicle access."),
        MissionRegion("Marina Burger Shot", (740.0, -1715.0, 905.0, -1530.0), "Ryder and food-job restaurant/shop access."),
        MissionRegion("Temple Burger Shot", (1100.0, -1025.0, 1295.0, -790.0), "OG Loc job and Management Issues mission staging."),
        MissionRegion("Beach party and pier", (365.0, -2165.0, 710.0, -1900.0), "Life's a Beach, pier and beach-vehicle access."),
        MissionRegion("Santa Maria waterfront", (100.0, -1905.0, 680.0, -1580.0), "Waterfront mission staging and road-to-pier access."),
        MissionRegion("Ocean Docks army depot", (2570.0, -2570.0, 2910.0, -2250.0), "Robbing Uncle Sam depot, gates, warehouses and forklift route."),
        MissionRegion("Ocean Docks import warehouses", (2630.0, -2250.0, 2910.0, -1960.0), "Gray Imports and late-story warehouse paths."),
        MissionRegion("Los Santos airport", (1330.0, -2890.0, 2175.0, -2150.0), "Airport terminals, fixed aircraft staging and runway approaches."),
        MissionRegion("Los Santos police and civic square", (1435.0, -1835.0, 1650.0, -1540.0), "Police compound, civic staging and return-story mission access."),
        MissionRegion("Mulholland intersection", (1160.0, -1170.0, 1540.0, -900.0), "The Green Sabre staging and road escape access."),
        MissionRegion("Los Santos storm drain", (2310.0, -2250.0, 2560.0, -1100.0), "Just Business recorded motorcycle route and drainage-channel mission access."),
        MissionRegion("East Los Santos finale", (2420.0, -1390.0, 2705.0, -1090.0), "End of the Line building and fixed mission entrance/exit staging."),
        MissionRegion("East Beach stadium", (2500.0, -1900.0, 2910.0, -1550.0), "Stadium frontage, events and vehicle approaches."),
        MissionRegion("Vinewood observatory", (1030.0, -315.0, 1270.0, -55.0), "Observatory fixed collision and north-city approaches."),
    )


def parse_vehicle_recording(data: bytes, name: str = "carrec.rrr") -> tuple[VehicleRecordingFrame, ...]:
    if len(data) < 32:
        raise ValueError(f"{name}: a vehicle recording needs at least one complete 32-byte frame.")
    frames = []
    for offset in range(0, len(data) - 31, 32):
        timestamp = struct.unpack_from("<I", data, offset)[0]
        if offset and timestamp == 0:
            break
        x, y, z = struct.unpack_from("<3f", data, offset + 20)
        if not all(math.isfinite(value) and -6000.0 <= value <= 6000.0 for value in (x, y, z)):
            raise ValueError(f"{name}: recording frame {offset // 32} contains an invalid XYZ position.")
        frames.append(VehicleRecordingFrame(timestamp, x, y, z))
    return tuple(frames)


def read_recording_archive(path: Path) -> dict[str, bytes]:
    recordings = {}
    with path.open("rb") as stream:
        header = stream.read(8)
        if len(header) != 8 or header[:4] != b"VER2":
            raise ValueError(f"{path.name}: vehicle recordings require a GTA SA VER2 IMG archive.")
        count = struct.unpack_from("<I", header, 4)[0]
        file_size = path.stat().st_size
        if count > 100000 or 8 + count * 32 > file_size:
            raise ValueError(f"{path.name}: invalid recording archive directory size.")
        directory = stream.read(count * 32)
        for index in range(count):
            sector, streaming_sectors, archive_sectors, raw_name = struct.unpack_from("<IHH24s", directory, index * 32)
            name = raw_name.split(b"\0", 1)[0].decode("ascii", errors="strict")
            if not re.fullmatch(r"carrec\d+\.rrr", name, re.IGNORECASE):
                continue
            length = (archive_sectors or streaming_sectors) * 2048
            position = sector * 2048
            if not length or position < 8 + count * 32 or position + length > file_size:
                raise ValueError(f"{path.name}: recording {name} points outside the IMG archive.")
            if name.lower() in recordings:
                raise ValueError(f"{path.name}: duplicate recording {name}.")
            stream.seek(position)
            recordings[name.lower()] = stream.read(length)
    return recordings


def collect_recording_plan(game_directory: Path, bounds: tuple[float, float, float, float] = WORLD_BOUNDS) -> RecordingPlan:
    paths_directory = find_child_directory(find_child_directory(game_directory, "data"), "paths")
    archive = find_case_insensitive(paths_directory, "carrec.img")
    if archive is None:
        return RecordingPlan(diagnostics=("Vehicle-recording archive carrec.img was not present in this installation.",))
    recordings = read_recording_archive(archive)
    segments = []
    identities = set()
    route_count = 0
    for name, data in recordings.items():
        frames = parse_vehicle_recording(data, name)
        if not frames:
            continue
        selected = [frames[0]]
        for frame in frames[1:-1]:
            previous = selected[-1]
            if math.hypot(frame.x - previous.x, frame.y - previous.y) >= 6.0:
                selected.append(frame)
        if len(frames) > 1:
            selected.append(frames[-1])
        retained = 0
        for first, second in zip(selected, selected[1:]):
            distance = math.hypot(second.x - first.x, second.y - first.y)
            if distance < 0.05 or distance > 120.0 or min(first.z, second.z) < -30.0 or max(first.z, second.z) > 120.0:
                continue
            segment = TrafficPathSegment(first.x, first.y, first.z, second.x, second.y, second.z, 20.0)
            if not segment_reaches_bounds(segment, bounds, 10.0):
                continue
            key = tuple(sorted(((round(first.x), round(first.y), round(first.z)), (round(second.x), round(second.y), round(second.z)))))
            if key in identities:
                continue
            identities.add(key)
            segments.append(segment)
            retained += 1
        route_count += retained > 0
    corridor_bounds = rail_corridor_bounds(segments, 25.0)
    hashes = {str(archive.relative_to(game_directory)): hashlib.sha256(archive.read_bytes()).hexdigest()}
    diagnostics = (
        f"Preserved {route_count} low-elevation native vehicle recordings and {len(segments)} scripted-route corridor segments; original CARREC bytes are unchanged.",
        "Aircraft/high-altitude recording samples and teleport-length discontinuities are not treated as ground roads.",
    )
    return RecordingPlan(tuple(segments), corridor_bounds, hashes, route_count, diagnostics)


def collect_gameplay_plan(scan: GameScan) -> GameplayPlan:
    rail = collect_rail_plan(Path(scan.game_directory), scan)
    recordings = collect_recording_plan(Path(scan.game_directory))
    regions = los_santos_mission_regions()
    bounds = tuple(dict.fromkeys((*rail.protected_bounds, *recordings.protected_bounds, *(region.bounds for region in regions))))
    model_ids = set(rail.protected_model_ids)
    for placement in scan.placements:
        if point_in_bounds(placement.x, placement.y, bounds):
            model_ids.add(placement.model_id)
    diagnostics = [*rail.diagnostics, *recordings.diagnostics]
    diagnostics.append(f"Reserved {len(regions)} named Los Santos mission compounds and their retail collision surfaces.")
    diagnostics.append("Mission compounds and recorded routes are preservation checks, not a decompilation of main.scm or proof of every mission's completion.")
    return GameplayPlan(rail, regions, bounds, frozenset(model_ids), tuple(diagnostics), recordings)


def validate_preserved_tracks(plan: RailPlan, game_directory: Path) -> list[str]:
    errors = []
    for track in plan.tracks:
        directory = find_child_directory(find_child_directory(game_directory, "data"), "paths")
        path = find_case_insensitive(directory, track.filename)
        if path is None or hashlib.sha256(path.read_bytes()).hexdigest() != track.sha256:
            errors.append(f"Original train path changed during generation: {track.filename}")
    return errors
