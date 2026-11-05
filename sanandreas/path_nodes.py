from __future__ import annotations

import math
import hashlib
import re
import struct
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gta_integration import TrafficPathSegment


DYNAMIC_LINK_SLOTS = 16 * 12
PATH_NODE_SIZE = 28
CAR_LINK_SIZE = 14
MAX_STATIC_LINKS = 32767 - DYNAMIC_LINK_SLOTS
MAX_CAR_LINKS = 1024
WORLD_MINIMUM = -3000.0
SECTOR_SIZE = 750.0
NODE_FILE_PATTERN = re.compile(r"^nodes(\d+)\.dat$", re.IGNORECASE)


@dataclass(frozen=True, order=True)
class NodeAddress:
    area: int
    node: int

    def packed(self) -> bytes:
        return struct.pack("<HH", self.area, self.node)


@dataclass
class BinaryNodeSector:
    area_id: int
    vehicle_count: int
    nodes: list[bytes]
    car_links: list[bytes]
    addresses: list[NodeAddress]
    navigation: list[int]
    lengths: bytearray
    intersections: bytearray
    dynamic_addresses: list[NodeAddress] = field(default_factory=list)
    dynamic_lengths: bytes = b""
    dynamic_intersections: bytes = b""
    trailing_data: bytes = b""

    @property
    def pedestrian_count(self) -> int:
        return len(self.nodes) - self.vehicle_count

    def position(self, node_id: int) -> tuple[float, float, float]:
        return tuple(value / 8.0 for value in struct.unpack_from("<3h", self.nodes[node_id], 8))


@dataclass
class BinaryPathNetwork:
    files: dict[str, bytes]
    generated_segments: list[TrafficPathSegment]
    nodes_added: int
    links_added: int
    retail_nodes_preserved: int
    pedestrian_nodes_remapped: int
    rejected_segments: int
    validation: list[str]
    sectors: dict[int, BinaryNodeSector] = field(default_factory=dict)
    source_hashes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedRoadEdge:
    first: tuple[int, int, int]
    second: tuple[int, int, int]
    width: float


@dataclass(frozen=True)
class AppendedLink:
    target: NodeAddress
    navigation: int
    length: int
    intersection: int = 0


def pack_navigation_address(area_id: int, car_link_id: int) -> int:
    if not 0 <= area_id < 64 or not 0 <= car_link_id < MAX_CAR_LINKS:
        raise ValueError("A navigation address exceeds its six-bit area or ten-bit link index.")
    value = (area_id << 10) | car_link_id
    if value == 0xFFFF:
        raise ValueError("The navigation address collides with the invalid-address sentinel.")
    return value


def unpack_navigation_address(value: int) -> tuple[int, int] | None:
    return None if value == 0xFFFF else (value >> 10, value & 0x3FF)


def sector_for_position(x: float, y: float) -> int:
    if not math.isfinite(x) or not math.isfinite(y) or not -3000.0 <= x <= 3000.0 or not -3000.0 <= y <= 3000.0:
        raise ValueError("Path coordinates are outside GTA San Andreas' exterior map bounds.")
    column = min(7, int(math.floor((x - WORLD_MINIMUM) / SECTOR_SIZE)))
    row = min(7, int(math.floor((y - WORLD_MINIMUM) / SECTOR_SIZE)))
    return row * 8 + column


def parse_binary_node_file(data: bytes, area_id: int) -> BinaryNodeSector:
    if not 0 <= area_id < 64:
        raise ValueError("Exterior NODES area must be between zero and 63.")
    if len(data) < 20:
        raise ValueError(f"NODES{area_id}.DAT has a truncated header.")
    node_count, vehicle_count, pedestrian_count, car_count, address_count = struct.unpack_from("<5I", data)
    if node_count != vehicle_count + pedestrian_count or node_count > 65535:
        raise ValueError(f"NODES{area_id}.DAT has inconsistent or unsupported node counts.")
    if car_count > MAX_CAR_LINKS or address_count > MAX_STATIC_LINKS:
        raise ValueError(f"NODES{area_id}.DAT exceeds native navigation or signed link-index limits.")
    stored_count = address_count + DYNAMIC_LINK_SLOTS if address_count else 0
    required = 20 + node_count * PATH_NODE_SIZE + car_count * CAR_LINK_SIZE + stored_count * 6 + address_count * 2
    if len(data) < required:
        raise ValueError(
            f"NODES{area_id}.DAT is truncated: {required} bytes are required for all native arrays "
            f"including {DYNAMIC_LINK_SLOTS} dynamic link slots, but only {len(data)} are present."
        )
    offset = 20
    nodes = [data[offset + index * PATH_NODE_SIZE:offset + (index + 1) * PATH_NODE_SIZE] for index in range(node_count)]
    offset += node_count * PATH_NODE_SIZE
    car_links = [data[offset + index * CAR_LINK_SIZE:offset + (index + 1) * CAR_LINK_SIZE] for index in range(car_count)]
    offset += car_count * CAR_LINK_SIZE
    all_addresses = [NodeAddress(*struct.unpack_from("<HH", data, offset + index * 4)) for index in range(stored_count)]
    offset += stored_count * 4
    navigation = list(struct.unpack_from(f"<{address_count}H", data, offset)) if address_count else []
    offset += address_count * 2
    all_lengths = data[offset:offset + stored_count]
    offset += stored_count
    all_intersections = data[offset:offset + stored_count]
    offset += stored_count
    sector = BinaryNodeSector(
        area_id, vehicle_count, nodes, car_links, all_addresses[:address_count], navigation,
        bytearray(all_lengths[:address_count]), bytearray(all_intersections[:address_count]),
        all_addresses[address_count:], all_lengths[address_count:], all_intersections[address_count:], data[offset:],
    )
    validate_sector_structure(sector)
    return sector


def validate_sector_structure(sector: BinaryNodeSector) -> None:
    count = len(sector.addresses)
    if not (count == len(sector.navigation) == len(sector.lengths) == len(sector.intersections)):
        raise ValueError(f"NODES{sector.area_id}.DAT parallel link arrays have different lengths.")
    if count > MAX_STATIC_LINKS or len(sector.car_links) > MAX_CAR_LINKS or len(sector.nodes) > 65535:
        raise ValueError(f"NODES{sector.area_id}.DAT exceeds a native index limit.")
    if not 0 <= sector.vehicle_count <= len(sector.nodes):
        raise ValueError("Vehicle nodes must precede pedestrian nodes.")
    for index, raw in enumerate(sector.nodes):
        if len(raw) != PATH_NODE_SIZE:
            raise ValueError("A path node does not contain exactly 28 bytes.")
        base = struct.unpack_from("<h", raw, 16)[0]
        degree = raw[24] & 15
        area, node_id = struct.unpack_from("<HH", raw, 18)
        if (area, node_id) != (sector.area_id, index):
            raise ValueError(f"NODES{sector.area_id}.DAT node {index} has a mismatched stored address.")
        if degree and (base < 0 or base + degree > count):
            raise ValueError(f"NODES{sector.area_id}.DAT node {index} references links outside its static array.")
    if any(len(raw) != CAR_LINK_SIZE for raw in sector.car_links):
        raise ValueError("A car navigation link does not contain exactly 14 bytes.")


def write_binary_node_file(sector: BinaryNodeSector) -> bytes:
    validate_sector_structure(sector)
    count = len(sector.addresses)
    header = struct.pack("<5I", len(sector.nodes), sector.vehicle_count, sector.pedestrian_count, len(sector.car_links), count)
    output = bytearray(header + b"".join(sector.nodes) + b"".join(sector.car_links))
    if count:
        dynamic_addresses = sector.dynamic_addresses or [NodeAddress(0xFFFF, 0xFFFF)] * DYNAMIC_LINK_SLOTS
        dynamic_lengths = sector.dynamic_lengths or bytes(DYNAMIC_LINK_SLOTS)
        dynamic_intersections = sector.dynamic_intersections or bytes(DYNAMIC_LINK_SLOTS)
        if not (len(dynamic_addresses) == len(dynamic_lengths) == len(dynamic_intersections) == DYNAMIC_LINK_SLOTS):
            raise ValueError("A nonempty NODES sector requires exactly 192 dynamic link slots.")
        output += b"".join(address.packed() for address in sector.addresses + dynamic_addresses)
        output += struct.pack(f"<{count}H", *sector.navigation)
        output += sector.lengths + dynamic_lengths
        output += sector.intersections + dynamic_intersections
    output += sector.trailing_data
    return bytes(output)


def load_binary_path_network(
    paths_directory: Path,
    source_files: dict[str | int, bytes] | None = None,
) -> tuple[dict[int, BinaryNodeSector], dict[int, bytes]]:
    payloads: dict[int, bytes] = {}
    if source_files is not None:
        for name, data in source_files.items():
            if isinstance(name, int):
                area_id = name
            else:
                match = NODE_FILE_PATTERN.match(Path(name).name)
                if match is None:
                    raise ValueError(f"Not a native path filename: {name}")
                area_id = int(match.group(1))
            if not 0 <= area_id < 64 or area_id in payloads:
                raise ValueError("Duplicate or invalid native path sector.")
            payloads[area_id] = bytes(data)
    elif paths_directory.is_dir():
        for path in paths_directory.iterdir():
            match = NODE_FILE_PATTERN.match(path.name)
            if match and path.is_file() and 0 <= int(match.group(1)) < 64:
                payloads[int(match.group(1))] = path.read_bytes()
    sectors = {area: parse_binary_node_file(data, area) for area, data in sorted(payloads.items())}
    validate_binary_path_network(sectors)
    return sectors, payloads


def validate_binary_path_network(sectors: dict[int, BinaryNodeSector]) -> None:
    for area, sector in sectors.items():
        validate_sector_structure(sector)
        for node_id, raw in enumerate(sector.nodes):
            base = struct.unpack_from("<h", raw, 16)[0]
            for index in range(base, base + (raw[24] & 15)):
                address = sector.addresses[index]
                target = sectors.get(address.area)
                if target is None or address.node >= len(target.nodes):
                    raise ValueError(f"NODES{area}.DAT node {node_id} links to missing node {address}.")
                navigation = unpack_navigation_address(sector.navigation[index])
                if navigation is not None:
                    navigation_area, navigation_id = navigation
                    navigation_sector = sectors.get(navigation_area)
                    if navigation_sector is None or navigation_id >= len(navigation_sector.car_links):
                        raise ValueError(f"NODES{area}.DAT contains an invalid car navigation address.")
        for raw in sector.car_links:
            address = NodeAddress(*struct.unpack_from("<HH", raw, 4))
            target = sectors.get(address.area)
            if target is None or address.node >= len(target.nodes):
                raise ValueError(f"NODES{area}.DAT car navigation target {address} does not exist.")


def position_key(x: float, y: float, z: float) -> tuple[int, int, int]:
    if any(not math.isfinite(value) for value in (x, y, z)):
        raise ValueError("Generated path positions must be finite.")
    key = tuple(round(value * 8.0) for value in (x, y, z))
    if any(value < -32768 or value > 32767 for value in key):
        raise ValueError("Generated path coordinates exceed the native fixed-point range.")
    sector_for_position(x, y)
    return key


def segment_crosses_bounds(
    start: tuple[float, float, float], end: tuple[float, float, float], bounds: tuple[float, float, float, float],
    margin: float = 0.0,
) -> bool:
    minimum_x, minimum_y, maximum_x, maximum_y = bounds
    minimum_factor, maximum_factor = 0.0, 1.0
    for coordinate, difference, lower, upper in (
        (start[0], end[0] - start[0], minimum_x - margin, maximum_x + margin),
        (start[1], end[1] - start[1], minimum_y - margin, maximum_y + margin),
    ):
        if abs(difference) <= 0.000001:
            if not lower <= coordinate <= upper:
                return False
        else:
            first, second = (lower - coordinate) / difference, (upper - coordinate) / difference
            minimum_factor, maximum_factor = max(minimum_factor, min(first, second)), min(maximum_factor, max(first, second))
            if minimum_factor > maximum_factor:
                return False
    return True


def split_generated_road_segments(
    segments: list[TrafficPathSegment],
    exclusion_bounds: tuple[tuple[float, float, float, float], ...],
    maximum_length: float = 100.0,
) -> tuple[list[GeneratedRoadEdge], int]:
    edges: dict[tuple[tuple[int, int, int], tuple[int, int, int]], GeneratedRoadEdge] = {}
    rejected = 0
    for segment in segments:
        start, end = (segment.start_x, segment.start_y, segment.start_z), (segment.end_x, segment.end_y, segment.end_z)
        width = float(segment.width)
        if not math.isfinite(width) or width <= 0.0 or width > 64.0:
            raise ValueError("Generated road widths must be finite and between zero and 64.")
        if any(segment_crosses_bounds(start, end, bounds, width * 0.5 + 1.0) for bounds in exclusion_bounds):
            rejected += 1
            continue
        length = math.dist(start, end)
        if length < 0.5:
            continue
        factors = {index / math.ceil(length / maximum_length) for index in range(math.ceil(length / maximum_length) + 1)}
        for axis in (0, 1):
            difference = end[axis] - start[axis]
            if abs(difference) > 0.000001:
                for boundary_index in range(1, 8):
                    factor = (WORLD_MINIMUM + boundary_index * SECTOR_SIZE - start[axis]) / difference
                    if 0.0 < factor < 1.0:
                        factors.add(factor)
        points = [position_key(*(start[axis] + (end[axis] - start[axis]) * factor for axis in range(3))) for factor in sorted(factors)]
        for first, second in zip(points, points[1:]):
            if first == second:
                continue
            key = tuple(sorted((first, second)))
            current = edges.get(key)
            if current is None or width > current.width:
                edges[key] = GeneratedRoadEdge(first, second, width)
    return list(edges.values()), rejected


def graph_components(edges: list[GeneratedRoadEdge]) -> list[set[tuple[int, int, int]]]:
    adjacency: dict[tuple[int, int, int], set[tuple[int, int, int]]] = defaultdict(set)
    for edge in edges:
        adjacency[edge.first].add(edge.second)
        adjacency[edge.second].add(edge.first)
    remaining = set(adjacency)
    result = []
    while remaining:
        first = min(remaining)
        component, pending = {first}, deque([first])
        remaining.remove(first)
        while pending:
            for neighbor in adjacency[pending.popleft()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
        result.append(component)
    return result


def make_path_node(address: NodeAddress, position: tuple[int, int, int], width: float, group: int, degree: int) -> bytes:
    if degree > 15:
        raise ValueError("A generated junction exceeds the native four-bit link count.")
    raw = bytearray(PATH_NODE_SIZE)
    struct.pack_into("<3hh", raw, 8, *position, 32766)
    struct.pack_into("<HH", raw, 18, address.area, address.node)
    raw[22] = min(255, max(0, round(width * 0.25 * 16.0)))
    raw[23] = group
    raw[24] = degree | (0x10 if degree == 1 else 0)
    raw[25] = 1 << 4
    raw[26] = 10
    return bytes(raw)


def make_car_navigation_link(
    first: tuple[float, float, float], second: tuple[float, float, float], target: NodeAddress, width: float,
) -> bytes:
    delta_x, delta_y = second[0] - first[0], second[1] - first[1]
    length = math.hypot(delta_x, delta_y)
    if length < 0.01:
        raise ValueError("A car link must have a nonzero horizontal direction.")
    lanes = 2 if width >= 18.0 else 1
    path_width = min(127, max(1, round(width * 0.25 * 16.0)))
    return struct.pack(
        "<hhHHbbbBH", round((first[0] + second[0]) * 4.0), round((first[1] + second[1]) * 4.0),
        target.area, target.node, round(delta_x / length * 100.0), round(delta_y / length * 100.0),
        path_width, lanes | (lanes << 3), 0,
    )


def remap_node_address(address: NodeAddress, sectors: dict[int, BinaryNodeSector], additions: dict[int, int]) -> NodeAddress:
    sector = sectors.get(address.area)
    if sector is not None and sector.vehicle_count <= address.node < len(sector.nodes):
        return NodeAddress(address.area, address.node + additions.get(address.area, 0))
    return address


def build_binary_path_network(
    paths_directory: Path,
    generated_segments: list[TrafficPathSegment],
    protected_bounds: tuple[tuple[float, float, float, float], ...] = (),
    rail_bounds: tuple[tuple[float, float, float, float], ...] = (),
    source_files: dict[str | int, bytes] | None = None,
    maximum_graft_distance: float = 80.0,
) -> BinaryPathNetwork:
    from .gta_integration import TrafficPathSegment

    sectors, originals = load_binary_path_network(paths_directory, source_files)
    source_hashes = {f"nodes{area}.dat": hashlib.sha256(data).hexdigest() for area, data in originals.items()}
    if not sectors:
        raise ValueError("No complete native NODES sectors were found; refusing to export an unconnected traffic graph.")
    exclusion_bounds = tuple(protected_bounds) + tuple(rail_bounds)
    edges, rejected = split_generated_road_segments(generated_segments, exclusion_bounds)
    retail_nodes = sum(len(sector.nodes) for sector in sectors.values())
    eligible: dict[tuple[int, int], list[tuple[NodeAddress, tuple[float, float, float], int]]] = defaultdict(list)
    for area, sector in sectors.items():
        for node_id, raw in enumerate(sector.nodes[:sector.vehicle_count]):
            position = sector.position(node_id)
            if raw[24] & 0xE0 or (raw[24] & 15) >= 15:
                continue
            if any(segment_crosses_bounds(position, position, bounds, 2.0) for bounds in exclusion_bounds):
                continue
            eligible[(math.floor(position[0] / 80.0), math.floor(position[1] / 80.0))].append((NodeAddress(area, node_id), position, raw[23]))

    groups: dict[tuple[int, int, int], int] = {}
    grafts: list[tuple[tuple[int, int, int], NodeAddress, float]] = []
    aliases: dict[tuple[int, int, int], NodeAddress] = {}
    graft_counts: dict[NodeAddress, int] = defaultdict(int)
    retained_keys: set[tuple[int, int, int]] = set()
    for component in graph_components(edges):
        options: list[tuple[float, tuple[int, int, int], NodeAddress, int]] = []
        for key in sorted(component):
            position = tuple(value / 8.0 for value in key)
            cell_x, cell_y = math.floor(position[0] / 80.0), math.floor(position[1] / 80.0)
            for offset_y in range(-1, 2):
                for offset_x in range(-1, 2):
                    for address, target, group in eligible.get((cell_x + offset_x, cell_y + offset_y), []):
                        distance = math.dist(position, target)
                        if not 0.0 <= distance <= maximum_graft_distance or abs(position[2] - target[2]) > 1.5:
                            continue
                        if abs(position[2] - target[2]) > max(0.125, distance * 0.10):
                            continue
                        if any(segment_crosses_bounds(position, target, bounds, 7.0) for bounds in exclusion_bounds):
                            continue
                        options.append((distance, key, address, group))
        options.sort()
        if not options:
            rejected += sum(edge.first in component for edge in edges)
            continue
        group = options[0][3]
        used_generated, used_retail = set(), set()
        for distance, key, address, option_group in options:
            original_degree = sectors[address.area].nodes[address.node][24] & 15
            if (option_group != group or key in used_generated or address in used_retail
                    or original_degree + graft_counts[address] >= 15):
                continue
            if any(math.dist(key, old) < 120.0 * 8.0 for old in used_generated):
                continue
            used_generated.add(key)
            used_retail.add(address)
            if distance < 0.125:
                aliases[key] = address
            else:
                grafts.append((key, address, 11.0))
            graft_counts[address] += 1
            if len(used_generated) >= 4:
                break
        if not used_generated:
            rejected += sum(edge.first in component for edge in edges)
            continue
        retained_keys.update(component)
        groups.update({key: group for key in component})
    edges = [edge for edge in edges if edge.first in retained_keys and edge.second in retained_keys]
    if not edges:
        return BinaryPathNetwork({}, [], 0, 0, retail_nodes, 0, rejected,
                                 ["No generated street component had a grade-compatible, unprotected retail road attachment."], sectors, source_hashes)

    keys_by_area: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for key in sorted(retained_keys - set(aliases)):
        area = sector_for_position(key[0] / 8.0, key[1] / 8.0)
        if area not in sectors:
            sectors[area] = BinaryNodeSector(area, 0, [], [], [], [], bytearray(), bytearray())
            originals[area] = write_binary_node_file(sectors[area])
        keys_by_area[area].append(key)
    additions = {area: len(keys) for area, keys in keys_by_area.items()}
    addresses: dict[tuple[int, int, int], NodeAddress] = dict(aliases)
    for area, keys in keys_by_area.items():
        for offset, key in enumerate(keys):
            addresses[key] = NodeAddress(area, sectors[area].vehicle_count + offset)
    original_vehicle_counts = {area: sector.vehicle_count for area, sector in sectors.items()}
    updated: dict[int, BinaryNodeSector] = {}
    for area, sector in sectors.items():
        added = additions.get(area, 0)
        nodes = []
        for old_id, raw in enumerate(sector.nodes):
            node = bytearray(raw)
            if old_id >= sector.vehicle_count:
                struct.pack_into("<H", node, 20, old_id + added)
            nodes.append(bytes(node))
        old_car_links = []
        for raw in sector.car_links:
            car_link = bytearray(raw)
            target = remap_node_address(NodeAddress(*struct.unpack_from("<HH", raw, 4)), sectors, additions)
            struct.pack_into("<HH", car_link, 4, target.area, target.node)
            old_car_links.append(bytes(car_link))
        updated[area] = BinaryNodeSector(
            area, sector.vehicle_count + added,
            nodes[:sector.vehicle_count] + [b""] * added + nodes[sector.vehicle_count:], old_car_links,
            [remap_node_address(address, sectors, additions) for address in sector.addresses], list(sector.navigation),
            bytearray(sector.lengths), bytearray(sector.intersections),
            [remap_node_address(address, sectors, additions) for address in sector.dynamic_addresses],
            sector.dynamic_lengths, sector.dynamic_intersections, sector.trailing_data,
        )
    appended: dict[NodeAddress, list[AppendedLink]] = defaultdict(list)
    effective_segments: list[TrafficPathSegment] = []
    widths: dict[tuple[int, int, int], float] = defaultdict(lambda: 11.0)
    node_positions: dict[NodeAddress, tuple[float, float, float]] = {}
    for key, address in addresses.items():
        node_positions[address] = tuple(value / 8.0 for value in key)
    for key, address, width in grafts:
        node_positions[address] = sectors[address.area].position(address.node)
    link_specs = [(edge.first, addresses[edge.second], edge.width) for edge in edges]
    link_specs.extend(grafts)
    generated_address_keys = {address: key for key, address in addresses.items()}
    for first_key, second_address, width in link_specs:
        first_address = addresses[first_key]
        first_position, second_position = node_positions[first_address], node_positions[second_address]
        if first_address == second_address:
            continue
        if first_address.node < original_vehicle_counts[first_address.area]:
            original_node = sectors[first_address.area].nodes[first_address.node]
            original_base = struct.unpack_from("<h", original_node, 16)[0]
            original_targets = sectors[first_address.area].addresses[original_base:original_base + (original_node[24] & 15)]
            if second_address in original_targets:
                continue
        owner = first_address.area
        owner_sector = updated[owner]
        car_id = len(owner_sector.car_links)
        navigation = pack_navigation_address(owner, car_id)
        owner_sector.car_links.append(make_car_navigation_link(first_position, second_position, second_address, width))
        length = max(1, min(255, round(math.dist(first_position, second_position))))
        appended[first_address].append(AppendedLink(second_address, navigation, length))
        appended[second_address].append(AppendedLink(first_address, navigation, length))
        widths[first_key] = max(widths[first_key], width)
        second_key = generated_address_keys.get(second_address)
        if second_key is not None:
            widths[second_key] = max(widths[second_key], width)
        effective_segments.append(TrafficPathSegment(*first_position, *second_position, width))
    for key, address in addresses.items():
        if key in aliases:
            continue
        updated[address.area].nodes[address.node] = make_path_node(address, key, widths[key], groups[key], len(appended[address]))
    for address, new_links in sorted(appended.items()):
        sector = updated[address.area]
        raw = bytearray(sector.nodes[address.node])
        old_links: list[AppendedLink] = []
        if address.node < original_vehicle_counts[address.area]:
            old = sectors[address.area].nodes[address.node]
            base = struct.unpack_from("<h", old, 16)[0]
            for link_id in range(base, base + (old[24] & 15)):
                old_links.append(AppendedLink(sector.addresses[link_id], sector.navigation[link_id], sector.lengths[link_id], sector.intersections[link_id]))
        links = old_links + new_links
        if len(links) > 15 or len(sector.addresses) + len(links) > MAX_STATIC_LINKS:
            raise ValueError("Generated traffic attachments exceed a native node degree or signed link-index limit.")
        struct.pack_into("<h", raw, 16, len(sector.addresses))
        raw[24] = (raw[24] & 0xF0) | len(links)
        if len(links) > 1:
            raw[24] &= ~0x10
        sector.nodes[address.node] = bytes(raw)
        sector.addresses.extend(link.target for link in links)
        sector.navigation.extend(link.navigation for link in links)
        sector.lengths.extend(link.length for link in links)
        sector.intersections.extend(link.intersection for link in links)
    validate_binary_path_network(updated)
    validate_preserved_retail_paths(sectors, updated, additions)
    validate_added_traffic_links(updated, appended)
    validate_generated_traffic_reachability(updated, original_vehicle_counts)
    files = {}
    for area, sector in sorted(updated.items()):
        payload = write_binary_node_file(sector)
        if payload != originals[area]:
            files[f"nodes{area}.dat"] = payload
    remapped_peds = sum(sector.pedestrian_count for area, sector in sectors.items() if additions.get(area))
    return BinaryPathNetwork(
        files, effective_segments, sum(additions.values()), len(effective_segments), retail_nodes, remapped_peds, rejected,
        [
            f"PASS: {sum(additions.values())} generated vehicle nodes and {len(effective_segments)} bidirectional car links serialized.",
            f"PASS: all original vehicle and pedestrian edges are retained; {remapped_peds} pedestrian indices remapped.",
            f"PASS: {DYNAMIC_LINK_SLOTS} reserved dynamic slots per nonempty sector, native lane references and cross-sector targets validated.",
            f"PASS: {len(grafts) + len(aliases)} grade-compatible connections join generated roads to their retail flood-fill groups.",
        ], updated, source_hashes,
    )


def validate_preserved_retail_paths(
    originals: dict[int, BinaryNodeSector],
    updated: dict[int, BinaryNodeSector],
    additions: dict[int, int],
) -> None:
    for area, old_sector in originals.items():
        sector = updated[area]
        for old_index, old in enumerate(old_sector.nodes):
            new_index = old_index + (additions.get(area, 0) if old_index >= old_sector.vehicle_count else 0)
            new = sector.nodes[new_index]
            if old[:16] != new[:16] or old[18:20] != new[18:20] or old[22:24] != new[22:24] or old[25:] != new[25:]:
                raise ValueError("A retained native node changed position, navigation state or gameplay flags.")
            if (old[24] & 0xE0) != (new[24] & 0xE0):
                raise ValueError("A retained native node changed its switched-off, roadblock or water state.")
            old_base, new_base = struct.unpack_from("<h", old, 16)[0], struct.unpack_from("<h", new, 16)[0]
            old_count, new_count = old[24] & 15, new[24] & 15
            if new_count < old_count:
                raise ValueError("A retained native node lost a navigation edge.")
            for offset in range(old_count):
                old_link, new_link = old_base + offset, new_base + offset
                expected = remap_node_address(old_sector.addresses[old_link], originals, additions)
                if sector.addresses[new_link] != expected:
                    raise ValueError("A retained native edge changed its target.")
                if (sector.navigation[new_link] != old_sector.navigation[old_link]
                        or sector.lengths[new_link] != old_sector.lengths[old_link]
                        or sector.intersections[new_link] != old_sector.intersections[old_link]):
                    raise ValueError("A retained native edge changed lane routing, length or intersection flags.")
        for index, old in enumerate(old_sector.car_links):
            new = sector.car_links[index]
            expected = remap_node_address(NodeAddress(*struct.unpack_from("<HH", old, 4)), originals, additions)
            if old[:4] != new[:4] or old[8:] != new[8:] or struct.unpack_from("<HH", new, 4) != (expected.area, expected.node):
                raise ValueError("A retained car navigation link changed its lane geometry or flags.")


def validate_added_traffic_links(
    sectors: dict[int, BinaryNodeSector],
    appended: dict[NodeAddress, list[AppendedLink]],
) -> None:
    for source, links in appended.items():
        sector = sectors[source.area]
        raw = sector.nodes[source.node]
        base = struct.unpack_from("<h", raw, 16)[0]
        emitted = [(sector.addresses[index], sector.navigation[index]) for index in range(base, base + (raw[24] & 15))]
        for link in links:
            if (link.target, link.navigation) not in emitted:
                raise ValueError("A generated road edge is missing from its decoded source node.")
            target_sector = sectors[link.target.area]
            target_raw = target_sector.nodes[link.target.node]
            target_base = struct.unpack_from("<h", target_raw, 16)[0]
            reverse = [
                (target_sector.addresses[index], target_sector.navigation[index])
                for index in range(target_base, target_base + (target_raw[24] & 15))
            ]
            if (source, link.navigation) not in reverse:
                raise ValueError("A generated road edge has no decoded reciprocal lane link.")
            navigation_area, navigation_id = unpack_navigation_address(link.navigation)
            navigation = sectors[navigation_area].car_links[navigation_id]
            if not (navigation[11] & 7) or not ((navigation[11] >> 3) & 7):
                raise ValueError("A generated two-way street lost one of its lane directions.")
            attached = NodeAddress(*struct.unpack_from("<HH", navigation, 4))
            if attached not in (source, link.target):
                raise ValueError("A generated navigation link is attached to an unrelated node.")
            start_address = link.target if attached == source else source
            first = sectors[start_address.area].position(start_address.node)
            second = sectors[attached.area].position(attached.node)
            direction_x, direction_y = struct.unpack_from("<bb", navigation, 8)
            length = math.hypot(second[0] - first[0], second[1] - first[1])
            alignment = ((second[0] - first[0]) * direction_x + (second[1] - first[1]) * direction_y) / (length * 100.0)
            if alignment < 0.985 or not 1 <= link.length <= 255:
                raise ValueError("A generated lane direction or traversal length is invalid.")


def validate_generated_traffic_reachability(
    sectors: dict[int, BinaryNodeSector],
    original_vehicle_counts: dict[int, int],
) -> None:
    pending = deque(
        NodeAddress(area, index)
        for area, count in sorted(original_vehicle_counts.items())
        for index in range(count)
    )
    reachable = set(pending)
    while pending:
        address = pending.popleft()
        sector = sectors[address.area]
        raw = sector.nodes[address.node]
        base = struct.unpack_from("<h", raw, 16)[0]
        for link_index in range(base, base + (raw[24] & 15)):
            target = sector.addresses[link_index]
            target_sector = sectors[target.area]
            if target.node < target_sector.vehicle_count and target not in reachable:
                reachable.add(target)
                pending.append(target)
    for area, sector in sectors.items():
        for index in range(original_vehicle_counts.get(area, 0), sector.vehicle_count):
            if NodeAddress(area, index) not in reachable:
                raise ValueError(
                    f"Generated vehicle node {area}:{index} has no decoded route to the retained retail road network."
                )
