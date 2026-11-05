from __future__ import annotations

from dataclasses import dataclass

from .model import Vec3


@dataclass(frozen=True)
class AnchorContract:
    anchor_id: str
    mission: str
    city_id: str
    original_position: Vec3
    protected_radius: float
    policy: str
    relocation_radius: float
    required_tags: tuple[str, ...]


@dataclass(frozen=True)
class EntranceContract:
    entrance_id: str
    anchor_id: str
    exterior_offset: Vec3
    interior_position: Vec3
    interior_id: int
    facing_degrees: float
    name: str


@dataclass(frozen=True)
class CheckpointContract:
    checkpoint_id: str
    anchor_id: str
    mission: str
    offset: Vec3
    radius: float
    vehicle_class: str
    maximum_road_distance: float


@dataclass(frozen=True)
class CullContract:
    cull_id: str
    anchor_id: str
    offset: Vec3
    size: Vec3
    purpose: str
    flags: int


ANCHOR_CONTRACTS = (
    AnchorContract("grove_street", "Sweet & Kendl", "ls", Vec3(2490.0, -1668.0, 13.3), 115.0, "protected", 0.0, ("residential", "cul_de_sac", "road_access", "ped_access")),
    AnchorContract("ls_airport", "Learning to Fly", "ls", Vec3(1685.0, -2334.0, 13.5), 180.0, "protected", 0.0, ("airport", "runway", "aircraft_access")),
    AnchorContract("unity_station", "Wrong Side of the Tracks", "ls", Vec3(1742.0, -1863.0, 13.6), 90.0, "relocatable", 95.0, ("rail_access", "motorcycle_route", "station")),
    AnchorContract("doherty_garage", "Wear Flowers in Your Hair", "sf", Vec3(-2027.0, 154.0, 29.0), 105.0, "relocatable", 120.0, ("garage", "road_access", "open_yard")),
    AnchorContract("wang_cars", "Zeroing In", "sf", Vec3(-1955.0, 300.0, 35.5), 85.0, "relocatable", 100.0, ("showroom", "road_access")),
    AnchorContract("four_dragons", "The Four Dragons", "lv", Vec3(2018.0, 1007.0, 10.8), 145.0, "protected", 0.0, ("casino", "road_access", "interior_entrance")),
    AnchorContract("caligulas", "Freefall", "lv", Vec3(2195.0, 1677.0, 12.4), 140.0, "relocatable", 110.0, ("casino", "road_access", "interior_entrance")),
    AnchorContract("verdant_meadows", "Green Goo", "lv", Vec3(404.0, 2530.0, 16.6), 220.0, "protected", 0.0, ("airstrip", "aircraft_access", "hangar")),
)


ENTRANCE_CONTRACTS = (
    EntranceContract("enex_johnson_house", "grove_street", Vec3(3.5, -8.0, 0.2), Vec3(2496.0, -1692.0, 1014.7), 3, 180.0, "Johnson House"),
    EntranceContract("enex_four_dragons", "four_dragons", Vec3(7.0, -18.0, 0.2), Vec3(2019.0, 1017.0, 996.9), 10, 0.0, "Four Dragons Casino"),
    EntranceContract("enex_caligulas", "caligulas", Vec3(-10.0, -15.0, 0.2), Vec3(2233.8, 1712.2, 1011.8), 1, 180.0, "Caligula's Casino"),
)


CHECKPOINT_CONTRACTS = (
    CheckpointContract("cp_sweet_kendl_home", "grove_street", "Sweet & Kendl", Vec3(-25.0, 18.0, 0.0), 4.0, "bicycle", 30.0),
    CheckpointContract("cp_wrong_side_start", "unity_station", "Wrong Side of the Tracks", Vec3(22.0, -12.0, 0.0), 5.0, "motorcycle", 32.0),
    CheckpointContract("cp_doherty_garage", "doherty_garage", "Wear Flowers in Your Hair", Vec3(18.0, 4.0, 0.0), 5.0, "car", 34.0),
    CheckpointContract("cp_four_dragons", "four_dragons", "The Four Dragons", Vec3(0.0, -30.0, 0.0), 5.0, "car", 34.0),
    CheckpointContract("cp_verdant_runway", "verdant_meadows", "Learning to Fly", Vec3(80.0, 0.0, 8.0), 12.0, "aircraft", 120.0),
)


CULL_CONTRACTS = (
    CullContract("cull_johnson_shelter", "grove_street", Vec3(4.0, -8.0, 4.0), Vec3(18.0, 16.0, 9.0), "interior_exterior_shell", 0x0008),
    CullContract("cull_four_dragons_entry", "four_dragons", Vec3(7.0, -18.0, 5.0), Vec3(24.0, 18.0, 12.0), "no_rain_area", 0x0008),
)


MISSION_CONNECTIONS = (
    ("grove_street", "unity_station", "mission_corridor"),
    ("unity_station", "ls_airport", "mission_corridor"),
    ("doherty_garage", "wang_cars", "mission_corridor"),
    ("four_dragons", "caligulas", "mission_corridor"),
    ("caligulas", "verdant_meadows", "highway"),
)


VOICE_LINES = (
    ("ambient_ls_001", "street_local", "That road was not here yesterday.", "ambient", None),
    ("ambient_ls_002", "street_local", "This whole block feels different.", "ambient", None),
    ("mission_route_001", "mission_dispatch", "Stay on the road and follow the next marker.", "route", None),
    ("mission_route_002", "mission_dispatch", "The entrance moved, but the job is still the same.", "route", None),
    ("ambient_sf_001", "mechanic", "These hills keep changing their mind.", "ambient", None),
    ("ambient_sf_002", "dock_worker", "The bay looks strange tonight.", "ambient", None),
    ("ambient_lv_001", "casino_guest", "I swear this casino was across town.", "ambient", None),
    ("ambient_lv_002", "desert_driver", "Every shortcut out here is a gamble.", "ambient", None),
)
