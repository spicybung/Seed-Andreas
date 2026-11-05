from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import hypot
from typing import Any


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float = 0.0

    def distance_2d(self, other: "Vec3") -> float:
        return hypot(self.x - other.x, self.y - other.y)

    def moved(self, dx: float, dy: float, dz: float = 0.0) -> "Vec3":
        return Vec3(self.x + dx, self.y + dy, self.z + dz)


@dataclass
class City:
    city_id: str
    display_name: str
    center: Vec3
    half_width: float
    half_height: float
    style: str


@dataclass
class MissionAnchor:
    anchor_id: str
    mission: str
    city_id: str
    original_position: Vec3
    generated_position: Vec3
    protected_radius: float
    policy: str
    required_tags: list[str]


@dataclass
class RoadNode:
    node_id: str
    city_id: str
    position: Vec3
    kind: str = "junction"
    mission_anchor_id: str | None = None


@dataclass
class RoadEdge:
    edge_id: str
    start_node: str
    end_node: str
    road_type: str
    width: float
    lanes: int
    mission_critical: bool = False


@dataclass
class Building:
    building_id: str
    city_id: str
    model_name: str
    model_id: int
    position: Vec3
    width: float
    depth: float
    height: float
    yaw_degrees: float
    district: str


@dataclass
class Entrance:
    entrance_id: str
    anchor_id: str
    exterior: Vec3
    interior: Vec3
    interior_id: int
    facing_degrees: float
    name: str


@dataclass
class Checkpoint:
    checkpoint_id: str
    anchor_id: str
    mission: str
    position: Vec3
    radius: float
    vehicle_class: str
    maximum_road_distance: float


@dataclass
class CullZone:
    cull_id: str
    anchor_id: str
    center: Vec3
    size: Vec3
    purpose: str
    flags: int


@dataclass
class VoiceVariation:
    line_id: str
    speaker: str
    text: str
    varied: bool
    accent_profile: str
    voice_bank: str
    pitch_semitones: float
    tempo: float
    grit: float
    source_audio: str | None = None


@dataclass
class ValidationIssue:
    severity: str
    code: str
    subject: str
    message: str


@dataclass
class GeneratedWorld:
    generator_version: str
    seed: int
    profile: str
    cities: list[City] = field(default_factory=list)
    anchors: list[MissionAnchor] = field(default_factory=list)
    road_nodes: list[RoadNode] = field(default_factory=list)
    road_edges: list[RoadEdge] = field(default_factory=list)
    buildings: list[Building] = field(default_factory=list)
    entrances: list[Entrance] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    cull_zones: list[CullZone] = field(default_factory=list)
    voices: list[VoiceVariation] = field(default_factory=list)
    validation_issues: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
