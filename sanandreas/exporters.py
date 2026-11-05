from __future__ import annotations

import html
import json
import math
import shlex
from pathlib import Path

from .model import GeneratedWorld, Vec3
from .voice import make_ffmpeg_filter


def export_world(world: GeneratedWorld, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    game_data = output_directory / "modloader" / "SeedAndreas" / "data" / "maps" / "seedandreas"
    game_data.mkdir(parents=True, exist_ok=True)
    report_directory = output_directory / "report"
    report_directory.mkdir(parents=True, exist_ok=True)

    write_json(output_directory / "seed_andreas_seed.json", world.to_dict())
    write_ide(game_data / "seedandreas_generated.ide", world)
    write_ipl(game_data / "seedandreas_generated.ipl", world)
    write_paths(game_data / "seedandreas_paths.prototype.json", world)
    write_gameplay_data(game_data, world)
    write_voice_manifest(output_directory, world)
    write_radar_svg(report_directory / "radar.svg", world)
    write_report(report_directory / "index.html", world)
    write_build_notes(output_directory / "INSTALLATION_BLOCKED.txt")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_ide(path: Path, world: GeneratedWorld) -> None:
    unique_models = sorted({(building.model_id, building.model_name) for building in world.buildings})
    lines = ["objs"]
    for model_id, model_name in unique_models:
        lines.append(f"{model_id}, {model_name}, seedandreas, 1, 300.0, 0")
    lines.append("end")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ipl(path: Path, world: GeneratedWorld) -> None:
    lines = ["inst"]
    for building in world.buildings:
        half_yaw = math.radians(building.yaw_degrees) * 0.5
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)
        lines.append(
            f"{building.model_id}, {building.model_name}, 0, "
            f"{building.position.x:.4f}, {building.position.y:.4f}, {building.position.z:.4f}, "
            f"0.000000, 0.000000, {qz:.6f}, {qw:.6f}, -1"
        )
    lines.append("end")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paths(path: Path, world: GeneratedWorld) -> None:
    node_lookup = {node.node_id: node for node in world.road_nodes}
    payload = {
        "format": "SeedAndreas path prototype; not GTA SA binary path data",
        "nodes": [
            {
                "id": node.node_id,
                "city": node.city_id,
                "position": node.position.__dict__,
                "kind": node.kind,
                "mission_anchor": node.mission_anchor_id,
            }
            for node in world.road_nodes
        ],
        "links": [
            {
                "id": edge.edge_id,
                "start": edge.start_node,
                "end": edge.end_node,
                "width": edge.width,
                "lanes": edge.lanes,
                "road_type": edge.road_type,
                "mission_critical": edge.mission_critical,
                "length": round(node_lookup[edge.start_node].position.distance_2d(node_lookup[edge.end_node].position), 3),
            }
            for edge in world.road_edges
        ],
    }
    write_json(path, payload)


def write_gameplay_data(game_data: Path, world: GeneratedWorld) -> None:
    enex_payload = [
        {
            "id": item.entrance_id,
            "anchor": item.anchor_id,
            "exterior": item.exterior.__dict__,
            "interior": item.interior.__dict__,
            "interior_id": item.interior_id,
            "facing": item.facing_degrees,
            "name": item.name,
        }
        for item in world.entrances
    ]
    checkpoint_payload = [item.__dict__ | {"position": item.position.__dict__} for item in world.checkpoints]
    cull_payload = [item.__dict__ | {"center": item.center.__dict__, "size": item.size.__dict__} for item in world.cull_zones]
    write_json(game_data / "enex.prototype.json", enex_payload)
    write_json(game_data / "mission_checkpoints.prototype.json", checkpoint_payload)
    write_json(game_data / "cull.prototype.json", cull_payload)


def write_voice_manifest(output_directory: Path, world: GeneratedWorld) -> None:
    voice_directory = output_directory / "voices"
    voice_directory.mkdir(parents=True, exist_ok=True)
    write_json(voice_directory / "voice_variations.json", [voice.__dict__ for voice in world.voices])

    command_lines = [
        "# Accent profiles select alternate recorded/TTS voice banks.",
        "# The ffmpeg filters below implement only pitch, tempo, and light grit.",
        "# Add source_audio paths to the manifest before rendering.",
        "",
    ]
    for voice in world.voices:
        if not voice.varied:
            continue
        source_name = voice.source_audio or f"input/{voice.line_id}.wav"
        output_name = f"output/{voice.line_id}.wav"
        audio_filter = make_ffmpeg_filter(voice)
        command_lines.append(
            "ffmpeg -y -i "
            + shlex.quote(source_name)
            + " -af "
            + shlex.quote(audio_filter)
            + " "
            + shlex.quote(output_name)
        )
    (voice_directory / "render_voice_variations.sh").write_text("\n".join(command_lines) + "\n", encoding="utf-8")


def world_bounds(world: GeneratedWorld) -> tuple[float, float, float, float]:
    points = [node.position for node in world.road_nodes]
    minimum_x = min(point.x for point in points) - 100.0
    maximum_x = max(point.x for point in points) + 100.0
    minimum_y = min(point.y for point in points) - 100.0
    maximum_y = max(point.y for point in points) + 100.0
    return minimum_x, minimum_y, maximum_x, maximum_y


def project(point: Vec3, bounds: tuple[float, float, float, float], size: int = 1400) -> tuple[float, float]:
    minimum_x, minimum_y, maximum_x, maximum_y = bounds
    scale_x = size / (maximum_x - minimum_x)
    scale_y = size / (maximum_y - minimum_y)
    scale = min(scale_x, scale_y)
    offset_x = (size - (maximum_x - minimum_x) * scale) * 0.5
    offset_y = (size - (maximum_y - minimum_y) * scale) * 0.5
    x = offset_x + (point.x - minimum_x) * scale
    y = size - (offset_y + (point.y - minimum_y) * scale)
    return x, y


def write_radar_svg(path: Path, world: GeneratedWorld) -> None:
    bounds = world_bounds(world)
    node_lookup = {node.node_id: node for node in world.road_nodes}
    size = 1400
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}">',
        '<rect width="1400" height="1400" fill="#9eb2a0"/>',
        '<g opacity="0.95">',
    ]
    for building in world.buildings:
        x, y = project(building.position, bounds, size)
        width = max(2.0, building.width * 0.09)
        depth = max(2.0, building.depth * 0.09)
        parts.append(
            f'<rect x="{x - width / 2:.2f}" y="{y - depth / 2:.2f}" width="{width:.2f}" height="{depth:.2f}" '
            f'fill="#9a846b" transform="rotate({-building.yaw_degrees:.2f} {x:.2f} {y:.2f})"/>'
        )
    parts.append("</g><g fill=\"none\" stroke-linecap=\"round\">")
    road_colours = {"street": "#d8c6a2", "collector": "#decba1", "arterial": "#e5d2a7", "mission_access": "#d2bd94", "mission_corridor": "#d9bd84", "highway": "#d6b57b", "intercity_highway": "#cba66b"}
    for edge in world.road_edges:
        start = project(node_lookup[edge.start_node].position, bounds, size)
        end = project(node_lookup[edge.end_node].position, bounds, size)
        road_width = max(1.2, edge.width * 0.12)
        colour = road_colours[edge.road_type]
        parts.append(f'<path d="M {start[0]:.2f} {start[1]:.2f} L {end[0]:.2f} {end[1]:.2f}" stroke="#725f4f" stroke-width="{road_width + 1.5:.2f}"/>')
        parts.append(f'<path d="M {start[0]:.2f} {start[1]:.2f} L {end[0]:.2f} {end[1]:.2f}" stroke="{colour}" stroke-width="{road_width:.2f}"/>')
    parts.append("</g><g>")
    for anchor in world.anchors:
        x, y = project(anchor.generated_position, bounds, size)
        colour = "#ed405f" if anchor.policy == "protected" else "#ffad33"
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="{colour}" stroke="#33161c" stroke-width="1.4"/>')
        parts.append(f'<text x="{x + 8:.2f}" y="{y - 7:.2f}" font-family="Arial" font-size="11" fill="#202020">{html.escape(anchor.anchor_id)}</text>')
    parts.append("</g></svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(path: Path, world: GeneratedWorld) -> None:
    errors = [issue for issue in world.validation_issues if issue.severity == "error"]
    status = "STATIC VALIDATION PASSED" if not errors else "STATIC VALIDATION FAILED"
    status_class = "pass" if not errors else "fail"
    varied_count = sum(voice.varied for voice in world.voices)
    issue_rows = "".join(
        f"<tr><td>{html.escape(issue.severity)}</td><td>{html.escape(issue.code)}</td><td>{html.escape(issue.subject)}</td><td>{html.escape(issue.message)}</td></tr>"
        for issue in world.validation_issues
    ) or '<tr><td colspan="4">No static validation issues.</td></tr>'
    anchor_rows = "".join(
        f"<tr><td>{html.escape(anchor.anchor_id)}</td><td>{html.escape(anchor.mission)}</td><td>{html.escape(anchor.policy)}</td>"
        f"<td>{anchor.generated_position.x:.1f}, {anchor.generated_position.y:.1f}, {anchor.generated_position.z:.1f}</td>"
        f"<td>{html.escape(', '.join(anchor.required_tags))}</td></tr>"
        for anchor in world.anchors
    )
    voice_rows = "".join(
        f"<tr><td>{html.escape(voice.line_id)}</td><td>{'yes' if voice.varied else 'no'}</td><td>{html.escape(voice.accent_profile)}</td>"
        f"<td>{voice.pitch_semitones:+.2f}</td><td>{voice.tempo:.3f}</td><td>{html.escape(voice.text)}</td></tr>"
        for voice in world.voices
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Seed Andreas seed {world.seed}</title>
<style>
:root {{ color-scheme: dark; --pink:#f54f75; --orange:#ffad33; --ink:#121216; --panel:#202029; --line:#383844; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--ink); color:#eee; font:15px/1.5 Segoe UI,Arial,sans-serif; }}
header {{ padding:28px 34px; background:linear-gradient(110deg,#1f2541,#6f214d); border-bottom:1px solid #744260; }}
h1 {{ margin:0; font-size:34px; }} .subtitle {{ opacity:.78; margin-top:5px; }}
main {{ max-width:1450px; margin:auto; padding:24px; }} .grid {{ display:grid; grid-template-columns:minmax(420px,1.2fr) minmax(360px,.8fr); gap:20px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:18px; box-shadow:0 14px 30px #0005; }}
.radar {{ width:100%; aspect-ratio:1; border-radius:8px; background:#9eb2a0; }}
.stats {{ display:grid; grid-template-columns:repeat(2,1fr); gap:10px; margin-bottom:18px; }} .stat {{ padding:12px; background:#191920; border-radius:7px; }}
.stat b {{ display:block; font-size:24px; color:#fff; }} .status {{ padding:12px; border-radius:7px; font-weight:700; letter-spacing:.04em; }}
.pass {{ background:#174832; color:#83e3af; }} .fail {{ background:#5d2028; color:#ff9aaa; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }} th,td {{ text-align:left; border-bottom:1px solid var(--line); padding:8px; vertical-align:top; }} th {{ color:#ffb1c3; }}
h2 {{ margin:0 0 14px; }} .wide {{ margin-top:20px; overflow:auto; }} .warning {{ color:#ffc66d; }}
@media(max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header><h1>Seed Andreas</h1><div class="subtitle">Deterministic mission-aware world prototype · seed {world.seed} · generator {world.generator_version}</div></header>
<main>
<div class="grid">
  <section class="panel"><h2>Generated radar</h2><img class="radar" src="radar.svg" alt="Generated road and mission anchor map"></section>
  <section class="panel">
    <div class="stats">
      <div class="stat"><b>{len(world.road_nodes)}</b>road nodes</div><div class="stat"><b>{len(world.road_edges)}</b>road links</div>
      <div class="stat"><b>{len(world.buildings)}</b>placements</div><div class="stat"><b>{len(world.anchors)}</b>mission anchors</div>
      <div class="stat"><b>{len(world.entrances)}</b>ENEX pairs</div><div class="stat"><b>{varied_count}/{len(world.voices)}</b>varied voices</div>
    </div>
    <div class="status {status_class}">{status}</div>
    <p class="warning">This certifies the prototype's internal graph only. It does not certify GTA SA story completion, binary path data, SCM rewriting, DFF/COL assets, or runtime behavior.</p>
    <p>Protected mission regions remain fixed. Relocatable anchors move deterministically, and their ENEX, CULL and checkpoint dependants move with them.</p>
  </section>
</div>
<section class="panel wide"><h2>Mission contracts</h2><table><thead><tr><th>Anchor</th><th>Mission</th><th>Policy</th><th>Generated position</th><th>Required semantics</th></tr></thead><tbody>{anchor_rows}</tbody></table></section>
<section class="panel wide"><h2>Occasional voice mutations</h2><table><thead><tr><th>Line</th><th>Varied</th><th>Accent bank</th><th>Pitch</th><th>Tempo</th><th>Original prototype line</th></tr></thead><tbody>{voice_rows}</tbody></table></section>
<section class="panel wide"><h2>Validation</h2><table><thead><tr><th>Severity</th><th>Code</th><th>Subject</th><th>Message</th></tr></thead><tbody>{issue_rows}</tbody></table></section>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def write_build_notes(path: Path) -> None:
    path.write_text(
        "Seed Andreas prototype output is intentionally not install-ready.\n\n"
        "The IDE/IPL files prove deterministic placement export, but the referenced DFF/TXD/COL assets are placeholders. "
        "Path, ENEX, CULL, checkpoint and voice outputs are explicit prototype JSON rather than retail GTA SA binary/script patches. "
        "Do not copy this package into a game installation and treat it as story-certified.\n",
        encoding="utf-8",
    )
