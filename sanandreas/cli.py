from __future__ import annotations

import argparse
import json
import sys
import tempfile
import webbrowser
from pathlib import Path

from .generator import generate_world
from .validator import validate_world
from .exporters import export_world
from .los_santos import generate_los_santos_world, write_los_santos_mod_package
from .gta_integration import (
    DEFAULT_GAME_DIRECTORY,
    install_mod_package,
    launch_game,
    scan_game_map,
    uninstall_mod_package,
    validate_game_directory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seed-andreas", description="Generate a deterministic mission-aware San Andreas world prototype.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate", help="Generate and validate a seed.")
    generate_parser.add_argument("--seed", type=int, default=105839204)
    generate_parser.add_argument("--profile", default="story_safe")
    generate_parser.add_argument("--voice-chance", type=float, default=0.28)
    generate_parser.add_argument("--output", type=Path)
    generate_parser.add_argument("--open-report", action="store_true")

    scan_parser = subparsers.add_parser("scan-game", help="Scan the installed GTA SA IDE and IPL files.")
    scan_parser.add_argument("--game", type=Path, default=DEFAULT_GAME_DIRECTORY)

    install_parser = subparsers.add_parser("install", help="Generate and install a seeded custom-mesh Los Santos replacement.")
    install_parser.add_argument("--game", type=Path, default=DEFAULT_GAME_DIRECTORY)
    install_parser.add_argument("--seed", type=int, default=105839204)
    install_parser.add_argument("--staging", type=Path, help="Parent directory for a fresh temporary build directory.")
    install_parser.add_argument("--launch", action="store_true")

    uninstall_parser = subparsers.add_parser("uninstall", help="Move the installed prototype to a recoverable backup.")
    uninstall_parser.add_argument("--game", type=Path, default=DEFAULT_GAME_DIRECTORY)

    launch_parser = subparsers.add_parser("launch", help="Launch the configured GTA SA installation.")
    launch_parser.add_argument("--game", type=Path, default=DEFAULT_GAME_DIRECTORY)
    return parser


def main(arguments: list[str] | None = None) -> int:
    parser = build_parser()
    options = parser.parse_args(arguments)
    if hasattr(options, "game"):
        options.game = options.game.expanduser().resolve()
    if options.command == "generate":
        return generate_command(options)
    if options.command == "scan-game":
        return scan_game_command(options)
    if options.command == "install":
        return install_command(options)
    if options.command == "uninstall":
        return uninstall_command(options)
    if options.command == "launch":
        launch_game(options.game)
        return 0
    parser.error("Unknown command.")
    return 2


def generate_command(options: argparse.Namespace) -> int:
    if not 0.0 <= options.voice_chance <= 1.0:
        print("--voice-chance must be between 0 and 1.", file=sys.stderr)
        return 2

    output_directory = options.output or Path("build") / f"seed_{options.seed}"
    world = generate_world(options.seed, options.profile, options.voice_chance)
    world.validation_issues = validate_world(world)
    export_world(world, output_directory)

    errors = [issue for issue in world.validation_issues if issue.severity == "error"]
    varied_voices = sum(voice.varied for voice in world.voices)
    summary = {
        "seed": world.seed,
        "output": str(output_directory.resolve()),
        "road_nodes": len(world.road_nodes),
        "road_edges": len(world.road_edges),
        "buildings": len(world.buildings),
        "anchors": len(world.anchors),
        "voice_variations": varied_voices,
        "static_validation": "PASS" if not errors else "FAIL",
        "errors": len(errors),
    }
    print(json.dumps(summary, indent=2))
    if options.open_report:
        webbrowser.open((output_directory / "report" / "index.html").resolve().as_uri())
    return 1 if errors else 0


def scan_game_command(options: argparse.Namespace) -> int:
    problems = validate_game_directory(options.game, require_mod_loader=True)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    scan = scan_game_map(options.game)
    summary = {
        "game_directory": scan.game_directory,
        "ide_files": scan.ide_files_scanned,
        "ipl_files": scan.ipl_files_scanned,
        "binary_ipl_files_skipped": scan.binary_ipl_files_skipped,
        "object_models": len(scan.models),
        "text_ipl_placements": len(scan.placements),
        "mod_loader": "detected",
    }
    print(json.dumps(summary, indent=2))
    return 0


def install_command(options: argparse.Namespace) -> int:
    problems = validate_game_directory(options.game, require_mod_loader=True)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    scan = scan_game_map(options.game)
    world = generate_los_santos_world(scan, options.seed)
    mode = "full_los_santos_seed"
    if options.staging is not None:
        options.staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"SeedAndreas_{options.seed}_", dir=options.staging) as temporary_directory:
        staged_mod = write_los_santos_mod_package(options.game, Path(temporary_directory), world)
        installed, backup = install_mod_package(options.game, staged_mod, world.manifest)
    summary = {
        "seed": options.seed,
        "mode": mode,
        "installed": str(installed),
        "previous_version_backup": str(backup) if backup else None,
        "custom_dff_models": world.manifest.custom_models,
        "custom_buildings": world.manifest.custom_buildings,
        "render_vertices": world.manifest.render_vertices,
        "render_triangles": world.manifest.render_triangles,
        "collision_triangles": world.manifest.collision_triangles,
        "model_ids": [world.manifest.model_id_first, world.manifest.model_id_last],
    }
    summary.update({
        "seed_fingerprint": world.manifest.seed_fingerprint,
        "bounds": world.manifest.bounds,
        "district_tiles": world.manifest.district_tiles,
        "text_ipls_overridden": len(world.manifest.text_ipls_overridden),
        "stream_ipls_overridden": len(world.manifest.stream_ipls_overridden),
        "collision_archives_overridden": world.manifest.collision_archives_overridden,
        "texture_dictionaries_overridden": world.manifest.texture_dictionaries_overridden,
        "radar_tiles_overridden": len(world.manifest.radar_tiles_overridden),
        "traffic_road_segments": world.manifest.traffic_path_segments_preserved,
        "traffic_path_nodes_changed": world.manifest.traffic_path_nodes_changed,
        "traffic_path_nodes_added": world.manifest.traffic_path_nodes_added,
        "traffic_car_links_added": world.manifest.traffic_car_links_added,
        "pedestrian_indices_remapped": world.manifest.traffic_pedestrian_nodes_remapped,
        "neighborhood_buildings": world.manifest.neighborhood_buildings,
        "building_conditions": world.manifest.building_conditions,
        "building_styles": world.manifest.building_styles,
        "dwelling_units": world.manifest.dwelling_units,
        "water": world.manifest.water,
        "mission_regions": world.manifest.mission_regions,
        "railway_tracks_preserved": world.manifest.railway_tracks_preserved,
        "traffic_path_nodes_preserved": world.manifest.traffic_path_nodes_preserved,
        "traffic_path_files_overridden": world.manifest.traffic_path_files_overridden,
        "mission_safe_instances_preserved": world.manifest.mission_safe_instances_preserved,
        "terrain_elevation_range": [
            world.manifest.terrain_elevation_minimum,
            world.manifest.terrain_elevation_maximum,
        ],
        "ground_rules": world.manifest.ground_rules,
        "validation": world.manifest.validation,
    })
    print(json.dumps(summary, indent=2))
    if options.launch:
        launch_game(options.game)
    return 0


def uninstall_command(options: argparse.Namespace) -> int:
    moved_to = uninstall_mod_package(options.game)
    if moved_to is None:
        print("Seed Andreas is not installed in this Mod Loader directory.")
    else:
        print(f"Seed Andreas was moved to the recoverable backup: {moved_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
