from __future__ import annotations

import queue
import random
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from sanandrais.generator import GENERATOR_VERSION
from sanandrais.gta_integration import (
    DEFAULT_GAME_DIRECTORY,
    install_mod_package,
    launch_game,
    scan_game_map,
    uninstall_mod_package,
    validate_game_directory,
    TrafficPathSegment,
)
from sanandrais.los_santos import LosSantosWorld, generate_los_santos_world, procedural_land_outline, write_los_santos_mod_package
from sanandrais.textures import TextureImage, radar_material_color, rasterize_radar_triangle, render_geometry_radar_tile
from sanandrais.street_graph import StreetBlock, StreetGraph, generate_hierarchical_street_graph


WOOD = "#714022"
WOOD_DARK = "#3b2114"
PANEL = "#b8b99d"
PANEL_LIGHT = "#d8d8be"
PANEL_DARK = "#686b59"
INK = "#171b17"
BLUE = "#183cff"
WATER = "#338f8d"
LAND = "#d7c78f"
ROAD = "#4f554f"
PARK = "#1f8d42"
ACTIVE = "#d8d08c"
PIXEL_FONT = ("Fixedsys", 12)
PIXEL_BOLD = ("Fixedsys", 15, "bold")


def inset_block(block: StreetBlock, scale: float) -> tuple[tuple[float, float], ...]:
    center_x = sum(point.x for point in block.points) / len(block.points)
    center_y = sum(point.y for point in block.points) / len(block.points)
    return tuple(
        (
            center_x + (point.x - center_x) * scale,
            center_y + (point.y - center_y) * scale,
        )
        for point in block.points
    )


def point_inside_polygon(
    x: float,
    y: float,
    polygon: list[tuple[float, float]],
) -> bool:
    inside = False
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        if (first[1] > y) == (second[1] > y):
            continue
        intersection_x = first[0] + (y - first[1]) * (second[0] - first[0]) / (second[1] - first[1])
        if x < intersection_x:
            inside = not inside
    return inside


class SeedAndreasLauncher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"Seed Andreas {GENERATOR_VERSION} — Map Select")
        self.geometry("930x730")
        self.minsize(820, 650)
        self.configure(background=WOOD)

        self.game_directory_variable = tk.StringVar(value=str(DEFAULT_GAME_DIRECTORY))
        self.seed_variable = tk.StringVar(value="105839204")
        self.status_variable = tk.StringVar(value="SELECT A SEED, THEN BUILD THE MAP")
        self.result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.preview_traffic_segments: list[TrafficPathSegment] = []
        self.preview_world: LosSantosWorld | None = None
        self.preview_photo: tk.PhotoImage | None = None
        self.preview_photo_size = 0
        self.busy = False

        self.create_styles()
        self.create_widgets()
        self.seed_variable.trace_add("write", self.on_seed_changed)
        self.game_directory_variable.trace_add("write", self.on_seed_changed)
        self.protocol("WM_DELETE_WINDOW", self.close_launcher)
        self.draw_seed_preview()
        self.after(100, self.poll_results)

    def create_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Retro.Horizontal.TProgressbar",
            troughcolor=PANEL_DARK,
            background=BLUE,
            bordercolor=INK,
            lightcolor=BLUE,
            darkcolor=BLUE,
        )

    def create_widgets(self) -> None:
        self.draw_wood_background()
        shell = tk.Frame(self, background=PANEL, highlightbackground=INK, highlightthickness=3)
        shell.place(relx=0.5, rely=0.5, anchor=tk.CENTER, relwidth=0.92, relheight=0.90)

        header = self.beveled_frame(shell, PANEL_LIGHT)
        header.pack(fill=tk.X, padx=16, pady=(15, 9))
        tk.Label(header, text="SEED ANDREAS", background=INK, foreground="#46e848", font=PIXEL_BOLD, padx=10, pady=6).pack(side=tk.LEFT, padx=8, pady=8)
        tk.Label(header, text="MAP SELECT", background=INK, foreground=BLUE, font=("Fixedsys", 20, "bold"), padx=18, pady=7).pack(side=tk.LEFT, expand=True, padx=10, pady=8)
        tk.Label(header, text=f"V{GENERATOR_VERSION}", background=PANEL_LIGHT, foreground=INK, font=PIXEL_FONT).pack(side=tk.RIGHT, padx=10)

        body = tk.Frame(shell, background=PANEL)
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=6)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        preview_panel = self.beveled_frame(body, PANEL_DARK)
        preview_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.preview_canvas = tk.Canvas(
            preview_panel,
            width=500,
            height=410,
            background=WATER,
            highlightbackground=INK,
            highlightthickness=4,
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=13, pady=13)

        controls = self.beveled_frame(body, PANEL_LIGHT)
        controls.grid(row=0, column=1, sticky="nsew")
        controls.columnconfigure(0, weight=1)

        tk.Label(controls, text="WORLD SEED", background=PANEL_LIGHT, foreground=INK, font=PIXEL_BOLD).grid(row=0, column=0, pady=(18, 6))
        seed_row = tk.Frame(controls, background=PANEL_LIGHT)
        seed_row.grid(row=1, column=0, sticky="ew", padx=18)
        seed_row.columnconfigure(1, weight=1)
        self.previous_button = self.retro_button(seed_row, "◀", lambda: self.change_seed(-1), width=3)
        self.previous_button.grid(row=0, column=0, padx=(0, 6))
        self.seed_entry = tk.Entry(
            seed_row,
            textvariable=self.seed_variable,
            justify=tk.CENTER,
            background=INK,
            foreground="#f0e8bf",
            insertbackground="#f0e8bf",
            relief=tk.FLAT,
            font=("Courier New", 14, "bold"),
        )
        self.seed_entry.grid(row=0, column=1, sticky="ew", ipady=6)
        self.next_button = self.retro_button(seed_row, "▶", lambda: self.change_seed(1), width=3)
        self.next_button.grid(row=0, column=2, padx=(6, 0))
        self.random_button = self.retro_button(controls, "RANDOM MAP", self.randomize_seed)
        self.random_button.grid(row=2, column=0, sticky="ew", padx=18, pady=8)

        tk.Label(controls, text="LOS SANTOS / FULL", background=PANEL_LIGHT, foreground=INK, font=PIXEL_FONT).grid(row=3, column=0, pady=(9, 2))
        tk.Label(
            controls,
            text="ACTIVE MISSION ROAD GRAPH\nROAD-CARVED BLOCKS\nHEIGHTFIELD + ISLAND MASK\nGEOMETRY + RADAR",
            background=PANEL_LIGHT,
            foreground="#304530",
            justify=tk.LEFT,
            font=("Courier New", 9, "bold"),
        ).grid(row=4, column=0, pady=6)

        self.install_button = self.retro_button(controls, "OK — BUILD + INSTALL", self.begin_install, accent=True)
        self.install_button.grid(row=5, column=0, sticky="ew", padx=18, pady=(12, 6))
        self.scan_button = self.retro_button(controls, "CHECK GTA INSTALL", self.begin_scan)
        self.scan_button.grid(row=6, column=0, sticky="ew", padx=18, pady=4)
        self.launch_button = self.retro_button(controls, "PLAY GTA SAN ANDREAS", self.launch_installed_game)
        self.launch_button.grid(row=7, column=0, sticky="ew", padx=18, pady=4)
        self.uninstall_button = self.retro_button(controls, "REMOVE GENERATED MAP", self.uninstall)
        self.uninstall_button.grid(row=8, column=0, sticky="ew", padx=18, pady=4)

        self.progress = ttk.Progressbar(controls, mode="indeterminate", style="Retro.Horizontal.TProgressbar")
        self.progress.grid(row=9, column=0, sticky="ew", padx=18, pady=(13, 16))

        path_panel = self.beveled_frame(shell, PANEL_LIGHT)
        path_panel.pack(fill=tk.X, padx=18, pady=(8, 7))
        tk.Label(path_panel, text="GTA SA FOLDER", background=PANEL_LIGHT, foreground=INK, font=PIXEL_FONT).pack(side=tk.LEFT, padx=(10, 8), pady=10)
        self.path_entry = tk.Entry(path_panel, textvariable=self.game_directory_variable, background="#ebe8cf", foreground=INK, font=("Courier New", 9))
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self.browse_button = self.retro_button(path_panel, "...", self.browse_game, width=4)
        self.browse_button.pack(side=tk.LEFT, padx=8)

        status = tk.Label(
            shell,
            textvariable=self.status_variable,
            background=INK,
            foreground="#e4d98b",
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=810,
            font=("Courier New", 9, "bold"),
            padx=10,
            pady=8,
        )
        status.pack(fill=tk.X, padx=18, pady=(0, 15))

    def draw_wood_background(self) -> None:
        background = tk.Canvas(self, background=WOOD, highlightthickness=0)
        background.place(relx=0, rely=0, relwidth=1, relheight=1)
        for y in range(8, 760, 18):
            background.create_line(0, y, 940, y + (4 if (y // 18) % 2 else -3), fill=WOOD_DARK, width=2)
            for x in range((y * 7) % 90, 940, 145):
                background.create_arc(x, y - 7, x + 65, y + 9, start=8, extent=165, style=tk.ARC, outline="#8e5330")

    @staticmethod
    def beveled_frame(parent: tk.Misc, background: str) -> tk.Frame:
        return tk.Frame(
            parent,
            background=background,
            relief=tk.RAISED,
            borderwidth=4,
            highlightbackground=PANEL_DARK,
            highlightthickness=1,
        )

    @staticmethod
    def retro_button(parent: tk.Misc, text: str, command: object, width: int = 0, accent: bool = False) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            background=ACTIVE if accent else PANEL,
            foreground=BLUE if accent else INK,
            activebackground="#ece7bf",
            activeforeground=INK,
            relief=tk.RAISED,
            borderwidth=3,
            font=("Fixedsys", 11, "bold"),
            padx=7,
            pady=5,
            cursor="hand2",
        )

    def on_seed_changed(self, *_: object) -> None:
        self.preview_traffic_segments = []
        self.preview_world = None
        self.preview_photo = None
        self.preview_photo_size = 0
        self.after_idle(self.draw_seed_preview)

    def draw_seed_preview(self) -> None:
        canvas = self.preview_canvas
        canvas.delete("all")
        width = max(420, canvas.winfo_width())
        height = max(340, canvas.winfo_height())
        try:
            seed = int(self.seed_variable.get().strip())
        except ValueError:
            canvas.create_text(width / 2, height / 2, text="ENTER A WHOLE-NUMBER SEED", fill="#f3ddb2", font=PIXEL_BOLD)
            return
        canvas.create_rectangle(0, 0, width, height, fill=WATER, outline="")
        margin = 22
        left, top = margin + 32, margin + 27
        right, bottom = width - margin - 45, height - margin - 30

        def project_land(world_x: float, world_y: float) -> tuple[float, float]:
            return (
                left + world_x / 3000.0 * (right - left),
                bottom - (world_y + 3000.0) / 3000.0 * (bottom - top),
            )

        land_points: list[float] = []
        land_polygon: list[tuple[float, float]] = []
        for world_x, world_y in procedural_land_outline(seed):
            projected_x, projected_y = project_land(world_x, world_y)
            land_polygon.append((projected_x, projected_y))
            land_points.extend((projected_x, projected_y))
        canvas.create_polygon(land_points, fill=LAND, outline="#ece1b0", width=3)

        if self.preview_world is not None:
            self.draw_generated_world_preview(canvas, self.preview_world, left, top, right, bottom)
            preview_label = "TRUE GEOMETRY PREVIEW — CHECKPOINT BUILD"
        else:
            graph = generate_hierarchical_street_graph(seed, left, top, right, bottom, neighborhood_count=8)
            for water_body in graph.water_bodies:
                water_points = [(point.x, point.y) for point in water_body.points]
                if len(water_points) < 3:
                    continue
                water_center_x = sum(point[0] for point in water_points) / len(water_points)
                water_center_y = sum(point[1] for point in water_points) / len(water_points)
                if land_polygon and not point_inside_polygon(water_center_x, water_center_y, land_polygon):
                    continue
                canvas.create_polygon(
                    *(coordinate for point in water_points for coordinate in point),
                    fill=WATER,
                    outline="#77b8b1",
                    width=2,
                )
            self.draw_graph_blocks(canvas, graph, seed, land_polygon)
            self.draw_graph_roads(canvas, graph, land_polygon)
            canvas.create_text(
                (left + right) * 0.5,
                bottom - 12,
                text="RUN CHECK GTA INSTALL FOR A TRUE PREVIEW BUILT FROM GENERATED GEOMETRY",
                anchor=tk.S,
                fill="#f0e7ba",
                font=("Courier New", 8, "bold"),
            )
            preview_label = "PLACEHOLDER LAYOUT — NOT INSTALLED GEOMETRY"

        canvas.create_text(13, height - 12, text=f"SEED {seed}  /  LOS SANTOS", anchor=tk.SW, fill="#f0e7ba", font=("Courier New", 9, "bold"))
        canvas.create_text(width - 13, 13, text=preview_label, anchor=tk.NE, fill="#f0e7ba", font=("Courier New", 8, "bold"))

    @staticmethod
    def draw_active_traffic_graph(
        canvas: tk.Canvas,
        segments: list[TrafficPathSegment],
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> None:
        def project(x: float, y: float) -> tuple[float, float]:
            return left + x / 3000.0 * (right - left), bottom - (y + 3000.0) / 3000.0 * (bottom - top)

        for segment in sorted(segments, key=lambda item: item.width):
            if max(segment.start_x, segment.end_x) < 0.0 or min(segment.start_x, segment.end_x) > 3000.0:
                continue
            if max(segment.start_y, segment.end_y) < -3000.0 or min(segment.start_y, segment.end_y) > 0.0:
                continue
            start = project(segment.start_x, segment.start_y)
            end = project(segment.end_x, segment.end_y)
            canvas.create_line(*start, *end, fill=ROAD, width=max(1, min(5, round(segment.width / 6.0))), capstyle=tk.ROUND)

    def draw_generated_world_preview(
        self,
        canvas: tk.Canvas,
        world: LosSantosWorld,
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> None:
        target_size = max(192, min(384, int(min(right - left, bottom - top))))
        if self.preview_photo is None or self.preview_photo_size != target_size:
            self.preview_photo = self.build_generated_world_photo(world, target_size)
            self.preview_photo_size = target_size
        center_x = (left + right) * 0.5
        center_y = (top + bottom) * 0.5
        canvas.create_image(center_x, center_y, image=self.preview_photo)

    def build_generated_world_photo(self, world: LosSantosWorld, size: int) -> tk.PhotoImage:
        preview = self.build_generated_world_texture(world, size)
        photo = tk.PhotoImage(width=preview.width, height=preview.height)
        for y in range(preview.height):
            row_colors: list[str] = []
            for x in range(preview.width):
                red, green, blue, _ = preview.get_pixel(x, y)
                row_colors.append(f"#{red:02x}{green:02x}{blue:02x}")
            photo.put("{" + " ".join(row_colors) + "}", to=(0, y))
        return photo

    @staticmethod
    def build_generated_world_texture(world: LosSantosWorld, size: int) -> TextureImage:
        image = TextureImage.solid(size, size, (49, 91, 108, 255))
        water_surfaces = world.water_override.surfaces if world.water_override else ()
        for radar_y in range(6, 12):
            for radar_x in range(6, 12):
                tile = render_geometry_radar_tile(
                    radar_x, radar_y, world.models, world.placements,
                    water_surfaces=water_surfaces, preserved_bounds=world.retail_surface_bounds,
                    retail_radar_data=world.retail_radar_tiles.get(f"radar{radar_y * 12 + radar_x:02d}.txd"),
                )
                left = round((radar_x - 6) * size / 6)
                right = round((radar_x - 5) * size / 6)
                top = round((radar_y - 6) * size / 6)
                bottom = round((radar_y - 5) * size / 6)
                for y in range(top, bottom):
                    for x in range(left, right):
                        source_x = min(255, int((x - left + 0.5) * 256 / max(1, right - left)))
                        source_y = min(255, int((y - top + 0.5) * 256 / max(1, bottom - top)))
                        image.set_pixel(x, y, tile.get_pixel(source_x, source_y))
        return image


    @staticmethod
    def draw_graph_blocks(
        canvas: tk.Canvas,
        graph: StreetGraph,
        seed: int,
        land_polygon: list[tuple[float, float]] | None = None,
    ) -> None:
        district_colors = ("#245a9a", "#a86b42", "#716a73", "#c7904e", "#8b473d", "#b9a35c")
        random_source = random.Random(seed ^ 0x424C4F434B53)
        water_polygons = [
            [(point.x, point.y) for point in water_body.points]
            for water_body in graph.water_bodies
        ]
        for block in graph.blocks:
            inset_points = inset_block(block, 0.83)
            center_x = sum(point[0] for point in inset_points) / len(inset_points)
            center_y = sum(point[1] for point in inset_points) / len(inset_points)
            if any(point_inside_polygon(center_x, center_y, polygon) for polygon in water_polygons):
                continue
            if land_polygon is not None and not all(
                point_inside_polygon(point[0], point[1], land_polygon) for point in inset_points
            ):
                continue
            color = PARK if random_source.random() < 0.12 else district_colors[block.district_index % len(district_colors)]
            canvas.create_polygon(
                *(coordinate for point in inset_points for coordinate in (point[0], point[1])),
                fill=color,
                outline="#c8b980",
                width=1,
            )

    @staticmethod
    def draw_graph_roads(
        canvas: tk.Canvas,
        graph: StreetGraph,
        land_polygon: list[tuple[float, float]] | None = None,
    ) -> None:
        widths = {"arterial": 5, "collector": 3, "local": 2}
        colors = {"arterial": "#3f443f", "collector": ROAD, "local": "#62675f"}
        ordering = {"local": 0, "collector": 1, "arterial": 2}
        for segment in sorted(graph.segments, key=lambda item: ordering[item.street_class]):
            start = graph.nodes[segment.start_node]
            end = graph.nodes[segment.end_node]
            if land_polygon is not None:
                midpoint_x = (start.x + end.x) * 0.5
                midpoint_y = (start.y + end.y) * 0.5
                if not (
                    point_inside_polygon(start.x, start.y, land_polygon)
                    and point_inside_polygon(end.x, end.y, land_polygon)
                    and point_inside_polygon(midpoint_x, midpoint_y, land_polygon)
                ):
                    continue
            canvas.create_line(
                start.x,
                start.y,
                end.x,
                end.y,
                fill=colors[segment.street_class],
                width=widths[segment.street_class],
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
            )
    def browse_game(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.game_directory_variable.get())
        if selected:
            self.game_directory_variable.set(selected)

    def change_seed(self, delta: int) -> None:
        try:
            value = int(self.seed_variable.get().strip())
        except ValueError:
            value = 0
        self.seed_variable.set(str(max(0, min(2_147_483_647, value + delta))))

    def randomize_seed(self) -> None:
        self.seed_variable.set(str(random.SystemRandom().randint(1, 2_147_483_647)))

    def read_options(self) -> tuple[Path, int]:
        directory_text = self.game_directory_variable.get().strip()
        if not directory_text:
            raise ValueError("Select the GTA SA folder.")
        game_directory = Path(directory_text).expanduser().resolve()
        try:
            seed = int(self.seed_variable.get().strip())
        except ValueError as error:
            raise ValueError("Seed must be a whole number.") from error
        return game_directory, seed

    def begin_scan(self) -> None:
        try:
            game_directory, seed = self.read_options()
        except ValueError as error:
            messagebox.showerror("INVALID SETTINGS", str(error))
            return
        self.begin_task("SCANNING GTA SA MAP DATA AND BUILDING TRUE PREVIEW...", self.scan_thread, game_directory, seed)

    def scan_thread(self, game_directory: Path, seed: int) -> None:
        try:
            problems = validate_game_directory(game_directory, require_mod_loader=True)
            if problems:
                raise RuntimeError("\n".join(problems))
            scan = scan_game_map(game_directory)
            world = generate_los_santos_world(scan, seed)
            self.result_queue.put(("scan", (scan, world)))
        except Exception as error:
            self.result_queue.put(("error", error))

    def begin_install(self) -> None:
        try:
            options = self.read_options()
        except ValueError as error:
            messagebox.showerror("INVALID SETTINGS", str(error))
            return
        if not messagebox.askyesno(
            "BUILD THIS MAP?",
            "Generate and install this complete Los Santos seed? Original files stay untouched. Use a new disposable save because later missions still require validation.",
        ):
            return
        self.begin_task("GENERATING WORLD MESH, COLLISION, PATHS AND RADAR...", self.install_thread, *options)

    def install_thread(self, game_directory: Path, seed: int) -> None:
        try:
            problems = validate_game_directory(game_directory, require_mod_loader=True)
            if problems:
                raise RuntimeError("\n".join(problems))
            scan = scan_game_map(game_directory)
            world = generate_los_santos_world(scan, seed)
            with tempfile.TemporaryDirectory(prefix=f"SeedAndreas_{seed}_") as temporary_directory:
                staged_mod = write_los_santos_mod_package(game_directory, Path(temporary_directory), world)
                destination, backup = install_mod_package(game_directory, staged_mod, world.manifest)
            self.result_queue.put(("installed", (world.manifest, destination, backup)))
        except Exception as error:
            self.result_queue.put(("error", error))

    def begin_task(self, status: str, target: object, *arguments: object) -> None:
        if self.busy:
            return
        self.busy = True
        for button in self.task_buttons():
            button.configure(state=tk.DISABLED)
        self.progress.start(12)
        self.status_variable.set(status)
        threading.Thread(target=target, args=arguments, daemon=True).start()

    def task_buttons(self) -> tuple[tk.Widget, ...]:
        return (
            self.previous_button,
            self.next_button,
            self.random_button,
            self.scan_button,
            self.install_button,
            self.uninstall_button,
            self.browse_button,
            self.launch_button,
            self.seed_entry,
            self.path_entry,
        )

    def poll_results(self) -> None:
        try:
            result_type, payload = self.result_queue.get_nowait()
        except queue.Empty:
            self.after(100, self.poll_results)
            return

        self.busy = False
        self.progress.stop()
        for button in self.task_buttons():
            button.configure(state=tk.NORMAL)

        if result_type == "error":
            self.status_variable.set("FAILED — ORIGINAL GTA SA FILES WERE NOT CHANGED")
            error_text = str(payload)
            if "Permission" in error_text or "Access is denied" in error_text:
                error_text += "\n\nOpen SeedAndreas.pyw and approve the Windows administrator prompt."
            messagebox.showerror("SEED ANDREAS", error_text)
        elif result_type == "scan":
            scan, world = payload
            try:
                selected_directory, selected_seed = self.read_options()
            except ValueError:
                selected_directory, selected_seed = None, None
            if selected_seed != world.manifest.seed or selected_directory != Path(scan.game_directory).resolve():
                self.on_seed_changed()
                self.status_variable.set("SELECTION CHANGED — CHECK GTA INSTALL TO BUILD THIS PREVIEW")
                self.after(100, self.poll_results)
                return
            self.preview_traffic_segments = list(scan.vehicle_path_segments)
            self.preview_world = world
            self.preview_photo = None
            self.preview_photo_size = 0
            self.draw_seed_preview()
            self.status_variable.set(
                f"TRUE PREVIEW READY — {scan.ide_files_scanned} IDE / {scan.ipl_files_scanned} IPL / "
                f"{len(scan.models)} MODELS / {len(scan.placements)} PLACEMENTS / "
                f"{world.manifest.custom_buildings} GENERATED BUILDINGS"
            )
        elif result_type == "installed":
            manifest, destination, backup = payload
            backup_text = " / PREVIOUS MAP BACKED UP" if backup else ""
            self.status_variable.set(
                f"INSTALLED SEED {manifest.seed} [{manifest.seed_fingerprint}] — "
                f"{manifest.custom_buildings} BUILDINGS / {manifest.collision_triangles} COLLISION TRIANGLES{backup_text}"
            )
            messagebox.showinfo("MAP INSTALLED", f"Seed {manifest.seed} is installed at:\n{destination}\n\nClick PLAY GTA SAN ANDREAS and start a new disposable save.")
        self.after(100, self.poll_results)

    def launch_installed_game(self) -> None:
        try:
            launch_game(Path(self.game_directory_variable.get().strip()))
        except Exception as error:
            messagebox.showerror("COULD NOT LAUNCH GTA SA", str(error))

    def close_launcher(self) -> None:
        if self.busy:
            messagebox.showinfo("MAP BUILD IN PROGRESS", "The map build must finish before this window can close.")
            return
        self.destroy()

    def uninstall(self) -> None:
        try:
            moved_to = uninstall_mod_package(Path(self.game_directory_variable.get().strip()))
        except Exception as error:
            messagebox.showerror("COULD NOT REMOVE MAP", str(error))
            return
        if moved_to is None:
            self.status_variable.set("NO SEED ANDREAS MAP IS INSTALLED")
        else:
            self.status_variable.set(f"MAP REMOVED SAFELY — BACKUP: {moved_to}")


def main() -> None:
    launcher = SeedAndreasLauncher()
    launcher.mainloop()


if __name__ == "__main__":
    main()
