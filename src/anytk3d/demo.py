"""
Runnable ANYtk3D showcase - shapes, lighting, an FE result field and animation.

Run it straight from an IDE (right-click -> Run in PyCharm), or from a shell::

    python -m anytk3d.demo

Four tabs, each with its own controls:

* **Shapes & lighting** - every built-in solid, with live control of the sun
  direction, ambient/diffuse balance and the specular highlight.
* **FE result** - a deformed plate coloured by a scalar field, built through
  :meth:`Tkinter3DCanvas.add_faces`, the batched path meant for result
  meshes.  Mesh density and deformation scale are adjustable so the frame
  cost can be watched as the element count grows.
* **Animation** - the same result field swept through a vibration cycle,
  pre-captured and replayed, with a measured playback frame rate.
* **Stiffened cylinder** - the structural view, with a transparent shell over
  internal stiffeners.

Imports are absolute so the file works both as a script and as a module.
"""

from __future__ import annotations

import math
import time
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, List, Optional, Sequence, Tuple

from anytk3d.canvas import Point3D, Tkinter3DCanvas, populate_stiffened_cylinder
from anytk3d.core import _interpolate_thickness_color
from anytk3d.shading import sun_direction
from anytk3d import shapes


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


def _viewport(parent: tk.Misc) -> Tuple[ttk.Frame, ttk.Frame, Tkinter3DCanvas]:
    """A tab holding a control strip above a 3D canvas."""
    frame = ttk.Frame(parent)
    controls = ttk.Frame(frame)
    controls.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))
    canvas = Tkinter3DCanvas(frame, width=900, height=560, bg="white")
    canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(2, 6))
    return frame, controls, canvas


def _view_buttons(controls: tk.Misc, canvas: Tkinter3DCanvas) -> None:
    for text, command in (
        ("Fit", canvas.fit_to_scene),
        ("Iso", canvas.set_iso_view),
        ("Top", canvas.set_top_view),
        ("Front", canvas.set_front_view),
        ("Side", canvas.set_side_view),
    ):
        ttk.Button(controls, text=text, width=6, command=command).pack(side=tk.LEFT, padx=2)


def _slider(
    parent: tk.Misc,
    label: str,
    low: float,
    high: float,
    initial: float,
    command: Callable[[float], None],
) -> tk.DoubleVar:
    box = ttk.Frame(parent)
    box.pack(side=tk.LEFT, padx=(10, 0))
    ttk.Label(box, text=label).pack(side=tk.TOP, anchor="w")
    variable = tk.DoubleVar(value=initial)
    ttk.Scale(
        box,
        from_=low,
        to=high,
        variable=variable,
        length=110,
        command=lambda _value: command(variable.get()),
    ).pack(side=tk.TOP)
    return variable


# ----------------------------------------------------------------------
# Tab 1 - shapes and lighting
# ----------------------------------------------------------------------


SHAPE_ROW_ONE = (
    ("box", lambda c, p: c.add_box(1.6, 1.6, 1.6, center=p(0.8), color="#4e79a7")),
    ("sphere", lambda c, p: c.add_sphere(0.9, center=p(0.9), color="#f28e2b")),
    ("cone", lambda c, p: c.add_cone(0.9, 1.8, center=p(0.9), color="#e15759")),
    ("frustum", lambda c, p: c.add_frustum(0.95, 0.45, 1.7, center=p(0.85), color="#76b7b2")),
    ("pyramid", lambda c, p: c.add_pyramid(1.0, 1.7, center=p(0.85), color="#59a14f")),
)

SHAPE_ROW_TWO = (
    ("tube", lambda c, p: c.add_tube(0.95, 0.6, 1.7, center=p(0.85), color="#edc948")),
    ("torus", lambda c, p: c.add_torus(0.8, 0.27, center=p(0.6), color="#b07aa1")),
    ("wedge", lambda c, p: c.add_wedge(1.7, 1.4, 1.4, center=p(0.7), color="#ff9da7")),
    (
        "prism (I)",
        lambda c, p: c.add_prism(
            shapes.profile_section("I", 1.4, 0.14, 0.8, 0.14),
            1.7,
            center=p(0.0),
            axis=Point3D(0.0, 1.0, 0.0),
            color="#9c755f",
        ),
    ),
    ("annulus", lambda c, p: c.add_disk(1.0, 0.5, center=p(0.02), color="#8d99ae")),
)


def build_shape_tab(notebook: ttk.Notebook) -> ttk.Frame:
    frame, controls, canvas = _viewport(notebook)
    canvas.set_mesh_lines(False)

    canvas.add_grid(
        size_x=15.0, size_y=11.0, step=1.0,
        center=Point3D(0.0, 0.6, 0.0), color="#dde4ec",
    )
    for row_y, row in ((4.5, SHAPE_ROW_ONE), (1.0, SHAPE_ROW_TWO)):
        for column, (name, place) in enumerate(row):
            x = (column - 2) * 3.0
            place(canvas, lambda z, x=x, y=row_y: Point3D(x, y, z))
            canvas.add_text(Point3D(x, row_y, -0.5), name, color="#475569")

    # A transparent shell with something solid inside it: the far wall, the
    # contents and the near wall all stay readable.
    canvas.add_sphere(
        1.6, center=Point3D(-4.2, -3.4, 1.7), segments=30, rings=20,
        color="#86bcd8", opacity=0.35,
    )
    canvas.add_box(1.1, 1.1, 1.1, center=Point3D(-4.2, -3.4, 1.7), color="#c0392b")
    canvas.add_text(Point3D(-4.2, -3.4, -0.5), "transparent shell", color="#475569")

    canvas.add_beam(
        Point3D(-1.4, -3.4, 0.0), Point3D(5.5, -3.4, 0.0), kind="I",
        web_height=1.0, web_thickness=0.07, flange_width=0.55,
        flange_thickness=0.09, up=Point3D(0.0, 0.0, 1.0), color="#7f8c9a",
    )
    for x in (-1.0, 2.05, 5.1):
        canvas.add_arrow(Point3D(x, -3.4, 3.0), Point3D(x, -3.4, 1.2))
    canvas.add_text(Point3D(2.05, -3.4, -0.5), "beam + arrows", color="#475569")

    _view_buttons(controls, canvas)

    state = {"azimuth": 300.0, "elevation": 50.0}

    def refresh_light(*_args: Any) -> None:
        canvas.set_light(direction=sun_direction(state["azimuth"], state["elevation"]))

    def set_azimuth(value: float) -> None:
        state["azimuth"] = value
        refresh_light()

    def set_elevation(value: float) -> None:
        state["elevation"] = value
        refresh_light()

    _slider(controls, "sun azimuth", 0.0, 360.0, state["azimuth"], set_azimuth)
    _slider(controls, "sun elevation", -20.0, 89.0, state["elevation"], set_elevation)
    _slider(controls, "ambient", 0.0, 1.0, 0.45,
            lambda value: canvas.set_light(ambient=value))
    _slider(controls, "specular", 0.0, 0.8, 0.12,
            lambda value: canvas.set_light(specular=value))

    shading = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        controls, text="Lighting", variable=shading,
        command=lambda: canvas.set_shading(shading.get()),
    ).pack(side=tk.LEFT, padx=(12, 0))

    follow = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        controls, text="Head lamp", variable=follow,
        command=lambda: canvas.set_light(follow_camera=follow.get()),
    ).pack(side=tk.LEFT, padx=(6, 0))

    refresh_light()
    canvas.after(120, canvas.fit_to_scene)
    return frame


# ----------------------------------------------------------------------
# FE result field, shared by the result and animation tabs
# ----------------------------------------------------------------------


def result_field(
    divisions: int,
    phase: float = 0.0,
) -> Tuple[List[List[Tuple[float, float, float]]], List[List[float]]]:
    """
    A plate response: node grid plus the scalar plotted on it.

    The wave travels along the plate rather than standing still, so the
    amplitude - and with it the colour map - stays alive all the way through
    the cycle instead of collapsing to zero twice per period.
    """
    span_x, span_y = 6.0, 4.0
    grid: List[List[Tuple[float, float, float]]] = []
    values: List[List[float]] = []
    for i in range(divisions + 1):
        x = -0.5 * span_x + span_x * i / divisions
        row_points: List[Tuple[float, float, float]] = []
        row_values: List[float] = []
        for j in range(divisions + 1):
            y = -0.5 * span_y + span_y * j / divisions
            envelope = math.sin(math.pi * (x + 0.5 * span_x) / span_x)
            shape = envelope * math.sin(
                2.0 * math.pi * (y + 0.5 * span_y) / span_y - phase
            )
            row_points.append((x, y, shape))
            row_values.append(abs(shape))
        grid.append(row_points)
        values.append(row_values)
    return grid, values


def add_result_field(
    canvas: Tkinter3DCanvas,
    divisions: int,
    scale: float,
    phase: float = 0.0,
    peak: float = 1.0,
) -> int:
    """
    Push one result field onto the canvas as a single batch.

    ``add_faces`` takes the whole element set in one call: one flat vertex
    array and one colour per element.  Adding the same elements one at a time
    through ``add_polygon`` costs a dictionary plus a Python centroid and
    normal for each of them, which is what dominates on real result meshes.
    """
    grid, values = result_field(divisions, phase)
    polygons = []
    colors = []
    for i in range(divisions):
        for j in range(divisions):
            corner_00 = grid[i][j]
            corner_10 = grid[i + 1][j]
            corner_11 = grid[i + 1][j + 1]
            corner_01 = grid[i][j + 1]
            polygons.append(
                (
                    (corner_00[0], corner_00[1], corner_00[2] * scale),
                    (corner_10[0], corner_10[1], corner_10[2] * scale),
                    (corner_11[0], corner_11[1], corner_11[2] * scale),
                    (corner_01[0], corner_01[1], corner_01[2] * scale),
                )
            )
            average = 0.25 * (
                values[i][j] + values[i + 1][j] + values[i + 1][j + 1] + values[i][j + 1]
            )
            colors.append(_interpolate_thickness_color(average, 0.0, peak))

    canvas.add_faces(polygons, colors=colors, outline="#64748b", layer=5)
    return len(polygons)


def _result_legend(canvas: Tkinter3DCanvas, peak: float) -> None:
    canvas.set_thickness_legend(
        [round(peak * step / 4.0, 2) for step in range(5)],
        unit="mm",
        title="Deflection",
    )


# ----------------------------------------------------------------------
# Tab 2 - FE result
# ----------------------------------------------------------------------


def build_result_tab(notebook: ttk.Notebook) -> ttk.Frame:
    frame, controls, canvas = _viewport(notebook)
    status = ttk.Label(controls, text="")

    divisions = tk.IntVar(value=48)
    scale = tk.DoubleVar(value=1.0)
    first = {"value": True}

    def rebuild(*_args: Any) -> None:
        canvas.clear(keep_canvas=True)
        started = time.perf_counter()
        elements = add_result_field(canvas, divisions.get(), scale.get())
        build_ms = 1000.0 * (time.perf_counter() - started)
        _result_legend(canvas, 1.0)

        started = time.perf_counter()
        canvas.redraw()
        draw_ms = 1000.0 * (time.perf_counter() - started)
        status.configure(
            text=f"{elements} elements   build {build_ms:5.1f} ms   draw {draw_ms:5.1f} ms"
        )
        if first["value"]:
            first["value"] = False
            canvas.after(80, canvas.fit_to_scene)

    _view_buttons(controls, canvas)
    ttk.Label(controls, text="  mesh").pack(side=tk.LEFT, padx=(12, 2))
    ttk.Spinbox(
        controls, from_=8, to=140, increment=8, width=5,
        textvariable=divisions, command=rebuild,
    ).pack(side=tk.LEFT)
    _slider(controls, "deformation", 0.0, 3.0, 1.0, lambda _value: rebuild())

    mesh_lines = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        controls, text="Mesh lines", variable=mesh_lines,
        command=lambda: canvas.set_mesh_lines(mesh_lines.get()),
    ).pack(side=tk.LEFT, padx=(12, 0))
    status.pack(side=tk.RIGHT, padx=8)

    canvas.after(60, rebuild)
    return frame


# ----------------------------------------------------------------------
# Tab 3 - animation
# ----------------------------------------------------------------------


def build_animation_tab(notebook: ttk.Notebook) -> ttk.Frame:
    frame, controls, canvas = _viewport(notebook)
    status = ttk.Label(controls, text="Capture, then play.")

    divisions = tk.IntVar(value=40)
    frame_count = tk.IntVar(value=24)
    fps = tk.IntVar(value=30)
    fast = tk.StringVar(value="auto")

    meter = {"last_index": 0, "last_time": time.perf_counter(), "after_id": None}

    def capture() -> None:
        steps = max(2, frame_count.get())
        canvas.stop_animation()
        status.configure(text=f"Capturing {steps} frames...")
        canvas.update_idletasks()

        started = time.perf_counter()
        canvas.begin_animation_cache()
        elements = 0
        for step in range(steps):
            canvas.clear(keep_canvas=True)
            elements = add_result_field(
                canvas, divisions.get(), 1.0,
                phase=2.0 * math.pi * step / steps,
            )
            _result_legend(canvas, 1.0)
            canvas.capture_animation_frame()
        elapsed = time.perf_counter() - started

        canvas.fit_to_scene()
        status.configure(
            text=f"{steps} frames x {elements} elements captured in "
                 f"{elapsed:.2f} s ({1000 * elapsed / steps:.0f} ms/frame)"
        )

    def play() -> None:
        if canvas.animation_frames == 0:
            capture()
        mode = fast.get()
        canvas.play_animation(
            fps=max(1, fps.get()),
            fast={"auto": None, "full": False, "fast": True}[mode],
        )
        meter["last_index"] = canvas.animation_frame_index
        meter["last_time"] = time.perf_counter()
        tick_meter()

    def stop() -> None:
        canvas.stop_animation()
        if meter["after_id"] is not None:
            canvas.after_cancel(meter["after_id"])
            meter["after_id"] = None

    def tick_meter() -> None:
        meter["after_id"] = None
        if not canvas.is_playing_animation:
            return
        now = time.perf_counter()
        index = canvas.animation_frame_index
        total = max(1, canvas.animation_frames)
        advanced = (index - meter["last_index"]) % total
        elapsed = now - meter["last_time"]
        if elapsed >= 0.5 and advanced:
            status.configure(
                text=f"Playing {total} frames at {advanced / elapsed:4.1f} fps"
                     f"   (render: {'reduced' if canvas._animation_fast else 'full'} detail)"
            )
            meter["last_index"] = index
            meter["last_time"] = now
        meter["after_id"] = canvas.after(250, tick_meter)

    _view_buttons(controls, canvas)
    ttk.Label(controls, text="  mesh").pack(side=tk.LEFT, padx=(12, 2))
    ttk.Spinbox(controls, from_=8, to=100, increment=8, width=5,
                textvariable=divisions).pack(side=tk.LEFT)
    ttk.Label(controls, text="  frames").pack(side=tk.LEFT, padx=(8, 2))
    ttk.Spinbox(controls, from_=4, to=80, increment=4, width=5,
                textvariable=frame_count).pack(side=tk.LEFT)
    ttk.Label(controls, text="  fps").pack(side=tk.LEFT, padx=(8, 2))
    ttk.Spinbox(controls, from_=1, to=60, increment=1, width=4,
                textvariable=fps).pack(side=tk.LEFT)
    ttk.Label(controls, text="  detail").pack(side=tk.LEFT, padx=(8, 2))
    ttk.Combobox(controls, textvariable=fast, width=6, state="readonly",
                 values=("auto", "full", "fast")).pack(side=tk.LEFT)

    ttk.Button(controls, text="Capture", command=capture).pack(side=tk.LEFT, padx=(12, 2))
    ttk.Button(controls, text="Play", command=play).pack(side=tk.LEFT, padx=2)
    ttk.Button(controls, text="Stop", command=stop).pack(side=tk.LEFT, padx=2)
    status.pack(side=tk.RIGHT, padx=8)

    canvas.after(80, capture)
    return frame


# ----------------------------------------------------------------------
# Tab 4 - structural view
# ----------------------------------------------------------------------


def build_cylinder_tab(notebook: ttk.Notebook) -> ttk.Frame:
    frame, controls, canvas = _viewport(notebook)
    populate_stiffened_cylinder(canvas)
    _view_buttons(controls, canvas)

    rulers = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        controls, text="Rulers", variable=rulers,
        command=lambda: canvas.set_axis_ruler(rulers.get()),
    ).pack(side=tk.LEFT, padx=(12, 0))

    mesh_lines = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        controls, text="Mesh lines", variable=mesh_lines,
        command=lambda: canvas.set_mesh_lines(mesh_lines.get()),
    ).pack(side=tk.LEFT, padx=(6, 0))

    ttk.Label(
        controls,
        text="Right-drag: rotate   Left-drag: pan   Wheel: zoom",
    ).pack(side=tk.RIGHT, padx=8)
    canvas.after(120, canvas.fit_to_scene)
    return frame


# ----------------------------------------------------------------------


def build_demo(container: tk.Misc) -> ttk.Notebook:
    """Populate ``container`` with the demo notebook."""
    notebook = ttk.Notebook(container)
    notebook.pack(fill=tk.BOTH, expand=True)
    notebook.add(build_shape_tab(notebook), text="Shapes & lighting")
    notebook.add(build_result_tab(notebook), text="FE result")
    notebook.add(build_animation_tab(notebook), text="Animation")
    notebook.add(build_cylinder_tab(notebook), text="Stiffened cylinder")
    return notebook


def main() -> None:
    root = tk.Tk()
    root.title("ANYtk3D demo - shapes, lighting, FE results and animation")
    root.geometry("1180x740")
    root.minsize(900, 600)
    build_demo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
