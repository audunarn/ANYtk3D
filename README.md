# ANYtk3D

Fast dependency-free 3D drawing on a Tkinter Canvas.

ANYtk3D is the standalone 3D viewport extracted from
[ANYstructure](https://github.com/audunarn/ANYstructure). It renders
solids, stiffened plates, cylinders and arbitrary meshes directly on a
`tkinter.Canvas` — no OpenGL, no matplotlib, only numpy and the standard
library.

## Features

- **Typical shapes out of the box** — box, sphere, cone, truncated cone,
  cylinder, tube/pipe, torus, pyramid, wedge, prism, swept extrusion,
  disk/annulus, plane, arrow, ground grid, structural beam profiles
  (FB, T, I, L, C, BOX) and arbitrary index meshes
- **Directional lighting** — ambient + Lambert diffuse + Blinn-Phong
  highlight, flat-shaded per face, two-sided, world-fixed or head-lamp
- **Correct depth order** — every face is painted back to front by
  camera-space depth, closed solids cull their back faces, faces
  straddling the near plane are clipped instead of dropped, and 3D lines
  take part in the same sort
- **Layered transparency** — 16 screen-door density steps with
  non-overlapping windows per layer, so a transparent shell shows its far
  wall, its contents *and* its near wall at once
- **Built for speed** — geometry is compiled to numpy arrays once and
  every frame is vectorised; canvas items are pooled and only
  reconfigured when their appearance changes; the interactive level of
  detail self-tunes from measured frame times
- **Result fields and animation** — a batched `add_faces` call for meshes
  coloured per element, plus frame capture and playback that adapts its
  detail to the requested frame rate
- Camera orbit, zoom and pan; plate colour-coding by thickness (or any
  scalar) with a fixed legend; X/Y/Z axis overlay and optional rulers;
  animation caching and playback

## Installation

```bash
pip install ANYtk3D
```

Tkinter ships with the standard CPython installers.

## Quick start

```python
import tkinter as tk
from anytk3d import Point3D, Tkinter3DCanvas

root = tk.Tk()
canvas = Tkinter3DCanvas(root, width=900, height=600, bg='white')
canvas.pack(fill='both', expand=True)

canvas.add_box(2.0, 1.0, 0.5, center=Point3D(-3, 0, 0.25), color='#4e79a7')
canvas.add_sphere(0.8, center=Point3D(0, 0, 0.8), color='#f28e2b')
canvas.add_tube(0.8, 0.5, 2.0, center=Point3D(3, 0, 1.0), color='#59a14f')
canvas.add_beam(Point3D(-4, 3, 0), Point3D(4, 3, 0), kind='I',
                web_height=0.6, flange_width=0.3)
canvas.add_grid(size_x=12, size_y=10)

canvas.fit_to_scene()
root.mainloop()
```

### Shapes

`add_box`, `add_box_from_bounds`, `add_sphere`, `add_cone`,
`add_frustum`, `add_cylinder`, `add_tube`, `add_torus`, `add_pyramid`,
`add_wedge`, `add_prism`, `add_extrusion`, `add_disk`, `add_plane`,
`add_arrow`, `add_beam`, `add_grid`, `add_mesh`, `add_shape`,
plus the original `add_polygon`, `add_line`, `add_text`,
`add_rectangular_plate`, `add_flat_stiffener`, `add_flat_girder`,
`add_longitudinal_stiffener` and `add_ring_stiffener`.

Every shape builder takes `center`/`axis` placement and the same material
keywords — `color`, `outline`, `opacity`, `layer`, `cull_backface`,
`lit`, `back_color`, `face_colors`, `tags`.

The tessellation lives in `anytk3d.shapes` as plain functions returning a
`Mesh` (vertices plus index faces), so it can be used, combined and
tested without a display:

```python
from anytk3d import shapes

mesh = shapes.torus(2.0, 0.4).merged(shapes.sphere(0.5))
canvas.add_shape(mesh, position=Point3D(0, 0, 3), color='#b07aa1')
```

### Lighting

```python
canvas.set_light(direction=Point3D(0.4, -0.6, 0.7), ambient=0.45,
                 diffuse=0.55, specular=0.12)
canvas.set_light(follow_camera=True)   # head lamp that orbits with the view
canvas.set_shading(False)              # flat colours, no shading
```

Ambient and diffuse sum to 1.0 by default, so a face turned toward the
light renders in its exact base colour and every other face only darkens.
That keeps colour-coded scenes matching their legend.

### Transparency

`opacity` runs from 0 to 1 in 16 usable steps. A Tk canvas has no alpha
channel, so this is screen-door stippling — but the front and back of a
surface are given non-overlapping dither windows, sized so that stacking
them reproduces ordinary alpha compositing (`1 - (1 - a)^2`). Passing an
explicit `stipple=` string still uses that exact Tk pattern.

```python
canvas.add_sphere(2.0, color='#7fb3d5', opacity=0.35)   # see the contents
canvas.add_box(1.0, 1.0, 1.0, color='#c0392b')          # ...through the shell
```

### Picking and highlighting

Every shape builder takes `tags`, and those tags reach the underlying Tk
canvas item. Picking uses Tk's own hit testing, so the topmost item under the
cursor is the nearest one — no ray casting, and occlusion is correct for free.

```python
canvas.add_box(2, 1, 0.5, tags='plate7')

def on_pick(pick):
    print(pick.tag, pick.shift)          # '' when the click missed everything
    canvas.set_highlight([pick.tag] if pick.tag else [])

canvas.set_pick_callback(on_pick, prefix='plate')   # prefix is optional
canvas.set_hover_callback(lambda pick: ...)         # fires only on change
canvas.pick_at(x, y)                                # query without an event
```

A click is a press and release without a drag, so picking coexists with pan
and orbit. Highlighting is applied while rendering rather than by
reconfiguring Tk items, so it survives the next redraw; the resolution from
tags to faces is cached per scene and highlight generation.

Picking is opt-in — with no callback set, the canvas behaves exactly as before.

For CAD/FE-style interaction, opt into the commercial profile. LMB selects
and draws a directional box, MMB pans, RMB orbits, and the wheel zooms. A
left-to-right box requires full containment; right-to-left is crossing.

```python
from anytk3d import (
    PickBinding, SelectionConfig, SelectionDepth, SelectionFilter,
)

canvas = Tkinter3DCanvas(root, interaction_profile="commercial")
canvas.add_faces(
    element_polygons,
    bindings=[
        PickBinding.one(f"element{number}", "mesh.element")
        for number in element_numbers
    ],
)
canvas.configure_selection(
    lambda event: print(event.operation, [hit.key for hit in event.hits]),
    hover_callback=lambda hit: print(None if hit is None else hit.key),
    config=SelectionConfig(
        filter=SelectionFilter(kinds=frozenset({"mesh.element"})),
        depth=SelectionDepth.VISIBLE,
    ),
)
```

No modifier replaces the selection, Shift adds, Ctrl toggles, and Alt
removes. `query_point`, `query_rectangle`, `screen_ray`, and
`unproject_to_plane` expose the same projected geometry for modelling tools.
The default interaction profile remains `legacy` for compatibility.

### Colour scale

Plate colour-coding, the legend and `thickness_color` all share one scale.
Replace it to use a different colour map — sample any ramp at a handful of
positions and hand the stops over:

```python
from matplotlib import colormaps, colors   # only if you want matplotlib maps
import anytk3d

stops = [(t / 16, colors.to_hex(colormaps['viridis'](t / 16))) for t in range(17)]
anytk3d.set_color_stops(stops)
anytk3d.get_color_stops()
anytk3d.reset_color_stops()                # back to the built-in blue→red ramp
```

Colours a canvas resolved itself (cylinder plate thickness, stiffener
thickness) refresh on the next redraw. `anytk3d.DEFAULT_COLOR_STOPS` is the
built-in scale; assigning to the old `_THICKNESS_COLOR_STOPS` constant has no
effect — use `set_color_stops`.

### Result fields

`add_faces` is the batched path for a mesh where every element carries its
own colour — an FE stress plot, a deformed shape, a utilisation map. It
takes the whole element set in one call and computes all the centroids and
normals as array operations:

```python
canvas.add_faces(polygons, colors=element_colors, outline='#64748b')
canvas.set_thickness_legend([0, 80, 160, 240], unit='MPa', title='von Mises')
```

`polygons` is a sequence of vertex sequences (`Point3D` or `(x, y, z)`), or
an `(faces, vertices, 3)` array. On a 4900-element field this compiles in
4.5 ms against 48 ms for the same elements added one at a time.

### Animation

Capture a scene per step, then replay it:

```python
canvas.begin_animation_cache()
for step in range(frames):
    canvas.clear(keep_canvas=True)
    canvas.add_faces(deformed_shape(step), colors=field_colors(step))
    canvas.capture_animation_frame()
canvas.play_animation(fps=30)          # fast=None/True/False
```

A scene without cylinders or stiffeners compiles to a single shared
representation, so each captured frame costs one build rather than two.
Playback defaults to `fast=None`, which starts at full detail and drops to
the reduced-detail path as soon as a frame overruns its slot; pass `True`
or `False` to pin it. `animation_frames`, `animation_frame_index` and
`is_playing_animation` are available for a progress readout.

### Performance

Interactive frames use a reduced-detail scene with a face budget that
adapts to the measured frame rate, so orbiting stays responsive on dense
models and full detail returns when the mouse is released.

```python
canvas.set_interactive_detail(4000)   # starting face budget while dragging
canvas.set_mesh_lines(False)          # drop per-face outlines
canvas.set_occlude_lines(False)       # keep 3D lines on top of geometry
```

## Demos

The interactive showcase — shapes with live light controls, a colour-coded
FE result field, animation playback with a measured frame rate, and a
stiffened cylinder — is a single file you can run straight from an IDE
(right-click → Run in PyCharm) or from a shell:

```bash
python -m anytk3d.demo
python run_gui.py
```

The original four-viewport demo is still there:

```bash
python -m anytk3d
```

or embed it in an existing Tk application with
`anytk3d.create_stiffened_cylinder_demo(root)`.

## Relation to ANYstructure

ANYstructure uses this package for its 3D previews. The module keeps its
original API (`Tkinter3DCanvas`, `Point3D`, `Camera3D` and every
`add_*` method) so it stays a drop-in dependency.

## Development

```bash
pip install -e .[dev]
pytest
```

The regular suite uses deterministic Tk-generated events and never moves the
desktop pointer. A small Windows-native acceptance suite covers event
translation that synthetic Tk events cannot reproduce: hover/click and
selection modifiers, directional window/crossing selection, plus middle-pan,
right-orbit and wheel zoom. Run it only on an interactive Windows desktop:

```powershell
$env:ANYTK3D_RUN_NATIVE_GUI = "1"
python -m pytest tests/test_native_gui_acceptance.py -q
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
