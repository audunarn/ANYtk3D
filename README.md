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

### Performance

Interactive frames use a reduced-detail scene with a face budget that
adapts to the measured frame rate, so orbiting stays responsive on dense
models and full detail returns when the mouse is released.

```python
canvas.set_interactive_detail(4000)   # starting face budget while dragging
canvas.set_mesh_lines(False)          # drop per-face outlines
canvas.set_occlude_lines(False)       # keep 3D lines on top of geometry
```

## Demo

Four viewports — a shape gallery with lighting and transparency, and
three structural scenes:

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

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
