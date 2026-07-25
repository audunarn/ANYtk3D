# ANYtk3D

Fast dependency-free 3D drawing on a Tkinter Canvas.

ANYtk3D is the standalone 3D viewport extracted from
[ANYstructure](https://github.com/audunarn/ANYstructure). It renders
stiffened plates, cylinders and general 3D geometry directly on a
`tkinter.Canvas` — no OpenGL, no matplotlib, only numpy and the
standard library.

## Features

- Camera orbit, zoom and pan with cached static geometry and throttled
  redraws for smooth interaction
- Painter-order occlusion with back-face culling and adaptive
  subdivision around ring girder elevations
- Stiffened cylinders (longitudinal and ring stiffeners, inside or
  outside the shell) and stiffened flat plates with girders
- Open-ended cylinders and stippled semi-transparent shells for viewing
  internal structure; two-sided shell rendering
- Plate colour-coding by thickness (or any scalar) with a fixed legend
- Global X/Y/Z axis-orientation overlay and optional axis rulers
- Simple animation caching / playback

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

canvas.add_cylinder(radius=6.5, height=20.0, center=Point3D(0, 0, 10.0),
                    color='lightsteelblue', opacity=0.6)
canvas.add_ring_stiffener(radius=6.5, z_position=5.0,
                          web_height=0.5, web_thickness=0.015)
canvas.add_rectangular_plate(x_start=-3, x_end=3, y_start=-2, y_end=2)
canvas.redraw()
root.mainloop()
```

Higher-level primitives: `add_cylinder`, `add_longitudinal_stiffener`,
`add_ring_stiffener`, `add_rectangular_plate`, `add_flat_stiffener`,
`add_flat_girder`, `add_polygon`, `add_line`, `add_text`.

## Demo

Four viewports showing a stiffened cylinder and a stiffened plate in
two rendering styles:

```bash
python -m anytk3d
```

or embed the demo in an existing Tk application with
`anytk3d.create_stiffened_cylinder_demo(root)`.

## Relation to ANYstructure

ANYstructure uses this package for its 3D previews. The module keeps
its original API (`Tkinter3DCanvas`, `Point3D`, `Camera3D`) so it can
be consumed as a drop-in dependency.

## Development

```bash
pip install -e .[dev]
pytest
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
