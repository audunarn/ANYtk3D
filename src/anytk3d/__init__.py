'''
ANYtk3D - fast dependency-free 3D drawing on a Tkinter Canvas.

Renders stiffened plates, cylinders and general 3D geometry directly on
a ``tkinter.Canvas`` with camera orbit/zoom, depth-sorted painter
rendering, flat-shaded directional lighting, layered screen-door
transparency, thickness colour-coding with a legend, and an
axis-orientation overlay.  Only numpy and the standard library are
required - no OpenGL, no matplotlib.

Quick start::

    import tkinter as tk
    from anytk3d import Point3D, Tkinter3DCanvas

    root = tk.Tk()
    canvas = Tkinter3DCanvas(root, width=800, height=600, bg='white')
    canvas.pack(fill='both', expand=True)
    canvas.add_cylinder(radius=1.0, height=4.0, center=Point3D(0, 0, 0))
    canvas.add_box(1.0, 1.0, 1.0, center=Point3D(2, 0, 0), color='#4e79a7')
    canvas.redraw()
    root.mainloop()

Run ``python -m anytk3d`` for a four-viewport demonstration, or
``python -m anytk3d.demo`` for the interactive showcase of shapes,
lighting, FE result fields and animation.
'''
from . import picking, shapes, shading, stipple
from .canvas import (
    DEFAULT_COLOR_STOPS,
    Camera3D,
    Light,
    Mesh,
    Point3D,
    Tkinter3DCanvas,
    create_stiffened_cylinder_demo,
    get_color_stops,
    main,
    populate_fe_gui_cylinder,
    populate_fe_gui_plate,
    populate_shape_gallery,
    populate_stiffened_cylinder,
    populate_stiffened_plate,
    reset_color_stops,
    set_color_stops,
)
from .picking import (
    Pick,
    PickBinding,
    PickOwner,
    SelectionConfig,
    SelectionDepth,
    SelectionEvent,
    SelectionFilter,
    SelectionGesture,
    SelectionHit,
    SelectionOperation,
    SelectionTool,
)
from any3dview import SectionPlane

__version__ = "0.2.3"

__all__ = [
    "DEFAULT_COLOR_STOPS",
    "Camera3D",
    "Light",
    "Mesh",
    "Pick",
    "PickBinding",
    "PickOwner",
    "Point3D",
    "SelectionConfig",
    "SelectionDepth",
    "SelectionEvent",
    "SelectionFilter",
    "SelectionGesture",
    "SelectionHit",
    "SelectionOperation",
    "SelectionTool",
    "SectionPlane",
    "Tkinter3DCanvas",
    "create_stiffened_cylinder_demo",
    "get_color_stops",
    "main",
    "populate_fe_gui_cylinder",
    "populate_fe_gui_plate",
    "populate_shape_gallery",
    "populate_stiffened_cylinder",
    "populate_stiffened_plate",
    "picking",
    "reset_color_stops",
    "set_color_stops",
    "shading",
    "shapes",
    "stipple",
]


def run_demo() -> None:
    """Open the interactive showcase (see :mod:`anytk3d.demo`)."""
    from .demo import main as _demo_main

    _demo_main()
