'''
ANYtk3D - fast dependency-free 3D drawing on a Tkinter Canvas.

Renders stiffened plates, cylinders and general 3D geometry directly on
a ``tkinter.Canvas`` with camera orbit/zoom, painter-order occlusion,
thickness colour-coding with a legend, and an axis-orientation overlay.
Only numpy and the standard library are required - no OpenGL, no
matplotlib.

Quick start::

    import tkinter as tk
    from anytk3d import Point3D, Tkinter3DCanvas

    root = tk.Tk()
    canvas = Tkinter3DCanvas(root, width=800, height=600, bg='white')
    canvas.pack(fill='both', expand=True)
    canvas.add_cylinder(radius=1.0, height=4.0, center=Point3D(0, 0, 0))
    canvas.redraw()
    root.mainloop()

Run ``python -m anytk3d`` for a four-viewport demonstration.
'''
from .canvas import (
    Camera3D,
    Point3D,
    Tkinter3DCanvas,
    create_stiffened_cylinder_demo,
    main,
    populate_fe_gui_cylinder,
    populate_fe_gui_plate,
    populate_stiffened_cylinder,
    populate_stiffened_plate,
)

__version__ = "0.1.0"

__all__ = [
    "Camera3D",
    "Point3D",
    "Tkinter3DCanvas",
    "create_stiffened_cylinder_demo",
    "main",
    "populate_fe_gui_cylinder",
    "populate_fe_gui_plate",
    "populate_stiffened_cylinder",
    "populate_stiffened_plate",
]
