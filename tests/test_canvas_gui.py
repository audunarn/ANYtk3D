'''
GUI tests - create a real Tkinter3DCanvas and populate the demo scenes.
Skipped automatically when no display is available.
'''
import tkinter as tk

import pytest

from anytk3d import (
    Point3D,
    Tkinter3DCanvas,
    populate_fe_gui_cylinder,
    populate_fe_gui_plate,
    populate_stiffened_cylinder,
    populate_stiffened_plate,
)


@pytest.fixture
def root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no display available for tkinter')
    root.withdraw()
    yield root
    root.destroy()


@pytest.mark.parametrize('populate', [
    populate_stiffened_cylinder,
    populate_stiffened_plate,
    populate_fe_gui_cylinder,
    populate_fe_gui_plate,
])
def test_demo_scenes_populate_and_render(root, populate):
    canvas = Tkinter3DCanvas(root, width=400, height=300, bg='white')
    canvas.pack()

    populate(canvas)

    assert len(canvas.objects) > 0
    canvas.redraw()


def test_primitives_and_thickness_legend(root):
    canvas = Tkinter3DCanvas(root, width=400, height=300, bg='white')
    canvas.pack()

    canvas.add_cylinder(radius=1.0, height=4.0, center=Point3D(0.0, 0.0, 2.0))
    canvas.add_ring_stiffener(radius=1.0, z_position=2.0)
    canvas.add_rectangular_plate(x_start=-1.0, x_end=1.0, y_start=-1.0, y_end=1.0)
    canvas.set_thickness_legend([10.0, 15.0, 20.0], unit='mm')
    canvas.redraw()

    assert canvas._thickness_legend is not None
    assert canvas._thickness_legend['unit'] == 'mm'
    assert any(obj.get('type') == 'cylinder' for obj in canvas.objects)


def test_camera_orbit_and_zoom(root):
    canvas = Tkinter3DCanvas(root, width=400, height=300, bg='white')
    canvas.add_cylinder(radius=1.0, height=4.0, center=Point3D(0.0, 0.0, 2.0))
    canvas.redraw()

    before = canvas.camera.position.to_tuple()
    canvas.camera.orbit(0.3, 0.1)
    canvas.camera.zoom(0.8)
    canvas.redraw()

    assert canvas.camera.position.to_tuple() != before
