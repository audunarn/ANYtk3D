'''
GUI tests - create a real Tkinter3DCanvas and populate the demo scenes.
Skipped automatically when no display is available.
'''
from dataclasses import replace
import tkinter as tk

import numpy as np
import pytest

from any3dview import MeshArrays
from anytk3d import (
    Point3D,
    Tkinter3DCanvas,
    populate_fe_gui_cylinder,
    populate_fe_gui_plate,
    populate_stiffened_cylinder,
    populate_stiffened_plate,
)
from anytk3d.canvas import _mix_color


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


def test_legend_accepts_result_palette_colors(root):
    canvas = Tkinter3DCanvas(root, width=500, height=300)
    canvas.pack()
    levels = [0.0, 50.0, 100.0]
    colors = ["#440154", "#21918c", "#fde725"]
    canvas.set_thickness_legend(
        levels, unit="MPa", title="von Mises", colors=colors
    )
    root.update()

    assert canvas._thickness_legend["colors"] == colors
    assert canvas._legend_color(canvas._thickness_legend, 50.0) == colors[1]


def test_camera_orbit_and_zoom(root):
    canvas = Tkinter3DCanvas(root, width=400, height=300, bg='white')
    canvas.add_cylinder(radius=1.0, height=4.0, center=Point3D(0.0, 0.0, 2.0))
    canvas.redraw()

    before = canvas.camera.position.to_tuple()
    canvas.camera.orbit(0.3, 0.1)
    canvas.camera.zoom(0.8)
    canvas.redraw()

    assert canvas.camera.position.to_tuple() != before


def test_shared_viewer_contract_on_real_tk_widget(root):
    canvas = Tkinter3DCanvas(root, width=320, height=180, bg="white")
    canvas.pack()
    root.update()

    canvas.set_section_plane((1, 0, 0), -1.0)
    state = canvas.export_view_state()
    replacement = replace(
        state,
        background="#112233",
        section_plane=None,
        mesh_lines=False,
        interaction_profile="commercial",
    )
    canvas.apply_view_state(replacement, redraw=False)

    assert canvas.backend_name == "software"
    assert canvas.event_widget is canvas.canvas
    assert canvas.viewport_size[0] > 1
    assert canvas.viewport_size[1] > 1
    assert canvas.project_point(canvas.camera.target) is not None
    assert canvas.section_plane is None
    assert canvas.bg == "#112233"
    assert canvas.interaction_profile == "commercial"


def test_viewer_destroy_is_idempotent(root):
    canvas = Tkinter3DCanvas(root, width=160, height=100)
    canvas.pack()
    root.update_idletasks()

    canvas.destroy()
    canvas.destroy()


def test_retained_selected_elements_are_visually_distinct(root):
    canvas = Tkinter3DCanvas(
        root, width=320, height=220, bg="white", shading=False
    )
    canvas.pack()
    mesh = MeshArrays(
        np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            dtype=np.float32,
        ),
        np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.uint32),
        triangle_to_element=np.asarray([0, 1], dtype=np.uint32),
    )
    handle = canvas.add_mesh_arrays(
        mesh, color="#224466", cull_backface=False
    )
    handle.set_selected_elements((1,))
    canvas.fit_to_scene(redraw=False)
    canvas.redraw()
    root.update()

    fills = {
        state[0]
        for state in canvas._polygon_state
        if state is not None
    }
    assert fills == {
        "#224466",
        _mix_color("#224466", canvas._pick.highlight_fill, 0.65),
    }


def test_capture_image_from_mapped_native_tk_viewport(root):
    pytest.importorskip("PIL.ImageGrab")
    root.geometry("360x260+50+50")
    root.deiconify()
    root.attributes("-topmost", True)
    canvas = Tkinter3DCanvas(root, bg="#f4f7fb")
    canvas.pack(fill="both", expand=True)
    canvas.add_box(1.0, color="#336699", outline="#102030")
    canvas.fit_to_scene(redraw=False)
    canvas.redraw()
    root.update()

    assert canvas.event_widget.winfo_ismapped()
    expected_size = canvas.viewport_size
    image = canvas.capture_image()

    assert image.mode == "RGBA"
    assert image.size == expected_size
    assert image.getbbox() is not None
