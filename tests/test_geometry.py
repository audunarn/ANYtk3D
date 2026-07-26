'''
Headless tests for the 3D math and colour helpers (no Tk display
needed).
'''
import math

import pytest

from anytk3d import Point3D
from anytk3d.canvas import (
    Tkinter3DCanvas,
    _hex_to_rgb,
    _interpolate_thickness_color,
    _rgb_to_hex,
)


def test_point3d_vector_algebra():
    a = Point3D(1.0, 0.0, 0.0)
    b = Point3D(0.0, 1.0, 0.0)

    assert a.dot(b) == 0.0
    assert a.cross(b).to_tuple() == (0.0, 0.0, 1.0)
    assert Point3D(3.0, 4.0, 0.0).length() == pytest.approx(5.0)
    assert Point3D(10.0, 0.0, 0.0).normalized().to_tuple() == (1.0, 0.0, 0.0)


def test_point3d_rotations():
    p = Point3D(1.0, 0.0, 0.0)
    rotated = p.rotate_z(math.pi / 2)

    assert rotated.x == pytest.approx(0.0, abs=1e-12)
    assert rotated.y == pytest.approx(1.0)


def test_hex_rgb_roundtrip():
    assert _rgb_to_hex(*_hex_to_rgb('#0a1b2c')) == '#0a1b2c'


def test_interpolate_thickness_color_endpoints_differ():
    low = _interpolate_thickness_color(0.0, 0.0, 1.0)
    high = _interpolate_thickness_color(1.0, 0.0, 1.0)
    mid = _interpolate_thickness_color(0.5, 0.0, 1.0)

    assert low.startswith('#') and high.startswith('#') and mid.startswith('#')
    assert len({low, mid, high}) == 3


def test_two_sided_cylinder_shell_does_not_occlude_backside_members():
    # A two-sided (back_color) shell must not act as an opaque occluder.
    canvas = Tkinter3DCanvas.__new__(Tkinter3DCanvas)
    canvas._explicit_opaque_cylinder_occluders = []
    canvas.objects = [
        {
            "type": "cylinder",
            "radius": 1.0,
            "height": 10.0,
            "center": Point3D(0.0, 0.0, 0.0),
            "opacity": 1.0,
            "show_backfaces": True,
            "back_color": "brown",
        }
    ]

    assert canvas._collect_opaque_cylinder_occluders() == []

    occluder = {"radius": 1.0, "height": 10.0, "center": Point3D(0.0, 0.0, 0.0)}
    assert canvas._primitive_hidden_by_opaque_cylinder(
        {"kind": "polygon", "layer": 20, "center": Point3D(0.0, -0.5, 0.0)},
        (occluder,),
        Point3D(0.0, -4.0, 0.0),
    )


def _bare_canvas():
    '''A Tkinter3DCanvas with only the fields the compiler needs, no display.'''
    canvas = Tkinter3DCanvas.__new__(Tkinter3DCanvas)
    canvas.objects = []
    canvas._explicit_opaque_cylinder_occluders = []
    canvas._thickness_legend = None
    canvas._world_primitive_cache = {}
    canvas._scene_cache = {}
    canvas.show_axis_ruler = False
    canvas._occlude_lines = True
    return canvas


def test_plate_front_and_back_colours_reach_the_compiled_scene():
    # Renderer feature contract carried over from ANYstructure: a polygon may
    # carry a separate colour for its far side, and a two-sided shell keeps
    # both halves instead of culling the back.
    canvas = _bare_canvas()
    canvas.objects = [
        {
            'type': 'polygon',
            'vertices': [Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(1, 1, 0)],
            'color': '#123456',
            'back_color': '#654321',
            'outline': 'black',
            'width': 1,
            'two_sided_shell': True,
        }
    ]

    scene = canvas._compile(canvas._get_world_primitives('full'))

    assert scene.base_front == ['#123456']
    assert scene.base_back == ['#654321']
    assert not scene.face_cull[0]
    # two_sided_shell faces are bracketed around the rest of the scene.
    assert scene.face_phase[0] == 0


def test_transparent_or_two_sided_cylinder_keeps_its_back_faces():
    canvas = _bare_canvas()
    canvas.objects = [
        {
            'type': 'cylinder',
            'radius': 1.0,
            'height': 2.0,
            'center': Point3D(0.0, 0.0, 0.0),
            'color': 'gray',
            'opacity': 0.4,
            'segments': 8,
            'height_segments': 2,
        }
    ]
    primitives = canvas._object_to_primitives(canvas.objects[0], 'full')
    assert primitives and not any(p['cull_backface'] for p in primitives)

    # An opaque shell without a back colour only needs its camera-facing half.
    canvas.objects[0]['opacity'] = 1.0
    canvas._invalidate_geometry_cache()
    primitives = canvas._object_to_primitives(canvas.objects[0], 'full')
    assert primitives and all(p['cull_backface'] for p in primitives)

    # A back colour means the shell is meant to be seen from inside.
    canvas.objects[0]['back_color'] = 'brown'
    primitives = canvas._object_to_primitives(canvas.objects[0], 'full')
    assert primitives and not any(p['cull_backface'] for p in primitives)
