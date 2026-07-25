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


def test_renderer_supports_plate_front_back_colours():
    # Renderer feature contract carried over from ANYstructure.
    from pathlib import Path

    import anytk3d.canvas as canvas_module

    source = Path(canvas_module.__file__).read_text(encoding="utf-8")

    assert "back_color: str = \"\"" in source
    assert "\"back_color\": back_color" in source
    assert "fill_color = primitive[\"color\"]" in source
    assert "fill_color = primitive.get(\"back_color\") or fill_color" in source
    assert "show_backfaces = bool(back_color) or opacity < 0.90" in source
