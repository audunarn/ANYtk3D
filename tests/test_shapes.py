'''
Headless tests for the shape tessellation library.

The key contract is that every closed builder produces a watertight mesh
wound counter-clockwise seen from outside.  That is exactly what the
divergence theorem measures: summing ``dot(centroid, normal) * area / 3``
over the faces gives the enclosed volume, which is positive only when the
winding points outward, and matches the analytic volume only when the
surface is closed.
'''
import math

import pytest

from anytk3d import shapes
from anytk3d.core import Point3D


def enclosed_volume(mesh):
    total = 0.0
    for face in mesh.faces:
        points = [mesh.vertices[index] for index in face]
        normal = [0.0, 0.0, 0.0]
        for index in range(len(points)):
            a = points[index]
            b = points[(index + 1) % len(points)]
            normal[0] += a[1] * b[2] - a[2] * b[1]
            normal[1] += a[2] * b[0] - a[0] * b[2]
            normal[2] += a[0] * b[1] - a[1] * b[0]
        centroid = [sum(p[axis] for p in points) / len(points) for axis in range(3)]
        total += sum(centroid[axis] * 0.5 * normal[axis] for axis in range(3)) / 3.0
    return total


CLOSED_SHAPES = [
    ('box', shapes.box(2.0, 3.0, 4.0), 24.0),
    ('sphere', shapes.sphere(1.5, 32, 24), 4.0 / 3.0 * math.pi * 1.5 ** 3),
    ('cylinder', shapes.cylinder(1.0, 3.0, 48, 3), math.pi * 3.0),
    ('cone', shapes.cone(1.0, 2.0, 48), math.pi * 2.0 / 3.0),
    ('frustum', shapes.frustum(1.0, 0.4, 2.0, 48, 4),
     math.pi * 2.0 / 3.0 * (1.0 + 0.4 + 0.16)),
    ('tube', shapes.tube(1.0, 0.6, 2.0, 48, 2), math.pi * (1.0 - 0.36) * 2.0),
    ('torus', shapes.torus(2.0, 0.5, 48, 24), 2.0 * math.pi ** 2 * 2.0 * 0.25),
    ('pyramid', shapes.pyramid(1.0, 2.0, 4), 4.0 / 3.0),
    ('wedge', shapes.wedge(2.0, 1.0, 1.0), 1.0),
    ('rectangular_tube', shapes.rectangular_tube(0.4, 0.3, 0.02, 3.0), 0.0792),
]


@pytest.mark.parametrize('name,mesh,expected', CLOSED_SHAPES,
                         ids=[item[0] for item in CLOSED_SHAPES])
def test_closed_shapes_are_watertight_and_outward_wound(name, mesh, expected):
    volume = enclosed_volume(mesh)
    assert volume > 0.0, f'{name} is wound inward'
    # Faceting always loses a little volume against the analytic solid.
    assert 0.9 * expected <= volume <= 1.001 * expected


@pytest.mark.parametrize('kind', ['FB', 'T', 'I', 'L', 'C'])
def test_profile_prisms_have_the_section_area(kind):
    profile = shapes.profile_section(kind, 0.3, 0.02, 0.15, 0.02)
    length = 3.0
    mesh = shapes.prism(profile, length)
    area = abs(shapes.polygon_area_2d(profile))

    assert enclosed_volume(mesh) == pytest.approx(area * length, rel=1e-9)


def test_profile_sections_are_simple_and_counter_clockwise():
    for kind in shapes.PROFILE_KINDS:
        profile = shapes.profile_section(kind, 0.3, 0.02, 0.15, 0.02)
        assert len(profile) >= 4
        assert shapes.polygon_area_2d(shapes.ensure_ccw(profile)) > 0.0


def test_triangulation_covers_a_concave_profile():
    profile = shapes.ensure_ccw(shapes.profile_section('I', 0.3, 0.02, 0.15, 0.02))
    triangles = shapes.triangulate_polygon(profile)

    covered = sum(
        abs(shapes.polygon_area_2d([profile[i] for i in triangle]))
        for triangle in triangles
    )
    assert covered == pytest.approx(abs(shapes.polygon_area_2d(profile)), rel=1e-9)


def test_beam_web_follows_the_requested_up_direction():
    mesh = shapes.beam(
        Point3D(0.0, 0.0, 0.0),
        Point3D(5.0, 0.0, 0.0),
        kind='T',
        web_height=1.0,
        web_thickness=0.1,
        flange_width=0.5,
        flange_thickness=0.1,
        up=Point3D(0.0, 0.0, 1.0),
    )
    low, high = mesh.bounds()

    assert low.x == pytest.approx(0.0)
    assert high.x == pytest.approx(5.0)
    # The section sits on the path line and rises along `up`.
    assert low.z == pytest.approx(0.0)
    assert high.z == pytest.approx(1.0)


def test_box_beam_is_hollow():
    mesh = shapes.beam(
        Point3D(0.0, 0.0, 0.0),
        Point3D(3.0, 0.0, 0.0),
        kind='BOX',
        web_height=0.4,
        flange_width=0.3,
        flange_thickness=0.02,
    )
    solid = 0.3 * 0.4 * 3.0
    volume = enclosed_volume(mesh)

    assert 0.0 < volume < 0.5 * solid


def test_flat_shapes_face_positive_z():
    for mesh in (shapes.disk(1.0), shapes.disk(1.0, 0.5), shapes.plane(2.0, 2.0, 3, 3)):
        for face in mesh.faces:
            a, b, c = (mesh.vertices[index] for index in face[:3])
            e1 = [b[i] - a[i] for i in range(3)]
            e2 = [c[i] - a[i] for i in range(3)]
            normal_z = e1[0] * e2[1] - e1[1] * e2[0]
            assert normal_z > 0.0


def test_placed_maps_local_z_onto_the_requested_axis():
    mesh = shapes.cylinder(0.5, 4.0, 8).placed(
        origin=Point3D(1.0, 2.0, 3.0), axis=Point3D(1.0, 0.0, 0.0)
    )
    low, high = mesh.bounds()

    assert low.x == pytest.approx(-1.0)
    assert high.x == pytest.approx(3.0)
    assert low.y == pytest.approx(1.5)
    assert high.y == pytest.approx(2.5)


def test_arrow_points_from_start_to_end():
    mesh = shapes.arrow(Point3D(0.0, 0.0, 0.0), Point3D(0.0, 0.0, 4.0))
    low, high = mesh.bounds()

    assert low.z == pytest.approx(0.0, abs=1e-9)
    assert high.z == pytest.approx(4.0, abs=1e-9)
    # The head is a cone, so the top ring collapses to a single apex vertex.
    apexes = [v for v in mesh.vertices if abs(v[2] - 4.0) < 1e-9]
    assert len(apexes) == 1


def test_grid_lines_span_the_requested_extent():
    segments = shapes.grid_lines(size_x=10.0, size_y=6.0, step=1.0, z=0.5)
    xs = [point.x for segment in segments for point in segment]
    ys = [point.y for segment in segments for point in segment]

    assert len(segments) == 11 + 7
    assert min(xs) == pytest.approx(-5.0)
    assert max(xs) == pytest.approx(5.0)
    assert min(ys) == pytest.approx(-3.0)
    assert max(ys) == pytest.approx(3.0)
    assert all(point.z == pytest.approx(0.5) for segment in segments for point in segment)
