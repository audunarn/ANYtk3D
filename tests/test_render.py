'''
Tests for the render pipeline: depth ordering, culling, near-plane
clipping, lighting and layered transparency.

Most of these drive the widget without needing a visible window, but the
pipeline needs a real ``tkinter.Canvas`` to hold its item pool, so they
skip when no display is available.
'''
import math
import tkinter as tk

import numpy as np
import pytest

from anytk3d import Light, Point3D, Tkinter3DCanvas, shapes, stipple
from anytk3d.shading import face_shade


@pytest.fixture
def root():
    try:
        window = tk.Tk()
    except tk.TclError:
        pytest.skip('no display available for tkinter')
    window.withdraw()
    yield window
    window.destroy()


@pytest.fixture
def canvas(root):
    widget = Tkinter3DCanvas(root, width=400, height=300, bg='white')
    widget.pack()
    root.update()
    return widget


def drawn_order(canvas):
    '''Face indices in the order they were handed to Tk, far to near.'''
    scene = canvas._get_scene('fast' if canvas._interactive_render else 'full')
    basis = canvas.camera.basis()
    canvas._apply_lighting(scene, basis)
    right, up, forward = basis
    position = canvas.camera.position
    y_scale = 1.0 / math.tan(canvas.camera.fov / 2.0)
    plot_width = canvas._plot_width()
    origin = np.array([position.x, position.y, position.z], dtype=np.float32)
    matrix = np.array(
        [
            [right.x, up.x, forward.x],
            [right.y, up.y, forward.y],
            [right.z, up.z, forward.z],
        ],
        dtype=np.float32,
    )
    order, front, _coords, _clipped = canvas._visible_faces(
        scene,
        origin,
        matrix,
        y_scale * canvas.height / plot_width,
        y_scale,
        0.5 * plot_width,
        0.5 * canvas.height,
        max(1.0e-9, canvas.camera.near),
        plot_width,
        [],
        False,
    )
    return scene, order, front


def test_faces_are_painted_back_to_front(canvas):
    # Two separated boxes: whichever is farther must be drawn first, and the
    # order has to flip when the camera orbits to the other side.
    canvas.add_box(1.0, 1.0, 1.0, center=Point3D(-3.0, 0.0, 0.0), color='#ff0000')
    canvas.add_box(1.0, 1.0, 1.0, center=Point3D(3.0, 0.0, 0.0), color='#0000ff')
    canvas.fit_to_scene(redraw=False)

    def near_box_drawn_last(azimuth):
        canvas.set_view(azimuth, 0.0)
        scene, order, _front = drawn_order(canvas)
        depths = [
            float(np.linalg.norm(
                scene.face_center[face]
                - np.array(canvas.camera.position.to_tuple(), dtype=np.float32)
            ))
            for face in order
        ]
        # Painter order: monotonically decreasing distance from the camera.
        return all(a >= b - 1.0e-3 for a, b in zip(depths, depths[1:]))

    assert near_box_drawn_last(0.0)
    assert near_box_drawn_last(180.0)


def test_section_plane_clips_overlay_lines_markers_and_text(canvas, root):
    canvas.add_line(
        Point3D(-2, 0, 0), Point3D(2, 0, 0), draw_overlay=True
    )
    canvas.add_markers([Point3D(-1, 0, 0), Point3D(1, 0, 0)])
    canvas.add_text(Point3D(-1, 0, 0), "clipped")
    canvas.add_text(Point3D(1, 0, 0), "kept")
    canvas.fit_to_scene(redraw=False)
    canvas.set_section_plane((1, 0, 0), 0)
    canvas.redraw()
    root.update()

    assert len(canvas._line_pool) == 1
    assert len(canvas._marker_pool) == 1
    assert len(canvas._text_pool) == 1
    assert canvas.canvas.itemcget(canvas._text_pool[0], "text") == "kept"


def test_orbiting_swaps_which_object_is_in_front(canvas):
    canvas.add_box(1.0, 1.0, 1.0, center=Point3D(-3.0, 0.0, 0.0), color='#ff0000')
    canvas.add_box(1.0, 1.0, 1.0, center=Point3D(3.0, 0.0, 0.0), color='#0000ff')
    canvas.fit_to_scene(redraw=False)

    def last_face_colour(azimuth):
        canvas.set_view(azimuth, 0.0)
        scene, order, front = drawn_order(canvas)
        last = order[-1]
        return scene.fill_front[last] if front[last] else scene.fill_back[last]

    # The camera sits on +X at azimuth 0, so the +X box is nearest there.
    near_at_0 = last_face_colour(0.0)
    near_at_180 = last_face_colour(180.0)
    assert near_at_0 != near_at_180


def test_closed_solids_cull_their_back_faces(canvas):
    canvas.add_sphere(1.0, segments=16, rings=12, color='#4e79a7')
    canvas.fit_to_scene(redraw=False)

    scene, order, _front = drawn_order(canvas)

    assert scene.face_cull.all()
    # Roughly half a closed convex solid faces away from the camera.
    assert 0.3 * scene.face_total < len(order) < 0.7 * scene.face_total


def test_transparent_solids_keep_their_back_faces(canvas):
    canvas.add_sphere(1.0, segments=16, rings=12, color='#4e79a7', opacity=0.4)
    canvas.fit_to_scene(redraw=False)

    scene, order, _front = drawn_order(canvas)

    assert not scene.face_cull.any()
    assert len(order) == scene.face_total


def test_transparent_front_and_back_use_non_overlapping_stipples(canvas):
    canvas.add_sphere(1.0, segments=12, rings=8, color='#4e79a7', opacity=0.4)
    scene = canvas._get_scene('full')

    front = scene.stipple_front[0]
    back = scene.stipple_back[0]
    assert front and back and front != back

    # The two windows must not light the same pixels, or the far wall of a
    # transparent surface contributes nothing.
    start_a, count_a = stipple.window(0.4, 0)
    start_b, count_b = stipple.window(0.4, 1)
    assert start_a + count_a <= start_b


def test_explicit_stipple_is_passed_through_unchanged(canvas):
    canvas.add_polygon(
        [Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(1, 1, 0)],
        color='#123456',
        stipple='gray50',
    )
    scene = canvas._get_scene('full')

    assert scene.stipple_front[0] == 'gray50'
    assert scene.stipple_back[0] == 'gray50'


def test_layered_stipple_matches_alpha_compositing():
    for opacity in (0.25, 0.5, 0.75):
        start_a, count_a = stipple.window(opacity, 0)
        start_b, count_b = stipple.window(opacity, 1)
        covered = (count_a + count_b) / 64.0
        assert covered == pytest.approx(1.0 - (1.0 - opacity) ** 2, abs=0.02)
        assert start_a + count_a == start_b


def test_lighting_darkens_faces_turned_away_from_the_light(canvas):
    canvas.add_box(2.0, 2.0, 2.0, color='#808080')
    # Point the light straight down the +Z axis so the box's top face is fully
    # lit and its bottom face is fully shadowed.
    canvas.set_light(direction=Point3D(0.0, 0.0, 1.0), specular=0.0)
    scene = canvas._get_scene('full')
    canvas._apply_lighting(scene, canvas.camera.basis())

    light = canvas.light
    direction = light.world_direction(canvas.camera.basis())
    shades = face_shade(scene.face_normal, light, direction, None)

    assert shades.max() > shades.min()
    # A face turned toward the light keeps its literal base colour, so a
    # thickness or utilisation colour code still matches its legend.
    assert shades.max() == pytest.approx(light.ambient + light.diffuse, abs=1e-6)
    assert shades.min() == pytest.approx(light.ambient, abs=1e-6)
    lit_index = int(np.argmax(shades))
    assert scene.fill_front[lit_index] == '#808080'
    assert scene.fill_front[int(np.argmin(shades))] != '#808080'


def test_shading_can_be_switched_off(canvas):
    canvas.add_box(2.0, 2.0, 2.0, color='#808080')
    canvas.set_shading(False)
    scene = canvas._get_scene('full')
    canvas._apply_lighting(scene, canvas.camera.basis())

    assert set(scene.fill_front) == {'#808080'}


def test_camera_following_light_updates_when_the_camera_moves(canvas):
    canvas.add_sphere(1.0, segments=16, rings=12, color='#4e79a7')
    canvas.set_light(follow_camera=True)
    scene = canvas._get_scene('full')

    canvas.set_view(0.0, 0.0)
    canvas._apply_lighting(scene, canvas.camera.basis())
    before = list(scene.fill_front)

    canvas.set_view(90.0, 0.0)
    canvas._apply_lighting(scene, canvas.camera.basis())

    assert before != scene.fill_front


def test_world_fixed_light_is_cached_across_camera_moves(canvas):
    canvas.add_sphere(1.0, segments=16, rings=12, color='#4e79a7')
    scene = canvas._get_scene('full')

    canvas.set_view(0.0, 0.0)
    canvas._apply_lighting(scene, canvas.camera.basis())
    key_before = scene.shade_key
    fills = scene.fill_front

    canvas.set_view(90.0, 30.0)
    canvas._apply_lighting(scene, canvas.camera.basis())

    assert scene.shade_key == key_before
    assert scene.fill_front is fills


def test_faces_crossing_the_near_plane_are_clipped_not_dropped(canvas):
    # A large plate with the camera sitting inside its span: every face that
    # straddles the near plane must survive as a clipped polygon.
    canvas.add_plane(20.0, 20.0, nx=4, ny=4, color='#59a14f')
    canvas.camera.set_target(Point3D(0.0, 0.0, 0.0))
    canvas.camera.set_orbit(azimuth=0.0, elevation=math.radians(2.0), distance=1.0)
    canvas.camera.near = 1.0e-4

    scene, order, _front = drawn_order(canvas)

    assert len(order) > 0
    _scene2, order2, _f2, clipped = _visible(canvas)
    assert clipped, 'expected at least one near-plane clipped face'
    assert set(clipped).issubset(set(order2))


def _visible(canvas):
    scene = canvas._get_scene('full')
    basis = canvas.camera.basis()
    canvas._apply_lighting(scene, basis)
    right, up, forward = basis
    position = canvas.camera.position
    y_scale = 1.0 / math.tan(canvas.camera.fov / 2.0)
    plot_width = canvas._plot_width()
    origin = np.array([position.x, position.y, position.z], dtype=np.float32)
    matrix = np.array(
        [
            [right.x, up.x, forward.x],
            [right.y, up.y, forward.y],
            [right.z, up.z, forward.z],
        ],
        dtype=np.float32,
    )
    order, front, _coords, clipped = canvas._visible_faces(
        scene,
        origin,
        matrix,
        y_scale * canvas.height / plot_width,
        y_scale,
        0.5 * plot_width,
        0.5 * canvas.height,
        max(1.0e-9, canvas.camera.near),
        plot_width,
        [],
        False,
    )
    return scene, order, front, clipped


def test_interactive_level_of_detail_keeps_the_largest_faces(canvas):
    for index in range(40):
        canvas.add_sphere(
            0.3, center=Point3D(index * 0.8, 0.0, 0.0), segments=12, rings=8,
            color='#4e79a7',
        )
    canvas.fit_to_scene(redraw=False)
    canvas._interactive_render = True
    canvas.set_interactive_detail(200)
    canvas.redraw()
    root_faces = sum(1 for state in canvas._polygon_state if state is not None)

    assert 0 < root_faces <= 200


def test_lines_without_overlay_are_depth_sorted_with_geometry(canvas):
    canvas.add_box(2.0, 2.0, 2.0, color='#4e79a7')
    canvas.add_line(Point3D(-4.0, 0.0, 0.0), Point3D(4.0, 0.0, 0.0), color='#ff0000')
    scene = canvas._get_scene('full')

    # The line joins the face pipeline, not the always-on-top overlay pool.
    assert scene.face_is_edge.sum() == 1
    assert len(scene.line_color) == 0

    canvas.set_occlude_lines(False)
    scene = canvas._get_scene('full')
    assert scene.face_is_edge.sum() == 0
    assert scene.line_color == ['#ff0000']


def test_overlay_lines_stay_on_top(canvas):
    canvas.add_box(2.0, 2.0, 2.0, color='#4e79a7')
    canvas.add_line(
        Point3D(-4.0, 0.0, 0.0), Point3D(4.0, 0.0, 0.0),
        color='#ff0000', draw_overlay=True,
    )
    scene = canvas._get_scene('full')

    assert scene.line_color == ['#ff0000']
    assert scene.face_is_edge.sum() == 0


def test_members_behind_an_opaque_shell_are_hidden(canvas):
    canvas.add_cylinder(
        radius=2.0, height=4.0, center=Point3D(0.0, 0.0, 0.0),
        opacity=1.0, show_backfaces=False, segments=16, height_segments=4,
    )
    canvas.add_ring_stiffener(radius=2.0, z_position=0.0, segments=16, inside=True)
    canvas.fit_to_scene(redraw=False)
    canvas.set_view(0.0, 0.0)

    scene = canvas._get_scene('full')
    origin = np.array(canvas.camera.position.to_tuple(), dtype=np.float32)
    hidden = canvas._faces_hidden_by_occluders(
        scene, canvas._collect_opaque_cylinder_occluders(), origin
    )

    assert hidden.any()
    # Only member layers take part; the shell itself is never filtered out.
    assert not hidden[~scene.face_occludable].any()


def test_redraw_is_stable_across_a_full_orbit(canvas):
    canvas.add_sphere(1.2, segments=20, rings=14, color='#4e79a7')
    canvas.add_box(1.0, 1.0, 1.0, center=Point3D(2.5, 0.0, 0.0), color='#e15759')
    canvas.add_torus(2.0, 0.2, color='#59a14f')
    canvas.add_line(Point3D(-3, -3, -3), Point3D(3, 3, 3), color='#333333')
    canvas.add_text(Point3D(0.0, 0.0, 2.0), 'top')
    canvas.fit_to_scene(redraw=False)

    for step in range(0, 360, 20):
        canvas.set_view(float(step), 20.0)
        canvas.redraw()

    assert any(state is not None for state in canvas._polygon_state)


def test_mesh_api_accepts_plain_tuples(canvas):
    mesh = shapes.box(1.0)
    canvas.add_mesh(mesh.vertices, mesh.faces, color='#4e79a7')
    canvas.redraw()

    assert canvas.objects[0]['type'] == 'mesh'
    assert canvas._get_scene('full').face_total == 6


def test_fit_to_scene_frames_a_wide_flat_model_tightly(canvas):
    # A bounding-sphere fit would pull the camera back by the box diagonal and
    # leave a wide, flat model floating in an empty viewport.
    canvas.add_plane(20.0, 20.0, nx=2, ny=2, color='#59a14f')
    canvas.set_view(-45.0, 25.0)
    canvas.fit_to_scene(redraw=False)

    corners = [
        Point3D(x, y, 0.0)
        for x in (-10.0, 10.0)
        for y in (-10.0, 10.0)
    ]
    projected = [
        canvas.camera.project_point(corner, canvas._plot_width(), canvas.height)
        for corner in corners
    ]
    assert all(point is not None for point in projected)

    xs = [point[0] for point in projected]
    ys = [point[1] for point in projected]
    width = canvas._plot_width()
    # Everything on screen ...
    assert min(xs) >= 0.0 and max(xs) <= width
    assert min(ys) >= 0.0 and max(ys) <= canvas.height
    # ... and filling most of the limiting dimension.
    fill = max((max(xs) - min(xs)) / width, (max(ys) - min(ys)) / canvas.height)
    assert fill > 0.6


def test_axis_ruler_adds_labelled_lines(canvas):
    canvas.add_box(2.0, 2.0, 2.0, color='#4e79a7')
    canvas.set_axis_ruler(True)
    canvas.fit_to_scene(redraw=False)
    scene = canvas._get_scene('full')

    assert len(scene.line_color) >= 3
    labels = {text for text, _color, _font, _anchor in scene.text_content}
    assert {'x [m]', 'y [m]', 'z [m]'} <= labels

    canvas.set_axis_ruler(False)
    assert len(canvas._get_scene('full').line_color) == 0


def test_animation_cache_replays_without_disturbing_the_live_scene(canvas):
    canvas.add_sphere(1.0, segments=12, rings=8, color='#4e79a7')
    canvas.fit_to_scene(redraw=False)

    canvas.begin_animation_cache()
    for offset in (0.0, 0.5, 1.0):
        canvas.clear(keep_canvas=True)
        canvas.add_sphere(
            1.0, center=Point3D(offset, 0.0, 0.0), segments=12, rings=8, color='#4e79a7'
        )
        canvas.capture_animation_frame()
    assert len(canvas._animation_cache) == 3

    live = canvas._get_scene('full')
    canvas.play_animation(fps=60)
    canvas._animation_tick()
    canvas.stop_animation()

    assert canvas._get_scene('full') is live


def test_light_can_be_reconfigured(canvas):
    canvas.set_light(direction=Point3D(1.0, 0.0, 0.0), ambient=0.2, diffuse=0.8)

    assert canvas.light.direction.x == pytest.approx(1.0)
    assert canvas.light.ambient == pytest.approx(0.2)
    assert isinstance(canvas.light, Light)
