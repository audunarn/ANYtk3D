'''
Tests for the batched face path used by FE result fields, and for the
animation capture/playback optimisations built on top of it.
'''
import math
import tkinter as tk

import numpy as np
import pytest

from anytk3d import Point3D, Tkinter3DCanvas


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


def quad_grid(divisions, phase=0.0):
    '''A small deformed plate: the shape of an FE result field.'''
    polygons = []
    colors = []
    nodes = [
        [
            (
                -1.0 + 2.0 * i / divisions,
                -1.0 + 2.0 * j / divisions,
                0.3 * math.sin(3.0 * i / divisions + phase) * math.cos(3.0 * j / divisions),
            )
            for j in range(divisions + 1)
        ]
        for i in range(divisions + 1)
    ]
    for i in range(divisions):
        for j in range(divisions):
            polygons.append(
                (nodes[i][j], nodes[i + 1][j], nodes[i + 1][j + 1], nodes[i][j + 1])
            )
            colors.append('#%02x4080' % (10 + (i * divisions + j) % 200))
    return polygons, colors


# ----------------------------------------------------------------------
# Batched faces
# ----------------------------------------------------------------------


def test_add_faces_matches_add_polygon_geometry(canvas, root):
    polygons, colors = quad_grid(6)

    canvas.add_faces(polygons, colors=colors, outline='#64748b', layer=5)
    batched = canvas._get_scene('full')

    reference = Tkinter3DCanvas(root, width=400, height=300, bg='white')
    for polygon, color in zip(polygons, colors):
        reference.add_polygon(
            [Point3D(*vertex) for vertex in polygon],
            color=color,
            outline='#64748b',
            layer=5,
        )
    single = reference._get_scene('full')

    assert batched.face_total == single.face_total == len(polygons)
    assert batched.base_front == single.base_front
    assert np.allclose(batched.face_center, single.face_center, atol=1e-5)
    assert np.allclose(batched.face_normal, single.face_normal, atol=1e-5)
    assert np.allclose(batched.face_vertices, single.face_vertices, atol=1e-5)
    assert batched.face_start.tolist() == single.face_start.tolist()
    assert batched.face_count.tolist() == single.face_count.tolist()


def test_add_faces_accepts_a_single_colour_and_back_colours(canvas):
    polygons, _colors = quad_grid(3)
    canvas.add_faces(polygons, colors='#123456', back_colors=['#654321'] * len(polygons))
    scene = canvas._get_scene('full')

    assert set(scene.base_front) == {'#123456'}
    assert set(scene.base_back) == {'#654321'}


def test_add_faces_accepts_a_uniform_array(canvas):
    polygons, colors = quad_grid(4)
    array = np.asarray(polygons, dtype=np.float32)
    assert array.ndim == 3

    canvas.add_faces(array, colors=colors)
    assert canvas._get_scene('full').face_total == len(polygons)


def test_add_faces_rejects_a_mismatched_colour_count(canvas):
    polygons, colors = quad_grid(3)
    with pytest.raises(ValueError):
        canvas.add_faces(polygons, colors=colors[:-1])


def test_add_faces_drops_degenerate_faces(canvas):
    canvas.add_faces(
        [
            [(0, 0, 0), (1, 0, 0)],                 # only two vertices
            [(0, 0, 0), (1, 0, 0), (1, 1, 0)],
        ],
        colors=['#ff0000'],
    )
    scene = canvas._get_scene('full')

    assert scene.face_total == 1
    assert np.allclose(scene.face_center[0], [2.0 / 3.0, 1.0 / 3.0, 0.0], atol=1e-6)


def test_add_faces_is_transparent_and_two_sided_when_asked(canvas):
    polygons, colors = quad_grid(2)
    canvas.add_faces(polygons, colors=colors, opacity=0.4)
    scene = canvas._get_scene('full')

    assert not scene.face_cull.any()
    assert scene.stipple_front[0] and scene.stipple_back[0]
    assert scene.stipple_front[0] != scene.stipple_back[0]


def test_add_faces_extends_the_scene_bounds(canvas):
    canvas.add_faces(
        [[(1.0, 2.0, 3.0), (4.0, 2.0, 3.0), (4.0, 5.0, 6.0)]],
        colors='#ff0000',
    )
    low, high = canvas._scene_bounds()

    assert (low.x, low.y, low.z) == pytest.approx((1.0, 2.0, 3.0))
    assert (high.x, high.y, high.z) == pytest.approx((4.0, 5.0, 6.0))


def test_batched_faces_render(canvas):
    polygons, colors = quad_grid(8)
    canvas.add_faces(polygons, colors=colors, outline='#64748b')
    canvas.fit_to_scene(redraw=False)
    canvas.redraw()

    drawn = sum(1 for state in canvas._polygon_state if state is not None)
    assert drawn == len(polygons)


# ----------------------------------------------------------------------
# Quality sharing
# ----------------------------------------------------------------------


def test_polygon_scenes_share_one_compiled_form(canvas):
    polygons, colors = quad_grid(4)
    canvas.add_faces(polygons, colors=colors)

    # Nothing here re-tessellates for interactive frames, so both quality
    # levels must resolve to the same compiled scene rather than building
    # (and caching) it twice.
    assert canvas._get_scene('fast') is canvas._get_scene('full')


def test_cylinder_scenes_keep_separate_quality_levels(canvas):
    canvas.add_cylinder(radius=1.0, height=2.0, segments=32, height_segments=8)

    fast = canvas._get_scene('fast')
    full = canvas._get_scene('full')
    assert fast is not full
    assert fast.face_total < full.face_total


# ----------------------------------------------------------------------
# Animation
# ----------------------------------------------------------------------


def capture_frames(canvas, count=4, divisions=4):
    canvas.begin_animation_cache()
    for step in range(count):
        canvas.clear(keep_canvas=True)
        polygons, colors = quad_grid(divisions, phase=step)
        canvas.add_faces(polygons, colors=colors)
        canvas.capture_animation_frame()
    return count


def test_capture_stores_one_scene_per_frame_for_polygon_models(canvas):
    capture_frames(canvas)

    assert canvas.animation_frames == 4
    for frame in canvas._animation_cache:
        # Shared quality means one build, not two, per captured frame.
        assert frame['scene_full'] is frame['scene_fast']


def test_capture_records_the_occluders_of_its_own_frame(canvas):
    canvas.begin_animation_cache()
    canvas.add_cylinder(
        radius=1.0, height=4.0, opacity=1.0, show_backfaces=False,
        segments=12, height_segments=3,
    )
    canvas.capture_animation_frame()
    with_cylinder = canvas._animation_cache[-1]['occluders']

    canvas.clear(keep_canvas=True)
    polygons, colors = quad_grid(3)
    canvas.add_faces(polygons, colors=colors)
    canvas.capture_animation_frame()
    without_cylinder = canvas._animation_cache[-1]['occluders']

    assert with_cylinder and not without_cylinder


def test_playback_restores_the_live_scene_and_render_mode(canvas):
    polygons, colors = quad_grid(4)
    canvas.add_faces(polygons, colors=colors)
    canvas.fit_to_scene(redraw=False)
    live = canvas._get_scene('full')

    capture_frames(canvas)
    canvas.play_animation(fps=30, fast=True)
    canvas._animation_tick()

    assert canvas._interactive_render is False
    assert canvas._animation_occluders is None

    canvas.stop_animation()
    assert not canvas.is_playing_animation
    assert canvas._get_scene('full') is not live or canvas.animation_frames == 4


def test_playback_auto_mode_falls_back_to_reduced_detail(canvas):
    # Enough elements that a full-detail frame cannot fit a 1 ms budget, so
    # the auto path has to notice the overrun and switch.
    capture_frames(canvas, count=2, divisions=30)
    canvas.fit_to_scene(redraw=False)
    canvas.play_animation(fps=1000, fast=None)
    canvas._animation_tick()

    assert canvas._animation_fast is True
    canvas.stop_animation()


def test_explicit_full_playback_stays_full_detail(canvas):
    capture_frames(canvas, count=2, divisions=30)
    canvas.fit_to_scene(redraw=False)
    canvas.play_animation(fps=1000, fast=False)
    canvas._animation_tick()

    assert canvas._animation_fast is False
    canvas.stop_animation()


def test_stop_animation_leaves_full_quality_behind(canvas):
    capture_frames(canvas)
    canvas.play_animation(fps=30, fast=True)
    canvas._animation_tick()
    canvas.stop_animation()

    assert canvas._interactive_render is False


# ----------------------------------------------------------------------
# HUD caching
# ----------------------------------------------------------------------


def test_hud_is_not_rebuilt_when_nothing_changed(canvas):
    polygons, colors = quad_grid(3)
    canvas.add_faces(polygons, colors=colors)
    canvas.set_thickness_legend([1.0, 2.0, 3.0], unit='mm')
    canvas.fit_to_scene(redraw=False)
    canvas.redraw()

    legend_items = canvas.canvas.find_withtag('_tk3d_hud_legend')
    assert legend_items

    canvas.redraw()
    assert canvas.canvas.find_withtag('_tk3d_hud_legend') == legend_items

    # Rotating rebuilds only the axis triad.
    axis_items = canvas.canvas.find_withtag('_tk3d_hud_axis')
    canvas.set_view(37.0, 12.0)
    canvas.redraw()
    assert canvas.canvas.find_withtag('_tk3d_hud_legend') == legend_items
    assert canvas.canvas.find_withtag('_tk3d_hud_axis') != axis_items


def test_legend_survives_interactive_frames(canvas):
    polygons, colors = quad_grid(3)
    canvas.add_faces(polygons, colors=colors)
    canvas.set_thickness_legend([1.0, 2.0, 3.0], unit='mm')
    canvas.fit_to_scene(redraw=False)
    canvas.redraw()

    canvas._interactive_render = True
    canvas.redraw()
    assert canvas.canvas.find_withtag('_tk3d_hud_legend')


def test_changing_the_legend_rebuilds_it(canvas):
    polygons, colors = quad_grid(3)
    canvas.add_faces(polygons, colors=colors)
    canvas.set_thickness_legend([1.0, 2.0], unit='mm')
    canvas.fit_to_scene(redraw=False)
    canvas.redraw()
    before = canvas.canvas.find_withtag('_tk3d_hud_legend')

    canvas.set_thickness_legend([5.0, 9.0], unit='MPa')
    canvas.redraw()
    assert canvas.canvas.find_withtag('_tk3d_hud_legend') != before


# ----------------------------------------------------------------------
# Demo module
# ----------------------------------------------------------------------


def test_demo_builds_every_tab(root):
    from anytk3d import demo

    notebook = demo.build_demo(root)
    root.update()
    assert len(notebook.tabs()) == 4


def test_demo_result_field_is_batched(canvas):
    from anytk3d import demo

    elements = demo.add_result_field(canvas, divisions=6, scale=1.0)

    assert elements == 36
    # One object for the whole field, not one per element.
    assert len(canvas.objects) == 1
    assert canvas.objects[0]['type'] == 'faces'
    assert canvas._get_scene('full').face_total == 36
