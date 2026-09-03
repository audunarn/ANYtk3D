'''
The colour scale is shared module state that applications swap when the
user picks a different colour map, so it needs a real API rather than a
patchable constant.
'''
import inspect
import tkinter as tk

import pytest

import anytk3d
from anytk3d import Point3D, Tkinter3DCanvas
from anytk3d.canvas import _interpolate_thickness_color


VIRIDIS = (
    (0.0, '#440154'),
    (0.5, '#21918c'),
    (1.0, '#fde725'),
)


def test_legend_api_exposes_configurable_text_size_without_opening_tk():
    signature = inspect.signature(Tkinter3DCanvas.set_thickness_legend)

    assert signature.parameters["font_size"].default == 10


def test_legend_retains_requested_text_size_without_opening_tk():
    canvas = Tkinter3DCanvas.__new__(Tkinter3DCanvas)
    canvas._thickness_legend = None
    canvas._invalidate_geometry_cache = lambda: None
    canvas._request_redraw = lambda: None

    canvas.set_thickness_legend(
        (0.0, 1.0), title="Displacement", width=220, font_size=12
    )

    assert canvas._thickness_legend["width"] == 220
    assert canvas._thickness_legend["font_size"] == 12


@pytest.fixture(autouse=True)
def restored_scale():
    yield
    anytk3d.reset_color_stops()


def sample():
    return tuple(_interpolate_thickness_color(v, 0.0, 1.0) for v in (0.0, 0.5, 1.0))


def test_setting_stops_changes_the_interpolated_colours():
    before = sample()
    anytk3d.set_color_stops(VIRIDIS)

    assert sample() != before
    assert sample() == ('#440154', '#21918c', '#fde725')


def test_reset_restores_the_built_in_scale():
    default = sample()
    anytk3d.set_color_stops(VIRIDIS)
    anytk3d.reset_color_stops()

    assert sample() == default
    assert anytk3d.get_color_stops() == anytk3d.DEFAULT_COLOR_STOPS


def test_stops_are_sorted_and_clamped():
    stops = anytk3d.set_color_stops(
        ((1.5, '#ffffff'), (-0.4, '#000000'), (0.5, '#ff0000'))
    )

    assert [position for position, _color in stops] == [0.0, 0.5, 1.0]
    assert sample() == ('#000000', '#ff0000', '#ffffff')


@pytest.mark.parametrize('bad', [
    (),
    ((0.0, '#000000'),),
    ((0.0, '#000000'), (1.0, 'not-a-colour')),
    ((float('nan'), '#000000'), (1.0, '#ffffff')),
])
def test_invalid_scales_are_rejected(bad):
    with pytest.raises(ValueError):
        anytk3d.set_color_stops(bad)


def test_patching_the_legacy_constant_no_longer_silently_does_nothing():
    # It used to be the only way in, and it broke when the interpolation
    # moved modules.  The constant is now purely the documented default.
    import anytk3d.canvas as canvas_module

    before = sample()
    canvas_module._THICKNESS_COLOR_STOPS = VIRIDIS
    try:
        assert sample() == before, 'assignment must not be the supported route'
    finally:
        canvas_module._THICKNESS_COLOR_STOPS = anytk3d.DEFAULT_COLOR_STOPS
    assert callable(anytk3d.set_color_stops)


def test_thickness_color_helper_follows_the_scale():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no display available for tkinter')
    root.withdraw()
    try:
        canvas = Tkinter3DCanvas(root, width=200, height=150, bg='white')
        canvas.set_thickness_legend([0.0, 10.0])
        before = canvas.thickness_color(10.0)
        anytk3d.set_color_stops(VIRIDIS)
        assert canvas.thickness_color(10.0) != before
        assert canvas.thickness_color(10.0) == '#fde725'
    finally:
        root.destroy()


def test_changing_the_scale_rebuilds_a_cached_scene():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('no display available for tkinter')
    root.withdraw()
    try:
        canvas = Tkinter3DCanvas(root, width=200, height=150, bg='white')
        canvas.add_cylinder(
            radius=1.0, height=2.0, segments=8, height_segments=2,
            plate_thickness=[10.0, 20.0], center=Point3D(0.0, 0.0, 0.0),
        )
        before = list(canvas._get_scene('full').base_front)

        anytk3d.set_color_stops(VIRIDIS)
        after = list(canvas._get_scene('full').base_front)

        assert before != after, 'the cached scene kept the old colours'
    finally:
        root.destroy()
