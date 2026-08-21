"""Commercial interaction profile and projected selection queries."""

from __future__ import annotations

import tkinter as tk

import pytest

from anytk3d import (
    PickBinding,
    PickOwner,
    Point3D,
    SelectionConfig,
    SelectionDepth,
    SelectionFilter,
    SelectionGesture,
    SelectionOperation,
    SelectionTool,
    Tkinter3DCanvas,
)
from anytk3d._selection import ProjectedPrimitive, ProjectedSelectionIndex
from anytk3d.picking import operation_from_modifiers


def binding(key: str, kind: str = "geometry.face", priority: int = 0):
    return PickBinding.one(key, kind, priority)


def square(index: int, key: str, depth: float):
    return ProjectedPrimitive(
        index=index,
        shape="polygon",
        points=((20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)),
        depths=(depth,) * 4,
        binding=binding(key),
    )


def test_public_selection_values_validate_and_filter():
    owner = PickOwner("face7", "geometry.face", priority=10)
    value = PickBinding((owner, PickOwner("element3", "mesh.element")))
    assert value.owners[0] == owner
    assert SelectionFilter(kinds=frozenset({"geometry.face"})).accepts(owner)
    assert not SelectionFilter(kinds=frozenset({"mesh.node"})).accepts(owner)
    with pytest.raises(ValueError, match="unique"):
        PickBinding((owner, owner))


def test_point_stack_marks_only_the_front_primitive_visible():
    index = ProjectedSelectionIndex(
        [square(0, "front", 2.0), square(1, "back", 5.0)], 100, 100
    )
    hits = index.point_hits(50, 50, SelectionFilter(), radius=0)
    assert [hit.key for hit in hits] == ["front", "back"]
    assert [hit.visible for hit in hits] == [True, False]


def test_visible_rectangle_stops_at_front_and_through_returns_both():
    index = ProjectedSelectionIndex(
        [square(0, "front", 2.0), square(1, "back", 5.0)], 100, 100
    )
    visible = index.rectangle_hits(
        (10, 10, 90, 90),
        SelectionFilter(),
        crossing=False,
        depth=SelectionDepth.VISIBLE,
    )
    through = index.rectangle_hits(
        (10, 10, 90, 90),
        SelectionFilter(),
        crossing=False,
        depth=SelectionDepth.THROUGH,
    )
    assert [hit.key for hit in visible] == ["front"]
    assert {hit.key for hit in through} == {"front", "back"}


def test_window_requires_the_whole_owner_but_crossing_needs_one_primitive():
    shared = binding("polyline", "geometry.edge")
    index = ProjectedSelectionIndex(
        [
            ProjectedPrimitive(0, "segment", ((10, 20), (40, 20)), (2, 2), shared),
            ProjectedPrimitive(1, "segment", ((40, 20), (90, 20)), (2, 2), shared),
        ],
        100,
        100,
    )
    window = index.rectangle_hits(
        (0, 0, 50, 50), SelectionFilter(), crossing=False, depth=SelectionDepth.THROUGH
    )
    crossing = index.rectangle_hits(
        (0, 0, 50, 50), SelectionFilter(), crossing=True, depth=SelectionDepth.THROUGH
    )
    assert window == ()
    assert [hit.key for hit in crossing] == ["polyline"]


def test_lasso_uses_the_polygon_not_only_its_bounding_box():
    inside = ProjectedPrimitive(
        0, "point", ((20, 20),), (2,), binding("inside", "mesh.node"), radius=2
    )
    outside_triangle = ProjectedPrimitive(
        1, "point", ((75, 75),), (2,), binding("outside", "mesh.node"), radius=2
    )
    index = ProjectedSelectionIndex([inside, outside_triangle], 100, 100)
    hits = index.polygon_hits(
        ((10, 10), (90, 10), (10, 90)),
        SelectionFilter(),
        depth=SelectionDepth.THROUGH,
    )
    assert [hit.key for hit in hits] == ["inside"]


def test_filtered_marker_does_not_hide_the_requested_face():
    face = square(0, "face", 3.0)
    marker = ProjectedPrimitive(
        1,
        "point",
        ((50, 50),),
        (2.0,),
        binding("node", "mesh.node", priority=20),
        radius=5,
        layer=32,
    )
    index = ProjectedSelectionIndex([face, marker], 100, 100)
    hits = index.point_hits(
        50,
        50,
        SelectionFilter(kinds=frozenset({"geometry.face"})),
        radius=4,
    )
    assert [hit.key for hit in hits] == ["face"]
    assert hits[0].visible


def test_screen_ray_and_plane_unprojection_are_inverse_at_target():
    # Test the public camera math without constructing Tk.
    from anytk3d import Camera3D

    camera = Camera3D()
    origin, direction = camera.screen_ray(200, 150, 400, 300)
    assert origin.to_tuple() == camera.position.to_tuple()
    assert direction.dot((camera.target - camera.position).normalized()) > 0.999999
    point = camera.unproject_to_plane(
        200, 150, 400, 300, camera.target, Point3D(0, 0, 1)
    )
    assert point is not None
    assert point.z == pytest.approx(camera.target.z)


@pytest.fixture(scope="module")
def root():
    try:
        window = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for tkinter")
    window.geometry("440x340+80+80")
    window.update()
    yield window
    window.destroy()


@pytest.fixture
def canvas(root):
    widget = Tkinter3DCanvas(
        root,
        width=400,
        height=300,
        bg="white",
        interaction_profile="commercial",
    )
    widget.pack()
    root.update()
    yield widget
    widget.destroy()
    root.update()


def test_default_profile_remains_legacy(root):
    widget = Tkinter3DCanvas(root, width=100, height=80)
    try:
        assert widget.interaction_profile == "legacy"
        widget.set_interaction_profile("commercial")
        assert widget.interaction_profile == "commercial"
        widget.set_interaction_profile("legacy")
        assert widget.interaction_profile == "legacy"
    finally:
        widget.destroy()


def populate_two_faces(canvas, root):
    polygons = [
        [(-2, -1, 0), (0, -1, 0), (0, 1, 0), (-2, 1, 0)],
        [(0, -1, 0), (2, -1, 0), (2, 1, 0), (0, 1, 0)],
    ]
    canvas.add_faces(
        polygons,
        bindings=[binding("left"), binding("right")],
        two_sided_shell=True,
    )
    canvas.add_markers(
        [(-1, 0, 0.05), (1, 0, 0.05)],
        bindings=[binding("node1", "mesh.node", 20), binding("node2", "mesh.node", 20)],
    )
    canvas.set_top_view()
    canvas.fit_to_scene()
    canvas.redraw()
    root.update()


def test_batched_face_and_marker_bindings_are_queryable(canvas, root):
    populate_two_faces(canvas, root)
    left = canvas.camera.project_point(Point3D(-1, 0, 0.05), canvas._plot_width(), canvas.height)
    assert left is not None
    hits = canvas.query_point(round(left[0]), round(left[1]))
    assert hits[0].key == "node1"
    assert "left" in {hit.key for hit in hits}


def test_projected_index_is_reused_until_the_view_changes(canvas, root):
    populate_two_faces(canvas, root)
    first = canvas._get_selection_index()
    assert canvas._get_selection_index() is first
    canvas.camera.orbit(0.1, 0.0)
    assert canvas._get_selection_index() is not first


def test_section_plane_removes_clipped_marker_from_projected_selection(canvas, root):
    canvas.add_markers(
        [Point3D(-1, 0, 0), Point3D(1, 0, 0)],
        bindings=[
            PickBinding.one("left", "mesh.node"),
            PickBinding.one("right", "mesh.node"),
        ],
    )
    canvas.fit_to_scene(redraw=False)
    canvas.set_section_plane((1, 0, 0), 0)
    root.update()

    owners = {
        owner.key
        for primitive in canvas._get_selection_index().primitives
        if primitive.binding is not None
        for owner in primitive.binding.owners
    }
    assert "right" in owners
    assert "left" not in owners


def test_commercial_lmb_click_and_directional_drag_emit_events(canvas, root):
    populate_two_faces(canvas, root)
    seen = []
    canvas.configure_selection(
        seen.append,
        config=SelectionConfig(
            filter=SelectionFilter(kinds=frozenset({"geometry.face"})),
            depth=SelectionDepth.THROUGH,
        ),
    )
    centre = canvas.camera.project_point(Point3D(-1, 0, 0), canvas._plot_width(), canvas.height)
    assert centre is not None
    x, y = round(centre[0]), round(centre[1])
    canvas.canvas.event_generate("<ButtonPress-1>", x=x, y=y)
    canvas.canvas.event_generate("<ButtonRelease-1>", x=x, y=y)
    root.update()
    assert seen[-1].gesture == SelectionGesture.CLICK
    assert seen[-1].hits[0].key == "left"

    canvas.canvas.event_generate("<ButtonPress-1>", x=390, y=290)
    canvas.canvas.event_generate("<B1-Motion>", x=10, y=10)
    canvas.canvas.event_generate("<ButtonRelease-1>", x=10, y=10)
    root.update()
    assert seen[-1].gesture == SelectionGesture.CROSSING
    assert {hit.key for hit in seen[-1].hits} == {"left", "right"}


def test_single_tool_can_commit_on_press_without_waiting_for_release(canvas, root):
    """CAD single-pick remains usable when Tk drops ButtonRelease."""

    populate_two_faces(canvas, root)
    seen = []
    canvas.configure_selection(
        seen.append,
        config=SelectionConfig(
            filter=SelectionFilter(kinds=frozenset({"mesh.node"})),
            tool=SelectionTool.SINGLE,
            click_on_press=True,
        ),
    )
    point = canvas.camera.project_point(
        Point3D(-1, 0, 0.05), canvas._plot_width(), canvas.height
    )
    assert point is not None
    x, y = round(point[0]), round(point[1])

    canvas.canvas.event_generate("<ButtonPress-1>", x=x, y=y)
    root.update()
    assert len(seen) == 1
    assert seen[0].gesture == SelectionGesture.CLICK
    assert seen[0].hits[0].key == "node1"

    # Motion cannot turn the explicit Single tool into a region, and a later
    # release must not apply the same click twice (important for Ctrl-toggle).
    canvas.canvas.event_generate("<B1-Motion>", x=x + 80, y=y + 80, state=0x0100)
    canvas.canvas.event_generate("<ButtonRelease-1>", x=x + 80, y=y + 80)
    root.update()
    assert len(seen) == 1
    assert canvas._selection_overlay is None


def test_toplevel_release_fallback_commits_active_window(canvas, root):
    """A release missed by the canvas must not strand the drag overlay."""

    populate_two_faces(canvas, root)
    seen = []
    canvas.configure_selection(
        seen.append,
        config=SelectionConfig(
            filter=SelectionFilter(kinds=frozenset({"geometry.face"})),
            depth=SelectionDepth.THROUGH,
        ),
    )
    canvas.canvas.event_generate("<ButtonPress-1>", x=10, y=10)
    canvas.canvas.event_generate("<B1-Motion>", x=390, y=290)
    root.update()
    assert canvas._selection_overlay is not None

    # Call the toplevel delivery path directly to reproduce the Windows/Tk
    # case where the canvas's own ButtonRelease binding is skipped.
    class Release:
        x_root = canvas.canvas.winfo_rootx() + 390
        y_root = canvas.canvas.winfo_rooty() + 290
        x = 0
        y = 0

    canvas._on_toplevel_selection_release(Release())
    root.update()

    assert seen[-1].gesture == SelectionGesture.WINDOW
    assert {hit.key for hit in seen[-1].hits} == {"left", "right"}
    assert canvas._selection_press is None
    assert canvas._selection_overlay is None


def test_commercial_navigation_uses_middle_pan_and_right_orbit(canvas, root):
    populate_two_faces(canvas, root)
    target_before = canvas.camera.target.to_tuple()
    position_before = canvas.camera.position.to_tuple()

    canvas.canvas.event_generate("<ButtonPress-2>", x=200, y=150)
    canvas.canvas.event_generate("<B2-Motion>", x=220, y=160)
    canvas.canvas.event_generate("<ButtonRelease-2>", x=220, y=160)
    root.update()
    assert canvas.camera.target.to_tuple() != target_before

    canvas.canvas.event_generate("<ButtonPress-3>", x=200, y=150)
    canvas.canvas.event_generate("<B3-Motion>", x=220, y=160)
    canvas.canvas.event_generate("<ButtonRelease-3>", x=220, y=160)
    root.update()
    assert canvas.camera.position.to_tuple() != position_before


def test_shift_ctrl_alt_map_to_add_toggle_remove(canvas, root, monkeypatch):
    from anytk3d import canvas as canvas_module

    # Low Mod1 is Alt on X11.  Windows uses its high bit and is covered by the
    # native-state regressions below.
    monkeypatch.setattr(canvas_module.sys, "platform", "linux")
    populate_two_faces(canvas, root)
    seen = []
    canvas.configure_selection(seen.append)
    centre = canvas.camera.project_point(Point3D(-1, 0, 0), canvas._plot_width(), canvas.height)
    assert centre is not None
    x, y = round(centre[0]), round(centre[1])
    for state, operation in (
        (0x0001, SelectionOperation.ADD),
        (0x0004, SelectionOperation.TOGGLE),
        (0x0008, SelectionOperation.REMOVE),
    ):
        canvas.canvas.event_generate("<ButtonPress-1>", x=x, y=y, state=state)
        canvas.canvas.event_generate("<ButtonRelease-1>", x=x, y=y, state=state)
        root.update()
        assert seen[-1].operation == operation


def test_stale_windows_alt_tracking_cannot_turn_plain_selection_into_remove(
    canvas, monkeypatch
):
    """Native menus may consume Alt KeyRelease while mouse state stays exact."""

    from types import SimpleNamespace
    from anytk3d import canvas as canvas_module

    monkeypatch.setattr(canvas_module.sys, "platform", "win32")
    canvas._tracked_modifiers["alt"] = True

    assert canvas._event_modifiers(SimpleNamespace(state=0)) == (
        False,
        False,
        False,
    )
    assert operation_from_modifiers(*canvas._event_modifiers(
        SimpleNamespace(state=0)
    )) == SelectionOperation.REPLACE
    # Tk 8.6 emits this low Mod1-looking bit for a native plain Windows click.
    assert canvas._event_modifiers(SimpleNamespace(state=0x0008)) == (
        False,
        False,
        False,
    )


def test_stale_windows_alt_mouse_bit_is_verified_against_physical_key(
    canvas, monkeypatch
):
    """A stale native Alt bit must not override plain or Shift selection."""

    from types import SimpleNamespace
    from anytk3d import canvas as canvas_module

    monkeypatch.setattr(canvas_module.sys, "platform", "win32")
    monkeypatch.setattr(canvas, "_windows_alt_is_down", lambda: False)

    assert canvas._event_modifiers(SimpleNamespace(state=0x20000)) == (
        False,
        False,
        False,
    )
    assert canvas._event_modifiers(SimpleNamespace(state=0x20001)) == (
        True,
        False,
        False,
    )

    monkeypatch.setattr(canvas, "_windows_alt_is_down", lambda: True)
    assert canvas._event_modifiers(SimpleNamespace(state=0x20000)) == (
        False,
        False,
        True,
    )


def test_stale_windows_alt_bit_cannot_remove_a_real_click(canvas, root, monkeypatch):
    """The complete click controller must still emit REPLACE with its hit."""

    from anytk3d import canvas as canvas_module

    populate_two_faces(canvas, root)
    seen = []
    canvas.configure_selection(
        seen.append,
        config=SelectionConfig(
            filter=SelectionFilter(kinds=frozenset({"mesh.node"}))
        ),
    )
    monkeypatch.setattr(canvas_module.sys, "platform", "win32")
    monkeypatch.setattr(canvas, "_windows_alt_is_down", lambda: False)
    point = canvas.camera.project_point(
        Point3D(-1, 0, 0.05), canvas._plot_width(), canvas.height
    )
    assert point is not None
    x, y = round(point[0]), round(point[1])

    canvas.canvas.event_generate("<ButtonPress-1>", x=x, y=y, state=0x20000)
    canvas.canvas.event_generate("<ButtonRelease-1>", x=x, y=y, state=0x20000)
    root.update()

    assert seen[-1].operation == SelectionOperation.REPLACE
    assert seen[-1].hits[0].key == "node1"


def test_macos_option_tracking_remains_available(canvas, monkeypatch):
    from types import SimpleNamespace
    from anytk3d import canvas as canvas_module

    monkeypatch.setattr(canvas_module.sys, "platform", "darwin")
    canvas._tracked_modifiers["alt"] = True
    assert canvas._event_modifiers(SimpleNamespace(state=0)) == (
        False,
        False,
        True,
    )
