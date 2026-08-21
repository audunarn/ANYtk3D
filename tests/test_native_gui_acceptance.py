"""Opt-in Windows desktop acceptance tests using native mouse input.

Tk ``event_generate`` deliberately fills unspecified state fields with zero.
That makes it excellent for deterministic unit tests but unable to reproduce
platform event-state translation bugs.  These tests drive the real widget
through Win32 mouse/key input while Tk's main loop is running.

Run explicitly on an interactive Windows desktop with::

    $env:ANYTK3D_RUN_GUI_TESTS = "1"
    $env:ANYTK3D_RUN_NATIVE_GUI = "1"
    python -m pytest tests/test_native_gui_acceptance.py -q

The suite temporarily moves the pointer and restores it after every action.
It is opt-in so CI agents and ordinary test runs are never disturbed.
"""

from __future__ import annotations

import ctypes
import os
import sys
import tkinter as tk
from typing import Callable, Iterable

import pytest

from anytk3d import (
    PickBinding,
    Point3D,
    SelectionConfig,
    SelectionDepth,
    SelectionFilter,
    SelectionGesture,
    SelectionOperation,
    SelectionTool,
    Tkinter3DCanvas,
)


RUN_NATIVE_GUI = (
    sys.platform == "win32"
    and os.environ.get("ANYTK3D_RUN_NATIVE_GUI", "").casefold()
    in {"1", "true", "yes"}
)

pytestmark = [
    pytest.mark.native_gui,
    pytest.mark.skipif(
        not RUN_NATIVE_GUI,
        reason=(
            "set ANYTK3D_RUN_NATIVE_GUI=1 on an interactive Windows desktop"
        ),
    ),
]


class _NativePoint(ctypes.Structure):
    _fields_ = (("x", ctypes.c_long), ("y", ctypes.c_long))


class NativeInput:
    """Schedule real Win32 input while Tk owns the foreground main loop."""

    LEFT = (0x0002, 0x0004)
    RIGHT = (0x0008, 0x0010)
    MIDDLE = (0x0020, 0x0040)
    SHIFT = 0x10
    CONTROL = 0x11
    ALT = 0x12

    def __init__(self, root: tk.Tk, widget: tk.Misc) -> None:
        self.root = root
        self.widget = widget
        self.user32 = ctypes.windll.user32

    def _screen(self, point: tuple[int, int]) -> tuple[int, int]:
        return (
            int(self.widget.winfo_rootx()) + int(point[0]),
            int(self.widget.winfo_rooty()) + int(point[1]),
        )

    def _move(self, point: tuple[int, int]) -> None:
        assert self.user32.SetCursorPos(*self._screen(point))

    def _mouse(self, flag: int, data: int = 0) -> None:
        self.user32.mouse_event(flag, 0, 0, data, 0)

    def _key(self, virtual_key: int, down: bool) -> None:
        self.user32.keybd_event(virtual_key, 0, 0 if down else 0x0002, 0)

    def run(self, actions: Iterable[tuple[int, Callable[[], None]]]) -> None:
        actions = tuple(actions)
        previous = _NativePoint()
        assert self.user32.GetCursorPos(ctypes.byref(previous))
        wrapper = self.user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.user32.SetForegroundWindow(wrapper)
        self.root.focus_force()
        self.root.update_idletasks()
        latest = max((delay for delay, _action in actions), default=0)
        for delay, action in actions:
            self.root.after(delay, action)
        self.root.after(latest + 250, self.root.quit)
        try:
            self.root.mainloop()
        finally:
            # Fail-safe releases keep a failed assertion/test interruption from
            # leaving the engineer's mouse or modifiers logically held.
            for _down, up in (self.LEFT, self.MIDDLE, self.RIGHT):
                self._mouse(up)
            for key in (self.SHIFT, self.CONTROL, self.ALT):
                self._key(key, False)
            self.user32.SetCursorPos(previous.x, previous.y)
            self.root.attributes("-topmost", False)
            self.root.update()

    def hover(self, point: tuple[int, int]) -> None:
        nearby = (point[0] + 25, point[1] + 25)
        self.run(
            (
                (30, lambda: self._move(nearby)),
                (100, lambda: self._move(point)),
            )
        )

    def click(
        self,
        point: tuple[int, int],
        *,
        modifier: int | None = None,
        button: tuple[int, int] = LEFT,
    ) -> None:
        actions: list[tuple[int, Callable[[], None]]] = [
            (30, lambda: self._move(point)),
        ]
        if modifier is not None:
            actions.append((80, lambda: self._key(modifier, True)))
        actions.extend(
            (
                (130, lambda: self._mouse(button[0])),
                (220, lambda: self._mouse(button[1])),
            )
        )
        if modifier is not None:
            actions.append((270, lambda: self._key(modifier, False)))
        self.run(actions)

    def drag(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        *,
        button: tuple[int, int] = LEFT,
    ) -> None:
        actions: list[tuple[int, Callable[[], None]]] = [
            (30, lambda: self._move(start)),
            (100, lambda: self._mouse(button[0])),
        ]
        steps = 12
        for index in range(1, steps + 1):
            fraction = index / steps
            point = (
                round(start[0] + (end[0] - start[0]) * fraction),
                round(start[1] + (end[1] - start[1]) * fraction),
            )
            actions.append((100 + index * 25, lambda point=point: self._move(point)))
        actions.append((100 + (steps + 2) * 25, lambda: self._mouse(button[1])))
        self.run(actions)

    def wheel(self, point: tuple[int, int], delta: int) -> None:
        self.run(
            (
                (30, lambda: self._move(point)),
                (100, lambda: self._mouse(0x0800, delta)),
            )
        )


@pytest.fixture(scope="module")
def native_root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("an interactive Windows desktop is required")
    root.geometry("760x560+120+80")
    root.title("ANYtk3D native GUI acceptance")
    root.update()
    yield root
    root.destroy()


@pytest.fixture
def native_canvas(native_root):
    root = native_root
    canvas = Tkinter3DCanvas(
        root,
        width=720,
        height=520,
        interaction_profile="commercial",
    )
    canvas.pack(fill="both", expand=True)
    canvas.add_markers(
        [Point3D(-1.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0)],
        size=9,
        bindings=[
            PickBinding.one("point1", "geometry.vertex", 30),
            PickBinding.one("point2", "geometry.vertex", 30),
        ],
    )
    canvas.set_top_view()
    canvas.fit_to_scene()
    canvas.redraw()
    root.update()
    try:
        yield root, canvas, NativeInput(root, canvas.canvas)
    finally:
        canvas.destroy()
        root.update()


def _project(canvas: Tkinter3DCanvas, x: float) -> tuple[int, int]:
    point = canvas.camera.project_point(
        Point3D(x, 0.0, 0.0), canvas._plot_width(), canvas.height
    )
    assert point is not None
    return round(point[0]), round(point[1])


def test_native_hover_click_and_modifier_operations(native_canvas):
    root, canvas, native = native_canvas
    events = []
    hovered = []
    canvas.configure_selection(
        events.append,
        hover_callback=lambda hit: hovered.append(None if hit is None else hit.key),
        config=SelectionConfig(
            filter=SelectionFilter(kinds=frozenset({"geometry.vertex"})),
            tool=SelectionTool.SINGLE,
            click_on_press=True,
        ),
    )
    first = _project(canvas, -1.0)
    second = _project(canvas, 1.0)

    native.hover(first)
    assert "point1" in hovered

    native.click(first)
    assert events[-1].gesture == SelectionGesture.CLICK
    assert events[-1].operation == SelectionOperation.REPLACE
    assert [hit.key for hit in events[-1].hits] == ["point1"]

    native.click(second, modifier=native.SHIFT)
    assert events[-1].operation == SelectionOperation.ADD
    assert [hit.key for hit in events[-1].hits] == ["point2"]

    native.click(first, modifier=native.CONTROL)
    assert events[-1].operation == SelectionOperation.TOGGLE
    assert [hit.key for hit in events[-1].hits] == ["point1"]

    native.click(second, modifier=native.ALT)
    assert events[-1].operation == SelectionOperation.REMOVE
    assert [hit.key for hit in events[-1].hits] == ["point2"]


def test_native_directional_window_and_crossing(native_canvas):
    _root, canvas, native = native_canvas
    events = []
    canvas.configure_selection(
        events.append,
        config=SelectionConfig(
            filter=SelectionFilter(kinds=frozenset({"geometry.vertex"})),
            tool=SelectionTool.BOX,
            depth=SelectionDepth.VISIBLE,
            click_on_press=True,
        ),
    )
    first = _project(canvas, -1.0)
    second = _project(canvas, 1.0)
    left = min(first[0], second[0]) - 40
    right = max(first[0], second[0]) + 40
    top = min(first[1], second[1]) - 40
    bottom = max(first[1], second[1]) + 40

    native.drag((left, top), (right, bottom))
    window = next(
        event for event in reversed(events) if event.gesture == SelectionGesture.WINDOW
    )
    assert {hit.key for hit in window.hits} == {"point1", "point2"}

    native.drag((right, bottom), (left, top))
    crossing = next(
        event
        for event in reversed(events)
        if event.gesture == SelectionGesture.CROSSING
    )
    assert {hit.key for hit in crossing.hits} == {"point1", "point2"}
    assert canvas._selection_overlay is None


def test_native_middle_pan_right_orbit_and_wheel_zoom(native_canvas):
    _root, canvas, native = native_canvas
    centre = (canvas._plot_width() // 2, canvas.height // 2)

    target_before = canvas.camera.target.to_tuple()
    native.drag(
        (centre[0] - 30, centre[1]),
        (centre[0] + 30, centre[1] + 20),
        button=native.MIDDLE,
    )
    assert canvas.camera.target.to_tuple() != target_before

    position_before = canvas.camera.position.to_tuple()
    native.drag(
        (centre[0] - 30, centre[1]),
        (centre[0] + 30, centre[1] + 20),
        button=native.RIGHT,
    )
    assert canvas.camera.position.to_tuple() != position_before

    distance_before = (canvas.camera.position - canvas.camera.target).length()
    native.wheel(centre, 120)
    distance_after = (canvas.camera.position - canvas.camera.target).length()
    assert distance_after < distance_before
