'''
Picking, hover and highlight.

The highlight state machine is tested without a display.  The click and hover
paths drive real Tk events, which requires a mapped window: Tk silently
discards synthetic button events aimed at an unviewable one.

One module-scoped root is used throughout.  Creating and destroying a Tk root
per test is unreliable on Windows, and a second root in the same interpreter is
not a second independent main window.
'''
import tkinter as tk

import pytest

from anytk3d import Point3D, Tkinter3DCanvas
from anytk3d.picking import Pick, PickState, entity_tag_at


@pytest.fixture(scope='module')
def root():
    try:
        window = tk.Tk()
    except tk.TclError:
        pytest.skip('no display available for tkinter')
    window.geometry('420x320+60+60')
    window.update()
    yield window
    window.destroy()


@pytest.fixture
def canvas(root):
    widget = Tkinter3DCanvas(root, width=400, height=300, bg='white')
    widget.pack()
    yield widget
    widget.destroy()
    root.update()


def show(canvas, root, **kwargs):
    '''Put one tagged box on screen, filling the view.'''

    canvas.add_box(2.0, 2.0, 0.1, center=Point3D(0, 0, 0), **kwargs)
    canvas.fit_to_scene()
    canvas.redraw()
    root.update()


class _FakeScene:
    '''Just enough of a compiled scene for the highlight resolver.'''

    def __init__(self, tags):
        self.tags = list(tags)
        self.any_tags = any(tags)


# ----------------------------------------------------------------------
# highlight state, no display needed
# ----------------------------------------------------------------------
def test_no_highlight_resolves_to_none():
    state = PickState()
    assert state.highlighted_faces(_FakeScene(['a', 'b', ''])) is None


def test_highlight_resolves_matching_faces():
    state = PickState()
    scene = _FakeScene(['plate1', 'plate2', 'plate1', ''])
    assert state.set_highlight(['plate1']) is True
    assert state.highlighted_faces(scene) == frozenset({0, 2})


def test_highlight_matches_one_tag_among_several():
    state = PickState()
    scene = _FakeScene(['plate1 group_a', 'plate2 group_b'])
    state.set_highlight(['group_b'])
    assert state.highlighted_faces(scene) == frozenset({1})


def test_setting_the_same_highlight_reports_no_change():
    state = PickState()
    assert state.set_highlight(['a']) is True
    assert state.set_highlight(['a']) is False
    assert state.set_highlight(['a', 'b']) is True


def test_highlight_resolution_is_cached_per_generation():
    state = PickState()
    scene = _FakeScene(['a', 'b'])
    state.set_highlight(['a'])
    first = state.highlighted_faces(scene)
    assert state.highlighted_faces(scene) is first

    state.set_highlight(['b'])
    assert state.highlighted_faces(scene) == frozenset({1})


def test_changing_colour_alone_still_invalidates():
    state = PickState()
    assert state.set_highlight(['a']) is True
    assert state.set_highlight(['a'], fill='#123456') is True
    assert state.highlight_fill == '#123456'


def test_scene_without_tags_never_highlights():
    state = PickState()
    state.set_highlight(['a'])
    assert state.highlighted_faces(_FakeScene(['', ''])) is None


def test_pick_dataclass_carries_modifiers():
    pick = Pick(tag='a', item=1, x=2, y=3, shift=True)
    assert pick.shift and not pick.ctrl and not pick.alt


# ----------------------------------------------------------------------
# picking against a real canvas
# ----------------------------------------------------------------------
def test_pick_at_returns_the_caller_tag(canvas, root):
    show(canvas, root, tags='plate7')
    assert canvas.pick_at(200, 150) == 'plate7'


def test_pick_returns_none_off_the_model(canvas, root):
    canvas.add_box(0.2, 0.2, 0.2, center=Point3D(0, 0, 0), tags='small')
    canvas.fit_to_scene()
    canvas.redraw()
    root.update()
    assert canvas.pick_at(2, 2) is None


def test_pick_never_returns_a_renderer_pool_tag(canvas, root):
    # No caller tag at all: the face pool tag must not leak out.
    show(canvas, root)
    assert canvas.pick_at(200, 150) is None


def test_prefix_filters_foreign_tags(canvas, root):
    show(canvas, root, tags='other7')
    canvas.set_pick_callback(lambda pick: None, prefix='ent_')
    assert canvas.pick_at(200, 150) is None


def test_entity_tag_at_prefers_the_topmost_item(root):
    '''Painter order means the last item drawn is the nearest one.'''

    widget = tk.Canvas(root, width=200, height=200)
    widget.pack()
    widget.create_rectangle(0, 0, 100, 100, fill='blue', tags='far')
    widget.create_rectangle(0, 0, 100, 100, fill='red', tags='near')
    root.update()
    try:
        tag, item = entity_tag_at(widget, 50, 50)
        assert tag == 'near'
        assert item is not None
    finally:
        widget.destroy()


# ----------------------------------------------------------------------
# real events
# ----------------------------------------------------------------------
def test_click_fires_the_callback_but_a_drag_does_not(canvas, root):
    show(canvas, root, tags='plate7')

    picks = []
    canvas.set_pick_callback(picks.append)

    canvas.canvas.event_generate('<ButtonPress-1>', x=200, y=150)
    canvas.canvas.event_generate('<ButtonRelease-1>', x=200, y=150)
    assert [pick.tag for pick in picks] == ['plate7']

    picks.clear()
    canvas.canvas.event_generate('<ButtonPress-1>', x=200, y=150)
    canvas.canvas.event_generate('<ButtonRelease-1>', x=260, y=150)
    assert picks == []


def test_click_on_empty_space_fires_with_an_empty_tag(canvas, root):
    canvas.add_box(0.2, 0.2, 0.2, center=Point3D(0, 0, 0), tags='small')
    canvas.fit_to_scene()
    canvas.redraw()
    root.update()

    picks = []
    canvas.set_pick_callback(picks.append)
    canvas.canvas.event_generate('<ButtonPress-1>', x=3, y=3)
    canvas.canvas.event_generate('<ButtonRelease-1>', x=3, y=3)

    assert len(picks) == 1
    assert picks[0].tag == ''


def test_picking_is_off_until_a_callback_is_set(canvas, root):
    '''Existing applications must see no behaviour change.'''

    show(canvas, root, tags='plate7')
    canvas.canvas.event_generate('<ButtonPress-1>', x=200, y=150)
    canvas.canvas.event_generate('<ButtonRelease-1>', x=200, y=150)
    assert canvas._pick.press is None


def test_hover_callback_fires_only_on_change(canvas, root):
    show(canvas, root, tags='plate7')

    seen = []
    canvas.set_hover_callback(
        lambda pick: seen.append(None if pick is None else pick.tag)
    )

    canvas.canvas.event_generate('<Motion>', x=200, y=150)
    canvas.canvas.event_generate('<Motion>', x=201, y=150)
    assert seen == ['plate7']

    canvas.canvas.event_generate('<Motion>', x=3, y=3)
    assert seen == ['plate7', None]


def test_highlight_survives_a_redraw(canvas, root):
    show(canvas, root, tags='plate7', color='#4e79a7')

    def fills():
        return {
            canvas.canvas.itemcget(item, 'fill')
            for item in canvas.canvas.find_withtag('plate7')
        }

    plain = fills()
    canvas.set_highlight(['plate7'], fill='#ff0000')
    canvas.redraw()
    root.update()
    assert fills() == {'#ff0000'}

    # And again after another redraw, because the tint is applied while
    # rendering rather than by poking the Tk items once.
    canvas.redraw()
    root.update()
    assert fills() == {'#ff0000'}

    canvas.clear_highlight()
    canvas.redraw()
    root.update()
    assert fills() == plain


# ----------------------------------------------------------------------
# lines carry tags too
# ----------------------------------------------------------------------
def test_lines_are_pickable(canvas, root):
    canvas.add_line(Point3D(-1, 0, 0), Point3D(1, 0, 0),
                    color='#000000', width=5, tags='beam3')
    canvas.fit_to_scene()
    canvas.redraw()
    root.update()

    assert canvas.pick_at(200, 150) == 'beam3'


def test_overlay_lines_are_pickable(canvas, root):
    '''Lines drawn on top of the geometry keep their tag as well.'''

    canvas.set_occlude_lines(False)
    canvas.add_line(Point3D(-1, 0, 0), Point3D(1, 0, 0),
                    color='#000000', width=5, tags='beam4')
    canvas.fit_to_scene()
    canvas.redraw()
    root.update()

    assert canvas.pick_at(200, 150) == 'beam4'


def test_a_highlighted_line_is_tinted(canvas, root):
    canvas.add_line(Point3D(-1, 0, 0), Point3D(1, 0, 0),
                    color='#000000', width=5, tags='beam3')
    canvas.fit_to_scene()
    canvas.redraw()
    root.update()

    canvas.set_highlight(['beam3'], outline='#00ff00')
    canvas.redraw()
    root.update()
    outlines = {
        canvas.canvas.itemcget(item, 'outline')
        for item in canvas.canvas.find_withtag('beam3')
    }
    assert outlines == {'#00ff00'}
