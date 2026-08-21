from types import SimpleNamespace

import numpy as np

import any3dview
import anytk3d
from anytk3d.canvas import Tkinter3DCanvas


def test_legacy_exports_share_exact_core_identity():
    for name in (
        "Point3D",
        "Camera3D",
        "Mesh",
        "Light",
        "PickOwner",
        "PickBinding",
        "SelectionConfig",
        "SelectionDepth",
        "SelectionEvent",
        "SelectionFilter",
        "SelectionGesture",
        "SelectionHit",
        "SelectionOperation",
        "SelectionTool",
        "SectionPlane",
    ):
        assert getattr(anytk3d, name) is getattr(any3dview, name), name


def test_private_selection_compatibility_path_is_preserved():
    from anytk3d._selection import ProjectedPrimitive, ProjectedSelectionIndex

    assert ProjectedPrimitive is any3dview.ProjectedPrimitive
    assert ProjectedSelectionIndex is any3dview.ProjectedSelectionIndex


def bare_canvas():
    canvas = Tkinter3DCanvas.__new__(Tkinter3DCanvas)
    canvas._section_plane = None
    canvas._selection_index = object()
    canvas._selection_index_key = object()
    canvas._fast_polygon_target = 100
    canvas.height = 100
    canvas.camera = SimpleNamespace(distance=10.0)
    canvas._reset_selection_cycle = lambda: None
    canvas._request_redraw = lambda: None
    return canvas


def one_face_scene():
    return SimpleNamespace(
        face_total=1,
        face_start=np.asarray([0], dtype=np.int64),
        face_count=np.asarray([4], dtype=np.int64),
        face_vertices=np.asarray(
            [(-1, -1, 5), (1, -1, 5), (1, 1, 5), (-1, 1, 5)],
            dtype=np.float32,
        ),
        face_is_edge=np.asarray([False]),
        face_normal=np.asarray([(0, 0, -1)], dtype=np.float32),
        face_center=np.asarray([(0, 0, 5)], dtype=np.float32),
        face_cull=np.asarray([False]),
        face_phase=np.asarray([1], dtype=np.int8),
        face_layer=np.asarray([0], dtype=np.float32),
    )


def projected_section(canvas, scene):
    return canvas._visible_faces_with_section(
        scene,
        np.asarray((0, 0, 0), dtype=np.float32),
        np.eye(3, dtype=np.float32),
        1.0,
        1.0,
        50.0,
        50.0,
        0.1,
        100,
        False,
    )


def test_section_plane_clips_faces_and_invalidates_selection_cache():
    canvas = bare_canvas()
    canvas.set_section_plane((2, 0, 0), 0)
    assert canvas.section_plane is not None
    assert canvas.section_plane.normal.to_tuple() == (1.0, 0.0, 0.0)
    assert canvas._selection_index is None
    assert canvas._selection_index_key is None

    order, _front, _coords, clipped = projected_section(canvas, one_face_scene())
    assert order == [0]
    assert min(clipped[0][0::2]) >= 50

    canvas.set_section_plane((1, 0, 0), 2)
    order, _front, _coords, clipped = projected_section(canvas, one_face_scene())
    assert order == []
    assert clipped == {}

    canvas.clear_section_plane()
    assert canvas.section_plane is None


def test_disabled_section_plane_leaves_section_path_inactive():
    canvas = bare_canvas()
    canvas.set_section_plane(enabled=False)
    assert canvas.section_plane is not None
    assert not canvas.section_plane.enabled
