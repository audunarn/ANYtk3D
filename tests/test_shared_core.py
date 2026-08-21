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
        "MeshArrays",
        "MeshHandle",
        "ViewerCapabilities",
        "ViewerScheduler",
        "ApplicationOwner",
        "DirtyGenerations",
        "ModelOwner",
        "PackedOwnerTable",
        "RetainedViewer",
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


def retained_canvas():
    canvas = Tkinter3DCanvas.__new__(Tkinter3DCanvas)
    canvas.objects = []
    canvas._occlude_lines = False
    canvas._invalidate_geometry_cache = lambda: None
    canvas._request_redraw = lambda: None
    return canvas


def retained_triangle(**fields):
    values = {
        "positions": np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
        ),
        "triangles": np.asarray([[0, 1, 2]], dtype=np.uint32),
    }
    values.update(fields)
    return any3dview.MeshArrays(**values)


def test_indexed_retained_mesh_reuses_geometry_for_scalar_only_updates():
    canvas = retained_canvas()
    handle = canvas.add_mesh_arrays(
        retained_triangle(element_scalars=np.asarray([0.0], np.float32)),
        scalar_range=(0.0, 1.0),
    )

    first = canvas._object_to_primitives(canvas.objects[0], "full")[0]
    first_vertices = first["vertices"]
    first_color = first["colors"]
    handle.update_element_scalars(np.asarray([1.0], np.float32))
    second = canvas._object_to_primitives(canvas.objects[0], "full")[0]

    assert second["vertices"] is first_vertices
    assert second["colors"] != first_color
    assert handle.generations.scalar == 1


def test_active_mask_and_remove_update_the_retained_software_scene():
    canvas = retained_canvas()
    handle = canvas.add_mesh_arrays(retained_triangle())

    assert len(canvas._object_to_primitives(canvas.objects[0], "full")) == 1
    handle.set_active_elements([False])
    assert canvas._object_to_primitives(canvas.objects[0], "full") == []

    handle.remove()
    assert canvas.objects == []


def test_packed_owner_is_resolved_only_for_selection():
    canvas = retained_canvas()
    binding = any3dview.PickBinding.one("element:7", "mesh.element")
    owners = any3dview.PackedOwnerTable.from_owners(triangles=[binding])
    canvas.add_mesh_arrays(retained_triangle(), owners=owners)

    primitive = canvas._object_to_primitives(canvas.objects[0], "full")[0]
    scene = canvas._compile([primitive])

    assert scene.face_bindings == [None]
    assert scene.face_binding(0) == binding
    assert scene.face_binding(0).owners[0].identity is None


def test_packed_line_and_point_owners_are_resolved_lazily():
    canvas = retained_canvas()
    line_binding = any3dview.PickBinding.one("member:4", "mesh.member")
    point_binding = any3dview.PickBinding.one("node:9", "mesh.node")
    owners = any3dview.PackedOwnerTable.from_owners(
        lines=[line_binding], points=[point_binding]
    )
    arrays = any3dview.MeshArrays(
        np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        np.empty((0, 3), dtype=np.uint32),
        lines=np.asarray([[0, 1]], dtype=np.uint32),
        point_indices=np.asarray([1], dtype=np.uint32),
    )
    canvas.add_mesh_arrays(arrays, owners=owners)

    scene = canvas._compile(canvas._object_to_primitives(canvas.objects[0], "full"))

    assert scene.line_bindings == [None]
    assert scene.marker_bindings == [None]
    assert scene.line_binding(0) == line_binding
    assert scene.marker_binding(0) == point_binding
