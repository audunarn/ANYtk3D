from types import SimpleNamespace
import importlib.util
import inspect

import numpy as np
import pytest

import any3dview
import anytk3d
from anytk3d.canvas import Tkinter3DCanvas


def _public_protocol_members():
    protocol = any3dview.ViewerBackend
    names = set(protocol.__dict__.get("__annotations__", ()))
    names.update(
        name
        for name, value in vars(protocol).items()
        if not name.startswith("_") and (callable(value) or isinstance(value, property))
    )
    return names


def _assert_shared_signature(contract, concrete, *, exact):
    expected = inspect.signature(contract).parameters
    actual = inspect.signature(concrete).parameters
    for name, parameter in expected.items():
        assert name in actual, name
        candidate = actual[name]
        assert candidate.kind is parameter.kind, name
        if parameter.default is not inspect.Parameter.empty:
            assert candidate.default == parameter.default, name
        elif exact:
            assert candidate.default is inspect.Parameter.empty, name
    if exact:
        assert tuple(actual) == tuple(expected)


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
        "Pick",
        "RetainedViewer",
        "ViewerBackend",
        "ViewerState",
    ):
        assert getattr(anytk3d, name) is getattr(any3dview, name), name


class _DummyCanvasWidget:
    def __init__(self, width=200, height=100):
        self.width = width
        self.height = height
        self.options = {"width": width, "height": height}

    def winfo_width(self):
        return self.width

    def winfo_height(self):
        return self.height

    def cget(self, name):
        return self.options[name]

    def configure(self, **options):
        self.options.update(options)

    def update_idletasks(self):
        return None

    def winfo_rootx(self):
        return 11

    def winfo_rooty(self):
        return 17


def contract_canvas():
    canvas = Tkinter3DCanvas.__new__(Tkinter3DCanvas)
    canvas.canvas = _DummyCanvasWidget()
    canvas.width = 200
    canvas.height = 100
    canvas.bg = "white"
    canvas.camera = any3dview.Camera3D()
    canvas.camera.set_target(any3dview.Point3D(0, 0, 0))
    canvas.camera.set_position(any3dview.Point3D(10, 0, 0))
    canvas._thickness_legend = None
    canvas._section_plane = any3dview.SectionPlane((1, 0, 0), -2.0)
    canvas._backend_diagnostics = ("software fallback",)
    canvas._light = any3dview.Light()
    canvas._shading_enabled = True
    canvas._occlude_lines = True
    canvas.show_mesh_lines = True
    canvas._show_axis_indicator = True
    canvas.show_axis_ruler = False
    canvas._interaction_profile = "commercial"
    canvas._selection_config = any3dview.SelectionConfig()
    canvas._pick = SimpleNamespace(preselection_key=None)
    canvas._animation_cache = []
    canvas._animation_frame_index = 0
    canvas._is_playing_animation = False
    canvas._interactive_render = False
    canvas.configure = lambda **_options: None
    canvas.set_interaction_profile = lambda value: setattr(
        canvas, "_interaction_profile", str(value)
    )
    canvas._invalidate_geometry_cache = lambda: None
    canvas._request_redraw = lambda **_options: None
    return canvas


def test_software_backend_exposes_shared_integration_contract():
    canvas = contract_canvas()

    assert canvas.backend_name == "software"
    assert canvas.event_widget is canvas.canvas
    assert canvas.viewport_size == (200, 100)
    assert canvas.backend_diagnostics == ("software fallback",)
    assert isinstance(canvas, any3dview.ViewerBackend)

    center = canvas.project_point(any3dview.Point3D(0, 0, 0))
    assert center == (100.0, 50.0, 10.0)
    assert canvas.project_points([]) == ()
    assert canvas.project_points([any3dview.Point3D(11, 0, 0)]) == (None,)


def test_protocol_covers_the_complete_documented_viewer_surface():
    required = {
        "backend_name", "backend_diagnostics", "capabilities", "event_widget",
        "viewport_size", "export_view_state", "apply_view_state", "screen_ray",
        "unproject_to_plane", "set_light", "set_shading", "set_occlude_lines",
        "set_thickness_legend", "set_axis_indicator", "set_axis_ruler",
        "set_mesh_lines", "configure_selection", "query_point", "query_rectangle",
        "query_lasso", "set_preselection", "set_highlight", "add_mesh_arrays",
        "add_faces", "add_line", "add_markers", "add_text", "add_shape",
        "add_box", "add_cylinder", "fit_to_scene", "reset_camera",
        "begin_animation_cache", "capture_animation_frame", "play_animation",
        "stop_animation", "capture_image", "clear", "redraw", "destroy",
    }
    assert required <= _public_protocol_members()


def test_tk_methods_exactly_match_the_shared_protocol_signatures():
    protocol = any3dview.ViewerBackend
    for name in sorted(_public_protocol_members()):
        if name in protocol.__dict__.get("__annotations__", ()):
            continue
        contract = getattr(protocol, name, None)
        concrete = getattr(Tkinter3DCanvas, name, None)
        assert concrete is not None, name
        if callable(contract):
            _assert_shared_signature(contract, concrete, exact=True)


def test_gpu_methods_accept_every_shared_call_and_retain_0_4_extensions():
    pytest.importorskip("moderngl")
    from any3dview.gpu import Any3DView

    protocol = any3dview.ViewerBackend
    for name in sorted(_public_protocol_members()):
        if name in protocol.__dict__.get("__annotations__", ()):
            continue
        contract = getattr(protocol, name, None)
        concrete = getattr(Any3DView, name, None)
        assert concrete is not None, name
        if callable(contract):
            _assert_shared_signature(contract, concrete, exact=False)

    rectangle = inspect.signature(Any3DView.query_rectangle).parameters
    assert rectangle["end"].default is None
    assert "config" in rectangle
    assert "config" in inspect.signature(Any3DView.query_point).parameters
    mesh = inspect.signature(Any3DView.add_mesh_arrays).parameters
    assert mesh["_appearance"].kind is inspect.Parameter.VAR_KEYWORD


def test_view_state_round_trips_camera_and_view_policy_without_scene_data():
    canvas = contract_canvas()
    state = canvas.export_view_state()

    assert isinstance(state, any3dview.ViewerState)
    assert state.camera_position.to_tuple() == (10.0, 0.0, 0.0)
    assert state.camera_target.to_tuple() == (0.0, 0.0, 0.0)
    assert state.section_plane == canvas.section_plane
    assert state.background == "white"
    assert state.interaction_profile == "commercial"

    replacement = any3dview.ViewerState(
        camera_position=any3dview.Point3D(0, -8, 2),
        camera_target=any3dview.Point3D(1, 2, 3),
        camera_world_up=any3dview.Point3D(0, 0, 1),
        fov=0.7,
        near=0.02,
        far=500.0,
        section_plane=None,
        background="#112233",
        shading_enabled=False,
        occlude_lines=False,
        mesh_lines=False,
        axis_indicator=False,
        axis_ruler=True,
        interaction_profile="legacy",
    )
    canvas.apply_view_state(replacement, redraw=False)

    exported = canvas.export_view_state()
    assert np.allclose(exported.camera_position.to_tuple(), (0.0, -8.0, 2.0))
    assert np.allclose(exported.camera_target.to_tuple(), (1.0, 2.0, 3.0))
    assert exported.section_plane is None
    assert exported.background == "#112233"
    assert not exported.shading_enabled
    assert not exported.occlude_lines
    assert not exported.mesh_lines
    assert not exported.axis_indicator
    assert exported.axis_ruler
    assert exported.interaction_profile == "legacy"


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"near": 0.0}, "near"),
        ({"near": 2.0, "far": 1.0}, "near"),
        ({"fov": 0.0}, "fov"),
        ({"camera_world_up": any3dview.Point3D(0, 0, 0)}, "world_up"),
        ({"interaction_profile": "unknown"}, "interaction profile"),
    ],
)
def test_software_rejects_invalid_portable_view_state(changes, message):
    canvas = contract_canvas()
    before = canvas.export_view_state()
    invalid = __import__("dataclasses").replace(before, **changes)

    with pytest.raises(ValueError, match=message):
        canvas.apply_view_state(invalid, redraw=False)

    after = canvas.export_view_state()
    assert after.camera_position.to_tuple() == before.camera_position.to_tuple()
    assert after.camera_target.to_tuple() == before.camera_target.to_tuple()
    assert after.camera_world_up.to_tuple() == before.camera_world_up.to_tuple()
    assert after.fov == before.fov
    assert after.near == before.near
    assert after.far == before.far
    assert after.section_plane == before.section_plane
    assert after.background == before.background
    assert after.interaction_profile == before.interaction_profile


def test_software_capabilities_describe_legacy_and_interaction_features():
    capabilities = contract_canvas().capabilities
    for name in (
        "legacy_primitives",
        "text_hud",
        "legends",
        "camera_controls",
        "work_plane_projection",
        "hover_selection",
        "region_selection",
        "lasso_selection",
        "animation",
        "line_occlusion",
        "stippled_transparency",
    ):
        assert getattr(capabilities, name), name
    assert capabilities.image_capture is (importlib.util.find_spec("PIL") is not None)


def test_software_capture_uses_only_the_inner_event_widget(monkeypatch):
    image_module = pytest.importorskip("PIL.Image")
    image_grab = pytest.importorskip("PIL.ImageGrab")
    captured = []

    def grab(*, bbox):
        captured.append(bbox)
        return image_module.new("RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]))

    monkeypatch.setattr(image_grab, "grab", grab)
    canvas = contract_canvas()

    image = canvas.capture_image()

    assert captured == [(11, 17, 211, 117)]
    assert image.mode == "RGBA"
    assert image.size == (200, 100)


def test_retained_application_selection_is_compiled_as_a_face_mask():
    mesh = any3dview.MeshArrays(
        np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            dtype=np.float32,
        ),
        np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.uint32),
        triangle_to_element=np.asarray([0, 1], dtype=np.uint32),
    )
    handle = any3dview.MeshHandle(mesh)
    handle.set_selected_elements((1,))
    canvas = contract_canvas()

    primitives = canvas._retained_mesh_primitives(
        {
            "handle": handle,
            "color": "#224466",
            "cull_backface": False,
        }
    )
    scene = canvas._compile(primitives)

    assert scene.face_application_selected.tolist() == [False, True]


def test_chunk_local_owner_tables_reach_all_software_selection_primitives():
    from anytk3d.picking import PickState

    base = any3dview.MeshArrays(
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.uint32),
    )
    chunk = any3dview.MeshArrays(
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.asarray([[0, 1, 2]], dtype=np.uint32),
        lines=np.asarray([[0, 1]], dtype=np.uint32),
        point_indices=np.asarray([2], dtype=np.uint32),
    )
    owners = any3dview.PackedOwnerTable.from_owners(
        triangles=(any3dview.PickBinding.one("chunk:face"),),
        lines=(any3dview.PickBinding.one("chunk:line"),),
        points=(any3dview.PickBinding.one("chunk:point"),),
    )
    handle = any3dview.MeshHandle(base)
    handle.add_chunk("local", chunk, owners=owners)
    canvas = contract_canvas()

    primitives = canvas._retained_mesh_primitives(
        {
            "handle": handle,
            "color": "#224466",
            "outline": "#111111",
            "cull_backface": False,
        }
    )
    scene = canvas._compile(primitives)

    assert scene.face_binding(0).owners[0].key == "chunk:face"
    assert scene.line_binding(0).owners[0].key == "chunk:line"
    assert scene.marker_binding(0).owners[0].key == "chunk:point"
    pick_state = PickState()
    pick_state.set_highlight(("chunk:face",))
    assert pick_state.highlighted_faces(scene) == frozenset((0,))
    pick_state.set_preselection("chunk:face")
    assert pick_state.preselected_faces(scene) == frozenset((0,))


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
