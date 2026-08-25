from __future__ import annotations

import numpy as np

import any3dview
import anytk3d
from anytk3d.canvas import Tkinter3DCanvas


def test_command_contracts_are_exact_shared_reexports():
    for name in (
        "SemanticRef",
        "VisibilityState",
        "ViewerCommand",
        "ViewerCommandController",
        "ViewerCommandDescriptor",
        "ViewerCommandPriority",
        "ViewerCommandResult",
        "ViewerObservation",
        "VIEWER_COMMANDS",
        "viewer_command_manifest",
    ):
        assert getattr(anytk3d, name) is getattr(any3dview, name)


def test_software_retained_batches_filter_hidden_semantic_owners():
    mesh = any3dview.MeshArrays(
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        np.asarray([[0, 1, 2]], np.uint32),
        lines=np.asarray([[0, 1]], np.uint32),
        point_indices=np.asarray([2], np.uint32),
    )
    owners = any3dview.PackedOwnerTable.from_owners(
        triangles=[any3dview.PickBinding.one("face-1", "face")],
        lines=[any3dview.PickBinding.one("edge-1", "edge")],
        points=[any3dview.PickBinding.one("vertex-1", "vertex")],
    )
    handle = any3dview.MeshHandle(mesh)
    canvas = object.__new__(Tkinter3DCanvas)
    canvas._visibility_state = any3dview.VisibilityState(
        hidden=(any3dview.SemanticRef("application", "face", "face-1"),),
        hidden_kinds=("edge", "vertex"),
    )
    primitives = canvas._retained_mesh_primitives(
        {"handle": handle, "owners": owners, "color": "#999999"}
    )
    assert not any(item["kind"] == "faces" for item in primitives)
    assert next(item for item in primitives if item["kind"] == "lines")["total"] == 0
    assert len(next(item for item in primitives if item["kind"] == "markers")["points"]) == 0
