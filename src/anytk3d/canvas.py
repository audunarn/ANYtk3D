"""
Fast dependency-free 3D drawing on a Tkinter Canvas.

Rendering model
---------------
Static world geometry is compiled once into flat numpy arrays.  Every frame
then runs as vectorised array work - projection, back-face classification,
screen-bounds and sub-pixel rejection, occluder tests and the painter-order
sort all happen without a Python loop over primitives.  Only the final
handover to Tk iterates, and it reuses a pool of canvas items and skips
``itemconfigure`` whenever a slot's appearance is unchanged.

Depth order
-----------
Surfaces are painted back to front by camera-space depth, so objects orbit
in front of and behind each other correctly.  Closed solids cull their back
faces, which both halves the work and removes the ordering ambiguity of
coincident front/back surfaces.  Layer numbers act only as a near-coplanar
tie-break.  Polygons that straddle the near plane are clipped instead of
dropped, so geometry no longer vanishes when the camera moves inside it.
Lines drawn without ``draw_overlay`` take part in the same sort and are
occluded by geometry in front of them.

Transparency
------------
A Tk canvas has no alpha channel, so transparency is screen-door stippling.
:mod:`anytk3d.stipple` generates 16 density steps with disjoint phase
variants, and front and back faces of the same surface are given different
phases, so a stack of transparent surfaces stays readable instead of the
nearest one masking everything behind it.

Lighting
--------
:mod:`anytk3d.shading` flat-shades every face with ambient + Lambert diffuse
+ a Blinn-Phong highlight.  With the default world-fixed light the shaded
colours are computed once per scene compile, so lighting costs nothing per
frame; ``follow_camera`` lights update as the camera orbits.

Shapes
------
Beyond the original cylinder/stiffener/plate helpers the canvas draws boxes,
spheres, cones, tubes, tori, pyramids, wedges, prisms, swept extrusions,
disks, arrows, planes, grids, structural beam profiles and arbitrary
index meshes; see :mod:`anytk3d.shapes`.

The cylinder axis is the global Z axis.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
import tkinter as tk
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from any3dview import (
    MeshArrays,
    MeshHandle,
    PackedOwnerTable,
    ViewerCapabilities,
    ViewerScheduler,
)
from any3dview.clipping import SectionPlane
from any3dview.contracts import ViewerBackend, ViewerState
from any3dview.semantic import SemanticRef, VisibilityState, semantic_refs

from . import shapes as shapes_module
from . import stipple as stipple_module
from .picking import (
    Pick,
    PickBinding,
    PickOwner,
    PickState,
    SelectionConfig,
    SelectionDepth,
    SelectionEvent,
    SelectionFilter,
    SelectionGesture,
    SelectionHit,
    SelectionOperation,
    SelectionTool,
    entity_tag_at,
    fallback_binding,
    modifiers_from_event,
    operation_from_modifiers,
)
from ._selection import ProjectedPrimitive, ProjectedSelectionIndex
from .core import (  # noqa: F401  (re-exported for backwards compatibility)
    _EPS,
    _NAMED_COLORS,
    _THICKNESS_COLOR_STOPS,
    Camera3D,
    DEFAULT_COLOR_STOPS,
    Point3D,
    _flatten_numeric_values,
    _hex_to_rgb,
    _interpolate_thickness_color,
    _rgb_to_hex,
    as_point,
    color_stop_generation,
    get_color_stops,
    parse_color,
    reset_color_stops,
    set_color_stops,
)
from .shading import (  # noqa: F401  (part of the public surface)
    Light,
    face_shade,
    shade_color,
    shade_level,
    shade_levels,
    shade_table,
    sun_direction,
)
from .shapes import Mesh

__all__ = [
    "DEFAULT_COLOR_STOPS",
    "Camera3D",
    "Light",
    "Mesh",
    "MeshArrays",
    "MeshHandle",
    "Pick",
    "PickBinding",
    "PickOwner",
    "Point3D",
    "SelectionConfig",
    "SelectionDepth",
    "SelectionEvent",
    "SelectionFilter",
    "SelectionGesture",
    "SelectionHit",
    "SelectionOperation",
    "SelectionTool",
    "SectionPlane",
    "Tkinter3DCanvas",
    "ViewerCapabilities",
    "ViewerBackend",
    "ViewerState",
    "create_stiffened_cylinder_demo",
    "get_color_stops",
    "reset_color_stops",
    "set_color_stops",
    "main",
    "populate_fe_gui_cylinder",
    "populate_fe_gui_plate",
    "populate_shape_gallery",
    "populate_stiffened_cylinder",
    "populate_stiffened_plate",
]


SOFTWARE_CAPABILITIES = ViewerCapabilities(
    dynamic_arrays=True,
    node_scalar_field=True,
    element_scalar_field=True,
    active_element_mask=True,
    incremental_chunks=True,
    through_selection=True,
    clipping_planes=True,
    transparency=True,
    software_fallback=True,
    legacy_primitives=True,
    text_hud=True,
    legends=True,
    camera_controls=True,
    work_plane_projection=True,
    hover_selection=True,
    region_selection=True,
    lasso_selection=True,
    animation=True,
    image_capture=importlib.util.find_spec("PIL") is not None,
    line_occlusion=True,
    stippled_transparency=True,
    semantic_selection=True,
    semantic_visibility=True,
    viewer_commands=True,
    command_history=True,
)


def _mix_color(base: str, overlay: str, amount: float) -> str:
    """Blend two renderer colours, falling back to the overlay when unknown."""

    first = parse_color(str(base))
    second = parse_color(str(overlay))
    if first is None or second is None:
        return str(overlay)
    weight = max(0.0, min(1.0, float(amount)))
    return _rgb_to_hex(
        *(left + weight * (right - left) for left, right in zip(first, second))
    )


# Canvas item tags used by the render pools.  Keeping them distinct lets the
# renderer restack whole pools with a single Tk call when a pool grows.
_TAG_POLYGON = "_tk3d_face"
_TAG_LINE = "_tk3d_line"
_TAG_TEXT = "_tk3d_text"
_TAG_MARKER = "_tk3d_marker"
_TAG_SELECTION = "_tk3d_selection"
_TAG_HUD = "_tk3d_hud"
_TAG_HUD_LEGEND = "_tk3d_hud_legend"
_TAG_HUD_AXIS = "_tk3d_hud_axis"

# Both HUD parts carry the shared tag so a single tag_raise keeps them above
# the face pool, and a specific tag so each can be rebuilt on its own.
_HUD_LEGEND_TAGS = (_TAG_HUD, _TAG_HUD_LEGEND)
_HUD_AXIS_TAGS = (_TAG_HUD, _TAG_HUD_AXIS)

# The renderer's own pool tags.  Picking never returns these, so a caller tag
# is always what comes back.
_RESERVED_TAGS = frozenset(
    {
        _TAG_POLYGON,
        _TAG_LINE,
        _TAG_TEXT,
        _TAG_MARKER,
        _TAG_SELECTION,
        _TAG_HUD,
        _TAG_HUD_LEGEND,
        _TAG_HUD_AXIS,
    }
)

# How far the cursor may travel between press and release and still count as a
# click rather than a pan.
_PICK_CLICK_SLOP = 3

# Screen coordinates are clamped before reaching Tk: enormous values from
# near-plane grazing geometry slow the rasteriser down for no visible gain.
_COORD_LIMIT = 32000.0

# Faces whose screen bounding box is smaller than this never show a pixel.
_MIN_SCREEN_EXTENT = 0.55


def _flatten_polygons(polygons: Iterable[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pack a sequence of faces into one vertex array plus per-face counts.

    Accepts ``Point3D`` objects, ``(x, y, z)`` sequences, or a single
    ``(faces, vertices, 3)`` array when every face has the same vertex count.
    """
    if isinstance(polygons, np.ndarray) and polygons.ndim == 3:
        face_total, vertex_total = polygons.shape[0], polygons.shape[1]
        return (
            np.ascontiguousarray(polygons.reshape(-1, 3), dtype=np.float32),
            np.full(face_total, vertex_total, dtype=np.int64),
        )

    flat: List[float] = []
    counts: List[int] = []
    for polygon in polygons:
        count = 0
        for vertex in polygon:
            if isinstance(vertex, Point3D):
                flat.append(vertex.x)
                flat.append(vertex.y)
                flat.append(vertex.z)
            else:
                x, y, z = vertex
                flat.append(float(x))
                flat.append(float(y))
                flat.append(float(z))
            count += 1
        if count < 3:
            # Drop degenerate faces here so the compiled arrays stay uniform.
            del flat[len(flat) - 3 * count:]
            continue
        counts.append(count)

    if not counts:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int64)
    return (
        np.asarray(flat, dtype=np.float32).reshape(-1, 3),
        np.asarray(counts, dtype=np.int64),
    )


def _coerce_pick_binding(value: Any, tags: str = "") -> Optional[PickBinding]:
    if value is None:
        return fallback_binding(tags)
    if isinstance(value, PickBinding):
        return value
    if isinstance(value, PickOwner):
        return PickBinding((value,))
    if isinstance(value, str):
        return PickBinding.one(value)
    raise TypeError("binding must be PickBinding, PickOwner, string, or None")


def _coerce_pick_bindings(
    value: Any, total: int, tags: str = ""
) -> List[Optional[PickBinding]]:
    if value is None or isinstance(value, (PickBinding, PickOwner, str)):
        binding = _coerce_pick_binding(value, tags)
        return [binding] * total
    bindings = [_coerce_pick_binding(binding, tags) for binding in value]
    if len(bindings) != total:
        raise ValueError(f"bindings has {len(bindings)} entries for {total} primitives")
    return bindings


class _CompiledScene:
    """Flat array form of a primitive list, ready for per-frame projection."""

    __slots__ = (
        "primitives",
        "face_vertices",
        "face_start",
        "face_count",
        "face_normal",
        "face_center",
        "face_layer",
        "face_phase",
        "face_cull",
        "face_occludable",
        "face_lit",
        "face_is_edge",
        "face_application_selected",
        "face_width",
        "base_front",
        "base_back",
        "table_front",
        "table_back",
        "fill_front",
        "fill_back",
        "outline",
        "stipple_front",
        "stipple_back",
        "opaque",
        "fast_no_outline",
        "tags",
        "face_bindings",
        "face_owner_table_slot",
        "face_owner_primitive",
        "face_owner_tables",
        "face_owner_resolvers",
        "any_tags",
        "line_vertices",
        "line_color",
        "line_width",
        "line_layer",
        "line_tags",
        "line_bindings",
        "line_owner_table_slot",
        "line_owner_primitive",
        "line_owner_tables",
        "line_owner_resolvers",
        "marker_points",
        "marker_colors",
        "marker_outlines",
        "marker_sizes",
        "marker_layers",
        "marker_tags",
        "marker_bindings",
        "marker_owner_table_slot",
        "marker_owner_primitive",
        "marker_owner_tables",
        "marker_owner_resolvers",
        "text_points",
        "text_layer",
        "text_content",
        "shade_key",
    )

    def __init__(self) -> None:
        self.primitives: List[Dict[str, Any]] = []
        self.face_vertices = np.empty((0, 3), dtype=np.float32)
        self.face_start = np.empty(0, dtype=np.int64)
        self.face_count = np.empty(0, dtype=np.int64)
        self.face_normal = np.empty((0, 3), dtype=np.float32)
        self.face_center = np.empty((0, 3), dtype=np.float32)
        self.face_layer = np.empty(0, dtype=np.float32)
        self.face_phase = np.empty(0, dtype=np.int8)
        self.face_cull = np.empty(0, dtype=bool)
        self.face_occludable = np.empty(0, dtype=bool)
        self.face_lit = np.empty(0, dtype=bool)
        self.face_is_edge = np.empty(0, dtype=bool)
        self.face_application_selected = np.empty(0, dtype=bool)
        self.face_width: List[int] = []
        self.base_front: List[str] = []
        self.base_back: List[str] = []
        self.table_front: List[Optional[Tuple[str, ...]]] = []
        self.table_back: List[Optional[Tuple[str, ...]]] = []
        self.fill_front: List[str] = []
        self.fill_back: List[str] = []
        self.outline: List[str] = []
        self.stipple_front: List[str] = []
        self.stipple_back: List[str] = []
        self.opaque = np.empty(0, dtype=bool)
        self.fast_no_outline = np.empty(0, dtype=bool)
        self.tags: List[str] = []
        self.face_bindings: List[Optional[PickBinding]] = []
        self.face_owner_table_slot = np.empty(0, dtype=np.int32)
        self.face_owner_primitive = np.empty(0, dtype=np.uint32)
        self.face_owner_tables: List[PackedOwnerTable] = []
        self.face_owner_resolvers: List[Optional[Callable[..., object]]] = []
        self.any_tags = False
        self.line_vertices = np.empty((0, 3), dtype=np.float32)
        self.line_color: List[str] = []
        self.line_width: List[int] = []
        self.line_tags: List[str] = []
        self.line_bindings: List[Optional[PickBinding]] = []
        self.line_owner_table_slot = np.empty(0, dtype=np.int32)
        self.line_owner_primitive = np.empty(0, dtype=np.uint32)
        self.line_owner_tables: List[PackedOwnerTable] = []
        self.line_owner_resolvers: List[Optional[Callable[..., object]]] = []
        self.line_layer = np.empty(0, dtype=np.float32)
        self.marker_points = np.empty((0, 3), dtype=np.float32)
        self.marker_colors: List[str] = []
        self.marker_outlines: List[str] = []
        self.marker_sizes: List[int] = []
        self.marker_layers = np.empty(0, dtype=np.float32)
        self.marker_tags: List[str] = []
        self.marker_bindings: List[Optional[PickBinding]] = []
        self.marker_owner_table_slot = np.empty(0, dtype=np.int32)
        self.marker_owner_primitive = np.empty(0, dtype=np.uint32)
        self.marker_owner_tables: List[PackedOwnerTable] = []
        self.marker_owner_resolvers: List[Optional[Callable[..., object]]] = []
        self.text_points = np.empty((0, 3), dtype=np.float32)
        self.text_layer = np.empty(0, dtype=np.float32)
        self.text_content: List[Tuple[str, str, Any, str]] = []
        self.shade_key: Any = None

    @property
    def face_total(self) -> int:
        return len(self.face_start)

    def face_binding(self, face: int) -> Optional[PickBinding]:
        """Resolve a packed binding only when selection or highlighting needs it."""

        if face < len(self.face_bindings) and self.face_bindings[face] is not None:
            return self.face_bindings[face]
        if face >= len(self.face_owner_table_slot):
            return None
        table_slot = int(self.face_owner_table_slot[face])
        if table_slot < 0:
            return None
        owners = self.face_owner_tables[table_slot].owners_for(
            "triangle",
            int(self.face_owner_primitive[face]),
            self.face_owner_resolvers[table_slot],
        )
        converted: List[PickOwner] = []
        for owner in owners:
            if isinstance(owner, PickOwner):
                converted.append(owner)
            else:
                converted.append(
                    PickOwner(
                        f"{owner.model_id}:{owner.kind}:{owner.id}",
                        f"geometry.{owner.kind}",
                        owner.priority,
                        owner,
                    )
                )
        return PickBinding(tuple(converted)) if converted else None

    @staticmethod
    def _packed_binding(
        primitive: int,
        primitive_kind: str,
        slots: np.ndarray,
        primitives: np.ndarray,
        tables: List[PackedOwnerTable],
        resolvers: List[Optional[Callable[..., object]]],
    ) -> Optional[PickBinding]:
        if primitive >= len(slots):
            return None
        table_slot = int(slots[primitive])
        if table_slot < 0:
            return None
        owners = tables[table_slot].owners_for(
            primitive_kind,
            int(primitives[primitive]),
            resolvers[table_slot],
        )
        converted = [
            owner
            if isinstance(owner, PickOwner)
            else PickOwner(
                f"{owner.model_id}:{owner.kind}:{owner.id}",
                f"geometry.{owner.kind}",
                owner.priority,
                owner,
            )
            for owner in owners
        ]
        return PickBinding(tuple(converted)) if converted else None

    def line_binding(self, line: int) -> Optional[PickBinding]:
        if line < len(self.line_bindings) and self.line_bindings[line] is not None:
            return self.line_bindings[line]
        return self._packed_binding(
            line,
            "line",
            self.line_owner_table_slot,
            self.line_owner_primitive,
            self.line_owner_tables,
            self.line_owner_resolvers,
        )

    def marker_binding(self, marker: int) -> Optional[PickBinding]:
        if marker < len(self.marker_bindings) and self.marker_bindings[marker] is not None:
            return self.marker_bindings[marker]
        return self._packed_binding(
            marker,
            "point",
            self.marker_owner_table_slot,
            self.marker_owner_primitive,
            self.marker_owner_tables,
            self.marker_owner_resolvers,
        )


class Tkinter3DCanvas(tk.Frame):
    """A fast pure-Tkinter 3D scene widget."""

    def __init__(
        self,
        master: tk.Misc,
        width: int = 800,
        height: int = 600,
        bg: str = "white",
        interactive_fps: int = 40,
        shading: bool = True,
        interaction_profile: str = "legacy",
        **canvas_kwargs: Any,
    ) -> None:
        super().__init__(master, background=bg)
        self._destroyed = False

        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.bg = bg
        self.camera = Camera3D()
        self.objects: List[Dict[str, Any]] = []
        self._explicit_opaque_cylinder_occluders: List[Dict[str, Any]] = []

        # Optional fixed 2D legend.  The 3D projection reserves space for it,
        # so the legend never covers the model.
        self._thickness_legend: Optional[Dict[str, Any]] = None
        self._show_axis_indicator = True
        self.show_mesh_lines = True
        self.show_axis_ruler = False

        self._light = Light()
        self._shading_enabled = bool(shading)
        self._occlude_lines = True
        self._section_plane: Optional[SectionPlane] = None
        self._backend_diagnostics: Tuple[str, ...] = ()

        canvas_kwargs.setdefault("highlightthickness", 0)
        canvas_kwargs.setdefault("borderwidth", 0)
        self.canvas = tk.Canvas(
            self,
            width=self.width,
            height=self.height,
            background=bg,
            **canvas_kwargs,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.interactive_fps = max(10, min(120, int(interactive_fps)))
        self._interactive_delay_ms = max(1, round(1000 / self.interactive_fps))

        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._is_dragging = False
        self._drag_mode = ""
        self._interactive_render = False
        # Interactive face budget.  It self-tunes from measured frame times so
        # a dense model degrades gracefully instead of dropping to a crawl.
        self._fast_polygon_target = 4000
        self._last_interactive_frame: Optional[float] = None

        self._redraw_after_id: Optional[str] = None
        self._finish_interaction_after_id: Optional[str] = None

        # World-space caches. Camera movement does not invalidate them.
        self._world_primitive_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._scene_cache: Dict[str, _CompiledScene] = {}
        self._quality_lod_flag: Optional[bool] = None
        self._color_stop_generation = color_stop_generation()

        # Animation Cache Fields
        self._animation_cache: List[Dict[str, Any]] = []
        self._is_capturing_animation = False
        self._is_playing_animation = False
        self._animation_frame_index = 0
        self._animation_after_id: Optional[str] = None
        self._animation_fps = 30
        self._animation_fast: Optional[bool] = None
        self._animation_frame_budget_ms = 33.0
        self._animation_occluders: Optional[List[Dict[str, Any]]] = None

        # HUD (legend and axis triad) is rebuilt only when it would differ,
        # which keeps animation playback from re-creating ~50 canvas items on
        # every frame.
        self._hud_signature: Optional[Tuple[Any, ...]] = None

        # Canvas item pools; slot order is display order, so assigning slots in
        # painter order gives the correct stacking without any restacking calls.
        self._polygon_pool: List[int] = []
        self._polygon_state: List[Optional[Tuple[Any, ...]]] = []
        self._line_pool: List[int] = []
        self._line_state: List[Optional[Tuple[Any, ...]]] = []
        self._text_pool: List[int] = []
        self._text_state: List[Optional[Tuple[Any, ...]]] = []
        self._marker_pool: List[int] = []
        self._marker_state: List[Optional[Tuple[Any, ...]]] = []

        # Picking is opt-in: without a callback the handlers below do nothing,
        # so existing applications keep exactly their previous behaviour.
        self._pick = PickState()

        # The commercial profile is deliberately opt-in for the 0.x series.
        # It reserves LMB for click/window selection, moves pan to MMB, and
        # leaves orbit on RMB.  The legacy profile keeps LMB pan + click-pick.
        self._interaction_profile = ""
        self._selection_config = SelectionConfig()
        self._semantic_selection: tuple[SemanticRef, ...] = ()
        self._visibility_state = VisibilityState()
        self._selection_callback: Optional[Callable[[SelectionEvent], None]] = None
        self._selection_hover_callback: Optional[
            Callable[[Optional[SelectionHit]], None]
        ] = None
        self._selection_index: Optional[ProjectedSelectionIndex] = None
        self._selection_index_key: Optional[Tuple[Any, ...]] = None
        self._selection_press: Optional[Tuple[int, int]] = None
        self._selection_modifiers = (False, False, False)
        self._selection_dragging = False
        self._selection_points: List[Tuple[int, int]] = []
        self._selection_overlay: Optional[int] = None
        self._selection_committed_on_press = False
        self._selection_press_hit_keys: frozenset[str] = frozenset()
        self._hover_after_id: Optional[str] = None
        self._hover_position: Optional[Tuple[int, int]] = None
        self._hover_key: Optional[str] = None
        self._cycle_signature: Optional[Tuple[Any, ...]] = None
        self._cycle_anchor: Optional[Tuple[int, int]] = None
        self._cycle_index = -1
        self._cycle_time = 0.0
        self._tracked_modifiers = {"shift": False, "ctrl": False, "alt": False}
        self._interaction_bindings: List[Tuple[str, str]] = []
        # Tk normally sends ButtonRelease back to the grabbed canvas, but on
        # Windows a real mouse release can occasionally land only on the
        # toplevel binding tag (notably after focus/menu transitions).  Keep a
        # second, instance-specific release binding so a visible selection
        # overlay can never be left waiting for a release that already
        # happened.  The normal canvas handler runs first and clears the press,
        # making this callback a no-op in the common path.
        self._selection_release_toplevel = self.winfo_toplevel()
        self._selection_release_binding = self._selection_release_toplevel.bind(
            "<ButtonRelease-1>", self._on_toplevel_selection_release, add="+"
        )

        self.bind("<Destroy>", self._on_destroy, add="+")
        self.canvas.bind("<Configure>", self._on_resize, add="+")
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel, add="+")
        self.canvas.bind("<Button-4>", self._on_mouse_wheel, add="+")
        self.canvas.bind("<Button-5>", self._on_mouse_wheel, add="+")
        self.canvas.bind("<KeyPress>", self._on_modifier_key, add="+")
        self.canvas.bind("<KeyRelease>", self._on_modifier_key, add="+")
        self.canvas.bind("<Escape>", self._cancel_selection_gesture, add="+")
        self.canvas.bind("<FocusOut>", self._on_selection_focus_out, add="+")
        self.set_interaction_profile(interaction_profile)

        self._update_scheduler = ViewerScheduler()
        self._update_poll_id: Optional[str] = None
        self._poll_updates()
        self.after_idle(self._request_redraw)

    def submit_update(
        self,
        callback: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Queue an immutable update payload for the owning Tk thread."""

        self._update_scheduler.submit(callback, *args, **kwargs)

    def _poll_updates(self) -> None:
        completed = self._update_scheduler.drain()
        if completed:
            self._request_redraw()
        self._update_poll_id = self.after(16, self._poll_updates)

    # ------------------------------------------------------------------
    # Lighting and render options
    # ------------------------------------------------------------------

    @property
    def light(self) -> Light:
        return self._light

    def set_light(
        self,
        direction: Optional[Point3D] = None,
        ambient: Optional[float] = None,
        diffuse: Optional[float] = None,
        specular: Optional[float] = None,
        shininess: Optional[float] = None,
        follow_camera: Optional[bool] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """
        Adjust the directional light.

        ``direction`` points from the surface toward the light.  With
        ``follow_camera`` it is interpreted in camera space (x right, y up,
        z toward the viewer) so the light orbits with the view.
        """
        if direction is not None:
            vector = as_point(direction).normalized()
            if vector.length() > 0.0:
                self._light.direction = vector
        if ambient is not None:
            self._light.ambient = float(ambient)
        if diffuse is not None:
            self._light.diffuse = float(diffuse)
        if specular is not None:
            self._light.specular = float(specular)
        if shininess is not None:
            self._light.shininess = max(1.0, float(shininess))
        if follow_camera is not None:
            self._light.follow_camera = bool(follow_camera)
        if enabled is not None:
            self._light.enabled = bool(enabled)
        self._request_redraw()

    def set_shading(self, enabled: bool = True) -> None:
        """Turn flat shading on or off; off draws every face in its raw colour."""
        enabled = bool(enabled)
        if enabled != self._shading_enabled:
            self._shading_enabled = enabled
            self._request_redraw()

    def set_occlude_lines(self, enabled: bool = True) -> None:
        """
        Choose whether 3D lines are hidden by geometry in front of them.

        Lines added with ``draw_overlay=True`` always stay on top; this
        setting controls the rest.
        """
        enabled = bool(enabled)
        if enabled != self._occlude_lines:
            self._occlude_lines = enabled
            self._invalidate_geometry_cache()
            self._request_redraw()

    @property
    def capabilities(self) -> ViewerCapabilities:
        """Read-only features implemented by the software backend."""

        return SOFTWARE_CAPABILITIES

    @property
    def backend_name(self) -> str:
        """Stable factory token for this renderer."""

        return "software"

    @property
    def backend_diagnostics(self) -> Tuple[str, ...]:
        """Diagnostics retained when ``create_viewer(auto)`` fell back here."""

        return self._backend_diagnostics

    @property
    def event_widget(self) -> tk.Canvas:
        """Widget applications should bind pointer and keyboard events to.

        ``canvas`` remains public for backwards compatibility.  New backend-
        neutral integrations use this property instead of knowing that the
        software renderer happens to draw into a nested :class:`tk.Canvas`.
        """

        return self.canvas

    @property
    def viewport_size(self) -> Tuple[int, int]:
        """Current drawable width and height in pixels."""

        return self._viewport_size()

    def export_view_state(self) -> ViewerState:
        """Return the renderer-neutral view policy for backend replacement."""

        camera = self.camera
        plane = self._section_plane
        if plane is not None:
            plane = SectionPlane(
                normal=plane.normal.to_tuple(),
                offset=plane.offset,
                enabled=plane.enabled,
            )
        return ViewerState(
            camera_position=Point3D(*camera.position.to_tuple()),
            camera_target=Point3D(*camera.target.to_tuple()),
            camera_world_up=Point3D(*camera.world_up.to_tuple()),
            fov=float(camera.fov),
            near=float(camera.near),
            far=float(camera.far),
            section_plane=plane,
            background=str(self.bg),
            shading_enabled=bool(self._shading_enabled),
            occlude_lines=bool(self._occlude_lines),
            mesh_lines=bool(self.show_mesh_lines),
            axis_indicator=bool(self._show_axis_indicator),
            axis_ruler=bool(self.show_axis_ruler),
            interaction_profile=str(self._interaction_profile),
            semantic_selection=getattr(self, "_semantic_selection", ()),
            visibility=getattr(self, "_visibility_state", VisibilityState()),
        )

    def apply_view_state(self, state: ViewerState, *, redraw: bool = True) -> None:
        """Apply portable camera and rendering policy in one invalidation.

        Scene objects deliberately do not belong to :class:`ViewerState`;
        applications repopulate a replacement backend before applying it.
        """

        if not isinstance(state, ViewerState):
            raise TypeError("state must be a ViewerState")
        if not (0.0 < float(state.near) < float(state.far)):
            raise ValueError("viewer state requires 0 < near < far")
        if not (0.0 < float(state.fov) < math.pi):
            raise ValueError("viewer state fov must be between 0 and pi")
        interaction_profile = str(state.interaction_profile).strip().lower()
        if interaction_profile not in {"legacy", "commercial"}:
            raise ValueError("interaction profile must be 'legacy' or 'commercial'")

        world_up = as_point(state.camera_world_up).normalized()
        if world_up.length() <= _EPS:
            raise ValueError("camera_world_up must be non-zero")
        camera = self.camera
        camera.world_up = world_up
        camera.fov = float(state.fov)
        camera.near = float(state.near)
        camera.far = float(state.far)
        camera.set_target(as_point(state.camera_target))
        camera.set_position(as_point(state.camera_position))

        self.bg = str(state.background)
        self.canvas.configure(background=self.bg)
        self.configure(background=self.bg)
        self._shading_enabled = bool(state.shading_enabled)
        self._occlude_lines = bool(state.occlude_lines)
        self.show_mesh_lines = bool(state.mesh_lines)
        self._show_axis_indicator = bool(state.axis_indicator)
        self.show_axis_ruler = bool(state.axis_ruler)
        self.set_interaction_profile(interaction_profile)
        self._semantic_selection = semantic_refs(state.semantic_selection)
        self._visibility_state = state.visibility
        plane = state.section_plane
        self._section_plane = (
            None
            if plane is None
            else SectionPlane(
                normal=plane.normal.to_tuple(),
                offset=plane.offset,
                enabled=plane.enabled,
            )
        )
        self._interactive_render = False
        self._invalidate_geometry_cache()
        if redraw:
            self._request_redraw(interactive=False)

    @property
    def section_plane(self) -> Optional[SectionPlane]:
        """The configured world-space section plane, or ``None`` when cleared."""

        return self._section_plane

    def set_section_plane(
        self,
        normal: Any = (1.0, 0.0, 0.0),
        offset: float = 0.0,
        *,
        enabled: bool = True,
    ) -> None:
        """Retain geometry where ``normal · point >= offset``.

        Clipping is a view policy: it leaves retained scene data untouched and
        intentionally does not create a cap over the cut surface.
        """

        plane = SectionPlane(normal=as_point(normal), offset=offset, enabled=enabled)
        if self._section_plane is not None and plane.key == self._section_plane.key:
            return
        self._section_plane = plane
        self._selection_index = None
        self._selection_index_key = None
        self._reset_selection_cycle()
        self._request_redraw()

    def clear_section_plane(self) -> None:
        """Disable section clipping while preserving the retained scene."""

        if self._section_plane is None:
            return
        self._section_plane = None
        self._selection_index = None
        self._selection_index_key = None
        self._reset_selection_cycle()
        self._request_redraw()

    def set_background(self, color: str) -> None:
        self.bg = color
        self.canvas.configure(background=color)
        self.configure(background=color)
        self._request_redraw()

    # ------------------------------------------------------------------
    # Thickness colour scale and fixed legend
    # ------------------------------------------------------------------

    def _viewport_size(self) -> Tuple[int, int]:
        """
        Current drawing size in pixels.

        A widget that has not been mapped yet reports 1x1, which would cull
        the whole scene.  Falling back to the requested size lets a canvas be
        rendered before its window is shown.
        """
        width = int(self.canvas.winfo_width())
        height = int(self.canvas.winfo_height())
        if width <= 1 or height <= 1:
            try:
                width = max(1, int(self.canvas.cget("width")))
                height = max(1, int(self.canvas.cget("height")))
            except (tk.TclError, ValueError):
                width = max(1, self.width)
                height = max(1, self.height)
        return max(1, width), max(1, height)

    def _plot_width(self) -> int:
        if self._thickness_legend is None:
            return max(1, self.width)
        legend_width = int(self._thickness_legend.get("width", 170))
        return max(120, self.width - legend_width)

    def set_thickness_legend(
        self,
        values: Sequence[float],
        unit: str = "mm",
        title: str = "Plate thickness",
        width: int = 170,
        value_range: Optional[Tuple[float, float]] = None,
        colors: Optional[Sequence[str]] = None,
    ) -> None:
        clean_values = sorted(
            {float(value) for value in values if math.isfinite(float(value))}
        )
        if not clean_values and value_range is None:
            self._thickness_legend = None
            self._invalidate_geometry_cache()
            self._request_redraw()
            return

        if value_range is None:
            minimum = clean_values[0]
            maximum = clean_values[-1]
        else:
            minimum = float(value_range[0])
            maximum = float(value_range[1])
            if maximum < minimum:
                minimum, maximum = maximum, minimum

        legend_colors: List[str] = []
        if colors is not None and len(colors) == len(values):
            by_value = {
                float(value): str(color)
                for value, color in zip(values, colors)
                if math.isfinite(float(value))
            }
            legend_colors = [by_value[value] for value in clean_values]

        self._thickness_legend = {
            "values": clean_values,
            "minimum": minimum,
            "maximum": maximum,
            "unit": str(unit),
            "title": str(title),
            "width": max(130, int(width)),
            "colors": legend_colors,
        }
        self._invalidate_geometry_cache()
        self._request_redraw()

    def clear_thickness_legend(self) -> None:
        self._thickness_legend = None
        self._invalidate_geometry_cache()
        self._request_redraw()

    def set_axis_indicator(self, visible: bool = True) -> None:
        visible = bool(visible)
        if visible != self._show_axis_indicator:
            self._show_axis_indicator = visible
            self._request_redraw()

    def set_mesh_lines(self, visible: bool = True) -> None:
        new_val = bool(visible)
        if self.show_mesh_lines != new_val:
            self.show_mesh_lines = new_val
            self._request_redraw()

    def set_axis_ruler(self, visible: bool = True) -> None:
        new_val = bool(visible)
        if self.show_axis_ruler != new_val:
            self.show_axis_ruler = new_val
            self._invalidate_geometry_cache()
            self._request_redraw()

    def thickness_color(
        self,
        thickness: float,
        value_range: Optional[Tuple[float, float]] = None,
    ) -> str:
        if value_range is not None:
            minimum, maximum = value_range
        elif self._thickness_legend is not None:
            minimum = float(self._thickness_legend["minimum"])
            maximum = float(self._thickness_legend["maximum"])
        else:
            minimum = maximum = float(thickness)
        return _interpolate_thickness_color(float(thickness), minimum, maximum)

    @staticmethod
    def _legend_color(legend: Dict[str, Any], value: float) -> str:
        levels = list(legend.get("values", ()))
        colors = list(legend.get("colors", ()))
        if len(levels) < 2 or len(colors) != len(levels):
            return _interpolate_thickness_color(
                value, float(legend["minimum"]), float(legend["maximum"])
            )
        if value <= levels[0]:
            return colors[0]
        for index in range(1, len(levels)):
            if value <= levels[index]:
                low, high = float(levels[index - 1]), float(levels[index])
                fraction = 0.0 if high <= low else (value - low) / (high - low)
                first = np.asarray(_hex_to_rgb(colors[index - 1]), dtype=float)
                second = np.asarray(_hex_to_rgb(colors[index]), dtype=float)
                return _rgb_to_hex(*(first + fraction * (second - first)))
        return colors[-1]

    @staticmethod
    def _format_legend_value(value: float) -> str:
        magnitude = abs(float(value))
        if magnitude >= 100.0:
            return f"{value:.0f}"
        if magnitude >= 10.0:
            return f"{value:.1f}".rstrip("0").rstrip(".")
        if magnitude >= 1.0:
            return f"{value:.2f}".rstrip("0").rstrip(".")
        if magnitude >= 1.0e-2:
            return f"{value:.4f}".rstrip("0").rstrip(".")
        if magnitude >= 1.0e-5 or magnitude == 0.0:
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return f"{value:.3e}"

    @staticmethod
    def _legend_text_lines(text: str, max_chars: int, max_lines: int = 3) -> List[str]:
        words = str(text).split()
        if not words:
            return [""]
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = word if not current else current + " " + word
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = word
            if len(lines) >= max_lines:
                break
        if current and len(lines) < max_lines:
            lines.append(current)
        if len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
            lines[-1] = lines[-1].rstrip(".") + "..."
        return lines

    def _draw_thickness_legend(self) -> None:
        legend = self._thickness_legend
        if legend is None:
            return

        panel_width = int(legend.get("width", 170))
        left = max(0, self.width - panel_width)
        right = self.width
        top = 0
        bottom = self.height

        self.canvas.create_rectangle(
            left,
            top,
            right,
            bottom,
            fill=self.bg,
            outline="#d0d0d0",
            width=1,
            tags=_HUD_LEGEND_TAGS,
        )

        padding = 14
        title = str(legend.get("title", "Plate thickness"))
        unit = str(legend.get("unit", ""))
        title_text = f"{title} [{unit}]" if unit else title
        max_title_chars = max(12, int((panel_width - 2 * padding) / 7))
        title_lines = self._legend_text_lines(title_text, max_title_chars)
        title_y = 18
        for line in title_lines:
            self.canvas.create_text(
                left + padding,
                title_y,
                text=line,
                anchor="nw",
                font=("TkDefaultFont", 10, "bold"),
                fill="#202020",
                tags=_HUD_LEGEND_TAGS,
            )
            title_y += 15

        values = list(legend.get("values", []))
        minimum = float(legend["minimum"])
        maximum = float(legend["maximum"])
        bar_top = max(54, title_y + 8)
        available_height = max(80, self.height - (bar_top + 16))

        # A short set of distinct thicknesses is clearer as labelled swatches.
        if 1 <= len(values) <= 10:
            values = sorted(values, reverse=True)
            row_height = min(34, max(23, available_height // len(values)))
            swatch_width = 34
            y_coord = bar_top
            for value in values:
                color = self._legend_color(legend, value)
                self.canvas.create_rectangle(
                    left + padding,
                    y_coord,
                    left + padding + swatch_width,
                    y_coord + 17,
                    fill=color,
                    outline="#505050",
                    width=1,
                    tags=_HUD_LEGEND_TAGS,
                )
                self.canvas.create_text(
                    left + padding + swatch_width + 10,
                    y_coord + 8,
                    text=self._format_legend_value(value),
                    anchor="w",
                    fill="#202020",
                    tags=_HUD_LEGEND_TAGS,
                )
                y_coord += row_height
            return

        # Continuous or highly populated scales are rendered as a gradient.
        bar_bottom = min(self.height - 28, bar_top + available_height)
        bar_left = left + padding
        bar_right = bar_left + 30
        steps = max(30, min(180, bar_bottom - bar_top))
        for index in range(steps):
            fraction_0 = index / steps
            fraction_1 = (index + 1) / steps
            value = maximum - fraction_0 * (maximum - minimum)
            color = self._legend_color(legend, value)
            y_0 = bar_top + fraction_0 * (bar_bottom - bar_top)
            y_1 = bar_top + fraction_1 * (bar_bottom - bar_top) + 1
            self.canvas.create_rectangle(
                bar_left,
                y_0,
                bar_right,
                y_1,
                fill=color,
                outline=color,
                tags=_HUD_LEGEND_TAGS,
            )
        self.canvas.create_rectangle(
            bar_left,
            bar_top,
            bar_right,
            bar_bottom,
            fill="",
            outline="#505050",
            width=1,
            tags=_HUD_LEGEND_TAGS,
        )

        tick_count = 6
        for index in range(tick_count):
            fraction = index / (tick_count - 1)
            value = maximum - fraction * (maximum - minimum)
            y_coord = bar_top + fraction * (bar_bottom - bar_top)
            self.canvas.create_line(
                bar_right, y_coord, bar_right + 5, y_coord, fill="#505050", tags=_HUD_LEGEND_TAGS
            )
            self.canvas.create_text(
                bar_right + 10,
                y_coord,
                text=self._format_legend_value(value),
                anchor="w",
                fill="#202020",
                tags=_HUD_LEGEND_TAGS,
            )

    def _draw_hud(self) -> None:
        """
        Redraw the fixed overlay, but only the parts that would differ.

        The legend depends on its own contents and the widget size; the axis
        triad additionally follows the camera.  Tracking them separately means
        orbiting rebuilds seven small items instead of the whole legend, and
        animation playback - where neither changes - rebuilds nothing at all.
        """
        legend = self._thickness_legend
        legend_signature: Any = None
        if legend is not None:
            legend_signature = (
                legend.get("title"),
                legend.get("unit"),
                legend.get("minimum"),
                legend.get("maximum"),
                tuple(legend.get("values", ())),
                tuple(legend.get("colors", ())),
                legend.get("width"),
                self.width,
                self.height,
                self.bg,
            )
        axis_signature: Any = None
        if self._show_axis_indicator:
            axis_signature = (
                self.width,
                self.height,
                round(self.camera.azimuth, 4),
                round(self.camera.elevation, 4),
                self._plot_width(),
            )

        previous_legend, previous_axis = self._hud_signature or (None, None)
        if legend_signature != previous_legend:
            self.canvas.delete(_TAG_HUD_LEGEND)
            if legend is not None:
                self._draw_thickness_legend()
        if axis_signature != previous_axis:
            self.canvas.delete(_TAG_HUD_AXIS)
            if self._show_axis_indicator:
                self._draw_axis_indicator()
        self._hud_signature = (legend_signature, axis_signature)

    def _draw_axis_indicator(self) -> None:
        if not self._show_axis_indicator:
            return

        plot_width = self._plot_width()
        if plot_width < 95 or self.height < 95:
            return

        origin_x = min(max(58.0, plot_width * 0.065), max(58.0, plot_width - 78.0))
        origin_y = max(58.0, self.height - 64.0)
        axis_length = min(58.0, max(34.0, min(plot_width, self.height) * 0.085))
        right, camera_up, forward = self.camera.basis()

        axes = [
            ("X", Point3D(1.0, 0.0, 0.0), "#9b111e"),
            ("Y", Point3D(0.0, 1.0, 0.0), "#159447"),
            ("Z", Point3D(0.0, 0.0, 1.0), "#0d47a1"),
        ]

        def screen_delta(vector: Point3D) -> Tuple[float, float]:
            return (
                vector.dot(right) * axis_length,
                -vector.dot(camera_up) * axis_length,
            )

        # Draw the visually deeper axis first so overlapping arrows read like a
        # small 3D triad instead of a flat logo.
        axes = sorted(axes, key=lambda item: item[1].dot(forward), reverse=True)

        self.canvas.create_oval(
            origin_x - 2,
            origin_y - 2,
            origin_x + 2,
            origin_y + 2,
            fill="#202020",
            outline="",
            tags=_HUD_AXIS_TAGS,
        )
        for label, vector, color in axes:
            dx, dy = screen_delta(vector)
            end_x = origin_x + dx
            end_y = origin_y + dy
            if abs(dx) + abs(dy) < 3.0:
                continue
            self.canvas.create_line(
                origin_x,
                origin_y,
                end_x,
                end_y,
                fill=color,
                width=2,
                arrow=tk.LAST,
                arrowshape=(9, 11, 4),
                tags=_HUD_AXIS_TAGS,
            )
            label_offset = 11.0
            length = max(math.hypot(dx, dy), 1.0)
            self.canvas.create_text(
                end_x + label_offset * dx / length,
                end_y + label_offset * dy / length,
                text=label,
                fill=color,
                font=("TkDefaultFont", 10, "bold"),
                tags=_HUD_AXIS_TAGS,
            )

    # ------------------------------------------------------------------
    # Event handling and redraw scheduling
    # ------------------------------------------------------------------

    def _on_destroy(self, event: tk.Event) -> None:
        """
        Drop every pending callback when the widget goes away.

        A redraw or animation step scheduled with ``after`` outlives the
        widget otherwise, and Tk reports the dangling handler as an "invalid
        command name" on stderr the moment it fires.
        """
        if event.widget is not self:
            return
        binding = getattr(self, "_selection_release_binding", None)
        toplevel = getattr(self, "_selection_release_toplevel", None)
        if binding and toplevel is not None:
            try:
                toplevel.unbind("<ButtonRelease-1>", binding)
            except tk.TclError:
                pass
            self._selection_release_binding = None
        for attribute in (
            "_redraw_after_id",
            "_finish_interaction_after_id",
            "_animation_after_id",
            "_hover_after_id",
        ):
            handle = getattr(self, attribute, None)
            if handle is None:
                continue
            setattr(self, attribute, None)
            try:
                self.after_cancel(handle)
            except (tk.TclError, ValueError):
                pass
        self._is_playing_animation = False

    def _on_resize(self, event: tk.Event) -> None:
        new_width = max(1, int(event.width))
        new_height = max(1, int(event.height))
        if new_width == self.width and new_height == self.height:
            return
        self.width = new_width
        self.height = new_height
        self._selection_index = None
        self._selection_index_key = None
        self._reset_selection_cycle()
        self._request_redraw()

    # ------------------------------------------------------------------
    # picking, hover and highlight
    # ------------------------------------------------------------------
    @property
    def interaction_profile(self) -> str:
        return self._interaction_profile

    def set_interaction_profile(self, profile: str) -> None:
        """Choose ``legacy`` or commercial CAD-style mouse interaction.

        ``legacy`` preserves the original LMB-pan/RMB-orbit bindings.
        ``commercial`` reserves LMB for selection, moves pan to MMB and keeps
        orbit on RMB.  Changing profile does not enable callbacks by itself.
        """

        profile = str(profile).strip().lower()
        if profile not in {"legacy", "commercial"}:
            raise ValueError("interaction profile must be 'legacy' or 'commercial'")
        if profile == self._interaction_profile:
            return
        self._cancel_selection_gesture()
        for sequence, identifier in self._interaction_bindings:
            self.canvas.unbind(sequence, identifier)
        self._interaction_bindings.clear()

        def bind(sequence: str, callback: Any) -> None:
            identifier = self.canvas.bind(sequence, callback, add="+")
            if identifier:
                self._interaction_bindings.append((sequence, identifier))

        self._interaction_profile = profile
        if profile == "legacy":
            bind(
                "<ButtonPress-1>",
                lambda event: self._on_mouse_down(event, "pan"),
            )
            bind("<B1-Motion>", self._on_mouse_drag)
            bind("<ButtonRelease-1>", self._on_mouse_up)
            bind("<ButtonPress-1>", self._on_pick_press)
            bind("<ButtonRelease-1>", self._on_pick_release)
            bind("<Motion>", self._on_pick_motion)
        else:
            bind("<ButtonPress-1>", self._on_selection_press)
            bind("<B1-Motion>", self._on_selection_drag)
            bind("<ButtonRelease-1>", self._on_selection_release)
            bind("<Motion>", self._on_selection_hover_motion)
            bind(
                "<ButtonPress-2>",
                lambda event: self._on_mouse_down(event, "pan"),
            )
            bind("<B2-Motion>", self._on_mouse_drag)
            bind("<ButtonRelease-2>", self._on_mouse_up)
        bind(
            "<ButtonPress-3>",
            lambda event: self._on_mouse_down(event, "rotate"),
        )
        bind("<B3-Motion>", self._on_mouse_drag)
        bind("<ButtonRelease-3>", self._on_mouse_up)

    def configure_selection(
        self,
        callback: Optional[Callable[[SelectionEvent], None]],
        *,
        hover_callback: Optional[Callable[[Optional[SelectionHit]], None]] = None,
        config: Optional[SelectionConfig] = None,
    ) -> None:
        """Configure rich click/drag selection callbacks.

        Region gestures are delivered only by the commercial interaction
        profile.  Programmatic point and rectangle queries work in either
        profile.
        """

        self._selection_callback = callback
        self._selection_hover_callback = hover_callback
        if config is not None:
            if not isinstance(config, SelectionConfig):
                raise TypeError("config must be a SelectionConfig")
            self._selection_config = config
        self._reset_selection_cycle()

    def set_selection_callback(
        self,
        callback: Optional[Callable[[SelectionEvent], None]],
    ) -> None:
        """Set only the semantic selection callback.

        This narrow alias matches the retained GPU viewer while
        :meth:`configure_selection` remains the full policy API.
        """

        self._selection_callback = callback

    @property
    def selection_config(self) -> SelectionConfig:
        return self._selection_config

    @property
    def semantic_selection(self) -> tuple[SemanticRef, ...]:
        return getattr(self, "_semantic_selection", ())

    @property
    def visibility_state(self) -> VisibilityState:
        return getattr(self, "_visibility_state", VisibilityState())

    def set_semantic_selection(self, values: Sequence[SemanticRef]) -> None:
        self._semantic_selection = semantic_refs(values)
        self._selection_index = None
        self._selection_index_key = None
        self._request_redraw()

    def set_visibility_state(self, state: VisibilityState) -> None:
        if not isinstance(state, VisibilityState):
            raise TypeError("state must be VisibilityState")
        self._visibility_state = state
        self._selection_index = None
        self._selection_index_key = None
        self._invalidate_geometry_cache()
        self._request_redraw()

    def update_selection_config(self, **changes: Any) -> SelectionConfig:
        """Replace selected fields of the immutable selection configuration."""

        self._selection_config = replace(self._selection_config, **changes)
        self._reset_selection_cycle()
        self.set_preselection(None)
        return self._selection_config

    def query_point(
        self,
        x: int,
        y: int,
        *,
        selection_filter: Optional[SelectionFilter] = None,
        radius: Optional[int] = None,
    ) -> Tuple[SelectionHit, ...]:
        """Return the front-to-back semantic candidate stack at a pixel."""

        config = self._selection_config
        return self._get_selection_index().point_hits(
            int(x),
            int(y),
            selection_filter or config.filter,
            radius=config.click_radius_px if radius is None else max(0, int(radius)),
        )

    def query_rectangle(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        *,
        crossing: Optional[bool] = None,
        selection_filter: Optional[SelectionFilter] = None,
        depth: Optional[SelectionDepth] = None,
    ) -> Tuple[SelectionHit, ...]:
        """Query a directional window/crossing rectangle."""

        config = self._selection_config
        if crossing is None:
            crossing = bool(config.directional and end[0] < start[0])
        return self._get_selection_index().rectangle_hits(
            (start[0], start[1], end[0], end[1]),
            selection_filter or config.filter,
            crossing=bool(crossing),
            depth=SelectionDepth(config.depth if depth is None else depth),
        )

    def query_lasso(
        self,
        points: Sequence[Tuple[int, int]],
        *,
        selection_filter: Optional[SelectionFilter] = None,
        depth: Optional[SelectionDepth] = None,
    ) -> Tuple[SelectionHit, ...]:
        """Query a freehand crossing polygon."""

        config = self._selection_config
        return self._get_selection_index().polygon_hits(
            points,
            selection_filter or config.filter,
            depth=SelectionDepth(config.depth if depth is None else depth),
        )

    def project_point(self, point: Any) -> Optional[Tuple[float, float, float]]:
        """Project one world point to ``(screen_x, screen_y, depth)``.

        ``None`` means the point is outside the camera's near/far depth range.
        Depth is positive camera-space distance and is therefore directly
        comparable across projected points.
        """

        return self.project_points((point,))[0]

    def project_points(
        self,
        points: Iterable[Any],
    ) -> Tuple[Optional[Tuple[float, float, float]], ...]:
        """Vector-project world points using the same viewport as rendering."""

        world_points = tuple(as_point(point) for point in points)
        if not world_points:
            return ()

        self.width, self.height = self._viewport_size()
        plot_width = max(1, self._plot_width())
        height = max(1, self.height)
        camera = self.camera
        right, camera_up, forward = camera.basis()
        position = camera.position
        values = np.asarray(
            [point.to_tuple() for point in world_points], dtype=np.float64
        )
        origin = np.asarray(position.to_tuple(), dtype=np.float64)
        basis = np.asarray(
            [
                [right.x, camera_up.x, forward.x],
                [right.y, camera_up.y, forward.y],
                [right.z, camera_up.z, forward.z],
            ],
            dtype=np.float64,
        )
        camera_space = (values - origin) @ basis
        depth = camera_space[:, 2]
        valid = (
            np.isfinite(camera_space).all(axis=1)
            & (depth > max(1.0e-9, float(camera.near)))
            & (depth < float(camera.far))
        )
        safe_depth = np.where(valid, depth, 1.0)
        y_scale = 1.0 / math.tan(float(camera.fov) / 2.0)
        x_scale = y_scale * float(height) / float(plot_width)
        screen_x = (camera_space[:, 0] * x_scale / safe_depth + 1.0) * (
            0.5 * plot_width
        )
        screen_y = (1.0 - camera_space[:, 1] * y_scale / safe_depth) * (
            0.5 * height
        )
        return tuple(
            None
            if not is_valid
            else (float(x), float(y), float(z))
            for x, y, z, is_valid in zip(screen_x, screen_y, depth, valid)
        )

    def screen_ray(self, x: float, y: float) -> Tuple[Point3D, Point3D]:
        """World-space camera ray through a canvas pixel."""

        self.width, self.height = self._viewport_size()
        return self.camera.screen_ray(x, y, self._plot_width(), self.height)

    def unproject_to_plane(
        self,
        x: float,
        y: float,
        plane_point: Any,
        plane_normal: Any,
    ) -> Optional[Point3D]:
        """Map a canvas pixel onto a world plane, or return ``None``."""

        self.width, self.height = self._viewport_size()
        return self.camera.unproject_to_plane(
            x,
            y,
            self._plot_width(),
            self.height,
            as_point(plane_point),
            as_point(plane_normal),
        )

    def set_preselection(self, key: Optional[str]) -> None:
        """Show one transient hover highlight, separate from selection."""

        if self._pick.set_preselection(key):
            self._request_redraw()

    @property
    def preselected_key(self) -> Optional[str]:
        return self._pick.preselection_key

    def set_pick_callback(
        self,
        callback: Optional[Any],
        *,
        prefix: str = "",
        radius: Optional[int] = None,
    ) -> None:
        """Call ``callback(pick)`` when the user clicks without dragging.

        ``prefix`` restricts hits to tags the caller owns.  A click on empty
        space still fires, with an empty ``pick.tag``, so the application can
        clear its selection.
        """

        self._pick.pick_callback = callback
        self._pick.prefix = str(prefix)
        if radius is not None:
            self._pick.radius = max(0, int(radius))

    def set_hover_callback(self, callback: Optional[Any]) -> None:
        """Call ``callback(pick_or_None)`` when the tag under the cursor changes."""

        self._pick.hover_callback = callback
        self._pick.hover_tag = None

    def pick_at(self, x: int, y: int) -> Optional[str]:
        """The topmost caller tag at a canvas position, or None."""

        tag, _item = entity_tag_at(
            self.canvas,
            int(x),
            int(y),
            prefix=self._pick.prefix,
            reserved=_RESERVED_TAGS,
            radius=self._pick.radius,
        )
        return tag

    def set_highlight(
        self,
        tags: Iterable[str],
        fill: Optional[str] = None,
        outline: Optional[str] = None,
    ) -> None:
        """Tint every face carrying one of ``tags``.

        The tint is applied while rendering rather than by reconfiguring Tk
        items directly, so it survives the next redraw.
        """

        if self._pick.set_highlight(tags, fill=fill, outline=outline):
            self._request_redraw()

    def clear_highlight(self) -> None:
        self.set_highlight(())

    def highlighted_tags(self) -> frozenset:
        return self._pick.highlight_tags

    def _on_pick_press(self, event: tk.Event) -> None:
        if self._pick.pick_callback is None:
            return
        self._pick.press = (int(event.x), int(event.y))

    def _on_pick_release(self, event: tk.Event) -> None:
        state = self._pick
        callback = state.pick_callback
        press = state.press
        state.press = None
        if callback is None or press is None:
            return

        x, y = int(event.x), int(event.y)
        if (
            abs(x - press[0]) > _PICK_CLICK_SLOP
            or abs(y - press[1]) > _PICK_CLICK_SLOP
        ):
            # The user was panning, not selecting.
            return

        tag, item = entity_tag_at(
            self.canvas,
            x,
            y,
            prefix=state.prefix,
            reserved=_RESERVED_TAGS,
            radius=state.radius,
        )
        shift, ctrl, alt = modifiers_from_event(event)
        callback(
            Pick(
                tag=tag or "",
                item=-1 if item is None else int(item),
                x=x,
                y=y,
                shift=shift,
                ctrl=ctrl,
                alt=alt,
            )
        )

    def _on_pick_motion(self, event: tk.Event) -> None:
        state = self._pick
        callback = state.hover_callback
        if callback is None or self._is_dragging:
            return

        x, y = int(event.x), int(event.y)
        tag, item = entity_tag_at(
            self.canvas,
            x,
            y,
            prefix=state.prefix,
            reserved=_RESERVED_TAGS,
            radius=state.radius,
        )
        if tag == state.hover_tag:
            return
        state.hover_tag = tag
        callback(
            None
            if tag is None
            else Pick(tag=tag, item=-1 if item is None else int(item), x=x, y=y)
        )

    def _on_modifier_key(self, event: tk.Event) -> None:
        keysym = str(getattr(event, "keysym", "")).lower()
        pressed = str(getattr(event, "type", "")) in {"2", "KeyPress"}
        if keysym.startswith("shift"):
            self._tracked_modifiers["shift"] = pressed
        elif keysym.startswith("control"):
            self._tracked_modifiers["ctrl"] = pressed
        elif keysym.startswith("alt") or keysym.startswith("option") or keysym.startswith("meta"):
            self._tracked_modifiers["alt"] = pressed

    def _event_modifiers(self, event: tk.Event) -> Tuple[bool, bool, bool]:
        shift, ctrl, alt = modifiers_from_event(event)
        state = int(getattr(event, "state", 0) or 0)
        if sys.platform == "win32":
            # The low X11 Mod1 bit (0x0008) is present on ordinary native
            # Windows button events in Tk 8.6; it does *not* mean Alt there.
            # Windows reports Alt through its high 0x20000 bit.  Confirm that
            # bit against the physical key because native menus can leave the
            # event bit stale.  Misreading either case makes every plain click
            # REMOVE, which looks exactly like selection is dead while hover
            # continues to work.
            alt = bool(state & 0x20000) and self._windows_alt_is_down()
        # Windows and X11 put the current modifiers directly in mouse-event
        # state (Windows Alt is 0x20000).  Merging key events here can leave
        # Alt permanently active when the native menu consumes KeyRelease;
        # every later click/box then becomes REMOVE and an empty selection
        # appears completely broken while hover still works.  macOS Option is
        # the platform that needs the separately tracked fallback.
        tracked = self._tracked_modifiers if sys.platform == "darwin" else {}
        return (
            shift or bool(tracked.get("shift", False)),
            ctrl or bool(tracked.get("ctrl", False)),
            alt or bool(tracked.get("alt", False)),
        )

    @staticmethod
    def _windows_alt_is_down() -> bool:
        """Return the physical Windows Alt state, conservatively on failure."""

        try:
            import ctypes

            return bool(ctypes.windll.user32.GetKeyState(0x12) & 0x8000)
        except (AttributeError, OSError):  # pragma: no cover - unusual runtime
            return True

    def _on_selection_press(self, event: tk.Event) -> None:
        # Recover cleanly if the platform dropped the preceding release.
        if self._selection_press is not None:
            self._finish_selection_gesture()
        self.canvas.focus_set()
        try:
            self.canvas.grab_set()
        except tk.TclError:
            # A containing dialog may already own a local grab.  Selection is
            # still usable inside the viewport; only release-outside capture
            # is unavailable in that unusual case.
            pass
        self._selection_press = (int(event.x), int(event.y))
        self._selection_modifiers = self._event_modifiers(event)
        self._selection_dragging = False
        self._selection_points = [self._selection_press]
        self.set_preselection(None)
        if self._selection_config.click_on_press:
            operation = operation_from_modifiers(*self._selection_modifiers)
            active = self._emit_click_selection(self._selection_press, operation)
            self._selection_committed_on_press = True
            self._selection_press_hit_keys = frozenset(hit.key for hit in active)

    def _on_selection_drag(self, event: tk.Event) -> None:
        press = self._selection_press
        if press is None:
            return
        point = (int(event.x), int(event.y))
        config = self._selection_config
        if config.tool == SelectionTool.SINGLE:
            return
        if not self._selection_dragging:
            if math.hypot(point[0] - press[0], point[1] - press[1]) < config.drag_threshold_px:
                return
            self._selection_dragging = True
        if config.tool == SelectionTool.LASSO:
            previous = self._selection_points[-1]
            if math.hypot(point[0] - previous[0], point[1] - previous[1]) >= 3.0:
                self._selection_points.append(point)
            coords = [coordinate for pair in self._selection_points for coordinate in pair]
            if self._selection_overlay is None:
                self._selection_overlay = self.canvas.create_line(
                    *coords,
                    fill="#2563eb",
                    width=2,
                    dash=(5, 3),
                    tags=_TAG_SELECTION,
                )
            else:
                self.canvas.coords(self._selection_overlay, *coords)
            return

        crossing = bool(config.directional and point[0] < press[0])
        outline = "#16a34a" if crossing else "#2563eb"
        fill = "#dcfce7" if crossing else "#dbeafe"
        dash = (5, 3) if crossing else ()
        if self._selection_overlay is None:
            self._selection_overlay = self.canvas.create_rectangle(
                press[0],
                press[1],
                point[0],
                point[1],
                outline=outline,
                fill=fill,
                stipple="gray25",
                width=2,
                dash=dash,
                tags=_TAG_SELECTION,
            )
        else:
            self.canvas.coords(
                self._selection_overlay, press[0], press[1], point[0], point[1]
            )
            self.canvas.itemconfigure(
                self._selection_overlay, outline=outline, fill=fill, dash=dash
            )

    def _on_selection_release(self, event: tk.Event) -> None:
        press = self._selection_press
        if press is None:
            return
        end = (int(event.x), int(event.y))
        operation = operation_from_modifiers(*self._selection_modifiers)
        if self._selection_dragging:
            config = self._selection_config
            if config.tool == SelectionTool.LASSO:
                points = tuple(self._selection_points + [end])
                hits = self.query_lasso(points)
                if (
                    operation == SelectionOperation.TOGGLE
                    and self._selection_committed_on_press
                    and self._selection_press_hit_keys
                ):
                    hits = tuple(
                        hit
                        for hit in hits
                        if hit.key not in self._selection_press_hit_keys
                    )
                event_value = SelectionEvent(
                    SelectionGesture.LASSO,
                    operation,
                    hits=hits,
                    start=press,
                    end=end,
                    points=points,
                )
            else:
                crossing = bool(config.directional and end[0] < press[0])
                hits = self.query_rectangle(press, end, crossing=crossing)
                if (
                    operation == SelectionOperation.TOGGLE
                    and self._selection_committed_on_press
                    and self._selection_press_hit_keys
                ):
                    # The start hit was already toggled on button press.  A
                    # region beginning on the same object must toggle it only
                    # once while still toggling every other enclosed owner.
                    hits = tuple(
                        hit
                        for hit in hits
                        if hit.key not in self._selection_press_hit_keys
                    )
                event_value = SelectionEvent(
                    SelectionGesture.CROSSING if crossing else SelectionGesture.WINDOW,
                    operation,
                    hits=hits,
                    start=press,
                    end=end,
                )
            self._emit_selection_event(event_value)
            self._reset_selection_cycle()
        elif not self._selection_committed_on_press:
            self._emit_click_selection(end, operation)
        self._finish_selection_gesture()

    def _on_toplevel_selection_release(self, event: tk.Event) -> None:
        """Commit an active gesture when Tk skips the canvas release bind."""

        if self._selection_press is None:
            return
        try:
            x = int(event.x_root) - int(self.canvas.winfo_rootx())
            y = int(event.y_root) - int(self.canvas.winfo_rooty())
        except (AttributeError, tk.TclError, TypeError, ValueError):
            # Synthetic/custom Tk events may not expose usable root
            # coordinates.  Their local coordinates are still the best safe
            # fallback and match a release delivered directly to the canvas.
            x, y = int(event.x), int(event.y)
        event.x, event.y = x, y
        self._on_selection_release(event)

    def _emit_click_selection(
        self, point: Tuple[int, int], operation: SelectionOperation
    ) -> Tuple[SelectionHit, ...]:
        candidates = self.query_point(*point)
        now = time.monotonic()
        config = self._selection_config
        signature = (
            tuple(hit.key for hit in candidates),
            config.filter,
            config.depth,
            operation,
            self._selection_index_key,
        )
        same_anchor = (
            self._cycle_anchor is not None
            and math.hypot(
                point[0] - self._cycle_anchor[0], point[1] - self._cycle_anchor[1]
            )
            <= config.cycle_radius_px
        )
        same_cycle = (
            candidates
            and signature == self._cycle_signature
            and same_anchor
            and (now - self._cycle_time) * 1000.0 <= config.cycle_timeout_ms
        )
        self._cycle_index = (
            (self._cycle_index + 1) % len(candidates) if same_cycle else (0 if candidates else -1)
        )
        self._cycle_signature = signature if candidates else None
        self._cycle_anchor = point if candidates else None
        self._cycle_time = now
        active = () if self._cycle_index < 0 else (candidates[self._cycle_index],)
        self._emit_selection_event(
            SelectionEvent(
                SelectionGesture.CLICK,
                operation,
                hits=active,
                candidates=candidates,
                start=point,
                end=point,
                cycle_index=max(0, self._cycle_index),
                cycle_total=len(candidates),
            )
        )
        return active

    def _emit_selection_event(self, event: SelectionEvent) -> None:
        if self._selection_callback is not None:
            self._selection_callback(event)
        callback = self._pick.pick_callback
        if callback is not None and event.gesture == SelectionGesture.CLICK:
            hit = event.hits[0] if event.hits else None
            shift, ctrl, alt = self._selection_modifiers
            callback(
                Pick(
                    tag="" if hit is None else hit.key,
                    item=-1 if hit is None else hit.item,
                    x=event.end[0],
                    y=event.end[1],
                    shift=shift,
                    ctrl=ctrl,
                    alt=alt,
                )
            )

    def _on_selection_hover_motion(self, event: tk.Event) -> None:
        if self._selection_press is not None:
            # Some Tk/Windows configurations lose ButtonRelease while the
            # pointer grab is active.  The first ordinary Motion after the
            # physical release has no Button1 state; use it to finish a click
            # or box that would otherwise remain stranded indefinitely.
            if not (int(getattr(event, "state", 0) or 0) & 0x0100):
                self._on_selection_release(event)
            return
        if self._is_dragging:
            return
        self._hover_position = (int(event.x), int(event.y))
        if self._hover_after_id is None:
            self._hover_after_id = self.after(33, self._run_selection_hover)

    def _run_selection_hover(self) -> None:
        self._hover_after_id = None
        point = self._hover_position
        if point is None or self._selection_press is not None or self._is_dragging:
            return
        candidates = self.query_point(*point)
        hit = next((candidate for candidate in candidates if candidate.visible), None)
        key = None if hit is None else hit.key
        if key == self._hover_key:
            return
        self._hover_key = key
        self.set_preselection(key)
        if self._selection_hover_callback is not None:
            self._selection_hover_callback(hit)
        if self._pick.hover_callback is not None:
            self._pick.hover_callback(
                None
                if hit is None
                else Pick(hit.key, hit.item, point[0], point[1])
            )

    def _finish_selection_gesture(self) -> None:
        self._selection_press = None
        self._selection_dragging = False
        self._selection_points = []
        self._selection_committed_on_press = False
        self._selection_press_hit_keys = frozenset()
        if self._selection_overlay is not None:
            try:
                self.canvas.delete(self._selection_overlay)
            except tk.TclError:
                pass
            self._selection_overlay = None
        try:
            if self.canvas.grab_current() is self.canvas:
                self.canvas.grab_release()
        except tk.TclError:
            pass

    def _cancel_selection_gesture(self, _event: Optional[tk.Event] = None) -> None:
        self._finish_selection_gesture()

    def _on_selection_focus_out(self, _event: tk.Event) -> None:
        for key in self._tracked_modifiers:
            self._tracked_modifiers[key] = False
        self._cancel_selection_gesture()

    def _reset_selection_cycle(self) -> None:
        self._cycle_signature = None
        self._cycle_anchor = None
        self._cycle_index = -1
        self._cycle_time = 0.0

    def _selection_view_key(self, scene: _CompiledScene) -> Tuple[Any, ...]:
        camera = self.camera
        return (
            id(scene),
            self.width,
            self.height,
            self._plot_width(),
            camera.position.x,
            camera.position.y,
            camera.position.z,
            camera.target.x,
            camera.target.y,
            camera.target.z,
            camera.fov,
            camera.near,
            camera.far,
            None if self._section_plane is None else self._section_plane.key,
        )

    def _get_selection_index(self) -> ProjectedSelectionIndex:
        """Return projected full-detail selection geometry for this view."""

        self.width, self.height = self._viewport_size()
        scene = self._get_scene("full")
        key = self._selection_view_key(scene)
        if self._selection_index is not None and self._selection_index_key == key:
            return self._selection_index

        right, camera_up, forward = self.camera.basis()
        position = self.camera.position
        origin = np.array([position.x, position.y, position.z], dtype=np.float32)
        basis = np.array(
            [
                [right.x, camera_up.x, forward.x],
                [right.y, camera_up.y, forward.y],
                [right.z, camera_up.z, forward.z],
            ],
            dtype=np.float32,
        )
        y_scale = 1.0 / math.tan(self.camera.fov / 2.0)
        plot_width = self._plot_width()
        x_scale = y_scale * self.height / plot_width
        half_width = 0.5 * plot_width
        half_height = 0.5 * self.height
        near = max(1.0e-9, float(self.camera.near))
        section_plane = self._section_plane
        if section_plane is not None and not section_plane.enabled:
            section_plane = None
        projected: List[ProjectedPrimitive] = []
        primitive_id = 0

        def world_values_to_camera(values: Sequence[Sequence[float]]) -> np.ndarray:
            world = np.asarray(values, dtype=np.float32).reshape(-1, 3)
            return (world - origin) @ basis

        def screen_points(
            values: Sequence[Sequence[float]],
        ) -> Tuple[Tuple[Tuple[float, float], ...], Tuple[float, ...]]:
            points: List[Tuple[float, float]] = []
            depths: List[float] = []
            for camera_x, camera_y, camera_depth in values:
                depth_value = max(float(camera_depth), near)
                points.append(
                    (
                        (float(camera_x) * x_scale / depth_value + 1.0) * half_width,
                        (1.0 - float(camera_y) * y_scale / depth_value) * half_height,
                    )
                )
                depths.append(depth_value)
            return tuple(points), tuple(depths)

        def clipped_segment(
            first: Sequence[float], second: Sequence[float]
        ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
            a = tuple(float(value) for value in first)
            b = tuple(float(value) for value in second)
            a_in = a[2] >= near
            b_in = b[2] >= near
            if not a_in and not b_in:
                return None
            if a_in and b_in:
                return a, b
            span = b[2] - a[2]
            if abs(span) <= _EPS:
                return None
            parameter = (near - a[2]) / span
            intersection = (
                a[0] + parameter * (b[0] - a[0]),
                a[1] + parameter * (b[1] - a[1]),
                near,
            )
            return (a, intersection) if a_in else (intersection, b)

        camera_space, _screen_x, _screen_y, valid = self._project(
            scene.face_vertices,
            origin,
            basis,
            x_scale,
            y_scale,
            half_width,
            half_height,
            near,
        )
        for face in range(scene.face_total):
            start = int(scene.face_start[face])
            stop = start + int(scene.face_count[face])
            if stop <= start:
                continue
            if bool(scene.face_is_edge[face]):
                if section_plane is None:
                    edge_camera = camera_space[start:stop]
                else:
                    section_line = section_plane.clip_segment(
                        scene.face_vertices[start],
                        scene.face_vertices[min(start + 1, stop - 1)],
                    )
                    if section_line is None:
                        continue
                    edge_camera = world_values_to_camera(
                        [point.to_tuple() for point in section_line]
                    )
                clipped_line = clipped_segment(edge_camera[0], edge_camera[-1])
                if clipped_line is None:
                    continue
                points_2d, depths = screen_points(clipped_line)
                shape = "segment"
                radius = max(1.5, 0.5 * float(scene.face_width[face]))
            else:
                if section_plane is None:
                    camera_polygon = camera_space[start:stop]
                else:
                    section_polygon = section_plane.clip_polygon(
                        scene.face_vertices[start:stop]
                    )
                    if len(section_polygon) < 3:
                        continue
                    camera_polygon = world_values_to_camera(
                        [point.to_tuple() for point in section_polygon]
                    )
                polygon_valid = camera_polygon[:, 2] > near
                if bool(np.all(polygon_valid)):
                    clipped_polygon = [tuple(float(value) for value in row) for row in camera_polygon]
                elif bool(np.any(polygon_valid)):
                    clipped_polygon = self._clip_near_plane(camera_polygon, near)
                else:
                    continue
                if len(clipped_polygon) < 3:
                    continue
                points_2d, depths = screen_points(clipped_polygon)
                shape = "polygon"
                radius = 0.0
            projected.append(
                ProjectedPrimitive(
                    index=primitive_id,
                    shape=shape,
                    points=points_2d,
                    depths=depths,
                    binding=scene.face_binding(face) or fallback_binding(scene.tags[face]),
                    layer=float(scene.face_layer[face]),
                    radius=radius,
                )
            )
            primitive_id += 1

        if len(scene.line_vertices):
            line_camera, _line_x, _line_y, _line_valid = self._project(
                scene.line_vertices,
                origin,
                basis,
                x_scale,
                y_scale,
                half_width,
                half_height,
                near,
            )
            for line in range(len(scene.line_vertices) // 2):
                start = 2 * line
                if section_plane is None:
                    candidate_line = line_camera[start:start + 2]
                else:
                    section_line = section_plane.clip_segment(
                        scene.line_vertices[start], scene.line_vertices[start + 1]
                    )
                    if section_line is None:
                        continue
                    candidate_line = world_values_to_camera(
                        [point.to_tuple() for point in section_line]
                    )
                clipped_line = clipped_segment(candidate_line[0], candidate_line[1])
                if clipped_line is None:
                    continue
                points_2d, depths = screen_points(clipped_line)
                projected.append(
                    ProjectedPrimitive(
                        index=primitive_id,
                        shape="segment",
                        points=points_2d,
                        depths=depths,
                        binding=scene.line_binding(line)
                        or fallback_binding(scene.line_tags[line]),
                        layer=float(scene.line_layer[line]),
                        radius=max(1.5, 0.5 * float(scene.line_width[line])),
                    )
                )
                primitive_id += 1

        if len(scene.marker_points):
            marker_camera, marker_x, marker_y, marker_valid = self._project(
                scene.marker_points,
                origin,
                basis,
                x_scale,
                y_scale,
                half_width,
                half_height,
                near,
            )
            for marker in range(len(scene.marker_points)):
                if not bool(marker_valid[marker]):
                    continue
                if section_plane is not None and not section_plane.contains(
                    scene.marker_points[marker]
                ):
                    continue
                projected.append(
                    ProjectedPrimitive(
                        index=primitive_id,
                        shape="point",
                        points=((float(marker_x[marker]), float(marker_y[marker])),),
                        depths=(float(marker_camera[marker, 2]),),
                        binding=scene.marker_binding(marker),
                        layer=float(scene.marker_layers[marker]),
                        radius=0.5 * float(scene.marker_sizes[marker]),
                    )
                )
                primitive_id += 1

        self._selection_index = ProjectedSelectionIndex(
            projected, self.width, self.height
        )
        self._selection_index_key = key
        return self._selection_index

    def _on_mouse_down(self, event: tk.Event, mode: str) -> None:
        self._hover_key = None
        self.set_preselection(None)
        self._last_mouse_x = int(event.x)
        self._last_mouse_y = int(event.y)
        self._is_dragging = True
        self._drag_mode = str(mode)
        self._interactive_render = True
        self.canvas.focus_set()

    def _on_mouse_up(self, _event: tk.Event) -> None:
        self._is_dragging = False
        self._drag_mode = ""
        self._interactive_render = False
        self._cancel_scheduled_redraw()
        self._request_redraw()

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if not self._is_dragging:
            return

        dx = int(event.x) - self._last_mouse_x
        dy = int(event.y) - self._last_mouse_y
        self._last_mouse_x = int(event.x)
        self._last_mouse_y = int(event.y)

        if self._drag_mode == "rotate":
            self.camera.orbit(
                delta_azimuth=-dx * 0.008,
                delta_elevation=dy * 0.008,
            )
        else:
            self.camera.pan_view_pixels(dx, dy, self._plot_width(), self.height)
        self._interactive_render = True
        self._request_redraw(interactive=True)

    def _on_mouse_wheel(self, event: tk.Event) -> str:
        event_num = getattr(event, "num", None)
        event_delta = getattr(event, "delta", 0)

        if event_num == 4 or event_delta > 0:
            self.camera.zoom(0.90)
        elif event_num == 5 or event_delta < 0:
            self.camera.zoom(1.10)
        else:
            return "break"

        self._interactive_render = True
        self._request_redraw(interactive=True)

        if self._finish_interaction_after_id is not None:
            try:
                self.after_cancel(self._finish_interaction_after_id)
            except tk.TclError:
                pass
        self._finish_interaction_after_id = self.after(120, self._finish_interaction)
        return "break"

    def _finish_interaction(self) -> None:
        self._finish_interaction_after_id = None
        if self._is_dragging:
            return
        self._interactive_render = False
        self._cancel_scheduled_redraw()
        self._request_redraw()

    def _cancel_scheduled_redraw(self) -> None:
        if self._redraw_after_id is not None:
            try:
                self.after_cancel(self._redraw_after_id)
            except tk.TclError:
                pass
            self._redraw_after_id = None

    def _request_redraw(self, interactive: Optional[bool] = None) -> None:
        if interactive is None:
            interactive = self._interactive_render
        if self._redraw_after_id is not None:
            return

        if interactive:
            self._redraw_after_id = self.after(
                self._interactive_delay_ms,
                self._run_scheduled_redraw,
            )
        else:
            self._redraw_after_id = self.after_idle(self._run_scheduled_redraw)

    def _run_scheduled_redraw(self) -> None:
        self._redraw_after_id = None
        self.redraw()

    # ------------------------------------------------------------------
    # Scene lifecycle and cache management
    # ------------------------------------------------------------------

    @property
    def animation_frames(self) -> int:
        """Number of frames currently held in the animation cache."""
        return len(self._animation_cache)

    @property
    def animation_frame_index(self) -> int:
        """Index of the frame shown last; useful for a progress readout."""
        return self._animation_frame_index

    @property
    def is_playing_animation(self) -> bool:
        return self._is_playing_animation

    def begin_animation_cache(self) -> None:
        self.stop_animation()
        self._animation_cache.clear()

    def capture_animation_frame(self) -> None:
        """
        Freeze the current scene as one animation frame.

        Both quality levels are stored, but a scene without cylinders or
        stiffeners compiles to a single shared representation, so a polygon
        or mesh result field costs one build rather than two.  The occluder
        set is captured too: playback must not test frames against whatever
        happens to be in ``objects`` when they are shown.
        """
        self._is_capturing_animation = True
        try:
            full_key = self._quality_key("full")
            fast_key = self._quality_key("fast")
            self._animation_cache.append(
                {
                    "scene_full": self._get_scene("full"),
                    "scene_fast": self._get_scene("fast"),
                    "primitives_full": self._world_primitive_cache.get(full_key, []),
                    "primitives_fast": self._world_primitive_cache.get(fast_key, []),
                    "legend": self._thickness_legend,
                    "occluders": self._collect_opaque_cylinder_occluders(),
                }
            )
        finally:
            self._is_capturing_animation = False

    def play_animation(self, fps: int = 30, fast: Optional[bool] = None) -> None:
        """
        Replay the captured frames.

        ``fast`` picks the reduced-detail render path (screen-space level of
        detail, no stipple, no per-face outlines).  The default adapts: full
        detail is used while it keeps up with ``fps``, and playback drops to
        the fast path once frames start running over their budget.
        """
        if not self._animation_cache:
            return
        self.stop_animation()
        self._animation_fps = max(1, fps)
        self._animation_fast = fast
        self._animation_frame_budget_ms = max(1.0, 1000.0 / self._animation_fps)
        self._is_playing_animation = True
        self._animation_frame_index = 0
        self._animation_tick()

    def stop_animation(self) -> None:
        was_playing = self._is_playing_animation
        self._is_playing_animation = False
        self._animation_occluders = None
        if self._animation_after_id is not None:
            self.after_cancel(self._animation_after_id)
            self._animation_after_id = None
        if was_playing:
            # Leave a full-quality frame behind, whatever playback was using.
            self._interactive_render = False
        # Restore normal view
        self._request_redraw(interactive=False)

    def capture_image(self):
        """Return the visible drawable area as a top-left-oriented Pillow image.

        Tk has no dependency-free raster readback.  When Pillow is installed,
        capture the mapped inner canvas rather than its surrounding frame,
        toolbars, or window chrome.  Pillow remains a lazy optional import.
        """

        try:
            from PIL import ImageGrab
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "capture_image requires Pillow (installed by the GUI application extras)"
            ) from error

        widget = self.event_widget
        try:
            widget.update_idletasks()
            x0 = int(widget.winfo_rootx())
            y0 = int(widget.winfo_rooty())
            width = int(widget.winfo_width())
            height = int(widget.winfo_height())
        except (AttributeError, TypeError, ValueError, tk.TclError) as error:
            raise RuntimeError("the software viewport is not available for capture") from error
        if width <= 1 or height <= 1:
            raise RuntimeError("the software viewport must be visible before capture")
        try:
            return ImageGrab.grab(
                bbox=(x0, y0, x0 + width, y0 + height)
            ).convert("RGBA")
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(
                "the operating system could not capture the visible software viewport"
            ) from error

    def _animation_tick(self) -> None:
        if not self._is_playing_animation or not self._animation_cache:
            return

        frame = self._animation_cache[self._animation_frame_index]

        saved_scenes = dict(self._scene_cache)
        saved_primitives = dict(self._world_primitive_cache)
        saved_legend = self._thickness_legend
        saved_interactive = self._interactive_render

        # Frames were captured under the object list of their own moment, so
        # the cache keys are recomputed here rather than reused from `objects`.
        self._scene_cache["full"] = frame["scene_full"]
        self._scene_cache["fast"] = frame["scene_fast"]
        self._world_primitive_cache["full"] = frame["primitives_full"]
        self._world_primitive_cache["fast"] = frame["primitives_fast"]
        self._thickness_legend = frame["legend"]
        self._animation_occluders = frame.get("occluders") or []
        self._interactive_render = bool(self._animation_fast)

        started = time.perf_counter()
        self.redraw()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        self._scene_cache = saved_scenes
        self._world_primitive_cache = saved_primitives
        self._thickness_legend = saved_legend
        self._interactive_render = saved_interactive
        self._animation_occluders = None

        # Auto mode: once a frame overruns its slot, switch to the fast path
        # and stay there for the rest of the playback.
        if self._animation_fast is None and elapsed_ms > self._animation_frame_budget_ms:
            self._animation_fast = True

        self._animation_frame_index = (self._animation_frame_index + 1) % len(self._animation_cache)
        delay_ms = max(1, int(1000.0 / self._animation_fps))
        self._animation_after_id = self.after(delay_ms, self._animation_tick)

    def _invalidate_geometry_cache(self) -> None:
        self._world_primitive_cache.clear()
        self._scene_cache.clear()
        self._quality_lod_flag = None
        self._selection_index = None
        self._selection_index_key = None
        pick = getattr(self, "_pick", None)
        if pick is not None:
            pick.invalidate()
        if hasattr(self, "_cycle_signature"):
            self._reset_selection_cycle()

    def _clear_canvas_only(self) -> None:
        self.canvas.delete("all")
        self._hud_signature = None
        self._polygon_pool.clear()
        self._polygon_state.clear()
        self._line_pool.clear()
        self._line_state.clear()
        self._marker_pool.clear()
        self._marker_state.clear()
        self._text_pool.clear()
        self._text_state.clear()

    def clear(self, keep_canvas: bool = False) -> None:
        retained = [
            obj.get("handle")
            for obj in self.objects
            if obj.get("type") == "mesh_arrays"
        ]
        for handle in retained:
            if isinstance(handle, MeshHandle):
                handle.remove()
        self.objects.clear()
        self._explicit_opaque_cylinder_occluders.clear()
        if hasattr(self, "_hover_key"):
            self._hover_key = None
        if hasattr(self, "_pick"):
            self._pick.set_preselection(None)
        self._invalidate_geometry_cache()
        if keep_canvas:
            # Retained-handle removal callbacks request a redraw.  Preserve
            # the historic keep_canvas contract by cancelling that idle draw
            # after the scene state has been cleared.
            self._cancel_scheduled_redraw()
        else:
            self._clear_canvas_only()

    def destroy(self) -> None:
        if getattr(self, "_destroyed", False):
            return
        self._destroyed = True
        scheduler = getattr(self, "_update_scheduler", None)
        if scheduler is not None:
            poll_id = getattr(self, "_update_poll_id", None)
            if poll_id is not None:
                try:
                    self.after_cancel(poll_id)
                except tk.TclError:
                    pass
                self._update_poll_id = None
            scheduler.close()
        super().destroy()

    def _get_ruler_primitives(self) -> List[Dict[str, Any]]:
        bounds = self._scene_bounds()
        if bounds is None:
            return []
        min_p, max_p = bounds
        # Extend bounds slightly so ruler sits outside geometry
        span_x = max_p.x - min_p.x
        span_y = max_p.y - min_p.y
        span_z = max_p.z - min_p.z
        padding = max(1.0, max(span_x, max(span_y, span_z))) * 0.05
        min_p = Point3D(min_p.x - padding, min_p.y - padding, min_p.z - padding)
        max_p = Point3D(max_p.x + padding, max_p.y + padding, max_p.z + padding)

        primitives: List[Dict[str, Any]] = []
        color = "#1f2937"

        def ruler_line(start: Point3D, end: Point3D, width: float) -> Dict[str, Any]:
            return {
                "kind": "line",
                "start": start,
                "end": end,
                "color": color,
                "width": width,
                "layer": 30,
                "draw_overlay": True,
            }

        primitives.append(ruler_line(Point3D(min_p.x, min_p.y, min_p.z), Point3D(max_p.x, min_p.y, min_p.z), 1.5))
        primitives.append(ruler_line(Point3D(min_p.x, min_p.y, min_p.z), Point3D(min_p.x, max_p.y, min_p.z), 1.5))
        primitives.append(ruler_line(Point3D(min_p.x, max_p.y, min_p.z), Point3D(min_p.x, max_p.y, max_p.z), 1.5))

        def get_ticks(start: float, end: float, count: int = 4) -> List[float]:
            if start >= end:
                return [start]
            return [start + i * (end - start) / count for i in range(count + 1)]

        tick_size = padding * 0.3

        def tick_text(point: Point3D, text: str, font: Tuple[Any, ...]) -> Dict[str, Any]:
            return {
                "kind": "text",
                "point": point,
                "text": text,
                "color": color,
                "font": font,
                "anchor": "center",
                "layer": 35,
            }

        tick_font = ("Arial", 9)
        for val in get_ticks(min_p.x + padding, max_p.x - padding):
            primitives.append(ruler_line(Point3D(val, min_p.y, min_p.z), Point3D(val, min_p.y - tick_size, min_p.z), 1.0))
            primitives.append(tick_text(Point3D(val, min_p.y - tick_size * 2.5, min_p.z), f"{val:.1f}", tick_font))

        for val in get_ticks(min_p.y + padding, max_p.y - padding):
            primitives.append(ruler_line(Point3D(min_p.x, val, min_p.z), Point3D(min_p.x - tick_size, val, min_p.z), 1.0))
            primitives.append(tick_text(Point3D(min_p.x - tick_size * 2.5, val, min_p.z), f"{val:.1f}", tick_font))

        for val in get_ticks(min_p.z + padding, max_p.z - padding):
            primitives.append(ruler_line(Point3D(min_p.x, max_p.y, val), Point3D(min_p.x - tick_size, max_p.y, val), 1.0))
            primitives.append(tick_text(Point3D(min_p.x - tick_size * 2.5, max_p.y, val), f"{val:.1f}", tick_font))

        # Axis name labels at the positive ends of each ruler line.
        label_font = ("Arial", 10, "bold")
        primitives.append(tick_text(Point3D(max_p.x + tick_size * 2.0, min_p.y, min_p.z), "x [m]", label_font))
        primitives.append(tick_text(Point3D(min_p.x, max_p.y + tick_size * 2.0, min_p.z), "y [m]", label_font))
        primitives.append(tick_text(Point3D(min_p.x, max_p.y, max_p.z + tick_size * 2.0), "z [m]", label_font))

        return primitives

    # ------------------------------------------------------------------
    # Scene compilation
    # ------------------------------------------------------------------

    def _quality_key(self, quality: str) -> str:
        """
        Collapse the two quality levels when they would produce the same thing.

        Only cylinders and stiffeners re-tessellate for interactive frames;
        a scene made of polygons and meshes - an FE result, for instance -
        builds one identical primitive list either way.  Sharing it halves
        both the build time and the memory, which matters most when caching
        an animation frame by frame.
        """
        if quality != "fast":
            return "full"
        flag = self._quality_lod_flag
        if flag is None:
            flag = any(
                obj.get("type") in ("cylinder", "stiffener") for obj in self.objects
            )
            self._quality_lod_flag = flag
        return "fast" if flag else "full"

    def _get_world_primitives(self, quality: str) -> List[Dict[str, Any]]:
        """Return (and cache) the world-space primitive dictionaries."""
        key = self._quality_key(quality)
        cached = self._world_primitive_cache.get(key)
        if cached is not None:
            return cached

        primitives: List[Dict[str, Any]] = []
        for index, obj in enumerate(self.objects):
            primitives.extend(self._object_to_primitives(obj, key, index))

        if self.show_axis_ruler:
            primitives.extend(self._get_ruler_primitives())

        self._world_primitive_cache[key] = primitives
        return primitives

    def _get_scene(self, quality: str) -> _CompiledScene:
        # Colours the scene resolved through the shared scale go stale when
        # the application swaps colour maps, so drop the cache on a change.
        generation = color_stop_generation()
        if generation != self._color_stop_generation:
            self._color_stop_generation = generation
            self._invalidate_geometry_cache()

        key = self._quality_key(quality)
        scene = self._scene_cache.get(key)
        if scene is not None:
            return scene
        scene = self._compile(self._get_world_primitives(key))
        self._scene_cache[key] = scene
        return scene

    @staticmethod
    def _resolve_stipples(primitive: Dict[str, Any]) -> Tuple[str, str]:
        """
        Front and back stipple patterns for one surface.

        Front and back faces of a transparent surface get non-overlapping
        stipple windows, so a shell shows its far wall through its near wall
        instead of masking it.  An explicit stipple string is used verbatim:
        the caller asked for that exact pattern.
        """
        explicit = primitive.get("stipple", "")
        if explicit:
            return explicit, explicit
        opacity = primitive.get("opacity")
        if opacity is None or float(opacity) >= stipple_module.OPAQUE_THRESHOLD:
            return "", ""
        rotation = int(primitive.get("stipple_phase", 0))
        return (
            stipple_module.for_opacity(opacity, 0, rotation),
            stipple_module.for_opacity(opacity, 1, rotation),
        )

    @staticmethod
    def _batch_centers_and_normals(
        vertices: np.ndarray,
        counts: np.ndarray,
        offsets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Centroids and Newell normals for a whole batch of faces at once.

        This is the same maths :meth:`_polygon_normal` does per face, run as
        array operations so that a mesh of tens of thousands of elements
        never pays Python's per-face overhead.
        """
        centers = np.add.reduceat(vertices, offsets, axis=0) / counts[:, None]

        # Pair every vertex with the next one *within its own face*, so the
        # edge sum wraps at each face boundary instead of running on.
        repeated_starts = np.repeat(offsets, counts)
        repeated_counts = np.repeat(counts, counts)
        position = np.arange(len(vertices), dtype=np.int64) - repeated_starts
        following = repeated_starts + (position + 1) % repeated_counts

        current = vertices
        successor = vertices[following]
        terms = np.empty_like(vertices)
        terms[:, 0] = (current[:, 1] - successor[:, 1]) * (current[:, 2] + successor[:, 2])
        terms[:, 1] = (current[:, 2] - successor[:, 2]) * (current[:, 0] + successor[:, 0])
        terms[:, 2] = (current[:, 0] - successor[:, 0]) * (current[:, 1] + successor[:, 1])

        normals = np.add.reduceat(terms, offsets, axis=0)
        lengths = np.sqrt(np.einsum("ij,ij->i", normals, normals))
        normals /= np.maximum(lengths, _EPS)[:, None]
        normals[lengths <= _EPS] = 0.0
        return centers.astype(np.float32, copy=False), normals.astype(np.float32, copy=False)

    def _compile(self, primitives: Sequence[Dict[str, Any]]) -> _CompiledScene:
        """Flatten primitive dictionaries into the per-frame array layout."""
        scene = _CompiledScene()
        scene.primitives = list(primitives)

        # Vertices arrive either one face at a time (polygons, lines) or as a
        # ready-made array (batched faces).  Collecting blocks and joining them
        # once at the end lets a batch skip the per-vertex Python entirely.
        pending: List[Tuple[float, float, float]] = []
        vertex_blocks: List[np.ndarray] = []
        next_vertex = 0

        def flush_pending() -> None:
            if pending:
                vertex_blocks.append(np.asarray(pending, dtype=np.float32))
                pending.clear()

        face_start: List[int] = []
        face_count: List[int] = []
        face_normal_blocks: List[np.ndarray] = []
        face_center_blocks: List[np.ndarray] = []
        face_normal: List[Tuple[float, float, float]] = []
        face_center: List[Tuple[float, float, float]] = []
        face_layer: List[float] = []
        face_phase: List[int] = []
        face_cull: List[bool] = []
        face_lit: List[bool] = []
        face_is_edge: List[bool] = []
        face_application_selected: List[bool] = []
        opaque: List[bool] = []
        fast_no_outline: List[bool] = []

        def flush_face_attributes() -> None:
            if face_normal:
                face_normal_blocks.append(np.asarray(face_normal, dtype=np.float32))
                face_center_blocks.append(np.asarray(face_center, dtype=np.float32))
                face_normal.clear()
                face_center.clear()

        line_vertices: List[Tuple[float, float, float]] = []
        line_vertex_blocks: List[np.ndarray] = []
        line_layer: List[float] = []
        marker_points: List[Tuple[float, float, float]] = []
        marker_layer: List[float] = []
        text_points: List[Tuple[float, float, float]] = []
        text_layer: List[float] = []
        packed_face_spans: List[
            Tuple[int, int, PackedOwnerTable, np.ndarray, Optional[Callable[..., object]]]
        ] = []
        packed_line_spans: List[
            Tuple[int, int, PackedOwnerTable, np.ndarray, Optional[Callable[..., object]]]
        ] = []
        packed_marker_spans: List[
            Tuple[int, int, PackedOwnerTable, np.ndarray, Optional[Callable[..., object]]]
        ] = []

        occlude_lines = self._occlude_lines

        for primitive in primitives:
            kind = primitive.get("kind")
            binding = primitive.get("binding")
            visibility_state = getattr(self, "_visibility_state", VisibilityState())
            if (
                not visibility_state.is_default
                and binding is not None
                and not visibility_state.accepts(binding.owners)
            ):
                continue

            if kind == "markers":
                points = primitive.get("points", ())
                marker_points.extend(
                    (float(point[0]), float(point[1]), float(point[2]))
                    if not isinstance(point, Point3D)
                    else (point.x, point.y, point.z)
                    for point in points
                )
                total = len(points)
                scene.marker_colors.extend(primitive.get("colors", ["#2563eb"] * total))
                scene.marker_outlines.extend(
                    primitive.get("outlines", [""] * total)
                )
                scene.marker_sizes.extend(primitive.get("sizes", [5] * total))
                marker_layer.extend([float(primitive.get("layer", 32))] * total)
                tags = primitive.get("tags") or ""
                scene.marker_tags.extend([tags] * total)
                scene.marker_bindings.extend(
                    primitive.get("bindings")
                    or _coerce_pick_bindings(None, total, tags)
                )
                owner_table = primitive.get("owner_table")
                if owner_table is not None:
                    owner_primitives = np.asarray(
                        primitive.get("owner_primitives", np.arange(total)),
                        dtype=np.uint32,
                    )
                    if len(owner_primitives) != total:
                        raise ValueError("owner_primitives must have one value per marker")
                    packed_marker_spans.append(
                        (
                            len(scene.marker_bindings) - total,
                            total,
                            owner_table,
                            owner_primitives,
                            primitive.get("owner_resolver"),
                        )
                    )
                    scene.any_tags = True
                if tags:
                    scene.any_tags = True
                continue

            if kind == "faces":
                flush_pending()
                flush_face_attributes()
                vertices = primitive["vertices"]
                counts = primitive["counts"]
                total = len(counts)
                offsets = np.zeros(total, dtype=np.int64)
                np.cumsum(counts[:-1], out=offsets[1:])

                centers = primitive.get("centers")
                normals = primitive.get("normals")
                if centers is None or normals is None:
                    centers, normals = self._batch_centers_and_normals(
                        vertices, counts, offsets
                    )
                face_center_blocks.append(centers)
                face_normal_blocks.append(normals)
                face_start.extend((offsets + next_vertex).tolist())
                face_count.extend(counts.tolist())
                vertex_blocks.append(vertices)
                next_vertex += len(vertices)

                layer = float(primitive.get("layer", 5))
                two_sided = bool(primitive.get("two_sided_shell", False))
                face_layer.extend([layer] * total)
                face_phase.extend([0 if two_sided else 1] * total)
                face_cull.extend([bool(primitive.get("cull_backface", False))] * total)
                face_lit.extend([bool(primitive.get("lit", True))] * total)
                face_is_edge.extend([False] * total)
                selected = np.asarray(
                    primitive.get("application_selected", np.zeros(total, dtype=bool)),
                    dtype=bool,
                )
                if selected.shape != (total,):
                    raise ValueError(
                        "application_selected must have one value per face"
                    )
                face_application_selected.extend(selected.tolist())
                fast_no_outline.extend(
                    [bool(primitive.get("fast_no_outline", True))] * total
                )

                colors = primitive["colors"]
                back_colors = primitive.get("back_colors") or colors
                scene.base_front.extend(colors)
                scene.base_back.extend(back_colors)
                scene.outline.extend([primitive.get("outline", "")] * total)
                scene.face_width.extend([primitive.get("width", 1)] * total)
                tags = primitive.get("tags") or ""
                scene.tags.extend([tags] * total)
                scene.face_bindings.extend(
                    primitive.get("bindings")
                    or _coerce_pick_bindings(None, total, tags)
                )
                owner_table = primitive.get("owner_table")
                if owner_table is not None:
                    owner_primitives = np.asarray(
                        primitive.get("owner_primitives", np.arange(total)),
                        dtype=np.uint32,
                    )
                    if len(owner_primitives) != total:
                        raise ValueError("owner_primitives must have one value per face")
                    packed_face_spans.append(
                        (
                            len(face_start) - total,
                            total,
                            owner_table,
                            owner_primitives,
                            primitive.get("owner_resolver"),
                        )
                    )
                    scene.any_tags = True
                if tags:
                    scene.any_tags = True

                front_stipple, back_stipple = self._resolve_stipples(primitive)
                opaque.extend([not front_stipple] * total)
                scene.stipple_front.extend([front_stipple] * total)
                scene.stipple_back.extend([back_stipple] * total)
                continue

            if kind == "lines":
                vertices = np.asarray(primitive.get("vertices"), dtype=np.float32)
                total = int(primitive.get("total", len(vertices) // 2))
                if vertices.shape != (2 * total, 3):
                    raise ValueError("line batch must contain two vertices per line")
                if line_vertices:
                    line_vertex_blocks.append(np.asarray(line_vertices, dtype=np.float32))
                    line_vertices.clear()
                line_vertex_blocks.append(vertices)
                scene.line_color.extend([primitive.get("color", "black")] * total)
                scene.line_width.extend([primitive.get("width", 1)] * total)
                tags = primitive.get("tags") or ""
                scene.line_tags.extend([tags] * total)
                scene.line_bindings.extend([None] * total)
                owner_table = primitive.get("owner_table")
                if owner_table is not None:
                    owner_primitives = np.asarray(
                        primitive.get("owner_primitives", np.arange(total)),
                        dtype=np.uint32,
                    )
                    if len(owner_primitives) != total:
                        raise ValueError("owner_primitives must have one value per line")
                    packed_line_spans.append(
                        (
                            len(scene.line_bindings) - total,
                            total,
                            owner_table,
                            owner_primitives,
                            primitive.get("owner_resolver"),
                        )
                    )
                    scene.any_tags = True
                line_layer.extend([float(primitive.get("layer", 30))] * total)
                if tags:
                    scene.any_tags = True
                continue

            if kind == "text":
                point = primitive["point"]
                text_points.append((point.x, point.y, point.z))
                text_layer.append(float(primitive.get("layer", 35)))
                scene.text_content.append(
                    (
                        str(primitive.get("text", "")),
                        primitive.get("color", "black"),
                        primitive.get("font", ("Segoe UI", 9, "bold")),
                        primitive.get("anchor", tk.CENTER),
                    )
                )
                continue

            if kind == "line":
                start = primitive["start"]
                end = primitive["end"]
                color = primitive.get("color", "black")
                width = primitive.get("width", 1)
                layer = float(primitive.get("layer", 30))
                if primitive.get("draw_overlay") or not occlude_lines:
                    line_vertices.append((start.x, start.y, start.z))
                    line_vertices.append((end.x, end.y, end.z))
                    scene.line_color.append(color)
                    scene.line_width.append(width)
                    overlay_tags = primitive.get("tags") or ""
                    scene.line_tags.append(overlay_tags)
                    scene.line_bindings.append(
                        _coerce_pick_binding(primitive.get("binding"), overlay_tags)
                    )
                    line_layer.append(layer)
                    continue
                # Depth-sorted lines share the face pipeline: a four-point
                # outline draws the segment and averages to its true midpoint.
                face_start.append(next_vertex)
                pending.extend(
                    (
                        (start.x, start.y, start.z),
                        (end.x, end.y, end.z),
                        (end.x, end.y, end.z),
                        (start.x, start.y, start.z),
                    )
                )
                next_vertex += 4
                face_count.append(4)
                face_normal.append((0.0, 0.0, 0.0))
                face_center.append(
                    (
                        0.5 * (start.x + end.x),
                        0.5 * (start.y + end.y),
                        0.5 * (start.z + end.z),
                    )
                )
                face_layer.append(layer)
                face_phase.append(1)
                face_cull.append(False)
                face_lit.append(False)
                face_is_edge.append(True)
                face_application_selected.append(False)
                opaque.append(True)
                fast_no_outline.append(False)
                scene.base_front.append("")
                scene.base_back.append("")
                scene.outline.append(color)
                scene.stipple_front.append("")
                scene.stipple_back.append("")
                scene.face_width.append(width)
                line_tags = primitive.get("tags") or ""
                scene.tags.append(line_tags)
                scene.face_bindings.append(
                    _coerce_pick_binding(primitive.get("binding"), line_tags)
                )
                if line_tags:
                    scene.any_tags = True
                continue

            if kind != "polygon":
                continue

            vertices = primitive["vertices"]
            face_start.append(next_vertex)
            pending.extend((v.x, v.y, v.z) for v in vertices)
            next_vertex += len(vertices)
            face_count.append(len(vertices))
            normal = primitive["normal"]
            center = primitive["center"]
            face_normal.append((normal.x, normal.y, normal.z))
            face_center.append((center.x, center.y, center.z))
            layer = float(primitive.get("layer", 0))
            face_layer.append(layer)
            two_sided = bool(primitive.get("two_sided_shell", False))
            face_phase.append(0 if two_sided else 1)
            face_cull.append(bool(primitive.get("cull_backface", False)))
            face_lit.append(bool(primitive.get("lit", True)))
            face_is_edge.append(False)
            face_application_selected.append(False)
            fast_no_outline.append(bool(primitive.get("fast_no_outline", True)))

            color = primitive["color"]
            back_color = primitive.get("back_color") or color
            scene.base_front.append(color)
            scene.base_back.append(back_color)
            scene.outline.append(primitive.get("outline", ""))
            scene.face_width.append(primitive.get("width", 1))
            tags = primitive.get("tags") or ""
            scene.tags.append(tags)
            scene.face_bindings.append(
                _coerce_pick_binding(primitive.get("binding"), tags)
            )
            if tags:
                scene.any_tags = True

            front_stipple, back_stipple = self._resolve_stipples(primitive)
            opaque.append(not front_stipple)
            scene.stipple_front.append(front_stipple)
            scene.stipple_back.append(back_stipple)

        flush_pending()
        flush_face_attributes()

        count = len(face_start)
        scene.face_vertices = (
            np.concatenate(vertex_blocks) if vertex_blocks
            else np.empty((0, 3), dtype=np.float32)
        )
        scene.face_start = np.asarray(face_start, dtype=np.int64) if count else np.empty(0, np.int64)
        scene.face_count = np.asarray(face_count, dtype=np.int64) if count else np.empty(0, np.int64)
        scene.face_normal = (
            np.concatenate(face_normal_blocks) if face_normal_blocks
            else np.empty((0, 3), np.float32)
        )
        scene.face_center = (
            np.concatenate(face_center_blocks) if face_center_blocks
            else np.empty((0, 3), np.float32)
        )
        scene.face_layer = np.asarray(face_layer, dtype=np.float32) if count else np.empty(0, np.float32)
        scene.face_phase = np.asarray(face_phase, dtype=np.int8) if count else np.empty(0, np.int8)
        scene.face_cull = np.asarray(face_cull, dtype=bool) if count else np.empty(0, bool)
        scene.face_lit = np.asarray(face_lit, dtype=bool) if count else np.empty(0, bool)
        scene.face_is_edge = np.asarray(face_is_edge, dtype=bool) if count else np.empty(0, bool)
        scene.face_application_selected = (
            np.asarray(face_application_selected, dtype=bool)
            if count else np.empty(0, bool)
        )
        scene.opaque = np.asarray(opaque, dtype=bool) if count else np.empty(0, bool)
        scene.fast_no_outline = (
            np.asarray(fast_no_outline, dtype=bool) if count else np.empty(0, bool)
        )
        # Member surfaces use layers 10-29; shells, result plates and selection
        # outlines are intentionally excluded from the hidden-surface filter.
        scene.face_occludable = (scene.face_layer >= 10.0) & (scene.face_layer < 30.0)

        scene.face_owner_table_slot = np.full(count, -1, dtype=np.int32)
        scene.face_owner_primitive = np.zeros(count, dtype=np.uint32)
        for start, total, table, primitives, resolver in packed_face_spans:
            table_slot = next(
                (
                    index
                    for index, existing in enumerate(scene.face_owner_tables)
                    if existing is table
                ),
                -1,
            )
            if table_slot < 0:
                table_slot = len(scene.face_owner_tables)
                scene.face_owner_tables.append(table)
                scene.face_owner_resolvers.append(resolver)
            scene.face_owner_table_slot[start : start + total] = table_slot
            scene.face_owner_primitive[start : start + total] = primitives

        def install_packed_spans(
            total: int,
            spans: List[
                Tuple[
                    int,
                    int,
                    PackedOwnerTable,
                    np.ndarray,
                    Optional[Callable[..., object]],
                ]
            ],
            tables: List[PackedOwnerTable],
            resolvers: List[Optional[Callable[..., object]]],
        ) -> Tuple[np.ndarray, np.ndarray]:
            slots = np.full(total, -1, dtype=np.int32)
            primitives = np.zeros(total, dtype=np.uint32)
            for start, span_total, table, source_primitives, resolver in spans:
                table_slot = next(
                    (
                        index
                        for index, existing in enumerate(tables)
                        if existing is table
                    ),
                    -1,
                )
                if table_slot < 0:
                    table_slot = len(tables)
                    tables.append(table)
                    resolvers.append(resolver)
                slots[start : start + span_total] = table_slot
                primitives[start : start + span_total] = source_primitives
            return slots, primitives

        (
            scene.line_owner_table_slot,
            scene.line_owner_primitive,
        ) = install_packed_spans(
            len(scene.line_bindings),
            packed_line_spans,
            scene.line_owner_tables,
            scene.line_owner_resolvers,
        )
        (
            scene.marker_owner_table_slot,
            scene.marker_owner_primitive,
        ) = install_packed_spans(
            len(scene.marker_bindings),
            packed_marker_spans,
            scene.marker_owner_tables,
            scene.marker_owner_resolvers,
        )

        scene.table_front = [shade_table(color) for color in scene.base_front]
        scene.table_back = [shade_table(color) for color in scene.base_back]

        if line_vertices:
            line_vertex_blocks.append(np.asarray(line_vertices, dtype=np.float32))
        scene.line_vertices = (
            np.concatenate(line_vertex_blocks)
            if line_vertex_blocks
            else np.empty((0, 3), dtype=np.float32)
        )
        scene.line_layer = np.asarray(line_layer, dtype=np.float32) if line_layer else np.empty(0, np.float32)
        scene.marker_points = (
            np.asarray(marker_points, dtype=np.float32)
            if marker_points
            else np.empty((0, 3), dtype=np.float32)
        )
        scene.marker_layers = (
            np.asarray(marker_layer, dtype=np.float32)
            if marker_layer
            else np.empty(0, np.float32)
        )
        scene.text_points = (
            np.asarray(text_points, dtype=np.float32)
            if text_points
            else np.empty((0, 3), dtype=np.float32)
        )
        scene.text_layer = np.asarray(text_layer, dtype=np.float32) if text_layer else np.empty(0, np.float32)
        return scene

    def _apply_lighting(
        self,
        scene: _CompiledScene,
        basis: Tuple[Point3D, Point3D, Point3D],
    ) -> None:
        """Refresh the per-face colour strings when the light or scene changed."""
        light = self._light
        enabled = self._shading_enabled and light.enabled
        key: Tuple[Any, ...] = (enabled,) + light.key()
        if enabled and light.follow_camera:
            forward = basis[2]
            key = key + (round(forward.x, 4), round(forward.y, 4), round(forward.z, 4))
        if scene.shade_key == key:
            return

        if not enabled or scene.face_total == 0:
            scene.fill_front = list(scene.base_front)
            scene.fill_back = list(scene.base_back)
            scene.shade_key = key
            return

        forward = basis[2]
        view_direction = np.array([-forward.x, -forward.y, -forward.z], dtype=np.float32)
        light_direction = light.world_direction(basis)

        normals = scene.face_normal
        front_levels = shade_levels(
            face_shade(normals, light, light_direction, view_direction)
        ).tolist()
        back_levels = shade_levels(
            face_shade(-normals, light, light_direction, view_direction)
        ).tolist()

        lit = scene.face_lit.tolist()
        scene.fill_front = [
            table[level] if (table is not None and is_lit) else base
            for base, table, level, is_lit in zip(
                scene.base_front, scene.table_front, front_levels, lit
            )
        ]
        scene.fill_back = [
            table[level] if (table is not None and is_lit) else base
            for base, table, level, is_lit in zip(
                scene.base_back, scene.table_back, back_levels, lit
            )
        ]
        scene.shade_key = key

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _project(
        points: np.ndarray,
        origin: np.ndarray,
        basis: np.ndarray,
        x_scale: float,
        y_scale: float,
        half_width: float,
        half_height: float,
        near: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Project world points; returns camera space, screen x/y and validity."""
        if len(points) == 0:
            empty = np.empty(0, dtype=np.float32)
            return np.empty((0, 3), np.float32), empty, empty, np.empty(0, bool)

        camera_space = (points - origin) @ basis
        depth = camera_space[:, 2]
        valid = depth > near
        safe_depth = np.where(valid, depth, 1.0)
        screen_x = (camera_space[:, 0] * x_scale / safe_depth + 1.0) * half_width
        screen_y = (1.0 - camera_space[:, 1] * y_scale / safe_depth) * half_height
        return camera_space, screen_x, screen_y, valid

    def set_interactive_detail(self, faces: int) -> None:
        """Set the starting face budget for interactive frames."""
        self._fast_polygon_target = max(200, min(200000, int(faces)))

    def _tune_interactive_detail(self) -> None:
        """
        Track the interval between interactive frames and adjust the budget.

        Redraws are scheduled with a fixed delay, so any interval beyond that
        delay is time Tk spent rasterising.  Shrinking the budget when frames
        run long keeps orbiting responsive on dense models; growing it back
        restores detail once there is headroom.
        """
        now = time.perf_counter()
        previous = self._last_interactive_frame
        self._last_interactive_frame = now
        if previous is None:
            return

        elapsed_ms = (now - previous) * 1000.0
        budget_ms = (
            self._animation_frame_budget_ms
            if self._is_playing_animation
            else float(self._interactive_delay_ms)
        )
        if elapsed_ms > 1.6 * budget_ms:
            self._fast_polygon_target = max(300, int(self._fast_polygon_target * 0.75))
        elif elapsed_ms < 1.15 * budget_ms:
            self._fast_polygon_target = min(200000, int(self._fast_polygon_target * 1.15) + 1)

    def redraw(self) -> None:
        """Render the scene; static world geometry is reused from cache."""
        if not self.winfo_exists() or not self.canvas.winfo_exists():
            return

        self.width, self.height = self._viewport_size()

        interactive = self._interactive_render
        if interactive:
            self._tune_interactive_detail()
        else:
            self._last_interactive_frame = None
        quality = "fast" if interactive else "full"
        scene = self._get_scene(quality)

        basis_vectors = self.camera.basis()
        right, camera_up, forward = basis_vectors
        self._apply_lighting(scene, basis_vectors)

        position = self.camera.position
        y_scale = 1.0 / math.tan(self.camera.fov / 2.0)
        plot_width = self._plot_width()
        x_scale = y_scale * self.height / plot_width
        half_width = 0.5 * plot_width
        half_height = 0.5 * self.height
        near = max(1.0e-9, float(self.camera.near))

        origin = np.array([position.x, position.y, position.z], dtype=np.float32)
        basis = np.array(
            [
                [right.x, camera_up.x, forward.x],
                [right.y, camera_up.y, forward.y],
                [right.z, camera_up.z, forward.z],
            ],
            dtype=np.float32,
        )

        # Hidden-member ray checks and stipple/legend drawing are restored on
        # mouse release.  Skipping them while dragging keeps orbiting responsive
        # on dense cylinder models without changing the final rendered view.
        if self._animation_occluders is not None:
            # A replayed frame is tested against the occluders it was captured
            # with, not against whatever `objects` holds right now.
            occluders = self._animation_occluders
        elif interactive:
            occluders = []
        else:
            occluders = self._collect_opaque_cylinder_occluders()

        order, front, coords, clipped = self._visible_faces(
            scene,
            origin,
            basis,
            x_scale,
            y_scale,
            half_width,
            half_height,
            near,
            plot_width,
            occluders,
            interactive,
        )
        try:
            self._draw_faces(scene, order, front, coords, clipped, interactive)
        except tk.TclError:
            # The only option Tk can reject here is a generated stipple bitmap
            # it failed to open.  Drop back to the built-in stipples and paint
            # the frame again rather than losing the whole view.
            stipple_module.disable_generated()
            self._invalidate_geometry_cache()
            scene = self._get_scene(quality)
            self._apply_lighting(scene, basis_vectors)
            order, front, coords, clipped = self._visible_faces(
                scene, origin, basis, x_scale, y_scale, half_width, half_height,
                near, plot_width, occluders, interactive,
            )
            self._draw_faces(scene, order, front, coords, clipped, interactive)
        self._draw_overlay_lines(
            scene, origin, basis, x_scale, y_scale, half_width, half_height, near
        )
        self._draw_markers(
            scene, origin, basis, x_scale, y_scale, half_width, half_height, near
        )
        self._draw_texts(
            scene, origin, basis, x_scale, y_scale, half_width, half_height, near, plot_width
        )

        if not self._is_capturing_animation:
            # The HUD is cached, so it no longer has to be dropped during
            # interaction to stay responsive: the legend now stays put while
            # orbiting and through animation playback.
            self._draw_hud()

    def _visible_faces(
        self,
        scene: _CompiledScene,
        origin: np.ndarray,
        basis: np.ndarray,
        x_scale: float,
        y_scale: float,
        half_width: float,
        half_height: float,
        near: float,
        plot_width: int,
        occluders: Sequence[Dict[str, Any]],
        interactive: bool,
    ) -> Tuple[List[int], List[bool], List[float], Dict[int, List[float]]]:
        """Cull, depth-sort and project every face that can show a pixel."""
        if self._section_plane is not None and self._section_plane.enabled:
            return self._visible_faces_with_section(
                scene,
                origin,
                basis,
                x_scale,
                y_scale,
                half_width,
                half_height,
                near,
                plot_width,
                interactive,
            )
        total = scene.face_total
        if total == 0:
            return [], [], [], {}

        camera_space, screen_x, screen_y, valid = self._project(
            scene.face_vertices, origin, basis, x_scale, y_scale, half_width, half_height, near
        )

        starts = scene.face_start
        counts = scene.face_count

        valid_float = valid.astype(np.float32)
        valid_count = np.add.reduceat(valid_float, starts)
        alive = valid_count > 0.0
        # Faces with some vertices behind the near plane get clipped below.  A
        # screen bounding box built from their surviving vertices understates
        # the clipped shape, so they skip the screen-space rejections.
        straddling = alive & (valid_count < counts.astype(np.float32))

        # Facing: positive when the outward normal points back at the camera.
        to_camera = origin - scene.face_center
        facing = np.einsum("ij,ij->i", scene.face_normal, to_camera)
        front_facing = facing > 0.0
        alive &= ~(scene.face_cull & ~front_facing)

        depth = camera_space[:, 2]
        depth_sum = np.add.reduceat(np.where(valid, depth, 0.0), starts)
        face_depth = depth_sum / np.maximum(valid_count, 1.0)

        big = np.float32(1.0e30)
        low_x = np.minimum.reduceat(np.where(valid, screen_x, big), starts)
        high_x = np.maximum.reduceat(np.where(valid, screen_x, -big), starts)
        low_y = np.minimum.reduceat(np.where(valid, screen_y, big), starts)
        high_y = np.maximum.reduceat(np.where(valid, screen_y, -big), starts)

        margin = 20.0
        on_screen = (
            (high_x >= -margin)
            & (low_x <= plot_width + margin)
            & (high_y >= -margin)
            & (low_y <= self.height + margin)
        )

        # Sub-pixel faces never show anything; edge-on slivers still do, so a
        # face only dies when it is thin in *both* directions.  Faces with no
        # valid vertex have an inverted bounding box; clamping keeps the
        # extent arithmetic finite for them.
        span_x = np.maximum(high_x - low_x, 0.0)
        span_y = np.maximum(high_y - low_y, 0.0)
        on_screen &= (span_x >= _MIN_SCREEN_EXTENT) | (span_y >= _MIN_SCREEN_EXTENT)
        alive &= on_screen | straddling
        # Level-of-detail metric: padded so long thin faces still rank above
        # genuinely tiny ones.
        extent = (span_x + 1.0) * (span_y + 1.0)

        if occluders:
            alive &= ~self._faces_hidden_by_occluders(scene, occluders, origin)

        index = np.nonzero(alive)[0]
        if len(index) == 0:
            return [], [], [], {}

        # Interactive level of detail: keep the largest faces on screen rather
        # than dropping every Nth face, which used to punch holes in surfaces.
        if interactive and len(index) > self._fast_polygon_target:
            keep = self._fast_polygon_target
            visible_extent = extent[index]
            threshold = np.partition(visible_extent, len(index) - keep)[len(index) - keep]
            index = index[visible_extent >= threshold]

        # Painter order: far to near, with the layer number acting only as a
        # near-coplanar tie-break so members never punch through a shell.
        layer_epsilon = max(1.0e-9, max(float(self.camera.distance), 1.0) * 1.0e-6)
        sort_key = -(face_depth[index] - scene.face_layer[index] * layer_epsilon)
        # two_sided_shell faces bracket the scene: their back half paints first
        # and their front half last, so internals stay inside the shell.
        phase = np.where(
            scene.face_phase[index] == 0,
            np.where(front_facing[index], 2, 0),
            1,
        )
        index = index[np.lexsort((sort_key, phase))]

        # Coordinates: one flat list of ints for the whole vertex array, so the
        # per-face handover is a list slice instead of numpy work.
        packed = np.empty((len(screen_x), 2), dtype=np.float32)
        packed[:, 0] = screen_x
        packed[:, 1] = screen_y
        np.nan_to_num(packed, copy=False, nan=0.0, posinf=_COORD_LIMIT, neginf=-_COORD_LIMIT)
        np.clip(packed, -_COORD_LIMIT, _COORD_LIMIT, out=packed)
        coords = np.rint(packed).astype(np.int32).ravel().tolist()

        # Faces straddling the near plane are clipped rather than dropped, so
        # geometry no longer vanishes when the camera moves inside a model.
        order = index.tolist()
        clipped: Dict[int, List[float]] = {}
        straddling_faces = set(np.nonzero(straddling)[0].tolist())
        if straddling_faces:
            for face in order:
                if face not in straddling_faces:
                    continue
                start = int(starts[face])
                stop = start + int(counts[face])
                polygon = self._clip_near_plane(camera_space[start:stop], near)
                if len(polygon) >= 2:
                    clipped[face] = self._project_camera_space(
                        polygon, x_scale, y_scale, half_width, half_height
                    )
            order = [
                face
                for face in order
                if face not in straddling_faces or face in clipped
            ]

        return order, front_facing.tolist(), coords, clipped

    def _visible_faces_with_section(
        self,
        scene: _CompiledScene,
        origin: np.ndarray,
        basis: np.ndarray,
        x_scale: float,
        y_scale: float,
        half_width: float,
        half_height: float,
        near: float,
        plot_width: int,
        interactive: bool,
    ) -> Tuple[List[int], List[bool], List[float], Dict[int, List[float]]]:
        """Project faces after world-space section and camera-near clipping.

        The normal renderer remains fully vectorised.  A section view takes
        this opt-in polygon path because intersection vertices have no slots in
        the compiled arrays.  The resulting cut polygons themselves determine
        occlusion, so removed portions of analytic cylinder occluders cannot
        hide retained geometry.
        """

        plane = self._section_plane
        if plane is None or not plane.enabled or scene.face_total == 0:
            return [], [False] * scene.face_total, [], {}
        front = [False] * scene.face_total
        entries: List[Tuple[int, float, float, int, List[float]]] = []
        margin = 20.0

        for face in range(scene.face_total):
            start = int(scene.face_start[face])
            stop = start + int(scene.face_count[face])
            world = scene.face_vertices[start:stop]
            if len(world) < 2:
                continue

            if bool(scene.face_is_edge[face]):
                segment = plane.clip_segment(world[0], world[-1])
                if segment is None:
                    continue
                retained_world = np.asarray(
                    [point.to_tuple() for point in segment], dtype=np.float32
                )
            else:
                polygon = plane.clip_polygon(world)
                if len(polygon) < 3:
                    continue
                retained_world = np.asarray(
                    [point.to_tuple() for point in polygon], dtype=np.float32
                )

            camera_polygon = (retained_world - origin) @ basis
            if bool(scene.face_is_edge[face]):
                camera_segment = self._clip_camera_segment(
                    camera_polygon[0], camera_polygon[-1], near
                )
                if camera_segment is None:
                    continue
                clipped_camera = list(camera_segment)
            else:
                valid = camera_polygon[:, 2] >= near
                if bool(np.all(valid)):
                    clipped_camera = [
                        tuple(float(value) for value in row) for row in camera_polygon
                    ]
                elif bool(np.any(valid)):
                    clipped_camera = self._clip_near_plane(camera_polygon, near)
                else:
                    continue
                if len(clipped_camera) < 3:
                    continue

            facing = float(np.dot(scene.face_normal[face], origin - scene.face_center[face]))
            is_front = facing > 0.0
            front[face] = is_front
            if bool(scene.face_cull[face]) and not is_front:
                continue

            coordinates = self._project_camera_space(
                clipped_camera, x_scale, y_scale, half_width, half_height
            )
            if len(coordinates) < 4:
                continue
            xs = coordinates[0::2]
            ys = coordinates[1::2]
            low_x, high_x = min(xs), max(xs)
            low_y, high_y = min(ys), max(ys)
            if (
                high_x < -margin
                or low_x > plot_width + margin
                or high_y < -margin
                or low_y > self.height + margin
            ):
                continue
            span_x = max(0.0, float(high_x - low_x))
            span_y = max(0.0, float(high_y - low_y))
            if span_x < _MIN_SCREEN_EXTENT and span_y < _MIN_SCREEN_EXTENT:
                continue

            depth = sum(point[2] for point in clipped_camera) / len(clipped_camera)
            extent = (span_x + 1.0) * (span_y + 1.0)
            phase = (
                (2 if is_front else 0)
                if int(scene.face_phase[face]) == 0
                else 1
            )
            entries.append((face, depth, extent, phase, coordinates))

        if interactive and len(entries) > self._fast_polygon_target:
            entries.sort(key=lambda entry: entry[2], reverse=True)
            entries = entries[: self._fast_polygon_target]

        layer_epsilon = max(1.0e-9, max(float(self.camera.distance), 1.0) * 1.0e-6)
        entries.sort(
            key=lambda entry: (
                entry[3],
                -(entry[1] - float(scene.face_layer[entry[0]]) * layer_epsilon),
            )
        )
        order = [entry[0] for entry in entries]
        clipped = {entry[0]: entry[4] for entry in entries}
        return order, front, [], clipped

    @staticmethod
    def _clip_camera_segment(
        first: Sequence[float], second: Sequence[float], near: float
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        a = tuple(float(value) for value in first)
        b = tuple(float(value) for value in second)
        a_inside = a[2] >= near
        b_inside = b[2] >= near
        if a_inside and b_inside:
            return a, b
        if not a_inside and not b_inside:
            return None
        span = b[2] - a[2]
        if abs(span) <= _EPS:
            return None
        parameter = (near - a[2]) / span
        intersection = (
            a[0] + parameter * (b[0] - a[0]),
            a[1] + parameter * (b[1] - a[1]),
            near,
        )
        return (a, intersection) if a_inside else (intersection, b)

    @staticmethod
    def _clip_near_plane(camera_space: np.ndarray, near: float) -> List[Tuple[float, float, float]]:
        """Sutherland-Hodgman clip of a camera-space polygon against z >= near."""
        polygon = [tuple(float(value) for value in row) for row in camera_space]
        result: List[Tuple[float, float, float]] = []
        count = len(polygon)
        for index in range(count):
            current = polygon[index]
            following = polygon[(index + 1) % count]
            current_in = current[2] >= near
            following_in = following[2] >= near
            if current_in:
                result.append(current)
            if current_in != following_in:
                span = following[2] - current[2]
                if abs(span) <= _EPS:
                    continue
                t = (near - current[2]) / span
                result.append(
                    (
                        current[0] + t * (following[0] - current[0]),
                        current[1] + t * (following[1] - current[1]),
                        near,
                    )
                )
        return result

    @staticmethod
    def _project_camera_space(
        polygon: Sequence[Tuple[float, float, float]],
        x_scale: float,
        y_scale: float,
        half_width: float,
        half_height: float,
    ) -> List[float]:
        coords: List[float] = []
        for x, y, depth in polygon:
            depth = max(depth, 1.0e-9)
            screen_x = (x * x_scale / depth + 1.0) * half_width
            screen_y = (1.0 - y * y_scale / depth) * half_height
            coords.append(round(max(-_COORD_LIMIT, min(_COORD_LIMIT, screen_x))))
            coords.append(round(max(-_COORD_LIMIT, min(_COORD_LIMIT, screen_y))))
        return coords

    def _draw_faces(
        self,
        scene: _CompiledScene,
        order: Sequence[int],
        front: Sequence[bool],
        coords: Sequence[float],
        clipped: Dict[int, List[float]],
        interactive: bool,
    ) -> None:
        needed = len(order)
        self._ensure_pool(
            self._polygon_pool,
            self._polygon_state,
            needed,
            lambda: self.canvas.create_polygon(0, 0, 0, 0, 0, 0, state="hidden", tags=_TAG_POLYGON),
            restack=True,
        )

        call = self.canvas.tk.call
        widget = self.canvas._w
        pool = self._polygon_pool
        states = self._polygon_state
        starts = scene.face_start
        counts = scene.face_count
        fill_front = scene.fill_front
        fill_back = scene.fill_back
        outlines = scene.outline
        widths = scene.face_width
        stipple_front = scene.stipple_front
        stipple_back = scene.stipple_back
        tags = scene.tags
        any_tags = scene.any_tags
        show_mesh_lines = self.show_mesh_lines
        fast_no_outline = scene.fast_no_outline
        opaque = scene.opaque
        is_edge = scene.face_is_edge
        application_selected = scene.face_application_selected

        # Resolved once per scene and highlight generation, so orbiting a
        # highlighted model does not re-split every tag string every frame.
        highlighted = self._pick.highlighted_faces(scene)
        preselected = self._pick.preselected_faces(scene)
        highlight_fill = self._pick.highlight_fill
        highlight_outline = self._pick.highlight_outline
        application_selection_fills: Dict[str, str] = {}

        for slot, face in enumerate(order):
            item = pool[slot]
            face_coords = clipped.get(face)
            if face_coords is None:
                start = 2 * int(starts[face])
                face_coords = coords[start:start + 2 * int(counts[face])]
            call((widget, "coords", item) + tuple(face_coords))

            is_front = front[face]
            fill = fill_front[face] if is_front else fill_back[face]
            stipple = stipple_front[face] if is_front else stipple_back[face]
            if interactive:
                stipple = ""

            if is_edge[face]:
                # A 3D line is drawn as an unfilled outline; it has no mesh.
                outline = outlines[face]
            elif interactive and fast_no_outline[face]:
                # Stroking every face costs about as much as filling it, so
                # interactive frames skip outlines entirely.
                outline = ""
            elif not show_mesh_lines:
                # Matching the fill removes the hairline seams between adjacent
                # facets; transparent faces keep an open outline so the stipple
                # still reads as see-through.
                outline = fill if opaque[face] else ""
            else:
                outline = outlines[face]

            if application_selected[face] and not is_edge[face]:
                selected_fill = application_selection_fills.get(fill)
                if selected_fill is None:
                    selected_fill = _mix_color(fill, highlight_fill, 0.65)
                    application_selection_fills[fill] = selected_fill
                fill = selected_fill

            if highlighted is not None and face in highlighted:
                if not is_edge[face]:
                    fill = highlight_fill
                # Always stroke a highlighted face, even when mesh lines are
                # off, so the selection reads against its neighbours.
                outline = highlight_outline
            elif preselected is not None and face in preselected:
                if not is_edge[face]:
                    fill = self._pick.preselection_fill
                outline = self._pick.preselection_outline

            state = (fill, outline, widths[face], stipple, tags[face])
            if states[slot] != state:
                options = (
                    "-fill", fill,
                    "-outline", outline,
                    "-width", widths[face],
                    "-stipple", stipple,
                    "-state", "normal",
                )
                if any_tags:
                    options += ("-tags", (_TAG_POLYGON + " " + tags[face]).strip())
                # Record only after Tk accepts it, so a rejected option cannot
                # leave the cache claiming a configuration that never applied.
                call((widget, "itemconfigure", item) + options)
                states[slot] = state

        self._hide_unused(self._polygon_pool, self._polygon_state, needed)

    def _draw_overlay_lines(
        self,
        scene: _CompiledScene,
        origin: np.ndarray,
        basis: np.ndarray,
        x_scale: float,
        y_scale: float,
        half_width: float,
        half_height: float,
        near: float,
    ) -> None:
        total = len(scene.line_color)
        if total == 0:
            self._hide_unused(self._line_pool, self._line_state, 0)
            return

        plane = self._section_plane
        if plane is not None and plane.enabled:
            source_lines: List[int] = []
            section_x0: List[float] = []
            section_y0: List[float] = []
            section_x1: List[float] = []
            section_y1: List[float] = []
            section_depth: List[float] = []
            for line in range(total):
                start = 2 * line
                segment = plane.clip_segment(
                    scene.line_vertices[start], scene.line_vertices[start + 1]
                )
                if segment is None:
                    continue
                world = np.asarray(
                    [point.to_tuple() for point in segment], dtype=np.float32
                )
                camera_segment = (world - origin) @ basis
                clipped_segment = self._clip_camera_segment(
                    camera_segment[0], camera_segment[1], near
                )
                if clipped_segment is None:
                    continue
                coordinates = self._project_camera_space(
                    clipped_segment, x_scale, y_scale, half_width, half_height
                )
                source_lines.append(line)
                section_x0.append(float(coordinates[0]))
                section_y0.append(float(coordinates[1]))
                section_x1.append(float(coordinates[2]))
                section_y1.append(float(coordinates[3]))
                section_depth.append(
                    0.5 * (clipped_segment[0][2] + clipped_segment[1][2])
                )
            if not source_lines:
                self._hide_unused(self._line_pool, self._line_state, 0)
                return
            x0 = np.asarray(section_x0)
            y0 = np.asarray(section_y0)
            x1 = np.asarray(section_x1)
            y1 = np.asarray(section_y1)
            depth = np.asarray(section_depth)
            source = np.asarray(source_lines, dtype=np.int64)
            index = np.argsort(-depth, kind="stable")
        else:
            _camera_space, screen_x, screen_y, valid = self._project(
                scene.line_vertices, origin, basis, x_scale, y_scale, half_width, half_height, near
            )
            pair_valid = valid[0::2] & valid[1::2]
            index = np.nonzero(pair_valid)[0]
            if len(index) == 0:
                self._hide_unused(self._line_pool, self._line_state, 0)
                return
            depth = (
                (scene.line_vertices[0::2] + scene.line_vertices[1::2]) * 0.5 - origin
            ) @ basis[:, 2]
            index = index[np.argsort(-depth[index], kind="stable")]
            source = np.arange(total, dtype=np.int64)
            x0 = np.clip(np.nan_to_num(screen_x[0::2]), -_COORD_LIMIT, _COORD_LIMIT)
            y0 = np.clip(np.nan_to_num(screen_y[0::2]), -_COORD_LIMIT, _COORD_LIMIT)
            x1 = np.clip(np.nan_to_num(screen_x[1::2]), -_COORD_LIMIT, _COORD_LIMIT)
            y1 = np.clip(np.nan_to_num(screen_y[1::2]), -_COORD_LIMIT, _COORD_LIMIT)

        needed = len(index)
        self._ensure_pool(
            self._line_pool,
            self._line_state,
            needed,
            lambda: self.canvas.create_line(0, 0, 0, 0, state="hidden", tags=_TAG_LINE),
        )

        call = self.canvas.tk.call
        widget = self.canvas._w
        pool = self._line_pool
        states = self._line_state
        colors = scene.line_color
        widths = scene.line_width
        line_tags = scene.line_tags
        line_bindings = scene.line_bindings
        any_line_tags = any(line_tags)
        selected = self._pick.highlight_tags
        preselected = self._pick.preselection_key

        for slot, projected_line in enumerate(index.tolist()):
            line = int(source[projected_line])
            item = pool[slot]
            call((widget, "coords", item, round(float(x0[projected_line])), round(float(y0[projected_line])),
                  round(float(x1[projected_line])), round(float(y1[projected_line]))))
            tag = line_tags[line] if any_line_tags else ""
            binding = scene.line_binding(line)
            keys = () if binding is None else tuple(owner.key for owner in binding.owners)
            is_selected = bool(selected.intersection(keys)) or tag in selected
            is_preselected = bool(
                preselected
                and not is_selected
                and (preselected in keys or preselected in tag.split())
            )
            color = (
                self._pick.highlight_outline
                if is_selected
                else self._pick.preselection_outline
                if is_preselected
                else colors[line]
            )
            width = widths[line] + (2 if (is_selected or is_preselected) else 0)
            state = (color, width, tag)
            if states[slot] != state:
                options = (
                    "-fill", color,
                    "-width", width,
                    "-state", "normal",
                )
                if any_line_tags:
                    options += ("-tags", (_TAG_LINE + " " + tag).strip())
                call((widget, "itemconfigure", item) + options)
                states[slot] = state

        self._hide_unused(self._line_pool, self._line_state, needed)

    def _draw_markers(
        self,
        scene: _CompiledScene,
        origin: np.ndarray,
        basis: np.ndarray,
        x_scale: float,
        y_scale: float,
        half_width: float,
        half_height: float,
        near: float,
    ) -> None:
        total = len(scene.marker_points)
        if total == 0:
            self._hide_unused(self._marker_pool, self._marker_state, 0)
            return
        camera_space, screen_x, screen_y, valid = self._project(
            scene.marker_points,
            origin,
            basis,
            x_scale,
            y_scale,
            half_width,
            half_height,
            near,
        )
        margin = 12.0
        visible = (
            valid
            & (screen_x >= -margin)
            & (screen_x <= self._plot_width() + margin)
            & (screen_y >= -margin)
            & (screen_y <= self.height + margin)
        )
        plane = self._section_plane
        if plane is not None and plane.enabled:
            normal = np.asarray(plane.normal.to_tuple(), dtype=np.float32)
            visible &= scene.marker_points @ normal >= float(plane.offset)
        index = np.nonzero(visible)[0]
        if len(index) == 0:
            self._hide_unused(self._marker_pool, self._marker_state, 0)
            return
        index = index[np.argsort(-camera_space[index, 2], kind="stable")]
        self._ensure_pool(
            self._marker_pool,
            self._marker_state,
            len(index),
            lambda: self.canvas.create_oval(
                0, 0, 0, 0, state="hidden", tags=_TAG_MARKER
            ),
        )
        selected = self._pick.highlight_tags
        preselected = self._pick.preselection_key
        for slot, marker in enumerate(index.tolist()):
            binding = scene.marker_binding(marker)
            keys = () if binding is None else tuple(owner.key for owner in binding.owners)
            is_selected = bool(selected.intersection(keys))
            is_preselected = bool(preselected and preselected in keys and not is_selected)
            size = scene.marker_sizes[marker] + (2 if is_preselected else 0)
            radius = 0.5 * size
            x = float(screen_x[marker])
            y = float(screen_y[marker])
            self.canvas.coords(
                self._marker_pool[slot], x - radius, y - radius, x + radius, y + radius
            )
            fill = (
                self._pick.highlight_fill
                if is_selected
                else self._pick.preselection_fill
                if is_preselected
                else scene.marker_colors[marker]
            )
            outline = (
                self._pick.highlight_outline
                if is_selected
                else self._pick.preselection_outline
                if is_preselected
                else scene.marker_outlines[marker]
            )
            tag = scene.marker_tags[marker]
            state = (fill, outline, size, tag)
            if self._marker_state[slot] != state:
                self.canvas.itemconfigure(
                    self._marker_pool[slot],
                    fill=fill,
                    outline=outline,
                    width=2 if (is_selected or is_preselected) else 1,
                    state="normal",
                    tags=(_TAG_MARKER + " " + tag).strip(),
                )
                self._marker_state[slot] = state
        self._hide_unused(self._marker_pool, self._marker_state, len(index))

    def _draw_texts(
        self,
        scene: _CompiledScene,
        origin: np.ndarray,
        basis: np.ndarray,
        x_scale: float,
        y_scale: float,
        half_width: float,
        half_height: float,
        near: float,
        plot_width: int,
    ) -> None:
        total = len(scene.text_content)
        if total == 0:
            self._hide_unused(self._text_pool, self._text_state, 0)
            return

        camera_space, screen_x, screen_y, valid = self._project(
            scene.text_points, origin, basis, x_scale, y_scale, half_width, half_height, near
        )
        margin = 20.0
        visible = (
            valid
            & (screen_x >= -margin)
            & (screen_x <= plot_width + margin)
            & (screen_y >= -margin)
            & (screen_y <= self.height + margin)
        )
        plane = self._section_plane
        if plane is not None and plane.enabled:
            normal = np.asarray(plane.normal.to_tuple(), dtype=np.float32)
            visible &= scene.text_points @ normal >= float(plane.offset)
        index = np.nonzero(visible)[0]
        if len(index) == 0:
            self._hide_unused(self._text_pool, self._text_state, 0)
            return
        index = index[np.argsort(-camera_space[index, 2], kind="stable")]

        needed = len(index)
        self._ensure_pool(
            self._text_pool,
            self._text_state,
            needed,
            lambda: self.canvas.create_text(0, 0, text="", state="hidden", tags=_TAG_TEXT),
        )

        call = self.canvas.tk.call
        widget = self.canvas._w
        pool = self._text_pool
        states = self._text_state
        content = scene.text_content

        for slot, text_index in enumerate(index.tolist()):
            item = pool[slot]
            call(
                (
                    widget, "coords", item,
                    round(float(screen_x[text_index])),
                    round(float(screen_y[text_index])),
                )
            )
            state = content[text_index]
            if states[slot] != state:
                label, color, font, anchor = state
                call(
                    (
                        widget, "itemconfigure", item,
                        "-text", label,
                        "-fill", color,
                        "-font", font,
                        "-anchor", anchor,
                        "-state", "normal",
                    )
                )
                states[slot] = state

        self._hide_unused(self._text_pool, self._text_state, needed)

    def _ensure_pool(
        self,
        pool: List[int],
        states: List[Optional[Tuple[Any, ...]]],
        needed: int,
        factory: Any,
        restack: bool = False,
    ) -> None:
        missing = needed - len(pool)
        if missing <= 0:
            return
        for _ in range(missing):
            pool.append(factory())
            states.append(None)
        if restack:
            # New face items are created on top; lines, texts and the HUD must
            # stay above them.
            for tag in (_TAG_LINE, _TAG_MARKER, _TAG_TEXT, _TAG_HUD, _TAG_SELECTION):
                try:
                    self.canvas.tag_raise(tag)
                except tk.TclError:
                    pass

    def _hide_unused(
        self,
        pool: List[int],
        states: List[Optional[Tuple[Any, ...]]],
        used: int,
    ) -> None:
        call = self.canvas.tk.call
        widget = self.canvas._w
        for slot in range(used, len(pool)):
            if states[slot] is None:
                continue
            states[slot] = None
            call((widget, "itemconfigure", pool[slot], "-state", "hidden"))

    # ------------------------------------------------------------------
    # Primitive construction
    # ------------------------------------------------------------------

    @staticmethod
    def _polygon_normal(vertices: Sequence[Point3D]) -> Point3D:
        """
        Newell's area-weighted normal.

        It uses every edge rather than one vertex triple, so it stays correct
        for concave and slightly non-planar faces, and it costs the same as
        the single cross product it replaced.  The arithmetic is inlined on
        raw floats because this runs once per face and FE result meshes bring
        tens of thousands of them.
        """
        count = len(vertices)
        if count < 3:
            return Point3D(0.0, 0.0, 0.0)

        normal_x = normal_y = normal_z = 0.0
        previous = vertices[-1]
        previous_x, previous_y, previous_z = previous.x, previous.y, previous.z
        for vertex in vertices:
            x, y, z = vertex.x, vertex.y, vertex.z
            normal_x += (previous_y - y) * (previous_z + z)
            normal_y += (previous_z - z) * (previous_x + x)
            normal_z += (previous_x - x) * (previous_y + y)
            previous_x, previous_y, previous_z = x, y, z

        length = math.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)
        if length <= _EPS:
            return Point3D(0.0, 0.0, 0.0)
        return Point3D(normal_x / length, normal_y / length, normal_z / length)

    def _polygon_primitive(
        self,
        vertices: Sequence[Point3D],
        color: str,
        outline: str,
        width: int = 1,
        layer: int = 0,
        cull_backface: bool = False,
        fast_no_outline: bool = True,
        stipple: str = "",
        tags: str = "",
        back_color: str = "",
        two_sided_shell: bool = False,
        lit: bool = True,
        stipple_phase: int = 0,
        opacity: Optional[float] = None,
        binding: Optional[PickBinding] = None,
    ) -> Optional[Dict[str, Any]]:
        vertices_list = vertices if type(vertices) is list else list(vertices)
        count = len(vertices_list)
        if count < 3:
            return None

        # One pass of scalar arithmetic instead of three generator sums: this
        # is the hot loop when a scene is one polygon per finite element.
        total_x = total_y = total_z = 0.0
        for vertex in vertices_list:
            total_x += vertex.x
            total_y += vertex.y
            total_z += vertex.z
        center = Point3D(total_x / count, total_y / count, total_z / count)
        return {
            "kind": "polygon",
            "vertices": vertices_list,
            "color": color,
            "back_color": back_color,
            "outline": outline,
            "width": width,
            "layer": layer,
            "center": center,
            "normal": self._polygon_normal(vertices_list),
            "cull_backface": cull_backface,
            "fast_no_outline": fast_no_outline,
            "stipple": stipple,
            "stipple_phase": stipple_phase,
            "opacity": opacity,
            "tags": tags,
            "two_sided_shell": bool(two_sided_shell),
            "lit": bool(lit),
            "binding": binding,
        }

    @staticmethod
    def _line_primitive(
        start: Point3D,
        end: Point3D,
        color: str,
        width: int,
        layer: int = 30,
        draw_overlay: bool = False,
        tags: str = "",
        binding: Optional[PickBinding] = None,
    ) -> Dict[str, Any]:
        return {
            "kind": "line",
            "start": start,
            "end": end,
            "color": color,
            "width": width,
            "layer": layer,
            "draw_overlay": bool(draw_overlay),
            "tags": tags,
            "binding": binding,
        }

    def _object_to_primitives(
        self,
        obj: Dict[str, Any],
        quality: str,
        object_index: int = 0,
    ) -> List[Dict[str, Any]]:
        object_type = obj.get("type")
        if object_type == "line":
            return [
                self._line_primitive(
                    obj["start"],
                    obj["end"],
                    obj.get("color", "black"),
                    obj.get("width", 1),
                    layer=int(obj.get("layer", 30)),
                    draw_overlay=bool(obj.get("draw_overlay", False)),
                    tags=obj.get("tags", ""),
                    binding=obj.get("binding"),
                )
            ]
        if object_type == "text":
            return [
                {
                    "kind": "text",
                    "point": obj["point"],
                    "text": obj.get("text", ""),
                    "color": obj.get("color", "black"),
                    "font": obj.get("font", ("Segoe UI", 9, "bold")),
                    "anchor": obj.get("anchor", tk.CENTER),
                    "layer": int(obj.get("layer", 35)),
                    "draw_overlay": bool(obj.get("draw_overlay", True)),
                }
            ]
        if object_type == "polygon":
            primitive = self._polygon_primitive(
                vertices=obj["vertices"],
                color=obj["color"],
                outline=obj["outline"],
                width=obj["width"],
                layer=obj.get("layer", 5),
                cull_backface=obj.get("cull_backface", False),
                stipple=obj.get("stipple", ""),
                tags=obj.get("tags", ""),
                back_color=obj.get("back_color", ""),
                two_sided_shell=bool(obj.get("two_sided_shell", False)),
                lit=bool(obj.get("lit", True)),
                stipple_phase=object_index,
                opacity=obj.get("opacity"),
                binding=obj.get("binding"),
            )
            return [primitive] if primitive else []
        if object_type == "faces":
            return [dict(obj, kind="faces", stipple_phase=object_index)]
        if object_type == "markers":
            return [dict(obj, kind="markers")]
        if object_type == "mesh":
            return self._mesh_primitives(obj, object_index)
        if object_type == "mesh_arrays":
            return self._retained_mesh_primitives(obj)
        if object_type == "cylinder":
            return self._cylinder_primitives(obj, quality)
        if object_type == "stiffener":
            if obj.get("stiffener_type") == "ring":
                return self._ring_stiffener_primitives(obj, quality)
            return self._longitudinal_stiffener_primitives(obj, quality)
        return []

    @staticmethod
    def _retained_colors(
        mesh: MeshArrays,
        face_indices: np.ndarray,
        obj: Dict[str, Any],
    ) -> List[str]:
        """Resolve retained result values without rebuilding mesh topology."""

        explicit = obj.get("face_colors")
        if explicit is not None:
            values = list(explicit)
            if len(values) == mesh.triangle_count:
                return [str(values[int(index)]) for index in face_indices]
            if len(values) == mesh.element_count:
                mapping = (
                    mesh.triangle_to_element
                    if mesh.triangle_to_element is not None
                    else np.arange(mesh.triangle_count, dtype=np.uint32)
                )
                return [str(values[int(mapping[index])]) for index in face_indices]
            raise ValueError("face_colors must match triangle or element count")

        scalar_values: Optional[np.ndarray]
        if mesh.element_scalars is not None:
            mapping = (
                mesh.triangle_to_element
                if mesh.triangle_to_element is not None
                else np.arange(mesh.triangle_count, dtype=np.uint32)
            )
            scalar_values = mesh.element_scalars[mapping[face_indices]]
        elif mesh.node_scalars is not None:
            scalar_values = np.mean(mesh.node_scalars[mesh.triangles[face_indices]], axis=1)
        else:
            return [str(obj.get("color", "#9aa7b4"))] * len(face_indices)

        finite = np.isfinite(scalar_values)
        configured = obj.get("scalar_range")
        if configured is not None:
            minimum, maximum = float(configured[0]), float(configured[1])
        elif np.any(finite):
            minimum = float(np.min(scalar_values[finite]))
            maximum = float(np.max(scalar_values[finite]))
        else:
            minimum, maximum = 0.0, 1.0
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError("scalar_range must contain finite values")
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        palette = [
            _interpolate_thickness_color(
                minimum + (maximum - minimum) * index / 255.0,
                minimum,
                maximum,
            )
            for index in range(256)
        ]
        span = max(_EPS, maximum - minimum)
        levels = np.clip(
            np.rint((scalar_values - minimum) * 255.0 / span), 0, 255
        ).astype(np.uint8)
        invalid = str(obj.get("invalid_color", "#808080"))
        return [palette[int(level)] if valid else invalid for level, valid in zip(levels, finite)]

    def _retained_mesh_primitives(self, obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        handle = obj["handle"]
        if handle.removed or not handle.visible:
            return []
        chunk_records = getattr(handle, "chunk_records", None)
        if chunk_records is None:  # compatibility with the 0.4 retained core
            chunk_records = tuple(
                (chunk_id, mesh, None, None)
                for chunk_id, mesh in handle.chunks
            )
        sources = [
            (
                None,
                handle.mesh,
                obj.get("owners"),
                obj.get("owner_resolver"),
            ),
            *chunk_records,
        ]
        result: List[Dict[str, Any]] = []
        geometry_cache = obj.setdefault("_array_geometry_cache", {})
        visibility_state = getattr(self, "_visibility_state", VisibilityState())

        for chunk_id, mesh, owner_table, owner_resolver in sources:
            mapping = (
                mesh.triangle_to_element
                if mesh.triangle_to_element is not None
                else np.arange(mesh.triangle_count, dtype=np.uint32)
            )
            if mesh.active_elements is None:
                face_indices = np.arange(mesh.triangle_count, dtype=np.uint32)
            else:
                face_indices = np.flatnonzero(mesh.active_elements[mapping]).astype(np.uint32)
            if owner_table is not None and not visibility_state.is_default:
                face_indices = np.asarray(
                    [
                        index for index in face_indices
                        if visibility_state.accepts(
                            owner_table.owners_for("triangle", int(index), owner_resolver)
                        )
                    ],
                    dtype=np.uint32,
                )

            cache_key = (
                chunk_id,
                id(mesh.positions),
                id(mesh.triangles),
                id(mesh.displacements),
                id(mesh.active_elements),
                handle.deformation_scale,
                handle.generations.transform,
                visibility_state,
            )
            cached = geometry_cache.get(cache_key)
            if cached is None:
                positions = mesh.positions
                if mesh.displacements is not None and handle.deformation_scale:
                    positions = positions + handle.deformation_scale * mesh.displacements
                transform = handle.transform
                positions = (
                    positions @ transform[:3, :3].T + transform[:3, 3]
                ).astype(np.float32, copy=False)
                triangles = mesh.triangles[face_indices]
                triangle_points = positions[triangles]
                vertices = np.ascontiguousarray(triangle_points.reshape((-1, 3)))
                counts = np.full(len(triangles), 3, dtype=np.int64)
                centers = np.mean(triangle_points, axis=1, dtype=np.float64).astype(np.float32)
                normals = np.cross(
                    triangle_points[:, 1] - triangle_points[:, 0],
                    triangle_points[:, 2] - triangle_points[:, 0],
                )
                lengths = np.linalg.norm(normals, axis=1)
                normals /= np.maximum(lengths, _EPS)[:, None]
                normals[lengths <= _EPS] = 0.0
                cached = (positions, vertices, counts, centers, normals.astype(np.float32))
                geometry_cache[cache_key] = cached
            positions, vertices, counts, centers, normals = cached

            if len(face_indices):
                colors = self._retained_colors(mesh, face_indices, obj)
                back = obj.get("back_color")
                application_selected = np.zeros(len(face_indices), dtype=bool)
                if chunk_id is None and len(handle.selected_elements):
                    application_selected = np.isin(
                        mapping[face_indices],
                        handle.selected_elements,
                        assume_unique=False,
                    )
                result.append(
                    {
                        "kind": "faces",
                        "vertices": vertices,
                        "counts": counts,
                        "centers": centers,
                        "normals": normals,
                        "colors": colors,
                        "back_colors": [str(back)] * len(colors) if back else None,
                        "outline": obj.get("outline", ""),
                        "width": obj.get("width", 1),
                        "layer": int(obj.get("layer", 5)),
                        "cull_backface": bool(obj.get("cull_backface", True)),
                        "opacity": float(obj.get("opacity", 1.0)),
                        "stipple": obj.get("stipple", ""),
                        "tags": obj.get("tags", ""),
                        "lit": bool(obj.get("lit", True)),
                        "two_sided_shell": bool(obj.get("two_sided_shell", False)),
                        "application_selected": application_selected,
                        "owner_table": owner_table,
                        "owner_primitives": face_indices,
                        "owner_resolver": owner_resolver,
                    }
                )

            if mesh.lines is not None and len(mesh.lines):
                line_indices = np.arange(len(mesh.lines), dtype=np.uint32)
                if owner_table is not None and not visibility_state.is_default:
                    line_indices = np.asarray(
                        [
                            index for index in line_indices
                            if visibility_state.accepts(
                                owner_table.owners_for("line", int(index), owner_resolver)
                            )
                        ],
                        dtype=np.uint32,
                    )
                line_vertices = np.ascontiguousarray(positions[mesh.lines[line_indices]].reshape((-1, 3)))
                result.append(
                    {
                        "kind": "lines",
                        "vertices": line_vertices,
                        "total": len(line_indices),
                        "color": str(obj.get("line_color") or obj.get("outline") or "black"),
                        "width": int(obj.get("line_width") or obj.get("width", 1)),
                        "layer": int(obj.get("line_layer", 30)),
                        "tags": obj.get("tags", ""),
                        "owner_table": owner_table,
                        "owner_primitives": line_indices,
                        "owner_resolver": owner_resolver,
                    }
                )
            if mesh.point_indices is not None and len(mesh.point_indices):
                point_primitives = np.arange(len(mesh.point_indices), dtype=np.uint32)
                if owner_table is not None and not visibility_state.is_default:
                    point_primitives = np.asarray(
                        [
                            index for index in point_primitives
                            if visibility_state.accepts(
                                owner_table.owners_for("point", int(index), owner_resolver)
                            )
                        ],
                        dtype=np.uint32,
                    )
                total = len(point_primitives)
                result.append(
                    {
                        "kind": "markers",
                        "points": positions[mesh.point_indices[point_primitives]],
                        "colors": [str(obj.get("point_color", "#2563eb"))] * total,
                        "outlines": [str(obj.get("point_outline", ""))] * total,
                        "sizes": [int(obj.get("point_size", 6))] * total,
                        "layer": int(obj.get("point_layer", 32)),
                        "tags": obj.get("tags", ""),
                        "owner_table": owner_table,
                        "owner_primitives": point_primitives,
                        "owner_resolver": owner_resolver,
                    }
                )
        return result

    def _mesh_primitives(
        self,
        obj: Dict[str, Any],
        object_index: int = 0,
    ) -> List[Dict[str, Any]]:
        points: Sequence[Point3D] = obj["points"]
        faces: Sequence[Sequence[int]] = obj["faces"]
        color = obj.get("color", "gray")
        face_colors = obj.get("face_colors")
        outline = obj.get("outline", "")
        width = obj.get("width", 1)
        layer = int(obj.get("layer", 5))
        cull = bool(obj.get("cull_backface", False))
        stipple = obj.get("stipple", "")
        opacity = obj.get("opacity")
        back_color = obj.get("back_color", "")
        tags = obj.get("tags", "")
        bindings = obj.get("bindings")
        lit = bool(obj.get("lit", True))
        two_sided = bool(obj.get("two_sided_shell", False))

        primitives: List[Dict[str, Any]] = []
        for face_index, face in enumerate(faces):
            if face_colors is not None:
                face_color = face_colors[face_index % len(face_colors)]
            else:
                face_color = color
            primitive = self._polygon_primitive(
                [points[index] for index in face],
                face_color,
                outline,
                width=width,
                layer=layer,
                cull_backface=cull,
                stipple=stipple,
                tags=tags,
                back_color=back_color,
                two_sided_shell=two_sided,
                lit=lit,
                stipple_phase=object_index,
                opacity=opacity,
                binding=(bindings[face_index] if bindings is not None else None),
            )
            if primitive:
                primitives.append(primitive)
        return primitives

    def _adaptive_z_breaks(
        self,
        z_bottom: float,
        z_top: float,
        requested_segments: int,
        quality: str,
    ) -> List[float]:
        """Build local vertical patches around ring girder elevations."""
        requested_segments = max(1, int(requested_segments))
        if quality == "fast":
            uniform_segments = max(3, min(5, round(requested_segments / 7)))
        else:
            uniform_segments = max(5, min(12, round(requested_segments / 4)))

        values = {
            z_bottom + (z_top - z_bottom) * index / uniform_segments
            for index in range(uniform_segments + 1)
        }

        for obj in self.objects:
            if obj.get("type") != "stiffener" or obj.get("stiffener_type") != "ring":
                continue
            z_position = float(obj.get("z_position", 0.0))
            half_width = 0.5 * max(
                float(obj.get("web_thickness", 0.0)),
                float(obj.get("flange_width", 0.0)),
            )
            if quality == "fast":
                candidates = (z_position,)
            else:
                candidates = (z_position - half_width, z_position, z_position + half_width)
            for value in candidates:
                if z_bottom + _EPS < value < z_top - _EPS:
                    values.add(value)

        return sorted(values)

    @staticmethod
    def _scaled_segments(segments: int, quality: str, minimum: int = 12) -> int:
        segments = max(3, int(segments))
        if quality == "fast":
            return max(minimum, segments // 2)
        return segments

    @staticmethod
    def _opacity_to_stipple(opacity: float) -> str:
        """Map a requested opacity to a stipple pattern."""
        return stipple_module.for_opacity(opacity)

    @staticmethod
    def _resolve_plate_thickness(
        specification: Any,
        angle: float,
        z_coord: float,
        z_bottom: float,
        z_top: float,
    ) -> Optional[float]:
        """
        Resolve a shell patch thickness.

        Supported specifications:
        * scalar: one thickness for the whole shell;
        * 1D sequence: axial bands ordered bottom to top;
        * 2D sequence: axial rows, with circumferential columns;
        * callable: called as ``fn(angle, z)``.  If that signature is not
          accepted, ``fn(angle, z, angle_fraction, height_fraction)`` is used.
        """
        if specification is None:
            return None

        angle_fraction = (angle % (2.0 * math.pi)) / (2.0 * math.pi)
        height_span = max(_EPS, z_top - z_bottom)
        height_fraction = max(0.0, min(1.0, (z_coord - z_bottom) / height_span))

        if callable(specification):
            try:
                value = specification(angle, z_coord)
            except TypeError:
                value = specification(angle, z_coord, angle_fraction, height_fraction)
            number = float(value)
            return number if math.isfinite(number) else None

        if isinstance(specification, (int, float)):
            number = float(specification)
            return number if math.isfinite(number) else None

        if isinstance(specification, (str, bytes)):
            return None

        try:
            rows = list(specification)
        except TypeError:
            return None
        if not rows:
            return None

        row_index = min(len(rows) - 1, int(height_fraction * len(rows)))
        selected = rows[row_index]

        if isinstance(selected, (int, float)):
            number = float(selected)
            return number if math.isfinite(number) else None

        if isinstance(selected, (str, bytes)):
            return None
        try:
            columns = list(selected)
        except TypeError:
            return None
        if not columns:
            return None

        column_index = min(len(columns) - 1, int(angle_fraction * len(columns)))
        number = float(columns[column_index])
        return number if math.isfinite(number) else None

    def _cylinder_primitives(
        self,
        obj: Dict[str, Any],
        quality: str,
    ) -> List[Dict[str, Any]]:
        radius = max(0.0, float(obj.get("radius", 1.0)))
        rt = obj.get("radius_top")
        radius_top = max(0.0, float(rt if rt is not None else radius))
        height = max(0.0, float(obj.get("height", 1.0)))
        center = obj.get("center", Point3D(0.0, 0.0, 0.0))
        color = obj.get("color", "lightgray")
        back_color = obj.get("back_color", "")
        outline = obj.get("outline", "black")
        plate_thickness = obj.get("plate_thickness")
        thickness_range = obj.get("thickness_range")
        if thickness_range is not None:
            thickness_minimum = float(thickness_range[0])
            thickness_maximum = float(thickness_range[1])
        elif self._thickness_legend is not None:
            thickness_minimum = float(self._thickness_legend["minimum"])
            thickness_maximum = float(self._thickness_legend["maximum"])
        else:
            thickness_values = _flatten_numeric_values(plate_thickness)
            thickness_minimum = min(thickness_values) if thickness_values else 0.0
            thickness_maximum = max(thickness_values) if thickness_values else 1.0
        segments = self._scaled_segments(int(obj.get("segments", 32)), quality)
        requested_height_segments = max(1, int(obj.get("height_segments", 24)))
        capped = bool(obj.get("capped", True))
        opacity = max(0.0, min(1.0, float(obj.get("opacity", 1.0))))

        # An opaque shell only needs the camera-facing half. A transparent or
        # stippled shell must retain the back-facing half as well; otherwise the
        # cylinder looks cut away rather than semi-transparent. The public
        # show_backfaces option can override this automatic behaviour.
        show_backfaces = obj.get("show_backfaces")
        if show_backfaces is None:
            show_backfaces = bool(back_color) or opacity < 0.90
        cull_shell_backfaces = not bool(show_backfaces)

        z_bottom = center.z - height / 2.0
        z_top = center.z + height / 2.0
        z_breaks = self._adaptive_z_breaks(
            z_bottom,
            z_top,
            requested_height_segments,
            quality,
        )

        angles = [2.0 * math.pi * index / segments for index in range(segments)]
        cosines = [math.cos(angle) for angle in angles]
        sines = [math.sin(angle) for angle in angles]

        rings: List[List[Point3D]] = []
        for z_coord in z_breaks:
            t = (z_coord - z_bottom) / height if height > 0 else 0.0
            r_z = radius + t * (radius_top - radius)
            rings.append(
                [
                    Point3D(
                        center.x + r_z * cosines[index],
                        center.y + r_z * sines[index],
                        z_coord,
                    )
                    for index in range(segments)
                ]
            )

        primitives: List[Dict[str, Any]] = []
        for z_index in range(len(rings) - 1):
            lower_ring = rings[z_index]
            upper_ring = rings[z_index + 1]
            for index in range(segments):
                next_index = (index + 1) % segments
                angle_mid = 2.0 * math.pi * (index + 0.5) / segments
                z_mid = 0.5 * (z_breaks[z_index] + z_breaks[z_index + 1])
                patch_thickness = self._resolve_plate_thickness(
                    plate_thickness,
                    angle_mid,
                    z_mid,
                    z_bottom,
                    z_top,
                )
                patch_color = (
                    _interpolate_thickness_color(
                        patch_thickness,
                        thickness_minimum,
                        thickness_maximum,
                    )
                    if patch_thickness is not None
                    else color
                )
                primitive = self._polygon_primitive(
                    [
                        lower_ring[index],
                        lower_ring[next_index],
                        upper_ring[next_index],
                        upper_ring[index],
                    ],
                    patch_color,
                    outline,
                    width=1,
                    layer=0,
                    cull_backface=cull_shell_backfaces,
                    opacity=opacity,
                    back_color=back_color,
                )
                if primitive:
                    primitive["two_sided_shell"] = bool(back_color)
                    primitives.append(primitive)

        if capped and rings:
            top_cap = self._polygon_primitive(
                rings[-1],
                color,
                outline,
                width=1,
                layer=1,
                cull_backface=cull_shell_backfaces,
                opacity=opacity,
                back_color=back_color,
            )
            bottom_cap = self._polygon_primitive(
                list(reversed(rings[0])),
                color,
                outline,
                width=1,
                layer=1,
                cull_backface=cull_shell_backfaces,
                opacity=opacity,
                back_color=back_color,
            )
            if top_cap:
                primitives.append(top_cap)
            if bottom_cap:
                primitives.append(bottom_cap)

        return primitives

    def _longitudinal_stiffener_primitives(
        self,
        obj: Dict[str, Any],
        quality: str,
    ) -> List[Dict[str, Any]]:
        radius = float(obj.get("radius", 1.0))
        rt = obj.get("radius_top")
        radius_top = float(rt if rt is not None else radius)
        height = float(obj.get("height", 1.0))
        angle = float(obj.get("angle", 0.0))
        web_height = max(0.0, float(obj.get("web_height", 0.1)))
        flange_width = max(0.0, float(obj.get("flange_width", 0.05)))
        flange_thickness = max(0.0, float(obj.get("flange_thickness", 0.01)))
        color = obj.get("color", "silver")
        outline = obj.get("outline", "black")
        width_segments = max(2, int(obj.get("segments", 4)))
        height_segments = max(4, int(obj.get("height_segments", 16)))
        inside = bool(obj.get("inside", False))
        radial_direction = -1.0 if inside else 1.0

        web_thickness = float(obj.get("web_thickness", 0.01))
        web_thickness_mm = web_thickness * 1000.0 if web_thickness < 1.0 else web_thickness
        flange_thickness_mm = flange_thickness * 1000.0 if flange_thickness < 1.0 else flange_thickness

        if self._thickness_legend is not None:
            web_color = self.thickness_color(web_thickness_mm)
            flange_color = self.thickness_color(flange_thickness_mm)
        else:
            web_color = flange_color = color

        if quality == "fast":
            width_segments = 2

        z_offset = float(obj.get("z_offset", 0.0))
        z_bottom = -height / 2.0 + z_offset
        z_top = height / 2.0 + z_offset
        # Longitudinal stiffeners need finer local vertical patches than the
        # shell.  In particular, a short patch immediately below the top cap
        # prevents the painter-order sort from hiding the visible outer part of
        # a front-side stiffener behind the large cap polygon.
        longitudinal_break_request = (
            height_segments if quality == "fast" else height_segments * 3
        )
        z_breaks = self._adaptive_z_breaks(
            z_bottom,
            z_top,
            longitudinal_break_request,
            quality,
        )

        cosine = math.cos(angle)
        sine = math.sin(angle)
        web_tip_radius = max(_EPS, radius + radial_direction * web_height)
        attachment_points = []
        tip_points = []
        for z in z_breaks:
            t = (z - z_bottom) / height if height > 0 else 0.0
            r_z = radius + t * (radius_top - radius)
            att_r = max(_EPS, r_z)
            tip_r = max(_EPS, r_z + radial_direction * web_height)
            attachment_points.append(Point3D(att_r * cosine, att_r * sine, z))
            tip_points.append(Point3D(tip_r * cosine, tip_r * sine, z))

        primitives: List[Dict[str, Any]] = []
        for z_index in range(len(z_breaks) - 1):
            web = self._polygon_primitive(
                [
                    attachment_points[z_index],
                    tip_points[z_index],
                    tip_points[z_index + 1],
                    attachment_points[z_index + 1],
                ],
                web_color,
                outline,
                width=1,
                layer=12,
                cull_backface=False,
            )
            if web:
                primitives.append(web)

        if flange_width > 0.0:
            flange_grid: List[List[Point3D]] = []
            for z in z_breaks:
                t = (z - z_bottom) / height if height > 0 else 0.0
                r_z = radius + t * (radius_top - radius)
                tip_r = max(_EPS, r_z + radial_direction * web_height)
                f_r = max(_EPS, tip_r + radial_direction * 0.5 * flange_thickness)
                half_angle = 0.5 * flange_width / f_r
                flange_angles = [
                    angle - half_angle + 2.0 * half_angle * index / (width_segments - 1)
                    for index in range(width_segments)
                ]
                flange_grid.append(
                    [
                        Point3D(f_r * math.cos(fa), f_r * math.sin(fa), z)
                        for fa in flange_angles
                    ]
                )

            for z_index in range(len(z_breaks) - 1):
                lower = flange_grid[z_index]
                upper = flange_grid[z_index + 1]
                for width_index in range(width_segments - 1):
                    flange = self._polygon_primitive(
                        [
                            lower[width_index],
                            lower[width_index + 1],
                            upper[width_index + 1],
                            upper[width_index],
                        ],
                        flange_color,
                        outline,
                        width=1,
                        layer=13,
                        cull_backface=False,
                    )
                    if flange:
                        primitives.append(flange)

        return primitives

    def _ring_stiffener_primitives(
        self,
        obj: Dict[str, Any],
        quality: str,
    ) -> List[Dict[str, Any]]:
        radius = float(obj.get("radius", 1.0))
        z_position = float(obj.get("z_position", 0.0))
        web_height = max(0.0, float(obj.get("web_height", 0.1)))
        web_thickness = max(0.0, float(obj.get("web_thickness", 0.01)))
        flange_width = max(0.0, float(obj.get("flange_width", 0.05)))
        flange_thickness = max(0.0, float(obj.get("flange_thickness", 0.01)))
        color = obj.get("color", "dimgray")
        outline = obj.get("outline", "black")
        segments = self._scaled_segments(int(obj.get("segments", 32)), quality)
        inside = bool(obj.get("inside", False))
        radial_direction = -1.0 if inside else 1.0

        web_thickness_mm = web_thickness * 1000.0 if web_thickness < 1.0 else web_thickness
        flange_thickness_mm = flange_thickness * 1000.0 if flange_thickness < 1.0 else flange_thickness

        if self._thickness_legend is not None:
            web_color = self.thickness_color(web_thickness_mm)
            flange_color = self.thickness_color(flange_thickness_mm)
        else:
            web_color = flange_color = color

        attachment_radius = max(_EPS, radius)
        tip_radius = max(_EPS, radius + radial_direction * web_height)
        z_lower = z_position - web_thickness / 2.0
        z_upper = z_position + web_thickness / 2.0

        angles = [2.0 * math.pi * index / segments for index in range(segments)]
        cosines = [math.cos(angle) for angle in angles]
        sines = [math.sin(angle) for angle in angles]

        attachment_lower = [
            Point3D(
                attachment_radius * cosines[index],
                attachment_radius * sines[index],
                z_lower,
            )
            for index in range(segments)
        ]
        tip_lower = [
            Point3D(tip_radius * cosines[index], tip_radius * sines[index], z_lower)
            for index in range(segments)
        ]
        attachment_upper = [
            Point3D(
                attachment_radius * cosines[index],
                attachment_radius * sines[index],
                z_upper,
            )
            for index in range(segments)
        ]
        tip_upper = [
            Point3D(tip_radius * cosines[index], tip_radius * sines[index], z_upper)
            for index in range(segments)
        ]

        primitives: List[Dict[str, Any]] = []
        for index in range(segments):
            next_index = (index + 1) % segments

            # During interaction use one web mid-surface instead of three solid
            # web surfaces. The full representation is restored on release.
            if quality == "fast":
                mid_attachment_0 = Point3D(
                    attachment_radius * cosines[index],
                    attachment_radius * sines[index],
                    z_position,
                )
                mid_tip_0 = Point3D(
                    tip_radius * cosines[index],
                    tip_radius * sines[index],
                    z_position,
                )
                mid_tip_1 = Point3D(
                    tip_radius * cosines[next_index],
                    tip_radius * sines[next_index],
                    z_position,
                )
                mid_attachment_1 = Point3D(
                    attachment_radius * cosines[next_index],
                    attachment_radius * sines[next_index],
                    z_position,
                )
                faces = [[mid_attachment_0, mid_tip_0, mid_tip_1, mid_attachment_1]]
            else:
                faces = [
                    [
                        attachment_lower[index],
                        tip_lower[index],
                        tip_lower[next_index],
                        attachment_lower[next_index],
                    ],
                    [
                        attachment_upper[next_index],
                        tip_upper[next_index],
                        tip_upper[index],
                        attachment_upper[index],
                    ],
                    [
                        tip_lower[index],
                        tip_upper[index],
                        tip_upper[next_index],
                        tip_lower[next_index],
                    ],
                ]

            for face in faces:
                primitive = self._polygon_primitive(
                    face,
                    web_color,
                    outline,
                    width=1,
                    layer=20,
                    cull_backface=False,
                )
                if primitive:
                    primitives.append(primitive)

        if flange_width > 0.0:
            flange_radius = max(
                _EPS,
                tip_radius + radial_direction * 0.5 * flange_thickness,
            )
            flange_z_lower = z_position - flange_width / 2.0
            flange_z_upper = z_position + flange_width / 2.0
            flange_lower = [
                Point3D(
                    flange_radius * cosines[index],
                    flange_radius * sines[index],
                    flange_z_lower,
                )
                for index in range(segments)
            ]
            flange_upper = [
                Point3D(
                    flange_radius * cosines[index],
                    flange_radius * sines[index],
                    flange_z_upper,
                )
                for index in range(segments)
            ]

            for index in range(segments):
                next_index = (index + 1) % segments
                primitive = self._polygon_primitive(
                    [
                        flange_lower[index],
                        flange_lower[next_index],
                        flange_upper[next_index],
                        flange_upper[index],
                    ],
                    flange_color,
                    outline,
                    width=1,
                    layer=21,
                    cull_backface=False,
                )
                if primitive:
                    primitives.append(primitive)

        return primitives

    # ------------------------------------------------------------------
    # Hidden-surface helpers
    # ------------------------------------------------------------------

    def set_opaque_cylinder_occluder(
            self,
            radius: float,
            height: float,
            center: Optional[Point3D] = None,
    ) -> None:
        # Register a non-rendered finite cylinder used for hidden-surface tests.
        self._explicit_opaque_cylinder_occluders.append(
            {
                "radius": max(0.0, float(radius)),
                "height": max(0.0, float(height)),
                "center": center if center is not None else Point3D(0.0, 0.0, 0.0),
            }
        )

    def _collect_opaque_cylinder_occluders(self) -> List[Dict[str, Any]]:
        occluders = list(self._explicit_opaque_cylinder_occluders)
        for obj in self.objects:
            if obj.get("type") != "cylinder":
                continue
            opacity = max(0.0, min(1.0, float(obj.get("opacity", 1.0))))
            show_backfaces = obj.get("show_backfaces")
            if show_backfaces is None:
                show_backfaces = opacity < 0.90
            if opacity < 0.94:
                continue
            if bool(show_backfaces):
                continue
            occluders.append(
                {
                    "radius": max(0.0, float(obj.get("radius", 0.0))),
                    "height": max(0.0, float(obj.get("height", 0.0))),
                    "center": obj.get("center", Point3D(0.0, 0.0, 0.0)),
                }
            )
        return occluders

    @staticmethod
    def _point_is_hidden_by_finite_cylinder(
            camera_position: Point3D,
            point: Point3D,
            occluder: Dict[str, Any],
    ) -> bool:
        radius = max(0.0, float(occluder.get("radius", 0.0)))
        height = max(0.0, float(occluder.get("height", 0.0)))
        center = occluder.get("center", Point3D(0.0, 0.0, 0.0))
        if radius <= _EPS or height <= _EPS:
            return False

        local_x = point.x - center.x
        local_y = point.y - center.y
        radial_distance = math.hypot(local_x, local_y)
        z_bottom = center.z - 0.5 * height
        z_top = center.z + 0.5 * height
        radial_tolerance = max(radius * 1.0e-6, 1.0e-8)
        z_tolerance = max(height * 1.0e-7, 1.0e-8)

        if radial_distance >= radius - radial_tolerance:
            return False
        if point.z < z_bottom - z_tolerance or point.z > z_top + z_tolerance:
            return False

        origin_x = camera_position.x - center.x
        origin_y = camera_position.y - center.y
        direction_x = point.x - camera_position.x
        direction_y = point.y - camera_position.y
        direction_z = point.z - camera_position.z

        coefficient_a = direction_x * direction_x + direction_y * direction_y
        if coefficient_a <= _EPS:
            return False
        coefficient_b = 2.0 * (
            origin_x * direction_x + origin_y * direction_y
        )
        coefficient_c = (
            origin_x * origin_x
            + origin_y * origin_y
            - radius * radius
        )
        discriminant = coefficient_b * coefficient_b - 4.0 * coefficient_a * coefficient_c
        if discriminant < 0.0:
            return False

        square_root = math.sqrt(max(0.0, discriminant))
        denominator = 2.0 * coefficient_a
        roots = sorted(
            (
                (-coefficient_b - square_root) / denominator,
                (-coefficient_b + square_root) / denominator,
            )
        )
        for parameter in roots:
            if parameter <= 1.0e-8 or parameter >= 1.0 - 1.0e-6:
                continue
            intersection_z = camera_position.z + parameter * direction_z
            if z_bottom - z_tolerance <= intersection_z <= z_top + z_tolerance:
                return True
        return False

    def _primitive_hidden_by_opaque_cylinder(
            self,
            primitive: Dict[str, Any],
            occluders: Sequence[Dict[str, Any]],
            camera_position: Point3D,
    ) -> bool:
        # Member surfaces use layers 10-29. Shells, result plates and selection
        # outlines are intentionally excluded from this hidden-surface filter.
        layer = int(primitive.get("layer", 0))
        if layer < 10 or layer >= 30 or not occluders:
            return False

        center = primitive.get("center")
        if not isinstance(center, Point3D):
            return False
        return any(
            self._point_is_hidden_by_finite_cylinder(
                camera_position,
                center,
                occluder,
            )
            for occluder in occluders
        )

    @staticmethod
    def _faces_hidden_by_occluders(
        scene: _CompiledScene,
        occluders: Sequence[Dict[str, Any]],
        camera_position: np.ndarray,
    ) -> np.ndarray:
        """Vectorised form of :meth:`_primitive_hidden_by_opaque_cylinder`."""
        centers = scene.face_center
        hidden = np.zeros(len(centers), dtype=bool)
        candidates = scene.face_occludable
        if not candidates.any():
            return hidden

        for occluder in occluders:
            radius = max(0.0, float(occluder.get("radius", 0.0)))
            height = max(0.0, float(occluder.get("height", 0.0)))
            if radius <= _EPS or height <= _EPS:
                continue
            center = occluder.get("center", Point3D(0.0, 0.0, 0.0))
            base = np.array([center.x, center.y, center.z], dtype=np.float32)

            local = centers - base
            radial = np.hypot(local[:, 0], local[:, 1])
            radial_tolerance = max(radius * 1.0e-6, 1.0e-8)
            z_tolerance = max(height * 1.0e-7, 1.0e-8)
            inside = (
                candidates
                & ~hidden
                & (radial < radius - radial_tolerance)
                & (local[:, 2] >= -0.5 * height - z_tolerance)
                & (local[:, 2] <= 0.5 * height + z_tolerance)
            )
            if not inside.any():
                continue

            eye = camera_position - base
            direction = centers[inside] - camera_position
            a = direction[:, 0] ** 2 + direction[:, 1] ** 2
            b = 2.0 * (eye[0] * direction[:, 0] + eye[1] * direction[:, 1])
            c = eye[0] ** 2 + eye[1] ** 2 - radius * radius
            discriminant = b * b - 4.0 * a * c
            usable = (a > _EPS) & (discriminant >= 0.0)

            root = np.sqrt(np.where(usable, discriminant, 0.0))
            denominator = np.where(usable, 2.0 * a, 1.0)
            crossed = np.zeros(len(direction), dtype=bool)
            for sign in (-1.0, 1.0):
                parameter = (-b + sign * root) / denominator
                valid = usable & (parameter > 1.0e-8) & (parameter < 1.0 - 1.0e-6)
                intersection_z = eye[2] + parameter * direction[:, 2]
                crossed |= (
                    valid
                    & (intersection_z >= -0.5 * height - z_tolerance)
                    & (intersection_z <= 0.5 * height + z_tolerance)
                )

            hidden[np.nonzero(inside)[0][crossed]] = True

        return hidden

    # ------------------------------------------------------------------
    # Public scene API
    # ------------------------------------------------------------------

    def _add_object(self, obj: Dict[str, Any]) -> None:
        self.objects.append(obj)
        self._invalidate_geometry_cache()
        self._request_redraw()

    def _retained_mesh_changed(self, handle: MeshHandle, change: str) -> None:
        matching = [obj for obj in self.objects if obj.get("handle") is handle]
        if change == "remove":
            self.objects[:] = [obj for obj in self.objects if obj.get("handle") is not handle]
        elif change in {"topology", "position", "displacement", "active", "transform"}:
            for obj in matching:
                obj.pop("_array_geometry_cache", None)
        self._invalidate_geometry_cache()
        self._request_redraw()

    def add_mesh_arrays(
        self,
        mesh: MeshArrays,
        *,
        color: str = "#9aa7b4",
        outline: str = "",
        width: int = 1,
        layer: int = 5,
        cull_backface: bool = True,
        opacity: float = 1.0,
        stipple: str = "",
        back_color: str = "",
        tags: str = "",
        lit: bool = True,
        two_sided_shell: bool = False,
        face_colors: Optional[Sequence[str]] = None,
        scalar_range: Optional[Tuple[float, float]] = None,
        invalid_color: str = "#808080",
        owners: Optional[PackedOwnerTable] = None,
        owner_resolver: Optional[Callable[..., object]] = None,
        line_color: Optional[str] = None,
        line_width: Optional[int] = None,
        point_color: str = "#2563eb",
        point_outline: str = "",
        point_size: int = 6,
    ) -> MeshHandle:
        """Add an indexed retained mesh without per-element scene objects."""

        if not isinstance(mesh, MeshArrays):
            raise TypeError("mesh must be MeshArrays")
        if owners is not None:
            if not isinstance(owners, PackedOwnerTable):
                raise TypeError("owners must be a PackedOwnerTable")
            mapped_triangles = len(owners.triangle_offsets) - 1
            if mapped_triangles not in (0, mesh.triangle_count):
                raise ValueError("triangle owner mappings must match triangle count")
        if scalar_range is not None:
            scalar_range = (float(scalar_range[0]), float(scalar_range[1]))
        handle = MeshHandle(mesh, on_change=self._retained_mesh_changed)
        self._add_object(
            {
                "type": "mesh_arrays",
                "handle": handle,
                "color": str(color),
                "outline": str(outline),
                "width": int(width),
                "layer": int(layer),
                "cull_backface": bool(cull_backface),
                "opacity": float(opacity),
                "stipple": str(stipple),
                "back_color": str(back_color),
                "tags": str(tags),
                "lit": bool(lit),
                "two_sided_shell": bool(two_sided_shell),
                "face_colors": face_colors,
                "scalar_range": scalar_range,
                "invalid_color": str(invalid_color),
                "owners": owners,
                "owner_resolver": owner_resolver,
                "line_color": line_color,
                "line_width": line_width,
                "point_color": str(point_color),
                "point_outline": str(point_outline),
                "point_size": max(1, int(point_size)),
            }
        )
        return handle

    def add_layer(self, layer: Any) -> Any:
        """Attach a renderer-neutral layer such as ``GeometryLayer``."""

        attach = getattr(layer, "attach", None)
        if not callable(attach):
            raise TypeError("layer must provide attach(viewer)")
        return attach(self)

    def add_line(
        self,
        start: Point3D,
        end: Point3D,
        color: str = "black",
        width: int = 1,
        layer: int = 30,
        draw_overlay: bool = False,
        tags: str = "",
        binding: Any = None,
    ) -> None:
        self._add_object(
            {
                "type": "line",
                "start": start,
                "end": end,
                "color": color,
                "width": width,
                "layer": int(layer),
                "draw_overlay": bool(draw_overlay),
                "tags": tags,
                "binding": _coerce_pick_binding(binding, tags),
            }
        )

    def add_text(
        self,
        point: Point3D,
        text: str,
        color: str = "black",
        font: Tuple[str, int, str] = ("Segoe UI", 9, "bold"),
        anchor: str = tk.CENTER,
        layer: int = 35,
        draw_overlay: bool = True,
    ) -> None:
        self._add_object(
            {
                "type": "text",
                "point": point,
                "text": text,
                "color": color,
                "font": font,
                "anchor": anchor,
                "layer": int(layer),
                "draw_overlay": bool(draw_overlay),
            }
        )

    def add_polygon(
        self,
        vertices: Iterable[Point3D],
        color: str = "gray",
        outline: str = "black",
        width: int = 1,
        cull_backface: bool = False,
        layer: int = 5,
        stipple: str = "",
        tags: str = "",
        back_color: str = "",
        two_sided_shell: bool = False,
        opacity: Optional[float] = None,
        lit: bool = True,
        binding: Any = None,
    ) -> None:
        self._add_object(
            {
                "type": "polygon",
                "vertices": list(vertices),
                "color": color,
                "back_color": back_color,
                "outline": outline,
                "width": width,
                "cull_backface": cull_backface,
                "layer": layer,
                "stipple": stipple,
                "opacity": opacity,
                "tags": tags,
                "two_sided_shell": two_sided_shell,
                "lit": lit,
                "binding": _coerce_pick_binding(binding, tags),
            }
        )

    def add_faces(
        self,
        polygons: Iterable[Any],
        colors: Any = "#9aa7b4",
        outline: str = "",
        width: int = 1,
        layer: int = 5,
        cull_backface: bool = False,
        opacity: float = 1.0,
        stipple: str = "",
        back_colors: Optional[Sequence[str]] = None,
        tags: str = "",
        lit: bool = True,
        two_sided_shell: bool = False,
        bindings: Any = None,
    ) -> None:
        """
        Add many independent faces in one call, each with its own colour.

        This is the fast path for result fields - an FE mesh coloured by
        stress, a deformed shape, a utilisation plot - where every element is
        a separate polygon with its own fill.  Adding them one at a time with
        :meth:`add_polygon` costs a dictionary and a Python centroid/normal
        per element; a batch stores one flat vertex array and computes all of
        its centroids and normals as array operations.

        ``polygons`` is a sequence of vertex sequences (``Point3D`` or
        ``(x, y, z)``), or an ``(faces, vertices, 3)`` array when every face
        has the same vertex count.  ``colors`` is one colour for the whole
        batch or one per face.
        """
        vertices, counts = _flatten_polygons(polygons)
        total = len(counts)
        if total == 0:
            return

        if isinstance(colors, str):
            color_list = [colors] * total
        else:
            color_list = [str(color) for color in colors]
            if len(color_list) != total:
                raise ValueError(
                    f"colors has {len(color_list)} entries for {total} faces"
                )
        if back_colors is None:
            back_list = None
        else:
            back_list = [str(color) for color in back_colors]
            if len(back_list) != total:
                raise ValueError(
                    f"back_colors has {len(back_list)} entries for {total} faces"
                )

        if stipple or opacity < stipple_module.OPAQUE_THRESHOLD:
            cull_backface = False

        binding_list = _coerce_pick_bindings(bindings, total, tags)

        self._add_object(
            {
                "type": "faces",
                "vertices": vertices,
                "counts": counts,
                "colors": color_list,
                "back_colors": back_list,
                "outline": outline,
                "width": width,
                "layer": int(layer),
                "cull_backface": bool(cull_backface),
                "opacity": float(opacity),
                "stipple": stipple,
                "tags": tags,
                "lit": bool(lit),
                "two_sided_shell": bool(two_sided_shell),
                "bindings": binding_list,
            }
        )

    def add_markers(
        self,
        points: Iterable[Any],
        colors: Any = "#2563eb",
        size: Any = 6,
        outline: Any = "",
        layer: int = 32,
        tags: str = "",
        bindings: Any = None,
    ) -> None:
        """Add fixed-pixel point markers as one compiled batch.

        Markers are intended for geometry points and mesh nodes.  Their
        screen-space size remains usable while zooming, and their bindings
        participate in point/window selection without one scene object per
        node.
        """

        point_list = [as_point(point) for point in points]
        total = len(point_list)
        if total == 0:
            return

        def expanded(value: Any, name: str, transform: Callable[[Any], Any]) -> List[Any]:
            if isinstance(value, str):
                return [transform(value)] * total
            try:
                raw_values = list(value)
            except TypeError:
                return [transform(value)] * total
            values = [transform(item) for item in raw_values]
            if len(values) != total:
                raise ValueError(f"{name} has {len(values)} entries for {total} markers")
            return values

        color_list = expanded(colors, "colors", str)
        outline_list = expanded(outline, "outline", str)
        size_list = expanded(size, "size", lambda value: max(1, int(value)))
        self._add_object(
            {
                "type": "markers",
                "points": point_list,
                "colors": color_list,
                "outlines": outline_list,
                "sizes": size_list,
                "layer": int(layer),
                "tags": tags,
                "bindings": _coerce_pick_bindings(bindings, total, tags),
            }
        )

    def add_mesh(
        self,
        vertices: Iterable[Any],
        faces: Iterable[Sequence[int]],
        color: str = "#9aa7b4",
        outline: str = "",
        width: int = 1,
        layer: int = 5,
        cull_backface: bool = True,
        opacity: float = 1.0,
        stipple: str = "",
        back_color: str = "",
        tags: str = "",
        lit: bool = True,
        face_colors: Optional[Sequence[str]] = None,
        two_sided_shell: bool = False,
        bindings: Any = None,
    ) -> None:
        """
        Add an indexed triangle/polygon mesh.

        ``vertices`` accepts ``Point3D`` objects or any ``(x, y, z)``
        sequence.  Faces must be wound counter-clockwise seen from outside
        for ``cull_backface`` and lighting to be correct - every builder in
        :mod:`anytk3d.shapes` already follows that convention.
        """
        if stipple or opacity < stipple_module.OPAQUE_THRESHOLD:
            # A see-through solid must keep its far side, or it looks cut open.
            cull_backface = False
        face_list = [tuple(int(index) for index in face) for face in faces]
        binding_list = _coerce_pick_bindings(bindings, len(face_list), tags)
        self._add_object(
            {
                "type": "mesh",
                "points": [as_point(vertex) for vertex in vertices],
                "faces": face_list,
                "color": color,
                "face_colors": list(face_colors) if face_colors is not None else None,
                "outline": outline,
                "width": width,
                "layer": int(layer),
                "cull_backface": bool(cull_backface),
                "stipple": stipple,
                "opacity": float(opacity),
                "back_color": back_color,
                "tags": tags,
                "lit": bool(lit),
                "two_sided_shell": bool(two_sided_shell),
                "bindings": binding_list,
            }
        )

    def add_shape(
        self,
        mesh: Mesh,
        position: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        up: Optional[Point3D] = None,
        **material: Any,
    ) -> None:
        """Place a :class:`anytk3d.shapes.Mesh` in the scene."""
        placed = mesh.placed(origin=position, axis=axis, up_hint=up)
        self.add_mesh(placed.points(), placed.faces, **material)

    # -- typical solids -------------------------------------------------

    def add_box(
        self,
        size_x: float = 1.0,
        size_y: Optional[float] = None,
        size_z: Optional[float] = None,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        **material: Any,
    ) -> None:
        self.add_shape(
            shapes_module.box(size_x, size_y, size_z), center, axis, **material
        )

    def add_box_from_bounds(
        self,
        minimum: Point3D,
        maximum: Point3D,
        **material: Any,
    ) -> None:
        self.add_shape(shapes_module.box_from_bounds(minimum, maximum), **material)

    def add_sphere(
        self,
        radius: float = 1.0,
        center: Optional[Point3D] = None,
        segments: int = 24,
        rings: int = 16,
        **material: Any,
    ) -> None:
        self.add_shape(shapes_module.sphere(radius, segments, rings), center, **material)

    def add_cone(
        self,
        radius: float = 1.0,
        height: float = 1.0,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        segments: int = 32,
        capped: bool = True,
        **material: Any,
    ) -> None:
        self.add_shape(
            shapes_module.cone(radius, height, segments, capped), center, axis, **material
        )

    def add_frustum(
        self,
        radius_bottom: float = 1.0,
        radius_top: float = 0.5,
        height: float = 1.0,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        segments: int = 32,
        height_segments: int = 1,
        capped: bool = True,
        **material: Any,
    ) -> None:
        self.add_shape(
            shapes_module.frustum(
                radius_bottom, radius_top, height, segments, height_segments, capped
            ),
            center,
            axis,
            **material,
        )

    def add_tube(
        self,
        outer_radius: float = 1.0,
        inner_radius: float = 0.7,
        height: float = 1.0,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        segments: int = 32,
        height_segments: int = 1,
        capped: bool = True,
        **material: Any,
    ) -> None:
        self.add_shape(
            shapes_module.tube(
                outer_radius, inner_radius, height, segments, height_segments, capped
            ),
            center,
            axis,
            **material,
        )

    def add_torus(
        self,
        major_radius: float = 1.0,
        minor_radius: float = 0.25,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        segments: int = 36,
        rings: int = 18,
        **material: Any,
    ) -> None:
        self.add_shape(
            shapes_module.torus(major_radius, minor_radius, segments, rings),
            center,
            axis,
            **material,
        )

    def add_pyramid(
        self,
        base_radius: float = 1.0,
        height: float = 1.0,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        sides: int = 4,
        capped: bool = True,
        **material: Any,
    ) -> None:
        self.add_shape(
            shapes_module.pyramid(base_radius, height, sides, capped),
            center,
            axis,
            **material,
        )

    def add_wedge(
        self,
        size_x: float = 1.0,
        size_y: float = 1.0,
        size_z: float = 1.0,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        **material: Any,
    ) -> None:
        self.add_shape(
            shapes_module.wedge(size_x, size_y, size_z), center, axis, **material
        )

    def add_prism(
        self,
        profile: Sequence[Tuple[float, float]],
        height: float = 1.0,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        capped: bool = True,
        **material: Any,
    ) -> None:
        self.add_shape(
            shapes_module.prism(profile, height, capped), center, axis, **material
        )

    def add_extrusion(
        self,
        profile: Sequence[Tuple[float, float]],
        path: Sequence[Point3D],
        capped: bool = True,
        up: Optional[Point3D] = None,
        **material: Any,
    ) -> None:
        self.add_shape(
            shapes_module.extrusion(profile, path, capped, up), **material
        )

    def add_disk(
        self,
        outer_radius: float = 1.0,
        inner_radius: float = 0.0,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        segments: int = 32,
        **material: Any,
    ) -> None:
        material.setdefault("cull_backface", False)
        self.add_shape(
            shapes_module.disk(outer_radius, inner_radius, segments),
            center,
            axis,
            **material,
        )

    def add_plane(
        self,
        size_x: float = 1.0,
        size_y: float = 1.0,
        center: Optional[Point3D] = None,
        axis: Optional[Point3D] = None,
        nx: int = 1,
        ny: int = 1,
        **material: Any,
    ) -> None:
        material.setdefault("cull_backface", False)
        self.add_shape(
            shapes_module.plane(size_x, size_y, nx, ny), center, axis, **material
        )

    def add_arrow(
        self,
        start: Point3D,
        end: Point3D,
        shaft_radius: Optional[float] = None,
        head_radius: Optional[float] = None,
        head_length: Optional[float] = None,
        segments: int = 16,
        **material: Any,
    ) -> None:
        material.setdefault("color", "#b45309")
        self.add_shape(
            shapes_module.arrow(
                start, end, shaft_radius, head_radius, head_length, segments
            ),
            **material,
        )

    def add_beam(
        self,
        start: Point3D,
        end: Point3D,
        kind: str = "T",
        web_height: float = 0.2,
        web_thickness: float = 0.01,
        flange_width: float = 0.1,
        flange_thickness: float = 0.015,
        up: Optional[Point3D] = None,
        capped: bool = True,
        **material: Any,
    ) -> None:
        """
        Add a structural profile swept from ``start`` to ``end``.

        Supported ``kind`` values are ``FB``, ``T``, ``I``, ``L``, ``C`` and
        ``BOX``.  The section sits on the start/end line and its web rises
        along ``up``, which matches how a stiffener is welded to plating.
        """
        self.add_shape(
            shapes_module.beam(
                start,
                end,
                kind,
                web_height,
                web_thickness,
                flange_width,
                flange_thickness,
                up,
                capped,
            ),
            **material,
        )

    def add_grid(
        self,
        size_x: float = 10.0,
        size_y: float = 10.0,
        step: float = 1.0,
        z: float = 0.0,
        center: Optional[Point3D] = None,
        color: str = "#c9d2dc",
        width: int = 1,
        layer: int = 2,
    ) -> None:
        """Add a ground reference grid as depth-sorted 3D lines."""
        for start, end in shapes_module.grid_lines(size_x, size_y, step, z, center):
            self.add_line(start, end, color=color, width=width, layer=layer)

    def add_cylinder(
        self,
        radius: float,
        height: float,
        radius_top: Optional[float] = None,
        center: Optional[Point3D] = None,
        color: str = "lightgray",
        back_color: str = "",
        outline: str = "black",
        segments: int = 32,
        height_segments: int = 24,
        capped: bool = True,
        opacity: float = 1.0,
        show_backfaces: Optional[bool] = None,
        plate_thickness: Any = None,
        thickness_range: Optional[Tuple[float, float]] = None,
        thickness_unit: str = "mm",
        thickness_legend_title: str = "Plate thickness",
        show_thickness_legend: bool = True,
    ) -> None:
        self.objects.append(
            {
                "type": "cylinder",
                "radius_top": radius_top,
                "radius": radius,
                "height": height,
                "center": center if center is not None else Point3D(0.0, 0.0, 0.0),
                "color": color,
                "back_color": back_color,
                "outline": outline,
                "segments": segments,
                "height_segments": height_segments,
                "capped": capped,
                "opacity": max(0.0, min(1.0, float(opacity))),
                "show_backfaces": show_backfaces,
                "plate_thickness": plate_thickness,
                "thickness_range": thickness_range,
            }
        )

        thickness_values = _flatten_numeric_values(plate_thickness)
        if show_thickness_legend and (thickness_values or thickness_range is not None):
            if thickness_range is not None and not thickness_values:
                thickness_values = [float(thickness_range[0]), float(thickness_range[1])]
            self.set_thickness_legend(
                thickness_values,
                unit=thickness_unit,
                title=thickness_legend_title,
                value_range=thickness_range,
            )
        else:
            self._invalidate_geometry_cache()
            self._request_redraw()

    def add_longitudinal_stiffener(
        self,
        radius: float,
        height: float,
        angle: float,
        radius_top: Optional[float] = None,
        web_height: float = 0.1,
        web_thickness: float = 0.01,
        flange_width: float = 0.05,
        flange_thickness: float = 0.01,
        color: str = "silver",
        outline: str = "black",
        segments: int = 4,
        height_segments: int = 16,
        inside: bool = False,
        z_offset: float = 0.0,
    ) -> None:
        self._add_object(
            {
                "type": "stiffener",
                "stiffener_type": "longitudinal",
                "radius_top": radius_top,
                "radius": radius,
                "height": height,
                "angle": angle,
                "web_height": web_height,
                "web_thickness": web_thickness,
                "flange_width": flange_width,
                "flange_thickness": flange_thickness,
                "color": color,
                "outline": outline,
                "segments": segments,
                "height_segments": height_segments,
                "inside": bool(inside),
                "z_offset": z_offset,
            }
        )

    def add_ring_stiffener(
        self,
        radius: float,
        z_position: float,
        web_height: float = 0.1,
        web_thickness: float = 0.01,
        flange_width: float = 0.05,
        flange_thickness: float = 0.01,
        color: str = "dimgray",
        outline: str = "black",
        segments: int = 32,
        inside: bool = False,
    ) -> None:
        self._add_object(
            {
                "type": "stiffener",
                "stiffener_type": "ring",
                "radius": radius,
                "z_position": z_position,
                "web_height": web_height,
                "web_thickness": web_thickness,
                "flange_width": flange_width,
                "flange_thickness": flange_thickness,
                "color": color,
                "outline": outline,
                "segments": segments,
                "inside": bool(inside),
            }
        )

    # ------------------------------------------------------------------
    # Camera API
    # ------------------------------------------------------------------

    def set_camera_position(self, position: Point3D) -> None:
        self.camera.set_position(position)
        self._request_redraw()

    def set_camera_target(self, target: Point3D) -> None:
        self.camera.set_target(target)
        self._request_redraw()

    def set_view(self, azimuth_degrees: float, elevation_degrees: float) -> None:
        self._interactive_render = False
        self.camera.set_orbit(
            azimuth=math.radians(azimuth_degrees),
            elevation=math.radians(elevation_degrees),
        )
        self._request_redraw()

    def set_iso_view(self) -> None:
        self.set_view(-45.0, 25.0)

    def add_rectangular_plate(
        self,
        x_start: float,
        x_end: float,
        y_start: float,
        y_end: float,
        z: float = 0.0,
        color: str = "gray",
        outline: str = "black",
        stipple: str = "",
        layer: int = 5,
        back_color: str = "",
        nx: int = 24,
        ny: int = 24,
        opacity: Optional[float] = None,
    ) -> None:
        dx = (x_end - x_start) / nx
        dy = (y_end - y_start) / ny
        for i in range(nx):
            for j in range(ny):
                x0 = x_start + i * dx
                x1 = x0 + dx
                y0 = y_start + j * dy
                y1 = y0 + dy
                self.add_polygon(
                    vertices=[
                        Point3D(x0, y0, z),
                        Point3D(x1, y0, z),
                        Point3D(x1, y1, z),
                        Point3D(x0, y1, z),
                    ],
                    color=color,
                    outline=outline,
                    stipple=stipple,
                    opacity=opacity,
                    layer=layer,
                    back_color=back_color,
                )

    def add_flat_stiffener(
        self,
        x_start: float,
        x_end: float,
        y: float,
        z_base: float,
        hw: float,
        b: float,
        color: str = "gray",
        outline: str = "black",
        stipple: str = "",
        layer_web: int = 12,
        layer_flange: int = 13,
        nx: int = 24,
        opacity: Optional[float] = None,
    ) -> None:
        dx = (x_end - x_start) / nx
        for i in range(nx):
            x0 = x_start + i * dx
            x1 = x0 + dx
            # Web
            self.add_polygon(
                vertices=[
                    Point3D(x0, y, z_base),
                    Point3D(x1, y, z_base),
                    Point3D(x1, y, z_base + hw),
                    Point3D(x0, y, z_base + hw),
                ],
                color=color,
                outline=outline,
                stipple=stipple,
                opacity=opacity,
                layer=layer_web,
            )
            # Flange
            if b > 0.0:
                self.add_polygon(
                    vertices=[
                        Point3D(x0, y - 0.5 * b, z_base + hw),
                        Point3D(x1, y - 0.5 * b, z_base + hw),
                        Point3D(x1, y + 0.5 * b, z_base + hw),
                        Point3D(x0, y + 0.5 * b, z_base + hw),
                    ],
                    color=color,
                    outline=outline,
                    stipple=stipple,
                    opacity=opacity,
                    layer=layer_flange,
                )

    def add_flat_girder(
        self,
        x: float,
        y_start: float,
        y_end: float,
        z_base: float,
        ghw: float,
        gb: float,
        color: str = "gray",
        outline: str = "black",
        stipple: str = "",
        layer_web: int = 14,
        layer_flange: int = 15,
        ny: int = 24,
        opacity: Optional[float] = None,
    ) -> None:
        dy = (y_end - y_start) / ny
        for j in range(ny):
            y0 = y_start + j * dy
            y1 = y0 + dy
            # Web
            self.add_polygon(
                vertices=[
                    Point3D(x, y0, z_base),
                    Point3D(x, y1, z_base),
                    Point3D(x, y1, z_base + ghw),
                    Point3D(x, y0, z_base + ghw),
                ],
                color=color,
                outline=outline,
                stipple=stipple,
                opacity=opacity,
                layer=layer_web,
            )
            # Flange
            if gb > 0.0:
                self.add_polygon(
                    vertices=[
                        Point3D(x - 0.5 * gb, y0, z_base + ghw),
                        Point3D(x - 0.5 * gb, y1, z_base + ghw),
                        Point3D(x + 0.5 * gb, y1, z_base + ghw),
                        Point3D(x + 0.5 * gb, y0, z_base + ghw),
                    ],
                    color=color,
                    outline=outline,
                    stipple=stipple,
                    opacity=opacity,
                    layer=layer_flange,
                )

    def set_top_view(self) -> None:
        self.set_view(-90.0, 89.0)

    def set_side_view(self) -> None:
        self.set_view(0.0, 0.0)

    def set_front_view(self) -> None:
        self.set_view(-90.0, 0.0)

    def reset_camera(self) -> None:
        self._interactive_render = False
        self.camera = Camera3D()
        self.fit_to_scene(redraw=False)
        self._request_redraw()

    def fit_to_scene(self, padding: float = 1.25, redraw: bool = True) -> None:
        bounds = self._scene_bounds()
        if bounds is None:
            if redraw:
                self._request_redraw()
            return

        minimum, maximum = bounds
        center = Point3D(
            0.5 * (minimum.x + maximum.x),
            0.5 * (minimum.y + maximum.y),
            0.5 * (minimum.z + maximum.z),
        )
        diagonal = (maximum - minimum).length()
        radius = max(0.5 * diagonal, 0.1)

        self.width, self.height = self._viewport_size()
        width = max(1, self._plot_width())
        height = self.height
        aspect = width / height
        vertical_half_fov = max(math.radians(5.0), self.camera.fov / 2.0)
        tan_vertical = math.tan(vertical_half_fov)
        tan_horizontal = tan_vertical * aspect

        # Fit the eight bounding-box corners to the frustum rather than the
        # bounding sphere.  A corner at camera-space (u, v, w) is inside the
        # view from distance d when |u| <= tan_h * (d + w) and likewise for v,
        # so the smallest workable distance is the largest of those bounds.
        # Sphere fitting would leave a wide, flat model floating in the middle
        # of an otherwise empty viewport.
        right, camera_up, forward = self.camera.basis()
        distance = 0.0
        for corner_x in (minimum.x, maximum.x):
            for corner_y in (minimum.y, maximum.y):
                for corner_z in (minimum.z, maximum.z):
                    offset = Point3D(corner_x, corner_y, corner_z) - center
                    depth = offset.dot(forward)
                    distance = max(
                        distance,
                        abs(offset.dot(right)) / tan_horizontal - depth,
                        abs(offset.dot(camera_up)) / tan_vertical - depth,
                    )
        distance = max(padding * distance, radius * 1.0e-3, 1.0e-6)

        self.camera.target = center
        self.camera.set_orbit(distance=distance)
        # Near-plane clipping keeps straddling faces on screen, so the near
        # plane can sit close in and geometry survives zooming and panning.
        self.camera.near = max(radius * 1.0e-4, 1.0e-6)
        self.camera.far = max(distance + 10.0 * radius, self.camera.near + 1.0)

        if redraw:
            self._interactive_render = False
            self._request_redraw()

    def _scene_bounds(self) -> Optional[Tuple[Point3D, Point3D]]:
        points: List[Point3D] = []

        for obj in self.objects:
            object_type = obj.get("type")
            if object_type == "line":
                points.extend((obj["start"], obj["end"]))
            elif object_type == "text":
                points.append(obj["point"])
            elif object_type == "polygon":
                points.extend(obj.get("vertices", []))
            elif object_type == "mesh":
                mesh_points = obj.get("points", [])
                if mesh_points:
                    xs = [point.x for point in mesh_points]
                    ys = [point.y for point in mesh_points]
                    zs = [point.z for point in mesh_points]
                    points.append(Point3D(min(xs), min(ys), min(zs)))
                    points.append(Point3D(max(xs), max(ys), max(zs)))
            elif object_type == "mesh_arrays":
                handle = obj.get("handle")
                if isinstance(handle, MeshHandle) and not handle.removed and handle.visible:
                    for _chunk_id, mesh in [(None, handle.mesh), *handle.chunks]:
                        positions = mesh.positions
                        if mesh.displacements is not None and handle.deformation_scale:
                            positions = positions + handle.deformation_scale * mesh.displacements
                        if not len(positions):
                            continue
                        transform = handle.transform
                        positions = positions @ transform[:3, :3].T + transform[:3, 3]
                        low = positions.min(axis=0)
                        high = positions.max(axis=0)
                        points.append(Point3D(*low))
                        points.append(Point3D(*high))
            elif object_type == "faces":
                vertices = obj.get("vertices")
                if vertices is not None and len(vertices):
                    low = vertices.min(axis=0)
                    high = vertices.max(axis=0)
                    points.append(Point3D(*low))
                    points.append(Point3D(*high))
            elif object_type == "markers":
                marker_points = obj.get("points", [])
                if marker_points:
                    xs = [point.x for point in marker_points]
                    ys = [point.y for point in marker_points]
                    zs = [point.z for point in marker_points]
                    points.append(Point3D(min(xs), min(ys), min(zs)))
                    points.append(Point3D(max(xs), max(ys), max(zs)))
            elif object_type == "cylinder":
                center = obj.get("center", Point3D(0.0, 0.0, 0.0))
                radius = max(
                    float(obj.get("radius", 1.0)),
                    float(obj.get("radius_top") or obj.get("radius", 1.0)),
                )
                half_height = 0.5 * float(obj.get("height", 1.0))
                points.extend(
                    (
                        Point3D(center.x - radius, center.y - radius, center.z - half_height),
                        Point3D(center.x + radius, center.y + radius, center.z + half_height),
                    )
                )
            elif object_type == "stiffener":
                radius = float(obj.get("radius", 1.0))
                web_height = float(obj.get("web_height", 0.0))
                flange_thickness = float(obj.get("flange_thickness", 0.0))
                inside = bool(obj.get("inside", False))
                outer_radius = (
                    radius
                    if inside
                    else radius + web_height + flange_thickness
                )
                if obj.get("stiffener_type") == "ring":
                    z_position = float(obj.get("z_position", 0.0))
                    half_width = 0.5 * max(
                        float(obj.get("web_thickness", 0.0)),
                        float(obj.get("flange_width", 0.0)),
                    )
                    points.extend(
                        (
                            Point3D(-outer_radius, -outer_radius, z_position - half_width),
                            Point3D(outer_radius, outer_radius, z_position + half_width),
                        )
                    )
                else:
                    half_height = 0.5 * float(obj.get("height", 1.0))
                    z_offset = float(obj.get("z_offset", 0.0))
                    points.extend(
                        (
                            Point3D(-outer_radius, -outer_radius, -half_height + z_offset),
                            Point3D(outer_radius, outer_radius, half_height + z_offset),
                        )
                    )

        if not points:
            return None

        minimum = Point3D(
            min(point.x for point in points),
            min(point.y for point in points),
            min(point.z for point in points),
        )
        maximum = Point3D(
            max(point.x for point in points),
            max(point.y for point in points),
            max(point.z for point in points),
        )
        return minimum, maximum


def populate_stiffened_cylinder(canvas_3d: Tkinter3DCanvas) -> None:
    """Populate a canvas with open-ended internal stiffening."""
    cylinder_radius = 2.0
    cylinder_height = 4.0

    canvas_3d.add_cylinder(
        radius=cylinder_radius,
        height=cylinder_height,
        center=Point3D(0.0, 0.0, 0.0),
        color="#d8e2ea",
        outline="#708090",
        segments=48,
        height_segments=24,
        capped=False,
        opacity=0.38,
        show_backfaces=True,
        # Four axial shell strakes, ordered from bottom to top.
        plate_thickness=[18.0, 16.0, 14.0, 12.0],
        thickness_unit="mm",
        thickness_legend_title="Plate thickness",
        show_thickness_legend=True,
    )

    number_of_longitudinals = 8
    for index in range(number_of_longitudinals):
        angle = 2.0 * math.pi * index / number_of_longitudinals
        canvas_3d.add_longitudinal_stiffener(
            radius=cylinder_radius,
            height=cylinder_height,
            angle=angle,
            web_height=0.15,
            web_thickness=0.01,
            flange_width=0.10,
            flange_thickness=0.02,
            color="#a0a0ff",
            outline="#404080",
            segments=4,
            height_segments=16,
            inside=True,
        )

    number_of_rings = 4
    for index in range(number_of_rings):
        z_position = (
            -cylinder_height / 2.0
            + (index + 1) * cylinder_height / (number_of_rings + 1)
        )
        canvas_3d.add_ring_stiffener(
            radius=cylinder_radius,
            z_position=z_position,
            web_height=0.12,
            web_thickness=0.02,
            flange_width=0.08,
            flange_thickness=0.015,
            color="#ffa0a0",
            outline="#804040",
            segments=48,
            inside=True,
        )

    canvas_3d.after_idle(canvas_3d.fit_to_scene)


def populate_stiffened_plate(canvas_3d: Tkinter3DCanvas) -> None:
    length = 4.0
    width = 4.0

    # Base plate
    canvas_3d.add_rectangular_plate(
        x_start=-length/2, x_end=length/2,
        y_start=-width/2, y_end=width/2,
        z=0.0,
        color="#d8e2ea",
        outline="#708090",
        stipple="gray50",
    )

    # Stiffeners along X
    num_stiffeners = 5
    for k in range(num_stiffeners):
        y = -width/2 + (k + 1) * width / (num_stiffeners + 1)
        canvas_3d.add_flat_stiffener(
            x_start=-length/2, x_end=length/2,
            y=y,
            z_base=0.0,
            hw=0.15,
            b=0.10,
            color="#a0a0ff",
            outline="#404080",
        )

    # Girders along Y
    num_girders = 1
    for k in range(num_girders):
        x = 0.0
        canvas_3d.add_flat_girder(
            x=x,
            y_start=-width/2, y_end=width/2,
            z_base=0.0,
            ghw=0.3,
            gb=0.2,
            color="#ffa0a0",
            outline="#804040",
        )
    canvas_3d.after_idle(canvas_3d.fit_to_scene)


def populate_fe_gui_cylinder(canvas_3d: Tkinter3DCanvas) -> None:
    cylinder_radius = 2.0
    cylinder_height = 4.0

    canvas_3d.add_cylinder(
        radius=cylinder_radius,
        height=cylinder_height,
        center=Point3D(0.0, 0.0, 0.0),
        color="#1f77b4",
        outline="black",
        segments=48,
        height_segments=24,
        capped=False,
        opacity=0.78,
        show_backfaces=True,
    )

    number_of_longitudinals = 8
    for index in range(number_of_longitudinals):
        angle = 2.0 * math.pi * index / number_of_longitudinals
        canvas_3d.add_longitudinal_stiffener(
            radius=cylinder_radius,
            height=cylinder_height,
            angle=angle,
            web_height=0.15,
            web_thickness=0.01,
            flange_width=0.10,
            flange_thickness=0.02,
            color="#2ca02c",
            outline="black",
            segments=4,
            height_segments=16,
            inside=True,
        )

    number_of_rings = 4
    for index in range(number_of_rings):
        z_position = (
            -cylinder_height / 2.0
            + (index + 1) * cylinder_height / (number_of_rings + 1)
        )
        canvas_3d.add_ring_stiffener(
            radius=cylinder_radius,
            z_position=z_position,
            web_height=0.12,
            web_thickness=0.02,
            flange_width=0.08,
            flange_thickness=0.015,
            color="#d62728",
            outline="black",
            segments=48,
            inside=True,
        )
    canvas_3d.after_idle(canvas_3d.fit_to_scene)


def populate_fe_gui_plate(canvas_3d: Tkinter3DCanvas) -> None:
    length = 4.0
    width = 4.0

    # Base plate
    canvas_3d.add_rectangular_plate(
        x_start=-length/2, x_end=length/2,
        y_start=-width/2, y_end=width/2,
        z=0.0,
        color="#1f77b4",
        outline="black",
        stipple="gray75",
    )

    # Stiffeners along X
    num_stiffeners = 5
    for k in range(num_stiffeners):
        y = -width/2 + (k + 1) * width / (num_stiffeners + 1)
        canvas_3d.add_flat_stiffener(
            x_start=-length/2, x_end=length/2,
            y=y,
            z_base=0.0,
            hw=0.15,
            b=0.10,
            color="#2ca02c",
            outline="black",
        )

    # Girders along Y
    num_girders = 1
    for k in range(num_girders):
        x = 0.0
        canvas_3d.add_flat_girder(
            x=x,
            y_start=-width/2, y_end=width/2,
            z_base=0.0,
            ghw=0.3,
            gb=0.2,
            color="#d62728",
            outline="black",
        )
    canvas_3d.after_idle(canvas_3d.fit_to_scene)


def populate_shape_gallery(canvas_3d: Tkinter3DCanvas) -> None:
    """Show the generic shape library with lighting and transparency."""
    canvas_3d.set_mesh_lines(False)
    canvas_3d.add_grid(
        size_x=15.0, size_y=11.0, step=1.0, z=0.0,
        center=Point3D(0.0, 0.6, 0.0), color="#dde4ec",
    )

    spacing = 3.0
    row_y = 4.5
    for column, (name, place) in enumerate(
        (
            ("box", lambda p: canvas_3d.add_box(1.6, 1.6, 1.6, center=p(0.8), color="#4e79a7")),
            ("sphere", lambda p: canvas_3d.add_sphere(0.9, center=p(0.9), color="#f28e2b")),
            ("cone", lambda p: canvas_3d.add_cone(0.9, 1.8, center=p(0.9), color="#e15759")),
            ("frustum", lambda p: canvas_3d.add_frustum(0.95, 0.45, 1.7, center=p(0.85), color="#76b7b2")),
            ("pyramid", lambda p: canvas_3d.add_pyramid(1.0, 1.7, center=p(0.85), color="#59a14f")),
        )
    ):
        x = (column - 2) * spacing
        place(lambda z, x=x: Point3D(x, row_y, z))
        canvas_3d.add_text(Point3D(x, row_y, -0.5), name, color="#475569")

    row_y = 1.0
    for column, (name, place) in enumerate(
        (
            ("tube", lambda p: canvas_3d.add_tube(0.95, 0.6, 1.7, center=p(0.85), color="#edc948")),
            ("torus", lambda p: canvas_3d.add_torus(0.8, 0.27, center=p(0.6), color="#b07aa1")),
            ("wedge", lambda p: canvas_3d.add_wedge(1.7, 1.4, 1.4, center=p(0.7), color="#ff9da7")),
            (
                "prism (I)",
                lambda p: canvas_3d.add_prism(
                    shapes_module.profile_section("I", 1.4, 0.14, 0.8, 0.14),
                    1.7,
                    center=p(0.0),
                    axis=Point3D(0.0, 1.0, 0.0),
                    color="#9c755f",
                ),
            ),
            ("annulus", lambda p: canvas_3d.add_disk(1.0, 0.5, center=p(0.02), color="#8d99ae")),
        )
    ):
        x = (column - 2) * spacing
        place(lambda z, x=x: Point3D(x, row_y, z))
        canvas_3d.add_text(Point3D(x, row_y, -0.5), name, color="#475569")

    # A transparent shell over solid contents: the layered stipple windows keep
    # the far wall, the contents and the near wall all readable at once.
    canvas_3d.add_sphere(
        1.6,
        center=Point3D(-4.2, -3.4, 1.7),
        segments=30,
        rings=20,
        color="#86bcd8",
        opacity=0.35,
    )
    canvas_3d.add_box(1.1, 1.1, 1.1, center=Point3D(-4.2, -3.4, 1.7), color="#c0392b")
    canvas_3d.add_text(Point3D(-4.2, -3.4, -0.5), "transparent shell", color="#475569")

    canvas_3d.add_beam(
        Point3D(-1.4, -3.4, 0.0),
        Point3D(5.5, -3.4, 0.0),
        kind="I",
        web_height=1.0,
        web_thickness=0.07,
        flange_width=0.55,
        flange_thickness=0.09,
        up=Point3D(0.0, 0.0, 1.0),
        color="#7f8c9a",
    )
    for x in (-1.0, 2.05, 5.1):
        canvas_3d.add_arrow(Point3D(x, -3.4, 3.0), Point3D(x, -3.4, 1.2))
    canvas_3d.add_text(Point3D(2.05, -3.4, -0.5), "beam + arrows", color="#475569")

    canvas_3d.after_idle(canvas_3d.fit_to_scene)


def _add_controls(parent: tk.Misc, canvas_3d: Tkinter3DCanvas) -> tk.Frame:
    controls = tk.Frame(parent)
    controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 4))

    tk.Button(controls, text="Fit", command=canvas_3d.fit_to_scene).pack(side=tk.LEFT, padx=3)
    tk.Button(controls, text="Reset", command=canvas_3d.reset_camera).pack(side=tk.LEFT, padx=3)
    tk.Button(controls, text="Top", command=canvas_3d.set_top_view).pack(side=tk.LEFT, padx=3)
    tk.Button(controls, text="Side", command=canvas_3d.set_side_view).pack(side=tk.LEFT, padx=3)
    tk.Button(controls, text="Front", command=canvas_3d.set_front_view).pack(side=tk.LEFT, padx=3)
    tk.Button(controls, text="Iso", command=canvas_3d.set_iso_view).pack(side=tk.LEFT, padx=3)

    shading = tk.BooleanVar(value=True)

    def toggle_shading() -> None:
        canvas_3d.set_shading(shading.get())

    tk.Checkbutton(
        controls, text="Light", variable=shading, command=toggle_shading
    ).pack(side=tk.LEFT, padx=(10, 3))

    tk.Label(
        controls,
        text="Right-drag: rotate | Left-drag: move | Wheel: zoom",
    ).pack(side=tk.RIGHT, padx=6)
    return controls


def _create_viewport(parent: tk.Misc, title: str, populate_func: Any) -> tk.Frame:
    frame = tk.Frame(parent, bd=2, relief=tk.GROOVE)

    lbl = tk.Label(frame, text=title, font=("TkDefaultFont", 10, "bold"))
    lbl.pack(side=tk.TOP, fill=tk.X, pady=2)

    canvas_3d = Tkinter3DCanvas(frame, width=400, height=300, bg="white")
    _add_controls(frame, canvas_3d)
    canvas_3d.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

    populate_func(canvas_3d)
    frame.canvas_3d = canvas_3d  # type: ignore[attr-defined]
    return frame


_DEMO_VIEWPORTS = (
    ("Shape gallery (lit)", populate_shape_gallery),
    ("Present cylinder", populate_stiffened_cylinder),
    ("Cylinder (fe-gui style)", populate_fe_gui_cylinder),
    ("Stiffened plate (fe-gui style)", populate_fe_gui_plate),
)


def _build_demo(container: tk.Misc) -> None:
    container.rowconfigure(0, weight=1)
    container.rowconfigure(1, weight=1)
    container.columnconfigure(0, weight=1)
    container.columnconfigure(1, weight=1)

    viewports = []
    for index, (title, populate) in enumerate(_DEMO_VIEWPORTS):
        viewport = _create_viewport(container, title, populate)
        viewport.grid(row=index // 2, column=index % 2, sticky="nsew", padx=4, pady=4)
        viewports.append(viewport)

    # Fit once the grid has settled: framing computed against a viewport's
    # initial requested size would be wrong for its final one.
    def fit_all() -> None:
        for frame in viewports:
            frame.canvas_3d.fit_to_scene()

    container.after(150, fit_all)


def create_stiffened_cylinder_demo(root: tk.Misc) -> tk.Toplevel:
    """Open the demonstration in a child window."""
    demo_window = tk.Toplevel(root)
    demo_window.title("Tkinter 3D - Four Viewports Demo")
    demo_window.geometry("1400x1000")
    demo_window.minsize(800, 600)
    _build_demo(demo_window)
    return demo_window


def main() -> None:
    """Run the four-viewport demonstration application."""
    root = tk.Tk()
    root.title("Tkinter 3D - Four Viewports Demo")
    root.geometry("1400x1000")
    root.minsize(800, 600)
    _build_demo(root)
    root.mainloop()


if __name__ == "__main__":
    main()
