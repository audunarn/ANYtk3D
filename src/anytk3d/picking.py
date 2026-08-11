"""Picking and highlighting support for :class:`Tkinter3DCanvas`.

Every face and line the canvas draws can carry caller-supplied ``tags``, which
reach the underlying Tk canvas item.  That makes picking almost free: Tk's own
hit testing already knows which items lie under the cursor, and because the
renderer paints back to front, the topmost item under the cursor *is* the
nearest one.  No ray casting is required for ordinary clicks.

The application decides what a tag means.  This module only carries it back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, FrozenSet, Iterable, Optional, Sequence, Tuple

__all__ = [
    "Pick",
    "PickBinding",
    "PickOwner",
    "SelectionConfig",
    "SelectionDepth",
    "SelectionEvent",
    "SelectionFilter",
    "SelectionGesture",
    "SelectionHit",
    "SelectionOperation",
    "SelectionTool",
    "entity_tag_at",
]

# Highlight colours chosen to read against both light and dark backgrounds.
DEFAULT_HIGHLIGHT_FILL = "#ff9d2e"
DEFAULT_HIGHLIGHT_OUTLINE = "#b35c00"

# How far from the cursor a hit still counts, in pixels.  A couple of pixels of
# slack makes thin lines and small markers pickable without a steady hand.
DEFAULT_PICK_RADIUS = 3


class SelectionDepth(str, Enum):
    """Whether a region query stops at the visible surface."""

    VISIBLE = "visible"
    THROUGH = "through"


class SelectionTool(str, Enum):
    """Shape made by an LMB drag in the commercial interaction profile."""

    SINGLE = "single"
    BOX = "box"
    LASSO = "lasso"


class SelectionOperation(str, Enum):
    """How an application should combine a gesture with its selection."""

    REPLACE = "replace"
    ADD = "add"
    REMOVE = "remove"
    TOGGLE = "toggle"


class SelectionGesture(str, Enum):
    CLICK = "click"
    WINDOW = "window"
    CROSSING = "crossing"
    LASSO = "lasso"


@dataclass(frozen=True)
class PickOwner:
    """One semantic object represented by a rendered primitive.

    ``kind`` is intentionally an application-owned string.  A finite-element
    application can therefore use ``geometry.face`` and ``mesh.element``
    without making ANYtk3D depend on either package.
    """

    key: str
    kind: str = ""
    priority: int = 0

    def __post_init__(self) -> None:
        if not str(self.key):
            raise ValueError("a pick owner needs a non-empty key")
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "priority", int(self.priority))


@dataclass(frozen=True)
class PickBinding:
    """Semantic owners attached to one face, line, or marker.

    More than one owner lets a mesh polygon be selected either as its parent
    geometry face or as its individual finite element without rebuilding the
    scene.
    """

    owners: Tuple[PickOwner, ...]

    def __post_init__(self) -> None:
        cleaned = tuple(
            owner if isinstance(owner, PickOwner) else PickOwner(*owner)
            for owner in self.owners
        )
        if not cleaned:
            raise ValueError("a pick binding needs at least one owner")
        if len({owner.key for owner in cleaned}) != len(cleaned):
            raise ValueError("owner keys in one pick binding must be unique")
        object.__setattr__(self, "owners", cleaned)

    @classmethod
    def one(cls, key: str, kind: str = "", priority: int = 0) -> "PickBinding":
        return cls((PickOwner(key, kind, priority),))


@dataclass(frozen=True)
class SelectionFilter:
    """Restrict queries by semantic kind and/or stable-key prefix."""

    kinds: FrozenSet[str] = frozenset()
    key_prefixes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kinds", frozenset(str(value) for value in self.kinds))
        object.__setattr__(
            self, "key_prefixes", tuple(str(value) for value in self.key_prefixes)
        )

    def accepts(self, owner: PickOwner) -> bool:
        if self.kinds and owner.kind not in self.kinds:
            return False
        if self.key_prefixes and not any(
            owner.key.startswith(prefix) for prefix in self.key_prefixes
        ):
            return False
        return True


@dataclass(frozen=True)
class SelectionConfig:
    """Interaction and query policy for the commercial selection profile."""

    filter: SelectionFilter = field(default_factory=SelectionFilter)
    depth: SelectionDepth = SelectionDepth.VISIBLE
    tool: SelectionTool = SelectionTool.BOX
    directional: bool = True
    drag_threshold_px: int = 4
    click_radius_px: int = 4
    cycle_radius_px: int = 5
    cycle_timeout_ms: int = 1500
    click_on_press: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.filter, SelectionFilter):
            raise TypeError("filter must be a SelectionFilter")
        object.__setattr__(self, "depth", SelectionDepth(self.depth))
        object.__setattr__(self, "tool", SelectionTool(self.tool))
        object.__setattr__(self, "click_on_press", bool(self.click_on_press))
        for name, minimum in (
            ("drag_threshold_px", 1),
            ("click_radius_px", 0),
            ("cycle_radius_px", 0),
            ("cycle_timeout_ms", 0),
        ):
            object.__setattr__(self, name, max(minimum, int(getattr(self, name))))


@dataclass(frozen=True)
class SelectionHit:
    """One semantic owner hit by a point or region query."""

    owner: PickOwner
    primitive: int
    depth: float
    screen_distance: float = 0.0
    visible: bool = True
    item: int = -1

    @property
    def key(self) -> str:
        return self.owner.key

    @property
    def kind(self) -> str:
        return self.owner.kind


@dataclass(frozen=True)
class SelectionEvent:
    """Completed click, box, or lasso gesture."""

    gesture: SelectionGesture
    operation: SelectionOperation
    hits: Tuple[SelectionHit, ...] = ()
    candidates: Tuple[SelectionHit, ...] = ()
    start: Tuple[int, int] = (0, 0)
    end: Tuple[int, int] = (0, 0)
    points: Tuple[Tuple[int, int], ...] = ()
    cycle_index: int = 0
    cycle_total: int = 0


@dataclass(frozen=True)
class Pick:
    """One hit returned by a click or a hover."""

    tag: str
    item: int
    x: int
    y: int
    shift: bool = False
    ctrl: bool = False
    alt: bool = False


def entity_tag_at(
    widget: Any,
    x: int,
    y: int,
    *,
    prefix: str = "",
    reserved: Iterable[str] = (),
    radius: int = DEFAULT_PICK_RADIUS,
) -> Tuple[Optional[str], Optional[int]]:
    """Return the topmost caller tag under ``(x, y)``, and its canvas item.

    ``prefix`` restricts the search to tags the caller owns; ``reserved`` names
    the renderer's own pool tags, which are never returned.
    """

    reserved_tags = set(reserved)
    try:
        items = widget.find_overlapping(
            x - radius, y - radius, x + radius, y + radius
        )
    except Exception:  # pragma: no cover - a destroyed widget during teardown
        return None, None

    # Tk returns items in stacking order, lowest first.  The renderer paints
    # back to front, so the last item is the front-most one.
    for item in reversed(items):
        for tag in widget.gettags(item):
            if tag in reserved_tags or tag == "current":
                continue
            if prefix and not tag.startswith(prefix):
                continue
            return tag, item
    return None, None


class PickState:
    """Picking, hover and highlight state held by one canvas."""

    def __init__(self) -> None:
        self.pick_callback: Optional[Callable[[Pick], None]] = None
        self.hover_callback: Optional[Callable[[Optional[Pick]], None]] = None
        self.prefix: str = ""
        self.radius: int = DEFAULT_PICK_RADIUS
        self.highlight_tags: FrozenSet[str] = frozenset()
        self.highlight_fill: str = DEFAULT_HIGHLIGHT_FILL
        self.highlight_outline: str = DEFAULT_HIGHLIGHT_OUTLINE
        self.generation: int = 0
        self.press: Optional[Tuple[int, int]] = None
        self.hover_tag: Optional[str] = None
        self._cache: Optional[Tuple[Any, int, FrozenSet[int]]] = None
        self.preselection_key: Optional[str] = None
        self.preselection_fill: str = "#ffd166"
        self.preselection_outline: str = "#b77900"
        self._preselection_cache: Optional[Tuple[Any, str, FrozenSet[int]]] = None

    # ------------------------------------------------------------------
    def set_highlight(
        self,
        tags: Sequence[str] | Iterable[str],
        fill: Optional[str] = None,
        outline: Optional[str] = None,
    ) -> bool:
        """Replace the highlighted tag set.  True when anything changed."""

        new_tags = frozenset(str(tag) for tag in tags)
        new_fill = self.highlight_fill if fill is None else str(fill)
        new_outline = self.highlight_outline if outline is None else str(outline)
        if (
            new_tags == self.highlight_tags
            and new_fill == self.highlight_fill
            and new_outline == self.highlight_outline
        ):
            return False
        self.highlight_tags = new_tags
        self.highlight_fill = new_fill
        self.highlight_outline = new_outline
        self.generation += 1
        self._cache = None
        return True

    def highlighted_faces(self, scene: Any) -> Optional[FrozenSet[int]]:
        """Face indices of the compiled scene that carry a highlighted tag.

        ``None`` means nothing is highlighted, which lets the render loop skip
        the check entirely.  The result is cached against the scene identity
        and the highlight generation, so orbiting a highlighted model does not
        re-split every tag string on every frame.
        """

        if not self.highlight_tags or not getattr(scene, "any_tags", False):
            return None

        cached = self._cache
        if cached is not None and cached[0] is scene and cached[1] == self.generation:
            return cached[2]

        wanted = self.highlight_tags
        bindings = getattr(scene, "face_bindings", ())
        faces = frozenset(
            index
            for index, tag_string in enumerate(scene.tags)
            if (
                tag_string and not wanted.isdisjoint(tag_string.split())
            ) or (
                index < len(bindings)
                and bindings[index] is not None
                and any(owner.key in wanted for owner in bindings[index].owners)
            )
        )
        self._cache = (scene, self.generation, faces)
        return faces

    def set_preselection(self, key: Optional[str]) -> bool:
        key = None if not key else str(key)
        if key == self.preselection_key:
            return False
        self.preselection_key = key
        self._preselection_cache = None
        return True

    def preselected_faces(self, scene: Any) -> Optional[FrozenSet[int]]:
        key = self.preselection_key
        if not key:
            return None
        cached = self._preselection_cache
        if cached is not None and cached[0] is scene and cached[1] == key:
            return cached[2]
        bindings = getattr(scene, "face_bindings", ())
        faces = frozenset(
            index
            for index, tag_string in enumerate(scene.tags)
            if key in tag_string.split()
            or (
                index < len(bindings)
                and bindings[index] is not None
                and any(owner.key == key for owner in bindings[index].owners)
            )
        )
        self._preselection_cache = (scene, key, faces)
        return faces

    def invalidate(self) -> None:
        """Drop the cached face resolution, e.g. when the scene is rebuilt."""

        self._cache = None
        self._preselection_cache = None


def modifiers_from_event(event: Any) -> Tuple[bool, bool, bool]:
    """Extract shift / control / alt from a Tk event state mask."""

    state = int(getattr(event, "state", 0) or 0)
    # Mod1 is Alt on X11; Windows uses the high bit.  Key tracking in the
    # commercial controller covers platforms where Option/Alt is not present
    # in the mouse event's state mask.
    return (
        bool(state & 0x0001),
        bool(state & 0x0004),
        bool(state & (0x0008 | 0x20000)),
    )


def operation_from_modifiers(
    shift: bool, ctrl: bool, alt: bool
) -> SelectionOperation:
    """Resolve combined modifiers with Alt > Ctrl > Shift precedence."""

    if alt:
        return SelectionOperation.REMOVE
    if ctrl:
        return SelectionOperation.TOGGLE
    if shift:
        return SelectionOperation.ADD
    return SelectionOperation.REPLACE


def fallback_binding(tags: str) -> Optional[PickBinding]:
    """Create a legacy binding from the first caller-supplied Tk tag."""

    for tag in str(tags or "").split():
        if tag and tag != "current" and not tag.startswith("_tk3d_"):
            return PickBinding.one(tag)
    return None
