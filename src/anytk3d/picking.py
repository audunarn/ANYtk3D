"""Picking and highlighting support for :class:`Tkinter3DCanvas`.

Every face and line the canvas draws can carry caller-supplied ``tags``, which
reach the underlying Tk canvas item.  That makes picking almost free: Tk's own
hit testing already knows which items lie under the cursor, and because the
renderer paints back to front, the topmost item under the cursor *is* the
nearest one.  No ray casting is required for ordinary clicks.

The application decides what a tag means.  This module only carries it back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, Iterable, Optional, Sequence, Tuple

__all__ = ["Pick", "PickState", "entity_tag_at"]

# Highlight colours chosen to read against both light and dark backgrounds.
DEFAULT_HIGHLIGHT_FILL = "#ff9d2e"
DEFAULT_HIGHLIGHT_OUTLINE = "#b35c00"

# How far from the cursor a hit still counts, in pixels.  A couple of pixels of
# slack makes thin lines and small markers pickable without a steady hand.
DEFAULT_PICK_RADIUS = 3


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
        faces = frozenset(
            index
            for index, tag_string in enumerate(scene.tags)
            if tag_string and not wanted.isdisjoint(tag_string.split())
        )
        self._cache = (scene, self.generation, faces)
        return faces

    def invalidate(self) -> None:
        """Drop the cached face resolution, e.g. when the scene is rebuilt."""

        self._cache = None


def modifiers_from_event(event: Any) -> Tuple[bool, bool, bool]:
    """Extract shift / control / alt from a Tk event state mask."""

    state = int(getattr(event, "state", 0) or 0)
    return bool(state & 0x0001), bool(state & 0x0004), bool(state & 0x20000)
