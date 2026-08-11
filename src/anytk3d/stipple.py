"""
Screen-door transparency patterns for the Tk canvas.

A Tk canvas has no alpha channel: the only way to see through a filled
polygon is a stipple bitmap, and Tk ships just four densities (gray12,
gray25, gray50, gray75).  Two problems follow when a scene stacks several
transparent surfaces:

* four densities are a coarse ladder for an opacity setting, and
* two polygons drawn with the *same* stipple cover exactly the same pixels,
  so the one behind contributes nothing and simply disappears.

This module generates 8x8 ordered-dither (Bayer) bitmaps on demand.  Every
pattern is a *window* into the 64-step Bayer ordering, which makes it easy
to hand successive layers of a stack windows that do not overlap.

The window widths follow ordinary alpha compositing.  Painting a surface of
opacity ``a`` over one already at ``a`` should end up covering
``1 - (1 - a)^2`` of the pixels, so layer 0 gets ``a`` of the tile, layer 1
gets ``a * (1 - a)`` of what is left, and so on.  A 38% shell therefore
shows its far wall through its near wall and still leaves 39% of the
background visible, instead of either hiding the far wall completely or
going fully opaque.

A ``rotation`` shifts the whole window set, which decorrelates unrelated
objects that happen to overlap.

Bitmaps are written once into a cache directory as X bitmap (``.xbm``)
files, the format Tk's ``-stipple`` option understands.  If the files
cannot be written the module falls back to Tk's built-in stipples, so
rendering never depends on a writable temp directory.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Dict, Optional, Tuple


#: Pixels per bitmap tile edge.
TILE = 8

#: Distinct window rotations used to decorrelate overlapping objects.
ROTATIONS = 4

# Classic 8x8 Bayer threshold matrix, values 0..63.
_BAYER = (
    (0, 32, 8, 40, 2, 34, 10, 42),
    (48, 16, 56, 24, 50, 18, 58, 26),
    (12, 44, 4, 36, 14, 46, 6, 38),
    (60, 28, 52, 20, 62, 30, 54, 22),
    (3, 35, 11, 43, 1, 33, 9, 41),
    (51, 19, 59, 27, 49, 17, 57, 25),
    (15, 47, 7, 39, 13, 45, 5, 37),
    (63, 31, 55, 23, 61, 29, 53, 21),
)

_CELLS = TILE * TILE

#: Opacity at or above this renders as a solid fill.
OPAQUE_THRESHOLD = 0.97

# Fallback ladder when custom bitmaps are unavailable.
_BUILTIN = ((0.20, "gray12"), (0.38, "gray25"), (0.65, "gray50"), (0.90, "gray75"))

_cache: Dict[Tuple[int, int, int], str] = {}
_cache_directory: Optional[str] = None
_generation_failed = False


def _bitmap_directory() -> Optional[str]:
    global _cache_directory, _generation_failed
    if _generation_failed:
        return None
    if _cache_directory is not None:
        return _cache_directory
    try:
        directory = os.path.join(tempfile.gettempdir(), "anytk3d-stipple")
        os.makedirs(directory, exist_ok=True)
    except OSError:
        _generation_failed = True
        return None
    _cache_directory = directory
    return directory


def _pattern_rows(start: int, count: int, rotation: int) -> Tuple[int, ...]:
    """One byte per row, least significant bit leftmost (the XBM convention)."""
    rows = []
    for y in range(TILE):
        bits = 0
        for x in range(TILE):
            threshold = (_BAYER[y][x] - rotation) % _CELLS
            if start <= threshold < start + count:
                bits |= 1 << x
        rows.append(bits)
    return tuple(rows)


def _write_bitmap(path: str, name: str, rows: Tuple[int, ...]) -> None:
    """
    Write an X bitmap file, then confirm it can be read back.

    Tk opens the file itself the first time a stipple is used.  Writing to a
    sibling temp file and renaming into place means Tk can only ever see a
    complete file, and the read-back settles the case where a virus scanner
    still holds a brand-new file open.
    """
    body = ", ".join(f"0x{value:02x}" for value in rows)
    content = (
        f"#define {name}_width {TILE}\n"
        f"#define {name}_height {TILE}\n"
        f"static unsigned char {name}_bits[] = {{\n {body}\n}};\n"
    )
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="ascii") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

    for attempt in range(3):
        try:
            with open(path, "r", encoding="ascii") as handle:
                if handle.read() == content:
                    return
        except OSError:
            pass
        time.sleep(0.01 * (attempt + 1))
    raise OSError(f"could not verify generated stipple {path!r}")


def disable_generated() -> None:
    """
    Stop using generated bitmaps and fall back to Tk's built-in stipples.

    The renderer calls this if Tk ever refuses a generated bitmap, so a
    hostile filesystem degrades the transparency ladder instead of breaking
    the drawing.
    """
    global _generation_failed
    _generation_failed = True
    _cache.clear()


def _builtin_for(coverage: float) -> str:
    for threshold, name in _BUILTIN:
        if coverage < threshold:
            return name
    return ""


def window(opacity: float, layer: int = 0) -> Tuple[int, int]:
    """
    Return ``(start, count)`` cells for ``layer`` of a stack at ``opacity``.

    Layer 0 takes the first ``opacity`` of the tile; each further layer takes
    the same fraction of whatever is still uncovered, and starts where the
    previous layer stopped.  Windows therefore never overlap and their union
    approaches - but never exceeds - the full tile.
    """
    opacity = max(0.0, min(1.0, float(opacity)))
    covered = 0.0
    remaining = 1.0
    for _ in range(max(0, int(layer))):
        step = opacity * remaining
        covered += step
        remaining -= step

    start = int(round(covered * _CELLS))
    end = int(round((covered + opacity * remaining) * _CELLS))
    start = max(0, min(_CELLS - 1, start))
    count = max(1, min(_CELLS - start, end - start))
    return start, count


def pattern(start: int, count: int, rotation: int = 0) -> str:
    """Tk stipple specification for one Bayer window."""
    start = max(0, min(_CELLS - 1, int(start)))
    count = int(count)
    if count >= _CELLS:
        return ""
    count = max(1, min(_CELLS - start, count))
    rotation = (int(rotation) % ROTATIONS) * (_CELLS // ROTATIONS)

    key = (start, count, rotation)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    directory = _bitmap_directory()
    if directory is None:
        return _builtin_for(count / _CELLS)

    name = f"anytk3d_{start:02d}_{count:02d}_{rotation:02d}"
    path = os.path.join(directory, name + ".xbm")
    try:
        _write_bitmap(path, name, _pattern_rows(start, count, rotation))
    except OSError:
        return _builtin_for(count / _CELLS)

    # Tcl paths use forward slashes on every platform.
    specification = "@" + path.replace("\\", "/")
    _cache[key] = specification
    return specification


def for_opacity(opacity: float, layer: int = 0, rotation: int = 0) -> str:
    """
    Map an opacity to a stipple specification.

    ``layer`` selects which surface of a stack this is: 0 for the face
    nearest the viewer, 1 for the one behind it.  ``rotation`` shifts the
    windows so that unrelated overlapping objects do not line up.
    """
    opacity = max(0.0, min(1.0, float(opacity)))
    if opacity >= OPAQUE_THRESHOLD:
        return ""
    start, count = window(opacity, layer)
    return pattern(start, count, rotation)


def available() -> bool:
    """True when generated bitmaps (rather than Tk built-ins) are in use."""
    return _bitmap_directory() is not None
