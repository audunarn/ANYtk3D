"""Rasterize viewer canvas items without reading pixels from the desktop."""

from __future__ import annotations

import math
from pathlib import Path
import tkinter.font as tkfont


def _font(canvas, item, scale):
    from PIL import ImageFont

    font = tkfont.Font(root=canvas, font=canvas.itemcget(item, "font"))
    actual = font.actual()
    size = int(actual["size"])
    pixels = abs(size) if size < 0 else size * float(canvas.winfo_fpixels("1i")) / 72
    family = str(actual["family"]).lower()
    names = {
        "arial": "arial", "segoe ui": "segoeui", "tahoma": "tahoma",
        "times new roman": "times", "courier new": "cour", "calibri": "calibri",
    }
    stem = names.get(family, family.replace(" ", ""))
    bold = actual["weight"] == "bold"
    italic = actual["slant"] == "italic"
    suffix = ("bi" if italic else "bd") if bold else ("i" if italic else "")
    candidates = [stem + suffix + ".ttf", stem + ".ttf"]
    fallback = "DejaVuSans" + ("-BoldOblique" if bold and italic else "-Bold" if bold else "-Oblique" if italic else "")
    candidates += [fallback + ".ttf", "DejaVuSans.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, max(1, round(pixels * scale)))
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=max(1, round(pixels * scale)))
    except TypeError:  # Pillow 10.0 predates the scalable default font.
        return ImageFont.load_default()


def _stipple(name, size, scale):
    from PIL import Image
    from .stipple import _BAYER

    if name.startswith("@"):
        with Image.open(Path(name[1:])) as source:
            tile = source.convert("L")
    else:
        fraction = {"gray12": .125, "gray25": .25, "gray50": .5, "gray75": .75}.get(name)
        if fraction is None:
            raise RuntimeError(f"image capture cannot render stipple {name!r}")
        tile = Image.new("L", (8, 8))
        tile.putdata([255 if value < 64 * fraction else 0 for row in _BAYER for value in row])
    tile = tile.resize((tile.width * scale, tile.height * scale), Image.Resampling.NEAREST)
    result = Image.new("L", size)
    for y in range(0, size[1], tile.height):
        for x in range(0, size[0], tile.width):
            result.paste(tile, (x, y))
    return result


def render_canvas(canvas, *, background, exclude=()):
    """Capture in painter order, including HUD, at the viewport's pixel size.

    Uses 2x supersampling. Fonts use the requested system face when available,
    with a Pillow fallback. This is independent of window visibility/occlusion.
    """
    try:
        from PIL import Image, ImageChops, ImageDraw
    except ImportError as error:
        raise RuntimeError('image capture requires Pillow; install "ANYtk3D[capture]"') from error
    scale = 2
    width, height = int(canvas.winfo_width()), int(canvas.winfo_height())
    if width <= 1 or height <= 1:
        raise RuntimeError("the software viewport must have a layout before capture")
    size = width * scale, height * scale

    def color(value):
        if not value:
            return None
        return tuple(round(channel / 257) for channel in canvas.winfo_rgb(value)) + (255,)

    image = Image.new("RGBA", size, color(background))
    patterns = {}
    for item in canvas.find_all():
        if item in exclude or canvas.itemcget(item, "state") == "hidden":
            continue
        kind = canvas.type(item)
        if kind not in {"polygon", "line", "rectangle", "oval", "text"}:
            raise RuntimeError(f"image capture cannot render canvas item type {kind!r}")
        coords = [float(value) * scale for value in canvas.coords(item)]
        points = list(zip(coords[::2], coords[1::2]))
        get = lambda name: canvas.itemcget(item, name)
        fill = color(get("fill"))
        draw = ImageDraw.Draw(image)
        if kind == "text":
            font = _font(canvas, item, scale)
            text = get("text")
            # Tk's bounding box carries its anchor and multiline placement.
            bbox = canvas.bbox(item)
            if bbox and fill:
                draw.multiline_text((bbox[0] * scale, bbox[1] * scale), text,
                                    font=font, fill=fill, anchor="la", align=get("justify") or "left")
            continue
        pen = max(1, round(float(get("width") or 1) * scale))
        if kind == "line":
            if fill:
                draw.line(points, fill=fill, width=pen, joint="curve")
                arrow = get("arrow")
                if arrow in {"first", "last", "both"}:
                    shape = [float(v) * scale for v in canvas.tk.splitlist(get("arrowshape"))]
                    for tip, previous in ([ (points[0], points[1]) ] if arrow == "first" else
                                          [ (points[-1], points[-2]) ] if arrow == "last" else
                                          [(points[0], points[1]), (points[-1], points[-2])]):
                        dx, dy = tip[0] - previous[0], tip[1] - previous[1]
                        length = math.hypot(dx, dy)
                        if length:
                            dx, dy = dx / length, dy / length
                            a, b, c = shape
                            draw.polygon([tip, (tip[0]-b*dx+c*dy, tip[1]-b*dy-c*dx),
                                          (tip[0]-a*dx, tip[1]-a*dy),
                                          (tip[0]-b*dx-c*dy, tip[1]-b*dy+c*dx)], fill=fill)
            continue
        outline = color(get("outline"))
        def paint(target, *, fill=None, outline=None):
            if kind == "polygon":
                if fill is not None:
                    target.polygon(points, fill=fill)
                if outline:
                    target.line(points + points[:1], fill=outline, width=pen, joint="curve")
            else:
                method = target.rectangle if kind == "rectangle" else target.ellipse
                method(coords, fill=fill, outline=outline, width=pen)
        stipple = get("stipple")
        if fill and stipple:
            mask = Image.new("L", size)
            paint(ImageDraw.Draw(mask), fill=255)
            if stipple not in patterns:
                patterns[stipple] = _stipple(stipple, size, scale)
            mask = ImageChops.multiply(mask, patterns[stipple])
            image.paste(fill, (0, 0, *size), mask)
            paint(ImageDraw.Draw(image), outline=outline)
        else:
            paint(draw, fill=fill, outline=outline)
    return image.resize((width, height), Image.Resampling.LANCZOS)
