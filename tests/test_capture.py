"""Report output must contain scene pixels, never desktop screenshots."""
from types import SimpleNamespace
import pytest

Image = pytest.importorskip("PIL.Image")
from anytk3d.capture import render_canvas


class Canvas:
    def __init__(self, items):
        self.items = items
        self.tk = SimpleNamespace(splitlist=lambda value: value.split())

    def winfo_width(self): return 32
    def winfo_height(self): return 32
    def find_all(self): return tuple(range(len(self.items)))
    def type(self, item): return self.items[item][0]
    def coords(self, item): return self.items[item][1]
    def itemcget(self, item, name): return self.items[item][2].get(name, "")
    def winfo_rgb(self, name):
        from PIL.ImageColor import getrgb
        return tuple(value * 257 for value in getrgb(name))


def test_capture_painter_order_hidden_items_and_shapes(monkeypatch):
    from PIL import ImageGrab
    monkeypatch.setattr(ImageGrab, "grab", lambda **kwargs: pytest.fail("desktop capture"))
    canvas = Canvas([
        ("polygon", [0, 0, 31, 0, 31, 31, 0, 31], {"fill": "red"}),
        ("rectangle", [5, 5, 25, 25], {"fill": "blue"}),
        ("oval", [10, 10, 20, 20], {"fill": "green"}),
        ("rectangle", [0, 0, 31, 31], {"fill": "black", "state": "hidden"}),
    ])
    image = render_canvas(canvas, background="white")
    assert image.size == (32, 32) and image.mode == "RGBA"
    assert image.getpixel((2, 2))[:3] == (255, 0, 0)
    assert image.getpixel((8, 8))[:3] == (0, 0, 255)
    assert image.getpixel((15, 15))[:3] == (0, 128, 0)


def test_capture_stipple_preserves_background_and_outline():
    canvas = Canvas([("polygon", [1, 1, 30, 1, 30, 30, 1, 30],
                      {"fill": "red", "outline": "blue", "width": "1", "stipple": "gray50"})])
    image = render_canvas(canvas, background="white")
    r, g, b, _ = image.getpixel((16, 16))
    assert r > 240 and g < r and b < r
    assert image.getpixel((1, 15))[2] > image.getpixel((1, 15))[0]


def test_capture_arrow_and_excluded_overlay():
    canvas = Canvas([
        ("line", [3, 16, 27, 16], {"fill": "black", "width": "2", "arrow": "last", "arrowshape": "8 10 3"}),
        ("rectangle", [0, 0, 31, 31], {"fill": "red"}),
    ])
    image = render_canvas(canvas, background="white", exclude=(1,))
    assert image.getpixel((16, 16))[0] < 30
    assert image.getpixel((16, 3))[:3] == (255, 255, 255)


def test_unknown_item_fails_instead_of_omitting_content():
    with pytest.raises(RuntimeError, match="item type"):
        render_canvas(Canvas([("window", [], {})]), background="white")


@pytest.mark.gui
def test_real_viewer_capture_includes_text_and_works_while_window_is_hidden():
    import tkinter as tk
    from anytk3d import Tkinter3DCanvas
    root = tk.Tk()
    try:
        root.geometry("320x240")
        viewer = Tkinter3DCanvas(root, width=320, height=240)
        viewer.pack(fill="both", expand=True)
        root.update()
        viewer.add_box(1, 1, 1)
        viewer.fit_to_scene()
        shown = viewer.capture_image()
        root.withdraw()
        hidden = viewer.capture_image()
        assert shown.size == hidden.size
        assert shown.tobytes() == hidden.tobytes()
        pixels = hidden.tobytes()
        assert len({pixels[index:index + 4] for index in range(0, len(pixels), 4)}) > 10
    finally:
        root.destroy()
