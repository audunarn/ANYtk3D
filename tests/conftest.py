"""Safe defaults for tests that can create native Tk windows."""

from __future__ import annotations

import inspect
import os

import pytest


RUN_GUI_TESTS = os.environ.get("ANYTK3D_RUN_GUI_TESTS", "").casefold() in {
    "1",
    "true",
    "yes",
}


def pytest_collection_modifyitems(items):
    if RUN_GUI_TESTS:
        return
    skip = pytest.mark.skip(
        reason="real Tk GUI test is opt-in; set ANYTK3D_RUN_GUI_TESTS=1"
    )
    for item in items:
        try:
            source = inspect.getsource(item.obj)
        except (OSError, TypeError):
            source = ""
        real_tk = any(name.endswith("root") for name in item.fixturenames) or any(
            token in source for token in ("tk.Tk(", "tkinter.Tk(")
        )
        if real_tk:
            item.add_marker("gui")
            item.add_marker(skip)
