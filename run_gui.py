#!/usr/bin/env python
"""Run the ANYtk3D interactive demo from this checkout."""

from __future__ import annotations

import sys
from pathlib import Path


_SOURCE = Path(__file__).resolve().parent / "src"
if _SOURCE.is_dir() and str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))


def main() -> None:
    """Launch the maintained interactive demo."""

    from anytk3d.demo import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
