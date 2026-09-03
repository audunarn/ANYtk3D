"""Fail release validation when licensing or dependency policy drifts."""

from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    if project["license"] != "MPL-2.0":
        raise SystemExit("project licence must be MPL-2.0")
    if project["version"] != "0.5.5":
        raise SystemExit("licensing gate is scoped to the 0.5.5 release")
    if project["dependencies"] != ["numpy", "ANY3dView>=0.5.5,<0.6"]:
        raise SystemExit("ANYtk3D must use the coordinated ANY3dView 0.5.5 range")

    licence = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if "Mozilla Public License Version 2.0" not in licence:
        raise SystemExit("LICENSE does not contain the MPL-2.0 text")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "Mozilla Public" not in readme or "License 2.0" not in readme:
        raise SystemExit("README does not declare MPL-2.0")
    if not (ROOT / "THIRD_PARTY_NOTICES.md").is_file():
        raise SystemExit("third-party notices are missing")


if __name__ == "__main__":
    main()
