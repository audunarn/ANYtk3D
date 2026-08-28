from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
ACTION = re.compile(r"uses:\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@([^\s#]+)")
COMMIT = re.compile(r"[0-9a-f]{40}")

EXPECTED = {
    ".github/workflows/release.yml": (
        ("actions/checkout", "d23441a48e516b6c34aea4fa41551a30e30af803"),
        ("actions/setup-python", "ece7cb06caefa5fff74198d8649806c4678c61a1"),
        ("actions/upload-artifact", "330a01c490aca151604b8cf639adc76d48f6c5d4"),
        ("actions/download-artifact", "018cc2cf5baa6db3ef3c5f8a56943fffe632ef53"),
        (
            "pypa/gh-action-pypi-publish",
            "dc37677b2e1c63e2034f94d8a5b11f265b73ba33",
        ),
    ),
}


def test_workflow_actions_are_closed_and_commit_pinned() -> None:
    workflows = tuple(
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))
    )
    assert workflows == tuple(sorted(EXPECTED))

    for relative, expected in EXPECTED.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        observed = tuple(ACTION.findall(text))
        assert observed == expected
        assert all(COMMIT.fullmatch(commit) for _action, commit in observed)
