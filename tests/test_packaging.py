'''
Release metadata guards.

These run against the working tree, so a version bump or a renamed entry
point is caught before a distribution is built rather than after it is on
PyPI.
'''
import pathlib
import shutil
import subprocess
import sys
import tomllib
import zipfile

import pytest

import anytk3d


ROOT = pathlib.Path(anytk3d.__file__).resolve().parents[2]
PYPROJECT = ROOT / 'pyproject.toml'


@pytest.fixture(scope='module')
def metadata():
    if not PYPROJECT.exists():
        pytest.skip('running against an installed package, not a source tree')
    return tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))['project']


def test_version_matches_pyproject(metadata):
    assert anytk3d.__version__ == metadata['version']


def test_declared_entry_point_is_importable(metadata):
    target = metadata['gui-scripts']['anytk3d-demo']
    module_name, _, attribute = target.partition(':')
    module = __import__(module_name, fromlist=['_'])
    assert callable(getattr(module, attribute))


def test_readme_and_license_are_shipped(metadata):
    assert (ROOT / metadata['readme']).exists()
    for pattern in metadata['license-files']:
        assert list(ROOT.glob(pattern)), f'no file matches {pattern!r}'


def test_every_public_name_is_importable():
    for name in anytk3d.__all__:
        assert hasattr(anytk3d, name), name


def test_runtime_dependencies_are_declared(metadata):
    # NumPy remains a direct renderer dependency and ANY3dView supplies the
    # shared core; tkinter ships with CPython and is not a distribution.
    assert metadata['dependencies'] == ['numpy', 'ANY3dView>=0.5,<0.6']
    sources = ' '.join(
        path.read_text(encoding='utf-8')
        for path in (ROOT / 'src' / 'anytk3d').glob('*.py')
    )
    for forbidden in ('import matplotlib', 'import scipy', 'import OpenGL'):
        assert forbidden not in sources


@pytest.mark.packaging
def test_release_candidate_builds_an_importable_wheel_pair(metadata, tmp_path):
    """The two-repository release candidate must contain every runtime module.

    Reconstructing ANYtk3D from Git-known and non-ignored candidate files keeps
    editor/build debris out while still allowing this gate to run before a
    commit.  The isolated probe loads the matching ANY3dView wheel as well.
    """

    if not (ROOT / '.git').exists():
        pytest.skip('tracked-source gate requires a Git source checkout')
    if shutil.which('git') is None:
        pytest.skip('tracked-source gate requires git')

    listed = subprocess.run(
        [
            'git', '-c', f'safe.directory={ROOT}', '-C', str(ROOT),
            'ls-files', '--cached', '--others', '--exclude-standard', '-z',
        ],
        check=True,
        capture_output=True,
    ).stdout.split(b'\0')
    relative_paths = [
        pathlib.Path(value.decode('utf-8')) for value in listed if value
    ]
    required = pathlib.Path('src/anytk3d/_selection.py')
    assert required in relative_paths, f'{required} is missing from the release candidate'

    snapshot = tmp_path / 'release-candidate'
    for relative in relative_paths:
        source = ROOT / relative
        if not source.is_file():
            continue
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    wheel_dir = tmp_path / 'wheel'
    completed = subprocess.run(
        [
            sys.executable,
            '-m',
            'build',
            '--wheel',
            '--no-isolation',
            '--outdir',
            str(wheel_dir),
            str(snapshot),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = tuple(wheel_dir.glob('*.whl'))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
    assert 'anytk3d/_selection.py' in names

    core_root = ROOT.parent / 'ANY3dView'
    if not (core_root / 'pyproject.toml').is_file():
        pytest.skip('paired release gate requires the sibling ANY3dView checkout')
    core_wheel_dir = tmp_path / 'core-wheel'
    core_build = subprocess.run(
        [
            sys.executable,
            '-m',
            'build',
            '--wheel',
            '--no-isolation',
            '--outdir',
            str(core_wheel_dir),
            str(core_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert core_build.returncode == 0, core_build.stdout + core_build.stderr
    core_wheels = tuple(core_wheel_dir.glob('*.whl'))
    assert len(core_wheels) == 1

    probe = subprocess.run(
        [
            sys.executable,
            '-I',
            '-c',
            (
                'import sys; '
                f'sys.path.insert(0, {str(core_wheels[0])!r}); '
                f'sys.path.insert(0, {str(wheels[0])!r}); '
                'import anytk3d; '
                'from anytk3d._selection import ProjectedSelectionIndex; '
                f'assert anytk3d.__version__ == {metadata["version"]!r}; '
                'assert ProjectedSelectionIndex is not None'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
