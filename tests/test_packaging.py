'''
Release metadata guards.

These run against the working tree, so a version bump or a renamed entry
point is caught before a distribution is built rather than after it is on
PyPI.
'''
import pathlib
import tomllib

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
    # numpy is the only third-party import in the package; tkinter ships with
    # CPython and must not be listed as a distribution dependency.
    assert metadata['dependencies'] == ['numpy']
    sources = ' '.join(
        path.read_text(encoding='utf-8')
        for path in (ROOT / 'src' / 'anytk3d').glob('*.py')
    )
    for forbidden in ('import matplotlib', 'import scipy', 'import OpenGL'):
        assert forbidden not in sources
