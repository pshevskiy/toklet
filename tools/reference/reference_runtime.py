"""Explicit Apple reference-runtime admission shared by new operator producers.

No numerical module is imported until its caller admits external file identities.
Legacy gated-delta v1 tooling stays frozen under its original runtime format.
"""
import hashlib
from importlib.machinery import ExtensionFileLoader, PathFinder, SourceFileLoader
from importlib.metadata import version
from importlib.util import cache_from_source, find_spec
import json
import math
from pathlib import Path
import re
import sys
import sysconfig

VERSIONS = {'numpy': '2.3.2', 'mlx': '0.32.3'}


def sha(value):
    if not isinstance(value, str) or re.fullmatch('[a-f0-9]{64}', value) is None:
        raise ValueError('Expected an external lowercase SHA256 identity')
    return value


def json_unique(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError('Duplicate reference JSON field')
        value[key] = item
    return value


def bad_json_number(value):
    raise ValueError('Nonfinite reference JSON number: ' + value)


def finite_json_number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite reference JSON number: ' + value)
    return number


def captured(path, maximum, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular captured reference file')
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Captured reference buffer identity differs')
    return raw


def runtime_paths(selected_sources):
    """Enumerate the installed selected producer closure without importing it."""
    site = Path(sysconfig.get_path('purelib')).resolve()
    root = Path(__file__).resolve().parents[2]
    if not site.is_relative_to(root / '.venv'):
        raise ValueError('Repository virtual environment required')
    paths = {Path(__file__).resolve(), Path(sys.executable).resolve(),
             *(Path(path).resolve() for path in selected_sources)}
    for package in ('numpy', 'mlx'):
        folder = site / package
        if not folder.is_dir():
            raise ValueError('Required reference runtime package absent')
        paths.update(path for path in folder.rglob('*')
            if path.is_file() and '__pycache__' not in path.parts)
    for pattern in ('numpy*.libs', 'numpy*.dist-info', 'mlx*.dist-info'):
        for folder in site.glob(pattern):
            paths.update(path for path in folder.rglob('*') if path.is_file() and '__pycache__' not in path.parts)
    if not any(path.name == 'mlx.metallib' for path in paths) or not any(path.name == 'libmlx.dylib' for path in paths):
        raise ValueError('Full MLX Metal/native runtime required')
    return sorted(paths)


def selected_imports(site, paths):
    """Resolve package roots without importing or running numerical code."""
    admitted = {str(path) for path in paths}
    selected = {}
    for name in ('numpy', 'mlx'):
        spec = find_spec(name)
        folder = site / name
        if spec is None or spec.submodule_search_locations is None:
            raise ValueError('Reference numerical package resolution differs')
        locations = list(spec.submodule_search_locations)
        if locations != [str(folder)]:
            raise ValueError('Reference numerical package search path differs')
        if name == 'numpy':
            if spec.origin != str(folder / '__init__.py') or not isinstance(spec.loader, SourceFileLoader) or spec.origin not in admitted:
                raise ValueError('Reference NumPy import source differs')
            selected[name] = spec.origin
        elif spec.origin is not None:
            raise ValueError('Expected the pinned MLX namespace package')
        for child, loader in ([('core', ExtensionFileLoader), ('nn', SourceFileLoader)] if name == 'mlx' else []):
            child_spec = PathFinder.find_spec(name + '.' + child, locations)
            if child_spec is None or child_spec.origin not in admitted or not isinstance(child_spec.loader, loader):
                raise ValueError('Reference MLX import source differs')
            selected[name + '.' + child] = child_spec.origin
    return selected


def admit_runtime(path, expected_sha, selected_sources):
    """Admit externally pinned exact runtime files before numerical imports.

    The owned caller creates the manifest and fresh pycache prefix before launch.
    Version strings alone, self-reported hashes and an already imported runtime
    cannot admit a reference execution.
    """
    raw = captured(path, 2 * 1024 * 1024, sha(expected_sha))
    manifest = json.loads(raw, object_pairs_hook=json_unique,
        parse_constant=bad_json_number, parse_float=finite_json_number)
    paths = runtime_paths(selected_sources)
    if (not isinstance(manifest, dict) or set(manifest) != {'format', 'versions', 'files'}
            or manifest['format'] != 'toklet-mlx-reference-runtime-v1'
            or manifest['versions'] != VERSIONS or not isinstance(manifest['files'], dict)
            or set(manifest['files']) != {str(path) for path in paths}):
        raise ValueError('Exact externally pinned reference runtime closure required')
    if any(name == 'numpy' or name.startswith('numpy.') or name == 'mlx' or name.startswith('mlx.') for name in sys.modules):
        raise ValueError('Reference numerical runtime was already imported')
    if not sys.pycache_prefix or not Path(sys.pycache_prefix).is_absolute():
        raise ValueError('Fresh owned Python cache prefix required')
    site = Path(sysconfig.get_path('purelib')).resolve()
    imports = selected_imports(site, paths)
    for item in paths:
        if item.is_relative_to(site) and item.suffix == '.py' and Path(cache_from_source(str(item))).exists():
            raise ValueError('Reference producer bytecode cache already exists')
        pin = manifest['files'][str(item)]
        if (not isinstance(pin, dict) or set(pin) != {'bytes', 'sha256'}
                or type(pin['bytes']) is not int or not 0 <= pin['bytes'] <= 512 * 1024 * 1024
                or item.is_symlink()):
            raise ValueError('Bounded regular runtime file identity required')
        expected = sha(pin['sha256'])
        digest = hashlib.sha256()
        count = 0
        with item.open('rb') as stream:
            while chunk := stream.read(128 * 1024):
                count += len(chunk)
                if count > pin['bytes']:
                    raise ValueError('Reference runtime file exceeds its admitted length')
                digest.update(chunk)
        if count != pin['bytes'] or digest.hexdigest() != expected:
            raise ValueError('Reference runtime bytes differ')
    if {name: version(name) for name in VERSIONS} != VERSIONS:
        raise ValueError('Reference runtime versions differ')
    return dict(manifest_sha256=expected_sha, files=len(paths), pycache_prefix=sys.pycache_prefix, imports=imports)
