"""Admit the exact cold text-reference runtime before third-party imports."""
import hashlib
from importlib.machinery import ExtensionFileLoader, PathFinder, SourceFileLoader
from importlib.metadata import version
from importlib.util import cache_from_source, find_spec
import json
from pathlib import Path
import re
import sys
import sysconfig

VERSIONS = {'python': '3.12.8', 'jinja2': '3.1.6', 'markupsafe': '3.0.4', 'tokenizers': '0.22.2'}
PACKAGES = ('jinja2', 'markupsafe', 'tokenizers')
_admitted = None


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate reference JSON field')
        result[key] = value
    return result


def pin(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Regular runtime file required: ' + str(path))
    count, digest = 0, hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(128 * 1024):
            count += len(chunk)
            if count > 512 * 1024 * 1024:
                raise ValueError('Runtime file exceeds bound')
            digest.update(chunk)
    return dict(bytes=count, sha256=digest.hexdigest())


def captured(path, maximum, expected):
    if not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None or path.is_symlink():
        raise ValueError('External SHA256 and regular input required')
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Admitted buffer identity differs')
    return raw


def paths(sources):
    site = Path(sysconfig.get_path('purelib')).resolve()
    venv = Path(sys.prefix).resolve()
    if venv.name != '.venv' or not site.is_relative_to(venv):
        raise ValueError('Project virtual environment required')
    result = {Path(__file__).resolve(), Path(sys.executable).resolve(), *(Path(p).resolve() for p in sources)}
    shared_python = Path(sys.base_prefix) / 'Python'
    if not shared_python.is_file():
        raise ValueError('Expected Apple Python shared runtime')
    result.add(shared_python.resolve())
    for package in PACKAGES:
        folder = site / package
        if not folder.is_dir():
            raise ValueError('Text runtime package absent')
        result.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        for info in site.glob(package + '-*.dist-info'):
            result.update(p for p in info.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return sorted(result)


def imports(selected):
    site = Path(sysconfig.get_path('purelib')).resolve()
    admitted = {str(p) for p in selected}
    result = {}
    for package in PACKAGES:
        spec = find_spec(package)
        folder = site / package
        if (spec is None or spec.origin != str(folder / '__init__.py')
                or not isinstance(spec.loader, SourceFileLoader)
                or list(spec.submodule_search_locations or []) != [str(folder)] or spec.origin not in admitted):
            raise ValueError('Text runtime import resolution differs')
        result[package] = spec.origin
    for package, child in [('tokenizers', 'tokenizers'), ('markupsafe', '_speedups')]:
        spec = PathFinder.find_spec(package + '.' + child, [str(site / package)])
        if spec is None or not isinstance(spec.loader, ExtensionFileLoader) or spec.origin not in admitted:
            raise ValueError('Native text runtime import differs')
        result[package + '.' + child] = spec.origin
    return result


def admit(manifest_path, expected_sha, sources):
    global _admitted
    if any(n == p or n.startswith(p + '.') for n in sys.modules for p in PACKAGES):
        raise ValueError('Third-party text runtime already imported before admission')
    raw = captured(manifest_path, 2 * 1024 * 1024, expected_sha)
    manifest = json.loads(raw, object_pairs_hook=unique)
    selected = paths(sources)
    if (set(manifest) != {'format', 'versions', 'files'} or manifest['format'] != 'toklet-chat-reference-runtime-v1'
            or manifest['versions'] != VERSIONS or set(manifest['files']) != {str(p) for p in selected}):
        raise ValueError('Exact externally pinned text runtime closure required')
    if not sys.pycache_prefix or not Path(sys.pycache_prefix).is_absolute():
        raise ValueError('Fresh owned Python cache prefix required')
    site = Path(sysconfig.get_path('purelib')).resolve()
    for path in selected:
        if path.is_relative_to(site) and path.suffix == '.py' and Path(cache_from_source(str(path))).exists():
            raise ValueError('Text runtime bytecode cache already exists')
        if pin(path) != manifest['files'][str(path)]:
            raise ValueError('Text runtime file bytes differ')
    if sys.version.split()[0] != VERSIONS['python'] or any(version(p) != VERSIONS[p] for p in PACKAGES):
        raise ValueError('Text runtime versions differ')
    selected_imports = imports(selected)
    _admitted = dict(path=manifest_path, sha256=expected_sha, sources=list(sources), manifest=manifest, imports=selected_imports)
    return dict(manifest_sha256=expected_sha, files=len(selected), imports=selected_imports)


def check():
    if _admitted is None:
        raise ValueError('Text runtime must be admitted before importing libraries')
    a = _admitted
    captured(a['path'], 2 * 1024 * 1024, a['sha256'])
    selected = paths(a['sources'])
    if set(a['manifest']['files']) != {str(p) for p in selected}:
        raise ValueError('Text runtime inventory changed')
    for path in selected:
        if pin(path) != a['manifest']['files'][str(path)]:
            raise ValueError('Text runtime bytes changed')
    for name, expected in a['imports'].items():
        if name in sys.modules and getattr(sys.modules[name], '__file__', None) != expected:
            raise ValueError('Loaded text runtime differs from admitted imports')
