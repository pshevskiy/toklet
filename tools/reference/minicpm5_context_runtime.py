"""Cold NumPy-only admission; no MLX, Rust, Torch or GPU reference execution."""
import ctypes
from importlib.metadata import version
from importlib.util import cache_from_source, find_spec
from importlib.machinery import SourceFileLoader, ExtensionFileLoader, PathFinder
import json
from pathlib import Path
import sys
import sysconfig

import reference_runtime as common
import text_reference_runtime as inventory

VERSIONS = {'numpy': '2.3.2'}
FORMAT = 'toklet-minicpm5-context-cpu-runtime-v1'


def bind_environment(root):
    root = Path(root).resolve()
    if (not (root/'Cargo.lock').is_file()
            or Path(sys.prefix).resolve() != (root/'.venv').resolve()):
        raise ValueError('Exact repository-owned numerical environment required')
    # Reuse only the established inventory builder, with explicit package roots.
    # Its text-only admission/import routine and shared source remain untouched.
    inventory.ROOT = root
    inventory.PACKAGES = ('numpy',)


def paths(sources):
    site = Path(sysconfig.get_path('purelib')).resolve()
    selected = set(inventory.paths(sources))
    for pattern in ('numpy*.libs', 'numpy*.dist-info'):
        for folder in site.glob(pattern):
            selected.update(p.resolve() for p in folder.rglob('*')
                            if p.is_file() and '__pycache__' not in p.parts)
    return sorted(selected)


def admit(path, expected, sources):
    if not sys.flags.no_user_site or not sys.dont_write_bytecode:
        raise ValueError('CPU context reference requires -s -B')
    if any(n == p or n.startswith(p+'.') for n in sys.modules for p in VERSIONS):
        raise ValueError('CPU context numerical package was already imported')
    if not sys.pycache_prefix or not Path(sys.pycache_prefix).is_absolute():
        raise ValueError('Fresh absolute CPU context cache prefix required')
    raw = common.captured(path, 2*1024**2, common.sha(expected))
    manifest = json.loads(raw, object_pairs_hook=common.json_unique,
                          parse_constant=common.bad_json_number)
    files = paths(sources)
    if (type(manifest) is not dict or set(manifest) != {'format', 'versions', 'files'}
            or manifest['format'] != FORMAT or manifest['versions'] != VERSIONS
            or set(manifest['files']) != {str(p) for p in files}):
        raise ValueError('Exact externally pinned CPU context runtime required')

    def check_file(p):
        pin = manifest['files'][str(p)]
        if (type(pin) is not dict or set(pin) != {'bytes', 'sha256'}
                or type(pin['bytes']) is not int or not 0 <= pin['bytes'] <= 512*1024**2):
            raise ValueError('Bounded CPU context runtime file identity required')
        if len(common.captured(p, pin['bytes'], common.sha(pin['sha256']))) != pin['bytes']:
            raise ValueError('Exact CPU context runtime extent differs')

    for p in files:
        if p.suffix == '.py' and Path(cache_from_source(str(p))).exists():
            raise ValueError('CPU context reference bytecode already exists')
        check_file(p)
    if {name: version(name) for name in VERSIONS} != VERSIONS:
        raise ValueError('CPU context package versions differ')
    site = Path(sysconfig.get_path('purelib')).resolve()
    admitted_files = {str(p) for p in files}
    spec = find_spec('numpy')
    if (spec is None or spec.origin != str(site/'numpy/__init__.py')
            or not isinstance(spec.loader, SourceFileLoader)
            or list(spec.submodule_search_locations or []) != [str(site/'numpy')]):
        raise ValueError('Exact NumPy root loader required')
    core = PathFinder.find_spec('numpy._core', [str(site/'numpy')])
    if core is None or not isinstance(core.loader, SourceFileLoader):
        raise ValueError('Exact NumPy core loader required')
    native = PathFinder.find_spec('numpy._core._multiarray_umath', list(core.submodule_search_locations or []))
    if native is None or not isinstance(native.loader, ExtensionFileLoader):
        raise ValueError('Native NumPy array implementation required')
    selected = {'numpy':spec.origin, 'numpy._core':core.origin,
                'numpy._core._multiarray_umath':native.origin}
    if any(origin not in admitted_files for origin in selected.values()):
        raise ValueError('Selected NumPy implementation is not admitted')

    def origins():
        modules = {}
        for name, module in list(sys.modules.items()):
            origin = getattr(module, '__file__', None)
            if not origin:
                continue
            p = Path(origin).resolve()
            if str(p) not in manifest['files']:
                raise ValueError('Unadmitted CPU context module: '+name+': '+str(p))
            check_file(p)
            modules[name] = str(p)
        if any(modules.get(name) != origin for name, origin in selected.items()):
            raise ValueError('Actual CPU context numerical import differs')
        library = ctypes.CDLL(None)
        count = library._dyld_image_count
        count.argtypes = [];count.restype = ctypes.c_uint32
        image = library._dyld_get_image_name
        image.argtypes = [ctypes.c_uint32];image.restype = ctypes.c_char_p
        native, system = {}, []
        for index in range(count()):
            name = image(index).decode();p = Path(name).resolve()
            if str(p).startswith(('/usr/lib/', '/System/')):
                system.append(name)
            else:
                if str(p) not in manifest['files']:
                    raise ValueError('Unadmitted CPU context native image: '+str(p))
                check_file(p)
                native[str(p)] = manifest['files'][str(p)]
        return dict(modules=modules, non_system_native_images=native,
                    system_native_images=sorted(system))

    import numpy
    initial = origins()

    def endpoint():
        if paths(sources) != files:
            raise ValueError('CPU context runtime inventory changed')
        for p in files:
            check_file(p)
        return origins()

    return dict(manifest_sha256=expected, files=len(files), versions=VERSIONS,
                pycache_prefix=sys.pycache_prefix, imports=selected,
                initial_origins=initial), endpoint
