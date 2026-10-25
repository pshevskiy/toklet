"""Cold, externally admitted runtime for pinned text-only reference captures.

Tokenizers/Jinja/MarkupSafe stay unimported until every source/native identity
is checked. This does not change the established NumPy/MLX runtime contract.
"""
from importlib.metadata import version
from importlib.machinery import ExtensionFileLoader, PathFinder, SourceFileLoader
from importlib.util import cache_from_source, find_spec
import json
from pathlib import Path
import sys
import sysconfig

import reference_runtime as common

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = {'tokenizers': '0.22.2', 'Jinja2': '3.1.6', 'MarkupSafe': '3.0.4'}
PACKAGES = ('tokenizers', 'jinja2', 'markupsafe')


def paths(sources):
    site = Path(sysconfig.get_path('purelib')).resolve()
    stdlib = Path(sysconfig.get_path('stdlib')).resolve()
    base = Path(sys.base_prefix).resolve()
    if not site.is_relative_to(ROOT/'.venv'):
        raise ValueError('Repository text reference environment required')
    selected = {Path(__file__).resolve(), Path(common.__file__).resolve(),
                Path(sys.executable).resolve(), *(Path(q).resolve() for q in sources),
                base/'Python', base/'Resources/Python.app/Contents/MacOS/Python'}
    for package in PACKAGES:
        folder = site/package
        if not folder.is_dir():raise ValueError('Text reference package absent')
        selected.update(q for q in folder.rglob('*') if q.is_file() and '__pycache__' not in q.parts)
    for pattern in ('tokenizers*.dist-info', 'jinja2*.dist-info', 'markupsafe*.dist-info',
                    '_distutils_hack', 'setuptools*.dist-info'):
        for folder in site.glob(pattern):
            selected.update(q for q in folder.rglob('*') if q.is_file() and '__pycache__' not in q.parts)
    selected.update(q for q in site.glob('*.pth') if q.is_file())
    # Pin the Python source library too. A fresh prefix and -B avoid accepting
    # unrelated pre-existing bytecode. Third-party packages are enumerated above.
    selected.update(q for q in stdlib.rglob('*.py') if 'site-packages' not in q.parts and '__pycache__' not in q.parts)
    selected.update(base.rglob('*.dylib'))
    selected.update((stdlib/'lib-dynload').glob('*.so'))
    return sorted({q.resolve() for q in selected})


def selected_imports(site,inventory):
    admitted={str(q) for q in inventory};selected={}
    for name in PACKAGES:
        folder=site/name;spec=find_spec(name)
        if (spec is None or spec.origin != str(folder/'__init__.py')
                or spec.origin not in admitted or not isinstance(spec.loader,SourceFileLoader)
                or list(spec.submodule_search_locations or []) != [str(folder)]):
            raise ValueError('Text reference package resolution/loader differs')
        selected[name]=spec.origin
    # This reference requires the installed native implementations explicitly;
    # MarkupSafe's possible Python fallback is not silently admitted instead.
    for name in ('tokenizers.tokenizers','markupsafe._speedups'):
        folder=site/name.split('.')[0];spec=PathFinder.find_spec(name,[str(folder)])
        if (spec is None or spec.origin not in admitted or not isinstance(spec.loader,ExtensionFileLoader)
                or Path(spec.origin).parent!=folder):
            raise ValueError('Text reference native resolution/loader differs')
        selected[name]=spec.origin
    return selected


def admit(path, expected, sources):
    if not sys.flags.no_user_site or not sys.dont_write_bytecode:
        raise ValueError('Text reference requires -s -B')
    if any(n == p or n.startswith(p+'.') for n in sys.modules for p in PACKAGES):
        raise ValueError('Text reference package already imported')
    if not sys.pycache_prefix or not Path(sys.pycache_prefix).is_absolute():
        raise ValueError('Fresh absolute text reference cache prefix required')
    raw = common.captured(path, 2*1024**2, common.sha(expected))
    manifest = json.loads(raw, object_pairs_hook=common.json_unique,
                          parse_constant=common.bad_json_number)
    inventory = paths(sources)
    if (set(manifest) != {'format', 'versions', 'files'}
            or manifest['format'] != 'toklet-text-reference-runtime-v1'
            or manifest['versions'] != VERSIONS
            or set(manifest['files']) != {str(q) for q in inventory}):
        raise ValueError('Exact text reference runtime inventory required')
    for q in inventory:
        pin = manifest['files'][str(q)]
        if (set(pin) != {'bytes', 'sha256'} or type(pin['bytes']) is not int
                or not 0 <= pin['bytes'] <= 512*1024**2):
            raise ValueError('Bounded text runtime file identity required')
        if q.suffix == '.py' and Path(cache_from_source(str(q))).exists():
            raise ValueError('Text reference bytecode cache already exists')
        if len(common.captured(q, pin['bytes'], common.sha(pin['sha256']))) != pin['bytes']:
            raise ValueError('Exact text runtime byte count required')
    if {name: version(name) for name in VERSIONS} != VERSIONS:
        raise ValueError('Text reference package versions differ')
    site = Path(sysconfig.get_path('purelib')).resolve()
    selected=selected_imports(site,inventory)

    def origins():
        import ctypes
        imported = {}
        for name, module in list(sys.modules.items()):
            origin = getattr(module, '__file__', None)
            if not origin:continue
            q = Path(origin).resolve();key = str(q)
            if key not in manifest['files']:
                raise ValueError('Unadmitted text module origin: '+name+': '+key)
            pin = manifest['files'][key]
            if len(common.captured(q, pin['bytes'], pin['sha256'])) != pin['bytes']:
                raise ValueError('Actual text module byte count differs')
            imported[name] = key
        for name,origin in selected.items():
            if imported.get(name) != origin:raise ValueError('Actual text import differs: '+name)
        library = ctypes.CDLL(None)
        count = library._dyld_image_count;count.argtypes=[];count.restype=ctypes.c_uint32
        image = library._dyld_get_image_name;image.argtypes=[ctypes.c_uint32];image.restype=ctypes.c_char_p
        non_system = {};system = []
        for i in range(count()):
            name=image(i).decode();q=Path(name).resolve();key=str(q)
            if key.startswith(('/usr/lib/', '/System/')):system.append(name);continue
            if key not in manifest['files']:raise ValueError('Unadmitted text native image: '+key)
            pin=manifest['files'][key]
            if len(common.captured(q,pin['bytes'],pin['sha256']))!=pin['bytes']:
                raise ValueError('Actual text native byte count differs')
            non_system[key]=pin
        return dict(modules=imported, non_system_native_images=non_system,
                    system_native_images=sorted(system))

    import tokenizers
    import jinja2
    import markupsafe
    initial = origins()

    def endpoint():
        if paths(sources) != inventory:raise ValueError('Text runtime inventory changed')
        for q in inventory:
            pin=manifest['files'][str(q)]
            if len(common.captured(q,pin['bytes'],pin['sha256']))!=pin['bytes']:
                raise ValueError('Endpoint text runtime byte count differs')
        return origins()
    return dict(manifest_sha256=expected, files=len(inventory), versions=VERSIONS,
                pycache_prefix=sys.pycache_prefix, imports=selected, initial_origins=initial), endpoint
