"""Granite's bounded runtime admission, hashing large files as byte streams.

The established affine runtime supplies the exact environment/inventory policy.
Only file verification changes: never capture the 190 MB Metal library in RAM.
The MiniCPM5 runtime and all its existing contracts remain untouched.
"""
import ctypes
from importlib.metadata import version
from importlib.util import cache_from_source
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import sysconfig

import minicpm5_quant_runtime as shared

common = shared.common
VERSIONS = shared.VERSIONS
FORMAT = 'toklet-granite-quant-runtime-v1'
HASH_CHUNK = 1024**2
bind_environment = shared.bind_environment
paths = shared.paths


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(value):
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def verify_file(path, pin):
    """Exact regular-file bytes plus descriptor/path identity, <=1 MiB/read."""
    require(type(pin) is dict and set(pin) == {'bytes', 'sha256'}
            and type(pin['bytes']) is int and 0 <= pin['bytes'] <= 512 * 1024**2,
            'Bounded runtime file identity required')
    expected = common.sha(pin['sha256'])
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        named = path.lstat()
        require(stat.S_ISREG(before.st_mode) and stat.S_ISREG(named.st_mode)
                and identity(before) == identity(named) and before.st_size == pin['bytes'],
                'Runtime file type, extent or named identity differs')
        remaining = before.st_size
        digest = hashlib.sha256()
        maximum_read = 0
        while remaining:
            request = min(remaining, HASH_CHUNK)
            maximum_read = max(maximum_read, request)
            raw = os.read(descriptor, request)
            require(bool(raw), 'Runtime file ended before its declared extent')
            remaining -= len(raw)
            digest.update(raw)
        require(os.read(descriptor, 1) == b'', 'Runtime file grew past its declared extent')
        require(identity(os.fstat(descriptor)) == identity(before)
                and identity(path.lstat()) == identity(before),
                'Runtime descriptor or path changed during hashing')
        require(digest.hexdigest() == expected, 'Runtime file SHA256 differs')
        return dict(bytes=before.st_size, sha256=expected,
                    maximum_read_bytes=maximum_read)
    finally:
        os.close(descriptor)


def admit(path, expected, sources):
    require(sys.flags.no_user_site and sys.dont_write_bytecode,
            'Conversion reference requires -s -B')
    require(not any(n == p or n.startswith(p + '.') for n in sys.modules for p in VERSIONS),
            'Conversion numerical package was already imported')
    require(sys.pycache_prefix and Path(sys.pycache_prefix).is_absolute(),
            'Fresh absolute conversion cache prefix required')
    raw = common.captured(path, 2 * 1024**2, common.sha(expected))
    manifest = json.loads(raw, object_pairs_hook=common.json_unique,
                          parse_constant=common.bad_json_number)
    files = paths(sources)
    require(type(manifest) is dict and set(manifest) == {'format', 'versions', 'files'}
            and manifest['format'] == FORMAT and manifest['versions'] == VERSIONS
            and set(manifest['files']) == {str(p) for p in files},
            'Exact externally pinned Granite conversion runtime required')

    def check_file(p):
        return verify_file(p, manifest['files'][str(p)])

    for p in files:
        require(p.suffix != '.py' or not Path(cache_from_source(str(p))).exists(),
                'Conversion reference bytecode already exists')
        check_file(p)
    require({name: version(name) for name in VERSIONS} == VERSIONS,
            'Conversion package versions differ')
    selected = common.selected_imports(Path(sysconfig.get_path('purelib')).resolve(), files)

    def origins():
        modules = {}
        for name, module in list(sys.modules.items()):
            origin = getattr(module, '__file__', None)
            if not origin:
                continue
            p = Path(origin).resolve()
            require(str(p) in manifest['files'], 'Unadmitted conversion module: ' + name)
            check_file(p)
            modules[name] = str(p)
        require(all(modules.get(name) == origin for name, origin in selected.items()),
                'Actual conversion numerical import differs')
        library = ctypes.CDLL(None)
        count = library._dyld_image_count
        count.argtypes = []
        count.restype = ctypes.c_uint32
        image = library._dyld_get_image_name
        image.argtypes = [ctypes.c_uint32]
        image.restype = ctypes.c_char_p
        native, system = {}, []
        for index in range(count()):
            name = image(index).decode()
            p = Path(name).resolve()
            if str(p).startswith(('/usr/lib/', '/System/')):
                system.append(name)
            else:
                require(str(p) in manifest['files'], 'Unadmitted conversion native image: ' + str(p))
                check_file(p)
                native[str(p)] = manifest['files'][str(p)]
        return dict(modules=modules, non_system_native_images=native,
                    system_native_images=sorted(system))

    import numpy  # noqa: F401 -- explicitly imported only after admission
    import mlx.core  # noqa: F401
    import mlx.nn  # noqa: F401
    initial = origins()

    def endpoint():
        require(paths(sources) == files, 'Conversion runtime inventory changed')
        for p in files:
            check_file(p)
        return origins()

    return dict(manifest_sha256=expected, files=len(files), versions=VERSIONS,
                pycache_prefix=sys.pycache_prefix, imports=selected,
                initial_origins=initial, streaming_hash_chunk_limit_bytes=HASH_CHUNK), endpoint
