"""Read authenticated Granite child tiles without mapping or retaining a model.

The caller supplies an externally pinned conversion layout, complete tile
journal and norm receipts. These are byte-consumption checks, not admission of
a numerical runtime or a managed model. No numerical package is imported here.
"""
import hashlib
import os
from pathlib import Path
import stat

from granite_quantize import need, ROWS
from reference_runtime import sha

MAX_READ = 4 * 1024**2


def identity(value):
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def read_exact(fd, count, offset):
    need(type(count) is int and 0 <= count <= MAX_READ
         and type(offset) is int and offset >= 0, 'Bounded exact read required')
    result = bytearray()
    while len(result) < count:
        raw = os.pread(fd, count - len(result), offset + len(result))
        need(bool(raw), 'Child descriptor ended before its declared extent')
        result.extend(raw)
    return bytes(result)


def tile_bindings(planned, rows, norms):
    """Bind every complete conversion tile before opening a weight descriptor."""
    need(type(rows) is list and type(norms) is list, 'Explicit tile/norm inventories required')
    expected = {}
    for name, matrix in planned['matrices'].items():
        count, columns = matrix['shape']
        for first in range(0, count, ROWS):
            expected[name, first] = min(ROWS, count - first), columns, matrix
    tiles = {}
    for row in rows:
        need(type(row) is dict and type(row.get('name')) is str
             and type(row.get('first_row')) is int, 'Typed conversion tile identity required')
        key = row['name'], row['first_row']
        need(key in expected and key not in tiles, 'Unique declared conversion tile required')
        count, columns, matrix = expected[key]
        for field, value in [('rows', count), ('columns', columns), ('bits', matrix['bits'])]:
            need(type(row.get(field)) is int and row[field] == value,
                 'Conversion tile geometry differs: ' + field)
        fields = row.get('fields')
        need(type(fields) is dict and set(fields) == set(matrix['fields']),
             'Complete conversion tile components required')
        pins = {}
        for field, item in matrix['fields'].items():
            pin = fields[field]
            need(type(pin) is dict and set(pin) == {'bytes', 'sha256'}
                 and type(pin['bytes']) is int and pin['bytes'] == count * item['row_stride'],
                 'Exact conversion component extent required')
            pins[field] = dict(bytes=pin['bytes'], sha256=sha(pin['sha256']))
        tiles[key] = dict(rows=count, fields=pins)
    need(set(tiles) == set(expected), 'Complete matrix tile coverage required')
    expected_norms = {name for name, shape in planned['logical'].items() if len(shape) == 1}
    norm_pins = {}
    for row in norms:
        need(type(row) is dict and type(row.get('name')) is str
             and row['name'] in expected_norms and row['name'] not in norm_pins,
             'Unique declared norm receipt required')
        name = row['name']
        start, end = planned['header'][name]['data_offsets']
        need(type(row.get('bytes')) is int and row['bytes'] == end - start,
             'Exact norm receipt extent required')
        norm_pins[name] = dict(bytes=row['bytes'], sha256=sha(row['copied_BF16_sha256']))
    need(set(norm_pins) == expected_norms, 'Complete norm coverage required')
    return tiles, norm_pins


class Child:
    """One regular FD; <=4 MiB reads and exact consumed-component hashes.

    Whole-file admission binds unrequested data too. Every actual matrix/norm
    buffer is checked again against externally pinned conversion receipts before
    it can be passed to a numerical decoder. Endpoint identity rejects mutation
    even when it affects an unrequested range. No source index is synthesized.
    """
    def __init__(self, path, planned, expected_file, tiles, norms):
        need(type(expected_file) is dict and set(expected_file) == {'bytes', 'sha256'}
             and type(expected_file['bytes']) is int
             and 0 < expected_file['bytes'] == planned['file_bytes'] <= 6 * 1024**3,
             'Exact bounded child file identity required')
        self.expected_sha = sha(expected_file['sha256'])
        self.tiles, self.norms = tile_bindings(planned, tiles, norms)
        self.path, self.planned = Path(path), planned
        self.fd = None
        self.open_identity = None
        self.component_reads = self.norm_reads = self.consumed_bytes = 0

    def _named_identity(self):
        need(self.fd is not None, 'Child descriptor is closed')
        owned, named = os.fstat(self.fd), self.path.lstat()
        need(stat.S_ISREG(owned.st_mode) and stat.S_ISREG(named.st_mode)
             and identity(owned) == identity(named), 'Child descriptor/named identity differs')
        return identity(owned)

    def __enter__(self):
        need(self.fd is None and self.open_identity is None, 'Child reader cannot be reopened')
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        self.fd = os.open(self.path, flags)
        try:
            self.open_identity = self._named_identity()
            need(self.open_identity[2] == self.planned['file_bytes'], 'Child file extent differs')
            need(read_exact(self.fd, self.planned['origin'], 0) == self.planned['prefix'],
                 'Exact externally planned child header required')
            digest = hashlib.sha256()
            size = self.planned['file_bytes']
            for start in range(0, size, 1024**2):
                digest.update(read_exact(self.fd, min(1024**2, size - start), start))
            need(os.pread(self.fd, 1, size) == b'' and digest.hexdigest() == self.expected_sha,
                 'Complete child object SHA256 differs')
            self.endpoint()
            return self
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def endpoint(self):
        need(self.open_identity is not None and self._named_identity() == self.open_identity,
             'Child changed during reference consumption')
        return dict(file_sha256=self.expected_sha, component_reads=self.component_reads,
                    norm_reads=self.norm_reads, consumed_bytes=self.consumed_bytes)

    def tile(self, name, first):
        need(self.fd is not None and type(first) is int and (name, first) in self.tiles,
             'Open child and exact declared matrix tile required')
        matrix = self.planned['matrices'][name]
        pin = self.tiles[name, first]
        result = {}
        for field, item in matrix['fields'].items():
            start = self.planned['header'][item['name']]['data_offsets'][0]
            raw = read_exact(self.fd, pin['fields'][field]['bytes'],
                             self.planned['origin'] + start + first * item['row_stride'])
            need(hashlib.sha256(raw).hexdigest() == pin['fields'][field]['sha256'],
                 'Actual consumed child component differs: ' + name + '.' + field)
            result[field] = raw
        # Count only complete, authenticated three-component consumptions.
        self.component_reads += 3
        self.consumed_bytes += sum(map(len, result.values()))
        return result

    def norm(self, name):
        need(self.fd is not None and name in self.norms, 'Open child and declared norm required')
        start = self.planned['header'][name]['data_offsets'][0]
        pin = self.norms[name]
        raw = read_exact(self.fd, pin['bytes'], self.planned['origin'] + start)
        need(hashlib.sha256(raw).hexdigest() == pin['sha256'], 'Actual consumed norm bytes differ')
        self.norm_reads += 1
        self.consumed_bytes += len(raw)
        return raw

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                self.endpoint()
        finally:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
