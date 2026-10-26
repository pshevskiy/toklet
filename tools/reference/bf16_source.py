"""Bounded reference access to immutable BF16 weights, outside the Rust engine.

One owner authenticates a regular file through its retained descriptor. Concrete
row views borrow it; no mapping, whole-tensor expansion or unbounded read ledger.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import re
import stat
import struct
import time

MAX_FILE_BYTES = 12 * 1024**3
HASH_BLOCK = 1024**2
MAX_ROWS = 262144
MAX_COLUMNS = 12288
MAX_REQUEST_ROWS = 128
MAX_READ_BYTES = MAX_REQUEST_ROWS * MAX_COLUMNS * 2


def integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('bounded integer required')
    return value


def check_deadline(deadline, stop):
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('finite absolute deadline required')
    if stop is not None and stop.is_set():
        raise InterruptedError('reference read cancelled')
    if time.monotonic() >= deadline:
        raise TimeoutError('reference read deadline')


def identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


class SourceChanged(ValueError):
    pass


class Source:
    """Single-caller owner; caller deadlines bound authentication and each request.

    The read chain binds returned bytes and spans. It is not an independent
    per-row proof against the original whole-file hash. Identity checks retain
    the original baseline permanently and reject changed/restored sources.
    """
    def __init__(self, path, size, sha256, *, deadline, stop=None):
        integer(size, 1, MAX_FILE_BYTES)
        if type(sha256) is not str or not re.fullmatch('[0-9a-f]{64}', sha256):
            raise ValueError('external full SHA-256 required')
        check_deadline(deadline, stop)
        self.path = Path(path).absolute()
        self.size, self.sha256 = size, sha256
        self.closed, self.reason = False, None
        self._audit = (hashlib.sha256(b'toklet-bf16-source-reads-v1\0'), 0, 0)
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            value = os.fstat(self.fd)
            if not stat.S_ISREG(value.st_mode) or value.st_size != size:
                raise ValueError('regular exact-size source required')
            self.original = identity(value)
            self._guard()
            digest, offset = hashlib.sha256(), 0
            while True:
                check_deadline(deadline, stop)
                raw = os.pread(self.fd, min(HASH_BLOCK, size - offset + 1), offset)
                if not raw:
                    break
                offset += len(raw)
                if offset > size:
                    raise SourceChanged('source grew during authentication')
                digest.update(raw)
            self._guard()
            if offset != size or digest.hexdigest() != sha256:
                raise ValueError('source authentication differs')
        except BaseException as error:
            try:
                os.close(self.fd)
            except OSError as cleanup:
                error.add_note('BF16 authentication cleanup: ' + str(cleanup))
            self.closed = True
            raise

    @property
    def poisoned(self):
        return self.reason is not None

    def poison(self, reason):
        if not self.poisoned:
            self.reason = str(reason)

    def _guard(self):
        if self.closed:
            raise ValueError('source is closed')
        if self.poisoned:
            raise SourceChanged('source permanently poisoned: ' + self.reason)
        try:
            if identity(os.fstat(self.fd)) != self.original or identity(self.path.lstat()) != self.original:
                raise SourceChanged('source descriptor/path identity changed')
        except (OSError, SourceChanged) as error:
            self.poison(str(error))
            raise SourceChanged(str(error)) from error

    def read(self, offset, count, *, deadline, stop=None):
        integer(offset, 0, self.size - 1)
        integer(count, 1, min(MAX_READ_BYTES, self.size - offset))
        chain, reads, read_bytes = self._audit
        integer(reads + 1, 1, 2**64 - 1)
        integer(read_bytes + count, 1, 2**64 - 1)
        check_deadline(deadline, stop)  # Caller rejection does not poison weights.
        self._guard()
        try:
            raw = os.pread(self.fd, count, offset)
            if len(raw) != count:
                raise SourceChanged('short reference row read')
            self._guard()
        except OSError as error:
            self.poison('source IO failure: ' + str(error))
            raise SourceChanged(self.reason) from error
        except SourceChanged as error:
            self.poison(str(error))
            raise
        check_deadline(deadline, stop)
        next_chain = chain.copy()
        next_chain.update(struct.pack('<QQQ', reads + 1, offset, count))
        next_chain.update(hashlib.sha256(raw).digest())
        self._audit = (next_chain, reads + 1, read_bytes + count)
        return raw

    def audit(self):
        chain, reads, read_bytes = self._audit
        return dict(bytes=self.size, sha256=self.sha256, reads=reads,
                    read_bytes=read_bytes, read_chain_sha256=chain.hexdigest(),
                    poisoned=self.poisoned, poison_reason=self.reason, closed=self.closed)

    def close(self):
        if self.closed:
            return
        try:
            if not self.poisoned:
                self._guard()
        finally:
            self.closed = True
            os.close(self.fd)

    def __enter__(self):
        self._guard()
        return self

    def __exit__(self, kind, error, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if error is None:
                raise
            error.add_note('BF16 source cleanup: ' + str(cleanup))
        return False


@dataclass(frozen=True)
class Rows:
    source: Source
    rows: int
    columns: int
    offset: int
    deadline: float
    stop: object = None

    def __post_init__(self):
        if not isinstance(self.source, Source):
            raise ValueError('authenticated source owner required')
        integer(self.rows, 1, MAX_ROWS)
        integer(self.columns, 1, MAX_COLUMNS)
        integer(self.offset, 0, self.source.size - 1)
        if self.offset % 2 or self.offset + self.rows * self.columns * 2 > self.source.size:
            raise ValueError('BF16 row-major extent differs')
        check_deadline(self.deadline, self.stop)
        self.source._guard()

    def read_raw(self, start, end):
        integer(start, 0, self.rows - 1)
        integer(end, start + 1, min(self.rows, start + MAX_REQUEST_ROWS))
        stride = self.columns * 2
        return self.source.read(self.offset + start * stride, (end - start) * stride,
                                deadline=self.deadline, stop=self.stop)

    def gather_raw(self, ids):
        if type(ids) is not list or not 1 <= len(ids) <= MAX_REQUEST_ROWS:
            raise ValueError('bounded aggregate gather rows required')
        for value in ids:
            integer(value, 0, self.rows - 1)
        check_deadline(self.deadline, self.stop)
        self.source._guard()
        stride = self.columns * 2
        result = bytearray(len(ids) * stride)
        for index, value in enumerate(ids):
            result[index * stride:(index + 1) * stride] = self.read_raw(value, value + 1)
        return bytes(result)

    def _decode(self, raw, count):
        # Caller admits the numerical runtime before invoking this lazy path.
        import numpy as np
        import affine4_tiles
        try:
            value = affine4_tiles.bf16(np.frombuffer(raw, dtype='<u2').reshape(count, self.columns))
        except ValueError:
            self.source.poison('BF16 conversion rejected')
            raise
        self.source._guard()
        check_deadline(self.deadline, self.stop)
        return value

    def read_rows(self, start, end):
        return self._decode(self.read_raw(start, end), end - start)

    def gather(self, ids):
        return self._decode(self.gather_raw(ids), len(ids))


def official_rows(source, tensor_name, layout_raw, *, deadline, stop=None):
    """Bind a name/shape/offset to the exact admitted official text layout."""
    if (not isinstance(source, Source) or type(layout_raw) is not bytes
            or type(tensor_name) is not str or not tensor_name
            or source.size != 10246621918
            or source.sha256 != '2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550'
            or len(layout_raw) > 2 * 1024**2
            or hashlib.sha256(layout_raw).hexdigest() != '2419f70e7232fd75a6d361e0ef8e8acaf37f87f52124e63e1e04e814d1040840'):
        raise ValueError('exact official source/layout identity required')
    entry = json.loads(layout_raw)['selected_text'].get(tensor_name)
    if entry is None or entry['dtype'] != 'BF16':
        raise ValueError('selected official text tensor required')
    shape = entry['shape']
    rows, columns = (shape[0], shape[1]) if len(shape) == 2 else (1, shape[0])
    return Rows(source, rows, columns, entry['file_offset'], deadline, stop)
