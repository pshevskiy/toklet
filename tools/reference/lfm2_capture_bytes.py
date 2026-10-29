"""Bounded native evidence parsing and physical-state reconstruction.

No source or model forward. Intentional failed-command NaN stamps are raw words;
only successful selected operands enter the separate conformance dispatcher.
This private draft is not an executable qualification entry point.
"""
import hashlib
import math
import mmap
import struct

import numpy as np
from lfm2_capture_contract import (
    ATTENTION, CONVOLUTION, File, HALF_PAD, WORD_PAD, STAGES,
    integer, parse_json, require,
)

RAW_CAP = 256 * 1024**2
RAW_PLANNED = 155483328
JOURNAL_CAP = 16 * 1024**2
DTYPES = {'f16le': np.dtype('<u2'), 'f32le': np.dtype('<u4'), 'u8': np.dtype('u1')}
ARRAY_KEYS = {'kind', 'seq', 'offset', 'bytes', 'sha256', 'dtype', 'id', 'shape', 'rank', 'query', 'keys'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def hash_words(words):
    require(words.flags.c_contiguous, 'physical word hash requires contiguous observation')
    return digest(memoryview(words).cast('B'))


def same_words(a, b, message):
    require(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), message)


def numeric_head_hash(words):
    require(words.dtype == np.dtype('<u4') and words.size == 65536, 'full numeric head hash')
    value = words.view('<f4')
    require(np.isfinite(value).all(), 'nonfinite successful head')
    normalized = words.copy()
    normalized[(normalized & 0x7fffffff) == 0] = 0
    return hash_words(normalized)


def array_layout(record, raw_size, expected_offset):
    require(type(record) is dict and set(record) == ARRAY_KEYS and record['kind'] == 'array',
            'exact array descriptor keys')
    require(record['dtype'] in DTYPES and type(record['id']) is list and len(record['id']) == 4,
            'array dtype/id')
    for value in record['id']:
        integer(value, 0, 15000, 'array identity integer')
    integer(record['id'][0], 1, 5, 'forward-owning Session id')
    integer(record['id'][1], 0, 104, 'global attempt id')
    integer(record['id'][2], 0, 11, 'array kind')
    rank = integer(record['rank'], 1, 3, 'array rank')
    require(type(record['shape']) is list and len(record['shape']) == 3, 'shape3')
    for value in record['shape']:
        integer(value, 1, RAW_CAP, 'array dimension')
    require(all(x == 1 for x in record['shape'][rank:]), 'unused dimensions must be one')
    for key in ('query', 'keys'):
        require(type(record[key]) is list and len(record[key]) == 2, 'absolute interval pair')
        for value in record[key]:
            integer(value, 0, 512, 'absolute interval value')
        require(sum(record[key]) <= 512, 'absolute interval overflow')
    count = math.prod(record['shape'])
    size = count * DTYPES[record['dtype']].itemsize
    require(record['offset'] == expected_offset and type(record['offset']) is int and
            record['bytes'] == size and type(record['bytes']) is int and
            size > 0 and expected_offset + size <= raw_size, 'contiguous exact raw extent')
    require(type(record['sha256']) is str and len(record['sha256']) == 64 and
            set(record['sha256']) <= set('0123456789abcdef'), 'array SHA encoding')
    return expected_offset + size


class Capture:
    """Metadata index plus separately accounted read-only native raw mapping."""
    def __init__(self, raw_path, journal_path, summary):
        require(set(summary) == {'raw_bytes','raw_sha256','journal_bytes','journal_sha256',
                'records','arrays','boundaries','other'}, 'complete summary schema')
        for name, cap in (('raw_bytes',RAW_PLANNED),('journal_bytes',JOURNAL_CAP),
                          ('records',17161),('arrays',14672),('boundaries',160),('other',2000)):
            integer(summary[name],0,cap,'summary count')
        self.raw_file = self.journal_file = self.map = None
        try:
            self.raw_file = File(raw_path, RAW_CAP,
                                 {'bytes': summary['raw_bytes'], 'sha256': summary['raw_sha256']})
            self.journal_file = File(journal_path, JOURNAL_CAP,
                                     {'bytes': summary['journal_bytes'], 'sha256': summary['journal_sha256']})
            require(self.raw_file.bytes <= RAW_PLANNED, 'prospectively planned raw upper bound')
            self.map = mmap.mmap(self.raw_file.file.fileno(), 0, access=mmap.ACCESS_READ)
            self._index(summary)
        except BaseException:
            self.close()
            raise

    def _index(self, summary):
        self.records = []
        self.by_key = {}
        self.by_seq = {}
        self.events = []
        offset = 0
        arrays = boundaries = other = 0
        self.journal_file.file.seek(0)
        while line := self.journal_file.file.readline(262145):
            seq = len(self.records)
            require(line.endswith(b'\n') and len(line) <= 262144, 'complete capped JSONL record')
            record = parse_json(line)
            require(type(record) is dict and record.get('seq') == seq and type(record['seq']) is int,
                    'monotonic journal sequence')
            if record.get('kind') == 'array':
                require(seq > 0 and len(line) <= 384, 'array record cap/order')
                offset = array_layout(record, self.raw_file.bytes, offset)
                view = memoryview(self.map)[record['offset']:offset]
                actual_hash = digest(view)
                del view
                require(actual_hash == record['sha256'], 'native array hash')
                arrays += 1
                kind = record['id'][2]
                key = tuple(record['id']) + tuple(record['query'])
                require(key not in self.by_key, 'duplicate native operand identity')
                self.by_key[key] = record
                self.by_seq[seq] = record
            else:
                require(set(record) == {'seq', 'kind', 'body'} and type(record['kind']) is str and
                        type(record['body']) is dict, 'exact event envelope')
                if seq == 0:
                    require(record['kind'] == 'start', 'first event must be start')
                elif record['kind'] in ('boundary', 'session_owners', 'session_begin', 'source_drop'):
                    require(len(line) <= 40000, 'boundary event cap')
                    boundaries += 1
                else:
                    require(len(line) <= 512 and record['kind'] != 'start', 'other event cap')
                    other += 1
                self.events.append(record)
            self.records.append(record)
            require(arrays <= 15000 and boundaries <= 160 and other <= 2000, 'journal inventory cap')
        require(offset == self.raw_file.bytes, 'unreferenced native raw suffix')
        require((len(self.records), arrays, boundaries, other) ==
                (summary['records'], summary['arrays'], summary['boundaries'], summary['other']),
                'completed writer inventory mismatch')

    def words(self, descriptor):
        require(self.by_seq.get(descriptor['seq']) is descriptor, 'unregistered raw descriptor')
        return np.frombuffer(self.map, dtype=DTYPES[descriptor['dtype']],
                             count=math.prod(descriptor['shape']), offset=descriptor['offset']).reshape(
                                 descriptor['shape'][:descriptor['rank']])

    def values(self, descriptor):
        require(descriptor['dtype'] in ('f16le', 'f32le'), 'numerical descriptor dtype')
        words = self.words(descriptor)
        values = words.view('<f2' if descriptor['dtype'] == 'f16le' else '<f4').astype('<f4')
        require(np.isfinite(values).all(), 'successful numerical operand nonfinite')
        return values

    def get(self, session, attempt, kind, index, query=(0, 0)):
        key = (session, attempt, kind, index, *query)
        require(key in self.by_key, 'missing actual captured operand')
        return self.by_key[key]

    def endpoint(self):
        self.raw_file.endpoint(); self.journal_file.endpoint()

    def close(self):
        for owner in (self.map, self.raw_file, self.journal_file):
            if owner is not None:
                owner.close()


def expected_owner_bytes(diagnostics, all_logits):
    """Independent concrete physical sizes; no Rust metadata-size guesses."""
    sizes = {}
    for owner in range(6):
        for role in range(4):
            sizes[4 * owner + role] = ((8 * (512 if role < 2 else 32) * 64) + 7) * 2
    for owner in range(20):
        sizes[24 + owner] = 2055 * 2
    widths = (1024, 1024, 1024, 3072, 1024, 1024, 1024, 1024, 512, 512, 512, 4608, 4608, 4608)
    for i, width in enumerate(widths):
        sizes[44 + i] = (32 * width + 7) * 2
    sizes[58] = sizes[59] = (16 * 32 * 512 + 7) * 4
    sizes[60] = (32 + 7) * 4
    sizes[61] = sizes[62] = ((32 if all_logits else 1) * 65536 + 7) * 4
    sizes[63] = 8 * 4
    if diagnostics:
        require(all_logits, 'diagnostics requires all-row logits')
        for slot, stage in enumerate(STAGES):
            count = 16 * 32 * 512 if stage.name.endswith(
                ('attention_scores', 'attention_probabilities')) else 32 * stage.width
            scalar = 2 if stage.dtype == 'f16le' else 4
            sizes[64 + slot] = ((count + 7) * scalar + 3) // 4 * 4
    require(len(sizes) == (357 if diagnostics else 64), 'physical owner inventory')
    return sizes


def expected_guard(id_):
    if id_ < 58:
        return struct.pack('<7H', *([HALF_PAD] * 7))
    if id_ < 64:
        return struct.pack('<7I', *([WORD_PAD] * 7))
    stage = STAGES[id_ - 64]
    if stage.dtype == 'f16le':
        return struct.pack('<8H', *([HALF_PAD] * 8))
    return struct.pack('<7I', *([WORD_PAD] * 7))


class PhysicalState:
    """One Session's actual initial KV plus all observed writes.

    Buffers are independent owned u16 copies (6,685,008 bytes total). They are
    never expanded to Q heads. Reset keeps dirty tails. Ordinary sessions have
    no producer evidence; their whole-prefix comparisons use the independently
    captured primary trajectory, with that narrower observation scope reported.
    """
    def __init__(self, capture, session, diagnostics, all_logits, model_addresses=()):
        self.capture, self.session = capture, session
        self.sizes = expected_owner_bytes(diagnostics, all_logits)
        self.diagnostics = diagnostics
        self.table = None
        self.last = None
        self.kv = {}
        self.model_addresses = set(model_addresses)
        self.device = None
        self.offset = 0
        self.tokens = np.zeros(512, dtype='<u4')
        self.history = {}
        self.boundaries = self.mutated_copy_words = 0

    def initial_kv(self):
        require(not self.kv, 'duplicate initial cache')
        for id_ in range(24):
            d = self.capture.get(self.session, 0, 6, id_)
            require(d['dtype'] == 'f16le' and d['rank'] == 1 and
                    d['shape'] == [self.sizes[id_] // 2, 1, 1] and d['keys'] == [0, 0],
                    'initial actual physical cache layout')
            words = self.capture.words(d).copy()
            require(np.all(words == HALF_PAD), 'initial actual cache canaries')
            self.kv[id_] = words

    def owner_table(self, rows):
        require(self.table is None and len(rows) == len(self.sizes), 'one complete Session owner table')
        table = {}
        addresses = set()
        for row in rows:
            require(type(row) is list and len(row) == 4, 'physical table [id,address,bytes,device]')
            id_, address, size, device = row
            require(type(id_) is int and id_ in self.sizes and id_ not in table, 'owner id')
            integer(address, 1, (1 << 64) - 1, 'actual owner address')
            integer(device, 1, (1 << 64) - 1, 'actual device identity')
            require(size == self.sizes[id_] and address not in addresses and
                    address not in self.model_addresses, 'physical size/owner alias')
            if self.device is None:
                self.device = device
            require(device == self.device, 'one-device Session')
            addresses.add(address); table[id_] = (address, size, device)
        require(set(table) == set(self.sizes), 'all physical owners required')
        ranges = sorted((address, address + size) for address, size, _ in table.values())
        require(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), 'overlapping physical ranges')
        self.table = table

    def initial_or_reset_histories(self, descriptor):
        require(descriptor['id'][0] == self.session and descriptor['id'][2] == 11 and
                0 <= descriptor['id'][3] <= 17 and
                descriptor['dtype'] == 'f16le' and descriptor['rank'] == 3 and
                descriptor['shape'] == [10, 2, 2055] and descriptor['keys'] == [0, 0] and
                descriptor['query'] == [0, 0], 'actual initial/reset history stack')
        words = self.capture.words(descriptor).reshape(20, 2055)
        require(np.all(words[:, :2048] == 0) and np.all(words[:, -7:] == HALF_PAD),
                'initial/reset actual histories are not cleared with intact guards')
        if self.history:
            for index in range(20):
                same_words(words[index], self.history[index], 'reset stack differs from declared reset writes')
        self.history = {index: value.copy() for index, value in enumerate(words)}

    def boundary(self, body):
        require(self.table is not None and body['session'] == self.session, 'Session boundary identity')
        owners = body['owners']
        require(len(owners) == len(self.table), 'complete boundary owners')
        observed = {}
        addresses = set()
        for row in owners:
            require(type(row) is list and len(row) == 3, 'compact boundary [id,address,hash]')
            id_, address, sha = row
            require(type(id_) is int and id_ in self.table and id_ not in observed and address not in addresses,
                    'boundary owner id/alias')
            # Role swaps are limited to each individual history pair and logits.
            candidates = [id_]
            if 24 <= id_ < 44:
                candidates = [24 + ((id_ - 24) // 2) * 2, 25 + ((id_ - 24) // 2) * 2]
            elif id_ in (61, 62):
                candidates = [61, 62]
            require(address in [self.table[i][0] for i in candidates], 'unexpected owner relocation')
            require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef'),
                    'boundary SHA encoding')
            observed[id_] = (address, sha); addresses.add(address)
        require(set(observed) == set(self.table), 'boundary completeness')
        guard = self.capture.by_seq[body['guard_array']]
        require(guard['id'][:3] == [self.session, body['attempt'], 9] and guard['dtype'] == 'u8',
                'guard observation binding')
        expected = b''.join(expected_guard(row[0]) for row in owners)
        require(self.capture.words(guard).tobytes() == expected, 'all actual physical guard tails')
        require(len(expected) == (5824 if self.diagnostics else 980), 'guard ledger')
        meta = body['cache']; status = meta['status']
        require(len(status['words']) == 8 and status['words'][1:] == [WORD_PAD] * 7 and
                status['address'] == observed[63][0] and status['bytes'] == 32 and
                status['device'] == self.device and
                digest(struct.pack('<8I', *status['words'])) == observed[63][1], 'actual status owner')
        for word in status['words']:
            integer(word,0,2**32-1,'status word')
        command = meta['command']
        if command is not None:
            require(command['terminal'] is True or command['status'] == 'MTLCommandBufferStatus(0)',
                    'nonterminal CPU observation')
        tokens = body['tokens']
        require(type(tokens) is list and len(tokens) == 512 and
                all(type(v) is int and 0 <= v <= 65535 for v in tokens), 'full physical token vector')
        require(np.array_equal(np.asarray(tokens, dtype='<u4'), self.tokens), 'publication token bytes')
        for id_, words in self.kv.items():
            require(hash_words(words) == observed[id_][1], 'physical KV hash reconstruction')
        for index, words in self.history.items():
            require(hash_words(words) == observed[24 + index][1], 'complete history physical hash')
        require(meta['offset'] == self.offset, 'cache published offset')
        self.last = body
        self.boundaries += 1
        return observed

    def raw_histories(self, attempt, kind):
        result = {}
        for index in range(20):
            d = self.capture.get(self.session, attempt, kind, index)
            require(d['dtype'] == 'f16le' and d['rank'] == 1 and d['shape'] == [2055, 1, 1] and
                    d['keys'] == [0, 0], 'actual guarded history bank')
            result[index] = self.capture.words(d)
            require(np.all(result[index][-7:] == HALF_PAD), 'history guard words')
        return result

    def reconstruct(self, attempt, start, tokens, fault=None):
        """Update predicted physical bytes solely from actual captured producers."""
        require(self.diagnostics and start == self.offset and 1 <= len(tokens) <= 32,
                'diagnostic actual-write reconstruction precondition')
        require(fault in (None, 'OmitHistory9', 'Head', 'FirstCopy'), 'frozen fault inventory')
        n = len(tokens)
        pre = self.raw_histories(attempt, 4)
        post = self.raw_histories(attempt, 5)
        if self.history:
            for index in range(20):
                same_words(pre[index], self.history[index], 'pre-history actual prior publication')
        for owner, layer in enumerate(CONVOLUTION):
            d = self.capture.get(self.session, attempt, 1, layer, (start, n))
            require(d['shape'] == [n, 1024, 1] and d['rank'] == 2 and d['dtype'] == 'f16le',
                    'full actual u producer')
            u = self.capture.words(d)
            candidate = np.full(2055, HALF_PAD, dtype='<u2')
            if fault == 'OmitHistory9' and owner == 9:
                candidate[:2048] = 0x7e7e
            elif n == 1:
                candidate[:1024] = pre[2 * owner][1024:2048]
                candidate[1024:2048] = u[0]
            else:
                candidate[:2048] = u[-2:].reshape(-1)
            if fault is None:
                same_words(post[2 * owner], candidate, 'published raw-u suffix')
                same_words(post[2 * owner + 1], pre[2 * owner], 'swapped prior old bank')
            else:
                same_words(post[2 * owner], pre[2 * owner], 'failed old history preserved')
                same_words(post[2 * owner + 1], candidate, 'failed candidate raw writes/stamps')
        for owner, layer in enumerate(ATTENTION):
            for role, kind in ((0, 2), (1, 3)):
                d = self.capture.get(self.session, attempt, kind, layer, (start, n))
                require(d['shape'] == [n, 512, 1] and d['rank'] == 2 and d['dtype'] == 'f16le',
                        'full actual Krot/V producer')
                source = self.capture.words(d).reshape(n, 8, 64)
                staged = self.kv[4 * owner + role + 2][:-7].reshape(8, 32, 64)
                staged[:, :n] = source.transpose(1, 0, 2)
                if fault is None:
                    self.kv[4 * owner + role][:-7].reshape(8, 512, 64)[:, start:start + n] = staged[:, :n]
                elif fault == 'FirstCopy' and owner == 0 and role == 0:
                    target = self.kv[0][:-7].reshape(8, 512, 64)[0, start:start + n]
                    self.mutated_copy_words += int(np.count_nonzero(target != staged[0, :n]))
                    require(self.mutated_copy_words == 128 and n == 2, 'exact128 actual first-run word mutations')
                    target[:] = staged[0, :n]
        if fault is None:
            self.tokens[start:start + n] = tokens
            self.offset = start + n
        self.history = {index: words.copy() for index, words in post.items()}

    def reset(self):
        """No KV, token, logit or role mutation is inferred from reset."""
        self.offset = 0
        for bank in self.history.values():
            bank[:2048] = 0

    def active(self, attempt):
        require(self.offset > 0, 'no active prefix at zero offset')
        for owner in range(6):
            for role in range(2):
                d = self.capture.get(self.session, attempt, 7, 2 * owner + role)
                require(d['dtype'] == 'f16le' and d['rank'] == 3 and
                        d['shape'] == [8, self.offset, 64] and d['keys'] == [0, self.offset],
                        'actual active prefix layout')
                expected = self.kv[4 * owner + role][:-7].reshape(8, 512, 64)[:, :self.offset]
                same_words(self.capture.words(d), expected, 'active KV prefix readback')

    def ordinary_commit(self, attempt, start, tokens):
        """Ordinary observations: actual terminal prefixes, no invented u/Krot/V.

        Full old/staged hashes must agree with observed terminal active prefixes
        and the reviewed copy layout. This checks physical consistency, not an
        independent ordinary producer or per-operator numerical reconstruction.
        Published prefix/history/head additionally compare to the primary run.
        """
        require(not self.diagnostics and start == self.offset and len(tokens) == 32,
                'ordinary fixed32-row attempt')
        end = start + 32
        pre = self.raw_histories(attempt, 4)
        post = self.raw_histories(attempt, 5)
        if self.history:
            for index in range(20):
                same_words(pre[index], self.history[index], 'ordinary pre-history continuity')
        for owner in range(10):
            same_words(post[2 * owner + 1], pre[2 * owner], 'ordinary previous old bank preserved by swap')
        for owner in range(6):
            for role in range(2):
                d = self.capture.get(self.session, attempt, 7, owner * 2 + role)
                require(d['dtype'] == 'f16le' and d['rank'] == 3 and
                        d['shape'] == [8, end, 64] and d['keys'] == [0, end], 'ordinary active prefix')
                actual = self.capture.words(d)
                old = self.kv[owner * 4 + role][:-7].reshape(8, 512, 64)
                same_words(actual[:, :start], old[:, :start], 'ordinary prior prefix preservation')
                old[:, :end] = actual
                self.kv[owner * 4 + role + 2][:-7].reshape(8, 32, 64)[:] = actual[:, start:end]
        self.tokens[start:end] = tokens; self.offset = end
        self.history = {i: words.copy() for i, words in post.items()}
