"""Literal pre-target workload and actual sampled-token checks.

Private reader support, no model invocation or source producer import. Dynamic
free continuations depend only on their already-saved complete native head.
"""
from dataclasses import dataclass
import numpy as np
from lfm2_capture_contract import File, PROTOCOL_SHA, integer, require, strict_equal
from lfm2_capture_bytes import numeric_head_hash

PLAIN = (1, 1024, 2048, 4096, 8192, 16384, 32768, 501)
FUTURE = PLAIN + (502, 503, 504)
JOURNEY = (1,) + tuple(501 + ((i * 997 + 37) % 63893) for i in range(63))
CASES = (('bos', (1,), (0,), ((1,),)),
         ('plain_ids8', PLAIN, (0, 7), ((8,), (3, 5))),
         ('future_tail11', FUTURE, (0, 7, 10), ((11,), (3, 5, 3))),
         ('journey64', JOURNEY, (0, 31, 32, 63), ((32, 32), (31, 1, 1, 31), (1,) * 64)))


@dataclass(frozen=True)
class Chunk:
    run: int
    session: int
    start: int
    tokens: tuple
    selected: tuple
    case: str
    fault: str | None = None

    @property
    def end(self):
        return self.start + len(self.tokens)

    @property
    def selected_queries(self):
        return tuple(row for row in self.selected if self.start <= row < self.end)


def primary_runs():
    runs = []
    for name, tokens, selected, partitions in CASES:
        for partition in partitions:
            offset = 0; chunks = []
            for count in partition:
                end = offset + count
                chunks.append(Chunk(len(runs), 1, offset, tokens[offset:end], selected, name))
                offset = end
            require(offset == len(tokens), 'literal partition coverage')
            runs.append(tuple(chunks))
    require(len(runs) == 8 and sum(map(len, runs)) == 78 and
            sum(len(chunk.tokens) for run in runs for chunk in run) == 231 and
            sum(len(chunk.selected_queries) for run in runs for chunk in run) == 23, 'primary frozen totals')
    return tuple(runs)


PRIMARY = primary_runs()
TEACHER = tuple(Chunk(8, 1, 0 if i == 0 else 7 + i,
                      PLAIN if i == 0 else (501 + i,), (), 'teacher') for i in range(4))
FAULTS = tuple(tuple((Chunk(11 + i, 2 + i, 0, PLAIN[:3], (), 'fault'),
                      Chunk(11 + i, 2 + i, 3, PLAIN[3:5], (), 'fault', fault),
                      Chunk(11 + i, 2 + i, 0, PLAIN[:3], (), 'fault'),
                      Chunk(11 + i, 2 + i, 3, PLAIN[3:8], (), 'fault')))
               for i, fault in enumerate(('OmitHistory9', 'Head', 'FirstCopy')))
ORDINARY = tuple(Chunk(14, 5, start, JOURNEY[start:start + 32], (), 'ordinary') for start in (0, 32))


def verify_protocol(path):
    f = File(path, 256 * 1024, {'bytes': 186511, 'sha256': PROTOCOL_SHA})
    try:
        protocol = f.json(256 * 1024)
        for literal, actual in zip(CASES, protocol['cases'], strict=True):
            name, tokens, selected, partitions = literal
            require((actual['name'], actual['tokens'], actual['selected_rows'], actual['native_partitions']) ==
                    (name, list(tokens), list(selected), [list(p) for p in partitions]),
                    'literal primary workload differs from frozen protocol')
        require([p['tokens'] for p in protocol['teacher']] == [list(FUTURE[:n]) for n in (8, 9, 10, 11)],
                'literal teacher histories')
        require(protocol['free_generation']['prompt_tokens'] == list(PLAIN) and
                protocol['free_generation']['maximum_sampled_tokens'] == 4 and
                protocol['free_generation']['stop_ids'] == [7], 'frozen free workload')
        f.endpoint()
        return protocol
    finally:
        f.close()


def validate_begin(body, expected, global_attempt, prior_boundary):
    require(set(body) == {'session', 'attempt', 'run', 'start', 'tokens', 'selected', 'prior_boundary'},
            'exact attempt-begin schema')
    require(type(body['attempt']) is int and body['attempt'] == global_attempt and
            type(body['session']) is int and body['session'] == expected.session and
            type(body['run']) is int and body['run'] == expected.run and
            type(body['start']) is int and body['start'] == expected.start and
            body['tokens'] == list(expected.tokens) and body['selected'] == list(expected.selected) and
            body['prior_boundary'] == prior_boundary, 'attempt differs from literal workload/current boundary')
    require(all(type(x) is int for x in body['tokens'] + body['selected']), 'exact token/query integer types')


def validate_terminal(body, expected, attempt):
    require(set(body) == {'session', 'attempt', 'result', 'error', 'phase_a', 'cache_completion',
                          'poisoned', 'offset'}, 'exact attempt-terminal schema')
    require(body['session'] == expected.session and body['attempt'] == attempt,
            'terminal attempt identity')
    require(strict_equal(body['phase_a'], [True, True]), 'actual Phase A terminal successful command required')
    if expected.fault is None:
        require(body['result'] == 'ok' and body['error'] is None and body['poisoned'] is False and
                body['offset'] == expected.end and strict_equal(body['cache_completion'], [True, True]),
                'successful numerical attempt publication/command')
    else:
        require(body['result'] == 'error' and type(body['error']) is str and body['error'] and
                body['offset'] == expected.start, 'injected failure not observed/prefix advanced')
        if expected.fault == 'FirstCopy':
            require(body['poisoned'] is True and strict_equal(body['cache_completion'], [True, True]),
                    'actual completed partial-copy poison required')
        else:
            require(body['poisoned'] is False and body['cache_completion'] is None,
                    'Phase A failure must discard command and retain valid prefix')


class FreeRepeat:
    def __init__(self, repeat):
        integer(repeat, 0, 1, 'free repeat')
        self.repeat = repeat
        self.committed = ()
        self.sampled = []
        self.hashes = []
        self.stopped = False
        self.codec_failure = None

    def next_chunk(self):
        require(not self.stopped and len(self.sampled) < 4 and self.codec_failure is None,
                'free attempt after stop/limit/codec failure')
        tokens = PLAIN if not self.sampled else (self.sampled[-1],)
        return Chunk(9 + self.repeat, 1, len(self.committed), tokens, (), 'free')

    def sample(self, words, event, descriptor_seq):
        expected = self.next_chunk()
        require(words.dtype == np.dtype('<u4') and words.shape == (65536,), 'entire native head required')
        values = words.view('<f4')
        require(np.isfinite(values).all(), 'nonfinite greedy distribution')
        token = int(np.argmax(values))  # Stable first/lowest index, no padding mask.
        self.committed += expected.tokens
        index = len(self.sampled)
        stop, unmapped = token == 7, token >= 64400
        require(set(event) == {'run', 'repeat', 'sample', 'distribution_ref', 'id', 'stop',
                              'unmapped', 'pending', 'committed_count'}, 'exact sample schema')
        require(strict_equal(event, dict(run=9 + self.repeat, repeat=self.repeat, sample=index,
                              distribution_ref=descriptor_seq, id=token, stop=stop, unmapped=unmapped,
                              pending=not stop, committed_count=len(self.committed))),
                'sample/stop/pending differs from unmasked native argmax')
        self.sampled.append(token); self.hashes.append(numeric_head_hash(words))
        self.stopped = stop
        if unmapped:
            self.codec_failure = {'id': token, 'sample': index, 'reason': 'unmapped head slot; no masking or remapping'}
        return token

    def finish(self):
        require(self.stopped or self.codec_failure is not None or len(self.sampled) == 4,
                'incomplete free repeat')
        return {'sampled': self.sampled, 'committed': list(self.committed), 'stopped': self.stopped,
                'pending': None if self.stopped else self.sampled[-1], 'codec_failure': self.codec_failure,
                'numeric_distribution_hashes': self.hashes}


def compare_free_repeats(first, second):
    require(first.finish() == second.finish(), 'same-native free repeats differ')
    require(first.codec_failure is None, 'sampled unmapped token is an explicit qualification failure')
