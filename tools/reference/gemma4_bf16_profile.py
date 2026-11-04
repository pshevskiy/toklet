"""Observational BF16 reference timings; no weight cache or arithmetic changes.

Intervals include their named validation/guard work. Projection remainder also
includes FP64 conversion and framework overhead; it is not a pure GEMM timer.
Only successfully read/decoded bytes count as returned bytes. Failed attempts
retain their timing and failure count without changing the original exception.
This is a single-caller diagnostic. Inclusive and remainder buckets overlap;
returned-byte counters describe traffic, not resident memory or allocations.
"""
import time
import gemma4_text_math as graph


class Ledger:
    def __init__(self, clock_ns=time.perf_counter_ns):
        self.clock_ns = clock_ns
        self.operations = {}
        self.raw_ns = 0
        self.decode_ns = 0

    def add(self, name, operation, elapsed_ns, *, bytes_returned=0, failed=False):
        row = self.operations.setdefault((name, operation),
                                         dict(calls=0, elapsed_ns=0, bytes_returned=0, failed_calls=0))
        row['calls'] += 1
        row['elapsed_ns'] += elapsed_ns
        row['bytes_returned'] += bytes_returned
        row['failed_calls'] += int(failed)
        if operation == 'read_guard_hash':
            self.raw_ns += elapsed_ns
        elif operation == 'decode_validate_guard':
            self.decode_ns += elapsed_ns

    def records(self):
        return [dict(matrix=name, operation=operation, **row)
                for (name, operation), row in sorted(self.operations.items())]


class Rows:
    """Borrow the authenticated reader and preserve every raw/decode call."""
    def __init__(self, name, inner, ledger):
        self.name, self.inner, self.ledger = name, inner, ledger
        self.rows, self.columns = inner.rows, inner.columns

    def _raw(self, request, *args):
        start = self.ledger.clock_ns()
        raw = None
        try:
            raw = request(*args)
            return raw
        finally:
            self.ledger.add(self.name, 'read_guard_hash', self.ledger.clock_ns()-start,
                            bytes_returned=0 if raw is None else len(raw), failed=raw is None)

    def _decode(self, raw, count):
        start = self.ledger.clock_ns()
        value = None
        try:
            value = self.inner._decode(raw, count)
            return value
        finally:
            self.ledger.add(self.name, 'decode_validate_guard', self.ledger.clock_ns()-start,
                            bytes_returned=0 if value is None else value.nbytes, failed=value is None)

    def read_rows(self, start, end):
        return self._decode(self._raw(self.inner.read_raw, start, end), end-start)

    def gather(self, ids):
        return self._decode(self._raw(self.inner.gather_raw, ids), len(ids))


class Weights(graph.Weights):
    """Original reference operators over instrumented borrowed BF16 readers."""
    def __init__(self, original, ledger):
        self.ledger = ledger
        super().__init__(original.profile,
                         {name: Rows(name, rows, ledger) for name, rows in original.matrices.items()},
                         original.vectors)

    def linear(self, name, x, dtype):
        start = self.ledger.clock_ns()
        raw_before, decode_before = self.ledger.raw_ns, self.ledger.decode_ns
        value = None
        try:
            value = super().linear(name, x, dtype)
            return value
        finally:
            elapsed = self.ledger.clock_ns()-start
            nested = self.ledger.raw_ns-raw_before + self.ledger.decode_ns-decode_before
            group = 'tied_head' if name == 'embed_tokens.weight' else 'projection'
            self.ledger.add(name, group+'_inclusive', elapsed,
                            bytes_returned=0 if value is None else value.nbytes, failed=value is None)
            self.ledger.add(name, group+'_fp64_conversion_compute_framework', elapsed-nested,
                            failed=value is None)
