"""Ordered source reads and bounded row workers for reference-only projections."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import struct
import threading
import affine4_tiles as tiles
from gemma_native_projection import Projection
import gemma_projection_domain as domain

_worker = threading.local()


def _initialize(library):
    _worker.projection = Projection(library)


def _tile(blocks, inputs, tokens, rows, columns, lanes):
    project = _worker.projection
    # Verify each executing worker's RNE and subnormal-preserving CPU environment
    # before the actual tile. This does not characterize the Metal environment.
    probe = project(bytes(32), bytes(2), bytes.fromhex('803f'),
                    struct.pack('<I', 1)+bytes(63*4), tokens=1, rows=1, columns=64, lanes=lanes)
    if probe != struct.pack('<I', 1):
        raise ValueError('CPU worker subnormal environment differs')
    return project(*blocks, inputs, tokens=tokens, rows=rows, columns=columns, lanes=lanes)


def project_rows(matrix, inputs, *, tokens, lanes, library, workers=1, normal_domain=False, matrix_name='unnamed'):
    """Return FP32 [tokens, rows], with at most four independent 128-row jobs.

    The caller authenticates the immutable source/compiled library and owns the
    borrowed seekable descriptor. Only this calling thread reads that descriptor;
    workers receive separate immutable tiles and preserve each full dot/tree.
    Pending jobs stay bounded by workers, and results assemble by source row.
    """
    if type(matrix) is not tiles.MatrixRows or type(workers) is not int or workers not in [1, 4]:
        raise ValueError('Validated source matrix and one/four workers required')
    if type(tokens) is not int or not 1 <= tokens <= 32 or type(inputs) is not bytes or len(inputs) != tokens*matrix.columns*4:
        raise ValueError('Bounded complete FP32 input rows required')
    if type(lanes) is not int or not 1 <= lanes <= 64 or lanes & (lanes-1):
        raise ValueError('Supported queried SIMD width required')
    if type(normal_domain) is not bool or type(matrix_name) is not str or not 1 <= len(matrix_name) <= 256:
        raise ValueError('Explicit domain policy and bounded matrix identity required')
    try:
        input_quantum = domain.input_grid(inputs) if normal_domain else None
    except domain.DomainError as error:
        error.evidence.update(matrix=matrix_name, tile_start=None)
        raise
    output = bytearray(tokens*matrix.rows*4)
    pending = deque()

    def finish():
        start, count, future = pending.popleft()
        block = future.result()
        for token in range(tokens):
            destination = (token*matrix.rows+start)*4
            output[destination:destination+count*4] = block[token*count*4:(token+1)*count*4]

    with ThreadPoolExecutor(max_workers=workers, initializer=_initialize, initargs=(library,)) as executor:
        for start in range(0, matrix.rows, 128):
            count = min(128, matrix.rows-start)
            blocks = []
            for offset, stride in matrix.components():
                matrix.stream.seek(offset+start*stride)
                block = matrix.stream.read(count*stride)
                if type(block) is not bytes or len(block) != count*stride:
                    raise ValueError('Truncated immutable packed tile')
                blocks.append(block)
            if normal_domain:
                try:
                    domain.admit_metadata(blocks[1], blocks[2], input_quantum)
                except domain.DomainError as error:
                    error.evidence.update(matrix=matrix_name, tile_start=start)
                    raise
            pending.append((start, count, executor.submit(_tile, blocks, inputs, tokens, count, matrix.columns, lanes)))
            if len(pending) == workers:
                finish()
        while pending:
            finish()
    return bytes(output)
