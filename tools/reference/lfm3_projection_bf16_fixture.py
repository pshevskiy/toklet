"""Independent pinned Torch BF16 linear outputs and NumPy FP64 dot fixtures.

Four selected physical rows preserve every real input width across all154 rank2
tensors. They establish row-wise source arithmetic, not whole-model execution.
No candidate projection reduction is implemented in this producer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

from lfm3_bf16_fixture import (
    HEADER_SHA, MODEL_BYTES, authenticate, digest, file_identity, input_pool,
    need, norm_inputs,
)

CONTRACT_SHA = 'ad76bdd309690fb7c7063930f714d26f09a49618565cce4a3aed52e305eccc6c'
RUNTIME_SHA = '1ca684c819f499a2ede6074c364b64c64c55b3fbd7fc29ccd23ff25a937202ed'


def verify_pins(pins, label):
    for path, pin in pins.items():
        need(digest(Path(path)) == pin, label+' changed')


def word_bytes(words):
    return struct.pack('<'+str(len(words))+'H', *words)


def selected_rows(rows):
    need(rows >= 4, 'At least four source rows required')
    return [0, 1, rows//2, rows-1]


def overflow_rescue(columns):
    need(columns >= 2, 'Rescue fixture requires two coordinates')
    x = [0]*columns
    w = [0]*columns
    x[0] = x[-1] = 0x7f7f
    w[0], w[-1] = 0xbf80, 0x4000
    return x, w


def export(args, policy, runtime):
    import numpy as np
    import torch
    import torch.nn.functional as functional
    need(torch.__version__ == '2.9.1' and np.__version__ == '2.3.2',
         'Pinned tensor versions differ')
    need(torch.version.git_version == policy['torch_git_commit'],
         'Pinned Torch source revision differs')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    def tensor(raw):
        return torch.from_numpy(np.frombuffer(raw, dtype='<u2').copy()).view(torch.bfloat16)

    def raw(value):
        return value.contiguous().view(torch.int16).numpy().astype('<u2').tobytes()

    blobs = {name: bytearray() for name in [
        'weights.bf16le', 'source.bf16le', 'reference.f64le',
        'corner-input.bf16le', 'corner-weight.bf16le', 'corner-source.bf16le',
        'finite-grid.bf16le',
    ]}

    def append(name, data):
        begin = len(blobs[name])
        blobs[name].extend(data)
        return dict(file=name, byte_offset=begin, byte_count=len(data),
                    sha256=hashlib.sha256(data).hexdigest())

    fd = os.open(args.model, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_size == MODEL_BYTES,
             'Canonical regular model file required')
        authenticate(fd)
        need(os.pread(fd, 8, 0) == (32368).to_bytes(8, 'little'), 'Header extent differs')
        header = os.pread(fd, 32368, 8)
        need(hashlib.sha256(header).hexdigest() == HEADER_SHA, 'Canonical header differs')
        schema = json.loads(header)
        actual = {name for name, item in schema.items()
                  if name != '__metadata__' and len(item['shape']) == 2}
        need(actual == {v['name'] for v in policy['matrices']} and len(actual) == 154,
             'Complete rank2 source inventory differs')
        args.fresh_directory.mkdir()
        pool = input_pool()
        matrices, cases = [], []
        for ordinal, entry in enumerate(policy['matrices']):
            item = schema[entry['name']]
            need(item['dtype'] == entry['dtype'] == 'BF16'
                 and item['shape'] == entry['shape']
                 and item['data_offsets'] == entry['data_offsets'], 'Source matrix layout differs')
            rows, columns = item['shape']
            selected = selected_rows(rows)
            need(selected == entry['selected_rows'], 'Selected source rows differ')
            begin, end = item['data_offsets']
            need(end-begin == rows*columns*2, 'Source matrix extent differs')
            parts = [os.pread(fd, columns*2, 32376+begin+row*columns*2) for row in selected]
            need(all(len(p) == columns*2 for p in parts), 'Short selected coefficient read')
            weight_raw = b''.join(parts)
            weight = tensor(weight_raw).reshape(4, columns)
            need(bool(torch.isfinite(weight).all()), 'Nonfinite selected source coefficient')
            matrices.append(dict(name=entry['name'], ordinal=ordinal, source_shape=item['shape'],
                selected_rows=selected, weights=append('weights.bf16le', weight_raw)))
            for tokens in policy['learned_tokens']:
                x_raw = word_bytes(norm_inputs(pool, ordinal, tokens, columns))
                x = tensor(x_raw).reshape(tokens, columns)
                output = functional.linear(x, weight)
                need(bool(torch.isfinite(output).all()), 'Nonfinite bounded learned output')
                # Different runtime/precision from the BF16 source computation;
                # no candidate FP32 source-order reduction is used here.
                reference = x.float().numpy().astype(np.float64) @ weight.float().numpy().astype(np.float64).T
                need(bool(np.isfinite(reference).all()), 'Nonfinite FP64 mathematical reference')
                cases.append(dict(name=f'learned-{ordinal}-t{tokens}', ordinal=ordinal,
                    tokens=tokens, in_features=columns, out_features=4,
                    input_sha256=hashlib.sha256(x_raw).hexdigest(),
                    output=append('source.bf16le', raw(output)),
                    mathematical_reference=append('reference.f64le', reference.astype('<f8').tobytes())))
        corners = []

        def corner(label, x_words, w_words, tokens, columns, rows):
            xb, wb = word_bytes(x_words), word_bytes(w_words)
            x, w = tensor(xb).reshape(tokens, columns), tensor(wb).reshape(rows, columns)
            output = functional.linear(x, w)
            corners.append(dict(name=label, tokens=tokens, in_features=columns, out_features=rows,
                input=append('corner-input.bf16le', xb), weight=append('corner-weight.bf16le', wb),
                output=append('corner-source.bf16le', raw(output)),
                finite_output_words=int(torch.isfinite(output).sum()),
                rejection_markers=int((~torch.isfinite(output)).sum())))

        for ordinal, columns in enumerate(policy['corner_columns']):
            for tokens in policy['corner_tokens']:
                corner(f'tail-k{columns}-t{tokens}',
                    norm_inputs(pool, ordinal+719, tokens, columns),
                    norm_inputs(pool, ordinal+997, policy['corner_rows'], columns),
                    tokens, columns, policy['corner_rows'])
        for columns in [32, 33, 40, 63, 64, 65]:
            x, w = overflow_rescue(columns)
            corner(f'fma-overflow-rescue-k{columns}', x, w, 1, columns, 1)
        for delta in [0, 0x3380, 0xb380, 0x3400, 0xb400]:
            x, w = [0]*33, [0]*33
            x[0], x[1], x[32] = 0x3f80, 0x3b80, delta
            w[0] = w[1] = w[32] = 0x3f80
            corner(f'bf16-midpoint-tail-delta{delta}', x, w, 1, 33, 1)
        # Nonzero terms in separate partials, then serial/vector tails, expose
        # cancellation and reduction trees without depending on random inputs.
        for columns in [7, 8, 17, 31, 32, 33, 63, 64, 65, 128, 129]:
            x, w = [0]*columns, [0x3f80]*columns
            for i in range(columns):
                x[i] = [0x4580, 0xc580, 0x3f80, 0xbf80, 1, 0x8001, 0, 0x8000][i % 8]
            corner(f'cancellation-k{columns}', x, w, 1, columns, 1)
        finite = np.asarray([v for v in range(65536) if v & 0x7f80 != 0x7f80], dtype='<u2')
        need(len(finite) == 65280, 'Finite BF16 grid differs')
        grid = np.zeros((len(finite), 32), dtype='<u2')
        grid[np.arange(len(finite)), np.arange(len(finite)) % 32] = finite
        identity = functional.linear(tensor(grid.tobytes()).reshape(len(finite), 32),
                                     tensor(word_bytes([0x3f80]*32)).reshape(1, 32))
        need(bool(torch.isfinite(identity).all()), 'Nonfinite identity-grid result')
        append('finite-grid.bf16le', raw(identity))
        authenticate(fd)
        need(file_identity(os.fstat(fd)) == file_identity(before), 'Source model changed')
    finally:
        os.close(fd)
    verify_pins(policy['source_sha256'], 'Source metadata')
    verify_pins(runtime['files'], 'Reference runtime')
    files = {}
    for name, data in blobs.items():
        destination = args.fresh_directory/name
        with destination.open('xb') as stream:
            stream.write(data)
        files[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    result = dict(format='toklet-lfm3-projection-bf16-fixture-v1', fixture_completed=True,
        contract_sha256=CONTRACT_SHA, runtime_sha256=RUNTIME_SHA,
        producer_sha256=digest(Path(__file__)), torch_version=torch.__version__,
        torch_git_commit=torch.version.git_version, numpy_version=np.__version__,
        source_model_fully_authenticated_before_after=True, matrices=matrices, cases=cases,
        corners=corners, finite_grid=dict(words=65280, columns=32, nonzero_lane='word_index%32',
            finite_word_order='ascending_u16excludingNaNInf', output_file='finite-grid.bf16le'),
        files=files, learned_output_words=sum(c['tokens']*c['out_features'] for c in cases),
        corner_output_words=sum(c['tokens']*c['out_features'] for c in corners),
        corner_rejection_markers=sum(c['rejection_markers'] for c in corners),
        remote_model_code_executed=False, model_execution_qualified=False, quality_adoption_qualified=False,
        scope='All154rank2source matrices via four selected physical rows; exact source BF16 outputs and independent FP64 dots, not complete projection matrices or model/loader/graph execution')
    with (args.fresh_directory/'fixture.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['matrices', 'cases', 'corners', 'files']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['contract', 'runtime', 'model', 'fresh_directory']:
        parser.add_argument('--'+name.replace('_', '-'), required=True, type=Path)
    args = parser.parse_args()
    need(not args.fresh_directory.exists() and args.fresh_directory.parent.is_dir(),
         'Fresh fixture destination required')
    need(digest(args.contract) == CONTRACT_SHA, 'Frozen projection contract differs')
    need(digest(args.runtime) == RUNTIME_SHA, 'Pinned reference runtime differs')
    policy, runtime = json.loads(args.contract.read_text()), json.loads(args.runtime.read_text())
    verify_pins(policy['source_sha256'], 'Source metadata')
    verify_pins(runtime['files'], 'Reference runtime')
    export(args, policy, runtime)


if __name__ == '__main__':
    main()
