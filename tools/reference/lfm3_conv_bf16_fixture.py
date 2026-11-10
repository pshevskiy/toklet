"""Canonical LFM3 parent BF16 binary and seven-tap convolution fixtures.

Pinned Torch CPU operations run over authenticated coefficient bytes. No remote
model code, projections, state mutation or complete model execution is involved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

from lfm3_bf16_fixture import (
    HEADER_SHA, MODEL_BYTES, MODEL_SHA, RUNTIME_SHA, authenticate, digest,
    file_identity, input_pool, need, norm_inputs, verify_runtime,
)

CONTRACT_SHA = '7c73e15f3974ea5aa78db212142441fd6a4aaa84d7875d275602e012dfb3bc94'


def conv_schema(config):
    need(config['hidden_size'] == 2048 and config['conv_L_cache'] == 7
         and config['conv_bias'] is False and len(config['layer_types']) == 24,
         'Pinned convolution geometry differs')
    names = sorted(f'model.layers.{layer}.conv.conv.weight'
                   for layer, kind in enumerate(config['layer_types']) if kind == 'conv')
    need(len(names) == 13, 'Expected all13 convolution banks')
    return names


def word_bytes(words):
    return struct.pack('<' + str(len(words)) + 'H', *words)


def head_gates(rows):
    pool = [0, 1, 128, 0x3c00, 0x3f00, 0x3f7f, 0x3f80]
    return [pool[(index*5 + index//16*3) % len(pool)] for index in range(rows*16)]


def export(args, policy, runtime):
    import numpy as np
    import torch
    import torch.nn.functional as functional
    need(torch.__version__ == '2.9.1' and np.__version__ == '2.3.2', 'Pinned CPU versions differ')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads((args.metadata/'child-config.json').read_text())
    names = conv_schema(config)

    def tensor(raw):
        return torch.from_numpy(np.frombuffer(raw, dtype='<u2').copy()).view(torch.bfloat16)

    def raw(value):
        return value.contiguous().view(torch.int16).numpy().astype('<u2').tobytes()

    def save(label, value):
        blob = raw(value)
        (args.fresh_directory/(label+'.bf16le')).write_bytes(blob)
        return hashlib.sha256(blob).hexdigest()

    fd = os.open(args.model, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_size == MODEL_BYTES, 'Canonical regular file required')
        authenticate(fd)
        need(os.pread(fd, 8, 0) == (32368).to_bytes(8, 'little'), 'Canonical header extent differs')
        header = os.pread(fd, 32368, 8)
        need(hashlib.sha256(header).hexdigest() == HEADER_SHA, 'Canonical header differs')
        schema = json.loads(header)
        pool = input_pool()
        weights = bytearray()
        entries, cases = [], []
        words = candidate_differences = diagnostic_failures = 0
        diagnostic_maximum = 0.0
        for ordinal, name in enumerate(names):
            item = schema[name]
            begin, end = item['data_offsets']
            need(item['dtype'] == 'BF16' and item['shape'] == [2048, 7, 1]
                 and end-begin == 2048*7*2 and 0 <= begin < end <= MODEL_BYTES-32376,
                 'Canonical convolution layout differs')
            bank_raw = os.pread(fd, end-begin, 32376+begin)
            need(len(bank_raw) == end-begin, 'Short coefficient read')
            bank = tensor(bank_raw).reshape(2048, 1, 7)
            need(bool(torch.isfinite(bank).all()), 'Nonfinite source coefficients')
            entries.append(dict(name=name, ordinal=ordinal, byte_offset=len(weights),
                byte_count=len(bank_raw), model_data_offset=begin,
                sha256=hashlib.sha256(bank_raw).hexdigest(),
                minimum=float(bank.float().min()), maximum=float(bank.float().max())))
            weights.extend(bank_raw)
            coeff = bank.float().numpy().reshape(2048, 7)
            for tokens in policy['convolution']['tokens']:
                x_raw = word_bytes(norm_inputs(pool, ordinal, tokens, 2048))
                x = tensor(x_raw).reshape(tokens, 2048)
                for profile in policy['convolution']['profiles']:
                    h_raw = (bytes(6*2048*2) if profile == 'zero-history'
                             else word_bytes(norm_inputs(pool, ordinal+31, 6, 2048)))
                    history = tensor(h_raw).reshape(6, 2048)
                    joined = torch.cat((history, x), dim=0)
                    # This is the actual parent CPU fallback, including BF16
                    # input/output and the source's padding/crop convention.
                    output = functional.conv1d(joined.T.unsqueeze(0), bank,
                        padding=6, groups=2048)[0, :, 6:6+tokens].T.contiguous()
                    need(bool(torch.isfinite(output).all()), 'Nonfinite canonical convolution')
                    values = joined.float().numpy()
                    partial = [np.zeros((tokens, 2048), dtype=np.float32) for _ in range(4)]
                    reference = np.zeros((tokens, 2048), dtype=np.float64)
                    for tap in range(7):
                        source = values[tap:tap+tokens]
                        term = np.multiply(source, coeff[:, tap], dtype=np.float32)
                        lane = tap if tap < 4 else 0
                        partial[lane] = np.add(partial[lane], term, dtype=np.float32)
                        reference += source.astype(np.float64)*coeff[:, tap].astype(np.float64)
                    unrounded = partial[0]
                    for lane in range(1, 4):
                        unrounded = np.add(unrounded, partial[lane], dtype=np.float32)
                    candidate = torch.from_numpy(unrounded).to(torch.bfloat16)
                    differing = int((candidate.view(torch.int16) != output.view(torch.int16)).sum())
                    errors = np.abs(unrounded.astype(np.float64)-reference)
                    failures = int(np.sum(errors > 1e-5 + 1e-4*np.abs(reference)))
                    need(bool(np.isfinite(unrounded).all()), 'Nonfinite candidate diagnostic')
                    label = f'conv-{ordinal}-t{tokens}-{profile}'
                    cases.append(dict(name=label, ordinal=ordinal, coefficient=name,
                        tokens=tokens, channels=2048, history_profile=profile,
                        input_sha256=hashlib.sha256(x_raw).hexdigest(),
                        history_sha256=hashlib.sha256(h_raw).hexdigest(),
                        output_sha256=save(label, output),
                        candidate_sha256=hashlib.sha256(raw(candidate)).hexdigest(),
                        candidate_word_differences=differing,
                        fp32_diagnostic_failures=failures,
                        fp32_diagnostic_maximum_absolute_error=float(errors.max())))
                    words += tokens*2048
                    candidate_differences += differing
                    diagnostic_failures += failures
                    diagnostic_maximum = max(diagnostic_maximum, float(errors.max()))
        weight_file = args.fresh_directory/'weights.bf16le'
        weight_file.write_bytes(weights)
        finite = [word for word in range(65536) if word & 0x7f80 != 0x7f80]
        need(len(finite) == 65280, 'Finite BF16 grid differs')
        binary_cases = []
        left_raw = word_bytes(finite)
        left = tensor(left_raw)
        for name, operation in [('add', torch.add), ('mul', torch.mul)]:
            for ordinal, anchor in enumerate(policy['binary']['anchors']):
                right = tensor(word_bytes([anchor]))
                output = operation(left, right)
                label = f'binary-{name}-{ordinal}'
                binary_cases.append(dict(name=label, operation=name, anchor=anchor,
                    values=65280, output_sha256=save(label, output),
                    nonfinite_output_words=int((~torch.isfinite(output)).sum())))
        gate_cases = []
        for tokens in policy['head_gate']['tokens']:
            x_raw = word_bytes(norm_inputs(pool, 73, tokens*16, 128))
            g_raw = word_bytes(head_gates(tokens))
            x = tensor(x_raw).reshape(tokens, 16, 128)
            gate = tensor(g_raw).reshape(tokens, 16, 1)
            output = x*gate
            label = f'head-gate-t{tokens}'
            gate_cases.append(dict(name=label, tokens=tokens,
                input_sha256=hashlib.sha256(x_raw).hexdigest(),
                gate_sha256=hashlib.sha256(g_raw).hexdigest(),
                output_sha256=save(label, output)))
        authenticate(fd)
        need(file_identity(before) == file_identity(os.fstat(fd)) == file_identity(args.model.lstat()),
             'Canonical source changed during capture')
    finally:
        os.close(fd)
    verify_runtime(runtime)
    return dict(format='toklet-lfm3-bf16-binary-conv-fixture-v1', fixture_completed=True,
        contract_sha256=CONTRACT_SHA, runtime_sha256=RUNTIME_SHA,
        producer_sha256=digest(Path(__file__)),
        common_producer_sha256=digest(Path(__file__).with_name('lfm3_bf16_fixture.py')),
        model_sha256=MODEL_SHA, model_bytes=MODEL_BYTES, header_sha256=HEADER_SHA,
        torch_version=torch.__version__, numpy_version=np.__version__, device='cpu', threads=1,
        convolution=dict(weights_sha256=digest(weight_file), weights_bytes=len(weights),
            banks=entries, cases=cases, output_words=words,
            candidate_word_differences=candidate_differences,
            candidate_exact_gate_passed=candidate_differences == 0,
            fp32_mathematical_diagnostic=dict(abs=1e-5, rel=1e-4,
                failures=diagnostic_failures, maximum_absolute_error=diagnostic_maximum)),
        binary=dict(finite_left_words=65280, left_sha256=hashlib.sha256(left_raw).hexdigest(),
            cases=binary_cases), head_gate_cases=gate_cases,
        child_runtime_correspondence_qualified=False, model_execution_qualified=False,
        model_memory_fit_qualified=False, quality_adoption_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['contract', 'runtime', 'metadata', 'model', 'fresh_directory', 'fresh_fixture']:
        parser.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    need(not args.fresh_directory.exists() and not args.fresh_fixture.exists()
         and args.fresh_directory.parent.is_dir() and args.fresh_fixture.parent.is_dir(),
         'Fresh fixture destinations required')
    need(digest(args.contract) == CONTRACT_SHA, 'Frozen convolution contract differs')
    need(digest(args.runtime) == RUNTIME_SHA, 'Admitted runtime manifest differs')
    policy = json.loads(args.contract.read_text())
    for name, pin in policy['source_sha256'].items():
        need(digest(args.metadata/name) == pin, 'Pinned source metadata differs')
    runtime = json.loads(args.runtime.read_text())
    verify_runtime(runtime)
    args.fresh_directory.mkdir()
    result = export(args, policy, runtime)
    with args.fresh_fixture.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(fixture_completed=True,
        banks=len(result['convolution']['banks']), cases=len(result['convolution']['cases']),
        words=result['convolution']['output_words'],
        candidate_differences=result['convolution']['candidate_word_differences'],
        diagnostic=result['convolution']['fp32_mathematical_diagnostic'],
        binary_cases=len(result['binary']['cases']), head_gate_cases=len(result['head_gate_cases']),
        fixture_sha256=digest(args.fresh_fixture))))


if __name__ == '__main__':
    main()
