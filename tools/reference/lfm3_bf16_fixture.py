"""LFM3 canonical parent BF16 norm/gate fixtures; no remote model execution.

Source formulas are written independently over pinned Torch CPU operations.
Authenticated child norm bytes equal the parent bytes; weight equality alone
does not grant child-runtime numerical correspondence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

CONTRACT_SHA = '1024f1494d60e7db8abe3e47148fc73396515df801b861156740552e44f7bf37'
RUNTIME_SHA = '038b083eb320ce9e24ab93750775e2d441ce139b244a1a58ea34e378853e3e45'
MODEL_SHA = 'b6ff17ac993dddf0f6f51bd872b12ae1437f4453a54a0b4878cedc7864fbc07e'
HEADER_SHA = '491ddbe9714ec2907c4d444294301198ae6eed8e1152af3b90237332b860b30b'
MODEL_BYTES = 3457635448


def need(value, reason):
    if not value:
        raise ValueError(reason)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_runtime(runtime):
    for name, pin in runtime['files'].items():
        need(digest(Path(name)) == pin, 'Pinned runtime changed: ' + name)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def bf16(value):
    word = struct.unpack('<I', struct.pack('<f', value))[0]
    need(word & 0x7f800000 != 0x7f800000, 'Finite BF16 input required')
    return ((word + 0x7fff + ((word >> 16) & 1)) >> 16) & 0xffff


def input_pool():
    # Every finite BF16 word within [-16,16], including both zeros and all
    # subnormals. Typical-magnitude lanes below make reduction rounding visible
    # instead of relying on a coarse grid whose square sums happen to be exact.
    return [magnitude | sign for sign in [0, 0x8000] for magnitude in range(0x4181)]


def norm_inputs(pool, ordinal, rows, width):
    result = []
    for index in range(rows * width):
        selector = index * 37 + index // width * 11 + ordinal * 3
        if index % 2:
            # BF16 magnitudes 0.0625..16, alternating signs via the same selector.
            value = (0x3d80 + selector % 1025) | ((selector // 1025 % 2) << 15)
        else:
            value = pool[selector % len(pool)]
        result.append(value)
    return result


def norm_schema(config):
    expected = {'model.embedding_norm.weight': (2048, 1)}
    need(config['hidden_size'] == 2048 and config['head_dim'] == 128
         and config['num_attention_heads'] == 16 and config['num_key_value_heads'] == 4
         and len(config['layer_types']) == config['num_hidden_layers'] == 24
         and config['norm_eps'] == 1e-5 and config.get('use_sandwich_norm', True) is True,
         'Pinned LFM3 norm geometry differs')
    for layer, kind in enumerate(config['layer_types']):
        for suffix in ['operator_norm', 'ffn_norm', 'post_operator_norm', 'post_ffn_norm']:
            expected[f'model.layers.{layer}.{suffix}.weight'] = (2048, 1)
        if kind in ['full_attention', 'sliding_attention']:
            for suffix, heads in [('q_norm', 16), ('k_norm', 4)]:
                expected[f'model.layers.{layer}.self_attn.{suffix}.weight'] = (128, heads)
        else:
            need(kind == 'conv', 'Unknown source layer')
    need(len(expected) == 119, 'Expected all119 learned norm vectors')
    return expected


def file_identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def authenticate(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(fd), 'rb') as stream:
        need(hashlib.file_digest(stream, 'sha256').hexdigest() == MODEL_SHA,
             'Full canonical child object differs')


def export(args, policy, runtime):
    import numpy as np
    import torch
    need(torch.__version__ == '2.9.1' and np.__version__ == '2.3.2', 'Admitted CPU versions differ')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads((args.metadata / 'child-config.json').read_text())
    expected = norm_schema(config)
    fd = os.open(args.model, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_size == MODEL_BYTES, 'Canonical regular file required')
        authenticate(fd)
        prefix = os.pread(fd, 8, 0)
        need(len(prefix) == 8 and int.from_bytes(prefix, 'little') == 32368, 'Pinned header extent differs')
        header = os.pread(fd, 32368, 8)
        need(hashlib.sha256(header).hexdigest() == HEADER_SHA, 'Canonical header differs')
        schema = json.loads(header)
        pool = input_pool()
        entries, cases = [], []
        weights = bytearray()
        total = diagnostic_failures = single_round_differences = 0
        diagnostic_maximum = 0.0

        def tensor(raw):
            return torch.from_numpy(np.frombuffer(raw, dtype='<u2').copy()).view(torch.bfloat16)

        def raw(value):
            return value.contiguous().view(torch.int16).numpy().astype('<u2').tobytes()

        def words32(value):
            return value.contiguous().view(torch.int32).reshape(-1).numpy().astype('<u4').tolist()

        for ordinal, name in enumerate(sorted(expected)):
            width, heads = expected[name]
            meta = schema[name]
            begin, end = meta['data_offsets']
            need(meta['dtype'] == 'BF16' and meta['shape'] == [width]
                 and end-begin == width*2 and 0 <= begin < end <= MODEL_BYTES-32376,
                 'Canonical learned gain layout differs')
            gain_raw = os.pread(fd, width*2, 32376+begin)
            need(len(gain_raw) == width*2, 'Short canonical gain read')
            gain = tensor(gain_raw)
            need(bool(torch.isfinite(gain).all()), 'Nonfinite canonical gain')
            entry = dict(name=name, ordinal=ordinal, width=width, heads=heads,
                byte_offset=len(weights), byte_count=len(gain_raw), model_data_offset=begin,
                sha256=hashlib.sha256(gain_raw).hexdigest(),
                minimum=float(gain.float().min()), maximum=float(gain.float().max()))
            entries.append(entry)
            weights.extend(gain_raw)
            for tokens in policy['norm_fixture_tokens']:
                rows = tokens*heads
                source = norm_inputs(pool, ordinal, rows, width)
                source_raw = struct.pack('<' + str(len(source)) + 'H', *source)
                x = tensor(source_raw).reshape(rows, width)
                xf = x.float()
                variance = xf.pow(2).mean(-1, keepdim=True)
                inverse = torch.rsqrt(variance + 1e-5)
                normalized = xf * inverse
                middle = normalized.to(torch.bfloat16)
                output = middle * gain
                single = (normalized * gain.float()).to(torch.bfloat16)
                need(bool(torch.isfinite(variance).all() and torch.isfinite(output).all()),
                     'Nonfinite canonical norm fixture')
                observed = raw(output)
                local_single = int((output.view(torch.int16) != single.view(torch.int16)).sum())
                single_round_differences += local_single
                # Independent FP64 mathematical diagnostic, before both BF16
                # boundaries; source epsilon is its actual FP32 value.
                independent = xf.numpy().astype(np.float64)
                reference = independent / np.sqrt(np.mean(independent*independent, axis=-1, keepdims=True)
                                                  + np.float64(np.float32(1e-5)))
                errors = np.abs(normalized.numpy().astype(np.float64)-reference)
                local_bad = int(np.sum(errors > 1e-5 + 1e-4*np.abs(reference)))
                diagnostic_failures += local_bad
                diagnostic_maximum = max(diagnostic_maximum, float(errors.max()))
                label = f'norm-{ordinal}-t{tokens}'
                (args.fresh_directory / (label + '.bf16le')).write_bytes(observed)
                cases.append(dict(name=label, gain=name, ordinal=ordinal, width=width, heads=heads,
                    tokens=tokens, rows=rows, input_sha256=hashlib.sha256(source_raw).hexdigest(),
                    variance_fp32_bits=words32(variance), inverse_fp32_bits=words32(inverse),
                    normalized_bf16_sha256=hashlib.sha256(raw(middle)).hexdigest(),
                    output_sha256=hashlib.sha256(observed).hexdigest(),
                    diagnostic_failures=local_bad, single_round_word_differences=local_single))
                total += len(source)
        norm_file = args.fresh_directory / 'norms.bf16le'
        norm_file.write_bytes(weights)
        finite_words = [word for word in range(65536) if word & 0x7f80 != 0x7f80]
        need(len(finite_words) == 65280, 'Finite BF16 enumeration differs')
        unary_input = struct.pack('<65280H', *finite_words)
        x = tensor(unary_input)
        pointwise = {}
        for name, output, explicit in [
            ('sigmoid', torch.sigmoid(x), torch.sigmoid(x.float()).to(torch.bfloat16)),
            ('silu', torch.nn.functional.silu(x), (x.float()*torch.sigmoid(x.float())).to(torch.bfloat16)),
        ]:
            need(bool(torch.isfinite(output).all()), 'Nonfinite canonical gate output')
            blob = raw(output)
            (args.fresh_directory / (name + '.bf16le')).write_bytes(blob)
            pointwise[name] = dict(output_sha256=hashlib.sha256(blob).hexdigest(), values=len(finite_words),
                explicit_fp32_formula_word_differences=int((output.view(torch.int16) != explicit.view(torch.int16)).sum()))
        need(diagnostic_failures == 0, 'Original FP32 mathematical diagnostic rejects')
        authenticate(fd)
        need(file_identity(before) == file_identity(os.fstat(fd)) == file_identity(args.model.lstat()),
             'Canonical source changed during capture')
    finally:
        os.close(fd)
    verify_runtime(runtime)
    return dict(format='toklet-lfm3-bf16-parent-fixture-v1', fixture_completed=True,
        contract_sha256=CONTRACT_SHA, runtime_sha256=RUNTIME_SHA, producer_sha256=digest(Path(__file__)),
        model_sha256=MODEL_SHA, model_bytes=MODEL_BYTES, header_sha256=HEADER_SHA,
        torch_version=torch.__version__, numpy_version=np.__version__, device='cpu', threads=1,
        norms=dict(weights_sha256=digest(norm_file), weights_bytes=len(weights), vectors=entries,
            input_pool=pool, input_rule='selector=index*37+row*11+ordinal*3; even:index->pool[selector%pool.len]; odd:(0x3d80+selector%1025)|((selector//1025%2)<<15)',
            cases=cases, output_words=total,
            single_round_vs_canonical_word_differences=single_round_differences,
            fp32_mathematical_diagnostic=dict(abs=1e-5, rel=1e-4, failures=diagnostic_failures,
                maximum_absolute_error=diagnostic_maximum)),
        pointwise=dict(input_sha256=hashlib.sha256(unary_input).hexdigest(),
            input_rule='all u16 ascending except exponent255; includes both zero signs/subnormals',
            finite_values=65280, functions=pointwise),
        child_runtime_correspondence_qualified=False, model_execution_qualified=False,
        model_memory_fit_qualified=False, quality_adoption_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['contract', 'runtime', 'metadata', 'model', 'fresh_directory', 'fresh_fixture']:
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    need(not args.fresh_directory.exists() and not args.fresh_fixture.exists()
         and args.fresh_directory.parent.is_dir() and args.fresh_fixture.parent.is_dir(),
         'Fresh fixture destinations required')
    need(digest(args.contract) == CONTRACT_SHA, 'Frozen BF16 source contract differs')
    need(digest(args.runtime) == RUNTIME_SHA, 'Admitted BF16 runtime manifest differs')
    policy = json.loads(args.contract.read_text())
    for name, pin in policy['source_sha256'].items():
        need(digest(args.metadata / name) == pin, 'Pinned BF16 metadata differs')
    runtime = json.loads(args.runtime.read_text())
    verify_runtime(runtime)
    args.fresh_directory.mkdir()
    result = export(args, policy, runtime)
    with args.fresh_fixture.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(fixture_completed=True, norm_vectors=len(result['norms']['vectors']),
        norm_cases=len(result['norms']['cases']), norm_output_words=result['norms']['output_words'],
        pointwise_values_per_function=65280,
        norm_single_round_word_differences=result['norms']['single_round_vs_canonical_word_differences'],
        pointwise=result['pointwise']['functions'], fixture_sha256=digest(args.fresh_fixture))))


if __name__ == '__main__':
    main()
