"""Prospective8192 preparation/replay only; no model execution or qualification.

Caller first authenticates and admits the actually retained4k decision. That
admission supplies real parent paths/hashes; this tool does not decide retention.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/reference'))
import qwen3_context8192_profile as profile
from qwen3_context_precision import Owner

SOURCES = 'dc2f82fb50b3f1f763c4757a992429b9122f3628c049709bdd7cb2252c69ea28'


def validate_selection_parents(original, complete, previous, previous4096):
    profile.validate_ids(original, 512)
    profile.validate_ids(previous, 10001)
    if (original != [9707, 11, 1879, 0] * 128
            or not isinstance(complete, list) or not 22289 <= len(complete) <= 1048576):
        raise ValueError('Original fixture or complete bounded corpus differs')
    profile.validate_ids(complete, len(complete))
    if set(previous4096) != {'A.u32le', 'B.u32le', 'Q.u32le'}:
        raise ValueError('All authenticated F2 inputs required')
    for values in previous4096.values():
        profile.validate_ids(values, 4096)
    old_a = original * 8
    if (complete[:10001] != previous or complete[10001:14097] != previous4096['Q.u32le']
            or previous4096['A.u32le'] != old_a
            or previous4096['B.u32le'] != old_a[:-1] + [(old_a[-1] + 1) % profile.VOCAB]):
        raise ValueError('Old10001 and F2 spans/structural controls must be preserved')


def choose(original, complete, previous, previous4096):
    validate_selection_parents(original, complete, previous, previous4096)
    a = original * 16
    return {'A.u32le': a, 'B.u32le': a[:-1] + [(a[-1] + 1) % profile.VOCAB],
            'Q.u32le': complete[14097:22289]}


def replay_selection(original, complete, previous, previous4096):
    """Separate index-based reconstruction, preserving both old corpus regions."""
    # Validation is shared; construction below does not use choose().
    validate_selection_parents(original, complete, previous, previous4096)
    a, b, q = [], [], []
    for position in range(8192):
        token = original[position % 512]
        a.append(token)
        b.append((token + 1) % 151936 if position == 8191 else token)
        q.append(complete[14097 + position])
    return {'A.u32le': a, 'B.u32le': b, 'Q.u32le': q}


def validate_admission(value):
    if (not isinstance(value, dict)
            or set(value) != {'format', 'retention_approved', 'retained_schedule', 'parent_receipts'}
            or value['format'] != 'toklet-qwen3-context8192-parent-admission-v1'
            or value['retention_approved'] is not True
            or value['retained_schedule'] != 'ParallelOutputsExperimental'
            or set(value['parent_receipts']) != {'f2_result', 'retained_parallel_result'}):
        raise ValueError('Explicit caller-admitted retained4k decision required')
    for binding in value['parent_receipts'].values():
        if set(binding) != {'path', 'bytes', 'sha256'} or not Path(binding['path']).is_absolute():
            raise ValueError('Absolute parent receipt binding required')
        profile.identity({key: binding[key] for key in ('bytes', 'sha256')}, 16 * 1024**2)
    if value['parent_receipts']['f2_result']['sha256'] != profile.F2_RESULT:
        raise ValueError('Accepted F2 receipt differs')


def derive(replay, report, admission_path, admission_sha256):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from importlib.metadata import version
    if version('pyarrow') != '20.0.0' or version('tokenizers') != '0.22.2':
        raise ValueError('Pinned preparation runtime required')
    if type(replay) is not bool or not profile.digest(admission_sha256):
        raise ValueError('Explicit mode and authenticated admission required')
    with ExitStack() as stack:
        def read(path, expected, maximum):
            path = Path(path)
            if isinstance(expected, str):
                expected = dict(bytes=path.stat().st_size, sha256=expected)
            profile.identity(expected, maximum)
            owner = Owner(path, expected)
            report['parents'].append(owner.audit)
            stack.enter_context(owner)
            owner.file.seek(0)
            raw = owner.file.read(expected['bytes'])
            if len(raw) != expected['bytes'] or profile.sha(raw) != expected['sha256']:
                raise ValueError('Actually consumed parent differs')
            return raw

        admission = json.loads(read(admission_path, admission_sha256, 65536))
        validate_admission(admission)
        parents = {}
        for name, binding in admission['parent_receipts'].items():
            pin = {key: binding[key] for key in ('bytes', 'sha256')}
            read(binding['path'], pin, 16 * 1024**2)
            parents[name] = pin
        report.update(admission_sha256=admission_sha256, parent_receipts=parents)
        sources = json.loads(read(ROOT / 'benchmarks/sources.json', SOURCES, 1024**2))
        numerical = json.loads(read(ROOT / 'tests/fixtures/qwen3-context512-v1/manifest.json', profile.NUMERICAL, 1024**2))
        quality = json.loads(read(ROOT / 'benchmarks/qwen3-quality-v1/manifest.json', profile.QUALITY, 1024**2))
        f2 = json.loads(read(ROOT / 'benchmarks/qwen3-context4096-v1/manifest.json', profile.F2_INPUT, 1024**2))
        assets = [v for v in sources['assets'] if v['id'] == 'wikitext.test']
        if len(assets) != 1:
            raise ValueError('Unique pinned WikiText source required')
        asset = assets[0]
        raw = read(ROOT / asset['path'], {key: asset[key] for key in ('bytes', 'sha256')}, 4 * 1024**2)
        table = pq.ParquetFile(pa.BufferReader(raw))
        if replay:
            rows = []
            for batch in table.iter_batches(batch_size=257, columns=['text'], use_threads=False):
                rows.extend(batch.column(0).to_pylist())
        else:
            rows = table.read(columns=['text'], use_threads=False).column(0).to_pylist()
        if not all(type(row) is str for row in rows):
            raise ValueError('Original string corpus rows required')
        corpus = '\n'.join(row for row in rows if row)
        if len(corpus.encode()) > 16 * 1024**2:
            raise ValueError('Corpus exceeds frozen bound')
        tokenpin = quality['tokenizer_assets']['tokenizer.json']
        tokenfile = ROOT / 'models/MLX-Qwen3-0.6B-8bit-11de968/tokenizer.json'
        tokenbytes = read(tokenfile, tokenpin, 16 * 1024**2)
        # Both parse consumed authenticated bytes. Separate batch corpus traversal
        # and constructor/index recipe form replay; outer Rust replay is separate.
        tokenizer = (Tokenizer.from_str(tokenbytes.decode()) if replay
                     else Tokenizer.from_buffer(tokenbytes))
        if tokenizer.truncation is not None or tokenizer.padding is not None:
            raise ValueError('Clipping/padding forbidden')
        complete = tokenizer.encode(corpus, add_special_tokens=False).ids
        previous_raw = read(ROOT / 'benchmarks/qwen3-quality-v1/held_out_loss.u32le',
                            quality['files']['held_out_loss.u32le'], 40004)
        previous = list(struct.unpack('<10001I', previous_raw))
        old = {}
        for name in ('A.u32le', 'B.u32le', 'Q.u32le'):
            old[name] = list(struct.unpack('<4096I', read(ROOT / 'benchmarks/qwen3-context4096-v1' / name,
                                                        f2['files'][name], 16384)))
        selected = (replay_selection if replay else choose)(numerical['plan']['cases']['original'],
                                                            complete, previous, old)
        report.update(dataset=asset, tokenizer=tokenpin, corpus_utf8_bytes=len(corpus.encode()),
                      corpus_sha256=profile.sha(corpus.encode()), complete_corpus_tokens=len(complete),
                      old10001_preserved=True, old4096_preserved=True,
                      new_span_global_half_open=[14097, 22289], add_special_tokens=False,
                      join_policy='newline_join_nonempty_original_rows')
    return {name: struct.pack('<8192I', *values) for name, values in selected.items()}


def run(output, admission_path, admission_sha256, replay=False, manifest_sha256=None):
    """Explicit preparation or authenticated replay; caller persists returned receipt."""
    output = Path(output)
    report = dict(passed=False, model_executed=False, parents=[], error=None, replay=replay)
    try:
        if type(replay) is not bool or (replay and not profile.digest(manifest_sha256)):
            raise ValueError('Replay requires frozen manifest identity')
        if not replay and (output.exists() or manifest_sha256 is not None):
            raise ValueError('Fresh preparation destination required')
        arrays = derive(replay, report, admission_path, admission_sha256)
        files = {name: dict(bytes=len(raw), sha256=profile.sha(raw)) for name, raw in arrays.items()}
        prefix = profile.sha(arrays['A.u32le'][:16384])
        if prefix != profile.F2_A:
            raise ValueError('Historical A4096 prefix identity differs')
        if replay:
            with ExitStack() as stack:
                def authenticated(path, pin):
                    owner = Owner(path, pin)
                    report['parents'].append(owner.audit)
                    stack.enter_context(owner)
                    owner.file.seek(0)
                    return owner.file.read(pin['bytes'])
                size = (output / 'manifest.json').stat().st_size
                profile.identity(dict(bytes=size, sha256=manifest_sha256), 1024**2)
                manifest_raw = authenticated(output / 'manifest.json', dict(bytes=size, sha256=manifest_sha256))
                manifest = json.loads(manifest_raw)
                profile.validate(manifest)
                if (manifest['files'] != files or manifest['parent_receipts'] != report['parent_receipts']
                        or manifest['admission_sha256'] != admission_sha256):
                    raise ValueError('Independent input reconstruction/parent differs')
                for name, raw in arrays.items():
                    if authenticated(output / name, files[name]) != raw:
                        raise ValueError('Saved input differs: ' + name)
        else:
            output.mkdir()
            manifest = dict(profile.REQUIRED, complete=False, files={}, primary_error=None,
                admission_sha256=admission_sha256, parent_receipts=report['parent_receipts'],
                a4096_prefix_sha256=prefix,
                provenance={key: value for key, value in report.items()
                            if key not in ('passed', 'parents', 'error', 'replay')},
                native_recipe=dict(B='A with last token incremented by1 modulo151936',
                                   source='sixteen original512 repeats; structural witness, not quality'),
                numerical_or_quality_executed=False, ordinary_app_capacity_changed=False)
            primary = None
            try:
                for name, raw in arrays.items():
                    with (output / name).open('xb') as stream:
                        written = stream.write(raw)
                        manifest['files'][name] = dict(bytes=written, sha256=profile.sha(raw[:written]))
                        if written != len(raw):
                            raise OSError('Incomplete token write: ' + name)
                manifest['complete'] = True
                profile.validate(manifest)
            except BaseException as error:
                primary = error
                manifest['complete'] = False
                manifest['primary_error'] = type(error).__name__ + ': ' + str(error)
                raise
            finally:
                try:
                    manifest_raw = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
                    with (output / 'manifest.json').open('xb') as stream:
                        if stream.write(manifest_raw) != len(manifest_raw):
                            raise OSError('Incomplete manifest write')
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note('Partial manifest: ' + repr(error))
        report.update(files=files, manifest_sha256=profile.sha(manifest_raw), passed=True)
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    return report


if __name__ == '__main__':
    if len(sys.argv) not in (5, 6) or sys.argv[1] not in ('prepare', 'replay'):
        raise SystemExit('prepare OUTPUT ADMISSION SHA | replay OUTPUT ADMISSION SHA MANIFEST_SHA')
    replay_mode = sys.argv[1] == 'replay'
    if len(sys.argv) != (6 if replay_mode else 5):
        raise SystemExit('Explicit mode argument count differs')
    result = run(sys.argv[2], sys.argv[3], sys.argv[4], replay_mode,
                 sys.argv[5] if replay_mode else None)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['passed'] else 1)
