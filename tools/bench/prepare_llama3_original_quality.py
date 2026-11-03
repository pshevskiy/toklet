"""Freeze original-Llama 10000-target inputs and long-probe prefix; no inference."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import prepare_llama3_loss1024 as parent

ROOT = parent.ROOT
PREFIXES = [512, 513, 1024, 1025]
BLOCKS = [dict(block=i, input_start=start, input_end=min(start+1025, 10001),
               score_rows=[0, min(1024, 10000-start)],
               global_target_indices=[start+1, min(start+1025, 10001)], reset_position=0)
          for i, start in enumerate(range(0, 10000, 1024))]
POLICY = dict(targets=10000, maximum_input_positions=1025, add_special_tokens=False,
              chat_rendering=False, stop_on_eos=False, reset_each_block=True,
              mean_nll_increase_limit=0.05, consistency_prefixes=PREFIXES,
              full_row_atol=1e-3, full_row_rtol=1e-4, target_nll_absolute=1e-4)


def select(ids):
    if len(ids) < 10001 or any(type(i) is not int or not 0 <= i < 128256 for i in ids[:10001]):
        raise ValueError('Quality corpus IDs differ')
    return struct.pack('<10001I', *ids[:10001])


def derive(replay=False):
    for name, pin in parent.PARENTS.items():
        parent.read_pinned(ROOT/name, pin)
    for name, version in [('pyarrow', '20.0.0'), ('tokenizers', '0.22.2')]:
        if importlib.metadata.version(name) != version:
            raise ValueError('Preparation runtime differs')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    data = parent.read_pinned(ROOT/parent.DATASET['path'], parent.DATASET)
    tokenraw = parent.read_pinned(ROOT/parent.TOKENIZER['path'], parent.TOKENIZER)
    parquet = pq.ParquetFile(pa.BufferReader(data))
    if replay:
        rows = [row for batch in parquet.iter_batches(batch_size=257, columns=['text'], use_threads=False)
                for row in batch.column(0).to_pylist()]
        tokenizer = Tokenizer.from_str(tokenraw.decode('utf-8'))
    else:
        rows = parquet.read(columns=['text'], use_threads=False).column(0).to_pylist()
        tokenizer = Tokenizer.from_buffer(tokenraw)
    if any(type(row) is not str for row in rows):
        raise ValueError('String corpus rows required')
    corpus = '\n'.join(row for row in rows if row)
    encoded = corpus.encode('utf-8')
    if len(encoded) != parent.CORPUS['utf8_bytes'] or parent.digest(encoded) != parent.CORPUS['sha256']:
        raise ValueError('Raw corpus identity differs')
    if tokenizer.padding is not None or tokenizer.truncation is not None:
        raise ValueError('Padding/truncation forbidden')
    ids = tokenizer.encode(corpus, add_special_tokens=False).ids
    raw = select(ids)
    old = ROOT/'benchmarks/llama3-loss1024-v1/tokens.u32le'
    pin = dict(bytes=4100, sha256='e4086f7621d9d781f5487db05f17df63e9693d0d2788d43a69c8108fc28aa493')
    if raw[:4100] != parent.read_pinned(old, pin):
        raise ValueError('Existing frozen prefix differs')
    for name, pin in parent.PARENTS.items():
        parent.read_pinned(ROOT/name, pin)
    parent.read_pinned(ROOT/parent.DATASET['path'], parent.DATASET)
    parent.read_pinned(ROOT/parent.TOKENIZER['path'], parent.TOKENIZER)
    return raw, len(ids)


def validate(value):
    if (value['format'] != 'toklet-llama3-original-quality-inputs-v1'
            or value['dataset'] != parent.DATASET or value['tokenizer'] != parent.TOKENIZER
            or value['corpus'] != parent.CORPUS or value['parents'] != parent.PARENTS
            or value['blocks'] != BLOCKS or value['policy'] != POLICY
            or value['model_executed'] is not False or value['complete_corpus_tokens'] != 288936):
        raise ValueError('Frozen quality policy differs')
    covered = []
    for block in value['blocks']:
        if not 2 <= block['input_end']-block['input_start'] <= 1025:
            raise ValueError('Block exceeds execution scope')
        covered.extend(range(*block['global_target_indices']))
    if covered != list(range(1, 10001)):
        raise ValueError('Targets not covered exactly once')
    for key, count in [('tokens', 10001), ('long_probe_prefix', 1026)]:
        pin = value[key]
        if (set(pin) != {'file', 'dtype', 'count', 'bytes', 'sha256'} or pin['file'] != 'tokens.u32le'
                or pin['dtype'] != 'u32le' or pin['count'] != count or pin['bytes'] != count*4
                or len(pin['sha256']) != 64 or any(c not in '0123456789abcdef' for c in pin['sha256'])):
            raise ValueError('Frozen token descriptor differs')


def load(folder, expected_sha):
    folder = Path(folder)
    raw = (folder/'manifest.json').read_bytes()
    if len(raw) > 65536 or parent.digest(raw) != expected_sha:
        raise ValueError('External manifest identity differs')
    value = json.loads(raw, object_pairs_hook=parent.unique)
    validate(value)
    raw = parent.read_pinned(folder/'tokens.u32le', value['tokens'])
    ids = list(struct.unpack('<10001I', raw))
    if (select(ids) != raw or parent.digest(raw[:4104]) != value['long_probe_prefix']['sha256']
            or parent.digest(raw[:4100]) != 'e4086f7621d9d781f5487db05f17df63e9693d0d2788d43a69c8108fc28aa493'):
        raise ValueError('Long prefix identity differs')
    return value, ids


def run(folder, replay=False, expected_sha=None):
    folder = Path(folder)
    if not replay and os.path.lexists(folder):
        raise FileExistsError('Fresh output required')
    raw, count = derive(replay)
    if replay:
        value, ids = load(folder, expected_sha)
        if select(ids) != raw or value['complete_corpus_tokens'] != count:
            raise ValueError('Input replay differs')
        return dict(replayed=True, model_executed=False)
    def pin(count):
        return dict(file='tokens.u32le', dtype='u32le', count=count, bytes=count*4,
                    sha256=parent.digest(raw[:count*4]))
    value = dict(format='toklet-llama3-original-quality-inputs-v1', dataset=parent.DATASET,
                 tokenizer=parent.TOKENIZER, corpus=parent.CORPUS, parents=parent.PARENTS,
                 blocks=BLOCKS, policy=POLICY, complete_corpus_tokens=count,
                 tokens=pin(10001), long_probe_prefix=pin(1026), model_executed=False)
    validate(value)
    folder.mkdir()
    (folder/'tokens.u32le').write_bytes(raw)
    (folder/'manifest.json').write_text(json.dumps(value, indent=2)+'\n')
    load(folder, parent.digest((folder/'manifest.json').read_bytes()))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path)
    group.add_argument('--replay', type=Path)
    parser.add_argument('--manifest-sha256')
    args = parser.parse_args()
    if args.replay and not args.manifest_sha256:
        parser.error('Replay requires external manifest SHA')
    print(json.dumps(run(args.prepare or args.replay, bool(args.replay), args.manifest_sha256)))
