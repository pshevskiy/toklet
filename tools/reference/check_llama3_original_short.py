"""Independent stdlib-only replay of saved source rows and frozen consistency gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def stats(row, target):
    maximum = max(row)
    mass = math.fsum(math.exp(x-maximum) for x in row)
    nll = (maximum-row[target]) + math.log(mass)
    return nll, row[target], next(i for i,x in enumerate(row) if x == maximum)


def check(folder, input_folder, manifest_sha):
    folder = Path(folder)
    inputs = Path(input_folder)
    raw_manifest = (inputs/'manifest.json').read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != manifest_sha:
        raise ValueError('Frozen manifest differs')
    manifest = json.loads(raw_manifest)
    raw_ids = (inputs/'tokens.u32le').read_bytes()
    if len(raw_ids) != 72 or hashlib.sha256(raw_ids).hexdigest() != manifest['tokens_sha256']:
        raise ValueError('Frozen input IDs differ')
    ids = struct.unpack('<18I', raw_ids)
    report = json.loads((folder/'source.json').read_bytes())
    if report['input_manifest_sha256'] != manifest_sha:
        raise ValueError('Source report has different inputs')
    if report['format'] != 'toklet-llama3-original-short-source-v1' or report['full_quality_qualified'] or not report['passed'] or [c['prefix_tokens'] for c in report['cases']] != [1,4,8,17]:
        raise ValueError('Incomplete source report')
    for case in report['cases']:
        n = case['prefix_tokens']
        if case['target_id'] != ids[n] or case['target_position'] != n or case['logit_row'] != n-1 or not case['consistency']['passed']:
            raise ValueError('Source target alignment differs')
        rows = {}
        for label in ['batch','prefix']:
            descriptor = case['arrays'][label]
            name = descriptor['file']
            if name != f'prefix-{n}-{label}.f32le' or descriptor['bytes'] != 128256*4 or descriptor['sha256'] != case['consistency'][label]['row_sha256']:
                raise ValueError('Invalid row descriptor')
            raw = (folder/name).read_bytes()
            if len(raw) != descriptor['bytes'] or hashlib.sha256(raw).hexdigest() != descriptor['sha256']:
                raise ValueError('Saved source row differs')
            row = struct.unpack('<128256f', raw)
            if not all(math.isfinite(x) for x in row):
                raise ValueError('Nonfinite saved row')
            nll, target, greedy = stats(row,case['target_id'])
            given = case['consistency'][label]
            if not math.isfinite(given['nll']) or abs(nll-given['nll']) > 1e-12 or target != struct.unpack('<f',struct.pack('<I',given['target_fp32_bits']))[0] or greedy != given['greedy_lowest_id']:
                raise ValueError('Source statistics do not replay')
            rows[label] = row
        if any(abs(a-b) > 1e-3+1e-4*abs(b) for a,b in zip(rows['batch'],rows['prefix'])):
            raise ValueError('Saved row consistency fails')
        if abs(stats(rows['batch'],case['target_id'])[0]-stats(rows['prefix'],case['target_id'])[0]) > 1e-4:
            raise ValueError('Saved target NLL consistency fails')
    return {'passed': True, 'rows': 8, 'vocabulary': 128256, 'full_quality_qualified': False}


def compare_native(folder, input_folder, manifest_sha, native_file):
    checked = check(folder, input_folder, manifest_sha)
    source = json.loads((Path(folder)/'source.json').read_bytes())
    native = json.loads(Path(native_file).read_bytes())
    if (native['format'] != 'toklet-llama3-original-short-native-v1' or not native['passed']
            or native['full_quality_qualified'] or native['input_manifest_sha256'] != manifest_sha
            or [c['prefix_tokens'] for c in native['cases']] != [1,4,8,17]):
        raise ValueError('Native short report differs')
    differences = []
    for original, candidate in zip(source['cases'], native['cases']):
        n, target = original['prefix_tokens'], original['target_id']
        score = candidate['score']
        if (candidate['target_position'] != n or candidate['target_id'] != target
                or score['target'] != target or score['evaluated'] != n or score['vocabulary'] != 128256
                or type(score['greedy']) is not int or not 0 <= score['greedy'] < 128256):
            raise ValueError('Native target alignment differs')
        values = [score[k] for k in ['targetLogit','greedyLogit','negativeLogLikelihood']]
        if (not all(type(x) in [int,float] and math.isfinite(x) for x in values)
                or score['negativeLogLikelihood'] < 0 or score['greedyLogit'] < score['targetLogit']
                or any(struct.unpack('<f',struct.pack('<f',x))[0] != x for x in values[:2])):
            raise ValueError('Invalid native scalar statistics')
        reference = original['consistency']['prefix']
        differences.append({'prefix_tokens':n,'target_id':target,'original_nll':reference['nll'],
                            'native_nll':score['negativeLogLikelihood'],
                            'native_minus_original':score['negativeLogLikelihood']-reference['nll'],
                            'greedy_matches':score['greedy']==reference['greedy_lowest_id'],
                            'original_greedy_margin':original['consistency']['prefix_greedy_margin']})
    return dict(checked, native_cases=4, differences=differences,
                short_mean_nll_difference=math.fsum(c['native_minus_original'] for c in differences)/4,
                adoption_decision='not_evaluated_short_alignment_only')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    parser.add_argument('inputs',type=Path);parser.add_argument('manifest_sha')
    parser.add_argument('--native',type=Path)
    args=parser.parse_args()
    result=compare_native(args.folder,args.inputs,args.manifest_sha,args.native) if args.native else check(args.folder,args.inputs,args.manifest_sha)
    print(json.dumps(result,indent=2))
