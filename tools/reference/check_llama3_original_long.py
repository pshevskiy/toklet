"""Stdlib replay of eight original long rows and descriptive native comparison."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import check_llama3_original_short as short
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'bench'))
import prepare_llama3_original_quality as inputs


def check(folder, input_folder, manifest_sha, native_folder=None, native_report_sha=None):
    folder = Path(folder)
    _, ids = inputs.load(input_folder, manifest_sha)
    report = json.loads((folder/'source.json').read_bytes())
    if (report['format'] != 'toklet-llama3-original-long-source-v1' or not report['passed']
            or report['full_quality_qualified'] or report['input_manifest_sha256'] != manifest_sha
            or [c['prefix_tokens'] for c in report['cases']] != inputs.PREFIXES
            or report['peak_rss_bytes'] > 6*1024**3):
        raise ValueError('Incomplete long source report')
    short.replay_rows(folder, report['cases'], ids)
    result = dict(passed=True, rows=8, vocabulary=128256, full_quality_qualified=False)
    if native_folder is not None:
        native_folder = Path(native_folder)
        native_bytes = (native_folder/'result.json').read_bytes()
        if native_report_sha is None or hashlib.sha256(native_bytes).hexdigest() != native_report_sha:
            raise ValueError('Pinned native report differs')
        native = json.loads(native_bytes)
        if (native['format'] != 'toklet-llama3-quality-context-v1' or not native['ok']
                or native['capacity'] != 1025 or native['full_quality_qualified']
                or native['revision'] != inputs.parent.REVISION):
            raise ValueError('Native long diagnostic differs')
        independent = [c for c in native['observations'] if c['kind']=='independent']
        if len(independent) != 5:
            raise ValueError('Native prefix count differs')
        cases = {c['prefix']:c for c in independent}
        if set(cases) != {17,512,513,1024,1025}:
            raise ValueError('Native prefixes differ')
        differences = []
        for case in report['cases']:
            n = case['prefix_tokens']; descriptor = cases[n]
            raw = (native_folder/f'prefix-{n}.f32le').read_bytes()
            if len(raw) != 128256*4 or hashlib.sha256(raw).hexdigest() != descriptor['raw_sha256']:
                raise ValueError('Native saved row differs')
            values = struct.unpack('<128256f',raw)
            if not all(math.isfinite(x) for x in values):
                raise ValueError('Nonfinite native row')
            nll, _, greedy = short.stats(values, ids[n])
            reference = case['consistency']['prefix']
            differences.append(dict(prefix_tokens=n,target_id=ids[n],original_nll=reference['nll'],
                                    native_nll=nll,native_minus_original=nll-reference['nll'],
                                    greedy_matches=greedy==reference['greedy_lowest_id']))
        result.update(native_cases=4,differences=differences,
                      long_mean_nll_difference=math.fsum(c['native_minus_original'] for c in differences)/4,
                      adoption_decision='not_evaluated_long_alignment_only')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path);parser.add_argument('inputs',type=Path)
    parser.add_argument('manifest_sha');parser.add_argument('--native-folder',type=Path)
    parser.add_argument('--native-report-sha256')
    args=parser.parse_args()
    print(json.dumps(check(args.folder,args.inputs,args.manifest_sha,args.native_folder,args.native_report_sha256),indent=2))
