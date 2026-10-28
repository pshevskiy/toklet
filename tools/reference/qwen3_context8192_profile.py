"""One prospective depth profile shared by input preparation and BF16 execution.

No runtime, model, input reads or arbitrary-depth configuration on import.
Future depths require a separately reviewed concrete profile.
"""
import hashlib
import json
import re

CAPACITY = 8192
VOCAB = 151936
PROFILE = 'qwen3-bf16-context8192-step8192-layer-barriers-chunk8-v1'
ENDPOINTS = (255, 256, 257, 1024, 1025)
PROBE_RANGES = ((253, 258), (1023, 1024))
QUALITY_RANGES = ((7167, 8191),)
NUMERICAL = '21d54e567025f70fdda01d7f1bede1a71f730c6b7c10aec62f5aaffec52a7f08'
QUALITY = '908e037de5b399d9e9a8dd09aa9fcc568f67abb714b06508672deac6bb8ac529'
F2_INPUT = '710054f18fe9a983399c230d7c31c7e99ddcff24f0748498dc3ce4a8b917c7c9'
F2_RESULT = '2961d4baef9c1fc488ea2497efda705ea5b8e96f3d4a021bebc1f889235cc9a6'
F2_A = '45f734c5fca32780b251eb14bf777bd4c512ba36a9a655c256d064aeea8b5c0b'
REQUIRED = dict(format='toklet-qwen3-context8192-inputs-v1', capacity=8192,
    numerical_parent_manifest_sha256=NUMERICAL, quality_parent_manifest_sha256=QUALITY,
    f2_input_manifest_sha256=F2_INPUT, preserved_quality_spans=[[0, 10001], [10001, 14097]],
    quality_global_token_start=14097, score_rows=[7167, 8191],
    score_target_indices=[7168, 8192], targets=1024, quality_mean_nll_increase_limit=0.05)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def identity(value, maximum, exact=None):
    if (not isinstance(value, dict) or set(value) != {'bytes', 'sha256'}
            or type(value['bytes']) is not int or not 0 < value['bytes'] <= maximum
            or exact is not None and value['bytes'] != exact or not digest(value['sha256'])):
        raise ValueError('Complete bounded identity required')


def validate(manifest):
    for key, expected in REQUIRED.items():
        # JSON spelling distinguishes bools from ints even inside scoring ranges.
        if json.dumps(manifest.get(key), sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError('Frozen8192 input/quality policy differs: ' + key)
    if manifest.get('complete') is not True:
        raise ValueError('Complete input publication required')
    files = manifest.get('files', {})
    if set(files) != {'A.u32le', 'B.u32le', 'Q.u32le'}:
        raise ValueError('Exactly three8192 arrays required')
    for item in files.values():
        identity(item, 32768, exact=32768)
    receipts = manifest.get('parent_receipts', {})
    if set(receipts) != {'f2_result', 'retained_parallel_result'}:
        raise ValueError('Caller-admitted F2 and retained parallel parents required')
    for item in receipts.values():
        identity(item, 16 * 1024**2)
    if (receipts['f2_result']['sha256'] != F2_RESULT
            or manifest.get('a4096_prefix_sha256') != F2_A
            or not digest(manifest.get('admission_sha256'))):
        raise ValueError('Historical A/F2 or admitted parent identity differs')


def validate_ids(values, count):
    if (not isinstance(values, list) or len(values) != count
            or any(type(v) is not int or not 0 <= v < VOCAB for v in values)):
        raise ValueError('Complete integer in-vocabulary IDs required')
