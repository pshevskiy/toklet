"""Join accepted M6 evidence; do not modify historical qualification scope."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'Cargo.toml').is_file())
OUT = ROOT / 'benchmarks/results/2026-10-19-m6-release-qualification-v1.json'
DEPS = {
    'm5': '2026-10-15-m5-release-qualification-v1.json',
    'component': '2026-10-15-m6-online-component-v1.json',
    'admission': '2026-10-15-m6-long-admission-component-v1.json',
    'loss': '2026-10-15-m6-online-full-loss-qualification-v1.json',
    'tasks': '2026-10-15-m6-online-task-quality-qualification-v1.json',
    'short': '2026-10-15-m6-short-performance-qualification-v1.json',
    'application': '2026-10-15-m6-full-application-qualification-v1.json',
    'kv': '2026-10-15-m6-kv-component-qualification-v1.json',
    'long_quality': '2026-10-16-m6-full-long-quality-qualification-v1.json',
    'winner': '2026-10-16-m6-winner-investigation-qualification-v1.json',
    'context': '2026-10-19-m6-context-qualification-v1.json',
}


EXPECTED = {'m5': '4d38941da89e94cfa357f15cec35923678af2a0b2e6b5a02e45dcc7d03da66b9', 'component': 'a5c596891a850669e9fd88e3ae7bef7412dd0f80fb414ff397a8069dd8cf4085', 'admission': 'fe50a340de7ada257479aa0c68ff841028965cf50ee183bde4e5a2c2eebe7c02', 'loss': '7b87646805ac5c6053150b3d7f02a06ea37daf46c967222401d6735d3a64def9', 'tasks': 'ca4a63ab48b934436c8b05cea16817e04355ab9cfd584a48db41c44199513842', 'short': '09a7aafeb1c3d448667d98aa6f629fe783ae338f2e4c6191c31841193e53775a', 'application': 'dde04b369f5564fb21e8704941f9120a8dd2e9c5eeb7ec62833cbbbf7d195fdf', 'kv': 'd2adcf5c8693c3f038658482d7e6dfcb7336344187990deafbe64f856c47848d', 'long_quality': 'bb371d5e0c9c12c4dfada6a841c7aa5cd5ccdfc4e0b44037d7f9537406fd5dcb', 'winner': '1d10ed435bcbe355af2ed2c204728795489624a311f17338e7fe8c839e57c735', 'context': 'f3e26bead96d52937e598c0f5c8889a80852d5909adb010f93279c24c2ad8b17'}

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if OUT.exists():
        raise ValueError('Exclusive release draft already exists')
    source = str(Path(__file__).resolve().relative_to(ROOT))
    pins = {source: sha(ROOT / source)}
    values = {}
    for key, name in DEPS.items():
        relative = 'benchmarks/results/' + name
        with (ROOT / relative).open('rb') as stream:
            raw = stream.read(16 * 1024**2 + 1)
        if len(raw) > 16 * 1024**2:
            raise ValueError('Oversize receipt: ' + relative)
        pins[relative] = hashlib.sha256(raw).hexdigest()
        if pins[relative] != EXPECTED[key]:
            raise ValueError('Accepted prerequisite bytes differ: ' + key)
        value = json.loads(raw)
        if key in ('component', 'admission'):
            if value['astra_review']['status'] != 'accepted' or value['astra_review']['remaining_findings']:
                raise ValueError('Unaccepted component: ' + key)
        elif value.get('passed') is not True or value.get('review', {}).get('accepted') is not True:
            raise ValueError('Unaccepted qualification: ' + key)
        values[key] = value
    context = values['context']
    if any(context[k] is not True for k in ('complete_frozen_context_coverage',
            'full_long_numerical_quality_qualified', 'context_boundary_coverage_complete')):
        raise ValueError('Incomplete context')
    if context['records'] != 32 or context['m6_complete'] is not False or context['kv_retention_qualified'] is not False:
        raise ValueError('Historical context scope differs')
    if values['kv']['adoption_qualified'] is not False:
        raise ValueError('Unexpected I8 adoption')
    for row in values['loss']['models']:
        if row['scored_targets'] != 10000 or row['loss_gate_passed'] is not True or row['increase_nats_per_token'] > .05:
            raise ValueError('Full loss gate')
    for row in values['tasks']['models']:
        if row['cases'] != 100 or row['resolved_outcomes'] != 100 or row['drop_percentage_points'] > 2:
            raise ValueError('Full task gate')
    if values['short']['confirmed_complete_request_useful'] is not True:
        raise ValueError('Short request benefit absent')
    budget = 6 * 1024**3
    for row in context['models'].values():
        if max(row['maximum_admission_bytes'], row['maximum_native_charge_bytes']) > budget:
            raise ValueError('Context memory budget')
    for row in values['application']['models']:
        if max(row['memory_admitted_bytes'], row['max_joint_native_bytes'], row['decoder_peak_bytes']) > budget:
            raise ValueError('Application memory budget')
    def merge(group):
        for path, expected in group.items():
            if path in pins and pins[path] != expected:
                raise ValueError('Conflicting closure binding: ' + path)
            pins[path] = expected

    # Read each exact replay from its accepted receipt, then admit the closure
    # it already checked. Historical source snapshots stay separate paths;
    # current runtime/model/contract bindings are required without filtering.
    replay_paths = [
        ('benchmarks/results/2026-10-19-m6-context-reviewed-closure-v1.json',
         context['published_copies'], context['evidence_pins']),
        ('benchmarks/results/2026-10-15-m6-application-full-replay-v1.json',
         values['application']['published_copies'], values['application']['evidence_pins'])]
    for path, copies, evidence in replay_paths:
        original = copies[path]
        candidates = {evidence[key] for key in (original, path) if key in evidence}
        if len(candidates) != 1:
            raise ValueError('Missing or conflicting replay pin: ' + path)
        expected = candidates.pop()
        with (ROOT / path).open('rb') as stream:
            data = stream.read(16 * 1024**2 + 1)
        if len(data) > 16 * 1024**2 or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Accepted replay differs: ' + path)
        replay = json.loads(data)
        if replay.get('passed') is not True:
            raise ValueError('Unpassed replay: ' + path)
        merge({path: expected})
        merge(replay['bindings'])
    for key in ('loss', 'tasks', 'application'):
        rows = values[key]['models']
        if len(rows) != 2 or {row['model'] for row in rows} != {'small', 'larger'}:
            raise ValueError('Exact model coverage required: ' + key)
    if set(context['models']) != {'small', 'larger'}:
        raise ValueError('Exact context model coverage required')
    required = {'Cargo.toml', 'Cargo.lock', 'src/metal/kernels.metal',
                'artifacts/m6-context-worker-native-v1/metal_context',
                'artifacts/m6-context-cli-native-v1/toklet',
                'benchmarks/m6-context-v1/manifest.json'}
    for model in ('Qwen2.5-0.5B-Instruct', 'Qwen2.5-1.5B-Instruct'):
        required.update('models/' + model + '/' + name for name in
                        ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'))
    required.update(str(path) for path in (ROOT / 'src').rglob('*') if path.is_file())
    required = {str(Path(path).relative_to(ROOT)) if Path(path).is_absolute() else path for path in required}
    if not required.issubset(pins):
        raise ValueError('Current runtime/payload closure incomplete: ' + str(required - pins.keys()))
    checks = context['checks']
    for path, expected in checks['protected_cpu_binaries'].items():
        if path in pins and pins[path] != expected:
            raise ValueError('Conflicting protected binary binding')
        pins[path] = expected
    result = dict(format='toklet-m6-release-qualification-v1', passed=True,
        review=dict(reviewer='Astra', accepted=False, publication='pending', remaining_findings=[]),
        source_parent=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        prerequisite_receipts={key: 'benchmarks/results/' + name for key, name in DEPS.items()},
        bindings=pins, checks=checks,
        provenance_scope='Exact accepted context and application replay closures include current Cargo/src/Metal, resident binaries, model/tokenizer/packed artifacts and contracts; historical snapshots retain distinct paths. Final documentation and audit-publication helpers are separate.',
        qualified_device='Apple M1 Pro / MacBookPro18,3 / 16 GiB / macOS26.6.2(25G83)',
        context_capacities=[4096,8192,16384,32768], output_reservation=128,
        models={
            'small': dict(model='Qwen2.5-0.5B-Instruct', projection='HalfCompensated', normalization='Original'),
            'larger': dict(model='Qwen2.5-1.5B-Instruct', projection='Q4CompensatedTiled v2 group8', normalization='Compensated')},
        attention='Online', kv_dtype='FP16', readback='FullLogits', prefill_chunk=128,
        full_loss_qualified=True, full_task_gate_qualified=True,
        complete_context_qualified=True, maximum_context_application_qualified=True,
        confirmed_short_numerical_request_benefit=True,
        long_timing_ranking_qualified=False, natural_long_document_ability_qualified=False,
        kv_retention_qualified=True,
        kv_decision='Retain FP16 after complete context/application <=6GiB qualification and accepted mixed-benefit I8 investigation; decline optional production integration.',
        i8_engine_adoption_qualified=False,
        i8_adoption_gates='Cache-write, full model loss/tasks, long-state and end-to-end gates not run or waived; required before future adoption.',
        m6_tasks={'M6.1': True, 'M6.2': True, 'M6.3': True}, m6_complete=True,
        m7_complete=False, defaults_unchanged=True,
        defaults=dict(backend='CPU', cpu_projection='Scalar', metal_projection='Original',
                      metal_attention='Reference', readback='FullLogits', prefill_chunk=128, context_capacity=2048),
        external_runtime_lead_qualified=False, physical_iphone_qualified=False,
        report_path='docs/experiments/2026-10-19-m6-release.md')
    # Recheck every admitted receipt/source and protected binary before publication.
    for path, expected in pins.items():
        if sha(ROOT / path) != expected:
            raise ValueError('Endpoint changed: ' + path)
    with OUT.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(OUT.relative_to(ROOT)), bindings=len(pins), review_pending=True)))


if __name__ == '__main__':
    main()
