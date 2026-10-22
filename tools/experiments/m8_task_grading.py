"""Native-app admission for the qualified literal grading window.

Never starts a VM or executes a response. The existing window/controller owns
those operations; this adapter authenticates immutable inputs and phone idleness.
"""
import hashlib
import json
from pathlib import Path
import sys

import m8_task_responses as responses
import task_grading_window as window

ROOT = responses.ROOT
QUAL = 'benchmarks/results/2026-10-22-native-mlx8-task-responses-v1.json'
QUAL_SHA = '72d682e9d62c1e7e3a36a8a3c3360b5603c7215a8b4559c2700daa7fcb025a5b'
NORMAL = 'artifacts/m8-task-responses-driver-v3/normalized/normalization.json'
NORMAL_SHA = '1838030221dc19df6a5f3c5dd81877a6d568b98f0da36997e1f91cf7d46524c3'
BASELINE = 'benchmarks/results/2026-10-15-m6-online-task-quality-qualification-v1.json'
BASELINE_SHA = 'ca4a63ab48b934436c8b05cea16817e04355ab9cfd584a48db41c44199513842'
require = responses.require


def validate_derivative(data, platform, decoded, pids, runtime, raw):
    require(data['format'] == 'toklet-m8-app-task-response-derivative-v1'
        and data['platform'] == platform and data['quality_completed'] is False
        and data['source_qualification'] == dict(path=responses.QUAL, sha256=responses.QUAL_SHA)
        and data['tokenizer'] == dict(path=responses.TOKENIZER, sha256=responses.TOKENIZER_SHA,
                                    skip_special_tokens=True, runtime=runtime)
        and data['local_producer_pids'] == pids and data['responses'] == decoded
        and data['raw_sha256'] == hashlib.sha256(raw).hexdigest(),
        'Exact source-replayed literal derivative required')


class Admission:
    def __init__(self, output):
        self.pins = {str(Path(__file__).resolve().relative_to(ROOT)):
                     responses.package.sha(Path(__file__).resolve())}
        self.output = Path(output)
        self.collections = {}
        self.barrier = None
        for module in list(sys.modules.values()):
            filename = getattr(module, '__file__', None)
            if filename and filename.endswith('.py') and Path(filename).resolve().is_relative_to(ROOT/'tools'):
                path = Path(filename).resolve()
                responses.merge(self.pins, {str(path.relative_to(ROOT)): responses.package.sha(path)})

    def admit(self):
        qualification = self.read(ROOT/QUAL, QUAL_SHA)
        require(qualification['passed'] is True and qualification['task_quality_qualified'] is False
                and qualification['new_vm_runs'] == qualification['new_generated_program_executions'] == 0,
                'Accepted response normalization required')
        responses.merge(self.pins, qualification['source_pins'])
        normal = self.read(ROOT/NORMAL, NORMAL_SHA)
        require(qualification['normalization'] == dict(path=NORMAL, sha256=NORMAL_SHA)
                and normal == qualification['execution']['normalization'] and normal['passed'] is True
                and normal['primary_error'] is None and not normal['endpoint_errors']
                and len(normal['idle']['observations']) == 2, 'Exact completed normalization required')
        responses.merge(self.pins, normal['bindings'])
        q, self.cases, packets, original_pids = responses.admit_source(self.pins)
        require(self.cases == window.probe.tasks.load_cases(), 'Frozen full grading cases differ')
        producers = qualification['execution']['commands']
        require(len(producers) == 1 and all(type(c['pid']) is int and c['pid'] > 1
                and c['exit_code'] == 0 and c['primary_error'] is None and not c['cleanup_errors']
                for c in producers), 'Naturally exited normalization producer required')
        self.pids = sorted(set(original_pids + [c['pid'] for c in producers]))
        self.protected = dict(qualification['execution']['protected_binary_pins'])
        signed = q['execution']['nested_receipts']['qwen8-tasks-full-run']['signed_bundle']
        self.barrier = responses.IdleBarrier(self.output/'private-logs', self.pins, self.pids, signed)
        self.barrier.check()
        # This independent reconstruction happens only after fresh host and phone absence.
        tokenizer_json, runtime = responses.tokenizer_inputs(self.pins)
        from tokenizers import Tokenizer
        decoder = Tokenizer.from_str(tokenizer_json)
        for platform in responses.PLATFORMS:
            path = ROOT/qualification['derivatives'][platform]['path']
            require(path == (ROOT/NORMAL).parent/(platform+'.json'), 'Fixed derivative path required')
            saved = self.read(path, qualification['derivatives'][platform]['sha256'])
            raw_path = path.with_suffix('.raw.jsonl')
            raw = responses.capture(raw_path, self.pins,
                                    qualification['derivatives'][platform]['raw_sha256'])
            require(raw == packets[platform]['raw'], 'Original app-event derivative differs')
            decoded = responses.decode_responses(self.cases, packets[platform]['checked'], decoder)
            validate_derivative(saved, platform, decoded, original_pids, runtime, raw)
            self.collections[str(path)] = dict(collection_path=str(path),
                collection_sha256=self.pins[str(path.relative_to(ROOT))],
                raw_sha256=self.pins[str(raw_path.relative_to(ROOT))], cases=self.cases,
                responses=decoded, identity=dict(backend='metal', model='small', platform=platform,
                    engine='native Rust/Metal', weight_dtype='MLX affine8/group64 + Half exclusions',
                    model_revision='6474d4c1a05bd7bd634a9c993560e4e7272eda3d'))
        self.paths = [(ROOT/NORMAL).parent/(platform+'.json') for platform in responses.PLATFORMS]
        require(self.collections[str(self.paths[0])]['responses'] ==
                self.collections[str(self.paths[1])]['responses'], 'Paired literal responses differ')
        self.guard()

    def read(self, path, expected=None):
        return json.loads(responses.capture(path, self.pins, expected))

    def guard(self):
        responses.guard(self.pins)

    def validate_collection(self, path):
        self.guard()
        key = str(Path(path).resolve())
        require(key in self.collections, 'Only exact admitted native-app collections may grade')
        return self.collections[key]

    def idle_proof(self):
        return dict(format='toklet-native-mlx8-task-idle-v1',
                    normalization=dict(path=NORMAL, sha256=NORMAL_SHA),
                    collections=[str(path.relative_to(ROOT)) for path in self.paths])

    def admit_idle(self, path, selected):
        require(Path(selected).resolve() == self.paths[0], 'One fresh Mac grading window required')
        raw = responses.capture(path, self.pins)
        require(json.loads(raw) == self.idle_proof(), 'Exact normalization idle proof required')
        self.guard()
        # Generic window checks every host producer/group and both derivative pairs.
        # Duplicated paths are deliberate: each row adds a distinct host PID.
        require(len(self.pids) >= 2, 'Both original and normalization producers required')
        collections = []
        for i, pid in enumerate(self.pids):
            item = self.collections[str(self.paths[i % 2])]
            collections.append(dict(path=item['collection_path'], sha256=item['collection_sha256'],
                raw_sha256=item['raw_sha256'], identity=item['identity'], worker_pid=pid))
        return dict(proof_path=str(Path(path).resolve()), proof_sha256=hashlib.sha256(raw).hexdigest(),
                    collections=collections, protected_binaries=self.protected)

    def idle_check(self, _binding):
        self.guard()
        self.barrier.check()
        self.guard()

    def finalize_private(self, primary, endpoints):
        """Bind original diagnostics before public redaction, including late guards."""
        errors = list(endpoints)
        try:
            self.guard()
        except BaseException as error:
            errors.append(type(error).__name__+': '+str(error))
        responses.write_bound(self.output/'private-execution.json', dict(primary_error=primary,
            endpoint_errors=list(errors), commands=self.barrier.commands if self.barrier else []), self.pins)
        try:
            self.guard()
        except BaseException as error:
            errors.append(type(error).__name__+': '+str(error))
        responses.write_bound(self.output/'private-final-guard.json',
                              dict(primary_error=primary, endpoint_errors=list(errors)), self.pins)
        return errors

    def public(self):
        return dict(normalization=dict(path=NORMAL, sha256=NORMAL_SHA),
                    idle=self.barrier.public() if self.barrier else None,
                    collection_paths=[str(p.relative_to(ROOT)) for p in getattr(self, 'paths', [])])


def replay_outcomes(cases, decoded, grading):
    """Replay exact literal controller outcomes; no generated source executes here."""
    tasks, evidence = window.probe.tasks, window.probe.evidence
    require(grading['completed'] is True and grading['quality_completed'] is True
            and not grading.get('failure') and grading['unresolved_case_ids'] == []
            and grading['task_payload_sha256'] == tasks.TASK_SHA and len(cases) == len(decoded) == 100
            and len(grading['runs']) == 40, 'Complete resolved qualified grading required')
    runs, outcomes = iter(grading['runs']), []
    for case, row in zip(cases, decoded, strict=True):
        require(row['id'] == case['id'] and row['category'] == case['category']
                and hashlib.sha256(row['visible_text'].encode()).hexdigest() == row['visible_text_sha256'],
                'Literal response identity/text binding differs')
        if case['category'] == 'code':
            run = next(runs)
            program = tasks.human_eval_program(case, row['visible_text'])
            payload = dict(case_id=case['id'], program=program,
                           program_sha256=hashlib.sha256(program.encode()).hexdigest())
            require(run['id'] == case['id'] and run['completion_sha256'] == row['visible_text_sha256']
                    and run['program_sha256'] == payload['program_sha256']
                    and not run.get('infrastructure_failure'), 'Literal program/test binding differs')
            resolved = evidence.classify(run['raw'], payload, grading['runtime'], grading['python_version'])
            require(resolved == run['resolved'], 'Qualified literal outcome replay differs')
            passed = resolved['passed']
        else:
            passed = tasks.grade_text(case, row['visible_text'])
        outcomes.append(dict(id=case['id'], category=case['category'], passed=passed,
                             status='model_pass' if passed else 'model_fail'))
    require(next(runs, None) is None and outcomes == grading['outcomes'], 'Full outcome replay differs')
    aggregate = tasks.aggregate(cases, outcomes)
    require(aggregate == grading['aggregate'], 'Frozen task aggregate differs')
    return outcomes, aggregate


def quality_gate(aggregate, pins):
    prior = json.loads(responses.capture(ROOT/BASELINE, pins, BASELINE_SHA))
    require(prior['passed'] is True and prior['quality_completed'] is True
            and prior['review']['accepted'] is True and prior['infrastructure_unresolved'] == 0,
            'Accepted matched FP32 baseline required')
    baseline = next(row for row in prior['models'] if row['model'] == 'small')
    require(type(baseline['fp32_passed']) is int and baseline['fp32_passed'] == 2
            and baseline['unchanged_allowed_drop_percentage_points'] == 2
            and type(aggregate['passed']) is int and 0 <= aggregate['passed'] <= 100
            and aggregate['cases'] == 100, 'Unchanged100-case baseline/gate required')
    drop = baseline['fp32_passed'] - aggregate['passed']
    return dict(cases=100, fp32_passed=2, passed=aggregate['passed'],
                drop_percentage_points=drop, maximum_drop_percentage_points=2,
                gate_passed=drop <= 2,
                limitation='The original2/100 baseline and two-point allowance admit zero passes at the boundary.')
