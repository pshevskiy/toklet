"""Ordinary-first literal grading with one qualified ambiguous-137 CPU retry."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import shlex
from unittest.mock import patch
import grading_support as support
from llama3_quality_grading import bind as bind_file
import linux_task_controller as literal
import linux_task_evidence as legacy
import linux_process_cpu_window as observed
import linux_process_cpu_evidence as cpu
import paired_task_grading as shared

ROOT = shared.ROOT
PROOF = ROOT/'benchmarks/results/2026-11-04-linux-process-cpu-witness-v1.json'
PROOF_SHA = 'd39b1d233512bd9afefb665bddf045742a05eb4f2efa81ea6fa27100b0964e0b'
AMBIGUOUS = 'Literal evaluator termination is ambiguous'


def ordinary(record, payload, runtime, python):
    """Return None only after the legacy validator admits natural unknown137."""
    try:
        return legacy.classify(record, payload, runtime, python)
    except ValueError as error:
        eligible = (str(error) == AMBIGUOUS and record['container_exit_code'] == 137
            and record['transport_exit_code'] == 137
            and not any(record[key] for key in ['wall_timed_out', 'output_limit_exceeded',
                                                'process_group_killed', 'container_forced_kill'])
            and record['container_state']['OOMKilled'] is False)
        if not eligible:
            raise
        captured = legacy.verified_capture(record)
        _, separator, rest = captured['stdout'].partition(b'\n')
        if not separator or rest or captured['stderr']:
            # The authored scope has one readiness frame and no subsequent
            # output. A contradictory terminal or unframed data stays unknown.
            raise
        return None


class Runtime:
    def __init__(self):
        self.meta = None
        self.policy = dict(kind='ordinary-first-qualified-process-prof-retry-v1', qualification_sha256=PROOF_SHA,
                           threshold_ns=4_000_000_000, retried_cases=[], original_attempts_preserved=True)

    def bind(self, bindings):
        bind_file(bindings, PROOF, PROOF_SHA)
        raw = PROOF.read_bytes()
        support.require(len(raw) <= 1024**2 and hashlib.sha256(raw).hexdigest() == PROOF_SHA,
                        'Bounded qualified CPU witness required')
        self.qualification = json.loads(raw)
        support.require(self.qualification['authored_witness_qualified'] is True
                        and self.qualification['threshold_ns'] == 4_000_000_000,
                        'Authored witness qualification required')
        for name, sha in self.qualification['receipts_sha256'].items():
            bind_file(bindings, ROOT/name, sha)
        for name in [__file__, 'tools/bench/gemma_quality_grading.py', 'tools/bench/linux_task_evidence.py',
                     'tools/bench/linux_task_controller.py', 'tools/bench/linux_task_transport.py']:
            bind_file(bindings, ROOT/name)
        self.observer_source_sha = self.qualification['receipts_sha256']['tools/bench/linux_process_cpu_guest.py']
        prior = shared.checked_json(shared.PRIOR, shared.PRIOR_SHA, 16*1024**2)
        self.literal_runtime = prior['admitted_runtime']
        self.literal_python = prior['admitted_python_version']

    def prepare(self, podman):
        if self.meta is not None:
            return
        code = ("import sys,hashlib,json,os,platform;from pathlib import Path;"
                "print(json.dumps(dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),"
                "kernel=platform.release(),observer_namespace=os.readlink('/proc/self/ns/pid'),"
                "observer_python_sha=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest())))")
        row = shared.boundary.capture([shared.PODMAN, 'machine', 'ssh', shared.lifecycle.MACHINE,
            shlex.join(['/usr/bin/python3', '-I', '-S', '-c', code])], 20, 65536)
        support.require(not row.get('failure') and row['exit_code'] == 0 and not row['wall_timed_out']
                        and not row['output_limit_exceeded'] and row['cleanup_errors'] == []
                        and row['surviving_group_members'] == [], 'CPU observer runtime query failed')
        meta = json.loads(shared.boundary.output(row))
        support.require(meta['kernel'] == self.qualification['kernel']
                        and meta['observer_python_sha'] == self.qualification['observer_python_sha256'],
                        'Qualified observer guest runtime changed')
        self.meta = meta
        self.policy.update(guest_runtime=meta, guest_runtime_capture=row)

    @staticmethod
    def bundled_source():
        return literal.bundled_source()

    @staticmethod
    def admit_bootstrap(source):
        return literal.admit_bootstrap(source)

    def execute(self, podman, image, source, payload):
        self.prepare(podman)
        original = literal.execute(podman, image, source, payload)
        try:
            result = ordinary(original, payload, self.literal_runtime, self.literal_python)
        except BaseException as error:
            original['ordinary_classification_failure'] = dict(type=type(error).__name__, message=str(error)[:2048])
            return original  # The shared runner journals before classifying.
        if result is None:
            self.policy['retried_cases'].append(payload['case_id'])
            try:
                retry = observed.execute(podman, image, source, payload)
            except BaseException as error:
                original['observed_cpu_retry_failure'] = dict(type=type(error).__name__, message=str(error)[:2048])
                return original
            original['observed_cpu_retry'] = dict(qualification_sha256=PROOF_SHA,
                program_sha256=payload['program_sha256'], raw=retry)
        return original

    def classify(self, original, payload, runtime, python):
        support.require('ordinary_classification_failure' not in original,
                        'Recorded ordinary classification failed; original remains unresolved')
        result = ordinary(original, payload, runtime, python)
        if result is not None:
            return result
        retry = original.get('observed_cpu_retry')
        support.require(retry is not None and retry['qualification_sha256'] == PROOF_SHA
                        and retry['program_sha256'] == payload['program_sha256'],
                        'Original137 remains unresolved without its identical qualified retry')
        raw = retry['raw']
        support.require(ordinary(raw, payload, runtime, python) is None,
                        'Observed retry changed termination or hit a watchdog; original remains unresolved')
        source = literal.bundled_source()
        for record in [original, raw]:
            before = record['container_inspect_before']
            support.require(before['Id'].removeprefix('sha256:') == record['container_id'].removeprefix('sha256:')
                            and before['Image'].removeprefix('sha256:') == shared.IMAGE
                            and legacy.canonical(literal.validate_configuration(before, source)) ==
                                legacy.canonical(record['configuration']), 'Retry image/launch admission differs')
        support.require(original['container_id'] != raw['container_id'] and original['container_name'] != raw['container_name']
                        and original['input_sha256'] == raw['input_sha256']
                        and legacy.canonical(original['configuration']) == legacy.canonical(raw['configuration']),
                        'Fresh retry payload/configuration differs')
        captures = raw['process_cpu_captures']
        support.require(len(captures) == 1, 'One independently owned CPU observation required')
        checked = cpu.validate(captures[0], container_id=raw['container_id'].removeprefix('sha256:'),
            image=shared.IMAGE, program_sha=payload['program_sha256'], observer_source_sha=self.observer_source_sha, **self.meta)
        support.require(checked['budget_exhausted'], 'Observed retry below CPU budget; original remains unresolved')
        return dict(outcome='resource_failure', passed=False, binding=legacy.child.admit_payload(payload),
                    reason='qualified_observed_retry_process_cpu_budget', process_cpu=checked,
                    original_unresolved_attempt_preserved=True)

    @contextmanager
    def install(self):
        # Only the shared runner's coarse interfaces change; the original
        # controller/evidence modules and qualified source hashes stay intact.
        with patch.object(shared, 'controller', self), patch.object(shared, 'evidence', self):
            yield
