"""Qualification runner for fixed authored literal programs only.

The enclosing window must admit the machine/API and restore its prior state.
This runner never accepts dataset programs, CLI vectors, or model responses.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import linux_code_isolation_probe as boundary
import linux_task_child as child
import linux_task_controller as controller
import linux_task_native_cases as authored
import linux_task_native_check as replay


def run(api, image_id, runtime, python_version, output):
    output = Path(output).resolve()
    raw = (boundary.ROOT / 'benchmarks/raw').resolve()
    if not output.is_relative_to(raw) or output.exists():
        raise ValueError('Fresh authored literal raw output is required')
    source = controller.bundled_source()
    controller.admit_bootstrap(source)
    inventory = authored.cases()
    names = [Path(__file__).resolve()]
    names += [Path(module.__file__).resolve() for module in
             (controller, authored, replay, child)]
    names += [boundary.ROOT / 'tools/bench' / name for name in
              ('linux_task_evidence.py', 'linux_task_transport.py',
               'linux_code_isolation_child.py', 'code_isolation_child.py',
               'linux_code_isolation_probe.py', 'task_cases_probe.py',
               'code_isolation_probe.py', 'prepare.py')]
    identities = {str(path.relative_to(boundary.ROOT)): boundary.digest(path) for path in names}
    record = {'format': 'toklet-authored-literal-native-probe-v1', 'completed': False,
              'scope': 'Fixed synthetic source tests only; no held-out or model program execution.',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': identities, 'image_id': image_id,
              'runtime': runtime, 'python_version': python_version,
              'bundle_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'cases': inventory, 'runs': []}
    try:
        for case in inventory:
            item = controller.execute(api, image_id, source, case['payload'])
            record['runs'].append(item)
            result = replay.check(item, case, runtime, python_version)
            item['qualification_result'] = result
            item['expectation_passed'] = True
            print(json.dumps({'case_id': case['payload']['case_id'],
                              'outcome': result['outcome']}), flush=True)
        if identities != {name: boundary.digest(boundary.ROOT / name) for name in identities}:
            raise ValueError('Authored literal qualification sources changed')
        record['completed'] = True
    except BaseException as error:
        record['failure'] = {'type': type(error).__name__, 'message': str(error)[:4096]}
        if isinstance(error, boundary.ControlError):
            record['failed_control'] = error.record
    record['completed_utc'] = datetime.now(timezone.utc).isoformat()
    with output.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return record
