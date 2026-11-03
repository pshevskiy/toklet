"""Qwen3 adapter around the existing qualified Linux literal evaluator/window.

Model processes must finish first. No containment, source-test construction,
text scorer, resource policy or unresolved-case aggregation rule changes.
"""
from pathlib import Path
import hashlib
import json
import os
import signal
import stat
import subprocess
import time

import qwen3_quality_results as results
import linux_code_isolation_probe as boundary
import linux_task_controller as controller
import linux_task_evidence as evidence
import podman_machine_lifecycle as lifecycle
from held_out_task_window import running_containers
from grading_support import verify_file, selected, write_json, finish_decision, supervise_window

ROOT = results.contract.ROOT
PODMAN = '/opt/podman/bin/podman'
IMAGE = 'eb7d16fec874f15b584063275012ac719885a2b87574d124c0da9c156d504629'
CONTRACT_SHA = '908e037de5b399d9e9a8dd09aa9fcc568f67abb714b06508672deac6bb8ac529'
PRIOR = ROOT/'benchmarks/results/2026-10-11-linux-code-isolation-probe-v8.json'
PRIOR_SHA = '0736ec7d8cf7481c5327c01c5f0c8555b6e3c8669c6c63e7a267f0044873d3f1'
LITERAL = ROOT/'benchmarks/results/2026-10-11-literal-native-v3-probe.json'
LITERAL_SHA = '77cdda26d80a079bc56da9a80d52f7fa09d7b2942ce768d464128f07798a9453'


def require_idle(record):
    results.require(record.get('passed') is True and record.get('primary_error') is None
        and record.get('endpoint_errors')==[] and record.get('observations',{}).get('scope',{}).get('full_model_collections_complete') is True,
        'complete successful model collections required before grading')
    commands = record.get('commands')
    results.require(isinstance(commands,list) and len(commands)==7, 'all seven owned collection phases required')
    pids = []
    for command in commands:
        pid = command.get('pid')
        results.require(type(pid) is int and pid>1 and pid not in pids and command.get('exit_code')==0
            and command.get('primary_error') is None and command.get('cleanup_errors')==[], 'natural owned collection exit required')
        for target in (pid,-pid):
            try:os.kill(target,0)
            except ProcessLookupError:pass
            else:raise ValueError('model process or owned group remains live')
        pids.append(pid)
    return pids


def run(checks_path, checks_sha, output, endpoint):
    checks_path = Path(checks_path).resolve();output = Path(output).resolve()
    results.require(output.is_relative_to(ROOT/'artifacts'), 'fresh artifact grading destination required')
    output.mkdir()
    record = dict(format='toklet-qwen3-full-quality-grading-v1',completed=False,commands=[],runs=[],outcomes={},
        started_by_this_task=False,restored_stopped=False,pipe_lifetime_finished=False,
        collection_sha256=checks_sha,generated_code_runs_on_host=False,unresolved_case_ids=[],
        scope='Literal source-test outcomes, no adversarial shared-interpreter grading attestation or timing acceptance')
    bindings = {};decoded = {};collections = {};cases = []

    def guard():
        for name,expected in bindings.items():verify_file(ROOT/name,expected)
        require_idle(collection)
        results.require(boundary.digest(checks_path)==checks_sha, 'collection receipt changed')
        endpoint()

    try:
        collection = results.contract.safe.document(results.contract.checked(checks_path,checks_sha,4*1024**2))
        record['exited_owned_model_pids'] = require_idle(collection)
        bindings = {**collection['bindings'],**collection['outputs']}
        m = results.contract.safe.document(results.contract.checked(ROOT/'benchmarks/qwen3-quality-v1/manifest.json',CONTRACT_SHA,2*1024**2))
        for name,sha in m['candidate']['numerical_source_sha256'].items():
            results.require(bindings[name]['sha256']==sha, 'collected numerical candidate differs from frozen contract')
        guard()
        inputs = ROOT/'artifacts/m8-qwen3-quality-inputs-v1/inputs'
        payload = results.contract.checked(inputs/'tasks.jsonl',m['files']['tasks.jsonl']['sha256'],4*1024**2)
        cases = [results.contract.safe.document(line) for line in payload.splitlines()]
        loss_raw = results.contract.checked(inputs/'held_out_loss.u32le',m['files']['held_out_loss.u32le']['sha256'],40004)
        import struct
        ids = list(struct.unpack('<10001I',loss_raw))
        for profile in ('baseline','native'):
            prefix = 'source' if profile=='baseline' else 'native'
            def rows(mode):
                path = checks_path.parent/(prefix+'-'+mode+('.jsonl'if profile=='baseline'else'.stdout'))
                return results.events(path,collection['outputs'][str(path.relative_to(ROOT))]['sha256'])
            collections[profile] = results.loss(rows('loss'),profile,m,ids)
            responses = results.tasks(rows('tasks'),profile,m,cases)
            decoded[profile] = results.decode(cases,responses)
            write_json(output/(profile+'-decoded.json'),decoded[profile])
        record['losses'] = collections
    except BaseException as error:
        record['failure']=dict(type=type(error).__name__,message=str(error)[:4096])
        record['source_and_evidence_bindings']=bindings
        write_json(output/'result.json',record)
        return record

    from paired_task_grading import grade_paired_cases
    return grade_paired_cases([results.legacy_case(c)for c in cases],decoded,guard,output,record,bindings,
        lambda outcomes:results.decision(collections['native'],collections['baseline'],cases,outcomes['native'],outcomes['baseline']))
