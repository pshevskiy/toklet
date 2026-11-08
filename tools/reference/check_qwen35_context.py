"""Frozen17-input/offset37 coupled fidelity and native recovery evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import check_qwen35_real_reference as prior
import check_qwen35_payload as pin
import qwen35_real_reference as producer

need = prior.need


def work(tokens, outcome='Success', committed=None):
    return dict(outcome=outcome, tokens_completed=tokens, tokens_committed=tokens if committed is None else committed,
        projections_completed=tokens*249, gathers_completed=tokens,
        convolution_steps_completed=tokens*24, recurrence_steps_completed=tokens*24,
        attention_heads_completed=tokens*128)


def native(folder):
    receipt = prior.document(folder/'checks.json')
    need(receipt['passed'] is True and type(receipt['exit_code']) is int and receipt['exit_code'] == 0
         and receipt['error'] is None and receipt['group_absent'] is True and receipt['changed_inputs'] == [],
         'Closed successful native context capture required')
    peak = receipt['lifetime_peak_physical_bytes']
    need(type(peak) is int and 0 < peak <= 6*1024**3, 'Measured native physical ceiling required')
    need(set(receipt['files_sha256']) == {'console.log', 'time-and-errors.log', 'inputs.json', 'capture/rows.f32le'}, 'Native seal inventory differs')
    for name, digest in receipt['files_sha256'].items():
        need(hashlib.sha256(prior.read(folder/name, 128*1024**2)).hexdigest() == digest, 'Sealed native evidence changed')
    log = prior.read(folder/'console.log', 2*1024**2).decode()
    labels = lambda tag: [prior.decode(s.split(' ',1)[1]) for s in log.splitlines() if s.startswith(tag+' ')]
    report, = labels('QWEN35_CONTEXT_REPORT')
    need(prior.exact(report, receipt['report']), 'Native report seal differs')
    tokens, position, values = producer.plan('context17')
    for name, value in dict(source_sha256=pin.FILE_SHA, container_bytes=pin.FILE_BYTES,
        config_sha256=pin.CONFIG_SHA, header_sha256=pin.HEADER_SHA, tokens=tokens,
        position_offset=position, capture_token=16, capacity=512, chunk=32,
        capture_bytes=values*4, arrays=1051, partitions=[17,1,3],
        all_logits_bits_exact=True, warmed_rust_allocations=0, admission_minus_one_rejected=True,
        failed_prefix_preserved=True, cancelled_prefix_preserved=True, retry_exact=True,
        observed_work=work(17), final_work=work(0,'Error',0),
        failed_work=dict(outcome='Error',tokens_completed=2,tokens_committed=0,projections_completed=502,
            gathers_completed=3,convolution_steps_completed=49,recurrence_steps_completed=49,attention_heads_completed=256),
        cancelled_work=dict(outcome='Cancelled',tokens_completed=0,tokens_committed=0,projections_completed=1,
            gathers_completed=1,convolution_steps_completed=0,recurrence_steps_completed=0,attention_heads_completed=0),
        independent_context_fidelity_qualified=False, quality_qualified=False, managed_adoption_qualified=False).items():
        need(name in report and prior.exact(report[name],value), 'Frozen native context contract differs: '+name)
    memory = report['memory']
    need(set(memory) == {'weights_and_model_bytes','recurrent_bank_bytes','convolution_bank_bytes','kv_bytes',
        'scratch_bytes','logits_bytes','history_bytes','metadata_bytes','metadata_allowance_bytes','admitted_bytes'},
        'Exact native owner inventory required')
    need(all(type(v) is int and v > 0 for v in memory.values()), 'Strict native owner counts required')
    need(2369040384 < memory['weights_and_model_bytes'] < 2369040384+1024**2, 'Native packed source ownership differs')
    need(sum(v for k,v in memory.items() if k != 'admitted_bytes') == memory['admitted_bytes'] <= 6*1024**3-128*1024**2,
         'Native admission/diagnostic reservation differs')
    for name, count in dict(recurrent_bank_bytes=150994944, convolution_bank_bytes=7077888,
        kv_bytes=33554432,scratch_bytes=504320,logits_bytes=63569920,history_bytes=2048,
        metadata_allowance_bytes=64*1024**2).items(): need(memory[name] == count, 'Native memory layout differs: '+name)
    for name in ['load_ns','bind_ns','observed_forward_ns']:
        need(type(report[name]) is int and report[name] > 0, 'Completed native phase timing required')
    need(type(report['partition_forward_ns']) is list and len(report['partition_forward_ns']) == 3
         and all(type(v) is int and v > 0 for v in report['partition_forward_ns']), 'Partition timing coverage differs')
    partitions = labels('QWEN35_CONTEXT_PARTITION')
    need(len(partitions) == 3, 'Partition proof coverage differs')
    for row, chunk in zip(partitions,[17,1,3],strict=True):
        need(set(row) == {'chunk','snapshot','all_logits_bits_exact','warmed_allocations'}
             and prior.exact(row['chunk'],chunk) and row['all_logits_bits_exact'] is True
             and type(row['warmed_allocations']) is int and row['warmed_allocations'] == 0
             and type(row['snapshot']) is str and re.fullmatch('[0-9a-f]{64}',row['snapshot']), 'Partition bit/allocation proof differs')
    need(len({v['snapshot'] for v in partitions}) == 1, 'Committed state bits differ across partitions')
    raw = prior.read(folder/'capture/rows.f32le',128*1024**2)
    need(len(raw) == values*4, 'Complete native context body required')
    expected = prior.expected(17); rows = {}; offset = 0
    arrays = labels('QWEN35_CONTEXT_ARRAY'); need(len(arrays) == 1051, 'Native stage inventory differs')
    for item in arrays:
        need(set(item) == {'token','position','layer','stage','head','count','offset','bytes','sha256'}, 'Exact native descriptor fields required')
        for field in ['token','position','count','offset','bytes']: need(type(item[field]) is int, 'Typed native extents required')
        for field in ['layer','head']: need(item[field] is None or type(item[field]) is int, 'Typed native owner/head required')
        need(type(item['stage']) is str and item['token'] == 16 and item['position'] == 53, 'Frozen native observation position differs')
        key = item['layer'],item['stage'],item['head']
        need(key in expected and key not in rows and item['count'] == expected[key]
             and item['offset'] == offset and item['bytes'] == item['count']*4, 'Complete native array coverage differs')
        block = memoryview(raw)[offset:offset+item['bytes']]
        need(len(block) == item['bytes'] and hashlib.sha256(block).hexdigest() == item['sha256'], 'Native array hash differs')
        data = np.frombuffer(block,dtype='<f4'); need(np.isfinite(data).all(), 'Nonfinite native array')
        rows[key] = data; offset += len(block)
    need(offset == len(raw), 'Trailing native bytes')
    need(type(report['greedy_lowest_id']) is int and int(rows[None,'logits',None].argmax()) == report['greedy_lowest_id'], 'Native lowest-ID greedy differs')
    return rows, peak, hashlib.sha256(raw).hexdigest()


def check(native_folder, reference_folder):
    need(np.__version__ == '2.3.2', 'Pinned NumPy required')
    actual, native_peak, body_sha = native(native_folder)
    expected, reference_peak = prior.reference(reference_folder,'context17')
    reports = []; greedy = []
    for key, data in actual.items():
        for profile, arrays in expected.items():
            want = arrays[key]
            reports.append(dict(profile=profile,layer=key[0],stage=key[1],head=key[2],**prior.statistics(data,want)))
            if key == (None,'logits',None): greedy.append(dict(profile=profile,native_id=int(data.argmax()),reference_id=int(want.argmax())))
    mismatches = [v for v in greedy if v['native_id'] != v['reference_id']]
    return dict(context17_coupled_fidelity_passed=not mismatches and all(v['failed_values'] == 0 for v in reports),
        compared_values=sum(v['values'] for v in reports),arrays=len(reports),rows=reports,greedy=greedy,
        unresolved_greedy_mismatches=mismatches,gate=producer.graph.GATE,native_capture_sha256=body_sha,
        native_peak_bytes=native_peak,independent_peak_bytes=reference_peak,
        longer_context_qualified=False,quality_qualified=False,managed_adoption_qualified=False,
        scope='Complete final-input stages/state/logits after17rawinputs/offset37, bit partition/reset/recovery and completed work only; no128/512/deepercontext,GPU,codec,quality,performance or adoption qualification')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native',type=Path);parser.add_argument('reference',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=check(args.native,args.reference)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','greedy']}))
    raise SystemExit(0 if result['context17_coupled_fidelity_passed'] else 1)
