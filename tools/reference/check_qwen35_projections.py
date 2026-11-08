"""Fail-closed all-output FP32 Metal projection comparison; no model adoption."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import check_qwen35_real_reference as prior
import qwen35_projection_reference as producer

need, exact, document, read = prior.need, prior.exact, prior.document, prior.read


def closed(folder, independent):
    receipt = document(folder/'checks.json')
    need(receipt['collection_passed' if independent else 'passed'] is True
         and type(receipt['exit_code']) is int and receipt['exit_code'] == 0
         and receipt['error'] is None and receipt['group_absent'] is True
         and receipt['changed_sources' if independent else 'changed_inputs'] == [], 'Closed successful collection required')
    peak = receipt['lifetime_peak_physical_bytes']
    need(type(peak) is int and 0 < peak <= 6*1024**3, 'Measured physical memory gate required')
    wanted = {'console.log','time-and-errors.log','inputs.json', 'capture/rows.f64le','capture/operands.f32le','capture/result.json','runtime.json','native.json'} if independent else {'console.log','time-and-errors.log','inputs.json','capture/rows.f32le'}
    need(set(receipt['files_sha256']) == wanted, 'Exact sealed collection inventory required')
    for name, digest in receipt['files_sha256'].items():
        need(hashlib.sha256(read(folder/name,producer.MAX_CAPTURE)).hexdigest() == digest, 'Sealed collection changed')
    peaks = re.findall(r'^\s*(\d+)\s+peak memory footprint\s*$',read(folder/'time-and-errors.log',2*1024**2).decode(),re.M)
    need(len(peaks) == 1 and int(peaks[0]) == peak, 'Physical peak must match sealed raw instrument output')
    return receipt


def reference(folder, context):
    receipt = closed(folder, True)
    result = document(folder/'capture/result.json')
    need(result['completed'] is True and result['primary_error'] is None and result['endpoint_errors'] == [], 'Complete independent calculation required')
    for name, value in dict(format='toklet-qwen35-projection-reference-v1',source_sha256=producer.pin.FILE_SHA,
        input_capture_sha256=producer.INPUT_SHA,gate=producer.graph.GATE,batches=[1,17],scalars=[1,-1,0.5],
        values=producer.VALUES,gpu_graph_qualified=False,quality_qualified=False,managed_adoption_qualified=False).items():
        need(exact(result[name],value), 'Frozen reference field differs: '+name)
    source = result['source']
    need(source['closed'] is True and source['poisoned'] is False and source['poison_reason'] is None
         and source['sha256'] == result['source_endpoint_sha256'] == producer.pin.FILE_SHA
         and type(source['bytes']) is int and source['bytes'] == producer.pin.FILE_BYTES
         and type(source['read_bytes']) is int and source['read_bytes'] >= producer.pin.FILE_BYTES,
         'Authenticated source closure required')
    inputs = document(folder/'inputs.json')
    freeze_raw = read(folder.parent/'adapter-freeze.json',2*1024**2)
    freeze = prior.decode(freeze_raw)
    need(inputs['freeze_sha256'] == hashlib.sha256(freeze_raw).hexdigest()
         and freeze['frozen_before_real_reference'] is True and exact(freeze['gate'],producer.graph.GATE)
         and freeze['contract_sha256'] == hashlib.sha256(read(folder.parent/'acceptance-contract.json',2*1024**2)).hexdigest(), 'Pre-execution freeze differs')
    runtime, native = document(folder/'runtime.json'), document(folder/'native.json')
    selected = {p:h for p,h in freeze['files'].items() if not p.endswith(('/reference_runtime.py','/test_qwen35_projection_reference.py'))}
    need(len(selected) == len(inputs['source_pins']) == 7, 'Exact numerical source closure required')
    for relative,digest in selected.items():
        names = [p for p in inputs['source_pins'] if Path(p).parts[-len(Path(relative).parts):] == Path(relative).parts]
        need(len(names) == 1 and inputs['source_pins'][names[0]] == runtime['files'][names[0]]['sha256'] == digest, 'Frozen producer source differs')
    for name in ['runtime','native']:
        need(inputs[name+'_manifest_sha256'] == hashlib.sha256(read(folder/(name+'.json'),2*1024**2)).hexdigest(), 'Runtime manifest seal differs')
    need(result['runtime_admission']['admitted']['manifest_sha256'] == inputs['runtime_manifest_sha256']
         and result['runtime_admission']['native_manifest_sha256'] == inputs['native_manifest_sha256'], 'Runtime admission differs')
    allowed = {**runtime['files'],**native['files']}; endpoint = result['runtime_endpoint']
    need(bool(endpoint['numpy_modules']) and bool(endpoint['non_system_native_images']), 'Loaded runtime endpoint required')
    for name,pin in endpoint['non_system_native_images'].items(): need(name in allowed and exact(pin,allowed[name]), 'Loaded native file differs')
    need(all(name in runtime['files'] for name in endpoint['numpy_modules'].values()), 'Loaded NumPy origin differs')
    raw,descriptors = producer.operands(context)
    need(exact(result['operands'],descriptors) and read(folder/'capture/operands.f32le',8*1024**2) == raw
         and hashlib.sha256(raw).hexdigest() == result['operands_sha256'], 'Frozen operand derivation differs')
    body = read(folder/'capture/rows.f64le',producer.MAX_CAPTURE)
    need(len(body) == producer.VALUES*8 and hashlib.sha256(body).hexdigest() == result['rows_sha256'], 'Complete independent output required')
    return receipt,result,body


def arrays(items, body, width, descriptors, native=False):
    import numpy as np
    need(type(items) is list and len(items) == 498, 'Every projection/batch required')
    result = {}; offset = 0
    for index,item in enumerate(items):
        desc = descriptors[index%249]; batch = [1,17][index//249]
        fields = {'name','batch','rows','columns','offset','bytes','sha256'}
        if native: fields |= {'warmed_rust_allocations','preflight_transactional'}
        need(set(item) == fields, 'Exact projection descriptor required')
        for name,value in dict(name=desc['name'],batch=batch,rows=desc['rows'],columns=desc['columns'],offset=offset,bytes=batch*desc['rows']*width).items():
            need(exact(item[name],value), 'Projection identity/shape/coverage differs: '+name)
        if native:
            need(exact(item['warmed_rust_allocations'],0) and item['preflight_transactional'] is True, 'Warm allocation/transactional proof differs')
        block = memoryview(body)[offset:offset+item['bytes']]
        need(len(block) == item['bytes'] and hashlib.sha256(block).hexdigest() == item['sha256'], 'Projection array seal differs')
        data = np.frombuffer(block,dtype='<f4' if width == 4 else '<f8')
        need(np.isfinite(data).all(), 'Finite projection array required')
        result[(batch,desc['name'])] = data
        offset += item['bytes']
    need(offset == len(body), 'No trailing projection body allowed')
    return result


def samples(rows):
    need(type(rows) is list and len(rows) == 52, 'Two complete paired cohorts required')
    seen = set(); groups = {}
    for index,row in enumerate(rows):
        need(set(row) == {'cohort','iteration','warmup','engine','elapsed_ns','projections'}, 'Exact sequence sample required')
        need(type(row['cohort']) is int and row['cohort'] in [0,1]
             and type(row['iteration']) is int and 0 <= row['iteration'] < 13
             and type(row['elapsed_ns']) is int and row['elapsed_ns'] > 0
             and exact(row['projections'],249) and row['engine'] in ['scalar_cpu','metal_fp32']
             and exact(row['warmup'],row['iteration'] < 3), 'Sample type/work/scope differs')
        key = row['cohort'],row['iteration'],row['engine']
        cohort, within = divmod(index,26)
        iteration, slot = divmod(within,2)
        engine = 'metal_fp32' if (slot+iteration+cohort)%2 == 1 else 'scalar_cpu'
        need(key == (cohort,iteration,engine), 'Frozen alternating sample order differs')
        need(key not in seen, 'Duplicate sequence sample'); seen.add(key)
        if not row['warmup']: groups.setdefault((row['cohort'],row['engine']),[]).append(row['elapsed_ns'])
    summaries = []
    for cohort in [0,1]:
        entries = {}
        for engine in ['scalar_cpu','metal_fp32']:
            data = groups.get((cohort,engine),[])
            need(len(data) == 10, 'Ten complete measured sequences required')
            median = statistics.median(data)
            entries[engine] = dict(samples_ns=data,median_ns=median,mad_ns=statistics.median([abs(v-median) for v in data]),minimum_ns=min(data),maximum_ns=max(data))
        improvement = 1 - entries['metal_fp32']['median_ns']/entries['scalar_cpu']['median_ns']
        summaries.append(dict(cohort=cohort,engines=entries,improvement_fraction=improvement))
    return summaries


def memory_ledger(report, owners, descriptors):
    """Reconcile fixed recipe ownership; the old footer lacks raw Vec capacity.

    Its residual must cover the exact operand bytes and remain within the
    diagnostic's 8 MiB input bound. This is a bounded residual, not an observed
    raw-capacity field. Other caller/scratch bytes reconcile exactly.
    """
    caller = 17*4*sum(d['rows']+d['columns'] for d in descriptors)
    need(exact(report['caller_bytes'],caller), 'Caller storage must reconcile with frozen batch geometry')
    scratch = gpu = 0
    for owner,desc in zip(owners,descriptors,strict=True):
        rows,columns = desc['rows'],desc['columns']
        # Three GPU buffers (+7 guard words each), full CPU transaction staging,
        # child CPU readback, and the pinned arm64 Projection struct (240B).
        expected = 4*(17*columns + 2*17*min(rows,16384) + 17*rows + 22) + 240
        need(exact(owner['scratch_bytes'],expected), 'Projection scratch must reconcile with frozen geometry')
        packed = rows*columns//2 + 2*(rows*columns//64)*2
        need(packed <= owner['gpu_weight_bytes'] <= packed + 16384*math.ceil(rows/16384), 'GPU owner differs from original packed representation')
        scratch += expected; gpu += owner['gpu_weight_bytes']
    need(exact(report['scratch_bytes'],scratch) and exact(report['packed_gpu_bytes'],gpu), 'Aggregate native owners must reconcile')
    tensors = overlap = 0
    for kind,shape in producer.graph.specs(producer.weights.GEOMETRY).values():
        count = math.prod(shape)
        if kind == 'matrix': tensors += count//2 + 2*(count//64)*2
        else:
            tensors += count*4; overlap = max(overlap,count*2)
    # Same independently derived pinned CPU recipe: retained tensors, cold
    # metadata allowance, streaming block and largest small BF16 conversion.
    cpu = tensors + 64*1024**2 + 64*1024 + overlap
    base = cpu + 64*1024**2 + 256*1024**2 + caller + scratch + gpu
    raw_minimum = 4*sum(d['columns'] for d in descriptors)
    admitted = report['admitted_bytes']
    need(type(admitted) is int and base+raw_minimum <= admitted <= min(base+8*1024**2,6*1024**3), 'Admission must reconcile with every native owner and bounded raw-input capacity')
    return dict(cpu_load_bytes=cpu,context_allowance_bytes=64*1024**2,diagnostic_allowance_bytes=256*1024**2,
        caller_bytes=caller,scratch_bytes=scratch,packed_gpu_bytes=gpu,raw_payload_bytes=raw_minimum,
        raw_capacity_bounded_residual_bytes=admitted-base,raw_capacity_upper_bound_bytes=8*1024**2,
        admitted_bytes=admitted)


def check(native, independent, context):
    receipt = closed(native,False)
    reference_receipt,result,body = reference(independent,context)
    expected = arrays(result['arrays'],body,8,result['operands'])
    log = read(native/'console.log',2*1024**2).decode()
    labels = lambda tag: [prior.decode(line.split(' ',1)[1]) for line in log.splitlines() if line.startswith(tag+' ')]
    reports = labels('QWEN35_PROJECTION_REPORT')
    need(len(reports) == 1, 'One complete native report required')
    report = reports[0]
    need(exact(report,receipt['report']), 'Receipt and raw footer reports must agree exactly')
    for name,value in dict(source_sha256=producer.pin.FILE_SHA,input_sha256=result['operands_sha256'],matrices=249,batches=[1,17],capture_bytes=producer.VALUES*4,warmed_rust_allocations=0,gpu_graph_qualified=False,quality_qualified=False,managed_adoption_qualified=False).items():
        need(exact(report[name],value), 'Native report contract differs: '+name)
    for name in ['admitted_bytes','packed_gpu_bytes','scratch_bytes','caller_bytes','load_ns','setup_ns']:
        need(type(report[name]) is int and report[name] > 0, 'Positive native owner/timing count required')
    need(report['admitted_bytes'] <= 6*1024**3, 'Native owner admission exceeds gate')
    owners = labels('QWEN35_PROJECTION_OWNER')
    need(len(owners) == 249, 'Every native projection owner required')
    for owner,desc in zip(owners,result['operands'],strict=True):
        need(set(owner) == {'name','rows','columns','gpu_weight_bytes','scratch_bytes'}
             and all(exact(owner[name],desc[name]) for name in ['name','rows','columns'])
             and all(type(owner[name]) is int and owner[name] > 0 for name in ['gpu_weight_bytes','scratch_bytes']), 'Native owner inventory differs')
    need(sum(v['gpu_weight_bytes'] for v in owners) == report['packed_gpu_bytes']
         and sum(v['scratch_bytes'] for v in owners) == report['scratch_bytes'], 'Native owner accounting differs')
    memory = memory_ledger(report,owners,result['operands'])
    inputs = document(native/'inputs.json')
    need(inputs['operand_sha256'] == result['operands_sha256']
         and inputs['contract_sha256'] == hashlib.sha256(read(independent.parent/'acceptance-contract.json',2*1024**2)).hexdigest(), 'Native workload pin differs')
    actual = arrays(labels('QWEN35_PROJECTION_ARRAY'),read(native/'capture/rows.f32le',producer.MAX_CAPTURE),4,result['operands'],True)
    rows = [dict(batch=k[0],name=k[1],**prior.statistics(actual[k],expected[k])) for k in expected]
    timings = samples(labels('QWEN35_PROJECTION_SAMPLE'))
    numerical = all(row['failed_values'] == 0 for row in rows)
    environment = receipt['environment']
    need(set(environment) == {'start','end'}, 'Both environment endpoints required')
    for name in ['hardware','os','configuration']:
        need(type(environment['start'][name]) is str and bool(environment['start'][name])
             and environment['start'][name] == environment['end'][name], 'Hardware/power configuration drift')
    power = all(type(v['power']) is str and v['power'].splitlines()[0] == "Now drawing from 'AC Power'" for v in environment.values())
    thermal = all(type(v['thermal']) is str and v['thermal'].splitlines() == [
        'Note: No thermal warning level has been recorded',
        'Note: No performance warning level has been recorded',
        'Note: No CPU power status has been recorded'] for v in environment.values())
    performance = power and thermal and all(row['improvement_fraction'] >= 0.05 for row in timings)
    return dict(projection_fidelity_passed=numerical,compared_values=sum(row['values'] for row in rows),failed_values=sum(row['failed_values'] for row in rows),maximum_gate_ratio=max(row['maximum_gate_ratio'] for row in rows),rows=rows,timings=timings,
        performance_threshold_passed=performance,retain_projection_candidate=numerical and performance,environment=environment,memory_ledger=memory,
        gate=producer.graph.GATE,native_peak_bytes=receipt['lifetime_peak_physical_bytes'],independent_peak_bytes=reference_receipt['lifetime_peak_physical_bytes'],
        gpu_graph_qualified=False,quality_qualified=False,managed_adoption_qualified=False,
        scope='249 isolated source projections, batches1/17; paired operator sequence only. No model TTFT, full GPU graph, sustained or adoption qualification.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['native','independent','context']: parser.add_argument(name,type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); result = check(args.native,args.independent,args.context)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','timings']}))
    raise SystemExit(0 if result['projection_fidelity_passed'] else 1)
