"""Independent full-head source, allocation, staging and complete-result replay.

Weight integrity uses whole-buffer digests reconstructed from original packed
bytes; all active device outputs and transactional callers are retained plaintext.
"""
from pathlib import Path
import contextlib
import copy
import hashlib
import json
import math
import re
import struct
import sys
import check_gemma_real_projections as base

N, K, B, TILE = 262144, 1536, 4, 16384
CHILD_OWNED, OWNED = 14442908, 235543088
TEST = 'metal::gemma_affine::partitioned::tests::head::real_tied_head_resident_reuse'
SENTINEL = struct.pack('<I', 0x5a170031)
GUARDS = SENTINEL * 7
same = base.same

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def identity(raw):
    if len(raw) % 4 or len(raw) < 28:
        raise ValueError('complete word buffer required')
    return dict(words=len(raw)//4, bytes=len(raw), sha256=sha(raw),
                guards=list(struct.unpack('<7I', raw[-28:])))

def policy(reference=4218880):
    if type(reference) is not int or not 4218880 <= reference <= 8 * 1024**2:
        raise ValueError('bounded actual reference allocation required')
    categories = dict(resident_owner=OWNED, context_policy=67108864, source_cpu_loading_overlap=226492416,
                      header_bytes=318228, frozen_binary=reference, frozen_json_and_manifest_caps=131072,
                      parsed_header_metadata_policy=4194304, parsed_reference_metadata_policy=1048576,
                      input_f32=24576, expected_f32=4194304, scalar_f32=4194304, caller_f32=4194304, baseline_f32=4194304,
                      caller_raw_conversion=4194304, child_raw_conversion=262144, hash_read_workspace=65536,
                      record_serialization_policy=4194304, record_metadata_policy=4194304)
    total = sum(categories.values())
    if total > 1024**3:
        raise ValueError('aggregate workload exceeds process policy')
    return dict(categories=categories, total_policy_bytes=total, native_process_allowance_bytes=1024**3,
                remainder_is_free_device_memory=False)

class Assets:
    def __init__(self, model, frozen):
        if model.stat().st_size != base.MODEL_BYTES or base.digest(model, base.MODEL_BYTES) != base.MODEL_SHA:
            raise ValueError('original checkpoint differs')
        base.load(frozen/'observation.json', 65536, '1c03155565a4a0ef5d0b71ce6c5d0600b3061f85c6c8275561530c9897f08579')
        base.load(frozen/'manifest.json', 65536, '40a56af7e34f975142cdb0fd8a89e8683846ebd9f29189a4fa4117f34062aeed')
        if base.digest(frozen/'arrays.bin', 4218880) != 'ba0e853cbf15459e3774afdac233237c68f36560395eb85c2ccb549409a7742c':
            raise ValueError('complete independent head reference differs')
        self.handles = contextlib.ExitStack()
        try:
            source = self.handles.enter_context(model.open('rb'))
            reference = self.handles.enter_context((frozen/'arrays.bin').open('rb'))
            self.input = reference.read(24576)
            self.expected = reference.read(4194304)
            if len(self.input) != 24576 or len(self.expected) != 4194304 or reference.read(1):
                raise ValueError('complete reference arrays required')
            if source.read(8) != struct.pack('<Q', 318220):
                raise ValueError('original header length differs')
            header_raw = source.read(318220)
            if sha(header_raw) != '4d1819ededded19812059691c0f77cfec0fe10c9c89c720950f919bfc26544a7':
                raise ValueError('original header differs')
            header = base.strict_json(header_raw)
            self.spans = []
            self.buffers = [dict() for _ in range(16)]
            for role, suffix, dtype, shape, count, stride in [
                ('codes', 'weight', 'U32', [N, K//8], 201326592, K//2),
                ('scales', 'scales', 'BF16', [N, K//64], 12582912, K//32),
                ('biases', 'biases', 'BF16', [N, K//64], 12582912, K//32)]:
                name = 'language_model.model.embed_tokens.' + suffix
                entry = header[name]
                same(entry['shape'], shape)
                same(entry['dtype'], dtype)
                start, end = entry['data_offsets']
                if type(start) is not int or type(end) is not int or end-start != count or not 0 <= start < end <= base.MODEL_BYTES-318228:
                    raise ValueError('original span differs')
                absolute = start + 318228
                source.seek(absolute)
                plain = hashlib.sha256()
                for chunk in range(16):
                    remaining = TILE * stride
                    device = hashlib.sha256()
                    while remaining:
                        raw = source.read(min(65536, remaining))
                        if not raw:
                            raise ValueError('short original packed span')
                        device.update(raw)
                        plain.update(raw)
                        remaining -= len(raw)
                    device.update(GUARDS)
                    words = TILE*stride//4 + 7
                    self.buffers[chunk][role] = dict(words=words, bytes=words*4, sha256=device.hexdigest(), guards=[0x5a170031]*7)
                self.spans.append(dict(name=name, dtype=dtype, shape=shape, offset=absolute, bytes=count,
                                       sha256=plain.hexdigest(), capacity=count))
        except BaseException:
            self.close()
            raise

    def close(self):
        self.handles.close()

def parse(raw):
    if len(raw) > 32 * 1024**2:
        raise ValueError('bounded native log required')
    result, wrapper = [], []
    for line in raw.decode().splitlines():
        if len(line.encode()) > 4 * 1024**2:
            raise ValueError('bounded native record required')
        if 'GEMMA_HEAD_' in line:
            prefix, rest = line.split('GEMMA_HEAD_', 1)
            if prefix not in ('', 'test ' + TEST + ' ... '):
                raise ValueError('unexpected native prefix')
            label, text = rest.split(' ', 1)
            result.append((label, base.strict_json(text)))
        elif line.strip():
            wrapper.append(line)
    if len(wrapper) != 3 or wrapper[:2] != ['running 1 test', 'ok'] or not re.fullmatch(
            r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s', wrapper[2]):
        raise ValueError('complete exact successful test required')
    return result

def verify(rows, binary, assets, *, overrides=None):
    events = iter(rows)
    def take(label, expected=None):
        try:
            actual_label, item = next(events)
        except StopIteration as error:
            raise ValueError('missing ' + label) from error
        if actual_label != label:
            raise ValueError('record order differs: '+actual_label+'/'+label)
        if expected is not None:
            same(item, expected)
        return item
    take('PLAN', policy())
    take('OLD_OWNER', dict(rejected=True, error='Gemma affine fixed geometry/format outside bounds', large_spans_read=False))
    meta = take('META')
    device, width = meta.get('device'), meta.get('simd_width')
    if type(device) is not str or not 1 <= len(device) <= 256 or type(width) is not int or not 1 <= width <= 65535:
        raise ValueError('named device and width required')
    actual_policy = policy(meta['policy']['categories']['frozen_binary'])
    same(meta, dict(device=device, simd_width=width, owner_struct_bytes=112, child_struct_bytes=240, policy=actual_policy,
                    source_spans=assets.spans, binary_cap_bytes=64*1024**2, whole_graph_qualified=False, native_activation_parity=False))
    if binary.stat().st_size != 37748736:
        raise ValueError('complete raw output file required')
    offset = 0
    count = 0
    def payload(stream, key, role, tokens, n):
        nonlocal offset, count
        entry = take('ARRAY_ATTEMPT')
        take('ARRAY', entry)
        size = tokens*n*4
        if not 0 < size <= B*N*4:
            raise ValueError('bounded full result required')
        stream.seek(offset)
        raw = stream.read(size)
        if overrides is not None and offset in overrides:
            raw = overrides[offset]
        if len(raw) != size or any(not math.isfinite(v) for (v,) in struct.iter_unpack('<f', raw)):
            raise ValueError('finite complete numerical output required')
        same(entry, dict(key=key, role=role, dtype='F32', shape=[tokens,n], offset=offset, bytes=size, sha256=sha(raw)))
        offset += size
        count += 1
        return raw, entry
    maximum = fraction = 0.
    comparisons = ordered_comparisons = 0
    def gate(actual, expected, contract):
        nonlocal maximum, fraction
        if len(actual) != len(expected):
            raise ValueError('complete numerical arrays required')
        for (a,), (b,) in zip(struct.iter_unpack('<f', actual), struct.iter_unpack('<f', expected), strict=True):
            error = abs(a-b)
            limit = 1e-5+1e-4*abs(b)
            if not math.isfinite(a) or error > limit:
                raise ValueError(contract+' numerical gate failed')
            maximum = max(maximum,error)
            fraction = max(fraction,error/limit)
    with binary.open('rb') as stream:
        ordered = b''
        for token in range(4):
            take('ORDERED_ATTEMPT', dict(token=token, contract='native ordered FP64 dot rounded FP32'))
            raw, entry = payload(stream, f'ordered/{token}', 'native-ordered-fp32', 1, N)
            take('ORDERED', dict(token=token, array=entry, contract='native ordered FP64 dot rounded FP32'))
            gate(raw, assets.expected[token*N*4:(token+1)*N*4], 'native ordered/mathematical')
            ordered_comparisons += N
            ordered += raw
        take('BUDGET', dict(required=OWNED, limit=OWNED-1, rejected=True, error=f'Metal admission requires {OWNED} bytes, budget {OWNED-1}'))
        take('CONSTRUCTOR_ATTEMPT', dict(rows=N, columns=K, maximum_rows=TILE, capacity=B, required=OWNED))
        take('CONSTRUCTOR_OUTCOME', dict(completed=True, error=None))
        cold = take('CONSTRUCTOR')
        state = dict(rows=N, columns=K, capacity=B, owned_bytes=OWNED, poisoned=False, owner_slots=16,
                     full_scratch_capacity=B*N, chunk_scratch_capacity=B*TILE,
                     staged_sha256=sha(bytes(B*N*4)), chunk_scratch_sha256=sha(bytes(B*TILE*4)), children=[])
        ids = set()
        for index in range(16):
            buffers = dict(assets.buffers[index], input=identity(bytes((B*K+7)*4)), output=identity(bytes((B*TILE+7)*4)), status=identity(bytes(32)))
            actual_device = cold['observation']['children'][index]['device']
            for name, buffer in buffers.items():
                allocation = actual_device[name]['allocation']
                if type(allocation) is not str or not re.fullmatch(r'0x[1-9a-f][0-9a-f]{0,15}', allocation) or allocation in ids:
                    raise ValueError('distinct live allocation identities required')
                ids.add(allocation)
                buffers[name] = dict(buffer, allocation=allocation)
            state['children'].append(dict(rows=TILE, owned_bytes=CHILD_OWNED, poisoned=False, pending=False, shared_read_safe=True, device=buffers))
        same(cold, dict(source_cpu_bytes_retained=0, observation=state))
        staged = bytearray(B*N*4)
        scratch = bytearray(B*TILE*4)
        baseline = None
        for phase,index,tokens in [('repeat',0,4),('repeat',1,4),*[('part',i,1) for i in range(4)],('resume',0,4)]:
            key=f'{phase}/{index}'
            active_input=assets.input if tokens==4 else assets.input[index*K*4:(index+1)*K*4]
            take('STEP_ATTEMPT',dict(key=key,phase=phase,index=index,tokens=tokens,input_sha256=sha(active_input)))
            outcome=take('STEP_OUTCOME')
            arrays=[]
            for child_index,child in enumerate(state['children']):
                raw,entry=payload(stream,key,f'child-{child_index}-observed-output',tokens,TILE)
                arrays.append(entry)
                for name,body in [('input',active_input+SENTINEL*((4-tokens)*K)+GUARDS),
                                  ('output',raw+SENTINEL*((4-tokens)*TILE)+GUARDS),('status',bytes(4)+GUARDS)]:
                    child['device'][name]=dict(identity(body),allocation=child['device'][name]['allocation'])
                scratch[:len(raw)]=raw
                first=child_index*TILE
                for token in range(tokens):
                    start=(token*N+first)*4
                    staged[start:start+TILE*4]=raw[token*TILE*4:(token+1)*TILE*4]
            raw,caller=payload(stream,key,'transactional-caller',tokens,N)
            if raw!=staged[:tokens*N*4]:
                raise ValueError('caller/device token-strided publication differs')
            state['staged_sha256']=sha(staged)
            state['chunk_scratch_sha256']=sha(scratch)
            same(outcome,dict(key=key,completed=True,error=None,caller_sha256=sha(raw),observation=state))
            take('STEP',dict(key=key,phase=phase,index=index,tokens=tokens,completed=True,error=None,caller=caller,
                             observed_child_outputs=arrays,observed_child_outputs_are_failed_request_results=False))
            math_raw=assets.expected if tokens==4 else assets.expected[index*N*4:(index+1)*N*4]
            scalar_raw=ordered if tokens==4 else ordered[index*N*4:(index+1)*N*4]
            gate(raw,math_raw,'GPU rounded mathematical')
            gate(raw,scalar_raw,'GPU native ordered')
            comparisons+=2*tokens*N
            if phase=='repeat' and index==0:
                baseline=raw
            elif phase=='part':
                if raw!=baseline[index*N*4:(index+1)*N*4]:
                    raise ValueError('single-token partition bits differ')
            elif raw!=baseline:
                raise ValueError('repeat or resume bits differ')
        take('FINAL',dict(steps=7,chunks=16,scalar_rows=4,raw_bytes=37748736,original_cpu_packed_table_retained=False,
                          all_chunks_resident=True,chunks_execute_sequentially=True,whole_graph_qualified=False,
                          native_activation_parity=False,performance_qualified=False))
        if offset!=37748736 or count!=123:
            raise ValueError('complete result inventory differs')
    try:
        next(events)
    except StopIteration:
        pass
    else:
        raise ValueError('trailing native record')
    return dict(passed=True,rows=N,columns=K,capacity=B,chunks=16,steps=7,arrays=123,raw_bytes=offset,
                native_ordered_math_comparisons=ordered_comparisons,gpu_contract_comparisons=comparisons,
                maximum_absolute_error=maximum,maximum_gate_fraction=fraction,owned_bytes=OWNED,
                workload_policy_bytes=actual_policy['total_policy_bytes'],device=device,simd_width=width,
                complete_active_device_outputs_retained=True,whole_gpu_weights_retained_plaintext=False,
                source_cpu_packed_table_retained=False,all_chunks_resident=True,chunks_execute_sequentially=True,
                whole_graph_qualified=False,native_activation_parity=False,performance_qualified=False)

if __name__=='__main__':
    if len(sys.argv)!=5:
        raise SystemExit('native stdout, native binary, original checkpoint, frozen head directory')
    stdout,binary,model,frozen=map(Path,sys.argv[1:])
    with stdout.open('rb') as source:
        raw=source.read(32*1024**2+1)
    assets=Assets(model,frozen)
    try:
        print(json.dumps(verify(parse(raw),binary,assets),allow_nan=False))
    finally:
        assets.close()
