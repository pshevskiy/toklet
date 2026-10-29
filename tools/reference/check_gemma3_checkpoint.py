#!/usr/bin/env python3
"""Independent saved-byte reader for the finite Gemma3 real capture, not quality adoption.

No model graph or native producer is imported. NumPy only decodes/reduces saved
arrays; tokenizers replays the frozen codec. Outer ownership authenticates the
executable, runtime, source-success receipts, process completion and memory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import mmap
from pathlib import Path
import sys
import numpy as np
import gemma3_conformance as local

V = 262144
# Frozen after source-v3 exact export/replay and independent saved-byte audit.
SOURCE_SHA = 'a62bb30d2fe5ed49bedecd07e4d801616afa31be6781e2f1a7dabae0922fa09d'
LEGACY_SOURCE_SHA = '60617b26647d2d4a6c96c41ac8675d978cfdec6dc251545ae308e52dd01c9033'
LEGACY_LIMITS_SHA = 'f40fa2275a9c9f193287f32f2764a3dbf9e4d8836a56706583df1d5555151bdd'
LEGACY_PROTOCOL_SHA = '8a28910d0452a36dcd8c0157a1f240a7ee49f9a23fab694dccb51181dff625a8'
LEGACY_CHECK_SHA = '808522aa2cd65017c45d18bac60232a2a31997ca56e47e07f9fe256cd38e6e2c'
SOURCE_ARRAY_SHA = '355683397397c40cc744e6ba32e69b42f9fa920f0a4376b3a90f7240bd5f9657'
LEGACY_ARRAY_SHA = '3385fff63c4260e8ceede9a03c9397d5bd5f4e48f10992c9651b0e519c595b81'
INPUT_SHA = '30c67871d9de7086d32e8c250416ba2134a4c65c8f54173f6cd17401aebba4ac'
PROTOCOL_SHA = '035bd42111d8e702a65a7442a805a3cffac6c04517c90a7cf95ccdc86e668c68'
TOKENIZER_SHA = '4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795'
PROFILE_CONTRACT = {'name': 'NativeHalf_certified_projection_serial_attention_v3', 'main': 'Original reconstructed FP32 weights and exactly Half inputs; certified binary64 projection stripes with exact dyadic direct-store fallback; FP32 serial Reference attention reductions/stores; existing FP32 Neumaier RMS/other explicit stores unchanged. Not GPU lane/SIMD emulation.', 'wide': 'same reconstructed FP32 weights widened per operation; FP64 BLAS and sums; FP32 scores/probabilities/logits stores and same Half boundaries', 'rms': 'main sequential column Neumaier FP32, vectorized only across independent rows; wide FP64 squares/reduction; stored FP32(1+rawBF16) scales', 'rotary': 'main FP32 NumPy power/angle/trig/separate products; wide Decimal80 frequency/angle then FP64 trig; Half store', 'gamma': 'selected card/0.26.3 converter no-fold provenance, raw BF16 +FP32 one once; not original-unquantized identity proof', 'embedding': 'original FP32 reconstruction selected row ->Half ->scale25.25 ->Half; tied head original FP32', 'gelu': 'explicit FP32 ufunc order main/FP64 wide thenHalf beforeHalf gating', 'residual': 'Both sites: validate finite Half operands, FP32 add, clamp to [-65504,65504], then Half store; identical explicit FP32 operation in MAIN and WIDE', 'projection_certificate': {'stripe_outputs': 1024, 'input_width_bound': 2048, 'token_bound': 544, 'products': 'finite binary16 times binary32 exact in binary64 (at most35 significand bits)', 'rounding_model': 'audited ordinary binary64 BLAS, round-to-nearest; gamma_n=nu/(1-nu),u=2^-53; n<=2048; no reduced-precision BLAS mode', 'absolute_error': 'outward rounded gamma_n * exact sumabs(Half input row) * maxabs(FP32 weight row)', 'certificate': 'outward dot interval endpoints must round to same finite Half/F32 destination bits', 'fallback': 'exact signed dyadic sum in units2^-173, direct nearest-ties-even destination word; no FP64 intermediary', 'finite': 'active inputs/coefficients/output finite; overflow errors; exact zero canonical+0, nonzero underflow sign retained', 'witnesses': 'first32 fallback exact integers/input+weight hashes, aggregate counts for all calls; no native target values'}, 'attention': 'MAIN explicit increasing-feature FP32 dot then FP32 sqrt/divide and score store, visible max, FP32 exp, increasing-key FP32 mass/divide and probability store, visible increasing-key FP32 product+sum thenHalf. CPU transcendentals are not claimed bit-identical to Metal. Raw masked scores retained; masked probabilities zero. WIDE unchanged.'}
POLICY_CONTRACT = {'version': 'legacy_fixed_limits_new_native_store_center_v1', 'fatal_reference': 'main', 'scope': 'each exact case/absolute selected row/stage and frozen teacher distribution', 'limits_sha256': 'f40fa2275a9c9f193287f32f2764a3dbf9e4d8836a56706583df1d5555151bdd', 'wide': 'diagnostic only; no native-WIDE/greedy/text gate', 'source_wide_identity': 'every current WIDE array must have the historical WIDE hash; this proves the unchanged independent path, not native acceptance'}
GATE_BASIS_CONTRACT = {'format': 'toklet-gemma3-legacy-numeric-limits-v1', 'file_pin': {'bytes': 441159, 'sha256': 'f40fa2275a9c9f193287f32f2764a3dbf9e4d8836a56706583df1d5555151bdd'}, 'source_manifest_pin': {'bytes': 2531015, 'sha256': '60617b26647d2d4a6c96c41ac8675d978cfdec6dc251545ae308e52dd01c9033'}, 'group_count': 3232, 'scope': 'same exactcase/row/stage names and teacher prefixes; MAIN center changes uniformly, literal old atol/rtol numbers do not change', 'comparison': 'abs(native - successor_MAIN) <= legacy_atol + legacy_rtol*abs(successor_MAIN)', 'new_source_envelope': 'diagnostic only; cannot update either literal numerical limit', 'history': 'v2 failure92 MAIN/98 WIDE remains rejected; successor must be reported as a distinct explicit profile correction'}
GATE_BASIS = "legacy_fixed_limits_new_native_store_center_v1"

LIMIT = 6 * 1024**3
APP = 512 * 1024**2
LAYER_STAGES = ('input_norm', 'q_projection', 'k_projection', 'v_projection',
 'q_norm', 'k_norm', 'q_rotary', 'k_rotary', 'attention_scores',
 'attention_probabilities', 'attention_context', 'attention_projection',
 'post_attention_norm', 'attention_residual', 'pre_ff_norm', 'gate_projection',
 'up_projection', 'gate_gelu', 'gated', 'down_projection', 'post_ff_norm', 'output')
STAGES = ['embedding_lookup', 'embedding'] + [f'layers.{layer}.{s}'
 for layer in range(18) for s in LAYER_STAGES] + ['final_norm', 'logits']
WIDTHS = dict(zip(LAYER_STAGES, [640,1024,256,256,1024,256,1024,256,
 None,None,1024,640,640,640,640,2048,2048,2048,2048,640,640,640]))
QK = {'q_projection','k_projection','v_projection','q_norm','k_norm','q_rotary','k_rotary'}
SCORE = {'attention_scores','attention_probabilities'}

class Failure(ValueError):
    pass

def need(condition, message):
    if not condition:
        raise Failure(message)

def integer(x, label):
    need(type(x) is int and x >= 0, f'{label}: unsigned integer required')
    return x

def sha(data):
    return hashlib.sha256(data).hexdigest()

def file_pin(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            h.update(block); size += len(block)
    return {'path':str(path), 'bytes':size, 'sha256':h.hexdigest()}

def strict_json(raw):
    def pairs(items):
        result={}
        for k,v in items:
            need(k not in result,'duplicate JSON key')
            result[k]=v
        return result
    def nonfinite(value):raise Failure('nonfinite JSON number: '+value)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=nonfinite)

def frozen_gate(main,wide,group,legacy_gate):
    need(main.shape==wide.shape and np.isfinite(main).all() and np.isfinite(wide).all(),'source pair shape/finite')
    envelope=float(np.max(np.abs(main.astype(np.float64)-wide.astype(np.float64))))
    need(group['envelope']==envelope,'successor source diagnostic envelope')
    need(group['gate_basis']==GATE_BASIS and group['envelope_scope']=='successor MAIN/WIDE diagnostic only; not a limit input','source gate basis')
    need(group['gate']==legacy_gate and set(legacy_gate)=={'atol','rtol','fatal_reference'},'literal legacy limits changed')
    need(math.isfinite(legacy_gate['atol']) and legacy_gate['atol']>=1e-4 and legacy_gate['rtol']==1e-3 and legacy_gate['fatal_reference']=='main','legacy finite MAIN limits')


def wide_identity(group,legacy):
    for profile in ('main','wide'):
        need(all(group[profile][key]==legacy[profile][key] for key in ('shape','dtype','bytes')),'historical source shape/store identity')
    need(group['wide']['sha256']==legacy['wide']['sha256'],'unchanged historical WIDE hash')


def source_metadata(source,input_,legacy,limits):
    need(source['format']=='toklet-mlx-gemma3-real-reference-v3' and source['source_reference_executed']is True and source['target_gpu_executed']is False,'frozen source execution/profile')
    need(source['profile']==PROFILE_CONTRACT and source['policy']==POLICY_CONTRACT and source['gate_basis']==GATE_BASIS_CONTRACT,'complete source arithmetic/limit policy')
    need(source['protocol_sha256']==PROTOCOL_SHA and source['input_sha256']==INPUT_SHA and source['cases']==input_['cases']==legacy['cases'],'source input/protocol binding')
    need(legacy['format']=='toklet-mlx-gemma3-real-reference-v2' and legacy['protocol_sha256']==LEGACY_PROTOCOL_SHA,'legacy source policy')
    need(limits['format']=='toklet-gemma3-legacy-numeric-limits-v1' and limits['source']['sha256']==LEGACY_SOURCE_SHA and limits['group_count']==3232,'fixed legacy gate-table basis')
    need(source['counts']==legacy['counts']==dict(graph_stage_groups=3200,teacher_stage_groups=32,total_arrays=6464),'fixed source inventory counts')


def capture_identity(start,mode,source_path,legacy_path,model_path,source_sha):
    need(mode in ('fresh-preregistered','historical-posthoc'),'explicit capture comparison mode')
    historical=mode=='historical-posthoc'
    expected_sha=LEGACY_SOURCE_SHA if historical else source_sha
    expected_protocol=LEGACY_PROTOCOL_SHA if historical else PROTOCOL_SHA
    expected_path=legacy_path if historical else source_path
    need(start['format']==('toklet-gemma3-real-native-v1'if historical else'toklet-gemma3-real-native-v2') and start['source_manifest_sha256']==expected_sha and start['input_sha256']==INPUT_SHA and start['protocol_sha256']==expected_protocol,'native initial source identity/mode')
    need(Path(start['source_manifest']).resolve()==Path(expected_path).resolve() and Path(start['model']).resolve()==Path(model_path).resolve(),'native consumed source/model paths')
    if not historical:
        need(start['source_profile']==PROFILE_CONTRACT['name'] and start['source_gate_basis']==GATE_BASIS and start['legacy_limits_sha256']==LEGACY_LIMITS_SHA and start['legacy_source_manifest_sha256']==LEGACY_SOURCE_SHA,'native preregistered successor metadata')
    return dict(retrospective_diagnostic=historical,numerical_qualification=False,
                numerical_qualification_eligible=False,
                qualification_scope='outer owned fresh-capture/source/runtime review still required'if not historical else'POSTHOC saved-v1 diagnosis only; old92-failure screen unchanged')


def probability_mask(values,layer,row,first):
    keys=values.shape[-1];positions=np.arange(first,first+keys)
    allowed=(positions<=row)&((positions>=max(0,row-511)) if layer%6!=5 else True)
    need(np.all(values[:,:,~allowed]==0),'actual masked probabilities nonzero')
    return int(np.count_nonzero(~allowed))*4

class Reader:
    def __init__(self):
        self.consumed = {}
        self.progress = {}
    def pin(self, path, cap, expected=None):
        path = Path(path).resolve()
        need(path.is_file() and path.stat().st_size <= cap, f'bounded file: {path}')
        pin = file_pin(path)
        if expected:
            for key in ('bytes','sha256'):
                if key in expected:
                    need(pin[key] == expected[key], f'file {key}: {path}')
        previous = self.consumed.setdefault(str(path), pin)
        need(previous == pin, f'changed input: {path}')
        return pin
    def doc(self, path, cap, expected=None):
        identity=self.pin(path,cap,expected);raw=Path(path).read_bytes()
        need(len(raw)==identity['bytes'] and sha(raw)==identity['sha256'],'consumed JSON differs from authenticated bytes')
        return strict_json(raw)
    def endpoints(self):
        for path,pin in self.consumed.items():
            need(file_pin(path)==pin, f'endpoint changed: {path}')

class Bytes:
    """Read-only mapping. Each decoded array owns a copy; no exported map view."""
    def __init__(self, path):
        self.file=Path(path).open('rb')
        self.map=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ)
    def close(self):
        self.map.close(); self.file.close()
    def array(self, d):
        shape=d['shape']
        need(isinstance(shape,list) and 1<=len(shape)<=3 and all(type(n)is int and n>0 for n in shape),'array shape')
        dtype=d['dtype'];need(dtype in ('u16le','f32le'),'array dtype')
        start=integer(d['offset'],'offset');length=integer(d['bytes'],'bytes')
        need(length==math.prod(shape)*(2 if dtype=='u16le' else 4),'descriptor byte/shape mismatch')
        need(start+length<=len(self.map),'descriptor outside raw file')
        raw=self.map[start:start+length]
        need(sha(raw)==d['sha256'],'descriptor raw digest')
        values=np.frombuffer(raw,dtype='<f2' if dtype=='u16le' else '<f4').astype(np.float32).reshape(shape)
        need(np.isfinite(values).all(),'nonfinite saved array')
        return values

def cover(descriptors, total):
    end=0
    for d in sorted(descriptors,key=lambda x:x['offset']):
        start=integer(d['offset'],'offset');length=integer(d['bytes'],'bytes')
        need(length>0 and start==end,'raw descriptors overlap or leave gap')
        end+=length
    need(end==total,'raw descriptor coverage')

def shape_for(stage, keys):
    if stage in ('embedding_lookup','embedding','final_norm'): return [1,640]
    if stage=='logits': return [1,V]
    layer=int(stage.split('.')[1]); leaf=stage.split('.')[2]
    need(0<=layer<18 and leaf in WIDTHS,'layer/stage inventory')
    if leaf in SCORE:return [4,1,keys]
    if leaf in QK:return [1,WIDTHS[leaf]//256,256]
    return [1,WIDTHS[leaf]]

def columns(stage,start,n):
    if stage.split('.')[-1] not in SCORE:return (0,0)
    layer=int(stage.split('.')[1])
    first=0 if layer%6==5 else max(0,start-511)
    return first,start+n-first

def source_slice(values, stage, row, start, n, record):
    first,keys=columns(stage,start,n)
    expected=shape_for(stage,keys)
    need(record['row']==row and record['absolute_key_start']==first,'wrong absolute query/key origin')
    need(record['shape']==expected,'wrong native stage shape')
    if stage.split('.')[-1] in SCORE:
        need(first+keys<=values.shape[2],'source absolute columns unavailable')
        return values[:,:,first:first+keys]
    need(list(values.shape)==expected,'source/native non-attention shape')
    return values

def numeric_hash(values):
    a=np.asarray(values,dtype='<f4').reshape(-1).copy()
    need(a.size==V and np.isfinite(a).all(),'finite vocabulary row required')
    a[a==0]=0
    return sha(a.tobytes())

def lowest_id(values):
    a=np.asarray(values).reshape(-1)
    need(a.size==V and np.isfinite(a).all(),'finite greedy row required')
    return int(np.argmax(a))

def compare(native, main, wide, gate):
    need(native.shape==main.shape==wide.shape,'comparison shape')
    need(all(np.isfinite(x).all() for x in (native,main,wide)),'comparison finite')
    a=native.astype(np.float64);m=main.astype(np.float64);w=wide.astype(np.float64)
    bound=gate['atol']+gate['rtol']*np.abs(m)
    wide_bound=gate['atol']+gate['rtol']*np.abs(w)
    error=np.abs(a-m);wide_error=np.abs(a-w)
    return dict(values=a.size,failures=int(np.count_nonzero(error>bound)),
        max_abs=float(error.max(initial=0)),max_fraction=float((error/bound).max(initial=0)),
        wide_exceedances=int(np.count_nonzero(wide_error>wide_bound)),
        wide_max_abs=float(wide_error.max(initial=0)),wide_max_fraction=float((wide_error/wide_bound).max(initial=0)),
        main_bit_differences=int(np.count_nonzero(native.view(np.uint32)!=main.copy().view(np.uint32))),
        wide_bit_differences=int(np.count_nonzero(native.view(np.uint32)!=wide.copy().view(np.uint32))))

def ledger(memory, model=None, diagnostic=None):
    fields=('packed_weights','effective_norms','zero_bias','cache','scratch','logits',
        'tokens_and_history','diagnostic_gpu','diagnostic_readback_reserved','metadata',
        'shared_framework_allowance','external_resident')
    need(set(memory)==set(fields)|{'admitted','cold_peak'},'memory ledger fields')
    for k,v in memory.items():integer(v,k)
    need(sum(memory[k] for k in fields)==memory['admitted']<=LIMIT,'memory admitted sum/cap')
    need(memory['admitted']<=memory['cold_peak']<=LIMIT,'accounted construction/admission bound')
    if model is not None:
        for key in ('packed_weights','effective_norms','zero_bias','shared_framework_allowance'):
            need(memory[key]==model[key],'model allocation ledger changed')
        need(memory['external_resident']==APP,'caller reserve')
        need(memory['cold_peak']==max(model['cold_peak'],memory['admitted']),'session accounted cold bound')
        expected_logits=8*((32 if diagnostic else 1)*V+7)
        need(memory['logits']==expected_logits,'two published/candidate logit owners')
        scratch=2*(4*(32*640+7)+2*(32*1024+7)+(32*256+7)+3*(32*2048+7))+8*(4*32*544+7)
        need(memory['scratch']==scratch,'fixed scratch byte formula')
        need(memory['tokens_and_history']>=4*(32+7+544),'tokens/history lower bound')
        need(memory['diagnostic_gpu']>0 and memory['diagnostic_readback_reserved']>=32*V*4 if diagnostic else memory['diagnostic_gpu']==memory['diagnostic_readback_reserved']==0,'diagnostic ownership')

class Events:
    def __init__(self,path,expected_sha=None):
        self.records=[];digest=hashlib.sha256()
        with Path(path).open('rb') as f:
            for line in f:
                need(len(line)<=1024**2,'journal event cap');digest.update(line)
                self.records.append(strict_json(line))
                need(len(self.records)<=20000,'journal record cap')
        if expected_sha is not None:need(digest.hexdigest()==expected_sha,'consumed native journal digest')
        self.at=0
    def take(self,kind):
        need(self.at<len(self.records),f'missing event {kind}')
        x=self.records[self.at];self.at+=1
        need(x.get('kind')==kind,f'event {self.at}: expected {kind}, got {x.get("kind")}')
        return x
    def end(self):need(self.at==len(self.records),'unconsumed journal records')

class Audit:
    def __init__(self,reader,source,input_,source_bytes,native_bytes,events,tokenizer):
        self.reader=reader;self.source=source;self.input=input_;self.sb=source_bytes;self.nb=native_bytes;self.e=events;self.tokenizer=tokenizer
        self.offset=0;self.records=0;self.stages=0;self.stage_values=0;self.commits=0;self.calls=0;self.masked=0
        self.numerical=dict(values=0,failures=0,max_abs=0.,max_fraction=0.,wide_exceedances=0,wide_max_abs=0.,wide_max_fraction=0.,main_bit_differences=0,wide_bit_differences=0)
        self.first_failure=None;self.source_groups=0;self.source_values=0
        self.coupled=dict(self.numerical);self.coupled_groups=0;self.local=None
        self.reader.progress={'numerical':self.numerical,'first_failure':None,'complete':False,'raw_records_checked':0}
    def source_audit(self):
        expected={}
        need(set(self.source['arrays'])==set(self.legacy['arrays'])==set(self.legacy_limits['gates']),'legacy exact group identities')
        for trial,old_trial in zip(self.source['greedy'],self.legacy['greedy'],strict=True):
            need(all(trial[k]==old_trial[k]for k in ('prompt','visible_ids','stop_reason')) and len(trial['steps'])==len(old_trial['steps']),'historical teacher identity')
            for step,old_step in zip(trial['steps'],old_trial['steps'],strict=True):
                need(all(step[k]==old_step[k]for k in ('prefix','wide_teacher_token','stop')),'historical fixed teacher prefix')
        for c in self.input['cases']:
            for row in c['selected_rows']:
                for stage in STAGES:expected[f'{c["name"]}.row_{row}.{stage}']=shape_for(stage,len(c['tokens']))
        for c,t in zip(self.input['cases'][:2],self.source['greedy'],strict=True):
            need(t['prompt']==c['name'],'source teacher prompt')
            prefix=list(c['tokens']);visible=[];steps=t['steps'];need(1<=len(steps)<=16,'source teacher bound')
            for index,step in enumerate(steps):
                need(step['prefix']==prefix,'source teacher input chain')
                token=integer(step['wide_teacher_token'],'teacher ID');need(token<V,'teacher in vocabulary')
                stop=token in (1,106);need(step['stop']==stop and (not stop or index+1==len(steps)),'source teacher stop')
                key=f'teacher.{c["name"]}.step_{index}.row_{len(prefix)-1}.logits';expected[key]=[1,V]
                group=self.source['arrays'][key]
                need(lowest_id(self.sb.array(group['wide']))==token,'saved wide teacher argmax')
                need(lowest_id(self.sb.array(group['main']))==step['main_greedy_diagnostic'],'saved main diagnostic argmax')
                if not stop:visible.append(token);prefix.append(token)
            need(t['visible_ids']==visible and t['stop_reason']==('stop_token' if stop else 'length'),'teacher visible/stop observation')
            need(stop or len(steps)==16,'teacher length cap')
        need(set(expected)==set(self.source['arrays']),'source exact stage inventory')
        descriptors=[]
        for name,shape in expected.items():
            g=self.source['arrays'][name];m=self.sb.array(g['main']);w=self.sb.array(g['wide'])
            need(g['main']['shape']==g['wide']['shape']==shape,'source semantic shape')
            dtype='f32le' if name.endswith(('attention_scores','attention_probabilities','logits')) else 'u16le'
            need(g['main']['dtype']==g['wide']['dtype']==dtype,'source store dtype')
            need(g['main']['file']==g['wide']['file']=='arrays.bin','source raw file')
            wide_identity(g,self.legacy['arrays'][name])
            need(self.legacy_limits['gates'][name]==self.legacy['arrays'][name]['gate'],'legacy table/source gate equality')
            frozen_gate(m,w,g,self.legacy_limits['gates'][name])
            descriptors.extend([g['main'],g['wide']]);self.source_groups+=1;self.source_values+=m.size
        cover(descriptors,len(self.sb.map));need(self.source_groups==3232 and self.source_values==13473568,'frozen3232 source groups/13473568 values per profile')
    def step(self,label,prefix,tokens):
        x=self.e.take('forward');need(x['label']==label and x['tokens']==tokens and x['before']==prefix and x['after']==prefix+tokens,'exact forward history')
        need(x['error'] is None and x['poisoned'] is False,'forward successful public result')
        need(isinstance(x['completed_wall_seconds'],(int,float)) and math.isfinite(x['completed_wall_seconds']) and x['completed_wall_seconds']>=0,'diagnostic wall interval')
        self.commits+=len(tokens);self.calls+=1
        return prefix+tokens
    def reset(self,label):
        x=self.e.take('reset');need(x['label']==label and x['history']==[] and x['poisoned'] is False and x['copy_logits_error']=='Gemma3 no logits/output shape','reset/publication contract')
    def capture(self,label,kind,row,stage,shape,key_start):
        start=self.e.take('capture_started');x=self.e.take('capture')
        fixed=dict(label=label,capture_kind=kind,row=row,stage=stage,shape=shape,absolute_key_start=key_start,offset=self.offset,bytes=math.prod(shape)*4)
        need(all(start.get(k)==v and x.get(k)==v for k,v in fixed.items()),'wrong case/row/stage/key/shape/extent')
        need(x['dtype']=='f32le' and x['finite'] is True,'native capture dtype/finite flag')
        a=self.nb.array(x);self.offset+=x['bytes'];self.records+=1
        self.reader.progress['raw_records_checked']=self.records
        need(self.offset<=256*1024**2 and self.records<=7000,'native capture bounds')
        return a,x
    def numerical_row(self,a,g,main,wide,identity,record):
        result=compare(a,main,wide,g['gate'])
        for k,v in result.items():
            if k.startswith('max_') or k.startswith('wide_max_'):self.numerical[k]=max(self.numerical[k],v)
            else:self.numerical[k]+=v
        if result['failures'] and self.first_failure is None:
            d=np.abs(a.astype(np.float64)-main.astype(np.float64));b=g['gate']['atol']+g['gate']['rtol']*np.abs(main.astype(np.float64));at=int(np.flatnonzero((d>b).reshape(-1))[0])
            self.first_failure=dict(identity=identity,index=at,native_bits=int(a.reshape(-1).view(np.uint32)[at]),main_bits=int(main.copy().reshape(-1).view(np.uint32)[at]),wide_bits=int(wide.copy().reshape(-1).view(np.uint32)[at]),raw_descriptor=record,main_source_descriptor=g['main'],wide_source_descriptor=g['wide'],gate=g['gate'])
            self.reader.progress['first_failure']=self.first_failure
        return result
    def fatal_coupled(self,result):
        self.coupled_groups+=1
        for key,value in result.items():
            if key.startswith('max_') or key.startswith('wide_max_'):self.coupled[key]=max(self.coupled[key],value)
            else:self.coupled[key]+=value
    def source_stage(self,label,case,row,stage,start,n,row_hashes):
        first,keys=columns(stage,start,n);shape=shape_for(stage,keys)
        a,record=self.capture(label,'source_stage',row,stage,shape,first)
        name=f'{case}.row_{row}.{stage}';g=self.source['arrays'][name]
        m=source_slice(self.sb.array(g['main']),stage,row,start,n,record);w=source_slice(self.sb.array(g['wide']),stage,row,start,n,record)
        if g['main']['dtype']=='u16le':need(np.array_equal(a,a.astype(np.float16).astype(np.float32)),'native Half storage widening')
        result=self.numerical_row(a,g,m,w,name,record);self.stages+=1;self.stage_values+=a.size
        if local.route(stage)=='coupled':self.fatal_coupled(result)
        self.local.observe(label,row,stage,a,record,g['gate'])
        if stage.endswith('attention_probabilities'):
            self.masked+=probability_mask(a,int(stage.split('.')[1]),row,first)
        if stage=='logits':need(numeric_hash(a)==row_hashes[row],'raw selected logits versus logical hash')
    def schedule(self,case,label,tokens,parts,selected,baseline=None,equal_until=0):
        x=self.e.take('schedule');need(x==dict(kind='schedule',label=label,source_case=label.split('.')[0],tokens=tokens,partitions=parts,selected_absolute_rows=selected,exact_numeric_prefix_rows=equal_until),'fixed schedule record')
        self.local.begin(label,tokens,selected)
        self.reset(label);start=0;hashes=[];prefix=[]
        for n in parts:
            prefix=self.step(label,prefix,tokens[start:start+n])
            for row in range(start,start+n):
                x=self.e.take('row');h=x['numeric_sha256'];need(x['label']==label and x['row']==row and x['vocabulary']==V and x['finite'] is True and isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h),'logical row record')
                if baseline is not None and row<equal_until:need(h==baseline[row],'exact numeric schedule/causal row hash')
                hashes.append(h)
            for stage in STAGES:
                for row in selected:
                    if start<=row<start+n:self.source_stage(label,case,row,stage,start,n,hashes)
            start+=n
        need(prefix==tokens and start==len(tokens),'schedule coverage')
        self.local.end()
        return hashes
    def rejection(self,label,request,history,published):
        x=self.e.take('rejection')
        need(x==dict(kind='rejection',label=label,request=request,before_history=history,after_history=history,before_published_row_hashes=published,after_published_row_hashes=published,published_rows=len(published),error='Gemma3 token/count/position admission',poisoned=False),'exact request rejection/publication preservation')
    def budgets(self,model,diagnostic):
        x=self.e.take('session_plan');need(x['diagnostics']==diagnostic,'session plan mode');ledger(x['memory'],model,diagnostic);required=x['memory']['admitted']
        under=self.e.take('session_budget_minus_one');need(under==dict(kind='session_budget_minus_one',diagnostics=diagnostic,required=required,limit=required-1,error=f'Metal admission requires {required} bytes, budget {required-1}'),'exact one-byte-under admission')
        exact=self.e.take('session_exact_budget');need(exact['diagnostics']==diagnostic and exact['memory']==x['memory'],'exact constructor ledger continuity')
    def codec(self):
        for c in self.input['cases'][:2]:
            x=self.e.take('codec_plain');ids=[2]+self.tokenizer.encode(c['text'],add_special_tokens=False).ids;decoded=self.tokenizer.decode(ids,skip_special_tokens=False)
            need(ids==c['tokens'] and decoded==c['decoded'] and ids.count(2)==1,'independent plain codec/single BOS')
            need(x==dict(kind='codec_plain',case=c['name'],encoded=ids,decoded=decoded,add_special_tokens=False,manual_bos=2),'native plain codec evidence')
        for c in self.codec_proof['chat_controls']:
            ids=self.tokenizer.encode(c['rendered'],add_special_tokens=False).ids;decoded=self.tokenizer.decode(ids,skip_special_tokens=False)
            need(ids==c['tokens'] and decoded==c['decoded'] and ids[0]==2 and ids.count(2)==1,'serialized template/BOS codec')
            x=self.e.take('codec_serialized_template');need(x==dict(kind='codec_serialized_template',rendered=c['rendered'],ids=ids,decoded=decoded,template_rendering_executed=False,skip_special_tokens=False),'native serialized codec observation')
    def teacher(self,c,t):
        name=c['name'];prefix=self.step(f'teacher.{name}.prefill',[],c['tokens'])
        for index,step in enumerate(t['steps']):
            need(prefix==step['prefix'],'native teacher prefix')
            label=f'teacher.{name}.step_{index}';row=len(prefix)-1;a,record=self.capture(label,'source_teacher',row,'logits',[1,V],0)
            key=f'teacher.{name}.step_{index}.row_{row}.logits';g=self.source['arrays'][key]
            self.fatal_coupled(self.numerical_row(a,g,self.sb.array(g['main']),self.sb.array(g['wide']),key,record))
            native=lowest_id(a);selected=step['wide_teacher_token'];x=self.e.take('teacher')
            need(x==dict(kind='teacher',prompt=name,step=index,prefix=prefix,wide_teacher_id=selected,native_greedy_diagnostic=native,agreement_diagnostic_only=native==selected),'teacher observation independent argmax')
            if index+1<len(t['steps']):prefix=self.step(f'teacher.{name}.append',prefix,[selected])
    def free(self,c,trial):
        name=c['name'];label=f'free.{name}.trial_{trial}';prefix=self.step(label,[],c['tokens']);sampled=[];visible=[];hashes=[];stopped=False
        for index in range(16):
            a,_=self.capture(f'{label}.step_{index}','native_free',len(prefix)-1,'logits',[1,V],0);token=lowest_id(a);need(self.tokenizer.id_to_token(token) is not None,'mapped sampled ID')
            hashes.append(numeric_hash(a));sampled.append(token);x=self.e.take('free_sample');need(x==dict(kind='free_sample',label=label,step=index,sampled_id=token,committed=prefix),'independent native free argmax/state')
            if token in (1,106):stopped=True;break
            visible.append(token)
            if index<15:prefix=self.step(label,prefix,[token])
        expected=dict(sampled_ids=sampled,visible_ids=visible,decoded_visible=self.tokenizer.decode(visible,skip_special_tokens=False),skip_special_tokens=False,stop_reason='stop_token' if stopped else 'length',generated_pending=0 if stopped else 1,committed=prefix,row_hashes=hashes)
        x=self.e.take('free_complete');need(x==dict(kind='free_complete',label=label,record=expected),'native free decode/pending/stop evidence')
        need(prefix==c['tokens']+visible[:len(visible)-(0 if stopped else 1)],'sampled versus committed generation history')
        return expected
    def run(self,summary):
        start=self.e.take('start');self.comparison_scope=capture_identity(start,self.capture_mode,self.source_path,self.legacy_path,self.model_path,SOURCE_SHA)
        need(start['caller_reserve_bytes']==APP and start['process_cap_bytes']==LIMIT and start['raw_cap_bytes']==256*1024**2 and start['maximum_records']==7000,'native fixed resource policy')
        need(start['repository']==self.source['model']['repository'] and start['revision']==self.source['model']['revision'],'native artifact identity')
        self.codec();x=self.e.take('model_loaded');model=x['memory'];ledger(model)
        need(x['capacity']==544 and x['declared_positions_not_admitted']==32768 and x['shared_tied_table'] is True and x['source_cpu_payload_dropped'] is True,'model ownership/scope record')
        need(model['external_resident']==0 and model['zero_bias']==2*(V+7) and model['effective_norms']==4*(55936+109*7),'model effective-norm/zero-bias byte equations')
        self.budgets(model,True);self.budgets(model,False)
        x=self.e.take('diagnostic_plan');ledger(x['memory'],model,True);need(x['host_readback_max_bytes']==32*V*4 and x['extra_caller_allowance']==APP,'readback and extra-caller ledger')
        for c in self.input['cases'][:2]:
            ids=c['tokens'];name=c['name'];selected=c['selected_rows'];base=self.schedule(name,f'{name}.whole',ids,[len(ids)],selected)
            second=self.schedule(name,f'{name}.tokenwise',ids,[1]*len(ids),selected,base,len(ids))
            for label,request in [('empty_request',[]),('outside_head_image_token',[262144]),('chunk33',[2]*33)]:self.rejection(label,request,ids,second[-1:])
        c=self.input['cases'][2];ids=c['tokens'];selected=c['selected_rows'];parts=[32]*17
        base=self.schedule('window544','window544.partition_a',ids,parts,selected)
        self.rejection('capacity545',[2],ids,base[-32:])
        self.schedule('window544','window544.partition_b',ids,[31]*16+[15,1,1,1,30],selected,base,544)
        changed=ids[:-1]+[self.input['native_future_last_token']];need(changed[-1]!=ids[-1],'future-tail is distinct')
        self.schedule('window544','window544.future_tail',changed,parts,[],base,543);self.reset('after_window_replay')
        x=self.e.take('ordinary_plan');ledger(x['memory'],model,False);prefix=[]
        for offset in range(0,544,32):
            prefix=self.step('window544.ordinary_last',prefix,ids[offset:offset+32]);x=self.e.take('ordinary_last');need(x==dict(kind='ordinary_last',row=offset+31,numeric_sha256=base[offset+31]),'ordinary/diagnostic last-row equality')
        for case,t in zip(self.input['cases'][:2],self.source['greedy'],strict=True):self.teacher(case,t)
        for case,t in zip(self.input['cases'][:2],self.source['greedy'],strict=True):
            a=self.free(case,0);b=self.free(case,1);need(a==b,'native free repeat exact numeric/text/state')
            x=self.e.take('free_replay');need(x==dict(kind='free_replay',prompt=case['name'],exact_native_replay=True,source_teacher_visible_ids=t['visible_ids'],source_agreement_diagnostic_only=a['visible_ids']==t['visible_ids']),'free replay/source-observation record')
        x=self.e.take('complete');need(x['summary']==summary,'summary file/journal agreement');self.e.end()
        need(summary['capture_completed'] is True and summary['numerical_qualification'] is False and summary['independent_checker_required'] is True and summary['source_reference_numerical_arrays_read'] is False,'capture-versus-qualification scope')
        need(self.stages==summary['source_stage_records']==6400 and self.stage_values==10148680,'exact stage inventory/values')
        need(self.records==summary['raw_records'] and self.offset==summary['raw_bytes']==len(self.nb.map),'native raw closure')
        need(self.commits==summary['native_committed_tokens']<=2351 and summary['maximum_committed_tokens']==2351 and self.calls==summary['forward_calls'],'finite native work')
        need(self.numerical['values']==self.stage_values+sum(len(t['steps']) for t in self.source['greedy'])*V,'compared stage+teacher count')
        need(self.coupled_groups==624 and self.coupled['values']==13185444,'exact fatal coupled inventory')
        need(self.local.groups==5808 and self.local.values==5351844,'exact exhaustive local inventory')
        need(self.coupled['values']+self.local.values==self.numerical['values'],'no skipped compared values')
        self.reader.progress['complete']=True
        return dict(stage_groups=self.stages,stage_values=self.stage_values,source_groups=self.source_groups,source_values_per_profile=self.source_values,raw_records=self.records,raw_bytes=self.offset,native_committed_tokens=self.commits,forward_calls=self.calls,masked_probability_zeros=self.masked,numerical=self.numerical,first_failure=self.first_failure,
                    coupled_fatal=dict(groups=self.coupled_groups,**self.coupled),own_input=self.local.summary())

def execute(args,reader):
    need(isinstance(SOURCE_SHA,str) and len(SOURCE_SHA)==64 and isinstance(SOURCE_ARRAY_SHA,str) and len(SOURCE_ARRAY_SHA)==64,'checker must be frozen to completed source-v3 output')
    need(np.__version__=='2.3.2','frozen NumPy2.3.2 reader runtime')
    reader.pin(args.conformance_policy,128*1024,{'sha256':local.POLICY_SHA})
    source_path=Path(args.source).resolve();native=Path(args.native).resolve();inputs_path=Path(args.inputs).resolve()
    source=reader.doc(source_path,16*1024**2,{'sha256':SOURCE_SHA})
    input_=reader.doc(inputs_path,1024**2,{'sha256':INPUT_SHA})
    legacy_path=Path(args.legacy_manifest).resolve()
    legacy=reader.doc(legacy_path,16*1024**2,{'sha256':LEGACY_SOURCE_SHA})
    legacy_limits=reader.doc(args.legacy_limits,1024**2,{'sha256':LEGACY_LIMITS_SHA})
    source_metadata(source,input_,legacy,legacy_limits)
    reader.pin(legacy_path.parent/'arrays.bin',128*1024**2,{'sha256':LEGACY_ARRAY_SHA})
    source_raw=source_path.parent/'arrays.bin';reader.pin(source_raw,128*1024**2,source['files']['arrays.bin']);need(source['files']['arrays.bin']['sha256']==SOURCE_ARRAY_SHA,'frozen source raw identity')
    reader.pin(source_path.parent/'progress.jsonl',16*1024**2,source['files']['progress.jsonl'])
    for c in input_['cases']:
        p=inputs_path.parent/c['token_file']['path'];reader.pin(p,4096,c['token_file']);need(p.read_bytes()==np.asarray(c['tokens'],dtype='<u4').tobytes(),'fixed raw input IDs')
    proof=reader.doc(inputs_path.parent/input_['tokenizer_probe']['path'],1024**2,input_['tokenizer_probe'])
    tokenizer_path=Path(args.model).resolve()/'tokenizer.json';reader.pin(tokenizer_path,34*1024**2,{'bytes':33384568,'sha256':TOKENIZER_SHA})
    import tokenizers
    need(tokenizers.__version__=='0.22.2','frozen tokenizers runtime')
    tokenizer=tokenizers.Tokenizer.from_file(str(tokenizer_path))
    for text,id_ in [('<bos>',2),('<eos>',1),('<end_of_turn>',106),('<image_soft_token>',V)]:need(tokenizer.token_to_id(text)==id_,'codec special mapping')
    need(not (native/'failure.json').exists(),'native failure record present')
    summary=reader.doc(native/'summary.json',1024**2)
    journal_pin=reader.pin(native/'native.jsonl',16*1024**2);reader.pin(native/'native.bin',256*1024**2)
    if args.capture_mode=='historical-posthoc':
        need(args.historical_check is not None,'posthoc requires original rejected check receipt')
        previous=reader.doc(args.historical_check,1024**2,{'sha256':LEGACY_CHECK_SHA})
        need(previous['passed']is False and previous['numerical']['failures']==92 and previous['numerical']['wide_exceedances']==98,'historical rejection preserved')
        need(args.historical_native_store_check is not None,'posthoc requires rejected source3 screen')
        rejected=reader.doc(args.historical_native_store_check,1024**2,{'sha256':'a0d59c80c2715209e45fd3590ac25eba7e805a755b0e7c5365188fe023545f8d'})
        need(rejected['passed']is False and rejected['numerical']['failures']==84,'historical84 rejection preserved')
        reader.pin(native/'native.bin',256*1024**2,{'sha256':'805ce524e061940919d0ec9cbde47f0e62ce2bca296b282ea82ebb5c15e2e557'})
        reader.pin(native/'native.jsonl',16*1024**2,{'sha256':'9dc813968cac2535e37b5de42783acfa9d2e1b4942013c5618875ddcc347a13a'})
    else:need(args.historical_check is None and args.historical_native_store_check is None,'fresh capture must not use posthoc receipt options')
    sb=Bytes(source_raw);nb=None;weights=None
    try:
        nb=Bytes(native/'native.bin');audit=Audit(reader,source,input_,sb,nb,Events(native/'native.jsonl',journal_pin['sha256']),tokenizer);audit.codec_proof=proof;audit.source_path=source_path;audit.model_path=Path(args.model).resolve();audit.legacy=legacy;audit.legacy_limits=legacy_limits;audit.legacy_path=legacy_path;audit.capture_mode=args.capture_mode
        weights=local.Weights(Path(args.model).resolve()/'model.safetensors',reader)
        audit.reader.progress['own_input']={}
        audit.local=local.Validator(weights,audit.reader.progress['own_input'])
        audit.source_audit();result=audit.run(summary)
        result['passed']=(result['coupled_fatal']['failures']==0 and result['own_input']['semantic_failures']==0 and result['own_input']['numeric_failures']==0)
        result.update(audit.comparison_scope)
        result['native_store_comparison_passed']=result['numerical']['failures']==0
        result['native_conformance_passed']=result['passed']
        result['native_conformance_policy_sha256']=local.POLICY_SHA
        result['original_weight_ranges']=weights.ranges
        result['original_weight_peak_decode_bytes']=weights.max_numeric_bytes
        result['historical_rejection_preserved']=True
        result['gate_basis']=GATE_BASIS
        return result
    finally:
        primary=sys.exception();failures=[]
        for owner in (weights,nb,sb):
            if owner is not None:
                try:owner.close()
                except BaseException as exc:failures.append(exc)
        if failures:
            if primary is not None:
                for exc in failures:primary.add_note('readback cleanup: '+repr(exc))
            else:raise Failure('readback cleanup: '+repr(failures))


def finalize_qualification(result):
    """Eligibility is assigned only after the final endpoint outcome."""
    result['numerical_qualification']=False
    result['native_conformance_passed']=(result.get('passed')is True and not result.get('error') and not result.get('endpoint_error'))
    result['numerical_qualification_eligible']=(
        result.get('capture_mode')=='fresh-preregistered'
        and result.get('passed')is True
        and not result.get('error') and not result.get('endpoint_error'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','inputs','model','native','output','legacy-manifest','legacy-limits','conformance-policy'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--capture-mode',choices=('fresh-preregistered','historical-posthoc'),default='fresh-preregistered')
    parser.add_argument('--historical-check')
    parser.add_argument('--historical-native-store-check')
    args=parser.parse_args();output=Path(args.output).resolve();need(not output.exists(),'fresh audit output required');reader=Reader()
    result={'format':'toklet-gemma3-real-native-conformance-check-v1','passed':False,'numerical_qualification':False,'capture_mode':args.capture_mode,'retrospective_diagnostic':args.capture_mode=='historical-posthoc','native_conformance_policy_sha256':local.POLICY_SHA,'scope':'prospective selected native conformance; coupled trajectory retained separately, no original-BF16/full-quality/performance/app qualification'}
    try:
        result.update(execute(args,reader))
    except BaseException as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    try:reader.endpoints()
    except BaseException as exc:result['passed']=False;result['endpoint_error']=f'{type(exc).__name__}: {exc}'
    result['consumed']=reader.consumed
    result['audit_progress']=reader.progress
    result['limits']={'logical_equality':'saved hashes for unselected rows; raw recomputation only where captured','state':'public history/publication observations; no independent full physical KV or fault reconstruction','memory':'checked reported ledger, exact-budget/error observations; process fit/owned groups external','source_envelope':'all successor envelopes recomputed diagnostically;3232 literal legacy limits and historical WIDE hashes checked; raw source producer never imported'}
    finalize_qualification(result)
    output.write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
