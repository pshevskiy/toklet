"""Pinned selected-artifact Gemma3 NativeHalf source export, independently CPU only.

Original Q4/BF16 rows are decoded in bounded stripes to one immutable typed FP32
model. No tensor values are read at import, no MLX/Torch/remote code is used.
"""
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import traceback
import numpy as np
import gemma3_real_half as graph
# These existing helpers are generic byte/descriptor/runtime operations only.
from mlx_phi3_real import bounded, descriptor_sha, package_pins, runtime_identity, sha, unique, validate_header

FORMAT='toklet-mlx-gemma3-real-reference-v3'
PROTOCOL_SHA='035bd42111d8e702a65a7442a805a3cffac6c04517c90a7cf95ccdc86e668c68'
LEGACY_MANIFEST_SHA='60617b26647d2d4a6c96c41ac8675d978cfdec6dc251545ae308e52dd01c9033'
LEGACY_LIMITS_SHA='f40fa2275a9c9f193287f32f2764a3dbf9e4d8836a56706583df1d5555151bdd'
INPUT_SHA='30c67871d9de7086d32e8c250416ba2134a4c65c8f54173f6cd17401aebba4ac'
HEADER_SHA='994efdf1aa3b4f6bc39d7935b1286380c6f9f49e926e02405fed6ed72f6e578f'
HEADER_BYTES=53490
FILE_BYTES=150939130
STRIPE=128
MAX_ARRAYS=128*1024**2
MAX_MANIFEST=16*1024**2


def identity(path):
    p=Path(path)
    return dict(bytes=p.stat().st_size,sha256=sha(p))


def specifications():
    matrices={'model.embed_tokens':(262144,640)}
    norms={'model.norm.weight':640}
    for layer in range(18):
        p=f'model.layers.{layer}'
        for module,rows,columns in [('self_attn.q_proj',1024,640),('self_attn.k_proj',256,640),('self_attn.v_proj',256,640),('self_attn.o_proj',640,1024),('mlp.gate_proj',2048,640),('mlp.up_proj',2048,640),('mlp.down_proj',640,2048)]:
            matrices[p+'.'+module]=(rows,columns)
        for name,width in [('input_layernorm',640),('post_attention_layernorm',640),('pre_feedforward_layernorm',640),('post_feedforward_layernorm',640),('self_attn.q_norm',256),('self_attn.k_norm',256)]:
            norms[p+'.'+name+'.weight']=width
    expected={}
    for name,(rows,columns) in matrices.items():
        expected[name+'.weight']=('U32',[rows,columns//8])
        expected[name+'.scales']=('BF16',[rows,columns//64])
        expected[name+'.biases']=('BF16',[rows,columns//64])
    expected.update({name:('BF16',[width]) for name,width in norms.items()})
    return matrices,norms,expected


def rows_from_stream(stream,origin,descriptor,begin,end):
    shape=descriptor['shape']
    if (type(begin) is not int or type(end) is not int or not 0<=begin<end<=shape[0]
            or descriptor['dtype'] not in ('U32','BF16')):
        raise ValueError('Original row range/dtype')
    item=4 if descriptor['dtype']=='U32' else 2
    width=math.prod(shape[1:]);lo,hi=descriptor['data_offsets']
    if lo<0 or hi-lo!=math.prod(shape)*item:raise ValueError('Original extent')
    stream.seek(origin+lo+begin*width*item)
    data=stream.read((end-begin)*width*item)
    if len(data)!=(end-begin)*width*item:raise ValueError('Truncated original rows')
    # frombuffer supports unaligned bytes. This owned copy outlives no borrowed mapping.
    return np.frombuffer(data,dtype='<u4' if item==4 else '<u2').copy().reshape((end-begin,*shape[1:]))


class Weights:
    layers=18
    dim=256
    vocab=262144
    def __init__(self,folder,pins):
        self.folder=Path(folder);self.pins=pins;self.matrices={};self.norms={};self.stream=None
    def endpoints(self):
        for name,want in self.pins.items():
            if identity(self.folder/name)!=want:raise ValueError('Artifact endpoint differs: '+name)
        if self.stream is not None and descriptor_sha(self.stream)!=self.pins['model.safetensors']['sha256']:
            raise ValueError('Open weight descriptor endpoint differs')
    def __enter__(self):
        try:
            self.endpoints()
            self.stream=(self.folder/'model.safetensors').open('rb')
            if os.fstat(self.stream.fileno()).st_size!=FILE_BYTES or descriptor_sha(self.stream)!=self.pins['model.safetensors']['sha256']:
                raise ValueError('Weight descriptor differs')
            prefix=self.stream.read(8)
            if len(prefix)!=8 or struct.unpack('<Q',prefix)[0]!=HEADER_BYTES:raise ValueError('Header prefix')
            raw=self.stream.read(HEADER_BYTES)
            if hashlib.sha256(raw).hexdigest()!=HEADER_SHA:raise ValueError('Header identity')
            header=json.loads(raw,object_pairs_hook=unique);header.pop('__metadata__',None)
            matrices,norms,expected=specifications();origin=8+HEADER_BYTES
            index=bounded(self.folder/'model.safetensors.index.json',1024**2,self.pins['model.safetensors.index.json']['sha256'])
            validate_header(header,index,expected,origin,FILE_BYTES)
            for name,(rows,columns) in matrices.items():
                value=np.empty((rows,columns),dtype=np.float32)
                for begin in range(0,rows,STRIPE):
                    end=min(rows,begin+STRIPE)
                    parts=[rows_from_stream(self.stream,origin,header[name+suffix],begin,end) for suffix in ('.weight','.scales','.biases')]
                    value[begin:end]=graph.decode(*parts,columns)
                value.flags.writeable=False;self.matrices[name]=value
            for name,width in norms.items():
                words=rows_from_stream(self.stream,origin,header[name],0,width)
                value=graph.finite(np.float32(1)+graph.bf16(words));value.flags.writeable=False;self.norms[name]=value
            if sum(v.nbytes for v in self.matrices.values())!=1072168960 or sum(v.nbytes for v in self.norms.values())!=223744:
                raise ValueError('Expanded owner accounting differs')
            self.endpoints()
            return self
        except BaseException as primary:
            self.matrices.clear();self.norms.clear()
            stream=self.stream;self.stream=None
            if stream is not None:
                try:stream.close()
                except BaseException as cleanup:primary.add_note('Load cleanup failure: '+repr(cleanup))
            raise
    def __exit__(self,typ,error,tb):
        failures=[]
        try:self.endpoints()
        except BaseException as exc:failures.append(exc)
        self.matrices.clear();self.norms.clear()
        stream=self.stream;self.stream=None
        if stream is not None:
            try:stream.close()
            except BaseException as exc:failures.append(exc)
        preserve_cleanup(error,failures)
        return False


def preserve_cleanup(primary,failures):
    if primary is not None:
        for failure in failures:primary.add_note('Cleanup/endpoint failure: '+repr(failure))
    elif failures:
        for failure in failures[1:]:failures[0].add_note('Additional cleanup failure: '+repr(failure))
        raise failures[0]


class Writer:
    """One raw file, incremental descriptor journal, no retained dense traces."""
    def __init__(self,out):
        self.out=Path(out);self.raw=(self.out/'arrays.bin').open('xb');self.journal=(self.out/'progress.jsonl').open('x');self.arrays={}
    def emit(self,case,profile,row,stage,a):
        if a.dtype not in (np.float16,np.float32):raise ValueError('Unexpected store dtype')
        graph.finite(a)
        name=f'{case}.row_{row}.{stage}'
        group=self.arrays.setdefault(name,{})
        if profile in group:raise ValueError('Duplicate capture')
        data=a.astype('<f2' if a.dtype==np.float16 else '<f4',copy=False).tobytes(order='C');offset=self.raw.tell()
        if offset+len(data)>MAX_ARRAYS:raise ValueError('Raw capture cap')
        self.raw.write(data);self.raw.flush()
        d=dict(file='arrays.bin',offset=offset,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),dtype='u16le' if a.dtype==np.float16 else 'f32le',shape=list(a.shape))
        group[profile]=d
        self.journal.write(json.dumps(dict(case=case,profile=profile,row=row,stage=stage,descriptor=d))+'\n');self.journal.flush()
    def close(self):
        failures=[]
        for stream in (self.raw,self.journal):
            try:stream.close()
            except BaseException as exc:failures.append(exc)
        preserve_cleanup(None,failures)
    def gates(self,legacy,limits):
        self.raw.flush()
        if set(self.arrays)!=set(legacy['arrays']) or set(self.arrays)!=set(limits['gates']):
            raise ValueError('Legacy exact group inventory differs')
        with (self.out/'arrays.bin').open('rb') as stream:
            for name,group in self.arrays.items():
                if not {'main','wide'}<=set(group) or set(group)-{'main','wide','envelope','envelope_scope','gate','gate_basis'}:
                    raise ValueError('Incomplete/unexpected pair: '+name)
                values=[]
                for profile in ('main','wide'):
                    d=group[profile];stream.seek(d['offset']);b=stream.read(d['bytes'])
                    if hashlib.sha256(b).hexdigest()!=d['sha256']:raise ValueError('Raw capture changed')
                    values.append(np.frombuffer(b,dtype='<f2' if d['dtype']=='u16le' else '<f4').astype(np.float64))
                if group['main']['shape']!=group['wide']['shape'] or group['main']['dtype']!=group['wide']['dtype']:raise ValueError('Profile shape differs')
                envelope=float(np.max(np.abs(values[0]-values[1])))
                previous=legacy['arrays'][name]
                for profile in ('main','wide'):
                    if any(group[profile][key]!=previous[profile][key] for key in ('shape','dtype','bytes')):
                        raise ValueError('Legacy exact shape/store differs: '+name)
                if group['wide']['sha256']!=previous['wide']['sha256']:
                    raise ValueError('Unchanged WIDE identity differs: '+name)
                gate=limits['gates'][name]
                if gate!=previous['gate'] or set(gate)!={'atol','rtol','fatal_reference'} or gate['fatal_reference']!='main':
                    raise ValueError('Legacy literal numerical limits differ: '+name)
                if not math.isfinite(gate['atol']) or gate['atol']<1e-4 or gate['rtol']!=1e-3:
                    raise ValueError('Invalid legacy limits')
                group['envelope']=envelope
                group['envelope_scope']='successor MAIN/WIDE diagnostic only; not a limit input'
                group['gate']=dict(gate)
                group['gate_basis']='legacy_fixed_limits_new_native_store_center_v1'


LAYER_STAGES=('input_norm','q_projection','k_projection','v_projection','q_norm','k_norm',
              'q_rotary','k_rotary','attention_scores','attention_probabilities',
              'attention_context','attention_projection','post_attention_norm',
              'attention_residual','pre_ff_norm','gate_projection','up_projection',
              'gate_gelu','gated','down_projection','post_ff_norm','output')


def capture_shape(stage,tokens):
    short=stage.rsplit('.',1)[-1]
    if short=='logits':return [1,262144],'f32le'
    if short in ('attention_scores','attention_probabilities'):return [4,1,tokens],'f32le'
    if short in ('q_projection','q_norm','q_rotary'):return [1,4,256],'u16le'
    if short in ('k_projection','v_projection','k_norm','k_rotary'):return [1,1,256],'u16le'
    if short=='attention_context':return [1,1024],'u16le'
    if short in ('gate_projection','up_projection','gate_gelu','gated'):return [1,2048],'u16le'
    return [1,640],'u16le'


def validate_inventory(arrays,cases,teacher):
    expected={}
    stages=['embedding_lookup','embedding','final_norm','logits']+[
        f'layers.{layer}.{stage}' for layer in range(18) for stage in LAYER_STAGES]
    assert len(stages)==400
    for case in cases:
        for row in case['selected_rows']:
            for stage in stages:expected[f'{case["name"]}.row_{row}.{stage}']=capture_shape(stage,len(case['tokens']))
    if len(expected)!=3200:raise ValueError('Fixed selected-row inventory')
    if [t['prompt'] for t in teacher]!=[c['name'] for c in cases[:2]]:raise ValueError('Teacher prompt inventory')
    for trial in teacher:
        if not 1<=len(trial['steps'])<=16:raise ValueError('Teacher step cap')
        for step,record in enumerate(trial['steps']):
            row=len(record['prefix'])-1
            expected[f'teacher.{trial["prompt"]}.step_{step}.row_{row}.logits']=([1,262144],'f32le')
    if set(arrays)!=set(expected):raise ValueError('Missing/unexpected captured stages')
    for name,(shape,dtype) in expected.items():
        group=arrays[name]
        if set(group)!=set(('main','wide')):raise ValueError('Incomplete main/wide pair')
        for desc in group.values():
            if desc['shape']!=shape or desc['dtype']!=dtype or desc['bytes']!=math.prod(shape)*(2 if dtype=='u16le' else 4):
                raise ValueError('Captured shape/store differs: '+name)
    return dict(graph_stage_groups=3200,teacher_stage_groups=len(expected)-3200,total_arrays=2*len(expected))


def inputs(path):
    path=Path(path);m=bounded(path,1024**2,INPUT_SHA)
    if m['format']!='toklet-gemma3-real-input-contract-v1' or [c['name'] for c in m['cases']]!=['plain_france','plain_sequence','window544']:
        raise ValueError('Input contract')
    for c in m['cases']:
        p=path.parent/c['token_file']['path']
        if identity(p)!={k:c['token_file'][k] for k in ('bytes','sha256')}:raise ValueError('Input token endpoint')
        b=p.read_bytes()
        if list(struct.unpack('<'+'I'*(len(b)//4),b))!=c['tokens']:raise ValueError('Input IDs differ')
    return m


def run(args):
    out=Path(args.out);out.mkdir(exist_ok=False)
    writer=None;primary=None;result=None;graph_started=False;graph_completed=False
    try:
        if np.__version__!='2.3.2':raise ValueError('Pinned NumPy2.3.2 required')
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
            if os.environ.get(name)!='1':raise ValueError('Explicit single-thread source environment required: '+name)
        protocol=bounded(Path(args.protocol),1024**2,PROTOCOL_SHA);contract=inputs(args.inputs)
        legacy=bounded(Path(args.legacy_manifest),MAX_MANIFEST,LEGACY_MANIFEST_SHA)
        limits=bounded(Path(args.legacy_limits),1024**2,LEGACY_LIMITS_SHA)
        if (legacy['format']!='toklet-mlx-gemma3-real-reference-v2' or legacy['cases']!=contract['cases']
                or limits['format']!='toklet-gemma3-legacy-numeric-limits-v1' or limits['group_count']!=3232
                or limits['source']['sha256']!=LEGACY_MANIFEST_SHA):
            raise ValueError('Legacy profile/input/limits basis')
        graph.reset_projection_audit()
        if protocol['memory']['admitted_bytes']!=4931096570 or protocol['process_bound']!=6*1024**3:raise ValueError('Ledger identity')
        runtime=runtime_identity();packages=package_pins();config=io.StringIO()
        with contextlib.redirect_stdout(config):np.show_config()
        if len(config.getvalue())>65536:raise ValueError('BLAS configuration output cap')
        own={Path(__file__).name:identity(Path(__file__)),Path(graph.__file__).name:identity(Path(graph.__file__)), 'mlx_phi3_real.py':identity(Path(__file__).with_name('mlx_phi3_real.py'))}
        writer=Writer(out);teacher=[]
        with Weights(args.model,protocol['model_file_pins']) as weights:
            for c in contract['cases']:
                for profile in ('main','wide'):
                    graph_started=True
                    result=graph.forward(weights,c['tokens'],c['selected_rows'],profile,lambda row,stage,a,c=c,profile=profile:writer.emit(c['name'],profile,row,stage,a))
                    del result
            for c,old_trial in zip(contract['cases'][:2],legacy['greedy']):
                if old_trial['prompt']!=c['name']:raise ValueError('Historical teacher prompt')
                history=list(c['tokens']);visible=[];steps=[]
                if not 1<=len(old_trial['steps'])<=16:raise ValueError('Historical teacher bound')
                for step,old_step in enumerate(old_trial['steps']):
                    if old_step['prefix']!=history:raise ValueError('Historical teacher prefix')
                    name=f'teacher.{c["name"]}.step_{step}';row=len(history)-1
                    wide=graph.forward(weights,history,[row],'wide',lambda r,s,a:writer.emit(name,'wide',r,s,a),logits_only=True)
                    token=int(np.argmax(wide[0]));del wide
                    if token!=old_step['wide_teacher_token']:raise ValueError('Unchanged WIDE teacher differs')
                    main=graph.forward(weights,history,[row],'main',lambda r,s,a:writer.emit(name,'main',r,s,a),logits_only=True)
                    main_token=int(np.argmax(main[0]));del main
                    stop=token in (1,106)
                    if stop!=old_step['stop']:raise ValueError('Historical teacher stop differs')
                    steps.append(dict(prefix=list(history),wide_teacher_token=token,main_greedy_diagnostic=main_token,stop=stop))
                    if stop:
                        if step+1!=len(old_trial['steps']):raise ValueError('Historical steps after stop')
                        break
                    visible.append(token);history.append(token)
                if visible!=old_trial['visible_ids']:raise ValueError('Historical teacher visible IDs')
                teacher.append(dict(prompt=c['name'],steps=steps,visible_ids=visible,stop_reason='stop_token' if steps[-1]['stop'] else 'length'))
            counts=validate_inventory(writer.arrays,contract['cases'],teacher)
            writer.gates(legacy,limits);graph_completed=True
        if package_pins()!=packages:raise ValueError('Runtime package endpoint changed')
        inputs(args.inputs)
        bounded(Path(args.protocol),1024**2,PROTOCOL_SHA)
        bounded(Path(args.legacy_manifest),MAX_MANIFEST,LEGACY_MANIFEST_SHA)
        bounded(Path(args.legacy_limits),1024**2,LEGACY_LIMITS_SHA)
        for name,want in own.items():
            if identity(Path(__file__).with_name(name))!=want:raise ValueError('Producer endpoint changed')
        result=dict(format=FORMAT,source_reference_executed=True,target_gpu_executed=False,protocol_sha256=PROTOCOL_SHA,input_sha256=INPUT_SHA,model=dict(repository=protocol['repository'],revision=protocol['revision'],file_pins=protocol['model_file_pins']),profile=protocol['profile'],policy=protocol['gates'],gate_basis=protocol['gate_basis'],projection_audit=graph.PROJECTION_AUDIT,memory=protocol['memory'],cases=contract['cases'],greedy=teacher,counts=counts,arrays=writer.arrays,runtime=runtime,package_pins=packages,blas_configuration=config.getvalue(),producer_files=own,scope='selected-artifact numerical profiles; no native output, original-unquantized quality or app qualification')
    except BaseException as exc:
        primary=exc
    finally:
        if writer is not None:
            try:writer.close()
            except BaseException as cleanup:
                if primary is None:primary=cleanup
                else:primary.add_note('Writer cleanup failure: '+repr(cleanup))
    try:
        if primary is not None:raise primary
        result['files']={name:identity(out/name) for name in ('arrays.bin','progress.jsonl')}
        encoded=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode()
        if len(encoded)>MAX_MANIFEST:raise ValueError('Manifest cap')
        (out/'manifest.json').write_bytes(encoded)
    except BaseException as exc:
        failure=dict(error=repr(exc),traceback=traceback.format_exc(),source_reference_started=graph_started,
                     source_reference_completed=graph_completed,target_gpu_executed=False,
                     partial_arrays=writer.arrays if writer is not None else {},projection_audit=graph.PROJECTION_AUDIT)
        try:(out/'failure.json').write_text(json.dumps(failure,indent=2,allow_nan=False)+'\n')
        except BaseException as report_error:exc.add_note('Failure report write: '+repr(report_error))
        raise
    print(json.dumps(dict(completed=True,stage_groups=len(writer.arrays),files={**result['files'],'manifest.json':identity(out/'manifest.json')},source_reference_executed=True,target_gpu_executed=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--inputs',required=True);parser.add_argument('--protocol',required=True);parser.add_argument('--out',required=True);parser.add_argument('--legacy-manifest',required=True);parser.add_argument('--legacy-limits',required=True);run(parser.parse_args())
