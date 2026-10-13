"""Bounded actual GPU counter/Metal-capture collector; no performance acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import metal_engine
import native_usage

ROOT=metal_engine.ROOT


def digest(path):
    with path.open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def tiny(directory):
    """Exact original FP32 fixture weights in a small standard safetensors artifact."""
    fixture=ROOT/'tests/fixtures/tiny-v1'
    if digest(fixture/'manifest.json')!='1d40cec4f24ece5fdc7b727d3e96f41237621120908833ee6c7df83352acfdd4':raise ValueError('tiny source manifest identity differs')
    manifest=json.loads((fixture/'manifest.json').read_text())
    directory.mkdir()
    config={'model_type':'qwen2','architectures':['Qwen2ForCausalLM'],'hidden_act':'silu','hidden_size':32,'intermediate_size':64,'num_hidden_layers':2,'num_attention_heads':4,'num_key_value_heads':2,'vocab_size':64,'max_position_embeddings':32,'rms_norm_eps':1e-6,'rope_theta':1e6,'tie_word_embeddings':True,'torch_dtype':'float32'}
    (directory/'config.json').write_text(json.dumps(config)+'\n')
    header={};data=bytearray()
    for name,item in sorted(manifest['arrays'].items()):
        if not name.startswith('weights.'):continue
        if item['dtype']!='f32le':raise ValueError('tiny weight dtype differs')
        payload=(fixture/item['file']).read_bytes()
        if len(payload)!=item['bytes'] or hashlib.sha256(payload).hexdigest()!=item['sha256']:raise ValueError('tiny weight hash differs')
        suffix=name.removeprefix('weights.')
        if suffix in ('embed_tokens','norm') or suffix.endswith('layernorm'):target=f'model.{suffix}.weight'
        else:
            _,layer,projection,kind=suffix.split('.')
            block='self_attn' if projection in ('q_proj','k_proj','v_proj','o_proj') else 'mlp'
            target=f'model.layers.{layer}.{block}.{projection}.{kind}'
        start=len(data);data.extend(payload)
        header[target]={'dtype':'F32','shape':item['shape'],'data_offsets':[start,len(data)]}
    metadata=json.dumps(header,separators=(',',':')).encode();metadata+=b' '*((-len(metadata))%8)
    with (directory/'model.safetensors').open('xb') as output:output.write(len(metadata).to_bytes(8,'little'));output.write(metadata);output.write(data)


def run(binary,kind,output,raw,capture):
    if any(p.exists() for p in (output,raw,output.with_suffix('.stderr'),capture)):raise ValueError('evidence destination exists')
    model=ROOT/'models/Qwen2.5-0.5B-Instruct' if kind=='qwen' else raw.parent/'tiny-model'
    if kind=='tiny':tiny(model)
    elif digest(model/'model.safetensors')!='fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe':raise ValueError('Qwen checkpoint identity differs')
    sources=sorted(set([Path(__file__),Path(metal_engine.__file__),Path(native_usage.__file__),ROOT/'Cargo.toml',ROOT/'Cargo.lock',ROOT/'examples/metal_trace.rs',Path(__file__).with_name('metal_trace_analysis.py')]+list((ROOT/'src').rglob('*.rs'))+list((ROOT/'src').rglob('*.metal'))))
    identities={str(p):digest(p) for p in sources+[binary,model/'config.json',model/'model.safetensors']}
    record={'scope':'M4.4 stage/host/allocation/transition attribution; instrumented intervals are not speed-parity acceptance','kind':kind,'identities':identities,'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'raw_path':str(raw),'capture_path':str(capture),'idle_usage':[],'build':'CARGO_TARGET_DIR=artifacts/m4-metal-trace-v1 cargo build --offline --release --features metal --example metal_trace','rustc':subprocess.check_output(['rustc','--version'],text=True).strip(),'platform':subprocess.check_output(['/usr/bin/sw_vers'],text=True).strip(),'power':subprocess.check_output(['/usr/bin/pmset','-g','batt'],text=True).strip()}
    child=None;transport=None
    with output.open('x') as summary,raw.open('x') as events,output.with_suffix('.stderr').open('xb') as stderr:
      try:
        child=subprocess.Popen([str(binary),str(model),str(capture)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,env={**os.environ,'MTL_CAPTURE_ENABLED':'1'})
        transport=metal_engine.Transport(child,events,stderr,stdout_limit=16*1024*1024,event_limit=32768)
        ready=transport.read();record['ready']=ready
        if ready.get('event')!='ready' or ready.get('model')!=kind or ready['runs']!=14 or ready['events_per_operation']!=(82 if kind=='tiny' else 918) or not 0<ready['combined_admitted_bytes']<=6*1024**3 or ready.get('debug_assertions') is not False or ready.get('capture_admitted_bytes',6*1024**3+1)>6*1024**3 or ready.get('capture_admitted_bytes')!=ready['combined_admitted_bytes']+ready.get('capture_memory_allowance_bytes',0):raise ValueError('worker admission/protocol differs')
        def proceed():child.stdin.write(b'continue\n');child.stdin.flush()
        record['idle_usage'].append({'run':-1,**native_usage.usage(child.pid)})
        proceed()
        for run in range(14):
            for operation in ('prefill','decode'):
                header=transport.read()
                if header.get('event')!='profile' or header.get('run')!=run or header.get('operation')!=operation or header.get('event_count')!=ready['events_per_operation']:raise ValueError('profile header differs')
                for index in range(ready['events_per_operation']):
                    event=transport.read()
                    if event.get('event')!='stage' or event.get('run')!=run or event.get('operation')!=operation or event.get('index')!=index:raise ValueError('profile coverage/order differs')
            idle=transport.read()
            if idle.get('event')!='idle' or idle.get('run')!=run or idle['baseline_memory']!=ready['baseline_memory'] or idle['profile_memory']!=ready['profile_memory']:raise ValueError('resident resources changed')
            usage=native_usage.usage(child.pid)
            if not 0<usage['lifetime_max_phys_footprint']<=6*1024**3:raise ValueError('native charge outside budget')
            record['idle_usage'].append({'run':run,**usage});proceed()
        if transport.read()!= {'event':'done'}:raise ValueError('trace completion differs')
        child.stdin.close()
        if transport.finish()!=0:raise ValueError('trace child failed')
      except Exception as error:
        record['error']=str(error);raise
      finally:
        if child is not None:
            metal_engine.kill_owned_group(child);child.wait(timeout=10)
            if transport:transport.close()
            for stream in (child.stdin,child.stdout,child.stderr):
                if stream and not stream.closed:stream.close()
        record['endpoint_identities']={str(p):digest(p) for p in sources+[binary,model/'config.json',model/'model.safetensors']}
        record['identities_unchanged']=record['endpoint_identities']==identities
        record['raw_sha256']=digest(raw)
        if capture.exists():record['capture_files']={str(p.relative_to(capture)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(capture.rglob('*')) if p.is_file()}
        summary.write(json.dumps(record,indent=2)+'\n');summary.flush()
    if not record['identities_unchanged']:raise ValueError('trace identities changed')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--kind',choices=('tiny','qwen'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--capture',type=Path,required=True)
    args=parser.parse_args();run(args.binary.resolve(),args.kind,args.output.resolve(),args.raw.resolve(),args.capture.resolve())
