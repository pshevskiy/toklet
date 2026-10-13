"""Qualify Qwen split-half rotary64 and preserve all tiny/front regressions."""
import argparse,copy,hashlib,json,struct,sys
from pathlib import Path
import rust_metal_rotary as rotary
import rust_metal_qwen_front as front
probe=front.probe;qwen=front.qwen


def cases():
    half,single,_=qwen.checked_manifests();config=json.loads(qwen.CONFIG.read_bytes())
    if(config['num_attention_heads'],config['num_key_value_heads'],config['hidden_size']//config['num_attention_heads'],config['rope_theta'])!=(14,2,64,1e6):raise ValueError('Qwen rotary topology differs')
    inputs=[];anchors={}
    for name,n in qwen.CASE_ROWS.items():
        for op,heads in(('q',14),('k',2)):
            prefix=f'cases.{name}.layers.0';key=f'{prefix}.{op}_rotary';source,si=qwen.rms.payload(qwen.HALF,half,f'{prefix}.{op}_projection','f16le');out,oi=qwen.rms.payload(qwen.HALF,half,key,'f16le');ref=single['arrays'][key];data,ri=qwen.rms.payload(qwen.SINGLE,{'arrays':{key:{'dtype':'f32le',**ref}}},key,'f32le');count=n*heads*64
            if oi['shape']!=[n,heads,64]or ri['shape']!=oi['shape']or si['shape']!=[n,heads*64]or oi['source_fp32_full_sha256']!=ri['sha256']:raise ValueError('Qwen rotary source shape differs')
            inputs.append({'id':key,'rows':n,'heads':heads,'dimension':64,'theta':1e6,'offset':half['cases'][name]['offset'],'input_bits':list(struct.unpack(f'<{count}H',source))})
            anchors[key]={'bits':list(struct.unpack(f'<{count}H',out)),'reference':list(struct.unpack(f'<{count}f',data)),'gate':oi['gate_against_source_fp32'],'input_fp16_sha256':si['sha256'],'oracle_fp16_sha256':oi['sha256'],'source_fp32_sha256':ri['sha256']}
    bits=list(struct.unpack('<128H',struct.pack('<128e',*range(1,129))));item={'id':'manual.qwen_position_zero','rows':1,'heads':2,'dimension':64,'theta':1e6,'offset':0,'input_bits':bits};inputs.append(item);anchors[item['id']]={'bits':bits,'reference':list(range(1,129)),'gate':None,'manual_oracle':'Exact nonzero identity at position0'}
    return inputs,anchors


def validate(e,item,a):
    rotary.validate(e,[item],{item['id']:a})
    rows=e['result']['cases'][0]['rows']
    if len({r['qualification']['output_sha256']for r in rows})!=1:raise ValueError('Qwen rotary repetitions differ')


def run(output,build,directory):
    probe.check_paths(output,build)
    evidence=(output,output.with_suffix(output.suffix+'.stderr.txt'),output.with_suffix(output.suffix+'.native.jsonl'))
    if directory.exists():raise ValueError('Refusing to overwrite rotary inputs')
    for path in(build,*evidence):
        if directory.resolve()==path.resolve()or directory.resolve()in path.resolve().parents or path.resolve()in directory.resolve().parents:raise ValueError('Rotary paths must be disjoint')
    for root in(qwen.HALF,qwen.SINGLE,qwen.WEIGHTS.parent,rotary.rms.FP16,rotary.rms.FP32):
        if any(path.resolve()==root.resolve()or path.resolve().is_relative_to(root.resolve())for path in(build,directory,*evidence)):raise ValueError('Rotary outputs must be outside sources')
    inputs,anchors=cases();ti,ta,to=rotary.cases();fi,fa=front.cases();di,da,_=front.qp.decoder.cases();directory.mkdir(parents=True);identities={};jobs=[]
    previous=json.loads((probe.ROOT/'benchmarks/results/2026-10-13-rust-metal-qwen-front-v2.json').read_text());origin=probe.ROOT/'artifacts/rust-metal-qwen-front-inputs-v2'
    for op in('q','k','v'):
        p=origin/f'{op}.f16le';expected=previous['extra_context']['input_sha256'][str(p.resolve())]
        if probe.common.digest(p)!=expected:raise ValueError('Qualified front weight identity differs')
        target=directory/p.name
        with target.open('xb')as f:f.write(p.read_bytes())
        identities[str(target.resolve())]=probe.common.digest(target)
    payloads=[(f'qwen_{i+1}',{'format':'toklet-metal-qwen-rotary-input-v1','cases':[item]},'--rotary-input',lambda e,item=item:validate(e,item,anchors[item['id']]))for i,item in enumerate(inputs)]
    payloads.extend([('tiny_rotary',{'format':'toklet-metal-rotary-input-v1','cases':ti},'--rotary-input',lambda e:rotary.validate(e,ti,ta)),('rotary_overflow',{'format':'toklet-metal-rotary-input-v1','cases':[to]},'--rotary-input',rotary.overflow_rejected)])
    for i,item in enumerate(fi):
        p=origin/f'front_{i+1}.json'
        if probe.common.digest(p)!=previous['extra_context']['input_sha256'][str(p.resolve())]:raise ValueError('Front fixture identity differs')
        payload=json.loads(p.read_text());a={**fa[item['id']],'norm_weight_sha256':previous['extra_context']['weights']['model.layers.0.input_layernorm.weight']['half_sha256']};payloads.append((f'front_{i+1}',payload,'--qwen-front-input',lambda e,payload=payload,a=a:front.validate(e,payload,a)))
    payloads.extend((f'decoder_{i+1}',item,'--decoder-input',lambda e,item=item:front.qp.decoder.validate(e,item,da[item['id']]))for i,item in enumerate(di))
    for label,payload,option,validator in payloads:
        p=directory/f'{label}.json'
        with p.open('x')as f:f.write(json.dumps(payload,separators=(',',':'))+'\n')
        identities[str(p.resolve())]=probe.common.digest(p);jobs.append((label,[option,str(p.resolve())],validator))
    drivers={str(Path(m.__file__).resolve()):probe.common.digest(Path(m.__file__))for name,m in list(sys.modules.items())if name.startswith('rust_metal_')and getattr(m,'__file__',None)};drivers[str(Path(__file__).resolve())]=probe.common.digest(Path(__file__));last=jobs[-1][2]
    def final(e):
        result=last(e)
        if cases()!=(inputs,anchors):raise ValueError('Rotary references changed')
        for name,h in{**identities,**drivers}.items():
            if probe.common.digest(Path(name))!=h:raise ValueError('Rotary identities changed')
        return result
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    probe.run(output,build,extra_runs=jobs,extra_context={'input_sha256':identities,'qualification_driver_sha256':drivers,'half_manifest_sha256':qwen.IDENTITIES[qwen.HALF],'source_manifest_sha256':qwen.IDENTITIES[qwen.SINGLE],'prior_front_record_sha256':probe.common.digest(probe.ROOT/'benchmarks/results/2026-10-13-rust-metal-qwen-front-v2.json'),'scope':'Eight saved Qwen layer0Q/K64dim rotary stages +exactposition0manual, originalsourcegates; tinyrotary/front/fulltinyregressions. No Qwen propagation into rotary/cache/GQA/full-layer/decoder/session/performance/M4closure.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True);a=p.parse_args();run(a.output,a.build_dir,a.input_dir)
if __name__=='__main__':main()
