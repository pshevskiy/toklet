"""Qualify layer0 Qwen Q/K/V projections from frozen normalized Half inputs."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import rust_metal_qwen_rms as qwen
import rust_metal_projection as projection
import rust_metal_decoder as decoder


def weights(path,expected_sha,expected_bytes):
    tensors={f'model.layers.0.self_attn.{op}_proj.{kind}':([out,896]if kind=='weight'else[out]) for op,out in [('q',896),('k',128),('v',128)]for kind in ('weight','bias')}
    result={};ranges=[]
    with path.open('rb')as file:
        before=os.fstat(file.fileno())
        if path.stat().st_size!=expected_bytes or qwen.stream_digest(file)!=expected_sha:raise ValueError('Qwen checkpoint identity differs')
        file.seek(0);prefix=file.read(8)
        if len(prefix)!=8:raise ValueError('Truncated Qwen header')
        size=struct.unpack('<Q',prefix)[0]
        if not 2<=size<=1024*1024 or size>expected_bytes-8:raise ValueError('Qwen header exceeds bounds')
        data=file.read(size)
        if len(data)!=size:raise ValueError('Truncated Qwen metadata')
        meta=json.loads(data,object_pairs_hook=qwen.unique_object)
        if not isinstance(meta,dict):raise ValueError('Qwen metadata object differs')
        for name,shape in tensors.items():
            item=meta.get(name);count=math.prod(shape)
            if not isinstance(item,dict)or item.get('dtype')!='BF16'or item.get('shape')!=shape:raise ValueError('Qwen projection weight dtype/shape differs')
            bounds=item.get('data_offsets')
            if not isinstance(bounds,list)or len(bounds)!=2 or any(type(n)is not int for n in bounds)or not 0<=bounds[0]<bounds[1]<=expected_bytes-8-size or bounds[1]-bounds[0]!=count*2 or bounds[0]%2:raise ValueError('Qwen projection weight range differs')
            if any(bounds[0]<end and start<bounds[1]for start,end in ranges):raise ValueError('Qwen projection tensors overlap')
            ranges.append(tuple(bounds));file.seek(8+size+bounds[0]);data=file.read(count*2)
            if len(data)!=count*2:raise ValueError('Truncated Qwen projection tensor')
            converted=bytearray(count*2)
            for i,(bits,)in enumerate(struct.iter_unpack('<H',data)):
                value=struct.unpack('<f',struct.pack('<I',bits<<16))[0]
                if not math.isfinite(value):raise ValueError('Nonfinite BF16 projection tensor')
                try:word=struct.pack('<e',value)
                except OverflowError as e:raise ValueError('Projection tensor exceeds Half')from e
                converted[i*2:i*2+2]=word
            result[name]={'data':bytes(converted),'shape':shape,'data_offsets':bounds,'source_bf16_sha256':hashlib.sha256(data).hexdigest(),'half_sha256':hashlib.sha256(converted).hexdigest()}
        after=os.fstat(file.fileno());entry=path.stat()
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)or (entry.st_dev,entry.st_ino)!=(after.st_dev,after.st_ino)or qwen.stream_digest(file)!=expected_sha:raise ValueError('Qwen checkpoint changed during conversion')
    return result


def cases():
    half,single,_=qwen.checked_manifests();inputs=[];anchors={}
    for name,rows in qwen.CASE_ROWS.items():
        prefix=f'cases.{name}.layers.0';data,item=qwen.rms.payload(qwen.HALF,half,prefix+'.input_norm','f16le')
        if item['shape']!=[rows,896]:raise ValueError('Qwen projection incoming shape differs')
        for op,out in [('q',896),('k',128),('v',128)]:
            key=f'{prefix}.{op}_projection';oracle,oi=qwen.rms.payload(qwen.HALF,half,key,'f16le');ref=single['arrays'][key]
            source,si=qwen.rms.payload(qwen.SINGLE,{'arrays':{key:{'dtype':'f32le',**ref}}},key,'f32le')
            if oi['shape']!=[rows,out]or si['shape']!=[rows,out]or oi['source_fp32_full_sha256']!=si['sha256']:raise ValueError('Qwen projection oracle/source shape differs')
            inputs.append({'id':key,'rows':rows,'inputs':896,'outputs':out,'input_bits':list(struct.unpack(f'<{rows*896}H',data)),'weight_file':f'{op}.f16le','op':op})
            anchors[key]={'source':list(struct.unpack(f'<{rows*out}f',source)),'oracle_bits':list(struct.unpack(f'<{rows*out}H',oracle)),'gate':oi['gate_against_source_fp32'],'input_sha256':item['sha256'],'oracle_sha256':oi['sha256'],'source_sha256':si['sha256']}
    return inputs,anchors


def validate(event,item,anchor):
    rows=event['result']['cases'][0]['rows']
    if len(rows)!=13 or any(row['output_bits']!=rows[0]['output_bits']or row['fp32_bits']!=rows[0]['fp32_bits']for row in rows):raise ValueError('Qwen repeated projection frames differ')
    for row in rows:row['fp32_sha256']=hashlib.sha256(struct.pack(f'<{len(row["fp32_bits"])}I',*row['fp32_bits'])).hexdigest()
    projection.validate(event,[item],{item['id']:anchor})


def run(output,build,directory):
    projection.rms.probe.check_paths(output,build)
    evidence=(output,output.with_suffix(output.suffix+'.stderr.txt'),output.with_suffix(output.suffix+'.native.jsonl'))
    if directory.exists():raise ValueError('Refusing to overwrite Qwen projection inputs')
    for path in (build,*evidence):
        if directory.resolve()==path.resolve()or directory.resolve()in path.resolve().parents or path.resolve()in directory.resolve().parents:raise ValueError('Qwen projection paths must be disjoint')
    for root in (qwen.HALF,qwen.SINGLE,qwen.HALF_PROBES,qwen.SINGLE_PROBES,qwen.WEIGHTS.parent,projection.rms.FP16,projection.rms.FP32):
        if any(path.resolve()==root.resolve()or path.resolve().is_relative_to(root.resolve())for path in (build,directory,*evidence)):raise ValueError('Qwen projection outputs must be outside source artifacts')
    inputs,anchors=cases();loaded=weights(qwen.WEIGHTS,qwen.WEIGHT_SHA,qwen.WEIGHT_BYTES);directory.mkdir(parents=True);identities={};jobs=[]
    for op in ('q','k','v'):
        tensor=loaded[f'model.layers.0.self_attn.{op}_proj.weight'];path=directory/f'{op}.f16le'
        with path.open('xb')as stream:stream.write(tensor['data'])
        identities[str(path.resolve())]=projection.rms.probe.common.digest(path)
    for i,item in enumerate(inputs):
        item=copy.deepcopy(item);op=item.pop('op');bias=loaded[f'model.layers.0.self_attn.{op}_proj.bias'];weight=loaded[f'model.layers.0.self_attn.{op}_proj.weight'];item['bias_bits']=list(struct.unpack(f'<{item["outputs"]}H',bias['data']))
        anchors[item['id']].update(weight_sha256=weight['half_sha256'],bias_sha256=bias['half_sha256'])
        path=directory/f'qwen_{i+1}.json'
        with path.open('x')as stream:stream.write(json.dumps({'format':'toklet-metal-qwen-projection-input-v1','cases':[item]},separators=(',',':'))+'\n')
        identities[str(path.resolve())]=projection.rms.probe.common.digest(path);jobs.append((f'qwen_{i+1}',['--projection-input',str(path.resolve())],lambda e,item=item:validate(e,item,anchors[item['id']])))
    tiny,ta,overflow=projection.cases();di,da,do=decoder.cases()
    payloads=[('tiny_regression',{'format':'toklet-metal-projection-input-v1','cases':tiny},'--projection-input',lambda e:projection.validate(e,tiny,ta)),('tiny_overflow',{'format':'toklet-metal-projection-input-v1','cases':[overflow]},'--projection-input',projection.overflow_rejected)]
    payloads.extend((f'decoder_{i+1}',item,'--decoder-input',lambda e,item=item:decoder.validate(e,item,da[item['id']]))for i,item in enumerate(di))
    for label,payload,option,validator in payloads:
        path=directory/f'{label}.json'
        with path.open('x')as stream:stream.write(json.dumps(payload,separators=(',',':'))+'\n')
        identities[str(path.resolve())]=projection.rms.probe.common.digest(path);jobs.append((label,[option,str(path.resolve())],validator))
    drivers={str(Path(m.__file__).resolve()):projection.rms.probe.common.digest(Path(m.__file__))for name,m in list(sys.modules.items())if name.startswith('rust_metal_')and getattr(m,'__file__',None)};drivers[str(Path(__file__).resolve())]=projection.rms.probe.common.digest(Path(__file__))
    previous=jobs[-1][2]
    def final(e):
        result=previous(e)
        if cases()!=(inputs,{k:{a:v for a,v in anchor.items()if a not in('weight_sha256','bias_sha256')}for k,anchor in anchors.items()}):raise ValueError('Qwen frozen reference changed')
        for path,expected in {**identities,**drivers}.items():
            if projection.rms.probe.common.digest(Path(path))!=expected:raise ValueError('Qwen projection identity changed')
        if projection.rms.probe.common.digest(qwen.WEIGHTS)!=qwen.WEIGHT_SHA:raise ValueError('Qwen checkpoint changed during qualification')
        return result
    # Identity hash runs after this last child's GPU completion/readback handshake.
    jobs[-1]=(jobs[-1][0],jobs[-1][1],final)
    provenance={name:{k:v for k,v in tensor.items()if k!='data'}for name,tensor in loaded.items()}
    projection.rms.probe.run(output,build,extra_runs=jobs,extra_context={'input_sha256':identities,'qualification_driver_sha256':drivers,'weights':provenance,'checkpoint_sha256':qwen.WEIGHT_SHA,'checkpoint_bytes':qwen.WEIGHT_BYTES,'half_manifest_sha256':qwen.IDENTITIES[qwen.HALF],'source_manifest_sha256':qwen.IDENTITIES[qwen.SINGLE],'scope':'12 layer0 Qwen Q/K/V projections from saved normalized Half input, pinned BF16 weights converted once to Half, unchanged source gates. Same-binary tiny projections and nine full tiny decoders. No Qwen propagated layer/cache/session/full decoder, quality, performance or M4 closure.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True);a=p.parse_args();run(a.output,a.build_dir,a.input_dir)
if __name__=='__main__':main()
