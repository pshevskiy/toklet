"""Reproduce all pinned affine8 matrices while authenticating both BF16 parents."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import minicpm5 as binding
import minicpm5_matrix_stream as conversion
import minicpm5_parent as parent
import minicpm5_quant_runtime as runtime
import minicpm5_quantize as quant

need = parent.need


def sources():
    folder = Path(__file__).resolve().parent
    return [folder/name for name in ['minicpm5_matrix_lineage.py', 'minicpm5_matrix_stream.py',
            'test_minicpm5_matrix_stream.py', 'minicpm5_quantize.py', 'test_minicpm5_quantize.py',
            'minicpm5_quant_runtime.py', 'minicpm5_parent.py', 'minicpm5.py']]


def transfer(shard, output, child_source, quantizer):
    name, size, pin, prefix, header, norms = shard
    url = f'https://huggingface.co/{parent.PARENT}/resolve/{parent.REVISION}/{name}?download=true&toklet_lineage={uuid.uuid4().hex}'
    command = ['/usr/bin/curl','-q','--fail','--location','--proto','=https',
               '--proto-redir','=https','--connect-timeout','20','--max-time','1200',
               '--max-filesize',str(size),'--no-progress-meter','-H','Accept-Encoding: identity',
               '--dump-header',str(output/(name+'.headers')),'--output','-',
               '--write-out','%{stderr}\nTOKLET_CURL_JSON%{json}\n',url]
    started = time.monotonic_ns();authenticated = None;converted = None;error = None;resource = None
    with (output/(name+'.curl.log')).open('x') as log, \
         (output/(name+'.tensors.jsonl')).open('x') as tensors, \
         (output/(name+'.conversion.jsonl')).open('x') as journal:
        child = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log)
        try:
            spans = parent.plan(prefix,header,size)
            wrapped = conversion.ConvertedStream(child.stdout,8+len(header),spans,
                                                  child_source,quantizer,journal)
            authenticated = parent.authenticate(wrapped,prefix,header,size,pin,norms,tensors)
            converted = wrapped.finish(size)
        except BaseException as exception:
            error = type(exception).__name__+': '+str(exception)
        finally:
            child.stdout.close()
            if error is not None and child.poll() is None:
                child.terminate()
            deadline = time.monotonic()+15
            while child.returncode is None:
                pid,status,usage = os.wait4(child.pid,os.WNOHANG)
                if pid:
                    child.returncode = os.waitstatus_to_exitcode(status);resource = usage;break
                if time.monotonic()>deadline:
                    child.kill();_,status,resource = os.wait4(child.pid,0)
                    child.returncode = os.waitstatus_to_exitcode(status);break
                time.sleep(.05)
    completed = time.monotonic_ns()
    matches = re.findall(r'^TOKLET_CURL_JSON(.+)$',(output/(name+'.curl.log')).read_text(),re.M)
    transport = {}
    try:
        need(len(matches)==1,'Missing/duplicate lineage transfer receipt')
        transport = json.loads(matches[0],object_pairs_hook=parent.unique)
    except ValueError as exception:
        error = (error+'; ' if error else '')+str(exception)
    authenticated_pass = (error is None and authenticated is not None and converted is not None
                          and child.returncode==0 and type(transport.get('http_code')) is int
                          and transport['http_code']==200
                          and type(transport.get('size_download')) is int and transport['size_download']==size
                          and type(transport.get('ssl_verify_result')) is int and transport['ssl_verify_result']==0
                          and resource is not None)
    result = dict(passed=bool(authenticated_pass and converted['passed']),
                  source_authentication_passed=authenticated_pass,name=name,pid=child.pid,
                  command=command,started_ns=started,completed_ns=completed,
                  exit_code=child.returncode,error=error,http_code=transport.get('http_code'),
                  size_download=transport.get('size_download'),ssl_verify_result=transport.get('ssl_verify_result'),
                  final_wait4_peak_rss_bytes=resource.ru_maxrss if resource else None,
                  payload_retained=False,result=authenticated,conversion=converted)
    (output/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    # A conversion mismatch is retained, and the second source is still audited.
    # An unauthenticated/truncated transfer cannot contribute to qualification.
    need(authenticated_pass,'Lineage source authentication failed; primary outcome retained: '+name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['environment_root','metadata_directory','child_safetensors','runtime_manifest',
                 'runtime_sha256','contract','contract_sha256','fresh_output']:
        parser.add_argument('--'+name.replace('_','-'),required=True)
    args = parser.parse_args();output = Path(args.fresh_output);output.mkdir()
    runtime.bind_environment(args.environment_root)
    raw = runtime.common.captured(Path(args.contract),2*1024**2,args.contract_sha256)
    contract = json.loads(raw,object_pairs_hook=parent.unique)
    need(contract['format']=='toklet-minicpm5-matrix-lineage-contract-v1'
         and contract['prospective_before_quantizer_or_parent_matrix_outcomes'] is True
         and contract['child_sha256']==parent.CHILD_SHA
         and contract['parent_repository']==parent.PARENT and contract['parent_revision']==parent.REVISION
         and [(r['rfilename'],r['size'],r['lfs']['sha256']) for r in contract['parent_objects']]==parent.OBJECTS
         and contract['recipe']==dict(runtime='MLX0.32.3',device='gpu',input='BF16 bits preserved; no FP16 cast',
                                     group_size=64,bits=8,mode='affine',maximum_tile_rows=256,cache_limit_bytes=0)
         and contract['source_audit_working_set_bytes']==536870912
         and contract['engine_ceiling_bytes']==6442450944 and contract['disk_floor_bytes']==2147483648,
         'Frozen source conversion recipe or limits differ')
    admitted,endpoint = runtime.admit(Path(args.runtime_manifest),args.runtime_sha256,sources())
    import numpy as np
    import mlx.core as mx
    mx.set_cache_limit(0)
    shards = parent.controls(Path(args.metadata_directory),Path(args.child_safetensors))
    source = conversion.Child(args.child_safetensors,binding.logical_shapes())
    try:
        records = []
        def quantizer(raw,rows,columns):
            return quant.quantize(raw,rows,columns,'gpu',np,mx)
        for shard in shards:
            row = transfer(shard,output,source,quantizer);records.append(row)
            print(json.dumps(dict(parent_object_authenticated=row['name'],
                                  matrices=row['conversion']['matrix_count'],
                                  exact_conversion=row['conversion']['passed'])),flush=True)
        source.verify()
    finally:
        source.close()
    final = endpoint()
    matrices = [m for row in records for m in row['conversion']['matrices']]
    need(len(matrices)==296 and len({m['name'] for m in matrices})==296
         and sum(row['result']['exact_child_norms'] for row in records)==85
         and sum(row['conversion']['norm_tensors'] for row in records)==85
         and sum(m['comparison']['weight']['bytes'] for m in matrices)==2516582400
         and sum(row['conversion']['tiles'] for row in records)==4128,
         'Full source matrix/norm/tile/byte coverage differs')
    passed = len(records)==2 and all(row['passed'] for row in records)
    result = dict(format='toklet-minicpm5-matrix-lineage-v1',passed=passed,
                  parent_repository=parent.PARENT,parent_revision=parent.REVISION,
                  child_sha256=parent.CHILD_SHA,authenticated_parent_bytes=sum(row['result']['bytes'] for row in records),
                  parent_objects=records,matrices=296,physical_quantized_arrays=888,
                  quantized_source_values=2516582400,exact_norms=85,tiles=4128,
                  recipe=contract['recipe'],contract_sha256=args.contract_sha256,
                  admitted_runtime=admitted,endpoint_origins=final,
                  retained_parent_payload_bytes=0,full_parent_identity_verified=True,
                  exact_matrix_reconstruction_verified=passed,conversion_recipe_reproduced=passed,
                  publisher_exact_binary_version_authenticated=False,
                  model_inference_executed=False,model_quality_adoption_qualified=False,
                  performance_or_overhead_qualified=False,physical_iphone_qualified=False)
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['parent_objects','admitted_runtime','endpoint_origins','recipe']}))
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main())
