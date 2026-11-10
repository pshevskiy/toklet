"""Seal exact conversion witnesses; discarded BF16 payload is not rehashed."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import check_minicpm5_parent as prior
import minicpm5 as binding
import minicpm5_matrix_stream as conversion
import minicpm5_parent as source

need, integer, sha, same = prior.need, prior.integer, prior.sha, prior.same
LIMIT = 512*1024**2


def read(path, maximum=32*1024**2):
    with path.open('rb') as stream:raw = stream.read(maximum+1)
    need(len(raw)<=maximum,'Oversize retained conversion witness')
    return raw


def document(path):
    return json.loads(read(path),object_pairs_hook=source.unique)


def validate_owner(owned):
    need(type(owned) is dict and owned.get('passed') is True
         and owned.get('owned_group_absent') is True and owned.get('source_inputs_unchanged') is True
         and type(owned.get('exit_code')) is int and owned['exit_code']==0
         and owned.get('failure') is None and owned.get('final_physical_peak_available') is False
         and type(owned.get('sample_errors')) is list and not owned['sample_errors']
         and integer(owned.get('started_ns'),1) and integer(owned.get('completed_ns'),1)
         and owned['started_ns']<owned['completed_ns']
         and integer(owned.get('minimum_observed_free_bytes'))
         and owned['minimum_observed_free_bytes']>=2*1024**3,
         'Source conversion owner is not successful/closed/bounded')
    for key in ['final_auditor_wait4_peak_rss_bytes','max_sampled_aggregate_rss_bytes',
                'max_sampled_aggregate_physical_bytes','max_sampled_aggregate_lifetime_physical_bytes']:
        need(integer(owned.get(key),1) and owned[key]<=LIMIT,'Conversion resource bound differs')


def validate_runtime(admitted, final, external, expected_sha, cache_prefix, environment_root):
    need(type(external) is dict and set(external)=={'format','versions','files'}
         and external['format']=='toklet-minicpm5-quant-runtime-v1'
         and external['versions']=={'numpy':'2.3.2','mlx':'0.32.3'}
         and type(external['files']) is dict and external['files']
         and any(Path(p).name=='mlx.metallib' for p in external['files'])
         and any(Path(p).name=='libmlx.dylib' for p in external['files']),
         'External numerical runtime schema differs')
    for path,pin in external['files'].items():
        p=Path(path)
        need(p.is_absolute() and type(pin) is dict and set(pin)=={'bytes','sha256'}
             and integer(pin['bytes']) and pin['bytes']<=LIMIT and sha(pin['sha256'])
             and not p.is_symlink() and p.is_file() and p.stat().st_size==pin['bytes']
             and source.digest(p)==pin['sha256'],'Pinned numerical runtime file changed')
    need(type(admitted) is dict and set(admitted)=={'manifest_sha256','files','versions',
         'pycache_prefix','imports','initial_origins'}
         and admitted['manifest_sha256']==expected_sha
         and type(admitted['files']) is int and admitted['files']==len(external['files'])
         and admitted['versions']==external['versions']
         and admitted['pycache_prefix']==cache_prefix and Path(cache_prefix).is_absolute()
         and type(admitted['imports']) is dict and set(admitted['imports'])=={'numpy','mlx.core','mlx.nn'},
         'Admitted runtime count/version/cache/import identity differs')
    imports=admitted['imports'];numpy=Path(imports['numpy']);site=numpy.parent.parent
    need(numpy.name=='__init__.py' and numpy.parent.name=='numpy'
         and site.name=='site-packages' and site.is_relative_to(environment_root/'.venv')
         and imports['mlx.nn']==str(site/'mlx/nn/__init__.py')
         and Path(imports['mlx.core']).parent==site/'mlx'
         and Path(imports['mlx.core']).name.startswith('core.')
         and Path(imports['mlx.core']).suffix=='.so'
         and all(p in external['files'] for p in imports.values()),
         'Selected numerical import roots differ')
    for origins in [admitted['initial_origins'],final]:
        need(type(origins) is dict and set(origins)=={'modules','non_system_native_images','system_native_images'}
             and type(origins['modules']) is dict and type(origins['non_system_native_images']) is dict
             and type(origins['system_native_images']) is list
             and all(type(n) is str and type(p) is str and p in external['files'] for n,p in origins['modules'].items())
             and all(origins['modules'].get(n)==p for n,p in imports.items())
             and all(p in external['files'] and same(pin,external['files'][p])
                     for p,pin in origins['non_system_native_images'].items())
             and imports['mlx.core'] in origins['non_system_native_images']
             and any(Path(p).name=='libmlx.dylib' for p in origins['non_system_native_images'])
             and origins['system_native_images']
             and all(type(p) is str and str(Path(p).resolve()).startswith(('/System/','/usr/lib/'))
                     for p in origins['system_native_images']),
             'Actual numerical module/native image origins differ')


def validate_samples(samples, started, command, owned, curl_pids):
    need(type(started) is dict and set(started)=={'pid','started_ns','command'}
         and integer(started['pid'],1) and integer(started['started_ns'],1)
         and started['started_ns']==owned['started_ns']
         and same(started['command'],command) and type(samples) is list and samples,
         'Owned producer/process sample association differs')
    allowed={started['pid'],*curl_pids};previous=owned['started_ns']
    peaks={key:0 for key in ['resident_size','phys_footprint','lifetime_max_phys_footprint']}
    for sample in samples:
        need(type(sample) is dict and set(sample)=={'observed_ns','processes'}
             and integer(sample['observed_ns'],1) and previous<=sample['observed_ns']<=owned['completed_ns']
             and type(sample['processes']) is list and sample['processes'],
             'Invalid source resource sample interval')
        previous=sample['observed_ns'];pids=set();total={key:0 for key in peaks}
        for row in sample['processes']:
            need(type(row) is dict and set(row)=={'pid',*peaks}
                 and integer(row['pid'],1) and row['pid'] in allowed and row['pid'] not in pids
                 and all(integer(row[key]) for key in peaks),'Unowned/untyped source resource sample')
            pids.add(row['pid'])
            for key in peaks:total[key]+=row[key]
        need(all(v<=LIMIT for v in total.values()),'Live source process sum exceeds limit')
        for key in peaks:peaks[key]=max(peaks[key],total[key])
    for key,field in [('resident_size','max_sampled_aggregate_rss_bytes'),
                      ('phys_footprint','max_sampled_aggregate_physical_bytes'),
                      ('lifetime_max_phys_footprint','max_sampled_aggregate_lifetime_physical_bytes')]:
        need(peaks[key]==owned[field],'Saved resource aggregate differs from raw samples')
    need(type(owned.get('sampled_observations')) is int and owned['sampled_observations']==len(samples),
         'Resource sample coverage differs')


def validate_transfer(row, shard, owned, destination):
    fields = {'passed','source_authentication_passed','name','pid','command','started_ns',
              'completed_ns','exit_code','error','http_code','size_download','ssl_verify_result',
              'final_wait4_peak_rss_bytes','payload_retained','result','conversion'}
    name,size,_,_,_,_ = shard
    need(type(row) is dict and set(row)==fields and row['passed'] is True
         and row['source_authentication_passed'] is True and row['name']==name
         and integer(row['pid'],1) and type(row['exit_code']) is int and row['exit_code']==0
         and row['error'] is None and row['payload_retained'] is False
         and type(row['http_code']) is int and row['http_code']==200
         and integer(row['size_download'],1) and row['size_download']==size
         and type(row['ssl_verify_result']) is int and row['ssl_verify_result']==0,
         'Partial/failed/retained source transfer differs')
    need(integer(row['started_ns'],1) and integer(row['completed_ns'],1)
         and owned['started_ns']<=row['started_ns']<row['completed_ns']<=owned['completed_ns']
         and integer(row['final_wait4_peak_rss_bytes'],1)
         and row['final_wait4_peak_rss_bytes']<=LIMIT,'Transfer clock/resource enclosure differs')
    command = row['command']
    expected = ['/usr/bin/curl','-q','--fail','--location','--proto','=https','--proto-redir',
                '=https','--connect-timeout','20','--max-time','1200','--max-filesize',str(size),
                '--no-progress-meter','-H','Accept-Encoding: identity','--dump-header']
    need(type(command) is list and len(command)==24 and all(type(x) is str for x in command)
         and command[:18]==expected and Path(command[18]).name==name+'.headers'
         and Path(command[18]).parent.resolve()==destination.resolve()
         and command[19:23]==['--output','-','--write-out','%{stderr}\nTOKLET_CURL_JSON%{json}\n']
         and re.fullmatch(re.escape(f'https://huggingface.co/{source.PARENT}/resolve/{source.REVISION}/{name}')
                          +r'\?download=true&toklet_lineage=[0-9a-f]{32}',command[-1]),
         'Exact bounded source command differs')
    prior.validate_result(row['result'],shard)


def fields(actual, expected):
    need(type(actual) is dict and set(actual)=={'weight','scales','biases'},
         'Complete affine field coverage differs')
    for key, value in actual.items():
        need(type(value) is dict and set(value)=={'bytes','mismatching_bytes',
             'reconstructed_sha256','expected_sha256'}
             and integer(value['bytes'],1) and value['bytes']==len(expected[key])
             and type(value['mismatching_bytes']) is int and value['mismatching_bytes']==0
             and value['expected_sha256']==hashlib.sha256(expected[key]).hexdigest()
             and value['reconstructed_sha256']==value['expected_sha256'],
             'Exact child code/scale/bias byte witness differs')


def validate_conversion(result, journal, shard, child):
    name,size,_,prefix,header,_ = shard
    spans = source.plan(prefix,header,size);matrices = [s for s in spans if len(s['shape'])==2]
    need(type(result) is dict and set(result)=={'matrix_count','tiles','norm_tensors','matrices',
         'passed','object_authenticated','retained_payload_bytes'}
         and result['passed'] is True and result['object_authenticated'] is False
         and type(result['retained_payload_bytes']) is int and result['retained_payload_bytes']==0
         and type(result['matrix_count']) is int and result['matrix_count']==len(matrices)
         and type(result['norm_tensors']) is int and result['norm_tensors']==len(spans)-len(matrices)
         and type(result['tiles']) is int and result['tiles']==sum((s['shape'][0]+255)//256 for s in matrices)
         and type(result['matrices']) is list and len(result['matrices'])==len(matrices),
         'Conversion report coverage/qualification differs')
    need(type(journal) is bytes and journal and journal.endswith(b'\n'),
         'Complete bounded conversion journal required')
    entries = iter(json.loads(line,object_pairs_hook=source.unique) for line in journal.splitlines())
    phases = 0;tile_count = 0;origin = len(prefix+header)
    for span, report in zip(matrices,result['matrices']):
        need(type(report) is dict and set(report)=={'name','shape','comparison','passed'}
             and report['name']==span['name'] and same(report['shape'],span['shape'])
             and report['passed'] is True,'Matrix report identity/shape differs')
        hashes = {key:hashlib.sha256() for key in ['weight','scales','biases']}
        byte_counts = {key:0 for key in hashes}
        for first in range(0,span['shape'][0],256):
            rows = min(256,span['shape'][0]-first);columns = span['shape'][1]
            try:event = next(entries)
            except StopIteration:raise ValueError('Missing conversion tile') from None
            need(type(event) is dict and set(event)=={'event','object_authenticated','name',
                 'first_row','rows','columns','parent_offset','parent_bytes','parent_tile_sha256',
                 'child_read_ns','completed_quantize_readback_ns','compare_hash_ns','comparison'}
                 and event['event']=='matrix_tile' and event['object_authenticated'] is False
                 and event['name']==span['name'] and sha(event['parent_tile_sha256']),
                 'Conversion tile identity/parent digest differs')
            for key,value in [('first_row',first),('rows',rows),('columns',columns),
                              ('parent_offset',origin+span['start']+first*columns*2),
                              ('parent_bytes',rows*columns*2)]:
                need(type(event[key]) is int and event[key]==value,'Tile source/row coverage differs')
            for key in ['child_read_ns','completed_quantize_readback_ns','compare_hash_ns']:
                need(integer(event[key],1),'Invalid completed conversion phase clock');phases += event[key]
            expected = child.tile(span['name'],first,rows)
            fields(event['comparison'],expected)
            for key in hashes:
                hashes[key].update(expected[key]);byte_counts[key] += len(expected[key])
            tile_count += 1
        need(type(report['comparison']) is dict and set(report['comparison'])==set(hashes),
             'Matrix aggregate field coverage differs')
        for key, digest in hashes.items():
            row = report['comparison'][key]
            need(type(row) is dict and set(row)=={'bytes','mismatching_bytes','reconstructed_sha256','expected_sha256'}
                 and type(row['bytes']) is int and row['bytes']==byte_counts[key]
                 and type(row['mismatching_bytes']) is int and row['mismatching_bytes']==0
                 and row['expected_sha256']==digest.hexdigest()
                 and row['reconstructed_sha256']==digest.hexdigest(),'Complete child matrix digest differs')
        try:completed = next(entries)
        except StopIteration:raise ValueError('Missing completed matrix journal') from None
        need(same(completed,dict(event='matrix_completed',object_authenticated=False,**report)),
             'Matrix journal/completed report differs')
    need(next(entries,None) is None and tile_count==result['tiles'],'Trailing/missing matrix journal rows')
    return phases


def bind_single_tile_sources(journal, tensors):
    # A single tile's BF16 digest is independently bound by the authenticated
    # tensor digest. Multiple tile hashes cannot reconstruct that SHA-256.
    required={t['name']:t['sha256'] for t in tensors
              if len(t['shape'])==2 and t['shape'][0]<=256}
    seen=set()
    for line in journal.splitlines():
        row=json.loads(line,object_pairs_hook=source.unique)
        if row.get('event')=='matrix_tile' and row.get('name') in required:
            need(row['name'] not in seen and row.get('first_row')==0
                 and row.get('parent_tile_sha256')==required[row['name']],
                 'Single-tile source/tensor digest differs')
            seen.add(row['name'])
    need(seen==set(required),'Single-tile source digest coverage differs')


def check(folder, metadata_directory, child_path, owned_path, freeze_path):
    owned = document(owned_path);validate_owner(owned)
    freeze = document(freeze_path)
    need(type(freeze.get('source_sha256')) is dict and freeze['source_sha256']
         and all(sha(h) and source.digest(Path(p))==h for p,h in freeze['source_sha256'].items()),
         'Frozen source/runtime/contract/runner changed')
    command = freeze.get('command')
    need(type(command) is list and len(command)==22 and all(type(p) is str for p in command)
         and command[1:4]==['-s','-B','-X'] and command[4].startswith('pycache_prefix=')
         and Path(command[5]).name=='minicpm5_matrix_lineage.py',
         'Admitted producer command differs')
    options = dict(zip(command[6::2],command[7::2]))
    need(len(options)==8 and set(options)=={'--environment-root','--metadata-directory',
         '--child-safetensors','--runtime-manifest','--runtime-sha256','--contract',
         '--contract-sha256','--fresh-output'}
         and Path(options['--metadata-directory']).resolve()==metadata_directory.resolve()
         and Path(options['--child-safetensors']).resolve()==child_path.resolve(),
         'Frozen source inputs differ')
    need(source.digest(Path(options['--runtime-manifest']))==options['--runtime-sha256']
         and source.digest(Path(options['--contract']))==options['--contract-sha256'],
         'Admitted runtime or prospective contract changed')
    manifest = document(folder/'manifest.json')
    names = {'format','passed','parent_repository','parent_revision','child_sha256',
             'authenticated_parent_bytes','parent_objects','matrices','physical_quantized_arrays',
             'quantized_source_values','exact_norms','tiles','recipe','contract_sha256',
             'admitted_runtime','endpoint_origins','retained_parent_payload_bytes',
             'full_parent_identity_verified','exact_matrix_reconstruction_verified',
             'conversion_recipe_reproduced','publisher_exact_binary_version_authenticated',
             'model_inference_executed','model_quality_adoption_qualified',
             'performance_or_overhead_qualified','physical_iphone_qualified'}
    need(type(manifest) is dict and set(manifest)==names
         and manifest['format']=='toklet-minicpm5-matrix-lineage-v1'
         and all(manifest[k] is True for k in ['passed','full_parent_identity_verified',
                  'exact_matrix_reconstruction_verified','conversion_recipe_reproduced'])
         and all(manifest[k] is False for k in ['publisher_exact_binary_version_authenticated',
                  'model_inference_executed','model_quality_adoption_qualified',
                  'performance_or_overhead_qualified','physical_iphone_qualified'])
         and manifest['parent_repository']==source.PARENT and manifest['parent_revision']==source.REVISION
         and manifest['child_sha256']==source.CHILD_SHA
         and manifest['contract_sha256']==options['--contract-sha256'],
         'Completed source conversion identity/qualification differs')
    for key,value in [('authenticated_parent_bytes',sum(row[1] for row in source.OBJECTS)),
                      ('matrices',296),('physical_quantized_arrays',888),('quantized_source_values',2516582400),
                      ('exact_norms',85),('tiles',4128),('retained_parent_payload_bytes',0)]:
        need(type(manifest[key]) is int and manifest[key]==value,'Whole conversion coverage differs')
    contract = document(Path(options['--contract']))
    need(same(manifest['recipe'],contract['recipe'])
         and manifest['admitted_runtime']['manifest_sha256']==options['--runtime-sha256'],
         'Runtime/recipe association differs')
    validate_runtime(manifest['admitted_runtime'],manifest['endpoint_origins'],
                     document(Path(options['--runtime-manifest'])),options['--runtime-sha256'],
                     command[4].removeprefix('pycache_prefix='),Path(options['--environment-root']))
    shards = source.controls(metadata_directory,child_path)
    need(type(manifest['parent_objects']) is list and len(manifest['parent_objects'])==len(shards),
         'Missing complete source object')
    evidence = [owned_path,freeze_path,folder/'manifest.json'];pids = set();previous = owned['started_ns'];phases = 0
    child = conversion.Child(child_path,binding.logical_shapes())
    try:
        for row, shard in zip(manifest['parent_objects'],shards):
            validate_transfer(row,shard,owned,Path(options['--fresh-output']))
            need(previous<=row['started_ns'] and row['pid'] not in pids,
                 'Transfer intervals overlap or reuse a PID')
            previous = row['completed_ns'];pids.add(row['pid']);name = row['name']
            need(same(row,document(folder/(name+'.json'))),'Per-object completed report differs')
            tensors = [json.loads(line,object_pairs_hook=source.unique)
                       for line in read(folder/(name+'.tensors.jsonl')).splitlines()]
            need(same(tensors,[dict(event='tensor_read',object_authenticated=False,**t)
                               for t in row['result']['tensors']]),'Tensor journal differs')
            conversion_journal=read(folder/(name+'.conversion.jsonl'))
            phase = validate_conversion(row['conversion'],conversion_journal,shard,child)
            bind_single_tile_sources(conversion_journal,row['result']['tensors'])
            need(phase<=row['completed_ns']-row['started_ns'],'Disjoint conversion clocks exceed transfer lifetime')
            phases += phase
            log = read(folder/(name+'.curl.log')).decode();receipts = re.findall(r'^TOKLET_CURL_JSON(.+)$',log,re.M)
            need(len(receipts)==1,'Missing/duplicate raw transport receipt')
            transport = json.loads(receipts[0],object_pairs_hook=source.unique)
            need(type(transport.get('exitcode')) is int and transport['exitcode']==0
                 and all(same(transport.get(k),row[k]) for k in ['http_code','size_download','ssl_verify_result']),
                 'Raw curl/transport result differs')
            responses = re.split(r'(?m)^HTTP/\S+ ',read(folder/(name+'.headers')).decode('latin1'))
            need(len(responses)>1 and re.match(r'200(?:\r?\n| )',responses[-1])
                 and re.search(r'(?im)^content-range:',responses[-1]) is None,
                 'Final source headers are not a complete object')
            lengths = re.findall(r'(?im)^content-length:\s*(\d+)\s*$',responses[-1])
            need(not lengths or (len(lengths)==1 and int(lengths[0])==shard[1]),
                 'Final complete source size differs')
            evidence.extend(folder/(name+suffix) for suffix in ['.json','.headers','.curl.log','.tensors.jsonl','.conversion.jsonl'])
        child.verify()
    finally:child.close()
    curl_rss = [r['final_wait4_peak_rss_bytes'] for r in manifest['parent_objects']]
    need(same(owned.get('final_curl_rss_bytes'),curl_rss)
         and owned['final_auditor_wait4_peak_rss_bytes']+sum(curl_rss)<=LIMIT,
         'Conservative final source process RSS sum exceeds bound')
    owner_started=owned_path.with_name('auth-owner-1.json')
    raw_samples=owned_path.with_name('auth-memory-samples-1.json')
    validate_samples(document(raw_samples),document(owner_started),command,owned,pids)
    evidence.extend([owner_started,raw_samples,Path(options['--runtime-manifest']),Path(options['--contract'])])
    return dict(format='toklet-minicpm5-lineage-witness-seal-v1',passed=True,
                exact_matrix_reconstruction_verified=True,matrices=296,quantized_arrays=888,
                copied_norms=85,tiles=4128,completed_conversion_phase_ns=phases,
                evidence_sha256={str(p):source.digest(p) for p in evidence},
                checker_sha256={str(Path(__file__)):source.digest(Path(__file__)),
                                str(Path(prior.__file__)):source.digest(Path(prior.__file__))},
                parent_payload_rehashed_offline=False,quantization_reexecuted_offline=False,
                child_bytes_independently_rehashed=True,model_quality_adoption_qualified=False,
                scope='Source/transport/runtime/owner and exact child row-hash witness seal. Discarded parent/tile digests rely on the checked full-stream producer, not offline payload replay or Merkle proofs.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['capture','metadata_directory','child','owned','freeze','fresh_seal']:
        parser.add_argument(name,type=Path)
    args = parser.parse_args();result = check(args.capture,args.metadata_directory,args.child,args.owned,args.freeze)
    with args.fresh_seal.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['evidence_sha256','checker_sha256']}))
