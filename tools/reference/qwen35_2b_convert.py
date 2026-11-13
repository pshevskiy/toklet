"""Bounded affine4/G64 Qwen3.5-2B conversion with exact typed small tensors.

No model is constructed. A prior externally pinned parent-lineage certificate
and setup receipt are mandatory. Every saved matrix has independent and completed
upstream FP32 reconstruction witnesses; a separate cold audit checks saved bytes.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import granite_quantize as quant
import granite_quant_probe as probe
import granite_quant_runtime as runtime
import ministral3_convert as files
import qwen35_2b_layout as layout
import qwen35_2b_lineage as lineage
import qwen35_2b_parent as parent

need = layout.need


def sources():
    folder = Path(__file__).resolve().parent
    return sorted({*lineage.sources(), *probe.sources(), *(folder/name for name in (
        'qwen35_2b_convert.py', 'test_qwen35_2b_convert.py',
        'qwen35_2b_audit.py', 'test_qwen35_2b_audit.py'))})


def contract(source_directory, lineage_sha256, protocol_sha256, setup_sha256):
    need(type(source_directory) is str and Path(source_directory).is_absolute(), 'Absolute selected source directory required')
    plan = layout.selected_layout()
    return dict(format='toklet-qwen35-2b-affine-conversion-protocol-v1', source_directory=source_directory,
        source=dict(layout.SOURCE), source_bytes=layout.SOURCE_BYTES,
        lineage_certificate_sha256=layout.identifier(lineage_sha256),
        parent_protocol_sha256=layout.identifier(protocol_sha256),
        source_setup_receipt_sha256=layout.identifier(setup_sha256),
        storage=dict(layout.STORAGE), header_bytes=plan['origin']-8,
        header_sha256=hashlib.sha256(plan['prefix'][8:]).hexdigest(), container_bytes=plan['file_bytes'],
        recipe=dict(device='gpu', input_dtype='BF16', metadata_dtype='BF16', maximum_rows_per_tile=256,
                    maximum_columns=6144, quantization_runtime=runtime.VERSIONS),
        small='Exact115originalBF16_and18originalF32_tensors_no_cast_no_offset',
        independent_reconstruction='all_matrix_values_exact_completed_upstream_F32_bytes',
        expected_coverage=dict(matrix_tiles=3238, reconstructed_values=1881276416,
                               exact_small_tensors=133, excluded_tensors=297, journal_records=3855),
        engine_limit_bytes=6*1024**3, audit_limit_bytes=512*1024**2,
        helper_reserve_bytes=32*1024**2, disk_floor_bytes=2*1024**3, workspace_reserve_bytes=256*1024**2,
        model_numerics_quality_fit_performance_adoption_qualified=False)


def validate_contract(value):
    need(type(value) is dict, 'Frozen conversion protocol required')
    expected = contract(value.get('source_directory'), value.get('lineage_certificate_sha256'),
                        value.get('parent_protocol_sha256'), value.get('source_setup_receipt_sha256'))
    layout.saved.same(value, expected, 'Exact selected conversion protocol differs')
    return value


def component(plan, name, first, rows, raw):
    declaration = plan['header'][name]
    stride = (declaration['data_offsets'][1]-declaration['data_offsets'][0])//declaration['shape'][0]
    return dict(name=name, dtype=declaration['dtype'], shape=[rows, declaration['shape'][1]],
                offset=plan['origin']+declaration['data_offsets'][0]+first*stride,
                bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def convert_matrix(fd, output_fd, origin, span, plan, np, mx, event):
    name, shape = span['name'], span['shape']
    fields = plan['matrices'][name]['fields']
    original, oracle_hash, upstream_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    hashes = {key:hashlib.sha256() for key in fields}
    count = 0
    for first in range(0, shape[0], 256):
        rows = min(256, shape[0]-first)
        raw = files.read_exact(fd, rows*shape[1]*2, origin+span['start']+first*shape[1]*2)
        started = time.monotonic_ns()
        buffers = quant.quantize(raw, rows, shape[1], 4, 'gpu', np, mx)
        completed_quantize_ns = time.monotonic_ns()-started
        independent = quant.reconstruct_f32(buffers, rows, shape[1], 4, np)
        upstream = probe.upstream_f32(buffers, rows, shape[1], 4, 'gpu', np, mx)
        record = dict(event='matrix_tile', name=name, first_row=first, rows=rows, columns=shape[1], bits=4,
            source_offset=origin+span['start']+first*shape[1]*2, source_bytes=len(raw),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            fields={key:component(plan, item['name'], first, rows, buffers[key]) for key,item in fields.items()},
            independent_F32_sha256=hashlib.sha256(independent).hexdigest(),
            upstream_F32_sha256=hashlib.sha256(upstream).hexdigest(),
            reconstructed_values=rows*shape[1], reconstruction_equal=independent==upstream,
            component_readback=False, completed_quantize_readback_ns=completed_quantize_ns)
        if independent != upstream:
            event(record)
            need(False, 'Independent and upstream FP32 reconstructions differ')
        try:
            layout.writes.write_matrix_tile(output_fd, plan, name, first, rows, buffers)
        except BaseException:
            event(record)
            raise
        record['component_readback'] = True
        event(record)
        original.update(raw); oracle_hash.update(independent); upstream_hash.update(upstream)
        for key,value in buffers.items():
            hashes[key].update(value)
        count += 1
        del raw, buffers, independent, upstream
    stored = {}
    for key,item in fields.items():
        declaration = plan['header'][item['name']]
        start,end = declaration['data_offsets']
        actual = files.hash_range(output_fd, end-start, plan['origin']+start)
        need(actual == hashes[key].hexdigest(), 'Whole saved matrix component differs')
        stored[key] = dict(name=item['name'], dtype=declaration['dtype'], shape=declaration['shape'],
                           offset=plan['origin']+start, bytes=end-start, sha256=actual)
    return dict(name=name, shape=shape, bits=4, source_sha256=original.hexdigest(), fields=stored,
                independent_F32_sha256=oracle_hash.hexdigest(), upstream_F32_sha256=upstream_hash.hexdigest(),
                reconstructed_values=math.prod(shape), tiles=count, reconstruction_equal=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('environment-root', 'runtime-manifest', 'runtime-sha256', 'contract', 'contract-sha256',
                 'source-setup-receipt', 'lineage-certificate', 'fresh-output', 'fresh-evidence'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    output,evidence = Path(args.fresh_output),Path(args.fresh_evidence)
    files.reserve_output(output,evidence)
    result = dict(format='toklet-qwen35-2b-affine-conversion-v1', passed=False,
                  contract_sha256=args.contract_sha256,
                  model_numerics_quality_fit_performance_adoption_qualified=False)
    inputs, destination, completed, records = [],None,False,0
    partial,final,journal_path = output/'model.safetensors.partial',output/'model.safetensors',evidence/'records.jsonl'
    directories = [(p,files.directory_identity(p)) for p in (output,evidence)]
    def capture(path, maximum, expected):
        path=Path(path); raw,identity=files.capture_named(path,maximum,expected)
        inputs.append((path,maximum,expected,raw,identity))
        return raw
    def endpoints():
        for path,maximum,expected,raw,identity in inputs:
            actual,after=files.capture_named(path,maximum,expected)
            need(raw==actual and identity==after, 'Captured input endpoint replaced or mutated')
        need(all(files.directory_identity(p)==identity for p,identity in directories), 'Owned output/evidence directory replaced')
    try:
        contract_raw=capture(args.contract,256*1024,layout.identifier(args.contract_sha256))
        selected_contract=validate_contract(layout.saved.loads(contract_raw))
        source_dir=Path(selected_contract['source_directory'])
        need(source_dir.is_dir() and not source_dir.is_symlink(), 'Regular selected source directory required')
        source_directory_identity=files.directory_identity(source_dir)
        auth_raw=capture(args.source_setup_receipt,2*1024**2,selected_contract['source_setup_receipt_sha256'])
        auth=layout.saved.loads(auth_raw)
        need(type(auth) is dict and auth.get('passed') is True, 'Positive guarded selected source setup receipt required')
        cert_raw=capture(args.lineage_certificate,2*1024**2,selected_contract['lineage_certificate_sha256'])
        cert=parent.validate_certificate(layout.saved.loads(cert_raw),selected_contract['parent_protocol_sha256'])
        need(cert['selected_source'].get('path')==str(source_dir/'model.safetensors'), 'Lineage source path differs')
        journal=cert.get('journal')
        need(type(journal) is dict and type(journal.get('bytes')) is int and 0<journal['bytes']<=32*1024**2,
             'Bounded complete parent comparison journal required')
        journal_raw=capture(journal['path'],32*1024**2,layout.identifier(journal['sha256']))
        need(len(journal_raw)==journal['bytes'], 'Parent comparison journal extent differs')
        del journal_raw
        config_raw=capture(source_dir/'config.json',64*1024,layout.SOURCE_CONFIG_SHA256)
        child_config=layout.converted_config(config_raw); plan=layout.selected_layout()
        space=os.statvfs(output)
        need(space.f_bavail*space.f_frsize>=plan['file_bytes']+2*1024**3+256*1024**2,
             'Conversion must preserve disk floor and workspace reserve')
        runtime_raw=capture(args.runtime_manifest,2*1024**2,args.runtime_sha256)
        with layout.saved.SavedFile(source_dir/'model.safetensors',layout.SOURCE_BYTES) as source:
            need(source.before==cert['selected_source']['after_identity'], 'Lineage source descriptor endpoint differs')
            header=source.read(layout.SOURCE_HEADER_BYTES,8)
            spans=layout.source_spans(source.read(8),header,source.before[2]); origin=8+len(header)
            need(source.digest()==layout.SOURCE_SHA256, 'Whole selected source differs before conversion')
            endpoints()
            runtime.bind_environment(args.environment_root)
            admitted,endpoint=runtime.admit(Path(args.runtime_manifest),args.runtime_sha256,sources())
            import numpy as np
            import mlx.core as mx
            mx.set_cache_limit(0)
            destination=os.open(partial,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
            os.ftruncate(destination,plan['file_bytes']); layout.writes.write_all(destination,plan['prefix'],0)
            matrices,small,excluded=[],[],[]
            lineage_hashes={row['name']:row['source_sha256'] for row in cert['tensors']}
            small_recipe={row['name']:row for row in layout.selected_recipe()[1]}
            with journal_path.open('x') as journal_file:
                def event(value):
                    nonlocal records
                    journal_file.write(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n');journal_file.flush();records+=1
                for span in spans:
                    name=span['name']; size=span['end']-span['start']
                    if not span['selected']:
                        record=dict(name=name,source_sha256=source.digest(size,origin+span['start']),bytes=size,excluded=True)
                        excluded.append(record);event(dict(event='source_exclusion',**record));continue
                    if name in plan['matrices']:
                        record=convert_matrix(source.fd,destination,origin,span,plan,np,mx,event)
                        need(record['source_sha256']==lineage_hashes[name], 'Converted matrix differs from lineage source tensor')
                        matrices.append(record);event(dict(event='matrix_completed',**record));continue
                    raw=source.read(size,origin+span['start']);layout.validate_small(small_recipe[name],raw)
                    digest=hashlib.sha256(raw).hexdigest()
                    need(digest==lineage_hashes[name], 'Small original differs from lineage source tensor')
                    layout.write_small(destination,plan,small_recipe[name],raw)
                    spec=plan['header'][name]; offset=plan['origin']+spec['data_offsets'][0]
                    need(files.read_exact(destination,size,offset)==raw, 'Typed original small readback differs')
                    record=dict(name=name,dtype=span['dtype'],shape=span['shape'],bytes=size,
                                source_sha256=digest,offset=offset,sha256=digest,exact_bytes_equal=True)
                    small.append(record);event(dict(event='small_copied',**record));del raw
                os.fsync(journal_file.fileno())
            need(len(matrices)==187 and len(small)==133 and len(excluded)==297
                 and sum(row['reconstructed_values'] for row in matrices)==1881276416
                 and sum(row['tiles'] for row in matrices)==3238 and records==3855,
                 'Complete selected conversion coverage required')
            config_record=files.write_exclusive(output/'config.json',child_config)
            need(source.digest()==layout.SOURCE_SHA256, 'Whole selected source differs after conversion')
            source.check();endpoints()
            need(files.directory_identity(source_dir)==source_directory_identity,'Selected source directory replaced')
            os.fchmod(destination,0o444);os.fsync(destination)
            need(files.read_exact(destination,plan['origin'],0)==plan['prefix'], 'Saved child header differs')
            child_before=files.verify_named(destination,partial)
            child_sha=files.hash_range(destination,plan['file_bytes'])
            files.verify_named(destination,partial,child_before)
            endpoint_origins=endpoint();endpoints();source.check()
            files.verify_metadata_records([config_record])
            pins=layout.artifact_pins(child_config,plan,child_sha)
            with layout.saved.SavedFile(journal_path) as journal_file:
                journal_record=dict(path=str(journal_path),bytes=journal_file.before[2],sha256=journal_file.digest(),records=records)
            child_identity=files.promote_partial(destination,partial,final,child_before)
            result.update(passed=True,protocol=selected_contract,artifact_pins=pins,
                contract=dict(path=args.contract,bytes=len(contract_raw),sha256=args.contract_sha256),
                source=dict(layout.SOURCE,path=str(source.path),before_identity=source.before,after_identity=source.check()),
                lineage_certificate=dict(path=args.lineage_certificate,bytes=len(cert_raw),sha256=selected_contract['lineage_certificate_sha256']),
                source_setup_receipt=dict(path=args.source_setup_receipt,bytes=len(auth_raw),sha256=selected_contract['source_setup_receipt_sha256']),
                runtime=dict(path=args.runtime_manifest,bytes=len(runtime_raw),sha256=args.runtime_sha256),
                admitted_runtime=admitted,endpoint_origins=endpoint_origins,journal=journal_record,
                matrices=matrices,small=small,exclusions=excluded,config=config_record,
                matrix_count=187,exact_small_count=133,excluded_count=297,matrix_tiles=3238,
                independently_reconstructed_values=1881276416,
                child=dict(path=str(final),before_identity=list(child_identity),after_identity=list(child_identity)),
                complete_positive_predicates=dict(source_lineage=True,whole_source_SHA256=True,
                    all_component_readback=True,independent_upstream_F32_equal=True,
                    exact_original_small_bytes=True,whole_child_SHA256=True,endpoint_unchanged=True))
            manifest=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode()
            files.write_exclusive(evidence/'manifest.json',manifest)
            identity=layout.child_manifest(pins,selected_contract['lineage_certificate_sha256'],hashlib.sha256(manifest).hexdigest())
            files.write_exclusive(output/'child-manifest.json',(json.dumps(identity,indent=2,allow_nan=False)+'\n').encode())
            files.verify_metadata_records([config_record])
            files.verify_named(destination,final,child_identity)
            completed=True
    except BaseException as error:
        result.update(passed=False,error_type=type(error).__name__,error=str(error)[:2048])
        target=evidence/('failure.json' if (evidence/'manifest.json').exists() else 'manifest.json')
        files.write_exclusive(target,(json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    finally:
        if destination is not None:
            try:
                if not completed:
                    files.preserve_failed_partial(destination,partial,final)
            finally:
                os.close(destination)
    print(json.dumps(dict(passed=completed,matrix_count=len(result.get('matrices',[])),
                          small_count=len(result.get('small',[])),model_constructed=False)))
    return 0 if completed else 1


if __name__=='__main__':
    raise SystemExit(main())
