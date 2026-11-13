"""Cold saved-byte audit of the typed 2B conversion and ordered witnesses.

This verifies actual source/child bytes and reconciles every recorded completed
independent/upstream FP32 equality. It does not execute those numerical runtimes
again, or qualify a model graph, quality, fit, performance or managed adoption.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import qwen35_2b_convert as convert
import qwen35_2b_layout as layout
import qwen35_2b_parent as parent

need = layout.need


def journal_rows(saved):
    """Bounded lines over a held identity-checked descriptor."""
    need(saved.before[2] <= 32*1024**2, 'Bounded conversion journal required')
    pending = b''
    for first in range(0,saved.before[2],1024**2):
        pending += saved.read(min(1024**2,saved.before[2]-first),first)
        while b'\n' in pending:
            line,pending = pending.split(b'\n',1)
            need(0<len(line)<=64*1024, 'Bounded nonempty journal line required')
            value=layout.saved.loads(line)
            need(type(value) is dict,'Typed conversion journal record required')
            yield value
        need(len(pending)<=64*1024,'Oversized unfinished journal line')
    need(not pending,'Journal must end with a complete newline')
    saved.check()


def reconcile_tile(record,plan,span,first,rows,source,child,origin):
    name,columns=span['name'],span['shape'][1]
    need(record.get('event')=='matrix_tile' and record.get('name')==name,'Ordered complete matrix tiles required')
    for key,expected in dict(first_row=first,rows=rows,columns=columns,bits=4,
        source_offset=origin+span['start']+first*columns*2,source_bytes=rows*columns*2,
        reconstructed_values=rows*columns,reconstruction_equal=True,component_readback=True).items():
        layout.saved.same(record.get(key),expected,'Complete saved matrix tile witness differs')
    raw=source.read(rows*columns*2,record['source_offset'])
    need(hashlib.sha256(raw).hexdigest()==record.get('source_sha256'),'Actual source tile hash differs')
    fields=record.get('fields')
    need(type(fields) is dict and set(fields)=={'weight','scales','biases'},'All actual packed fields required')
    for key,field in plan['matrices'][name]['fields'].items():
        declaration=plan['header'][field['name']]
        size=rows*field['row_stride'];offset=plan['origin']+declaration['data_offsets'][0]+first*field['row_stride']
        raw=child.read(size,offset)
        expected=dict(name=field['name'],dtype=declaration['dtype'],shape=[rows,declaration['shape'][1]],
                      offset=offset,bytes=size,sha256=hashlib.sha256(raw).hexdigest())
        layout.saved.same(fields[key],expected,'Actual saved component witness differs')
        if key!='weight':
            need(all(word&0x7f80!=0x7f80 for (word,) in struct.iter_unpack('<H',raw)), 'Nonfinite saved BF16 metadata')
    witness=layout.identifier(record.get('independent_F32_sha256'))
    need(witness==layout.identifier(record.get('upstream_F32_sha256')),'Recorded FP32 witness inequality')


def audit(model,manifest_sha256,conversion_manifest,source_dir):
    inputs=[]
    def capture(path,maximum,expected):
        raw,identity=convert.files.capture_named(path,maximum,expected)
        inputs.append((path,maximum,expected,raw,identity))
        return raw
    artifact_raw=capture(model/'child-manifest.json',64*1024,manifest_sha256)
    artifact=layout.capture_child_manifest(artifact_raw,manifest_sha256)
    raw=capture(conversion_manifest,2*1024**2,artifact['conversion_manifest_sha256'])
    need(not (conversion_manifest.parent/'failure.json').exists(), 'Failed conversion finalization cannot pass a saved-byte audit')
    manifest=layout.saved.loads(raw)
    need(type(manifest) is dict and manifest.get('format')=='toklet-qwen35-2b-affine-conversion-v1'
         and manifest.get('passed') is True,'Completed conversion manifest required')
    need(manifest.get('model_numerics_quality_fit_performance_adoption_qualified') is False,
         'Conversion is a diagnostic storage proof only')
    layout.saved.same(manifest.get('complete_positive_predicates'),dict(source_lineage=True,whole_source_SHA256=True,
        all_component_readback=True,independent_upstream_F32_equal=True,exact_original_small_bytes=True,
        whole_child_SHA256=True,endpoint_unchanged=True),'Complete conversion predicates required')
    protocol=convert.validate_contract(manifest.get('protocol'))
    contract_pin=manifest.get('contract')
    need(type(contract_pin) is dict and contract_pin.get('sha256')==manifest.get('contract_sha256'),
         'Actual conversion contract descriptor required')
    contract_raw=capture(Path(contract_pin['path']),256*1024,layout.identifier(contract_pin['sha256']))
    need(len(contract_raw)==contract_pin.get('bytes'),'Conversion contract extent differs')
    layout.saved.same(layout.saved.loads(contract_raw),protocol,'Frozen conversion protocol bytes differ')
    need(protocol['source_directory']==str(source_dir),'Actual selected source directory differs')
    need(protocol['lineage_certificate_sha256']==artifact['lineage_certificate_sha256'],
         'Protocol and artifact lineage certificate bindings differ')
    layout.saved.same(manifest.get('artifact_pins'),artifact['artifact_pins'],'Actual artifact pins differ')
    cert_pin=manifest.get('lineage_certificate')
    need(type(cert_pin) is dict and cert_pin.get('sha256')==artifact['lineage_certificate_sha256'], 'Parent certificate binding differs')
    cert_raw=capture(Path(cert_pin['path']),2*1024**2,cert_pin['sha256'])
    cert=parent.validate_certificate(layout.saved.loads(cert_raw),protocol['parent_protocol_sha256'])
    convert.lineage.verify_ca(cert['transport']['ca'])
    need(len(cert_raw)==cert_pin.get('bytes'),'Parent certificate extent differs')
    setup=manifest.get('source_setup_receipt')
    need(type(setup) is dict and setup.get('sha256')==protocol['source_setup_receipt_sha256'],'Setup receipt binding differs')
    setup_raw=capture(Path(setup['path']),2*1024**2,setup['sha256'])
    need(layout.saved.loads(setup_raw).get('passed') is True and len(setup_raw)==setup.get('bytes'),'Positive setup receipt required')
    runtime_pin=manifest.get('runtime')
    need(type(runtime_pin) is dict,'Actual admitted runtime manifest descriptor required')
    runtime_raw=capture(Path(runtime_pin['path']),2*1024**2,layout.identifier(runtime_pin['sha256']))
    need(len(runtime_raw)==runtime_pin.get('bytes'),'Runtime manifest extent differs')
    runtime_manifest=layout.saved.loads(runtime_raw)
    need(type(runtime_manifest) is dict and runtime_manifest.get('format')=='toklet-granite-quant-runtime-v1',
         'Explicit admitted runtime manifest format required')
    layout.saved.same(runtime_manifest.get('versions'),convert.runtime.VERSIONS,'Admitted numerical package versions differ')
    pins=artifact['artifact_pins'];plan=layout.selected_layout()
    config=capture(model/'config.json',64*1024,pins['config_sha256'])
    original_config=capture(source_dir/'config.json',64*1024,layout.SOURCE_CONFIG_SHA256)
    need(config==layout.converted_config(original_config) and len(config)==pins['config_bytes'],'Exact preserved child config required')
    need(pins['header_bytes']==plan['origin']-8 and pins['header_sha256']==hashlib.sha256(plan['prefix'][8:]).hexdigest()
         and pins['container_bytes']==plan['file_bytes'],'Independent complete child storage layout differs')
    journal_pin=manifest.get('journal');need(type(journal_pin) is dict,'Complete conversion journal required')
    matrices,small,excluded=[],[],[]
    tiles=values=records=0
    lineage_hashes={row['name']:row['source_sha256'] for row in cert['tensors']}
    with layout.saved.SavedFile(source_dir/'model.safetensors',layout.SOURCE_BYTES) as source, \
            layout.saved.SavedFile(model/'model.safetensors',plan['file_bytes']) as child, \
            layout.saved.SavedFile(Path(journal_pin['path']),journal_pin['bytes']) as journal:
        need(source.before==cert['selected_source']['after_identity'],'Source lineage identity endpoint differs')
        need(source.digest()==layout.SOURCE_SHA256 and child.digest()==pins['container_sha256'],'Whole source/child SHA256 differs')
        need(journal.digest()==journal_pin['sha256'],'Whole conversion journal SHA256 differs')
        need(child.read(plan['origin'])==plan['prefix'],'Actual complete child header differs')
        header=source.read(layout.SOURCE_HEADER_BYTES,8);origin=8+len(header)
        spans=layout.source_spans(source.read(8),header,source.before[2])
        iterator=iter(journal_rows(journal))
        def next_record():
            nonlocal records
            try:value=next(iterator)
            except StopIteration:raise ValueError('Conversion journal ended before complete tensor inventory') from None
            records+=1
            return value
        for span in spans:
            name=span['name'];size=span['end']-span['start']
            if not span['selected']:
                row=next_record();expected=dict(event='source_exclusion',name=name,
                    source_sha256=source.digest(size,origin+span['start']),bytes=size,excluded=True)
                layout.saved.same(row,expected,'Complete source exclusion differs')
                excluded.append({key:value for key,value in row.items() if key!='event'});continue
            if name in plan['matrices']:
                shape=span['shape'];matrix_tiles=0
                for first in range(0,shape[0],256):
                    rows=min(256,shape[0]-first);record=next_record()
                    reconcile_tile(record,plan,span,first,rows,source,child,origin)
                    tiles+=1;matrix_tiles+=1;values+=rows*shape[1]
                row=next_record();need(row.get('event')=='matrix_completed','Ordered matrix completion required')
                report={key:value for key,value in row.items() if key!='event'}
                need(report.get('name')==name and report.get('shape')==shape and report.get('bits')==4
                     and report.get('tiles')==matrix_tiles and report.get('reconstructed_values')==math.prod(shape)
                     and report.get('reconstruction_equal') is True,'Complete matrix summary differs')
                original_sha=source.digest(size,origin+span['start'])
                need(report.get('source_sha256')==original_sha==lineage_hashes[name],'Actual complete source matrix differs')
                need(layout.identifier(report.get('independent_F32_sha256'))==layout.identifier(report.get('upstream_F32_sha256')),
                     'Complete recorded FP32 witness differs')
                expected_fields={}
                for key,field in plan['matrices'][name]['fields'].items():
                    item=plan['header'][field['name']];start,end=item['data_offsets']
                    expected_fields[key]=dict(name=field['name'],dtype=item['dtype'],shape=item['shape'],
                        offset=plan['origin']+start,bytes=end-start,sha256=child.digest(end-start,plan['origin']+start))
                layout.saved.same(report.get('fields'),expected_fields,'Complete actual matrix field differs')
                matrices.append(report);continue
            row=next_record();item=plan['header'][name];offset=plan['origin']+item['data_offsets'][0]
            original=source.read(size,origin+span['start']);actual=child.read(size,offset)
            need(original==actual,'Original F32/BF16 small bytes changed')
            small_spec=dict(name=name,dtype=span['dtype'],shape=span['shape'],bytes=size)
            layout.validate_small(small_spec,actual);digest=hashlib.sha256(actual).hexdigest()
            need(digest==lineage_hashes[name],'Small source lineage hash differs')
            expected=dict(event='small_copied',**small_spec,source_sha256=digest,offset=offset,sha256=digest,exact_bytes_equal=True)
            layout.saved.same(row,expected,'Typed exact small witness differs')
            small.append({key:value for key,value in row.items() if key!='event'})
        need(next(iterator,None) is None,'Surplus conversion journal record')
        need(records==journal_pin['records'],'Complete ordered journal record count differs')
        need(tiles==3238 and values==1881276416 and records==3855,
             'Complete frozen conversion tile/value/record inventory required')
        for key,actual in [('matrices',matrices),('small',small),('exclusions',excluded)]:
            layout.saved.same(manifest.get(key),actual,'Complete ordered conversion summary differs')
        need(source.digest()==layout.SOURCE_SHA256 and child.digest()==pins['container_sha256'],'Final whole source/child bytes differ')
        source.check();child.check();journal.check()
    for path,maximum,expected,raw,identity in inputs:
        actual,after=convert.files.capture_named(path,maximum,expected)
        need(actual==raw and after==identity,'Pinned metadata descriptor/path endpoint differs')
    convert.lineage.verify_ca(cert['transport']['ca'])
    return dict(format='toklet-qwen35-2b-saved-byte-audit-v1',passed=True,
                artifact_manifest_sha256=manifest_sha256,conversion_manifest_sha256=artifact['conversion_manifest_sha256'],
                logical_tensors=320,physical_tensors=694,matrices=len(matrices),exact_small=len(small),excluded=len(excluded),
                matrix_tiles=tiles,recorded_reconstructed_values=values,journal_records=records,
                actual_source_child_and_component_bytes_verified=True,numerical_runtime_reexecuted=False,
                historical_publisher_lineage_proved=False,model_numerics_quality_fit_performance_adoption_qualified=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','artifact-manifest-sha256','conversion-manifest','source-dir','fresh-output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv);output=Path(args.fresh_output)
    need(output.is_absolute() and not output.exists() and not output.is_symlink(),'Fresh absolute audit output required')
    output.mkdir(mode=0o700)
    try:
        result=audit(Path(args.model_dir),args.artifact_manifest_sha256,Path(args.conversion_manifest),Path(args.source_dir))
    except BaseException as error:
        result=dict(format='toklet-qwen35-2b-saved-byte-audit-v1',passed=False,error_type=type(error).__name__,error=str(error)[:2048],
                    numerical_runtime_reexecuted=False,model_numerics_quality_fit_performance_adoption_qualified=False)
    convert.files.write_exclusive(output/'manifest.json',(json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
