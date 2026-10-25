"""Freeze complete staged-FP32 projection inputs/outputs from the accepted trace.

The70 derived GELU products are mathematical-recipe inputs, not newly observed
native activations. No model, engine, NumPy or GPU execution occurs here.
"""
from pathlib import Path
import hashlib,json,math,struct,sys
import gemma_affine_admission as accepted
import gemma_affine_l1_math as proof

COORDINATE_SHA='2d082715177522d4ea353d699a8415b792fa52ffa943b9d083d8aee832c7d821'
L1_SHA='14d479c1f218f3531512568e286f008a869375959cec2c7c31fdeac3e5964d44'
RECIPE_SHA='6c04a81b93c95edfad4a4de646da3731b71a6dd175163fb856d2009bd103b096'
MAX_BINARY=32*1024**2

def output_name(name):
    if name=='per_layer_model_projection.weight':return 'ple_projection'
    if name=='embed_tokens.weight':return 'head'
    parts=name.split('.')
    if len(parts)<4 or parts[0]!='layers' or parts[-1]!='weight':raise ValueError('known matrix name required')
    return 'layer.'+parts[1]+'.'+'.'.join(parts[2:-1])

def inputs(kind,names,read):
    if kind!='product-envelope':return proof.input_rows(kind,names,read)
    gate,shape=read(names[0]);up,other=read(names[1])
    if type(shape) is not list or type(other) is not list or shape!=other or len(shape)!=3 or any(type(v) is not int for v in shape) or shape[:2]!=[1,4] or not 1<=shape[-1]<=12288 or len(gate)!=4*shape[-1] or len(up)!=len(gate):raise ValueError('paired product geometry required')
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in gate+up):raise ValueError('finite typed product inputs required')
    product=[]
    for g,u in zip(gate,up,strict=True):
        # Preserve the acknowledged Python x**3 source recipe, then both FP32
        # boundaries; do not substitute target CPU activation arithmetic.
        activated=accepted.f32(.5*g*(1+math.tanh(math.sqrt(2/math.pi)*(g+.044715*g**3))))
        value=accepted.f32(activated*u)
        if not math.isfinite(value):raise ValueError('finite derived product required')
        product.append(value)
    return [product[t*shape[-1]:(t+1)*shape[-1]] for t in range(4)]

def capture(coordinate,l1_receipt,trace,raw_trace,compact,recipe,output):
    output.mkdir();report=dict(format='toklet-gemma-projection-trace-v1',passed=False,records=[],primary_error=None,
        expected_contract='Acknowledged mathematical graph: rounded staged-FP32 linear output over reconstructed FP32 weights',
        tokens=[2,100,257,100],positions=[0,1,2,3],binary_bytes=0,input_bytes=0,output_bytes=0,
        captured_state_inputs=0,derived_recipe_inputs=0,geometry_rejections=0,
        native_activation_parity=False,actual_gpu_executed=False,whole_graph_qualified=False,
        source_or_precision_failures_waived=False,app_or_phone_qualified=False)
    primary=None
    try:
        required=[(coordinate,COORDINATE_SHA),(l1_receipt,L1_SHA),(trace,accepted.TRACE_SHA),
                  (raw_trace,accepted.ARRAY_SHA),(compact,accepted.COMPACT_SHA),(recipe,RECIPE_SHA)]
        for path,wanted in required:
            if accepted.sha(path)!=wanted:raise ValueError('authenticated prerequisite differs')
        report['prerequisites']=[dict(path=str(path),sha256=wanted) for path,wanted in required]
        old=accepted.decode(accepted.bounded(coordinate,2*1024**2));receipt=accepted.decode(accepted.bounded(l1_receipt,2*1024**2))
        if old['passed'] is not True or receipt['passed'] is not True:raise ValueError('accepted prior evidence required')
        matrices={m['name']:m for m in old['matrices']}
        observation=accepted.decode(accepted.bounded(trace,4*1024**2));frozen=accepted.decode(accepted.bounded(compact,2*1024**2))
        with raw_trace.open('rb') as source,(output/'arrays.bin').open('xb') as destination:
            ledger=accepted.trace_ledger(observation,frozen,source)
            if len(ledger)!=1426:raise ValueError('complete accepted trace required')
            def read(name):
                entry=ledger[('staged_fp32',name)]
                if not 0<entry['bytes']<=196608:raise ValueError('bounded complete projection state required')
                source.seek(entry['offset']);body=source.read(entry['bytes'])
                if len(body)!=entry['bytes'] or hashlib.sha256(body).hexdigest()!=entry['sha256']:raise ValueError('raw source state differs')
                values=[v for (v,) in struct.iter_unpack('<f',body)]
                if any(not math.isfinite(v) for v in values):raise ValueError('finite captured state required')
                return values,entry['shape']
            def save(record,role,values,width):
                if len(values)!=4*width:raise ValueError('complete per-token array required')
                body=struct.pack('<'+'f'*len(values),*values)
                if destination.tell()+len(body)>MAX_BINARY:raise ValueError('projection binary cap exceeded')
                item=dict(dtype='F32',shape=[4,width],offset=destination.tell(),bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),completed=False)
                record['arrays'][role]=item
                if destination.write(body)!=len(body):raise OSError('short array write')
                destination.flush();item['completed']=True;report['binary_bytes']=destination.tell()
                report['input_bytes' if role=='input' else 'output_bytes']+=len(body)
            for previous in old['invocations']:
                name=previous['name'];m=matrices[name];out=output_name(name)
                record=dict(name=name,input_kind=previous['input_kind'],sources=previous['sources'],output_source=out,
                    rows=m['rows'],columns=m['columns'],arrays={},completed=False,
                    input_evidence='derived-mathematical-gelu-product' if previous['input_kind']=='product-envelope' else 'captured-state-reconstruction')
                report['records'].append(record)
                if not m['geometry_admitted']:
                    if name!='embed_tokens.weight':raise ValueError('unexpected geometry rejection')
                    record.update(status='geometry-rejected',completed=True);report['geometry_rejections']+=1;continue
                rows=inputs(previous['input_kind'],[v['name'] for v in previous['sources']],read)
                if len(rows)!=4 or any(len(row)!=m['columns'] for row in rows):raise ValueError('complete input geometry required')
                flat=[v for row in rows for v in row]
                result,shape=read(out)
                if shape!=[1,4,m['rows']]:raise ValueError('pre-norm linear output shape differs')
                magnitudes=[proof.upper_sum(abs(v) for v in row) for row in rows]
                old_bound=accepted.bound(m['columns'],m['maximum_weight'],max(map(abs,flat)))
                new_bound=proof.l1_bound(m['columns'],m['maximum_weight'],max(magnitudes))
                if not (old_bound['admitted'] or new_bound['admitted']):raise ValueError('actual input proof fails')
                record.update(maximum_weight=m['maximum_weight'],per_token_l1=magnitudes,legacy_max=old_bound,l1=new_bound)
                save(record,'input',flat,m['columns']);save(record,'expected_staged_fp32',result,m['rows'])
                record.update(status='proved-max' if old_bound['admitted'] else 'proved-l1',completed=True)
                report['derived_recipe_inputs' if previous['input_kind']=='product-envelope' else 'captured_state_inputs']+=1
        if (len(report['records']),report['captured_state_inputs'],report['derived_recipe_inputs'],report['geometry_rejections'],report['input_bytes'],report['output_bytes'])!=(277,206,70,1,11128832,15204352):raise ValueError('complete fixed projection inventory required')
        report['passed']=True
    except BaseException as error:
        primary=error;report['passed']=False;report['primary_error']=type(error).__name__+': '+str(error)
    finally:
        try:
            with (output/'observation.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
        except BaseException as error:
            if primary is None:
                primary=error;report['passed']=False;report['primary_error']=type(error).__name__+': '+str(error)
            else:primary.add_note('Partial report publication failed: '+str(error))
        try:
            paths=[output/name for name in ('observation.json','arrays.bin') if (output/name).exists()]
            manifest=dict(format='toklet-gemma-projection-manifest-v1',passed=report['passed'],producer_sha256=accepted.sha(Path(__file__)),
                          files={p.name:dict(bytes=p.stat().st_size,sha256=accepted.sha(p)) for p in paths})
            with (output/'manifest.json').open('x') as stream:json.dump(manifest,stream,indent=2);stream.write('\n')
        except BaseException as error:
            if primary is None:
                primary=error;report['passed']=False;report['primary_error']=type(error).__name__+': '+str(error)
            else:primary.add_note('Partial manifest publication failed: '+str(error))
    if primary is not None:raise primary.with_traceback(primary.__traceback__)
    return dict(passed=True,records=277,captured_state_inputs=206,derived_recipe_inputs=70,arrays=552,binary_bytes=report['binary_bytes'])

if __name__=='__main__':
    if len(sys.argv)!=8:raise SystemExit('coordinate l1-receipt trace-json trace-bin compact-json recipe-source new-output')
    print(json.dumps(capture(*(Path(v) for v in sys.argv[1:]))))
