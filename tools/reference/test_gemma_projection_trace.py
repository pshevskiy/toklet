"""Portable full-map, layout, rounding and late-failure regression checks."""
from fractions import Fraction
from pathlib import Path
import hashlib,json,math,struct,tempfile,unittest
from unittest.mock import patch
import gemma_projection_trace as producer
import check_gemma_projection_trace as checker

class Trace(unittest.TestCase):
    def test_independent_full_mapping_matches_acknowledged_invocation_inventory(self):
        mapped=checker.mapping();shapes={}
        for item in mapped:
            for name in item['sources']:
                shape=[1,8,4,item['columns']//8] if name.endswith('.context') else [1,4,item['columns']]
                if name in shapes:self.assertEqual(shapes[name],shape)
                shapes[name]=shape
        def read(name):
            shape=shapes[name];return [1.]*math.prod(shape),shape
        actual=list(producer.accepted.inputs(read,True))
        self.assertEqual(len(actual),277)
        for (name,kind,names,_),item in zip(actual,mapped,strict=True):
            self.assertEqual((name,kind,names),(item['name'],item['kind'],item['sources']))
            self.assertEqual(producer.output_name(name),item['output'])
        self.assertEqual(sum(v['kind']=='product-envelope' for v in mapped),70)

    def test_fixed_552_array_byte_inventory_and_ff_width_transition(self):
        mapping=checker.mapping();active=[v for v in mapping if v['rows']<=16384]
        self.assertEqual(sum(4*v['columns']*4 for v in active),11128832)
        self.assertEqual(sum(4*v['rows']*4 for v in active),15204352)
        names={v['name']:v for v in mapping}
        for layer,width in [(14,6144),(15,12288)]:
            self.assertEqual(names[f'layers.{layer}.mlp.down_proj.weight']['columns'],width)
            self.assertEqual(names[f'layers.{layer}.mlp.gate_proj.weight']['rows'],width)
        self.assertEqual(len(active)*2,552)
        self.assertEqual([v['name'] for v in mapping if v not in active],['embed_tokens.weight'])

    def test_output_mapping_is_before_norm_scale_and_residual(self):
        mapping={v['name']:v for v in checker.mapping()}
        for suffix in ('self_attn.q_proj','self_attn.k_proj','self_attn.o_proj','mlp.down_proj','per_layer_projection'):
            self.assertEqual(mapping['layers.0.'+suffix+'.weight']['output'],'layer.0.'+suffix)
        self.assertEqual(mapping['per_layer_model_projection.weight']['output'],'ple_projection')
        self.assertEqual(mapping['embed_tokens.weight']['output'],'head')

    def test_head_major_context_preserves_each_token_and_head(self):
        item=next(v for v in checker.mapping() if v['name']=='layers.0.self_attn.o_proj.weight')
        values=[float(100*head+token) for head in range(8) for token in range(4) for _ in range(256)]
        def read(_):return values,[1,8,4,256]
        first=producer.inputs(item['kind'],item['sources'],read)
        second=checker.reconstruct(item,read)
        self.assertEqual([v for row in first for v in row],second)
        self.assertEqual([first[token][head*256] for token in range(4) for head in range(8)],
                         [float(100*head+token) for token in range(4) for head in range(8)])

    def test_residual_keeps_both_fp32_rounding_boundaries(self):
        item=next(v for v in checker.mapping() if v['kind']=='exact-residual')
        values=dict(zip(item['sources'],[2.**24,1.,-2.**24],strict=True))
        def read(name):return [values[name]]*6144,[1,4,1536]
        a=[v for row in producer.inputs(item['kind'],item['sources'],read) for v in row]
        b=checker.reconstruct(item,read)
        self.assertEqual(a,b);self.assertEqual(set(a),{0.})
        self.assertEqual(sum((Fraction(v) for v in values.values()),Fraction()),1)

    def test_product_has_separate_activation_and_product_rounding(self):
        # Find a fixed-grid discriminator using only the acknowledged recipe;
        # neither candidate evaluates any native target implementation.
        found=None
        for g in [producer.accepted.f32(v/16) for v in range(-32,33)]:
            for u in [3.,7.,17.,.125]:
                activation=.5*g*(1+math.tanh(math.sqrt(2/math.pi)*(g+.044715*g**3)))
                two=producer.accepted.f32(producer.accepted.f32(activation)*u)
                one=producer.accepted.f32(activation*u)
                if two!=one:found=(g,u,two,one);break
            if found:break
        self.assertIsNotNone(found)
        g,u,two,one=found
        item=dict(kind='product-envelope',sources=['g','u'],columns=64)
        def read(name):return [g if name=='g' else u]*256,[1,4,64]
        a=[v for row in producer.inputs(item['kind'],item['sources'],read) for v in row]
        self.assertEqual(a,checker.reconstruct(item,read));self.assertEqual(set(a),{two});self.assertNotEqual(two,one)

    def test_product_shape_and_nonfinite_admission(self):
        item=dict(kind='product-envelope',sources=['g','u'],columns=64)
        for shape,values in [([1,4,32],[1.]*128),([1,4,64],[True]*256),([1,4,64],[math.inf]*256)]:
            def read(name):return (values,shape) if name=='g' else ([1.]*256,[1,4,64])
            with self.assertRaises((ValueError,OverflowError)):
                producer.inputs(item['kind'],item['sources'],read)
            with self.assertRaises((ValueError,OverflowError)):
                checker.reconstruct(item,read)

    def test_derived_zero_product_signs_are_retained_bitwise(self):
        item=dict(kind='product-envelope',sources=['g','u'],columns=64)
        def read(name):return ([-20.]*256 if name=='g' else [1.,-1.]*128),[1,4,64]
        a=[v for row in producer.inputs(item['kind'],item['sources'],read) for v in row]
        b=checker.reconstruct(item,read)
        expected=struct.pack('<256f',*([-0.,0.]*128))
        self.assertEqual(struct.pack('<256f',*a),expected);self.assertEqual(struct.pack('<256f',*b),expected)

    def test_all_negative_zero_residuals_keep_their_sign(self):
        item=next(v for v in checker.mapping() if v['kind']=='exact-residual')
        def read(_):return [-0.]*6144,[1,4,1536]
        first=[v for row in producer.inputs(item['kind'],item['sources'],read) for v in row]
        second=checker.reconstruct(item,read)
        expected=struct.pack('<6144f',*([-0.]*6144))
        self.assertEqual(struct.pack('<6144f',*first),expected);self.assertEqual(struct.pack('<6144f',*second),expected)

    def test_later_read_failure_keeps_first_arrays_and_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            coordinate,l1,trace,raw,compact,recipe=[root/name for name in ('coordinate','l1','trace','raw','compact','recipe')]
            first='per_layer_model_projection.weight';second='layers.0.self_attn.q_proj.weight'
            matrices=[dict(name=name,rows=n,columns=1536,geometry_admitted=True,maximum_weight=1.) for name,n in [(first,8960),(second,2048)]]
            previous=dict(passed=True,matrices=matrices,invocations=[dict(name=first,input_kind='direct',sources=[dict(name='scaled_embedding')]),dict(name=second,input_kind='direct',sources=[dict(name='layer.0.input_layernorm')])])
            coordinate.write_text(json.dumps(previous));l1.write_text('{"passed":true}')
            for p in (trace,compact,recipe):p.write_text('{}')
            blocks=[struct.pack('<'+'f'*n,*([1.]*n)) for n in (6144,35840,6144)]
            raw.write_bytes(b''.join(blocks));ledger={};offset=0
            for index,(name,width) in enumerate([('scaled_embedding',1536),('ple_projection',8960),('layer.0.input_layernorm',1536)]):
                body=blocks[index];ledger[('staged_fp32',name)]=dict(bytes=len(body),shape=[1,4,width],offset=offset,sha256=hashlib.sha256(body).hexdigest() if index<2 else '0'*64);offset+=len(body)
            ledger.update({('unused',str(i)):{} for i in range(1423)})
            fixed=dict(zip((coordinate,l1,trace,raw,compact,recipe),(producer.COORDINATE_SHA,producer.L1_SHA,producer.accepted.TRACE_SHA,producer.accepted.ARRAY_SHA,producer.accepted.COMPACT_SHA,producer.RECIPE_SHA),strict=True))
            original=producer.accepted.sha
            def sha(path):return fixed[path] if path in fixed else original(path)
            output=root/'capture'
            with patch.object(producer.accepted,'sha',side_effect=sha),patch.object(producer.accepted,'trace_ledger',return_value=ledger):
                with self.assertRaisesRegex(ValueError,'raw source state differs'):producer.capture(coordinate,l1,trace,raw,compact,recipe,output)
            report=json.loads((output/'observation.json').read_text());manifest=json.loads((output/'manifest.json').read_text())
            self.assertFalse(report['passed']);self.assertFalse(manifest['passed'])
            self.assertEqual(report['primary_error'],'ValueError: raw source state differs')
            self.assertEqual([v['completed'] for v in report['records']],[True,False])
            self.assertEqual((output/'arrays.bin').read_bytes(),blocks[0]+blocks[1])
            self.assertTrue(all(v['completed'] for v in report['records'][0]['arrays'].values()))

    def test_report_write_failure_after_success_retains_arrays_and_failed_manifest(self):
        # Run the whole real capture/publication path on a complete zero-valued
        # synthetic trace. Mock only prerequisite authentication and the ledger.
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            coordinate,l1,trace,raw,compact,recipe=[root/name for name in ('coordinate','l1','trace','raw','compact','recipe')]
            mapped=checker.mapping();shapes={}
            for item in mapped:
                if item['rows']>16384:continue
                definitions=[(name,[1,8,4,item['columns']//8] if name.endswith('.context') else [1,4,item['columns']]) for name in item['sources']]
                definitions.append((item['output'],[1,4,item['rows']]))
                for name,shape in definitions:
                    if name in shapes:self.assertEqual(shapes[name],shape)
                    shapes[name]=shape
            ledger={('staged_fp32',name):dict(shape=shape,bytes=math.prod(shape)*4,offset=0,
                     sha256=hashlib.sha256(bytes(math.prod(shape)*4)).hexdigest()) for name,shape in shapes.items()}
            ledger.update({('unused',str(i)):{} for i in range(1426-len(ledger))})
            matrices=[dict(name=i['name'],rows=i['rows'],columns=i['columns'],geometry_admitted=i['rows']<=16384,maximum_weight=1.) for i in mapped]
            invocations=[dict(name=i['name'],input_kind=i['kind'],sources=[dict(name=n) for n in i['sources']]) for i in mapped]
            coordinate.write_text(json.dumps(dict(passed=True,matrices=matrices,invocations=invocations)));l1.write_text('{"passed":true}')
            for path in (trace,compact,recipe):path.write_text('{}')
            raw.write_bytes(bytes(196608))
            fixed=dict(zip((coordinate,l1,trace,raw,compact,recipe),(producer.COORDINATE_SHA,producer.L1_SHA,producer.accepted.TRACE_SHA,producer.accepted.ARRAY_SHA,producer.accepted.COMPACT_SHA,producer.RECIPE_SHA),strict=True))
            original_sha=producer.accepted.sha;original_dump=json.dump;failure=OSError('injected report write failure')
            def sha(path):return fixed[path] if path in fixed else original_sha(path)
            def dump(data,stream,**kwargs):
                if Path(stream.name).name=='observation.json':
                    stream.write('{"partial":');stream.flush();raise failure
                return original_dump(data,stream,**kwargs)
            output=root/'capture'
            with patch.object(producer.accepted,'sha',side_effect=sha),patch.object(producer.accepted,'trace_ledger',return_value=ledger),patch.object(producer.json,'dump',side_effect=dump):
                with self.assertRaises(OSError) as caught:producer.capture(coordinate,l1,trace,raw,compact,recipe,output)
            self.assertIs(caught.exception,failure)
            self.assertEqual((output/'arrays.bin').stat().st_size,26333184)
            self.assertEqual((output/'observation.json').read_text(),'{"partial":')
            manifest=json.loads((output/'manifest.json').read_text());self.assertFalse(manifest['passed'])
            self.assertEqual(manifest['files']['arrays.bin']['bytes'],26333184)
            self.assertEqual(manifest['files']['observation.json']['sha256'],hashlib.sha256(b'{"partial":').hexdigest())

if __name__=='__main__':unittest.main()
