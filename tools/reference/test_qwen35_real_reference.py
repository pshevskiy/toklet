import copy
import hashlib
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import bf16_source
import qwen35_cpu_graph as graph
import qwen35_real_reference as producer
import qwen35_real_weights as adapter


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.table, cls.matrices, cls.vectors, _ = graph.artifact()

    def wire(self):
        table = {}
        for name, spec in self.table.items():
            if spec['kind'] == 'matrix':
                rows, columns = spec['shape']
                for key, part, dtype, shape in [(name, 'codes', 'U32', [rows, columns//8]),
                    (name.removesuffix('.weight')+'.scales', 'scales', 'BF16', [rows, columns//64]),
                    (name.removesuffix('.weight')+'.biases', 'biases', 'BF16', [rows, columns//64])]:
                    item = spec[part]
                    table[key] = dict(dtype=dtype, shape=shape, data_offsets=[item['offset'], item['offset']+item['bytes']])
            else:
                item = spec['data']
                table[name] = dict(dtype='BF16', shape=spec['shape'], data_offsets=[item['offset'], item['offset']+item['bytes']])
        return table

    def test_original_fixture_regenerates_every_byte(self):
        root = Path(__file__).resolve().parents[2]/'tests/fixtures/qwen35-cpu-graph-v1'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'regenerated'
            graph.generate(path)
            for name in ['weights.bin', 'reference.bin', 'manifest.json']:
                self.assertEqual((path/name).read_bytes(), (root/name).read_bytes())

    def test_streaming_and_explicit_geometry_preserve_all_stages(self):
        for staged in [False, True]:
            rows = []
            expected = graph.forward(self.matrices, self.vectors, [1, 7, 13], 4079, staged)
            result = graph.forward(self.matrices, self.vectors, [1, 7, 13], 4079, staged,
                geometry=dict(graph.GEOMETRY), math_type=graph.Math, sink=rows.append)
            self.assertEqual(result, [])
            self.assertEqual(rows, expected)

    def test_geometry_controls_rotary_and_rms_without_global_mutation(self):
        before = dict(graph.GEOMETRY)
        config = dict(before, rope_theta=100., rms_epsilon=.125)
        m = graph.Math(False, config)
        x = [float(v)/32 for v in range(32)]
        self.assertNotEqual(m.rotary(x, 1, 32, 8, 11), graph.Math(False).rotary(x, 1, 32, 8, 11))
        self.assertEqual(m.norm([1., 2.], [1., 1.], 2), [v/math.sqrt(2.5+graph.f32(.125)) for v in [1., 2.]])
        self.assertEqual(graph.GEOMETRY, before)

    def test_odd_origin_rows_vectors_and_bounded_projection_match_scalar(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'source'
            raw = b'\xa5'+self.raw
            path.write_bytes(raw)
            with bf16_source.Source(path, len(raw), hashlib.sha256(raw).hexdigest(), deadline=time.monotonic()+120) as owner:
                matrices, vectors = adapter.bind(owner, self.wire(), 1, graph.GEOMETRY, time.monotonic()+120)
                self.assertEqual(vectors, self.vectors)
                for name, dense in self.matrices.items():
                    for row in [0, len(dense)//2, len(dense)-1]: self.assertEqual(matrices[name][row], dense[row])
                    x = [float(j%7-3)/8 for j in range(len(dense[0]))]
                    for staged in [False, True]:
                        actual = adapter.Math(staged).linear(x, matrices[name])
                        expected = graph.Math(staged).linear(x, dense)
                        self.assertEqual(actual, expected)
                rows = []
                graph.forward(matrices, vectors, [1, 7], 4079, True, math_type=adapter.Math, sink=rows.append)
                expected = graph.forward(self.matrices, self.vectors, [1, 7], 4079, True)
                self.assertEqual([(v['token'], v['layer'], v['stage']) for v in rows], [(v['token'], v['layer'], v['stage']) for v in expected])
                for a, b in zip(rows, expected, strict=True):
                    for got, want in zip(a['values'], b['values'], strict=True): self.assertLessEqual(abs(got-want), 1e-5+1e-4*abs(want))
                with self.assertRaises(ValueError): adapter.weights(owner, b'', time.monotonic()+120)

    def test_descriptor_types_extents_and_extra_owners_reject(self):
        table = self.wire()
        name = next(iter(table))
        item = table[name]
        for change in [dict(shape=[True, item['shape'][1]]), dict(data_offsets=[True, item['data_offsets'][1]]),
                       dict(dtype='F32'), dict(data_offsets=[0, 1]), dict(extra=0)]:
            broken = copy.deepcopy(table)
            broken[name].update(change)
            with self.assertRaises(ValueError): adapter.component(broken, 1, len(self.raw)+1, name, item['dtype'], item['shape'])
        with self.assertRaises(ValueError): adapter.unique([('x', 0), ('x', 1)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'source'; raw = b'\xa5'+self.raw; path.write_bytes(raw)
            with bf16_source.Source(path, len(raw), hashlib.sha256(raw).hexdigest(), deadline=time.monotonic()+120) as owner:
                table['extra'] = dict(dtype='BF16', shape=[1], data_offsets=[0, 2])
                with self.assertRaises(ValueError): adapter.bind(owner, table, 1, graph.GEOMETRY, time.monotonic()+120)

    def test_collection_rejects_bad_config_and_closes_authenticated_owner(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); model = root/'model'; model.mkdir()
            raw = b'not model weights'; (model/'model.safetensors').write_bytes(raw)
            (root/'artifacts').mkdir()
            with patch.object(producer, 'ROOT', root), patch.object(producer.pin, 'FILE_BYTES', len(raw)), \
                    patch.object(producer.pin, 'FILE_SHA', hashlib.sha256(raw).hexdigest()), \
                    patch.object(producer.runtime_owner.runtime, 'captured', return_value=b''):
                result = producer.collect(root/'artifacts/output', model, {}, lambda: {'checked': True})
            self.assertFalse(result['completed'])
            self.assertEqual(result['primary_error']['message'], 'Pinned Qwen config differs')
            self.assertTrue(result['source']['closed'])
            self.assertFalse(result['source']['poisoned'])
            self.assertEqual(result['endpoint_errors'], [])

    def test_source_mutation_permanently_poisoned_before_row_return(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'source'; raw = b'\xa5'+self.raw; path.write_bytes(raw)
            with bf16_source.Source(path, len(raw), hashlib.sha256(raw).hexdigest(), deadline=time.monotonic()+120) as owner:
                matrices, _ = adapter.bind(owner, self.wire(), 1, graph.GEOMETRY, time.monotonic()+120)
                path.write_bytes(b'\xa4'+self.raw)
                with self.assertRaises(bf16_source.SourceChanged): matrices[graph.P+'embed_tokens.weight'][1]
                self.assertTrue(owner.poisoned)

    def test_real_plan_is_fixed_and_within_budget(self):
        self.assertEqual((len(graph.specs(adapter.GEOMETRY)), sum(k == 'matrix' for k, s in graph.specs(adapter.GEOMETRY).values())), (426, 249))
        self.assertEqual(producer.VALUES*12, 192807936)
        self.assertLess(producer.VALUES*12, producer.MAX_CAPTURE)
        self.assertLess(sum(producer.LEDGER.values()), 6*1024**3)
        self.assertEqual(adapter.ORIGIN % 4, 1)


if __name__ == '__main__': unittest.main()
