"""Portable explicit-variant/source-provenance and unchanged scalar controls."""
import copy
import hashlib
from pathlib import Path
import unittest
import build_qwen35_vector_graph as builder
import check_qwen35_9b_reference as check
import qwen35_9b_reference as scalar
import qwen35_9b_vector_reference as vector

ROOT=Path(__file__).resolve().parents[2]


def freeze():
    return dict(reference_variant='vector',files={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in check.VECTOR_SOURCES},
        graph_provenance=builder.validate())


class Controls(unittest.TestCase):
    def test_vector_is_explicit_and_scalar_default_remains_separate(self):
        filename,fields=check.reference_variant('vector',freeze(),ROOT)
        self.assertEqual(filename,'qwen35_9b_vector_reference.py')
        self.assertEqual(fields['format'],'toklet-qwen35-9b-vector-reference-v1')
        self.assertEqual(fields['graph_provenance'],builder.validate())
        self.assertEqual(check.reference_variant('scalar',{},ROOT),('qwen35_9b_reference.py',dict(format='toklet-qwen35-9b-real-reference-v1')))
        with self.assertRaises(ValueError):check.reference_variant('scalar',freeze(),ROOT)
        for name in ['','fp32','auto',None,True,1]:
            with self.assertRaises(ValueError):check.reference_variant(name,freeze(),ROOT)

    def test_missing_changed_extra_and_unfrozen_sources_reject(self):
        for change in [lambda f:f.pop('reference_variant'),lambda f:f.update(reference_variant='scalar'),
            lambda f:f['files'].pop('tools/reference/qwen35_9b_weights.py'),
            lambda f:f['files'].update({'unadmitted.py':'0'*64}),lambda f:f.update(files=[]),
            lambda f:f['files'].update({'tools/reference/qwen35_recurrent_vector.py':'0'*64}),
            lambda f:f['graph_provenance'].update(inverse_patch_exact=False),
            lambda f:f['graph_provenance'].update(original_sha256='0'*64),
            lambda f:f['graph_provenance'].update(extra=True)]:
            value=freeze();change(value)
            with self.assertRaises(ValueError):check.reference_variant('vector',value,ROOT)

    def test_original_workloads_geometry_gate_and_inventory_are_unchanged(self):
        for phase in ['one','context17']:self.assertEqual(scalar.plan(phase),vector.plan(phase))
        self.assertEqual(scalar.graph.GATE,vector.graph.GATE)
        self.assertEqual(scalar.LEDGER,vector.LEDGER)
        self.assertEqual(scalar.MAX_CAPTURE,vector.MAX_CAPTURE)
        self.assertEqual(scalar.weights.GEOMETRY,vector.weights.GEOMETRY)
        expected={p.name for p in vector.extras()}|{'reference_runtime.py','gemma4_affine_reference.py','affine4_tiles.py'}
        self.assertEqual(expected,{Path(p).name for p in check.VECTOR_SOURCES})
        with self.assertRaises(ValueError):vector.plan('context256')


if __name__=='__main__':unittest.main()
