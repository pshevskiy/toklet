"""Cold ordered witness, saved-component mutation and failure-output controls."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import qwen35_2b_audit as audit
from test_qwen35_2b_layout import tiny_recipe


class Audit(unittest.TestCase):
    def test_saved_tile_reconciliation_and_numeric_witness_mutations(self):
        matrices,small=tiny_recipe();plan=audit.layout.output_layout(matrices,small)
        name=matrices[0]['module']+'.weight';span=dict(name=name,shape=[3,64],start=0,end=384)
        raw=bytes(384);buffers=dict(weight=bytes(96),scales=struct.pack('<H',0x3f80)*3,biases=bytes(6))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source_path=root/'source';child_path=root/'child';source_path.write_bytes(raw)
            with child_path.open('w+b') as child:
                child.truncate(plan['file_bytes'])
                audit.layout.writes.write_matrix_tile(child.fileno(),plan,name,0,3,buffers)
            fields={key:audit.convert.component(plan,item['name'],0,3,buffers[key]) for key,item in plan['matrices'][name]['fields'].items()}
            record=dict(event='matrix_tile',name=name,first_row=0,rows=3,columns=64,bits=4,source_offset=0,
                        source_bytes=384,source_sha256=hashlib.sha256(raw).hexdigest(),fields=fields,
                        reconstructed_values=192,reconstruction_equal=True,component_readback=True,
                        independent_F32_sha256='1'*64,upstream_F32_sha256='1'*64)
            with audit.layout.saved.SavedFile(source_path) as source,audit.layout.saved.SavedFile(child_path) as child:
                audit.reconcile_tile(record,plan,span,0,3,source,child,0)
                for key,value in [('first_row',False),('upstream_F32_sha256','2'*64),('component_readback',False),
                                  ('source_sha256','3'*64)]:
                    broken=copy.deepcopy(record);broken[key]=value
                    with self.assertRaises(ValueError):audit.reconcile_tile(broken,plan,span,0,3,source,child,0)
                broken=copy.deepcopy(record);broken['fields']['scales']['sha256']='4'*64
                with self.assertRaises(ValueError):audit.reconcile_tile(broken,plan,span,0,3,source,child,0)

    def test_journal_requires_complete_newline_unique_JSON_and_bounded_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'journal'
            for raw in (b'{}',b'{"a":1,"a":1}\n',b'\n',b' '*65537+b'\n'):
                path.write_bytes(raw)
                with audit.layout.saved.SavedFile(path) as saved:
                    with self.assertRaises(ValueError):list(audit.journal_rows(saved))
            path.write_bytes(b'{"event":"one"}\n{"event":"two"}\n')
            with audit.layout.saved.SavedFile(path) as saved:
                self.assertEqual([row['event'] for row in audit.journal_rows(saved)],['one','two'])

    def test_actual_cli_retains_failed_audit_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            args=['--model-dir',str(root/'model'),'--artifact-manifest-sha256','1'*64,
                  '--conversion-manifest',str(root/'conversion'),'--source-dir',str(root/'source'),
                  '--fresh-output',str(root/'audit')]
            self.assertEqual(audit.main(args),1)
            result=json.loads((root/'audit/manifest.json').read_bytes())
            self.assertFalse(result['passed'])
            self.assertFalse(result['numerical_runtime_reexecuted'])
            self.assertFalse(result['model_numerics_quality_fit_performance_adoption_qualified'])


if __name__=='__main__':unittest.main()
