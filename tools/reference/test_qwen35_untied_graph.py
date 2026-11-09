import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import qwen35_untied_graph as fixture

ROOT=Path(__file__).resolve().parents[2]


class Controls(unittest.TestCase):
    def test_regeneration_and_original_prefix_are_exact(self):
        current=ROOT/'tests/fixtures/qwen35-untied-graph-v1'
        with tempfile.TemporaryDirectory() as tmp:
            generated=Path(tmp)/'new';fixture.generate(generated)
            for name in ['weights.bin','reference.bin','manifest.json','README.md']:
                self.assertEqual((current/name).read_bytes(),(generated/name).read_bytes())
        old=(ROOT/'tests/fixtures/qwen35-cpu-graph-v1/weights.bin').read_bytes()
        self.assertEqual((current/'weights.bin').read_bytes()[:len(old)],old)

    def test_only_final_logits_change_in_both_frozen_profiles(self):
        old=json.loads((ROOT/'tests/fixtures/qwen35-cpu-graph-v1/manifest.json').read_text())
        current=json.loads((ROOT/'tests/fixtures/qwen35-untied-graph-v1/manifest.json').read_text())
        self.assertEqual(old['geometry'],current['geometry']);self.assertEqual(old['gate'],current['gate'])
        changed=0;preserved=0
        for before,after in zip(old['cases'],current['cases'],strict=True):
            for key in ['name','tokens','position_offset']:self.assertEqual(before[key],after[key])
            for profile in ['ideal_fp64','staged_fp32']:
                for a,b in zip(before['profiles'][profile],after['profiles'][profile],strict=True):
                    for key in ['token','layer','stage','count','bytes']:self.assertEqual(a[key],b[key])
                    if a['stage']=='logits':self.assertNotEqual(a['sha256'],b['sha256']);changed+=1
                    else:self.assertEqual(a['sha256'],b['sha256']);preserved+=1
        self.assertEqual(changed,104);self.assertGreater(preserved,10000)

    def test_distinct_head_shape_bytes_and_producer_identity(self):
        folder=ROOT/'tests/fixtures/qwen35-untied-graph-v1';m=json.loads((folder/'manifest.json').read_text())
        head=m['weights'][m['output_head_name']];embedding=m['weights'][m['embedding_name']]
        self.assertEqual(head['shape'],embedding['shape']);self.assertEqual(head['shape'],[32,64])
        self.assertNotEqual(head['codes']['sha256'],embedding['codes']['sha256'])
        self.assertTrue(m['fixture_only']);self.assertFalse(m['real_model_qualified'])
        for name,sha in m['producer_source_sha256'].items():self.assertEqual(hashlib.sha256((ROOT/'tools/reference'/name).read_bytes()).hexdigest(),sha)


if __name__=='__main__':unittest.main()
