"""Proposed portable controls; no dataset/tokenizer/model is read."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
import prepare_llama3_loss1024 as prep


def manifest(raw):
    return dict(format=prep.FORMAT,repository=prep.REPOSITORY,revision=prep.REVISION,dataset=prep.DATASET,
                tokenizer=prep.TOKENIZER,corpus=prep.CORPUS,parents=prep.PARENTS,blocks=prep.BLOCKS,policy=prep.POLICY,
                complete_corpus_tokens=1200,model_executed=False,
                tokens=dict(file='tokens.u32le',dtype='u32le',count=1025,bytes=4100,sha256=prep.digest(raw)))


class Inputs(unittest.TestCase):
    def test_prefix_selection_and_exact_unique_target_partition(self):
        raw=prep.select(list(range(1200)));self.assertEqual(len(raw),4100)
        self.assertEqual(list(struct.unpack('<1025I',raw)),list(range(1025)))
        targets=[]
        for block in prep.BLOCKS:
            self.assertEqual(block['input_end']-block['input_start'],257)
            self.assertEqual(block['score_rows'],[0,256]);self.assertEqual(block['reset_position'],0)
            targets.extend(range(*block['global_target_indices']))
        self.assertEqual(targets,list(range(1,1025)))
        for bad in [list(range(1024)),[False]+list(range(1,1025)),[128256]+list(range(1,1025))]:
            with self.assertRaises(ValueError):prep.select(bad)

    def test_literal_contract_and_mutation_rejection(self):
        raw=prep.select(list(range(1025)));m=manifest(raw);prep.validate(m)
        changes=[lambda x:x['blocks'][1].update(input_start=257),lambda x:x['policy'].update(add_special_tokens=True),
                 lambda x:x['policy'].update(mean_nll_increase_limit=.051),lambda x:x['tokens'].update(file='../escape'),
                 lambda x:x['corpus'].update(recipe='strip rows'),lambda x:x['tokenizer'].update(sha256='0'*64),
                 lambda x:x.update(complete_corpus_tokens=True)]
        for change in changes:
            bad=copy.deepcopy(m);change(bad)
            with self.assertRaises(ValueError):prep.validate(bad)

    def test_reader_pins_manifest_and_exact_consumed_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);raw=prep.select(list(range(1025)));m=manifest(raw)
            (folder/'tokens.u32le').write_bytes(raw);metadata=(json.dumps(m)+'\n').encode();(folder/'manifest.json').write_bytes(metadata)
            _,ids=prep.load(folder,prep.digest(metadata));self.assertEqual(ids[-1],1024)
            with self.assertRaisesRegex(ValueError,'manifest'):prep.load(folder,'0'*64)
            (folder/'tokens.u32le').write_bytes(raw[:-4]+struct.pack('<I',1))
            with self.assertRaisesRegex(ValueError,'identity'):prep.load(folder,prep.digest(metadata))


if __name__=='__main__':unittest.main()
