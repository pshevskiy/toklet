"""Portable numerical/boundary checks and first-failure retention."""
from pathlib import Path
import hashlib,json,math,struct,tempfile,unittest
from unittest import mock
import gemma_table_math as producer
import check_gemma_table_math as checker

class TableWitness(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)/'capture';producer.capture(self.directory)
        self.report=json.loads((self.directory/'math.json').read_bytes())
    def rewrite(self):
        raw=(json.dumps(self.report,indent=2)+'\n').encode();(self.directory/'math.json').write_bytes(raw)
        m=json.loads((self.directory/'manifest.json').read_bytes())
        for name in ('math.json','math.bin'):
            raw=(self.directory/name).read_bytes();m['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        (self.directory/'manifest.json').write_text(json.dumps(m)+'\n')
    def alter_array(self,entry):
        path=self.directory/'math.bin';body=bytearray(path.read_bytes());body[entry['offset']]^=1;path.write_bytes(body)
        changed=body[entry['offset']:entry['offset']+entry['bytes']];entry['sha256']=hashlib.sha256(changed).hexdigest()
        for a in self.report['arrays']:
            if a['name']==entry['name']:a['sha256']=entry['sha256']
        self.rewrite()
    def test_complete_independent_replay(self):
        result=checker.replay(self.directory)
        self.assertEqual(result['binary_bytes'],22664);self.assertEqual(result['negative_zero_weights'],192)
    def test_odd_bf16_rows_and_padding(self):
        case=self.report['cases'][1]
        tile=case['partitions'][1]['chunks'][1]['views']['scales']
        self.assertEqual(tile['source_offset'],14)
        self.assertEqual(case['partitions'][1]['chunks'][0]['views']['scales']['zero_padding_bytes'],2)
        self.assertGreater(checker.replay(self.directory)['two_byte_metadata_views'],0)
    def test_negative_zero_reconstruction(self):
        entry=self.report['cases'][0]['arrays']['reconstructed'];raw=(self.directory/'math.bin').read_bytes()
        self.assertEqual(raw[entry['offset']:entry['offset']+512],bytes.fromhex('00000080')*128)
    def test_coherent_code_mutation(self):
        self.alter_array(self.report['cases'][0]['arrays']['codes'])
        with self.assertRaises(ValueError):checker.replay(self.directory)
    def test_coherent_gather_mutation(self):
        self.alter_array(self.report['cases'][0]['arrays']['gather'])
        with self.assertRaises(ValueError):checker.replay(self.directory)
    def test_padding_digest_mutation(self):
        self.report['cases'][1]['partitions'][1]['chunks'][0]['views']['scales']['padded_sha256']='0'*64;self.rewrite()
        with self.assertRaises(ValueError):checker.replay(self.directory)
    def test_chunk_overlap(self):
        self.report['cases'][0]['partitions'][1]['chunks'][1]['first_row']=6;self.rewrite()
        with self.assertRaises(ValueError):checker.replay(self.directory)
    def test_output_placement(self):
        indices=self.report['cases'][0]['partitions'][1]['chunks'][0]['output_indices'];indices[0],indices[1]=indices[1],indices[0];self.rewrite()
        with self.assertRaises(ValueError):checker.replay(self.directory)
    def test_shared_source_counted_once(self):
        self.report['cases'][0]['source_bytes_counted_once']*=2;self.rewrite()
        with self.assertRaises(ValueError):checker.replay(self.directory)
    def test_late_gather_failure_is_atomic(self):
        destination=[42.]*256;before=producer.raw32(destination);weights=[0.]*128+[math.inf]*64
        with self.assertRaisesRegex(ValueError,'reconstruction overflow'):producer.gather(3,64,weights,[1,0,2,1],destination)
        self.assertEqual(producer.raw32(destination),before)
    def test_late_invalid_id_is_atomic(self):
        destination=[42.]*128;before=producer.raw32(destination)
        with self.assertRaisesRegex(ValueError,'index or destination'):producer.gather(3,64,[0.]*192,[0,1<<64],destination)
        self.assertEqual(producer.raw32(destination),before)
    def test_checked_allocation_and_interval(self):
        with self.assertRaisesRegex(ValueError,'allocation overflow'):producer.checked_layout((1<<64)-1,128)
        for args in ((19,0,0),(19,8,7),(19,19,20),(19,1<<64,(1<<64)+1)):
            with self.assertRaisesRegex(ValueError,'row interval'):producer.interval(*args)
    def test_missing_source_is_atomic(self):
        destination=[42.]*128;before=producer.raw32(destination)
        with self.assertRaisesRegex(ValueError,'geometry or source'):producer.gather(3,64,[0.]*128,[0,2],destination)
        self.assertEqual(producer.raw32(destination),before)
    def test_unbounded_partition_rejected(self):
        with self.assertRaisesRegex(ValueError,'partition bound'):producer.partition((1<<64)-1,1)
    def test_extra_overflow_array(self):
        self.report['overflow']['arrays']['extra']=self.report['overflow']['arrays']['codes'];self.rewrite()
        with self.assertRaises(ValueError):checker.replay(self.directory)
    def test_duplicate_json(self):
        with self.assertRaisesRegex(ValueError,'duplicate field'):checker.document('{"passed":true,"passed":false}')

class FailureRetention(unittest.TestCase):
    def test_late_recipe_failure_retains_completed_case(self):
        original=OSError('original late recipe failure');recipe=producer.recipe
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'capture'
            with mock.patch.object(producer,'recipe',side_effect=[recipe(19,128),original]):
                with self.assertRaises(OSError) as raised:producer.capture(out)
            self.assertIs(raised.exception,original)
            report=json.loads((out/'math.json').read_bytes());manifest=json.loads((out/'manifest.json').read_bytes())
            self.assertFalse(report['passed']);self.assertEqual(len(report['cases']),1);self.assertEqual(len(report['arrays']),8)
            self.assertEqual(report['attempts'][-1],dict(name='rows11-k64',completed=False))
            self.assertEqual((out/'math.bin').stat().st_size,15876);self.assertFalse(manifest['passed'])
            self.assertEqual(manifest['primary_error'],'OSError: original late recipe failure')
    def test_report_write_failure_retains_binary_and_primary(self):
        original=OSError('original report write failure');open_file=Path.open
        class FailingWriter:
            def __init__(self,file):self.file=file
            def __enter__(self):return self
            def __exit__(self,*_):self.file.close()
            def write(self,text):self.file.write(text[:1]);self.file.flush();raise original
        def intercepted(path,*args,**kwargs):
            file=open_file(path,*args,**kwargs)
            return FailingWriter(file) if path.name=='math.json' and args==('x',) else file
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'capture'
            with mock.patch.object(Path,'open',intercepted):
                with self.assertRaises(OSError) as raised:producer.capture(out)
            self.assertIs(raised.exception,original)
            self.assertEqual((out/'math.bin').stat().st_size,22664)
            self.assertGreater((out/'math.json').stat().st_size,0)
            manifest=json.loads((out/'manifest.json').read_bytes());self.assertFalse(manifest['passed'])
            self.assertEqual(manifest['primary_error'],'OSError: original report write failure')
            self.assertEqual(manifest['files']['math.bin']['bytes'],22664)

if __name__=='__main__':unittest.main()
