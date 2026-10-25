"""Post-capture mutations of actual complete outputs; no synthetic GPU credit."""
from pathlib import Path
import copy,hashlib,io,struct,sys,unittest
from unittest import mock
import check_gemma_real_projections as check

class ReaderFailures(unittest.TestCase):
    def test_duplicate_json(self):
        with self.assertRaisesRegex(ValueError,'duplicate JSON field'):
            check.parse(b'GEMMA_REAL_META {"device":"one","device":"two"}\n')
    def test_nonfinite_json(self):
        with self.assertRaisesRegex(ValueError,'finite JSON required'):
            check.parse(b'GEMMA_REAL_META {"value":NaN}\n')
    def test_hash_growth_bound(self):
        path=mock.Mock();stream=io.BytesIO(b'123456789');path.open.return_value=stream
        with self.assertRaisesRegex(ValueError,'exceeds exact byte bound'):check.digest(path,4)
        self.assertTrue(stream.closed)
    def test_hash_short(self):
        path=mock.Mock();path.open.return_value=io.BytesIO(b'123')
        with self.assertRaisesRegex(ValueError,'file is short'):check.digest(path,4)
    def test_second_open_closes_first(self):
        model=mock.Mock();model.stat.return_value.st_size=check.MODEL_BYTES
        first=io.BytesIO(b'');model.open.return_value=first
        frozen=mock.MagicMock();array=frozen.__truediv__.return_value
        array.stat.return_value.st_size=26333184
        original=OSError('original second-open error');array.open.side_effect=original
        with mock.patch.object(check,'digest',side_effect=[check.MODEL_SHA,'037ca2048d2ee9f1fea58bd3ea1ba6008543dc8a13e7a064e2d755817c7f021e']),mock.patch.object(check,'load',side_effect=[{'passed':True},{}]):
            with self.assertRaises(OSError) as raised:check.Assets(model,frozen)
        self.assertIs(raised.exception,original);self.assertTrue(first.closed)

class RealCaptureMutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stdout,cls.binary,model,frozen=map(Path,sys.argv[1:5])
        with stdout.open('rb') as stream:cls.rows=check.parse(stream.read(32*1024**2+1))
        cls.assets=check.Assets(model,frozen)
        try:cls.baseline=check.verify(cls.rows,cls.binary,cls.assets)
        except BaseException:cls.assets.close();raise
    @classmethod
    def tearDownClass(cls):cls.assets.close()
    def setUp(self):self.rows=copy.deepcopy(self.rows);self.overrides={}
    def verify(self):return check.verify(self.rows,self.binary,self.assets,overrides=self.overrides)
    def replace(self,step_index,value):
        step=self.rows['STEP'][step_index];entry=step['output']
        with self.binary.open('rb') as stream:
            stream.seek(entry['offset']);raw=bytearray(stream.read(entry['bytes']))
        raw[:4]=struct.pack('<f',value);raw=bytes(raw);self.overrides[entry['offset']]=raw
        sha=hashlib.sha256(raw).hexdigest()
        for label in ('ARRAY','ARRAY_ATTEMPT'):
            for item in self.rows[label]:
                if item['offset']==entry['offset']:item['sha256']=sha
        step['output']['sha256']=sha
        for item in (step,self.rows['STEP_OUTCOME'][step_index]):
            item['caller_sha256']=sha
            n=entry['shape'][1];tokens=entry['shape'][0]
            item['device']['output']=check.identity(raw+check.SENTINEL*((4-tokens)*n)+check.GUARDS)
    def next_value(self,step_index):
        entry=self.rows['STEP'][step_index]['output']
        with self.binary.open('rb') as stream:
            stream.seek(entry['offset']);bits=struct.unpack('<I',stream.read(4))[0]
        next_bits=1 if bits==0x80000000 else bits+1 if bits<0x7f7fffff else bits-1
        return struct.unpack('<f',struct.pack('<I',next_bits))[0]
    def test_complete_original(self):
        self.assertIs(self.baseline['passed'],True)
        self.assertEqual(self.baseline['cases'],276)
    def test_unexpected_meta(self):
        self.rows['META'][0]['unqualified']=True
        with self.assertRaises(ValueError):self.verify()
    def test_boolean_word_count(self):
        for label in ('STEP','STEP_OUTCOME'):self.rows[label][0]['device']['status']['words']=True
        with self.assertRaises(ValueError):self.verify()
    def test_original_weight_hash(self):
        self.rows['CASE'][0]['loaded_spans'][0]['sha256']='0'*64
        with self.assertRaises(ValueError):self.verify()
    def test_original_weight_offset(self):
        self.rows['CASE'][0]['loaded_spans'][0]['offset']+=4
        with self.assertRaises(ValueError):self.verify()
    def test_device_weight_digest(self):
        for label in ('STEP','STEP_OUTCOME'):self.rows[label][0]['device']['codes']['sha256']='0'*64
        with self.assertRaises(ValueError):self.verify()
    def test_device_guard(self):
        for label in ('STEP','STEP_OUTCOME'):self.rows[label][0]['device']['output']['guards'][0]=0
        with self.assertRaises(ValueError):self.verify()
    def test_device_status_digest(self):
        for label in ('STEP','STEP_OUTCOME'):self.rows[label][0]['device']['status']['sha256']='0'*64
        with self.assertRaises(ValueError):self.verify()
    def test_partition_input(self):
        self.rows['STEP_ATTEMPT'][2]['input_sha256']=self.rows['STEP_ATTEMPT'][3]['input_sha256']
        with self.assertRaises(ValueError):self.verify()
    def test_coherent_repeat_within_gate(self):
        self.replace(1,self.next_value(1))
        with self.assertRaisesRegex(ValueError,'repeat output bits differ'):self.verify()
    def test_coherent_partition(self):
        self.replace(2,self.next_value(2))
        with self.assertRaisesRegex(ValueError,'partition output bits differ'):self.verify()
    def test_coherent_wrong_result_all_phases(self):
        for index in (0,1,2,6):self.replace(index,1000.)
        with self.assertRaisesRegex(ValueError,'GPU rounded-mathematical gate failed'):self.verify()
    def test_missing_completed_record(self):
        self.rows['STEP'].pop()
        with self.assertRaises(ValueError):self.verify()

if __name__=='__main__':
    if len(sys.argv)!=5:raise SystemExit('native-stdout native-output-binary checkpoint frozen-reference-directory')
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (ReaderFailures,RealCaptureMutations))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
