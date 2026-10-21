from pathlib import Path
import io,json,runpy,sys,unittest
from unittest.mock import patch
ROOT=Path('/Users/coloner/work/rust/toklet');sys.path.insert(0,str(ROOT/'tools/experiments'))
import m7_physical_measurement as m
DRIVER=Path(__file__).with_name('driver.py').resolve()
class Capture(io.StringIO):
 def close(self):self.saved=self.getvalue();super().close()
class DriverChecks(unittest.TestCase):
 def execute(self,mode):
  calls=[];source_changed=False;receipt_changed=False;capture=Capture();original=Path.open
  def opened(path,access='r',*args,**kwargs):
   if path==DRIVER.parent/'execution.json' and access=='x':return capture
   if path.name=='measurement.json':return io.BytesIO(b'replaced receipt' if receipt_changed else b'fixed receipt')
   if path==Path(m.__file__).resolve() and source_changed:return io.BytesIO(b'changed source')
   return original(path,access,*args,**kwargs)
  def main(*args,**kwargs):
   nonlocal source_changed,receipt_changed
   calls.append((args,kwargs))
   if mode=='primary-and-endpoint':source_changed=True;raise RuntimeError('retained primary')
   if mode=='post-call-source':source_changed=True
   if mode=='changed-ready' and len(calls)==3:receipt_changed=True
   return 1 if mode=='returned-failure' else 0
  with patch.object(m,'main',side_effect=main),patch.object(Path,'open',opened):
   with self.assertRaises(SystemExit) as stopped:runpy.run_path(str(DRIVER),run_name='__main__')
  return stopped.exception.code,json.loads(capture.saved),calls
 def test_complete_sequence_uses_frozen_ready_sha(self):
  code,record,calls=self.execute('pass');self.assertEqual(code,0);self.assertEqual(len(calls),24)
  self.assertIsNone(record['primary_error']);self.assertEqual(record['endpoint_errors'],[])
  self.assertEqual(len(record['ready_receipts']),6)
  for offset in range(0,24,4):
   self.assertEqual(calls[offset+2][1]['transfer_sha'],calls[offset+3][1]['transfer_sha'])
   self.assertIn(calls[offset+2][1]['transfer_sha'],record['ready_receipts'].values())
 def test_primary_survives_final_source_failure(self):
  code,record,calls=self.execute('primary-and-endpoint');self.assertEqual(code,1)
  self.assertIn('retained primary',record['primary_error']);self.assertEqual(len(record['endpoint_errors']),1)
  self.assertEqual(len(calls),1)
 def test_completed_stage_retained_after_source_failure(self):
  code,record,calls=self.execute('post-call-source');self.assertEqual(code,1)
  self.assertEqual(len(record['results']),1);self.assertEqual(record['results'][0]['exit_code'],0)
  self.assertEqual(len(calls),1);self.assertEqual(len(record['endpoint_errors']),1)
 def test_replaced_ready_blocks_second_trial(self):
  code,record,calls=self.execute('changed-ready');self.assertEqual(code,1)
  self.assertEqual(len(calls),3);self.assertEqual(len(record['results']),3)
  self.assertEqual(len(record['ready_receipts']),1);self.assertEqual(len(record['endpoint_errors']),1)
 def test_returned_failure_stops_and_retains_record(self):
  code,record,calls=self.execute('returned-failure');self.assertEqual(code,1)
  self.assertEqual(len(calls),1);self.assertEqual(record['results'][0]['exit_code'],1)
  self.assertIn('storage-proof failed',record['primary_error'])
 def test_existing_execution_prevents_children(self):
  original=Path.exists
  with patch.object(Path,'exists',lambda path:True if path==DRIVER.parent/'execution.json' else original(path)),patch.object(m,'main') as child:
   with self.assertRaises(FileExistsError):runpy.run_path(str(DRIVER),run_name='__main__')
   child.assert_not_called()
if __name__=='__main__':unittest.main()
