import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import check_qwen35_9b_payload as check
import qwen35_9b_storage_fixture as fixture
from test_check_qwen35_payload import Controls as ArithmeticControls


class ClosedCapture(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.native=self.root/'native';self.native.mkdir()
        self.binary=self.root/'pinned-binary';self.binary.write_bytes(b'unit fixture binary')
        self.source=self.root/'source.rs';self.source.write_text('unit source fixture\n')
        self.source_inputs=self.root/'source-inputs.json'
        self.source_inputs.write_text(json.dumps(dict(source_pins={'source.rs':self.sha(self.source)})))
        test='mlx::qwen35::nine_b::payload::tests::actual_payload_probe'
        self.inputs=dict(binary=str(self.binary),binary_sha256=self.sha(self.binary),test=test,
            command=['/usr/bin/time','-l',str(self.binary),test,'--exact','--ignored','--nocapture','--test-threads=1'],
            child_allocator={'MallocLargeCache':'0','MallocAggressiveMadvise':None,'MallocNanoZone':None},
            source_pin_count=1,source_inputs=str(self.source_inputs),source_inputs_sha256=self.sha(self.source_inputs),
            source_worktree=str(self.root))
        (self.native/'console.log').write_text('test result: ok. 1 passed; 0 failed; 0 ignored;\n')
        (self.native/'time-and-errors.log').write_text('     4096  peak memory footprint\n')
        self.receipt=dict(format='toklet-qwen35-storage-native-v1',passed=True,group_absent=True,error=None,exit_code=0,
            changed_inputs=[],binary_sha256=self.inputs['binary_sha256'],lifetime_peak_physical_bytes=4096,elapsed_seconds=.1,
            owned_pid=42,numerical_execution_qualified=False,managed_adoption_qualified=False)
        self.seal()

    def sha(self,path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def seal(self):
        (self.native/'inputs.json').write_text(json.dumps(self.inputs))
        for name,key in [('inputs.json','inputs_sha256'),('console.log','console_sha256'),('time-and-errors.log','time_log_sha256')]:
            self.receipt[key]=self.sha(self.native/name)
        (self.native/'checks.json').write_text(json.dumps(self.receipt))

    def reject(self):
        with self.assertRaises((ValueError,KeyError,OSError)):check.closed_capture(self.native)

    def test_closed_source_bound_control(self):
        self.assertIs(check.closed_capture(self.native)['passed'],True)

    def test_boolean_and_nonfinite_numbers_reject(self):
        for field,value in [('exit_code',False),('lifetime_peak_physical_bytes',True),('owned_pid',True),
                            ('elapsed_seconds',True),('elapsed_seconds',float('nan')),('elapsed_seconds',float('inf'))]:
            with self.subTest(field=field,value=value):
                previous=self.receipt[field];self.receipt[field]=value;self.seal();self.reject();self.receipt[field]=previous

    def test_scope_outcome_and_group_changes_reject(self):
        for field,value in [('passed',False),('group_absent',False),('error','failed'),('exit_code',1),
                            ('changed_inputs',['source.rs']),('numerical_execution_qualified',True),('managed_adoption_qualified',True)]:
            with self.subTest(field=field):
                previous=self.receipt[field];self.receipt[field]=value;self.seal();self.reject();self.receipt[field]=previous

    def test_changed_console_time_or_binary_reject(self):
        for path in [self.native/'console.log',self.native/'time-and-errors.log',self.binary]:
            previous=path.read_bytes();path.write_bytes(previous+b'changed');self.reject();path.write_bytes(previous)

    def test_resealed_peak_and_zero_test_claim_reject(self):
        (self.native/'time-and-errors.log').write_text('8192 peak memory footprint\n');self.seal();self.reject()
        (self.native/'time-and-errors.log').write_text('4096 peak memory footprint\n')
        (self.native/'console.log').write_text('test result: ok. 0 passed; 0 failed;\n');self.seal();self.reject()

    def test_resealed_command_and_source_selection_reject(self):
        self.inputs['command'][-1]='--test-threads=2';self.seal();self.reject()
        self.inputs['command'][-1]='--test-threads=1';self.inputs['test']='mlx::qwen35::payload::tests::actual_payload_probe';self.seal();self.reject()

    def test_resealed_allocator_and_source_pin_coverage_reject(self):
        for field,value in [('child_allocator',{'MallocLargeCache':'0','MallocAggressiveMadvise':None,'MallocNanoZone':'0'}),
                            ('source_pin_count',True),('source_pin_count',2)]:
            previous=self.inputs[field];self.inputs[field]=value;self.seal();self.reject();self.inputs[field]=previous

    def test_changed_source_or_source_manifest_reject(self):
        self.source.write_text('changed source\n');self.reject()
        self.source_inputs.write_text('{}');self.reject()


    def prepare_consumer_update(self):
        entries=[];pins={'source.rs':self.sha(self.source)}
        for number,name in enumerate(sorted(check.CONSUMER_PATHS)):
            file=self.root/name;file.parent.mkdir(parents=True,exist_ok=True);snapshot=self.root/('old-'+str(number)+'.py')
            snapshot.write_text('old reviewed consumer'+str(number));pins[name]=self.sha(snapshot)
            file.write_text('corrected consumer'+str(number))
            entries.append(dict(path=name,before_sha256=self.sha(snapshot),after_sha256=self.sha(file),snapshot=str(snapshot)))
        self.source_inputs.write_text(json.dumps(dict(source_pins=pins)))
        self.inputs.update(source_pin_count=3,source_inputs_sha256=self.sha(self.source_inputs));self.seal()
        path=self.root/'consumer-update.json';path.write_text(json.dumps(dict(format='toklet-qwen35-reader-consumer-update-v1',source_inputs_sha256=self.inputs['source_inputs_sha256'],updates=entries)))
        return path

    def test_consumer_changes_reject_without_explicit_hash_bound_update(self):
        path=self.prepare_consumer_update();self.reject()
        update=check.consumer_update(path,self.sha(path))
        self.assertIs(check.closed_capture(self.native,update)['passed'],True)
        with self.assertRaises(ValueError):check.consumer_update(path,'0'*64)

    def test_consumer_update_cannot_authorize_producer_or_unknown_paths(self):
        path=self.prepare_consumer_update();update=check.consumer_update(path,self.sha(path))
        update['updates'][0]['path']='source.rs'
        path.write_text(json.dumps(update))
        with self.assertRaises(ValueError):check.consumer_update(path,self.sha(path))
        update=check.consumer_update(self.prepare_consumer_update(),self.sha(path))
        self.source.write_text('changed producer')
        with self.assertRaises(ValueError):check.closed_capture(self.native,update)

    def test_timing_enclosure_and_order(self):
        check.timing_enclosure(dict(load_ns=10,load_and_capture_ns=20),.00000003)
        for load,capture,elapsed in [(20,10,1),(10,31,.00000003),(0,20,1),(True,20,1),(10,20,True),(10,20,float('inf')),(10,20,1e300)]:
            with self.subTest(load=load,capture=capture,elapsed=elapsed):
                with self.assertRaises(ValueError):check.timing_enclosure(dict(load_ns=load,load_and_capture_ns=capture),elapsed)


class IndependentFixture(unittest.TestCase):
    def test_generator_reproduces_all_frozen_bytes_and_separates_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'generated';record=fixture.fixture(out)
            original=Path(__file__).resolve().parents[2]/'tests/fixtures/qwen35-9b-storage-v1'
            for name in ['stream.bin','inventory.json','README.md']:self.assertEqual((out/name).read_bytes(),(original/name).read_bytes())
            self.assertNotEqual(record['expected_rows'][fixture.EMBED+'.weight'],record['expected_rows'][fixture.HEAD+'.weight'])
            self.assertEqual(record['source_bytes'],211)
            self.assertEqual(record['expected_small_fp32_bits'][fixture.NORM][-1],0x80000000)


if __name__=='__main__':unittest.main()
