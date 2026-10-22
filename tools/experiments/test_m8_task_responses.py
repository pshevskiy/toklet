"""Authored portable response/idle checks; no model, phone, tokenizer or VM work."""
from contextlib import ExitStack
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import m8_task_responses as m


def fixture():
    cases = [dict(id='case/'+str(i), category='code' if i < 40 else 'text',
                  prompt_tokens=[7], max_output_tokens=2) for i in range(100)]
    samples = [dict(caseID=c['id'], response=dict(ids=[1,2], sampled=2, reason=1,
               history=3, evaluated=2, pending=1, chunks=1)) for c in cases]
    return cases, dict(responses=samples)


class Responses(unittest.TestCase):
    def test_literal_decode_keeps_fences_whitespace_and_fixed_policy(self):
        cases, checked = fixture()
        text = '  ```python\nreturn "é"\n```\n'
        decoder = Mock();decoder.decode.return_value = text
        rows = m.decode_responses(cases, checked, decoder)
        self.assertEqual(len(rows), 100)
        self.assertEqual(rows[0]['visible_text'], text)
        self.assertEqual(rows[0]['visible_text_sha256'], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(rows[0]['stop_reason'], dict(kind='output_limit'))
        self.assertTrue(all(call.kwargs == dict(skip_special_tokens=True) for call in decoder.decode.call_args_list))
        self.assertEqual(rows[0]['visible_token_ids'], [1,2])

    def test_hidden_stops_remain_explicit_and_never_enter_visible_ids(self):
        cases, checked = fixture();row = checked['responses'][0]['response']
        row.update(reason=2, stopToken=151645)
        result = m.decode_responses(cases, checked, Mock(decode=Mock(return_value='literal')))
        self.assertEqual(result[0]['stop_reason'], dict(kind='stop_token', token_id=151645))
        self.assertEqual(result[0]['visible_token_ids'], [1,2])

    def test_missing_reordered_bool_id_or_unknown_stop_rejects(self):
        cases, original = fixture()
        for mutate in (lambda c:c['responses'].pop(), lambda c:c['responses'].reverse(),
                       lambda c:c['responses'][0]['response'].update(ids=[True]),
                       lambda c:c['responses'][0]['response'].update(reason=True),
                       lambda c:c['responses'][0]['response'].update(reason=2,stopToken=17)):
            checked=copy.deepcopy(original);mutate(checked)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                m.decode_responses(cases, checked, Mock(decode=Mock(return_value='text')))

    def test_raw_derivative_preserves_original_json_and_rejects_error_marker(self):
        literal='{"negativeZero":-0.0,"time":1e-7}'
        raw=m.raw_derivative('host chatter\nTOKLET_TASK_CASE_JSON='+literal+'\n')
        self.assertEqual(json.loads(raw),dict(marker='TOKLET_TASK_CASE_JSON',original_json=literal))
        for text in ['host chatter', 'TOKLET_TASKS_ERROR=bad']:
            with self.assertRaises(ValueError):m.raw_derivative(text)

    def test_capture_binds_consumed_bytes_and_rejects_changed_or_oversize(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(m,'ROOT',Path(directory)):
            path=Path(directory)/'data';path.write_bytes(b'first');pins={}
            self.assertEqual(m.capture(path,pins),b'first');expected=pins['data']
            path.write_bytes(b'second')
            with self.assertRaises(ValueError):m.capture(path,pins,expected)
            with self.assertRaises(ValueError):m.capture(path,{},maximum=1)


class Idle(unittest.TestCase):
    def setup_barrier(self):
        barrier=m.IdleBarrier.__new__(m.IdleBarrier)
        barrier.pins={};barrier.pids=[111,222];barrier.observations=[]
        barrier.selected=dict(device_identifier='authored-device')
        barrier.remote='/var/containers/Bundle/Application/authored/TokletHarness.app'
        barrier.app=Path('/authored/app');barrier.signed={'binary':'authored'}
        barrier.profile=dict(expires_utc='2030-01-01T00:00:00+00:00')
        current=dict(info=dict(outcome='success'),result=dict(deviceIdentifier='authored-device',
            matchingBundleIdentifier=m.physical.BUNDLE,apps=[dict(bundleIdentifier=m.physical.BUNDLE,
            builtByDeveloper=True,url='file://'+barrier.remote)]))
        processes=dict(result=dict(runningProcesses=[]))
        barrier.command=Mock(return_value=(b'',b''))
        barrier.device=Mock(side_effect=[{},current,processes]*2)
        return barrier,current,processes

    def invoke(self, barrier):
        with patch.object(m,'guard'),patch.object(m.package,'group_exists',return_value=False), \
             patch.object(m,'utc_now',return_value=datetime.datetime(2026,1,1,tzinfo=datetime.timezone.utc)), \
             patch.object(m.physical,'admit_device',return_value={'physical':True}), \
             patch.object(m.physical,'inventory',return_value=barrier.signed), \
             patch.object(m.physical,'stop_remote') as stop:
            barrier.check();stop.assert_not_called()

    def test_repeated_checks_use_only_host_pids_and_never_signal_phone(self):
        barrier,_,_=self.setup_barrier();self.invoke(barrier);self.invoke(barrier)
        self.assertEqual(len(barrier.observations),2)
        self.assertEqual(barrier.command.call_count,2)
        self.assertEqual(barrier.command.call_args.args[0],['ps','-p','111,222','-o','pid='])
        self.assertTrue(all(o['remote_absent'] and o['local_groups_absent'] for o in barrier.observations))

    def test_local_live_pid_or_group_prevents_all_device_queries(self):
        barrier,_,_=self.setup_barrier();barrier.command.return_value=(b'111',b'')
        with self.assertRaisesRegex(ValueError,'local producer'):self.invoke(barrier)
        barrier.device.assert_not_called()
        barrier.command.return_value=(b'',b'')
        with patch.object(m,'guard'),patch.object(m.package,'group_exists',return_value=True):
            with self.assertRaisesRegex(ValueError,'local producer'):barrier.check()
        barrier.device.assert_not_called()

    def test_replaced_installation_live_remote_or_foreign_process_rejects(self):
        for kind in ('installation','live','foreign'):
            barrier,current,processes=self.setup_barrier()
            if kind=='installation':current['result']['apps'][0]['url']='file:///different/app'
            else:processes['result']['runningProcesses']=[dict(processIdentifier=987654,
                executable=barrier.remote+'/TokletHarness' if kind=='live' else '/foreign/TokletHarness')]
            barrier.device=Mock(side_effect=[{},current,processes])
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.invoke(barrier)
            self.assertFalse(barrier.observations)
            self.assertNotIn('987654',barrier.command.call_args.args[0])

    def test_profile_clock_advance_rejects_before_next_command(self):
        barrier,_,_=self.setup_barrier()
        with patch.object(m,'guard'),patch.object(m.package,'group_exists',return_value=False), \
             patch.object(m.physical,'admit_device',return_value={'physical':True}), \
             patch.object(m.physical,'inventory',return_value=barrier.signed), \
             patch.object(m,'utc_now',side_effect=[datetime.datetime(2029,1,1,tzinfo=datetime.timezone.utc),
                                                  datetime.datetime(2031,1,1,tzinfo=datetime.timezone.utc)]):
            barrier.check()
            with self.assertRaisesRegex(ValueError,'profile expired'):barrier.check()
        self.assertEqual(barrier.command.call_count,1);self.assertEqual(barrier.device.call_count,3)
        self.assertEqual(len(barrier.observations),1)

    def test_timeout_and_first_capture_failure_keep_second_stream_and_private_cause(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'artifacts') as directory:
            folder=Path(directory);barrier,_,_=self.setup_barrier()
            barrier.output=folder;barrier.env={};barrier.commands=[]
            barrier.command=m.IdleBarrier.command.__get__(barrier)
            stderr=folder/'timeout.stderr';stderr.write_bytes(b'captured second stream')
            row=dict(command=['query','authored-device'],exit_code=-15,
                primary_error='TimeoutError: authored-device',cleanup_errors=['cleanup authored-device'],
                pid=333,stdout=str((folder/'timeout.stdout').relative_to(m.ROOT)),
                stderr=str(stderr.relative_to(m.ROOT)))
            with patch.object(m,'guard'),patch.object(m.package,'owned',return_value=row):
                with self.assertRaisesRegex(ValueError,'TimeoutError') as failure:
                    barrier.command(['query','authored-device'],'timeout')
            saved=barrier.commands[0]
            self.assertEqual(saved['primary_error'],'TimeoutError: authored-device')
            self.assertEqual(saved['cleanup_errors'],['cleanup authored-device'])
            self.assertEqual(len(saved['capture_errors']),1);self.assertIn('stdout',saved['capture_errors'][0])
            self.assertEqual(barrier.pins[str(stderr.relative_to(m.ROOT))],hashlib.sha256(stderr.read_bytes()).hexdigest())
            private=folder/'private.json';m.write_bound(private,saved,barrier.pins)
            self.assertEqual(json.loads(private.read_bytes())['primary_error'],row['primary_error'])
            public=m.physical.redact_record(dict(primary_error=str(failure.exception),idle=barrier.public()),
                                            barrier.selected,barrier.remote)
            self.assertNotIn('authored-device',json.dumps(public))

    def test_device_details_use_strict_hardware_identity_without_result_identifier(self):
        barrier,_,_=self.setup_barrier();barrier.device=m.IdleBarrier.device.__get__(barrier)
        barrier.selected.update(device_udid='authored-udid',product_type='iPhone15,2',os_version='27.0.1')
        data=dict(info=dict(outcome='success'),result=dict(
            hardwareProperties=dict(udid='authored-udid',productType='iPhone15,2',
                                    reality='physical',marketingName='iPhone 14 Pro'),
            deviceProperties=dict(osVersionNumber='27.0.1',developerModeStatus='enabled',ddiServicesAvailable=True),
            connectionProperties=dict(pairingState='paired',transportType='wired',tunnelState='connected')))
        barrier.command.return_value=(json.dumps(data).encode(),b'')
        self.assertEqual(barrier.device(['info','details'],'details'),data)
        for change in ('hardware','outcome'):
            altered=copy.deepcopy(data)
            if change=='hardware':altered['result']['hardwareProperties']['udid']='other'
            else:altered['info']['outcome']='failure'
            barrier.command.return_value=(json.dumps(altered).encode(),b'')
            with self.subTest(change=change),self.assertRaises(ValueError):barrier.device(['info','details'],'details')

    def test_wrong_device_result_is_not_absence(self):
        barrier,_,_=self.setup_barrier();barrier.device=m.IdleBarrier.device.__get__(barrier)
        barrier.command.return_value=(json.dumps(dict(info=dict(outcome='success'),
            result=dict(deviceIdentifier='other'))).encode(),b'')
        with self.assertRaisesRegex(ValueError,'identity differs'):barrier.device(['info','processes'],'query')


class Normalization(unittest.TestCase):
    def invoke(self, directory, failure=None):
        root=Path(directory).resolve();(root/'artifacts').mkdir(exist_ok=True)
        source=root/'producer.py';source.write_text('# authored producer\n')
        cases,checked=fixture();packets={p:dict(checked=checked,raw=b'{"app":"original"}\n') for p in m.PLATFORMS}
        q=dict(execution=dict(nested_receipts={'qwen8-tasks-full-run':dict(signed_bundle={})}))
        order=[]
        class Barrier:
            def __init__(self,*args):self.calls=0;self.commands=[];self.selected={};self.remote='authored'
            def check(self):
                self.calls+=1;order.append('absence')
                if failure==self.calls:raise ValueError('authored absence failure')
            def public(self):return dict(observations=self.calls)
        class Decoder:
            @classmethod
            def from_str(cls,value):order.append('construct');return cls()
            def decode(self,ids,skip_special_tokens):return 'literal'
        with ExitStack() as stack:
            for name,value in [('ROOT',root),('__file__',str(source)),('IdleBarrier',Barrier)]:stack.enter_context(patch.object(m,name,value))
            stack.enter_context(patch.object(m,'admit_source',return_value=(q,cases,packets,[111])))
            stack.enter_context(patch.object(m,'tokenizer_inputs',return_value=('authored',{'version':'authored','files':{}})))
            stack.enter_context(patch.object(m,'guard'))
            stack.enter_context(patch.dict(sys.modules,{'tokenizers':SimpleNamespace(Tokenizer=Decoder)}))
            stack.enter_context(patch('builtins.print'))
            result=m.normalize(root/'artifacts/output')
        return result,order,root/'artifacts/output'

    def test_decoder_constructs_after_absence_and_derivatives_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            result,order,out=self.invoke(directory)
            self.assertTrue(result['passed']);self.assertEqual(order,['absence','construct','absence'])
            record=json.loads((out/'mac.json').read_bytes())
            self.assertEqual(record['format'],'toklet-m8-app-task-response-derivative-v1')
            self.assertFalse(record['quality_completed']);self.assertEqual(len(record['responses']),100)
            self.assertEqual(record['raw_sha256'],hashlib.sha256((out/'mac.raw.jsonl').read_bytes()).hexdigest())
            self.assertEqual(result['new_vm_runs'],0);self.assertEqual(result['new_generated_program_executions'],0)

    def test_failed_initial_absence_never_constructs_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            result,order,out=self.invoke(directory,failure=1)
            self.assertFalse(result['passed']);self.assertEqual(order,['absence'])
            self.assertFalse((out/'mac.json').exists());self.assertTrue((out/'private-execution.json').exists())

    def test_failed_final_absence_keeps_partial_outputs_but_never_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            result,order,out=self.invoke(directory,failure=2)
            self.assertFalse(result['passed']);self.assertTrue((out/'mac.json').exists())
            self.assertFalse(json.loads((out/'normalization.json').read_bytes())['passed'])

    def test_existing_and_dangling_outputs_reject_before_source_admission(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(m,'ROOT',Path(directory)),patch.object(m,'admit_source') as admit:
            root=Path(directory).resolve();(root/'artifacts').mkdir();existing=root/'artifacts/existing';existing.mkdir()
            dangling=root/'artifacts/dangling';dangling.symlink_to(root/'missing')
            for path in [existing,dangling]:
                with self.assertRaises(ValueError):m.normalize(path)
            admit.assert_not_called()


if __name__=='__main__':unittest.main()
