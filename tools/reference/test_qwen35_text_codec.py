import copy
import json
from pathlib import Path
import tempfile
import unittest
import check_qwen35_text_codec as check


def rows():return [dict(iteration=i,warmup=i<3,load_ns=1000,encode_ns=200,decode_ns=50,stream_ns=100,prompt_ids=47,decode_ids=8) for i in range(13)]
def environment():
    commands=dict(hardware=['/usr/sbin/sysctl','-n','hw.model','machdep.cpu.brand_string','hw.memsize'],os=['/usr/bin/sw_vers'],configuration=['/usr/bin/pmset','-g','custom'],power=['/usr/bin/pmset','-g','batt'],thermal=['/usr/bin/pmset','-g','therm'],memory_pressure=['/usr/bin/memory_pressure','-Q'])
    output=dict(hardware='M1 Pro',os='macOS',configuration='stable',power="Now drawing from 'AC Power'\n",thermal='Note: No thermal warning level has been recorded\nNote: No performance warning level has been recorded\nNote: No CPU power status has been recorded\n',memory_pressure='System-wide memory free percentage: 50%')
    start={k:dict(command=c,exit_code=0,output=output[k]) for k,c in commands.items()}
    return dict(start=start,end=copy.deepcopy(start),sampling='External start/end CLI observations; text-only processes, no continuous telemetry')


class Samples(unittest.TestCase):
    def test_exact_types_preserve_boolean_distinction(self):
        self.assertFalse(check.exact(True,1));self.assertFalse(check.exact([False],[0]));self.assertTrue(check.exact(dict(x=[1,False]),dict(x=[1,False])))
    def test_samples_keep_counts_raw_outliers_and_dispersion(self):
        data=rows();data[-1]['load_ns']=1000000
        result=check.samples(data,47,8)
        self.assertEqual(result['load_ns']['median_ns'],1000);self.assertEqual(result['load_ns']['mad_ns'],0)
        self.assertEqual(result['load_ns']['maximum_ns'],1000000);self.assertEqual(len(result['load_ns']['samples_ns']),10)
    def test_missing_typed_reordered_wrong_work_samples_reject(self):
        cases=[rows()[:-1],list(reversed(rows()))]
        for key,value in [('iteration',True),('warmup',1),('load_ns',False),('decode_ns',0),('prompt_ids',46),('decode_ids',7)]:
            data=rows();data[0][key]=value;cases.append(data)
        for data in cases:
            with self.subTest(data=data[:1]),self.assertRaises(ValueError):check.samples(data,47,8)


class Closure(unittest.TestCase):
    def test_closed_typed_peak_scope_and_inventory_required(self):
        good=dict(passed=True,exit_code=0,error=None,group_absent=True,lifetime_peak_physical_bytes=50,elapsed_seconds=.5,files_sha256={})
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);check.closure(good,root,[],100)
            for key,value in [('exit_code',False),('group_absent',False),('lifetime_peak_physical_bytes',True),('lifetime_peak_physical_bytes',101),('elapsed_seconds',float('inf')),('error','failure')]:
                data={**good,key:value}
                with self.subTest(key=key),self.assertRaises(ValueError):check.closure(data,root,[],100)
            with self.assertRaises(ValueError):check.closure(good,root,['missing'],100)
    def test_duplicate_nonfinite_oversized_or_symlinked_documents_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'file.json'
            for text in ['{"x":1,"x":2}','{"x":NaN}']:
                path.write_text(text)
                with self.subTest(text=text),self.assertRaises(ValueError):check.document(path)
            path.write_text('{}');link=Path(temp)/'link';link.symlink_to(path)
            with self.assertRaises(ValueError):check.document(link)
            with self.assertRaises(ValueError):check.read(path,1)


class Environment(unittest.TestCase):
    def test_complete_stable_external_endpoints(self):self.assertTrue(check.environment(environment()))
    def test_battery_warning_unknown_or_typed_failure_not_qualified(self):
        for key,value in [('power',"Now drawing from 'Battery Power'"),('thermal','Thermal warning:2'),('memory_pressure','System-wide memory free percentage: 101%')]:
            data=environment();data['end'][key]['output']=value
            with self.subTest(key=key):self.assertFalse(check.environment(data))
        data=environment();data['end']['thermal']['exit_code']=False
        self.assertFalse(check.environment(data))
    def test_missing_commands_scope_and_configuration_drift_reject(self):
        cases=[];data=environment();del data['end']['power'];cases.append(data)
        data=environment();data['sampling']='No observations';cases.append(data)
        data=environment();data['end']['power']['command']=['test'];cases.append(data)
        data=environment();data['end']['configuration']['output']='other';cases.append(data)
        data=environment();data['end']['power']['output']=None;cases.append(data)
        for data in cases:
            with self.subTest(data=data),self.assertRaises(ValueError):check.environment(data)


class AllocatorProfile(unittest.TestCase):
    def sample(self):
        values=dict(MallocLargeCache='0',MallocAggressiveMadvise=None,MallocNanoZone=None)
        inputs=dict(allocator_profile='apple_no_large_cache',allocator_contract_sha256='a'*64,allocator_environment=values,
            inherited_allocator_environment=dict(MallocLargeCache=None,MallocAggressiveMadvise=None,MallocNanoZone='0'))
        process=dict(pid=100,group=100,parent_pid=99)
        receipt=dict(allocator_profile='apple_no_large_cache',process=process,nonbudget_checks_passed=True,budget_rejected=False,
            component_physical_limit_bytes=check.LIMIT,engine_process_limit_bytes=6*1024**3)
        return inputs,dict(allocator_environment=copy.deepcopy(values)),receipt,process
    def test_inherited_nano_zero_is_distinct_from_sanitized_child(self):
        check.allocator_profile(*self.sample(),'a'*64)
    def test_wrong_effective_child_or_contract_rejects(self):
        for key,value in [('MallocLargeCache','1'),('MallocLargeCache',False),('MallocNanoZone','0')]:
            data=self.sample();data[1]['allocator_environment'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):check.allocator_profile(*data,'a'*64)
        with self.assertRaises(ValueError):check.allocator_profile(*self.sample(),'b'*64)
    def test_process_identity_is_typed_positive_and_not_self_parented(self):
        for key,value in [('pid',True),('group',101),('parent_pid',0),('parent_pid',100)]:
            data=self.sample();data[3][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):check.allocator_profile(*data,'a'*64)
    def test_caps_and_success_cannot_be_relabelled(self):
        for key,value in [('component_physical_limit_bytes',6*1024**3),('engine_process_limit_bytes',8*1024**3),('budget_rejected',True),('nonbudget_checks_passed',1)]:
            data=self.sample();data[2][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):check.allocator_profile(*data,'a'*64)


if __name__=='__main__':unittest.main()
