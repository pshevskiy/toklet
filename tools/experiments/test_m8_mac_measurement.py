"""Portable Mac raw-measurement fixtures; no model or GPU workload."""
import copy
import json
import unittest

import m8_mac_measurement as m
import m7_physical_measurement as physical
from test_m7_physical_measurement import fixture as physical_fixture, console as physical_console


def fixture():
    config,signed,result,samples,proofs=physical_fixture()
    config.update(version=2,mlx=True,metal=True,packedFolder='')
    result['backend']='metal-mlx'
    for sample in samples:
        sample['environment']={k:v for k,v in sample['environment'].items()
                               if k not in ['applicationState','batteryState','batteryLevel']}
    return config,result,samples


def console(result,samples):
    return '\n'.join(['TOKLET_MEASUREMENT_SAMPLE_JSON='+json.dumps(s) for s in samples]
                     +['TOKLET_MEASUREMENT_JSON='+json.dumps(result)])


class MacMeasurement(unittest.TestCase):
    def test_completed_13_samples_have_ten_completed_timing_brackets(self):
        c,r,s=fixture();observed=m.validate_console(console(r,s),c)
        self.assertEqual(observed['sample_count'],13);self.assertEqual(len(observed['measured']),10)
        self.assertEqual(observed['measured'][0]['native_ttft_seconds'],0.1)
        self.assertEqual(observed['measured'][0]['caller_ttft_seconds'],0.1)
        self.assertEqual(observed['measured'][0]['decode_tokens_per_second'],10)
        self.assertEqual(observed['installed_proofs'],[])
        self.assertEqual(observed['scope'],'mac-native-app-coordinator')

    def test_physical_defaults_remain_strict_and_mac_never_invents_proofs(self):
        c,f,r,s,p=physical_fixture()
        physical.validate_console(physical_console(r,s,p),c,f)
        with self.assertRaises(ValueError):physical.validate_console(console(r,s),c,f)
        c,r,s=fixture()
        for field,value in [('applicationState',0),('batteryState',2),('batteryLevel',1)]:
            bad=copy.deepcopy(s);bad[4]['environment'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):m.validate_console(console(r,bad),c)
        with self.assertRaises(ValueError):m.validate_console('TOKLET_INSTALLATION_JSON={}\n'+console(r,s),c)
        c['version']=1;c.pop('mlx')
        with self.assertRaises(ValueError):m.validate_console(console(r,s),c)

    def test_sustained_stops_at_first_eligible_native_duration(self):
        c,r,s=fixture();c['minimumNativeSeconds']=600
        for i,sample in enumerate(s):
            start=(i+1)*62*10**9;sample['submittedUptimeNanoseconds']=start;sample['completedUptimeNanoseconds']=start+61*10**9
            for j,event in enumerate(sample['events']):
                event['nativeElapsedSeconds']=30*(j+1);event['callbackUptimeNanoseconds']=start+30*(j+1)*10**9
        r['sumNativeLastEventSeconds']=600
        self.assertEqual(len(m.validate_console(console(r,s),c)['measured']),10)
        extra=copy.deepcopy(s[-1]);extra['index']=13;extra['submittedUptimeNanoseconds']+=62*10**9;extra['completedUptimeNanoseconds']+=62*10**9
        for event in extra['events']:event['callbackUptimeNanoseconds']+=62*10**9
        s.append(extra);r['sumNativeLastEventSeconds']=660;r['measuredCompleted']=11
        with self.assertRaisesRegex(ValueError,'first eligible'):m.validate_console(console(r,s),c)

    def test_tokens_are_typed_in_vocabulary_and_repeat_exactly(self):
        for token in [True,-1,64,2**32]:
            c,r,s=fixture();s[4]['tokens'][0]=token
            with self.subTest(token=token),self.assertRaises(ValueError):m.validate_console(console(r,s),c,64)
        c,r,s=fixture();c['expectedPrefix']=[];s[4]['tokens'][0]=46
        with self.assertRaisesRegex(ValueError,'Repeated greedy'):m.validate_console(console(r,s),c,64)
        for vocabulary in [True,0,2**32+1]:
            with self.subTest(vocabulary=vocabulary),self.assertRaises(ValueError):m.validate_console(console(r,s),c,vocabulary)

    def test_serial_callback_and_native_clock_failures_reject(self):
        for kind in ['overlap','reverse','nativeOutside','callbackBefore','callbackAfter','nativeNan','zeroDecodeInterval']:
            c,r,s=fixture();sample=s[4]
            if kind=='overlap':sample['submittedUptimeNanoseconds']=s[3]['completedUptimeNanoseconds']-1
            elif kind=='reverse':sample['events'][1]['nativeElapsedSeconds']=0.01
            elif kind=='nativeOutside':sample['events'][0]['nativeElapsedSeconds']=0.15
            elif kind=='callbackBefore':sample['events'][0]['callbackUptimeNanoseconds']=0
            elif kind=='callbackAfter':sample['events'][1]['callbackUptimeNanoseconds']=sample['completedUptimeNanoseconds']+1
            elif kind=='nativeNan':sample['events'][0]['nativeElapsedSeconds']=float('nan')
            else:sample['events'][1]['nativeElapsedSeconds']=0.1
            with self.subTest(kind=kind),self.assertRaises(ValueError):m.validate_console(console(r,s),c)

    def test_memory_state_summary_and_cleanup_failures_reject(self):
        for kind in ['budget','peakBackwards','pending','admission','role','count','sum','warmups','cleanup','source','claim','lowPower','thermal']:
            c,r,s=fixture()
            if kind=='budget':s[4]['footprint']['lifetimePeakPhysicalBytes']=c['processBudget']+1
            elif kind=='peakBackwards':r['checkpoints'][3]['footprint']['lifetimePeakPhysicalBytes']=100
            elif kind=='pending':s[4]['pending']=0
            elif kind=='admission':s[4]['admittedBytes']=True
            elif kind=='role':s[4]['role']='warmup'
            elif kind=='count':r['measuredCompleted']=9
            elif kind=='sum':r['sumNativeLastEventSeconds']+=1
            elif kind=='warmups':r['warmupsCompleted']=2
            elif kind=='cleanup':r['cleanupErrors']=['failure']
            elif kind=='source':r['filesAfter']={}
            elif kind=='claim':r['physicalDeviceQualified']=True
            elif kind=='lowPower':s[4]['environment']['lowPowerMode']=0
            else:s[4]['environment']['thermalState']=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):m.validate_console(console(r,s),c)

    def test_marker_order_duplicates_errors_and_partial_samples_reject(self):
        c,r,s=fixture();text=console(r,s);lines=text.splitlines()
        for bad in [console(r,s[:-1]),text+'\n'+lines[-1],text+'\nTOKLET_MAC_JOB_ERROR=bad',
                    '\n'.join([lines[-1],*lines[:-1]]),text+'\nTOKLET_OTHER_JSON={}']:
            with self.subTest(tail=bad[-20:]),self.assertRaises(ValueError):m.validate_console(bad,c)

    def test_filesystem_scope_and_first_request_boundary_rejects(self):
        for kind in ['bool','instrument','scope','late','extra']:
            c,r,s=fixture();fs=r['filesystemBefore']
            if kind=='bool':fs['freeBytes']=True
            elif kind in ['instrument','scope']:fs[kind]='unknown'
            elif kind=='late':fs['readCompletedUptimeNanoseconds']=s[0]['submittedUptimeNanoseconds']+1
            else:fs['extra']=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):m.validate_console(console(r,s),c)


if __name__=='__main__':unittest.main()
