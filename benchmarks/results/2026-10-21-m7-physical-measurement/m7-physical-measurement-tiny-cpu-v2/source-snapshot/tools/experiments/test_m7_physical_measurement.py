import copy
import json
import plistlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import m7_physical_measurement as m


def fixture():
    config=dict(metal=False,prompt=[1,7,13,3],maximum=2,expectedPrefix=[47,47],numericalBudget=512*1024,
                processBudget=1024*1024,minimumNativeSeconds=0,files={'Model/config.json':'a'*64})
    signed={'TokletHarness':'b'*64}
    def memory(peak):return dict(physicalBytes=100,lifetimePeakPhysicalBytes=peak,returnedNaturalCount=93)
    samples=[]
    for index in range(13):
        start=(index+1)*1000000000
        samples.append(dict(index=index,role='warmup' if index<3 else 'measured',status=0,tokens=[47,47],
            history=6,evaluated=5,pending=1,reason=1,admittedBytes=4096,footprint=memory(400+index),
            environment=dict(applicationState=0,lowPowerMode=False,thermalState=0,batteryState=2,batteryLevel=1),
            submittedUptimeNanoseconds=start,completedUptimeNanoseconds=start+250000000,
            events=[dict(index=i,nativeElapsedSeconds=(i+1)/10,callbackUptimeNanoseconds=start+(i+1)*100000000) for i in range(2)]))
    phases=['before-artifact-proof','after-artifact-proof','loaded','unloaded','closed','after-final-artifact-proof']
    total=0
    for sample in samples[3:]:total+=sample['events'][-1]['nativeElapsedSeconds']
    result=dict(passed=True,backend='cpu',interruptionCount=0,cleanupErrors=[],endpointErrors=[],closed=True,released=True,
        physicalDeviceQualified=False,m7Complete=False,filesBefore=config['files'],filesAfter=config['files'],
        checkpoints=[dict(phase=p,footprint=memory(n)) for p,n in zip(phases,[100,200,300,500,600,700])],
        warmupsCompleted=3,measuredCompleted=10,sumNativeLastEventSeconds=total,loadAndSessionSeconds=1)
    proofs=[dict(phase=p,simulator=False,files=signed) for p in ['before','after']]
    return config,signed,result,samples,proofs


def console(result,samples,proofs):
    return '\n'.join(['TOKLET_INSTALLATION_JSON='+json.dumps(proofs[0])]
                     +['TOKLET_MEASUREMENT_SAMPLE_JSON='+json.dumps(s) for s in samples]
                     +['TOKLET_INSTALLATION_JSON='+json.dumps(proofs[1]),'TOKLET_MEASUREMENT_JSON='+json.dumps(result)])


class PhysicalMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):(m.ROOT/'artifacts').mkdir(exist_ok=True)

    def test_valid_complete_raw_measurement(self):
        c,f,r,s,p=fixture();observed=m.validate_console(console(r,s,p),c,f)
        self.assertEqual(observed['sample_count'],13);self.assertEqual(len(observed['measured']),10)
        self.assertEqual(observed['measured'][0]['decode_tokens_per_second'],10)

    def test_marker_order_and_extra_repetitions_rejected(self):
        c,f,r,s,p=fixture();lines=console(r,s,p).splitlines()
        reordered='\n'.join([lines[0],lines[-2],*lines[1:-2],lines[-1]])
        with self.assertRaises(ValueError):m.validate_console(reordered,c,f)
        extra=copy.deepcopy(s[-1]);extra['index']=13;s.append(extra)
        r['measuredCompleted']+=1;r['sumNativeLastEventSeconds']+=extra['events'][-1]['nativeElapsedSeconds']
        with self.assertRaises(ValueError):m.validate_console(console(r,s,p),c,f)
        # Native duration600s is first reached by the tenth measured request.
        c,f,r,s,p=fixture();c['minimumNativeSeconds']=600
        for index,sample in enumerate(s):
            start=(index+1)*62*10**9;sample['submittedUptimeNanoseconds']=start
            sample['completedUptimeNanoseconds']=start+61*10**9
            for i,event in enumerate(sample['events']):
                event['nativeElapsedSeconds']=30*(i+1);event['callbackUptimeNanoseconds']=start+30*(i+1)*10**9
        r['sumNativeLastEventSeconds']=600
        m.validate_console(console(r,s,p),c,f)
        extra=copy.deepcopy(s[-1]);extra['index']=13;s.append(extra);r['measuredCompleted']=11;r['sumNativeLastEventSeconds']=660
        with self.assertRaises(ValueError):m.validate_console(console(r,s,p),c,f)

    def test_overlapping_or_reversed_serial_requests_rejected(self):
        for overlap in (1,250000001):
            c,f,r,s,p=fixture();s[4]['submittedUptimeNanoseconds']=s[3]['completedUptimeNanoseconds']-overlap
            with self.subTest(overlap=overlap),self.assertRaises(ValueError):m.validate_console(console(r,s,p),c,f)

    def test_numeric_wire_bounds_and_own_callback_window(self):
        for kind in ('infinity','boolTimestamp','negativeAdmission','hugeID','boolIndex','loadNan','loadNegative','eventBeforeNative','floatPeak'):
            with self.subTest(kind=kind):
                c,f,r,s,p=fixture();sample=s[4]
                if kind=='infinity':sample['completedUptimeNanoseconds']=float('inf')
                elif kind=='boolTimestamp':sample['submittedUptimeNanoseconds']=True
                elif kind=='negativeAdmission':sample['admittedBytes']=-1
                elif kind=='hugeID':sample['tokens'][0]=2**32
                elif kind=='boolIndex':sample['events'][0]['index']=False
                elif kind=='loadNan':r['loadAndSessionSeconds']=float('nan')
                elif kind=='loadNegative':r['loadAndSessionSeconds']=-1
                elif kind=='eventBeforeNative':sample['events'][0]['nativeElapsedSeconds']=0.15
                else:sample['footprint']['physicalBytes']=100.0
                with self.assertRaises(ValueError):m.validate_console(console(r,s,p),c,f)

    def test_prompt_hash_and_unpack_use_one_bounded_buffer(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'artifacts') as temporary:
            base=Path(temporary);tokens=base/'tokens';original=(1).to_bytes(4,'little');expected=m.package.hashlib.sha256(original).hexdigest()
            tokens.write_bytes(original)
            self.assertEqual(m.read_prompt(tokens,1,expected),[1])
            # A prior external hash can still match after replacement. The
            # actual read buffer must independently reject the changed bytes.
            tokens.write_bytes((2).to_bytes(4,'little'))
            with patch.object(m.package,'sha',return_value=expected),self.assertRaises(ValueError):m.read_prompt(tokens,1,expected)
            private=base/'private';private.write_text('{}')
            with patch.object(m.physical,'selection',return_value={}),patch.object(m.integration,'admit_build',return_value={}), \
                 patch.object(m.physical,'admit_reuse',return_value=(base,{},'/verified/TokletHarness.app',{})), \
                 patch.object(m,'job',side_effect=lambda *_:m.read_prompt(tokens,1,expected)), \
                 patch.object(m.package,'owned') as owned,patch('builtins.print'):
                code=m.main(base,private,m.package.sha(private),base,'unused','small','128',base/'output')
            self.assertEqual(code,1);owned.assert_not_called()

    def test_source_result_and_installation_mismatch_rejected(self):
        for field in ('filesBefore','filesAfter','backend','closed','released','interruptionCount','cleanupErrors','endpointErrors','physicalDeviceQualified'):
            with self.subTest(field=field):
                c,f,r,s,p=fixture();r[field]={'wrong':'x'} if field.startswith('files') else {'backend':'metal','closed':False,'released':False,'interruptionCount':1,'cleanupErrors':['failure'],'endpointErrors':['mutation'],'physicalDeviceQualified':True}[field]
                with self.assertRaises(ValueError):m.validate_console(console(r,s,p),c,f)
        c,f,r,s,p=fixture();p[0]['simulator']=True
        with self.assertRaises(ValueError):m.validate_console(console(r,s,p),c,f)

    def test_partial_error_and_duplicate_output_rejected(self):
        c,f,r,s,p=fixture()
        for text in [console(r,s[:-1],p),console(r,s,p)+'\nTOKLET_MEASUREMENT_JSON='+json.dumps(r),console(r,s,p)+'\nTOKLET_MEASUREMENT_ERROR=failed']:
            with self.subTest(text=text[-32:]), self.assertRaises(ValueError):m.validate_console(text,c,f)

    def test_wrong_tokens_state_budget_or_foreground_rejected(self):
        for kind in ('tokens','pending','nativeBudget','processBudget','appState','thermal','role'):
            with self.subTest(kind=kind):
                c,f,r,s,p=fixture();sample=s[4]
                if kind=='tokens':sample['tokens']=[0,0]
                elif kind=='pending':sample['pending']=0
                elif kind=='nativeBudget':sample['admittedBytes']=c['numericalBudget']+1
                elif kind=='processBudget':sample['footprint']['lifetimePeakPhysicalBytes']=c['processBudget']+1
                elif kind=='appState':sample['environment']['applicationState']=2
                elif kind=='thermal':sample['environment']['thermalState']=99
                else:sample['role']='warmup'
                with self.assertRaises(ValueError):m.validate_console(console(r,s,p),c,f)

    def test_invalid_clocks_duration_and_peak_history_rejected(self):
        for kind in ('nan','negative','reverse','callback','outside','total','warmups','sustained','peak'):
            with self.subTest(kind=kind):
                c,f,r,s,p=fixture();sample=s[4]
                if kind=='nan':sample['events'][0]['nativeElapsedSeconds']=float('nan')
                elif kind=='negative':sample['events'][0]['nativeElapsedSeconds']=-1
                elif kind=='reverse':sample['events'][1]['nativeElapsedSeconds']=0.01
                elif kind=='callback':sample['events'][1]['callbackUptimeNanoseconds']=0
                elif kind=='outside':sample['events'][1]['nativeElapsedSeconds']=1
                elif kind=='total':r['sumNativeLastEventSeconds']+=1
                elif kind=='warmups':r['warmupsCompleted']=2
                elif kind=='sustained':c['minimumNativeSeconds']=600
                else:r['checkpoints'][3]['footprint']['lifetimePeakPhysicalBytes']=100
                with self.assertRaises(ValueError):m.validate_console(console(r,s,p),c,f)

    def test_listing_requires_exact_selected_container_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'source';path.write_bytes(b'known')
            config=dict(files={'Inputs/Model/weights':'x'},host={'Inputs/Model/weights':'source'})
            data=dict(info=dict(outcome='success'),result=dict(deviceIdentifier='selected',domain='appDataContainer',
                domainIdentifier=m.physical.BUNDLE,files=[dict(name='Model/weights',relativePath='Model/weights',
                    resources=dict(isDirectory=False,isSymbolicLink=False),metadata=dict(size=5))]))
            with patch.object(m,'ROOT',root):
                m.validate_listing(data,'selected',config,'Inputs')
                for kind in ('device','size','symlink','duplicate'):
                    d=copy.deepcopy(data)
                    if kind=='device':d['result']['deviceIdentifier']='other'
                    elif kind=='size':d['result']['files'][0]['metadata']['size']=4
                    elif kind=='symlink':d['result']['files'][0]['resources']['isSymbolicLink']=True
                    else:d['result']['files']*=2
                    with self.subTest(kind=kind),self.assertRaises(ValueError):m.validate_listing(d,'selected',config,'Inputs')

    def test_failed_prepared_admission_never_launches_child(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'artifacts') as temporary:
            base=Path(temporary);private=base/'private.json';private.write_text('{}')
            with patch.object(m.physical,'selection',return_value={'device_identifier':'selected'}), \
                 patch.object(m.integration,'admit_build',return_value={}), \
                 patch.object(m.physical,'admit_reuse',side_effect=ValueError('changed signed source')), \
                 patch.object(m.package,'owned') as owned,patch('builtins.print'):
                code=m.main(base,private,'unused',base,'unused','tiny-cpu','short',base/'output')
            self.assertEqual(code,1);owned.assert_not_called()
            record=json.loads((base/'output/measurement.json').read_text());self.assertFalse(record['passed']);self.assertEqual(record['commands'],[])

    def test_mutated_admitted_input_cannot_block_remote_cleanup(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'artifacts') as temporary:
            base=Path(temporary);private=base/'private.json';private.write_text('{}');private_sha=m.package.sha(private)
            payload=base/'payload';payload.write_bytes(b'known');parent='M7Inputs/tiny-cpu'
            c,f,r,s,p=fixture();c['packedFolder']='';c['files']={parent+'/Model/config.json':m.package.sha(payload)}
            r['filesBefore']=c['files'];r['filesAfter']=c['files'];host={next(iter(c['files'])):str(payload.relative_to(m.ROOT))}
            remote='/var/containers/Bundle/Application/known/TokletHarness.app';running=False
            def stage(config,files,output):
                folder=output/'transfer';(folder/'M7Jobs').mkdir(parents=True);job=folder/'M7Jobs/job.json';job.write_text('{}')
                return folder,'M7Jobs/job.json',m.package.sha(job)
            def owned(args,stdout,stderr,env,timeout):
                nonlocal running
                label=stdout.stem;data={}
                if label.startswith('process-'):
                    data={'result':{'runningProcesses':[{'executable':remote+'/TokletHarness','processIdentifier':123}] if running else []}}
                elif label.startswith('signal-'):running=False
                elif label=='copied-inventory':
                    data=dict(info=dict(outcome='success'),result=dict(deviceIdentifier='selected',domain='appDataContainer',domainIdentifier=m.physical.BUNDLE,
                        files=[dict(name='Model/config.json',relativePath='Model/config.json',resources=dict(isDirectory=False,isSymbolicLink=False),metadata=dict(size=5))]))
                if label in ('profile','entitlements'):stdout.write_bytes(plistlib.dumps({}))
                elif label=='measurement':
                    stdout.write_text(console(r,s,p));Path(args[args.index('--json-output')+1]).write_text('{}')
                    private.write_text('{"changed":true}');running=True
                else:stdout.write_text(json.dumps(data))
                stderr.write_text('')
                return dict(command=args,exit_code=0,primary_error=None,cleanup_errors=[],pid=1,
                            stdout=str(stdout.relative_to(m.ROOT)),stderr=str(stderr.relative_to(m.ROOT)))
            with patch.object(m.physical,'selection',return_value={'device_identifier':'selected'}), \
                 patch.object(m.integration,'admit_build',return_value={}), \
                 patch.object(m.physical,'admit_reuse',return_value=(base,{},remote,f)), \
                 patch.object(m,'job',return_value=(c,host,{str(payload.relative_to(m.ROOT)):m.package.sha(payload)})), \
                 patch.object(m,'stage',side_effect=stage),patch.object(m.package,'owned',side_effect=owned), \
                 patch.object(m.physical,'admit_device',return_value={}),patch.object(m.physical,'admit_profile',return_value={}), \
                 patch.object(m.physical,'launch_verified',side_effect=lambda current,launch,*_:launch()), \
                 patch.object(m.physical,'check_launch',return_value={}),patch.object(m.physical,'inventory',return_value=f),patch('builtins.print'):
                code=m.main(base,private,private_sha,base,'unused','tiny-cpu','short',base/'output')
            record=json.loads((base/'output/measurement.json').read_text())
            self.assertEqual(code,1);self.assertTrue(record['endpoint_errors']);self.assertIsNotNone(record['checked'])
            self.assertEqual(len(record['remote_cleanup']),2);self.assertTrue(all(x['absent'] for x in record['remote_cleanup']))
            self.assertEqual(record['remote_cleanup'][-1]['signals'],[dict(pid=123,signal='SIGTERM')]);self.assertFalse(running)

    def test_unsupported_jobs_rejected_before_model_reads(self):
        for model,workload in [('other','short'),('tiny-cpu','512'),('small','unknown')]:
            with self.subTest(model=model),self.assertRaises(ValueError):m.job(model,workload)


if __name__=='__main__':unittest.main()
