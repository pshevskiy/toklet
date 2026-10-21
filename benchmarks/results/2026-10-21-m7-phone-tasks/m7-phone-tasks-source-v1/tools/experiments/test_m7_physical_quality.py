import copy
import json
import math
from pathlib import Path
import plistlib
import struct
import tempfile
import unittest
from unittest.mock import patch
import m7_physical_quality as q
m=q.measurement


def fixture(targets=3,block=3):
    tokens=[i%64 for i in range(targets+1)];sha=m.package.hashlib.sha256(struct.pack('<'+str(len(tokens))+'I',*tokens)).hexdigest()
    config=dict(version=1,modelFolder='M7Inputs/tiny-cpu/Model',packedFolder='',files={'M7Inputs/tiny-cpu/Model/config.json':'a'*64},metal=False,
                capacity=block+1,chunk=1,numericalBudget=128*1024**2,processBudget=192*1024**2,
                tokensFile='M7Corpora/'+sha+'/tokens.u32le',tokensSHA256=sha,targets=targets,blockSize=block)
    signed={'TokletHarness':'b'*64};fp=dict(physicalBytes=100,lifetimePeakPhysicalBytes=200,returnedNaturalCount=93)
    samples=[]
    for i in range(targets):
        samples.append(dict(target=dict(targetIndex=i,block=i//block,position=i%block+1,committedHistory=i%block+2,
                                        score=dict(target=tokens[i+1],greedy=0,vocabulary=64,evaluated=i%block+1,targetLogit=0.0,greedyLogit=1.0,negativeLogLikelihood=1.0)),
                            footprint=copy.deepcopy(fp),environment=dict(applicationState=0,lowPowerMode=False,thermalState=0,batteryState=3,batteryLevel=1.0)))
    blocks=(targets+block-1)//block
    result=dict(backend='cpu',passed=True,cleanupErrors=[],endpointErrors=[],closed=True,released=True,interruptionCount=0,loadAndSessionSeconds=1.0,
                fullQualityQualified=False,physicalDeviceQualified=False,m7Complete=False,samplesCompleted=targets,filesBefore=config['files'],filesAfter=config['files'],
                tokensBeforeSHA256=sha,tokensAfterSHA256=sha,
                checkpoints=[dict(phase=p,footprint=copy.deepcopy(fp)) for p in ['before-artifact-proof','after-artifact-proof','loaded']+['block-'+str(i) for i in range(blocks)]+['unloaded','closed','after-final-artifact-proof']],
                summary=dict(targets=targets,blocks=blocks,blockSize=block,policy=q.POLICY,negativeLogLikelihoodSum=float(targets),meanNegativeLogLikelihood=1.0))
    proofs=[dict(phase=p,simulator=False,files=signed) for p in ('before','after')]
    return config,signed,tokens,result,samples,proofs


def console(result,samples,proofs):
    return '\n'.join(['TOKLET_INSTALLATION_JSON='+json.dumps(proofs[0])]+['TOKLET_QUALITY_SAMPLE_JSON='+json.dumps(s) for s in samples]
                     +['TOKLET_INSTALLATION_JSON='+json.dumps(proofs[1]),'TOKLET_QUALITY_JSON='+json.dumps(result)])


class PhysicalQualityChecks(unittest.TestCase):
    def reject(self,mutate):
        c,s,t,r,rows,p=fixture();mutate(c,r,rows,p)
        with self.assertRaises((ValueError,KeyError,TypeError)):q.check_console(console(r,rows,p),c,s,t,64)
    def test_exact_short_and_full_policy(self):
        for targets,block in [(3,3),(10000,1024)]:
            c,s,t,r,rows,p=fixture(targets,block);out=q.check_console(console(r,rows,p),c,s,t,64)
            self.assertEqual(out['sample_count'],targets);self.assertEqual(out['computed_mean_nll'],1.0)
            if targets==10000:self.assertEqual(rows[-1]['target']['committedHistory'],785)
    def test_prefix_and_teacher_ids_reject(self):
        for field,value in [('targetIndex',1),('targetIndex',False),('block',1),('position',2),('committedHistory',1)]:
            with self.subTest(field=field):self.reject(lambda c,r,s,p:s[0]['target'].update({field:value}))
        for field,value in [('target',7),('vocabulary',65),('evaluated',2),('greedy',64),('targetLogit',0.1),('targetLogit',float('nan')),('negativeLogLikelihood',True)]:
            with self.subTest(field=field):self.reject(lambda c,r,s,p:s[0]['target']['score'].update({field:value}))
    def test_stream_and_installation_proofs_reject(self):
        c,s,t,r,rows,p=fixture();lines=console(r,rows,p).splitlines()
        for text in ('\n'.join([lines[0],lines[-1],*lines[1:-1]]),console(r,rows,p)+'\nTOKLET_QUALITY_ERROR=bad'):
            with self.assertRaises(ValueError):q.check_console(text,c,s,t,64)
        self.reject(lambda c,r,s,p:p[0].update(simulator=True))
        self.reject(lambda c,r,s,p:p[1].update(files={}))
    def test_peak_foreground_and_power_reject(self):
        self.reject(lambda c,r,s,p:s[0]['footprint'].update(lifetimePeakPhysicalBytes=193*1024**2))
        self.reject(lambda c,r,s,p:s[0]['footprint'].update(lifetimePeakPhysicalBytes=201))
        for field,value in [('applicationState',1),('thermalState',4),('lowPowerMode',0),('batteryState',0),('batteryLevel',float('nan'))]:
            with self.subTest(field=field):self.reject(lambda c,r,s,p:s[0]['environment'].update({field:value}))
    def test_result_summary_and_reference_identity_reject(self):
        for field,value in [('passed',False),('closed',False),('released',False),('interruptionCount',True),('samplesCompleted',2),('tokensAfterSHA256','bad'),('fullQualityQualified',True),('loadAndSessionSeconds',True)]:
            with self.subTest(field=field):self.reject(lambda c,r,s,p:r.update({field:value}))
        for field,value in [('blocks',True),('blockSize',1024),('policy','changed'),('negativeLogLikelihoodSum',3.1),('meanNegativeLogLikelihood',float('inf'))]:
            with self.subTest(field=field):self.reject(lambda c,r,s,p:r['summary'].update({field:value}))
        self.reject(lambda c,r,s,p:c.update(tokensSHA256='a'*64))
        self.reject(lambda c,r,s,p:c.update(capacity=5))
    def test_frozen_corpus_admits_original_and_rejects_replacement(self):
        raw,pins=q.corpus('small','full');self.assertEqual(len(raw),40004);self.assertEqual(pins[q.CORPUS],q.CORPUS_SHA)
        short,_=q.corpus('larger','short');self.assertEqual(short,raw[:16])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();p=root/q.CORPUS;p.parent.mkdir(parents=True);p.write_bytes(raw+b'\0')
            with patch.object(q,'ROOT',root),self.assertRaises(ValueError):q.corpus('small','full')
        with self.assertRaises(ValueError):q.corpus('tiny-cpu','full')
    def test_job_seals_score_proof_and_full_block_policy(self):
        c,_,_,_,_,_=fixture();base={k:c[k] for k in ('version','modelFolder','packedFolder','files','metal','numericalBudget','processBudget')}
        proof=dict(passed=True,package_build_sha256=q.NATIVE_SHA,physical_device_qualified=False)
        raw=bytes(40004)
        with patch.object(m,'job',return_value=(base,{},{})),patch.object(m.integration,'metadata',return_value=proof),patch.object(q,'corpus',return_value=(raw,{})):
            result,_,pins=q.job('small','full');self.assertEqual((result['targets'],result['blockSize'],result['capacity'],result['chunk']),(10000,1024,1025,1));self.assertEqual(pins[q.SCORE_PROOF],q.SCORE_PROOF_SHA)
            proof['package_build_sha256']='a'*64
            with self.assertRaises(ValueError):q.job('small','full')
    def test_corpus_listing_rejects_wrong_device_size_or_symlink(self):
        c,*_=fixture();row=dict(name='tokens.u32le',relativePath='tokens.u32le',resources=dict(isSymbolicLink=False,isDirectory=False),metadata=dict(size=16))
        data=dict(info=dict(outcome='success'),result=dict(deviceIdentifier='device',domain='appDataContainer',domainIdentifier=m.physical.BUNDLE,files=[row]))
        q.validate_corpus_listing(data,'device',c)
        for field,value in [('size',17),('size',True)]:
            changed=copy.deepcopy(data);changed['result']['files'][0]['metadata'][field]=value
            with self.assertRaises(ValueError):q.validate_corpus_listing(changed,'device',c)
        changed=copy.deepcopy(data);changed['result']['files'][0]['resources']['isSymbolicLink']=True
        with self.assertRaises(ValueError):q.validate_corpus_listing(changed,'device',c)
        with self.assertRaises(ValueError):q.validate_corpus_listing(data,'different',c)
    def test_storage_counts_corpus_and_requires_its_copy_clock(self):
        c,*_=fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();base=root/'artifacts/staged';base.mkdir(parents=True);model=base/next(iter(c['files']));model.parent.mkdir(parents=True);model.write_bytes(b'model');c['files']={str(model.relative_to(base)):m.package.sha(model)}
            corpus=base/c['tokensFile'];corpus.parent.mkdir(parents=True);raw=struct.pack('<4I',0,1,2,3);corpus.write_bytes(raw)
            job=b'{}\n';digest=m.package.hashlib.sha256(job).hexdigest();name='M7Jobs/'+digest+'.json';path=base/name;path.parent.mkdir();path.write_bytes(job)
            with patch.object(m.storage,'ROOT',root):
                payload,total=m.storage.inventory(base,c,name,digest);self.assertEqual(total,len(b'model')+16+len(job));self.assertIn(c['tokensFile'],payload)
                saved=dict(proof=dict(path='artifacts/proof/measurement.json',sha256='a'*64),checks=[dict(label=label,wallUnixNanoseconds=1,monotonicNanoseconds=1) for label in ['copy-Model','copy-corpus','copy-job']])
                def fresh(admission,boot,wall,mono,label):return dict(label=label,wallUnixNanoseconds=wall,monotonicNanoseconds=mono)
                with patch.object(m.storage,'admit_proof',return_value=({'host_boot_token':'b'*64},{})),patch.object(m.storage,'inventory',return_value=(payload,total)),patch.object(m.storage,'decision',side_effect=lambda *a:copy.deepcopy(saved)),patch.object(m.storage,'fresh',side_effect=fresh):
                    value,_=m.storage.admit_ready(dict(storage_admission=saved),base,c,name,digest,{},None);self.assertEqual(value,saved)
                    bad=copy.deepcopy(saved);bad['checks'].pop(1)
                    with self.assertRaises(ValueError):m.storage.admit_ready(dict(storage_admission=bad),base,c,name,digest,{},None)
    def test_quality_prepare_transport_selects_package_corpus_and_receipt(self):
        (m.ROOT/'artifacts').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=m.ROOT/'artifacts') as temp:
            folder=Path(temp).resolve();output=folder/'run';app=folder/'app';app.mkdir()
            c,*_=fixture();selected=dict(device_identifier='device',device_udid='udid',development_team='team')
            signed={'TokletHarness':'b'*64};remote={'path':'owned'};labels=[]
            corpus_row=dict(name='tokens.u32le',relativePath='tokens.u32le',resources=dict(isSymbolicLink=False,isDirectory=False),metadata=dict(size=16))
            listed=dict(info=dict(outcome='success'),result=dict(deviceIdentifier='device',domain='appDataContainer',domainIdentifier=m.physical.BUNDLE,files=[corpus_row]))
            def stage(config,host,destination,*unused):
                base=destination/'transfer';base.mkdir();name='M7Jobs/'+('a'*64)+'.json'
                path=base/name;path.parent.mkdir();path.write_text('{}')
                (base/'M7Inputs/tiny-cpu/Model').mkdir(parents=True)
                cp=base/config['tokensFile'];cp.parent.mkdir(parents=True);cp.write_bytes(struct.pack('<4I',0,1,2,3))
                return base,name,'a'*64
            def owned(args,stdout,stderr,env,timeout):
                label=stdout.stem;labels.append(label)
                data=plistlib.dumps({}).decode() if label in ('profile','entitlements') else json.dumps(listed) if label=='copied-corpus-inventory' else '{}'
                stdout.write_text(data);stderr.write_text('')
                return dict(command=args,exit_code=0,primary_error=None,cleanup_errors=[],pid=1,stdout=str(stdout.relative_to(m.ROOT)),stderr=str(stderr.relative_to(m.ROOT)))
            with patch.object(m.physical,'selection',return_value=selected),patch.object(m.integration,'admit_build',return_value={}) as package_admission,\
                 patch.object(m.physical,'admit_reuse',return_value=(app,{},remote,signed)),patch.object(m,'verify_inputs'),\
                 patch.object(q,'job',return_value=(c,{},{})),patch.object(q,'stage',side_effect=stage),\
                 patch.object(m.package,'owned',side_effect=owned),patch.object(m.physical,'admit_device',return_value={'productType':'test'}),\
                 patch.object(m.physical,'admit_profile',return_value={}),patch.object(m.physical,'launch_verified',side_effect=lambda query,action,*a:action()),\
                 patch.object(m.physical,'stop_remote',return_value=dict(absent=True,observations=[[]],signals=[],errors=[])),\
                 patch.object(m,'validate_listing'),patch.object(m.physical,'inventory',return_value=signed),patch.object(m.physical,'redact_record',side_effect=lambda record,*a:record):
                code=m.main(folder/'native',folder/'private.json','c'*64,folder/'prepared','d'*64,'tiny-cpu','short',output,prepare_only=True,native_sha=q.NATIVE_SHA,quality=True)
            self.assertEqual(code,0)
            package_admission.assert_called_once_with((folder/'native').resolve(),q.NATIVE_SHA)
            record=json.loads((output/'measurement.json').read_text())
            self.assertEqual(record['format'],q.READY_FORMAT);self.assertFalse(record['full_quality_qualified'])
            self.assertEqual([label for label in labels if label.startswith('copy-')],['copy-Model','copy-corpus','copy-job'])
            self.assertIn('copied-corpus-inventory',labels)
            self.assertEqual(len(record['remote_cleanup']),2)

    def test_wrong_native_package_blocks_quality_before_output_or_children(self):
        with patch.object(m.package,'owned') as owned:
            with self.assertRaises(ValueError):m.main(Path('no'),Path('no'),'a'*64,Path('no'),'b'*64,'small','full',Path('no'),quality=True,native_sha=m.NATIVE_SHA)
            owned.assert_not_called()
    def test_ready_formats_cannot_be_interchanged(self):
        c,*_=fixture();folder=m.ROOT/'artifacts/unused-quality-ready';prepared=m.ROOT/'artifacts/unused-prepared';digest='a'*64
        r=dict(format=q.READY_FORMAT,passed=True,checked=None,config=c,signed_bundle={'app':'b'*64},prepared_receipt=dict(path=str((prepared/'integration.json').relative_to(m.ROOT)),sha256='c'*64),primary_error=None,endpoint_errors=[],cleanup_errors=[],remote_cleanup=[dict(absent=True,errors=[])]*2,bindings={},
               private_evidence_pins={str((folder/'private-logs'/n).relative_to(m.ROOT)):'d'*64 for n in ['profile.stdout','entitlements.stdout','signature-details.stderr']},job_receipt=dict(relative_path='M7Jobs/'+digest+'.json',sha256=digest))
        with patch.object(m,'verify_inputs'),patch.object(m.integration,'metadata',side_effect=[r,c]):
            result=m.admit_ready(folder,'e'*64,c,{},prepared,'c'*64,r['signed_bundle'],q.READY_FORMAT);self.assertEqual(result[2],digest)
        with patch.object(m.integration,'metadata',return_value=r),self.assertRaises(ValueError):m.admit_ready(folder,'e'*64,c,{},prepared,'c'*64,r['signed_bundle'])


if __name__=='__main__':unittest.main()
