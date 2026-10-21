"""Offline storage policy regressions; no models, SDKs, GPU or phone required."""
import copy
import builtins
import json
import plistlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import m7_physical_storage as policy
import m7_physical_measurement as measurement
from test_m7_physical_measurement import fixture,console

BOOT='00000000-0000-4000-8000-000000000001'


def proof_fixture(folder):
    folder.mkdir();logs=folder/'private-logs';logs.mkdir()
    config,signed,result,samples,proofs=fixture()
    result['filesystemBefore']['freeBytes']=policy.RESERVE+1024
    (logs/'measurement.stdout').write_text(console(result,samples,proofs));(logs/'measurement.stderr').write_text('')
    (logs/'launch.json').write_text('{}');(logs/'host-boot.stdout').write_text(BOOT+'\n')
    private=folder/'selection.json';private.write_text('{}');private_name=str(private.relative_to(policy.ROOT))
    launch=dict(pid=1,executable_matches=True,reported_exit_code=0,reported_core_dump=False)
    checked=measurement.validate_console(console(result,samples,proofs),config,signed);checked['launch']=launch
    inputs={private_name:policy.package.sha(private)}
    prepared=dict(path=str((folder/'prepared/integration.json').relative_to(policy.ROOT)),sha256='b'*64)
    context=dict(tiny_config=config,signed_bundle=signed,selection_path=private_name,selection_sha=inputs[private_name],
                 prepared_receipt=prepared,required_bindings=dict(inputs),remote='/owned/TokletHarness.app',selected={})
    record=dict(format='toklet-m7-physical-measurement-v1',passed=True,config=config,prepared_receipt=prepared,signed_bundle=signed,
                primary_error=None,endpoint_errors=[],cleanup_errors=[],remote_cleanup=[dict(absent=True,errors=[]) for _ in range(2)],
                bindings=inputs,private_evidence_pins={str(p.relative_to(policy.ROOT)):policy.package.sha(p) for p in logs.iterdir()},
                measurement_host_start=dict(wallUnixNanoseconds=10**12,monotonicNanoseconds=10**12),host_boot_token=policy.boot_token(BOOT),
                checked=checked,device=dict(product_type='test'),physical_device_qualified=False,m7_complete=False)
    return record,context,launch


class StorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):(policy.ROOT/'artifacts').mkdir(exist_ok=True)

    def test_boot_identity_requires_exact_uuid_and_hashes_case_consistently(self):
        self.assertEqual(policy.boot_token(BOOT),policy.boot_token(BOOT.upper()))
        self.assertNotIn(BOOT,policy.boot_token(BOOT))
        for value in ('','invalid','00000000-0000-0000-0000-000000000000',BOOT.replace('-','')):
            with self.subTest(value=value),self.assertRaises(ValueError):policy.boot_token(value)

    def test_payload_inventory_includes_job_and_rejects_changed_or_extra_entries(self):
        with tempfile.TemporaryDirectory(dir=policy.ROOT/'artifacts') as temporary:
            base=Path(temporary);(base/'Model').mkdir();(base/'M7Jobs').mkdir()
            model=base/'Model/weights';model.write_bytes(b'12345');job=base/'M7Jobs/job';job.write_bytes(b'{}')
            config=dict(files={'Model/weights':policy.package.sha(model)})
            inventory,total=policy.inventory(base,config,'M7Jobs/job',policy.package.sha(job))
            self.assertEqual(total,7);self.assertEqual(set(inventory),{'Model/weights','M7Jobs/job'})
            model.write_bytes(b'changed')
            with self.assertRaises(ValueError):policy.inventory(base,config,'M7Jobs/job',policy.package.sha(job))
            model.write_bytes(b'12345');extra=base/'Model/extra';extra.write_bytes(b'x')
            with self.assertRaises(ValueError):policy.inventory(base,config,'M7Jobs/job',policy.package.sha(job))
            extra.unlink();model.unlink();model.symlink_to(job)
            with self.assertRaises(ValueError):policy.inventory(base,config,'M7Jobs/job',policy.package.sha(job))

    def test_capacity_accepts_exact_requirement_rejects_one_byte_short_and_overflow(self):
        proof=dict(checked=dict(result=dict(filesystemBefore=dict(freeBytes=policy.RESERVE+7))),
                   measurement_host_start=dict(wallUnixNanoseconds=1,monotonicNanoseconds=1),host_boot_token=policy.boot_token(BOOT))
        folder=policy.ROOT/'artifacts/proof'
        admitted=policy.decision(proof,folder,'a'*64,{},7);self.assertEqual(admitted['required_bytes'],policy.RESERVE+7)
        proof['checked']['result']['filesystemBefore']['freeBytes']-=1
        with self.assertRaises(ValueError):policy.decision(proof,folder,'a'*64,{},7)
        with self.assertRaises(ValueError):policy.decision(proof,folder,'a'*64,{},2**64-policy.RESERVE)

    def test_file_replacement_between_metadata_and_hash_cannot_undercount_payload(self):
        with tempfile.TemporaryDirectory(dir=policy.ROOT/'artifacts') as temporary:
            base=Path(temporary);(base/'Model').mkdir();(base/'M7Jobs').mkdir()
            model=base/'Model/weights';model.write_bytes(b'12345');expected=policy.package.sha(model)
            job=base/'M7Jobs/job';job.write_bytes(b'{}');job_sha=policy.package.sha(job)
            # Metadata now sees one byte. Restore the admitted content exactly
            # when hashing opens it: old separate stat/hash code counted1+2.
            model.write_bytes(b'1');actual_open=Path.open
            def replaced_open(path,*args,**kwargs):
                if path==model:
                    with builtins.open(model,'wb') as stream:stream.write(b'12345')
                return actual_open(path,*args,**kwargs)
            with patch.object(Path,'open',replaced_open):
                inventory,total=policy.inventory(base,dict(files={'Model/weights':expected}),'M7Jobs/job',job_sha)
            self.assertEqual(total,7);self.assertEqual(inventory['Model/weights']['bytes'],5)

    def test_freshness_uses_start_both_clocks_and_same_boot(self):
        start=10**12;a=dict(host_boot_token=policy.boot_token(BOOT),measurement_host_start=dict(wallUnixNanoseconds=start,monotonicNanoseconds=start))
        observed=policy.fresh(a,a['host_boot_token'],start+policy.MAX_AGE_NS-1,start+1,'copy-job')
        self.assertEqual(observed['wall_age_nanoseconds'],policy.MAX_AGE_NS-1)
        for wall,mono,boot in [(start+policy.MAX_AGE_NS,start+1,a['host_boot_token']),
                              (start+1,start+policy.MAX_AGE_NS,a['host_boot_token']),
                              (start-1,start+1,a['host_boot_token']),(start+1,start-1,a['host_boot_token']),
                              (start,start,'a'*64),(True,start,a['host_boot_token'])]:
            with self.subTest(wall=wall,mono=mono),self.assertRaises(ValueError):policy.fresh(a,boot,wall,mono,'copy-Model')
        # A receipt completing just now cannot freshen a start older than5min.
        a['receipt_completed_unix_ns']=start+policy.MAX_AGE_NS+1
        with self.assertRaises(ValueError):policy.fresh(a,a['host_boot_token'],start+policy.MAX_AGE_NS+1,start+2,'copy-Model')

    def test_proof_replays_same_install_and_rejects_identity_clock_binding_or_console_changes(self):
        with tempfile.TemporaryDirectory(dir=policy.ROOT/'artifacts') as temporary:
            folder=Path(temporary)/'proof';r,context,launch=proof_fixture(folder)
            def admit(record):
                p=folder/'measurement.json';p.write_text(json.dumps(record))
                with patch.object(policy.physical,'check_launch',return_value=launch):
                    return policy.admit_proof(folder,policy.package.sha(p),context,measurement.validate_console)
            admitted,pins=admit(r);self.assertEqual(admitted,r);self.assertIn(str((folder/'measurement.json').relative_to(policy.ROOT)),pins)
            for kind in ('failed','config','app','selection','missingInput','missingClock','missingBoot','rawPin','cleanup','replay','scope'):
                bad=copy.deepcopy(r)
                if kind=='failed':bad['passed']=False
                elif kind=='config':bad['config']['maximum']=1
                elif kind=='app':bad['signed_bundle']['TokletHarness']='c'*64
                elif kind=='selection':bad['bindings'][context['selection_path']]='c'*64
                elif kind=='missingInput':bad['bindings'].clear()
                elif kind=='missingClock':bad.pop('measurement_host_start')
                elif kind=='missingBoot':bad.pop('host_boot_token')
                elif kind=='rawPin':bad['private_evidence_pins'].pop(str((folder/'private-logs/host-boot.stdout').relative_to(policy.ROOT)))
                elif kind=='cleanup':bad['remote_cleanup'][0]['errors']=['failed']
                elif kind=='replay':bad['checked']['result']['measuredCompleted']=9
                else:bad['physical_device_qualified']=True
                with self.subTest(kind=kind),self.assertRaises(ValueError):admit(bad)
            (folder/'private-logs/measurement.stdout').write_text('changed')
            with self.assertRaises(ValueError):admit(r)

    def test_ready_requires_complete_original_inventory_and_every_copy_decision(self):
        with tempfile.TemporaryDirectory(dir=policy.ROOT/'artifacts') as temporary:
            root=Path(temporary);folder=root/'proof';r,context,launch=proof_fixture(folder)
            record=folder/'measurement.json';record.write_text(json.dumps(r));proof_sha=policy.package.sha(record)
            base=root/'transfer';(base/'Model').mkdir(parents=True);(base/'M7Jobs').mkdir();model=base/'Model/w';model.write_bytes(b'12345')
            job=base/'M7Jobs/job';job.write_bytes(b'{}');config=dict(files={'Model/w':policy.package.sha(model)},packedFolder='')
            payload,total=policy.inventory(base,config,'M7Jobs/job',policy.package.sha(job));decision=policy.decision(r,folder,proof_sha,payload,total)
            stamp=r['measurement_host_start']['wallUnixNanoseconds']+1
            decision['checks']=[policy.fresh(decision,r['host_boot_token'],stamp,stamp,label) for label in ['copy-Model','copy-job']]
            ready=dict(storage_admission=decision)
            def admit(value):
                with patch.object(policy.physical,'check_launch',return_value=launch):
                    return policy.admit_ready(value,base,config,'M7Jobs/job',policy.package.sha(job),context,measurement.validate_console)
            observed,_=admit(ready);self.assertEqual(observed,decision)
            # No live clock is consulted by no-copy reuse; original decisions
            # are recomputed from their hash-bound evidence and recorded times.
            for kind in ('missing','missingCopy','wrongBytes','wrongReserve','wrongLabel','stale'):
                bad=copy.deepcopy(ready)
                if kind=='missing':bad.clear()
                elif kind=='missingCopy':bad['storage_admission']['checks'].pop()
                elif kind=='wrongBytes':bad['storage_admission']['payload_bytes']+=1
                elif kind=='wrongReserve':bad['storage_admission']['reserve_bytes']-=1
                elif kind=='wrongLabel':bad['storage_admission']['checks'][0]['label']='other'
                else:bad['storage_admission']['checks'][0]['wallUnixNanoseconds']+=policy.MAX_AGE_NS
                with self.subTest(kind=kind),self.assertRaises(ValueError):admit(bad)

    def test_options_reject_conflicts_duplicates_and_unused_flags(self):
        self.assertEqual(measurement.options(['--prepare-only','--storage-proof','folder','sha']),dict(prepare_only=True,storage_proof=Path('folder'),storage_sha='sha'))
        for flags in [['--prepare-only','--prepare-only'],['--unknown'],['--storage-proof'],['--prepare-only','--reuse-transfer','folder','sha'],
                      ['--storage-proof','folder','sha','--reuse-transfer','folder','sha']]:
            with self.subTest(flags=flags),self.assertRaises(ValueError):measurement.options(flags)

    def test_real_copy_without_storage_proof_has_zero_children(self):
        with tempfile.TemporaryDirectory(dir=policy.ROOT/'artifacts') as temporary:
            base=Path(temporary);private=base/'private';private.write_text('{}')
            with patch.object(measurement.physical,'selection',return_value={}),patch.object(measurement.integration,'admit_build',return_value={}), \
                 patch.object(measurement.physical,'admit_reuse',return_value=(base,{},'/owned/app',{})), \
                 patch.object(measurement,'job',return_value=({}, {}, {})),patch.object(measurement,'stage') as stage, \
                 patch.object(measurement.package,'owned') as owned,patch('builtins.print'):
                code=measurement.main(base,private,policy.package.sha(private),base,'prepared','small','short',base/'output',prepare_only=True)
            self.assertEqual(code,1);owned.assert_not_called();stage.assert_not_called()

    def test_main_checks_age_after_hashing_before_model_and_job_copy_and_still_cleans(self):
        for mode in ('success','model-expiry','job-expiry'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir=policy.ROOT/'artifacts') as temporary:
                base=Path(temporary);private=base/'private';private.write_text('{}');payload=base/'payload';payload.write_bytes(b'known')
                config,signed,_,_,_=fixture();target='M7Inputs/small/Model/config.json'
                config.update(modelFolder='M7Inputs/small/Model',packedFolder='',files={target:policy.package.sha(payload)})
                tiny,_,_,_,_=fixture();host={target:str(payload.relative_to(policy.ROOT))};boot=policy.boot_token(BOOT)
                start=10**12;clock=start+1;labels=[];installs=0;expire_after_hash=False
                proof=dict(device={},host_boot_token=boot,measurement_host_start=dict(wallUnixNanoseconds=start,monotonicNanoseconds=start),
                           checked=dict(result=dict(filesystemBefore=dict(freeBytes=16*1024**3))))
                def job(model,*_,**__):
                    return (tiny,{}, {}) if model=='tiny-cpu' else (config,host,{str(payload.relative_to(policy.ROOT)):policy.package.sha(payload)})
                actual_verify=measurement.verify_inputs
                def verify(pins):
                    nonlocal expire_after_hash,clock
                    actual_verify(pins)
                    if expire_after_hash:clock=start+policy.MAX_AGE_NS;expire_after_hash=False
                def owned(args,stdout,stderr,env,timeout):
                    nonlocal installs,expire_after_hash
                    label=stdout.stem;labels.append(label);data={}
                    if label.startswith('current-install-'):
                        installs+=1
                        expire_after_hash=(mode=='model-expiry' and installs==2) or (mode=='job-expiry' and installs==3)
                    if label.startswith('process-'):data={'result':{'runningProcesses':[]}}
                    elif label=='copied-inventory':
                        data=dict(info=dict(outcome='success'),result=dict(deviceIdentifier='selected',domain='appDataContainer',domainIdentifier=measurement.physical.BUNDLE,
                            files=[dict(name='Model/config.json',relativePath='Model/config.json',resources=dict(isDirectory=False,isSymbolicLink=False),metadata=dict(size=5))]))
                    if label=='host-boot':stdout.write_text(BOOT+'\n')
                    elif label in ('profile','entitlements'):stdout.write_bytes(plistlib.dumps({}))
                    else:stdout.write_text(json.dumps(data))
                    stderr.write_text('')
                    return dict(command=args,exit_code=0,primary_error=None,cleanup_errors=[],pid=1,stdout=str(stdout.relative_to(policy.ROOT)),stderr=str(stderr.relative_to(policy.ROOT)))
                def guard(current,action,*_):current();return action()
                with patch.object(measurement.physical,'selection',return_value={'device_identifier':'selected'}), \
                     patch.object(measurement.integration,'admit_build',return_value={}),patch.object(measurement.physical,'admit_reuse',return_value=(base,{},'/owned/app',signed)), \
                     patch.object(measurement,'job',side_effect=job),patch.object(policy,'admit_proof',return_value=(proof,{})), \
                     patch.object(measurement.package,'owned',side_effect=owned),patch.object(measurement,'verify_inputs',side_effect=verify), \
                     patch.object(measurement.physical,'admit_device',return_value={}),patch.object(measurement.physical,'admit_profile',return_value={}), \
                     patch.object(measurement.physical,'launch_verified',side_effect=guard),patch.object(measurement.physical,'inventory',return_value=signed), \
                     patch.object(measurement.time,'time_ns',side_effect=lambda:clock),patch.object(measurement.time,'monotonic_ns',side_effect=lambda:clock),patch('builtins.print'):
                    code=measurement.main(base,private,policy.package.sha(private),base,'prepared','small','short',base/'output',prepare_only=True,storage_proof=base,storage_sha='proof')
                record=json.loads((base/'output/measurement.json').read_text())
                self.assertEqual(code,0 if mode=='success' else 1)
                self.assertEqual([x for x in labels if x.startswith('copy-')],[] if mode=='model-expiry' else ['copy-Model'] if mode=='job-expiry' else ['copy-Model','copy-job'])
                self.assertEqual(len(record['remote_cleanup']),2);self.assertTrue(all(c['absent'] and not c['errors'] for c in record['remote_cleanup']))
                self.assertIsNone(record['checked'])


if __name__=='__main__':unittest.main()
