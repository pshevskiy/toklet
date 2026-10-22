"""Portable raw-ID recipe/staging checks; no GPU, MLX package or model download."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import shutil
import struct
import unittest
from unittest import mock

import m8_mlx_jobs as jobs


class Jobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (jobs.ROOT/'artifacts').mkdir(exist_ok=True)

    def test_explicit_backend_and_immutable_selected_identity(self):
        for model in ['tiny','qwen','qwen8']:
            mode,config,host,pins,payloads=jobs.job(model,'measurement-short')
            self.assertEqual(mode,'measurement');self.assertEqual(config['version'],2)
            self.assertIs(config['mlx'],True);self.assertIs(config['metal'],True)
            self.assertEqual(config['packedFolder'],'');self.assertEqual(config['warmups'],3);self.assertEqual(config['repetitions'],10)
            self.assertEqual(set(host),set(config['files']));self.assertFalse(payloads)
            self.assertEqual(len(host),2 if model=='tiny' else 3)
            self.assertEqual(config['expectedPrefix'],{'tiny':[47]*4,'qwen':[264,5458,315,17407],'qwen8':[4460,311,1855,264]}[model])
            reference,reference_sha=(jobs.REFERENCE8,jobs.REFERENCE8_SHA) if model=='qwen8' else (jobs.REFERENCE,jobs.REFERENCE_SHA)
            self.assertEqual(pins[reference],reference_sha)
            if model=='qwen8':self.assertTrue(all('8bit-6474d4c' in source for source in host.values()))
            for target,source in host.items():self.assertEqual(pins[source],config['files'][target])
            self.assertLess(config['numericalBudget'],config['processBudget'])
            self.assertLessEqual(config['processBudget'],4*1024**3)

    def test_contract_prompts_and_completed_work_policy_are_unchanged(self):
        for workload,count in [('128',128),('512',512),('1024',1024),('sustained',512)]:
            _,config,_,pins,_=jobs.job('qwen','measurement-'+workload)
            self.assertEqual(len(config['prompt']),count);self.assertEqual(config['chunk'],128)
            self.assertEqual(config['capacity'],1152);self.assertEqual(config['maximum'],128)
            self.assertEqual(config['minimumNativeSeconds'],600 if workload=='sustained' else 0)
            self.assertEqual(pins[jobs.CONTRACT],jobs.CONTRACT_SHA)

    def test_state512_uses_original_prompt_and_exact_native_state_fields(self):
        for model in ['qwen','qwen8']:
            mode,c,host,pins,payloads=jobs.job(model,'state-512')
            self.assertEqual(mode,'state');self.assertFalse(payloads)
            raw=(jobs.ROOT/'benchmarks/contract-v1/prompt_512.u32le').read_bytes()
            self.assertEqual(c['prompt'],list(struct.unpack('<512I',raw)))
            self.assertEqual((c['capacity'],c['chunk'],c['maximum'],c['vocabulary']),(1152,128,128,151936))
            self.assertEqual(set(c),{'version','mlx','modelFolder','packedFolder','files','metal','numericalBudget','processBudget','capacity','chunk','prompt','maximum','vocabulary'})
            self.assertIs(c['mlx'],True);self.assertEqual(c['version'],2)
            self.assertEqual(set(host),set(c['files']));self.assertEqual(pins[jobs.CONTRACT],jobs.CONTRACT_SHA)
            measurement=jobs.job(model,'measurement-512')[1]
            for key in c.keys()-{'vocabulary'}:self.assertEqual(c[key],measurement[key])
        with self.assertRaises(ValueError):jobs.job('tiny','state-512')

    def test_shared_prompt_capture_rejects_metadata_and_consumed_size_changes(self):
        real=jobs.captured
        for kind in ['metadata','size']:
            def captured(name,pin,maximum):
                raw=real(name,pin,maximum)
                if name==jobs.CONTRACT and kind=='metadata':
                    data=json.loads(raw);data['arrays']['prompt_512']['count']=511;return json.dumps(data).encode()
                if name.endswith('prompt_512.u32le') and kind=='size':return raw[:-4]
                return raw
            with self.subTest(kind=kind),mock.patch.object(jobs,'captured',captured):
                for workload in ['state-512','measurement-512']:
                    with self.assertRaises(ValueError):jobs.job('qwen8',workload)

    def test_lifecycle_uses_the_same_native_schema_with_explicit_operation(self):
        for model in ['qwen','qwen8']:
            mode,c,host,pins,payloads=jobs.job(model,'lifecycle-512')
            _,state,state_host,state_pins,state_payloads=jobs.job(model,'state-512')
            self.assertEqual(mode,'lifecycle');self.assertEqual(jobs.encoded(c),jobs.encoded(state))
            self.assertEqual((host,pins,payloads),(state_host,state_pins,state_payloads))
        with self.assertRaises(ValueError):jobs.job('tiny','lifecycle-512')

    def test_full_quality_scores_every_original_target_once(self):
        _,config,_,pins,payloads=jobs.job('qwen','quality-full')
        self.assertEqual(config['targets'],10000);self.assertEqual(config['blockSize'],1024)
        self.assertEqual(config['capacity'],1025);self.assertEqual(config['chunk'],1)
        raw=payloads[config['tokensFile']]
        self.assertEqual(len(raw),40004);self.assertEqual(hashlib.sha256(raw).hexdigest(),jobs.CORPUS_SHA)
        self.assertEqual(pins[jobs.CORPUS],jobs.CORPUS_SHA)

    def test_tasks_preserve_literal_ids_caps_stops_and_order(self):
        original=[dict(id='HumanEval/'+str(i),prompt_tokens=[9707,11],max_output_tokens=256 if i<40 else 32) for i in range(100)]
        real=jobs.captured
        def captured(name,pin,maximum):
            return b'\n'.join(json.dumps(item).encode() for item in original) if name==jobs.TASKS else real(name,pin,maximum)
        with mock.patch.object(jobs,'captured',captured):
            for workload in ['tasks-short','tasks-full']:
                mode,config,_,pins,payloads=jobs.job('qwen',workload)
                data=json.loads(payloads[config['tasksFile']]);cases=data['cases']
                self.assertEqual(mode,'tasks');self.assertEqual(data['stops'],[151645,151643])
                self.assertEqual(len(cases),1 if workload=='tasks-short' else 100)
                for actual,source in zip(cases,original):
                    self.assertEqual(actual,dict(id=source['id'],promptTokens=source['prompt_tokens'],maximum=min(8,source['max_output_tokens']) if workload=='tasks-short' else source['max_output_tokens']))
                self.assertEqual(pins[jobs.TASKS],jobs.TASKS_SHA)

    def test_eight_bit_preserves_full_workload_contracts(self):
        for workload in jobs.WORKLOADS:
            mode4,config4,_,pins4,payload4=jobs.job('qwen',workload)
            mode8,config8,_,pins8,payload8=jobs.job('qwen8',workload)
            self.assertEqual(mode8,mode4);self.assertEqual(payload8,payload4)
            for key in config4.keys()-{'files','modelFolder','expectedPrefix'}:self.assertEqual(config8[key],config4[key])
            self.assertEqual(config8['modelFolder'],'M8Inputs/qwen8/Model')
            self.assertIn(jobs.REFERENCE8,pins8);self.assertNotIn(jobs.REFERENCE,pins8)
            for name in [jobs.CONTRACT,jobs.CORPUS,jobs.TASKS]:
                if name in pins4:self.assertEqual(pins8[name],pins4[name])
        with self.assertRaises(ValueError):jobs.captured(jobs.REFERENCE8,'0'*64,256*1024)

    def test_rejects_unsupported_and_changed_captured_input(self):
        for model,workload in [('unknown','measurement-short'),('tiny','quality-full'),('tiny','tasks-full'),('qwen','unknown')]:
            with self.assertRaises(ValueError):jobs.job(model,workload)
        with self.assertRaises(ValueError):jobs.captured(jobs.REFERENCE,'0'*64,256*1024)
        with self.assertRaises(ValueError):jobs.captured(jobs.REFERENCE,jobs.REFERENCE_SHA,1)

    def test_staged_tiny_is_exclusive_hashed_and_shares_only_immutable_weights(self):
        with tempfile.TemporaryDirectory(dir=jobs.ROOT/'artifacts') as directory:
            output=Path(directory)/'trial';record=jobs.stage('tiny','quality-short',output)
            base=output/'transfer';config=record['config']
            self.assertTrue(record['passed']);self.assertFalse(record['physical_device_qualified'])
            raw=(base/record['job']['relative_path']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),record['job']['sha256']);self.assertEqual(json.loads(raw),config)
            for target,source in record['host'].items():
                self.assertEqual((base/target).stat().st_ino,(jobs.ROOT/source).stat().st_ino)
            for name,item in record['staged_files'].items():
                self.assertEqual(jobs.sha(base/name),item['sha256']);self.assertEqual((base/name).stat().st_size,item['bytes'])
            with self.assertRaises(FileExistsError):jobs.stage('tiny','quality-short',output)
            dangling=Path(directory)/'dangling';dangling.symlink_to(Path(directory)/'missing')
            with self.assertRaises(FileExistsError):jobs.stage('tiny','quality-short',dangling)
        with tempfile.TemporaryDirectory() as outside:
            with self.assertRaises(ValueError):jobs.stage('tiny','quality-short',Path(outside)/'trial')

    def test_initial_artifact_mismatch_rejects_before_staging(self):
        # Fail admission before staging rather than altering any actual fixture.
        with tempfile.TemporaryDirectory(dir=jobs.ROOT/'artifacts') as directory:
            output=Path(directory)/'trial';mode,config,host,pins,payloads=jobs.job('tiny','measurement-short')
            config=copy.deepcopy(config);config['files'][next(iter(host))]='0'*64
            with mock.patch.object(jobs,'job',return_value=(mode,config,host,pins,payloads)):
                with self.assertRaises(ValueError):jobs.stage('tiny','measurement-short',output)
            self.assertFalse(output.exists())

    def test_post_link_endpoint_mutation_leaves_no_success_receipt(self):
        with tempfile.TemporaryDirectory(dir=jobs.ROOT/'artifacts') as directory:
            root=Path(directory);inputs=root/'inputs';inputs.mkdir()
            mode,config,host,_,payloads=jobs.job('tiny','measurement-short')
            copies={};pins={jobs.REFERENCE:jobs.REFERENCE_SHA}
            for target,source in host.items():
                path=inputs/Path(source).name;shutil.copyfile(jobs.ROOT/source,path)
                copies[target]=str(path.relative_to(jobs.ROOT));pins[copies[target]]=jobs.sha(path)
            victim=inputs/'config.json';original_encoded=jobs.encoded
            def encoded(value):
                raw=original_encoded(value)
                with victim.open('ab') as stream:stream.write(b' ')
                return raw
            output=root/'trial'
            with mock.patch.object(jobs,'job',return_value=(mode,config,copies,pins,payloads)),mock.patch.object(jobs,'encoded',encoded):
                with self.assertRaisesRegex(ValueError,'endpoint changed'):jobs.stage('tiny','measurement-short',output)
            self.assertTrue((output/'transfer').exists())
            self.assertFalse((output/'staged.json').exists())


if __name__=='__main__':unittest.main()
