"""Offline loss alignment, numerical, ownership and evidence-boundary tests."""
import copy
import hashlib
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

import quality


def fixture(tokens,projection='scalar'):
    count=len(tokens)-1;sha=hashlib.sha256(struct.pack(f'<{len(tokens)}I',*tokens)).hexdigest()
    ready={'event':'ready','format':'toklet-held-out-loss-events-v1','cpu_projection':projection,
        'target_tokens':count,'corpus_sha256':sha,'context_policy':quality.POLICY,'block_targets':1024,
        'context_capacity':1025,'prefill_chunk_size':1,'weight_dtype':'FP32','kv_dtype':'FP32',
        'logical_weights_bytes':1000,'allocated_kv_bytes':200,'scratch_bytes':50,'history_bytes':40,
        'logits_bytes':100,'metadata_bytes':8,'metadata_allowance_bytes':10000,'application_allowance_bytes':65536}
    ready['memory_admitted_bytes']=sum(ready[key] for key in ('logical_weights_bytes','allocated_kv_bytes','scratch_bytes','history_bytes','logits_bytes','metadata_bytes','metadata_allowance_bytes','application_allowance_bytes'))
    rows=[{'event':'loss','target_index':i,'block_index':i//1024,'position':i%1024,'prefix_tokens':i%1024+1,
           'target_token':token,'negative_log_likelihood':1.0+(i%13)/16,'target_logit_f32_bits':0,'logits_sha256':'0'*64} for i,token in enumerate(tokens[1:])]
    blocks=[]
    for start in range(0,count,1024):
        end=min(start+1024,count);size=end-start
        blocks.append({'event':'block_done','block_index':start//1024,'target_start':start,'target_end':end,
            'scored_targets':size,'evaluated_tokens':size+1,'pending_tokens':0,
            'total_negative_log_likelihood':math.fsum(row['negative_log_likelihood'] for row in rows[start:end])})
    total=math.fsum(row['negative_log_likelihood'] for row in rows)
    done={'event':'done','scored_targets':count,'total_negative_log_likelihood':total,'mean_negative_log_likelihood':total/count,
          'evaluated_tokens':blocks[-1]['scored_targets']+1,'pending_tokens':0}
    return {'ready':ready,'losses':rows,'blocks':blocks,'done':done,'physical_process_charge_peak_bytes':100000},sha


class Quality(unittest.TestCase):
    def test_independent_stable_loss_preserves_equal_extreme_and_shifted_logits(self):
        self.assertEqual(quality.stable_loss([42.0],0),0)
        largest=struct.unpack('<f',struct.pack('<I',0x7f7fffff))[0]
        self.assertAlmostEqual(quality.stable_loss([largest]*3,2),math.log(3),places=14)
        self.assertEqual(quality.stable_loss([largest,-largest],1),2*largest)
        self.assertEqual(quality.stable_loss([-1000,-999],0),quality.stable_loss([1000,1001],0))
        for values,target in [([],0),([0],1),([math.nan],0),([0,math.inf],0)]:
            with self.subTest(values=values),self.assertRaises(ValueError):quality.stable_loss(values,target)

    def test_block_transition_and_full_tail_cover_every_target_once(self):
        for count in (1,1024,1025,10000):
            tokens=[i%127 for i in range(count+1)];run,sha=fixture(tokens)
            self.assertEqual(quality.validate_run(run,tokens,'scalar',sha),run['done']['mean_negative_log_likelihood'])
            self.assertEqual(run['blocks'][-1]['evaluated_tokens'],(count-1)%1024+2)

    def test_alignment_pending_and_missing_or_extra_scores_fail_closed(self):
        tokens=list(range(1026));original,sha=fixture(tokens)
        for kind in ('label','position','prefix','pending','missing','extra','block_start','total','mean','nonfinite','logit','sha'):
            run=copy.deepcopy(original)
            if kind=='label':run['losses'][1024]['target_token']+=1
            elif kind=='position':run['losses'][1024]['position']=1024
            elif kind=='prefix':run['losses'][1024]['prefix_tokens']=1025
            elif kind=='pending':run['blocks'][0]['pending_tokens']=1
            elif kind=='missing':run['losses'].pop(0)
            elif kind=='extra':run['losses'].append(run['losses'][-1])
            elif kind=='block_start':run['blocks'][1]['target_start']-=1
            elif kind=='total':run['done']['total_negative_log_likelihood']+=.01
            elif kind=='mean':run['done']['mean_negative_log_likelihood']+=.01
            elif kind=='nonfinite':run['losses'][0]['negative_log_likelihood']=math.nan
            elif kind=='logit':run['losses'][0]['target_logit_f32_bits']=0x7f800000
            elif kind=='sha':run['losses'][0]['logits_sha256']='g'*64
            with self.subTest(kind=kind),self.assertRaises(ValueError):quality.validate_run(run,tokens,'scalar',sha)

    def test_backend_policy_and_memory_require_actual_acknowledgement(self):
        tokens=[1,2,3];original,sha=fixture(tokens)
        for field,value in [('cpu_projection','tiled'),('context_policy','sliding'),('context_capacity',2048),('prefill_chunk_size',128),
                            ('weight_dtype','BF16'),('corpus_sha256','f'*64),('memory_admitted_bytes',quality.LIMIT+1),('scratch_bytes',-1)]:
            run=copy.deepcopy(original);run['ready'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):quality.validate_run(run,tokens,'scalar',sha)
        original['physical_process_charge_peak_bytes']=quality.LIMIT+1
        with self.assertRaises(ValueError):quality.validate_run(original,tokens,'scalar',sha)

    def test_exact_hash_agreement_does_not_replace_independent_nll_gate(self):
        losses=[{'negative_log_likelihood':1.0},{'negative_log_likelihood':2.0}]
        self.assertEqual(quality.check_probe(losses,[1,2]),[0,0])
        with self.assertRaisesRegex(ValueError,'NLL gate'):quality.check_probe(losses,[1,2.01])
        with self.assertRaisesRegex(ValueError,'coverage'):quality.check_probe(losses,[1])

    def test_output_and_build_are_fresh_and_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'artifacts').mkdir();(root/'benchmarks/raw').mkdir(parents=True)
            output=root/'benchmarks/raw/result.json';build=root/'artifacts/build'
            with patch.object(quality,'ROOT',root):
                quality.check_paths(output,build)
                with self.assertRaisesRegex(ValueError,'directories'):quality.check_paths(root/'outside.json',build)
                with self.assertRaisesRegex(ValueError,'directories'):quality.check_paths(output,root/'benchmarks/build')
                output.write_text('{}')
                with self.assertRaisesRegex(ValueError,'overwrite'):quality.check_paths(output,build)
                output.unlink();build.mkdir()
                with self.assertRaisesRegex(ValueError,'overwrite'):quality.check_paths(output,build)

    def test_interrupted_check_kills_and_reaps_its_owned_group_and_retains_output(self):
        for failure in (KeyboardInterrupt('interrupted'),SystemExit(7),RuntimeError('read failed')):
            worker=Mock(pid=12345,returncode=None)
            def communicate(*,timeout):
                if timeout==1200:raise failure
                self.assertEqual(timeout,20);worker.returncode=-9
                return 'retained stdout','retained stderr'
            worker.communicate.side_effect=communicate;record={'checks':[]}
            with self.subTest(failure=type(failure).__name__),patch.object(quality.subprocess,'Popen',return_value=worker),patch.object(quality.os,'killpg') as kill:
                with self.assertRaises(type(failure)) as caught:quality.checked(['cargo','test'],record)
                self.assertIs(caught.exception,failure)
                kill.assert_called_once_with(worker.pid,quality.signal.SIGKILL)
                self.assertEqual(worker.communicate.call_count,2)
                result=record['checks'][0]
                self.assertEqual((result['exit_code'],result['stdout'],result['stderr']),(-9,'retained stdout','retained stderr'))
                self.assertEqual(result['failure']['type'],type(failure).__name__)
                self.assertTrue(result['process_group_killed']);self.assertFalse(result['timed_out'])
                self.assertEqual(result['cleanup_errors'],[])
                worker.stdout.close.assert_called_once();worker.stderr.close.assert_called_once()


if __name__=='__main__':unittest.main()
