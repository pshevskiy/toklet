"""Exact v2 CLI admission and shared owned-transport hook; no GPU/models."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

import metal_cli_v2 as v2


def fixture():
    memory={k:10 for k in v2.shared.FIELDS};memory['weights_bytes']=v2.TINY_WEIGHTS
    return {'event':'ready','backend':'metal','cpu_projection':'none','context_capacity':32,
            'execution':v2.execution(3),'logical_memory':memory,'memory_admitted_bytes':sum(memory.values())}


class V2CLI(unittest.TestCase):
    def test_exact_format_modes_and_complete_memory_are_required(self):
        event=fixture();v2.ready(event,v2.MODE,32,3,weights=v2.TINY_WEIGHTS)
        for field,value in [('weight_dtype','FP16'),('packed_format_version',1),('quantization_group_size',64),
                            ('packed_format_version',2.0),('quantization_group_size',8.0),
                            ('quantizer','other'),('projection_mode','Original'),('normalization_mode','Original'),
                            ('readback_mode','TokenOnly'),('prefill_chunk_size',True)]:
            bad=copy.deepcopy(event);bad['execution'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):v2.ready(bad,v2.MODE,32,3,weights=v2.TINY_WEIGHTS)
        bad=copy.deepcopy(event);bad['logical_memory']['weights_bytes']-=1;bad['memory_admitted_bytes']-=1
        with self.assertRaisesRegex(ValueError,'residency'):v2.ready(bad,v2.MODE,32,3,weights=v2.TINY_WEIGHTS)
        with self.assertRaises(ValueError):v2.ready(event,'HalfCompensated',32,3,weights=v2.TINY_WEIGHTS)
        with self.assertRaises(ValueError):v2.ready(event,v2.MODE,32,3,weights=v2.TINY_WEIGHTS,external=20)
        bad=copy.deepcopy(event);bad['memory_admitted_bytes']+=1
        with self.assertRaises(ValueError):v2.ready(bad,v2.MODE,32,3,weights=v2.TINY_WEIGHTS)

    def test_custom_admission_rejection_retains_owned_exit_and_cleanup(self):
        real=subprocess.Popen;children=[];commands=[]
        def launch(command,**kwargs):
            commands.append(command)
            child=real([sys.executable,'-c','print(\'{"event":"unexpected"}\',flush=True);import time;time.sleep(60)'],**kwargs)
            child.wait=Mock(wraps=child.wait);children.append(child);return child
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with (root/'raw').open('x') as raw,(root/'stderr').open('xb') as stderr,patch.object(v2.shared.subprocess,'Popen',launch):
                out=v2.call(root/'binary',root/'source',root/'packed',raw,stderr,lambda *_:self.fail('bad admission ran script'),weights=v2.TINY_WEIGHTS)
        self.assertFalse(out['passed']);self.assertIn('metadata differs',out['error']);self.assertEqual(out['cleanup_errors'],[])
        self.assertIsNotNone(children[0].returncode);children[0].wait.assert_called_once_with(timeout=5)
        self.assertEqual(commands[0][-2:],['--metal-projection',v2.ARG]);self.assertIn('--q4-weights',commands[0])

    def test_ready_and_statistics_metadata_must_agree_without_relabeling(self):
        event=fixture();g=dict(reused_tokens=0,prefill_tokens=4,sampled_tokens=1,emitted_tokens=1,history_tokens=5,evaluated_tokens=4,pending_tokens=1)
        stats=dict(backend='metal',cpu_projection='none',execution=v2.execution(3),logical_memory=event['logical_memory'],memory_admitted_bytes=event['memory_admitted_bytes'],generation=g)
        v2.shared.stats({'event':'done','statistics':stats},event)
        stats['execution']['weight_dtype']='FP16'
        with self.assertRaises(ValueError):v2.shared.stats({'event':'done','statistics':stats},event)

    def test_tiny_v1_rejection_command_uses_asset_free_json_raw_ids(self):
        command=v2.v1_command(Path('/binary'),Path('/tiny'),Path('/packed'))
        self.assertIn('--json',command);self.assertIn('--tokens',command)
        self.assertNotIn('--prompt',command);self.assertNotIn('--messages',command)
        self.assertEqual(command[command.index('--q4-weights')+1],'/packed')

    def test_authenticated_v1_artifact_loading_rejection_requires_runtime_exit_one(self):
        good=subprocess.CompletedProcess([],1,b'',b'error: --q4-weights requires a Q4 v2 artifact\n')
        v2.check_v1_rejection(good)
        for code,out,err in [(2,b'',good.stderr),(0,b'',good.stderr),(1,b'event',good.stderr),(1,b'',b'error: wrong source')]:
            with self.subTest(code=code,out=out,err=err),self.assertRaises(ValueError):
                v2.check_v1_rejection(subprocess.CompletedProcess([],code,out,err))

    def test_existing_output_rejects_before_qualification_or_child_access(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'out.json';p.write_text('immutable')
            with patch.object(v2.loss.shared,'frozen_bytes') as read,patch.object(v2.shared,'child') as child:
                with self.assertRaisesRegex(ValueError,'evidence exists'):v2.run(Path('/missing'),p)
                read.assert_not_called();child.assert_not_called()
            self.assertEqual(p.read_text(),'immutable')


if __name__=='__main__':unittest.main()
