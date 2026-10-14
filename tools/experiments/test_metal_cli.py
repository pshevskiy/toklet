"""Offline actual-mode/memory/state rejection and owned worker cleanup."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
import metal_cli as cli

class MetalCli(unittest.TestCase):
    def text_events(self,count,*,stop=False):
        sampled=1 if stop else 8
        return [{'event':'done','statistics':{'generation':{'reused_tokens':0,'prefill_tokens':count,'sampled_tokens':sampled,'emitted_tokens':0 if stop else 8,'evaluated_tokens':count+sampled-1,'pending_tokens':1,'reason':{'kind':'stop_token' if stop else 'output_limit'}}}}]

    def test_text_generation_error_cannot_pass_empty_decode(self):
        request=Mock(return_value=[{'event':'error','message':'rejected'}])
        with self.assertRaisesRegex(ValueError,'did not complete'):cli.text_script(request,{},4,{'messages':[],'tools':None,'ids':[1]*31})
        request.assert_called_once()

    def test_text_completion_and_both_resets_required(self):
        case={'messages':[],'tools':None,'ids':[1]*31}
        for stop in (False,True):
            request=Mock(side_effect=[self.text_events(4,stop=stop),[{'event':'reset'}],self.text_events(31,stop=stop),[{'event':'reset'}]])
            cli.text_script(request,{},4,case);self.assertEqual(request.call_count,4)
        for final in ([],[{'event':'error'}],[{'event':'done'}]):
            with self.subTest(final=final),self.assertRaisesRegex(ValueError,'reset not acknowledged'):
                cli.text_script(Mock(side_effect=[self.text_events(4),[{'event':'reset'}],self.text_events(31),final]),{},4,case)
        with self.assertRaisesRegex(ValueError,'prompt/reset'):
            cli.text_script(Mock(return_value=self.text_events(5)),{},4,case)

    def test_frozen_chat_manifest_admitted_before_parse(self):
        value=cli.chat_case(cli.ROOT/'tests/fixtures/chat-v1/manifest.json');self.assertEqual(len(value['ids']),31)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'manifest differs'):cli.chat_case(path)

    def test_run_preflight_reaches_owned_worker_after_fixed_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'models/Qwen2.5-0.5B-Instruct';source.mkdir(parents=True)
            fixture=root/'tests/fixtures/chat-v1/manifest.json';fixture.parent.mkdir(parents=True);fixture.write_bytes((cli.ROOT/'tests/fixtures/chat-v1/manifest.json').read_bytes())
            for name in ('model.safetensors','config.json',*cli.ASSET_PINS):(source/name).touch()
            pins={**cli.ASSET_PINS,'model.safetensors':'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe','config.json':'18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45','manifest.json':cli.CHAT_SHA}
            with patch.object(cli,'ROOT',root),patch.object(cli.metal_engine,'digest',side_effect=lambda p:pins.get(p.name,'0'*64)),patch.object(cli,'child',side_effect=RuntimeError('first owned worker')) as child:
                with self.assertRaisesRegex(ValueError,'qualification failed'):cli.run(root/'binary',root/'result.json')
            child.assert_called_once();saved=json.loads((root/'result.json').read_text());self.assertEqual(saved['error'],'RuntimeError: first owned worker');self.assertEqual(len(saved['text_chat_case']['ids']),31)

    def fixture(self):
        memory={k:1 for k in cli.FIELDS}
        return {'event':'ready','backend':'metal','cpu_projection':'none','context_capacity':32,'execution':cli.execution('HalfCompensated',3),'logical_memory':memory,'memory_admitted_bytes':8}

    def test_actual_mode_representation_chunk_and_complete_admission_required(self):
        e=self.fixture();cli.ready(e,'HalfCompensated',32,3)
        changes=[('backend','cpu'),('cpu_projection','scalar'),('context_capacity',True),('memory_admitted_bytes',7)]
        for key,value in changes:
            v=copy.deepcopy(e);v[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):cli.ready(v,'HalfCompensated',32,3)
        for key,value in [('projection_mode','Original'),('weight_dtype','FP32'),('normalization_mode','Compensated'),('prefill_chunk_size',3.0)]:
            v=copy.deepcopy(e);v['execution'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):cli.ready(v,'HalfCompensated',32,3)
        v=copy.deepcopy(e);v['logical_memory']['weights_bytes']=True
        with self.assertRaises(ValueError):cli.ready(v,'HalfCompensated',32,3)
        v=copy.deepcopy(e);del v['logical_memory']['history_bytes']
        with self.assertRaises(ValueError):cli.ready(v,'HalfCompensated',32,3)

    def test_statistics_must_match_ready_and_valid_committed_prefix(self):
        ready=self.fixture();s={k:v for k,v in ready.items() if k not in ('event','context_capacity')};s['generation']={'reused_tokens':0,'prefill_tokens':4,'sampled_tokens':4,'emitted_tokens':4,'history_tokens':8,'evaluated_tokens':7,'pending_tokens':1};e={'event':'done','statistics':s};cli.stats(e,ready)
        for key,value in [('history_tokens',9),('pending_tokens',2),('emitted_tokens',5),('evaluated_tokens',True)]:
            v=copy.deepcopy(e);v['statistics']['generation'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):cli.stats(v,ready)
        v=copy.deepcopy(e);v['statistics']['execution']['projection_mode']='Original'
        with self.assertRaises(ValueError):cli.stats(v,ready)

    def test_bad_ready_preserves_error_reaps_owned_process(self):
        real=subprocess.Popen;children=[]
        def launch(command,**kwargs):
            self.assertTrue(kwargs['start_new_session']);p=real([sys.executable,'-c','print(\'{"event":"wrong"}\',flush=True);import time;time.sleep(60)'],**kwargs);p.wait=Mock(wraps=p.wait);children.append(p);return p
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with (root/'raw').open('x') as raw,(root/'stderr').open('xb') as stderr,patch.object(cli.subprocess,'Popen',launch):
                result=cli.child(root/'binary',root,'HalfCompensated',raw,stderr,lambda *_:self.fail('bad ready ran script'))
        self.assertFalse(result['passed']);self.assertIn('execution metadata',result['error']);self.assertEqual(result['cleanup_errors'],[]);self.assertIsNotNone(children[0].returncode);children[0].wait.assert_called_once_with(timeout=5)

if __name__=='__main__':unittest.main()
