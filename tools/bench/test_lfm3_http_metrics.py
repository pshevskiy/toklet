"""Portable external stream controls; no servers, model downloads or GPU needed."""
import copy
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lfm3_http_metrics as metrics


START = 1_000_000_000
MODEL = 'public-test'
PROMPT = [10,11]


def encoded(value):
    return json.dumps(value,separators=(',',':')).encode()


def ollama(done=False, **extra):
    row = dict(model=MODEL,done=done,response='')
    if done:
        row.update(done_reason='stop',prompt_eval_count=2,prompt_eval_cached_count=0,
            eval_count=2,total_duration=150,load_duration=10,prompt_eval_duration=100,eval_duration=20)
    row.update(extra)
    return row


def vllm(choices, usage=None, **extra):
    row = dict(id='cmpl-public-test',model=MODEL,object='text_completion',choices=choices)
    if usage is not None:
        row['usage'] = usage
    row.update(extra)
    return row


def choice(ids, text='', prompt=None, finish=None, stop=None):
    return dict(index=0,text=text,token_ids=ids,prompt_token_ids=prompt,
                finish_reason=finish,stop_reason=stop)


def usage(cached=0):
    return dict(prompt_tokens=2,completion_tokens=2,total_tokens=4,
                prompt_tokens_details=dict(cached_tokens=cached))


def vllm_rows():
    return [vllm([choice([7],'a',PROMPT)]),
            vllm([choice([metrics.EOS],'',finish='stop',stop=metrics.EOS)]),
            vllm([],usage())]


def timed(rows, done=False):
    events = [(encoded(row),START+(i+1)*100_000_000) for i,row in enumerate(rows)]
    if done:
        events.append((b'[DONE]',START+(len(rows)+1)*100_000_000))
    return events


def observe(engine, rows, done=False):
    state = metrics.state(engine)
    callback = metrics.observe_ollama if engine=='ollama' else metrics.observe_vllm
    callback(timed(rows,done),state,START,MODEL,PROMPT,2)
    return metrics.summarize(state,START,PROMPT)


class StreamTests(unittest.TestCase):
    def test_ollama_thinking_content_and_terminal_clocks_stay_separate(self):
        result = observe('ollama',[ollama(thinking='reason'),ollama(response='answer'),ollama(True)])
        self.assertEqual(result['client_metrics']['first_generated_text_seconds'],.1)
        self.assertEqual(result['client_metrics']['first_content_seconds'],.2)
        self.assertEqual(result['client_metrics']['request_seconds'],.3)
        self.assertEqual(result['engine_metrics']['api_eval_tokens_per_second'],100_000_000)
        self.assertIsNone(result['output_token_ids'])
        self.assertFalse(result['matched_input_qualified'])
        self.assertTrue(result['count_and_cache_match'])
        self.assertEqual(result['engine_metrics']['api_uncached_prefill_tokens_per_second'],20_000_000)

    def test_ollama_missing_cache_is_unavailable_rather_than_zero(self):
        final=ollama(True);del final['prompt_eval_cached_count']
        result=observe('ollama',[final])
        self.assertIsNone(result['cached_prompt_tokens'])
        self.assertIn('cached_prompt_tokens',result['unavailable'])
        self.assertFalse(result['count_and_cache_match'])
        self.assertIsNone(result['engine_metrics']['api_uncached_prefill_tokens_per_second'])
        self.assertIn('api_uncached_prefill_tokens_per_second',result['unavailable'])

    def test_ollama_empty_response_has_no_invented_first_text(self):
        result=observe('ollama',[ollama(True,eval_count=0,eval_duration=0)])
        self.assertIsNone(result['client_metrics']['first_generated_text_seconds'])
        self.assertIsNone(result['client_metrics']['first_content_seconds'])
        self.assertIsNone(result['engine_metrics']['api_eval_tokens_per_second'])
        self.assertIn('first_generated_text_seconds',result['unavailable'])
        self.assertIn('api_eval_tokens_per_second',result['unavailable'])

    def test_ollama_missing_and_duplicate_terminal_reject(self):
        for rows in [[ollama()],[ollama(True),ollama(True)],[ollama(True),ollama(response='late')]]:
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                observe('ollama',rows)

    def test_ollama_invalid_counters_and_identity_reject(self):
        for changes in [dict(eval_count=True),dict(eval_count=-1),dict(eval_count=3),
                dict(total_duration=None),dict(prompt_eval_cached_count=3),dict(model='other'),
                dict(done=1),dict(response=42),dict(done_reason='unload'),dict(error='engine failed')]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                row=ollama(True);row.update(changes)
                observe('ollama',[row])

    def test_vllm_prompt_ids_hidden_stop_and_usage_match(self):
        result=observe('vllm',vllm_rows(),True)
        self.assertEqual(result['output_token_ids'],[7,metrics.EOS])
        self.assertEqual(result['reported_output_tokens'],2)
        self.assertEqual(result['client_metrics']['first_token_batch_seconds'],.1)
        self.assertEqual(result['client_metrics']['request_seconds'],.4)
        self.assertIsNone(result['client_metrics']['first_content_seconds'])
        self.assertEqual(result['engine_metrics'],{})
        self.assertTrue(result['matched_input_qualified'])

    def test_vllm_batched_tokens_are_not_counted_as_singletons(self):
        rows=[vllm([choice([7,8],'ab',PROMPT,'length')]),vllm([],usage())]
        result=observe('vllm',rows,True)
        self.assertEqual(result['frames'],3)
        self.assertEqual(result['reported_output_tokens'],2)
        self.assertNotIn('decode_tokens_per_second',result['client_metrics'])
        self.assertIn('per_token_decode_rate',result['unavailable'])

    def test_vllm_added_bos_and_cache_reuse_fail_workload_qualification(self):
        for prompt,cached in [([metrics.EOS,*PROMPT],0),(PROMPT,1)]:
            rows=vllm_rows();rows[0]['choices'][0]['prompt_token_ids']=prompt
            rows[-1]['usage']['prompt_tokens_details']['cached_tokens']=cached
            result=observe('vllm',rows,True)
            self.assertTrue(result['stream_complete'])
            self.assertFalse(result['matched_input_qualified'])
            self.assertTrue(result['workload_failures'])

    def test_vllm_missing_prompt_details_are_explicit(self):
        rows=vllm_rows();rows[0]['choices'][0]['prompt_token_ids']=None
        del rows[-1]['usage']['prompt_tokens_details']
        result=observe('vllm',rows,True)
        self.assertFalse(result['matched_input_qualified'])
        self.assertIn('prompt_token_ids',result['unavailable'])
        self.assertIn('cached_prompt_tokens',result['unavailable'])

    def test_vllm_missing_finish_usage_or_done_reject(self):
        cases=[(vllm_rows(),False),(vllm_rows()[:-1],True),
               ([vllm([choice([7,8],'ab',PROMPT)]),vllm([],usage())],True)]
        for rows,done in cases:
            with self.subTest(rows=rows,done=done),self.assertRaises(ValueError):
                observe('vllm',rows,done)

    def test_vllm_counter_id_model_and_order_damage_reject(self):
        cases=[]
        for value in [True,-1,None,3]:
            rows=vllm_rows();rows[-1]['usage']['completion_tokens']=value;cases.append(rows)
        rows=vllm_rows();rows[1]['id']='other';cases.append(rows)
        rows=vllm_rows();rows[1]['model']='other';cases.append(rows)
        rows=vllm_rows();rows[0]['choices'][0]['token_ids']=None;cases.append(rows)
        rows=vllm_rows();rows[0]['choices'][0]['index']=True;cases.append(rows)
        rows=vllm_rows();rows[1]['choices'][0]['prompt_token_ids']=PROMPT;cases.append(rows)
        rows=vllm_rows();rows.append(copy.deepcopy(rows[-1]));cases.append(rows)
        rows=vllm_rows();rows.insert(2,vllm([choice([])]));cases.append(rows)
        for rows in cases:
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                observe('vllm',rows,True)

    def test_regressing_clocks_reject_before_duration_calculation(self):
        state=metrics.state('ollama')
        with self.assertRaisesRegex(ValueError,'Regressing frame clock'):
            metrics.observe_ollama([(encoded(ollama()),START+10),(encoded(ollama(True)),START+9)],
                state,START,MODEL,PROMPT,2)

    def test_nonfinite_exponents_constants_and_duplicate_fields_reject(self):
        for raw in [b'{"x":NaN}',b'{"x":Infinity}',b'{"x":1e999}',b'{"x":1,"x":2}',b'[]']:
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                metrics.object_json(raw)

    def test_sse_comments_multiline_and_crlf_are_framed_before_parsing(self):
        raw=io.StringIO();wire=b': heartbeat\r\n\r\ndata: {"x":\r\ndata: 1}\r\n\r\ndata: [DONE]\r\n\r\n'
        result=list(metrics.frames(io.BytesIO(wire),'vllm',raw,clock=lambda:START))
        self.assertEqual(result,[(b'{"x":\n1}',START),(b'[DONE]',START)])
        self.assertIn('heartbeat',raw.getvalue())
        self.assertEqual(metrics.object_json(result[0][0]),{'x':1})

    def test_truncated_and_oversized_frames_reject(self):
        for engine,wire in [('vllm',b'data: {}\n'),('ollama',b'{}'),('vllm',b'event: wrong\n\n')]:
            with self.subTest(engine=engine,wire=wire),self.assertRaises(ValueError):
                list(metrics.frames(io.BytesIO(wire),engine,io.StringIO()))
        with patch.object(metrics,'MAX_FRAME',16),self.assertRaisesRegex(ValueError,'read bound exceeded'):
            list(metrics.frames(io.BytesIO(b'x'*17+b'\n'),'ollama',io.StringIO()))
        with patch.object(metrics,'MAX_WIRE',5),self.assertRaisesRegex(ValueError,'read bound exceeded'):
            list(metrics.frames(io.BytesIO(b'{}\n{}\n'),'ollama',io.StringIO()))
        with patch.object(metrics,'MAX_FRAMES',1),self.assertRaisesRegex(ValueError,'event bound exceeded'):
            list(metrics.frames(io.BytesIO(b'{}\n{}\n'),'ollama',io.StringIO()))

    def test_frame_error_after_terminal_does_not_qualify_complete_request(self):
        observation=metrics.state('ollama')
        with self.assertRaises(ValueError):
            metrics.observe_ollama(timed([ollama(True),ollama()]),observation,START,MODEL,PROMPT,2)
        result=metrics.summarize(observation,START,PROMPT)
        self.assertFalse(result['stream_complete'])
        self.assertIsNone(result['client_metrics']['request_seconds'])
        self.assertIsNotNone(result['client_metrics']['observed_terminal_event_seconds'])

    def test_invalid_utf8_is_retained_exactly_before_json_rejection(self):
        raw=io.StringIO();wire=b'{"response":"\xff"}\n'
        framed=list(metrics.frames(io.BytesIO(wire),'ollama',raw))
        self.assertEqual(base64.b64decode(json.loads(raw.getvalue())['wire_base64']),wire)
        with self.assertRaises(UnicodeError):
            metrics.object_json(framed[0][0])
        with self.assertRaises(UnicodeError):
            observe('ollama',[ollama(True,response='\ud800')])

    def test_vllm_short_length_stop_cannot_satisfy_output_work(self):
        rows=[vllm([choice([7],'a',PROMPT,'length')]),vllm([],dict(
            prompt_tokens=2,completion_tokens=1,total_tokens=3))]
        with self.assertRaisesRegex(ValueError,'length stop'):
            observe('vllm',rows,True)

    def test_unrequested_stop_is_retained_and_fails_workload_match(self):
        rows=vllm_rows();rows[1]['choices'][0]['stop_reason']='unrequested stop string'
        result=observe('vllm',rows,True)
        self.assertTrue(result['stream_complete'])
        self.assertFalse(result['matched_input_qualified'])
        self.assertIn('Unexpected vLLM stop reason',result['workload_failures'])

    def test_rejected_line_and_overflow_prefix_keep_exact_raw_bytes(self):
        cases=[(b'{"response":"partial"}',1024,4096),
               (b'x'*50+b'\n',16,4096),(b'{}\n'*20,1024,5)]
        for wire,line_limit,total_limit in cases:
            raw=io.StringIO()
            with self.subTest(wire=wire),patch.object(metrics,'MAX_FRAME',line_limit), \
                 patch.object(metrics,'MAX_WIRE',total_limit),self.assertRaises(ValueError):
                list(metrics.frames(io.BytesIO(wire),'ollama',raw))
            captured=b''.join(base64.b64decode(json.loads(line)['wire_base64']) for line in raw.getvalue().splitlines())
            self.assertTrue(captured)
            self.assertTrue(wire.startswith(captured))
            self.assertLessEqual(len(captured),total_limit+1)
            self.assertLessEqual(len(captured),line_limit+1 if len(wire)>line_limit else len(wire))

    def test_ollama_counter_and_duration_bounds_precede_rate_calculation(self):
        for field in ['prompt_eval_count','eval_count','eval_duration','prompt_eval_duration','total_duration']:
            for value in [10**300,10**400]:
                row=ollama(True);row[field]=value
                observation=metrics.state('ollama')
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    metrics.observe_ollama(timed([row]),observation,START,MODEL,PROMPT,2)
                self.assertFalse(observation['stream_complete'])

    def test_vllm_stop_ids_require_integer_type_and_vocabulary_bounds(self):
        for stop in [float(metrics.EOS),True,-1,metrics.VOCAB,{},[]]:
            rows=vllm_rows();rows[1]['choices'][0]['stop_reason']=stop
            with self.subTest(stop=stop),self.assertRaises(ValueError):
                observe('vllm',rows,True)


def workload():
    return dict(format='toklet-lfm3-external-workloads-v1',prompt_tokens=512,
        maximum_sampled_output_tokens=32,raw_text=True,chat_formatting=False,
        rows=[dict(trial=i,prompt='public fixture',prompt_token_ids=list(range(512))) for i in range(13)],
        ollama_request=dict(raw=True,stream=True,options=dict(num_predict=32,num_ctx=2048,
            temperature=0,repeat_penalty=1,repeat_last_n=0,presence_penalty=0,frequency_penalty=0,
            top_k=0,top_p=1,min_p=0,seed=0)),
        vllm_request=dict(stream=True,max_tokens=32,temperature=0,repetition_penalty=1,
            presence_penalty=0,frequency_penalty=0,top_p=1,top_k=-1,min_p=0,seed=0,
            stop_token_ids=[metrics.EOS],return_token_ids=True,skip_special_tokens=True,
            stream_options=dict(include_usage=True)))


class RequestTests(unittest.TestCase):
    def test_raw_tokenization_and_residency_overrides_preserve_manifest(self):
        source=workload();before=copy.deepcopy(source)
        body,ids=metrics.request_body('vllm',source,0,MODEL)
        self.assertIs(body['add_special_tokens'],False)
        self.assertEqual(len(ids),512)
        body,_=metrics.request_body('ollama',source,0,MODEL)
        self.assertEqual(body['keep_alive'],'10m')
        self.assertEqual(source,before)

    def test_sampling_and_workload_mutations_reject(self):
        for engine in ['vllm','ollama']:
            source=workload();source[engine+'_request']['extra_stop']='changed'
            with self.subTest(engine=engine),self.assertRaisesRegex(ValueError,'sampling policy differs'):
                metrics.request_body(engine,source,0,MODEL)
        for source in [dict(workload(),raw_text=False),dict(workload(),prompt_tokens=513)]:
            with self.assertRaisesRegex(ValueError,'geometry differs'):
                metrics.request_body('vllm',source,0,MODEL)

    def test_failed_transport_preserves_partial_observations_and_raw_evidence(self):
        class Response(io.BytesIO):
            status=200
            def getheader(self,key):return 'application/x-ndjson'
        class Connection:
            closed=False
            def request(self,*args,**kwargs):pass
            def getresponse(self):return Response(encoded(ollama(response='partial'))+b'\n')
            def close(self):self.closed=True
        connection=Connection()
        with tempfile.TemporaryDirectory() as folder,patch.object(metrics.http.client,'HTTPConnection',return_value=connection):
            output=Path(folder)/'out'
            result=metrics.capture('ollama',12345,{'model':MODEL},PROMPT,output)
            self.assertFalse(result['completed'])
            self.assertEqual(result['failure']['phase'],'stream')
            self.assertEqual(result['observation']['text'],'partial')
            self.assertIsNone(result['observation']['client_metrics']['request_seconds'])
            self.assertFalse(result['observation']['matched_input_qualified'])
            self.assertTrue((output/'wire.jsonl').read_text())
            self.assertEqual(json.loads((output/'report.json').read_bytes()),result)
            self.assertTrue(connection.closed)

    def test_extreme_counter_failure_report_is_valid_json_and_incomplete(self):
        for value in [10**300,10**400]:
            wire=encoded(ollama(True,prompt_eval_count=value))+b'\n'
            class Response(io.BytesIO):
                status=200
                def getheader(self,key):return 'application/x-ndjson'
            class Connection:
                def request(self,*args,**kwargs):pass
                def getresponse(self):return Response(wire)
                def close(self):pass
            with self.subTest(value=value),tempfile.TemporaryDirectory() as folder, \
                 patch.object(metrics.http.client,'HTTPConnection',return_value=Connection()):
                output=Path(folder)/'out'
                result=metrics.capture('ollama',12345,{'model':MODEL},PROMPT,output)
                self.assertEqual(json.loads((output/'report.json').read_bytes()),result)
                self.assertFalse(result['completed'])
                self.assertFalse(result['observation']['stream_complete'])
                self.assertIsNone(result['observation']['client_metrics']['request_seconds'])

    def test_http_error_body_keeps_exact_binary_prefix(self):
        wire=b'\xff\xfe binary error body'
        class Response(io.BytesIO):
            status=422
        class Connection:
            def request(self,*args,**kwargs):pass
            def getresponse(self):return Response(wire)
            def close(self):pass
        with tempfile.TemporaryDirectory() as folder,patch.object(metrics.http.client,'HTTPConnection',return_value=Connection()):
            result=metrics.capture('ollama',12345,{'model':MODEL},PROMPT,Path(folder)/'out')
            self.assertFalse(result['completed'])
            self.assertEqual(base64.b64decode(result['http_error_body_base64']),wire)
            self.assertEqual(result['http_error_body_bytes_captured'],len(wire))

    def test_invalid_request_json_writes_a_valid_failure_report(self):
        class Connection:
            def close(self):pass
        with tempfile.TemporaryDirectory() as folder,patch.object(metrics.http.client,'HTTPConnection',return_value=Connection()):
            output=Path(folder)/'out'
            result=metrics.capture('ollama',12345,{'model':MODEL,'bad':float('nan')},PROMPT,output)
            self.assertFalse(result['completed'])
            self.assertIsNone(result['request'])
            self.assertEqual(result['failure']['phase'],'send')
            self.assertEqual(json.loads((output/'report.json').read_bytes()),result)


if __name__ == '__main__':
    unittest.main()
