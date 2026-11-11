"""Bounded client observations for explicit LFM3 Ollama/vLLM benchmark requests.

The caller owns the server, process-memory guard, power gates and wall timeout.
HTTP read timeout is an inactivity timeout. No engine or display timing is
inferred from client frames, and a frame is never counted as one sampled token.
Stream/input checks do not qualify model quality, weights, memory or performance.
"""
import argparse
import base64
import hashlib
import http.client
import json
import math
from pathlib import Path
import time


MAX_FRAME = 1024 * 1024
MAX_WIRE = 8 * 1024 * 1024
MAX_FRAMES = 4096
MAX_WORKLOAD = 1024 * 1024
VOCAB = 128000
EOS = 124900
MAX_CONTEXT = 2048
MAX_U64 = (1 << 64) - 1


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, name, maximum=MAX_U64):
    require(type(value) is int and 0 <= value <= maximum, 'Invalid bounded nonnegative integer: '+name)
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON field: '+key)
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError('Nonfinite JSON number: '+value)


def finite_float(value):
    number = float(value)
    require(math.isfinite(number), 'Nonfinite JSON number: '+value)
    return number


def object_json(data):
    if isinstance(data,bytes):
        data = data.decode('utf-8')
    value = json.loads(data, object_pairs_hook=unique_object,
                       parse_constant=invalid_constant, parse_float=finite_float)
    require(type(value) is dict, 'Expected a JSON object')
    return value


def frames(response, engine, raw, clock=None):
    """Yield (complete frame bytes, monotonic arrival ns), before JSON parsing.

    Raw evidence includes framing lines and comments. The byte, line, frame and
    event bounds apply to actual reads, independently of Content-Length.
    """
    require(engine in ('ollama', 'vllm'), 'Unknown engine')
    clock = clock or time.monotonic_ns
    total = lines = events = size = 0
    parts = []
    while True:
        # One extra byte detects overflow while retaining the rejected prefix.
        line = response.readline(min(MAX_FRAME + 1, MAX_WIRE - total + 1))
        if not line:
            require(not parts, 'Truncated SSE event')
            return
        observed = clock()
        total += len(line)
        lines += 1
        raw.write(json.dumps(dict(received_monotonic_ns=observed,
                    line_complete=line.endswith(b'\n'),
                    wire_line=line.decode('utf-8',errors='replace'),
                    wire_base64=base64.b64encode(line).decode('ascii')),
                    ensure_ascii=True, allow_nan=False)+'\n')
        raw.flush()
        require(len(line) <= MAX_FRAME and total <= MAX_WIRE and lines <= MAX_FRAMES * 8,
                'Stream read bound exceeded')
        require(line.endswith(b'\n'), 'Truncated stream line')
        if engine == 'ollama':
            if not line.strip():
                continue
            data = line.rstrip(b'\r\n')
        else:
            line = line.rstrip(b'\r\n')
            if line.startswith(b':'):
                continue
            if line:
                require(line.startswith(b'data:'), 'Unexpected SSE field')
                part = line[5:]
                if part.startswith(b' '):
                    part = part[1:]
                size += len(part) + 1
                require(size <= MAX_FRAME, 'SSE frame bound exceeded')
                parts.append(part)
                continue
            if not parts:
                continue
            data = b'\n'.join(parts)
            parts, size = [], 0
        events += 1
        require(events <= MAX_FRAMES, 'Stream event bound exceeded')
        yield data, observed


def state(engine):
    return dict(engine=engine, stream_complete=False, frames=0,
        first_generated_text_ns=None, first_content_ns=None, first_token_batch_ns=None,
        terminal_ns=None, text='', thinking='', output_token_ids=None,
        prompt_token_ids=None, cached_prompt_tokens=None, reported_prompt_tokens=None,
        reported_output_tokens=None, engine_metrics={}, finish_reason=None,
        stop_reason=None, workload_failures=[], unavailable={})


def arrival(observation, timestamp, started):
    integer(timestamp, 'arrival clock')
    require(timestamp >= observation.get('last_frame_ns', started), 'Regressing frame clock')
    observation['last_frame_ns'] = timestamp
    observation['frames'] += 1


def append_text(observation, field, text, timestamp):
    require(type(text) is str, 'Invalid generated text')
    text.encode('utf-8')  # Reject unpaired JSON surrogates before retaining text.
    observation[field] += text
    if text:
        if observation['first_generated_text_ns'] is None:
            observation['first_generated_text_ns'] = timestamp
        if field == 'text' and observation['engine'] == 'ollama' and observation['first_content_ns'] is None:
            observation['first_content_ns'] = timestamp


def observe_ollama(events, observation, started, model, prompt_ids, cap):
    for data, timestamp in events:
        arrival(observation, timestamp, started)
        require(observation['terminal_ns'] is None, 'Data after Ollama terminal event')
        row = object_json(data)
        require('error' not in row, 'Ollama stream error: '+str(row.get('error')))
        require(row.get('model') == model and type(row.get('done')) is bool,
                'Ollama model or done field differs')
        append_text(observation, 'thinking', row.get('thinking', ''), timestamp)
        append_text(observation, 'text', row.get('response', ''), timestamp)
        if not row['done']:
            continue
        for key in ('total_duration','load_duration','prompt_eval_duration','eval_duration'):
            observation['engine_metrics'][key+'_ns'] = integer(row.get(key), key)
        observation['reported_prompt_tokens'] = integer(row.get('prompt_eval_count'), 'prompt_eval_count', MAX_CONTEXT)
        count = integer(row.get('eval_count'), 'eval_count', cap)
        observation['reported_output_tokens'] = count
        if 'prompt_eval_cached_count' in row:
            cached = integer(row['prompt_eval_cached_count'], 'prompt_eval_cached_count')
            require(cached <= observation['reported_prompt_tokens'], 'Invalid cached prompt count')
            observation['cached_prompt_tokens'] = cached
        reason = row.get('done_reason')
        require(reason in ('stop','length'), 'Unsupported Ollama finish reason')
        observation.update(terminal_ns=timestamp, finish_reason=reason)
    require(observation['terminal_ns'] is not None, 'Ollama stream ended without a terminal event')
    observation['unavailable'].update(
        prompt_token_ids='Generate API does not expose authoritative prompt IDs',
        output_token_ids='Text chunks and eval_count do not expose sampled IDs or hidden stops',
        first_token='No authoritative per-token ID event in the generate stream',
        pending_and_committed_tokens='Generate API has no KV publication journal')
    duration = observation['engine_metrics']['eval_duration_ns']
    observation['engine_metrics']['api_eval_tokens_per_second'] = (
        count * 1e9 / duration if duration else None)
    if not duration:
        observation['unavailable']['api_eval_tokens_per_second'] = 'API eval_duration is zero; rate is undefined'
    cached = observation['cached_prompt_tokens']
    duration = observation['engine_metrics']['prompt_eval_duration_ns']
    observation['engine_metrics']['api_uncached_prefill_tokens_per_second'] = (
        (observation['reported_prompt_tokens']-cached)*1e9/duration
        if cached is not None and duration else None)
    if cached is None or not duration:
        observation['unavailable']['api_uncached_prefill_tokens_per_second'] = (
            'API cached count is absent or prompt_eval_duration is zero')
    observation['stream_complete'] = True


def token_ids(value, name):
    require(type(value) is list and len(value) <= 2048
            and all(type(v) is int and 0 <= v < VOCAB for v in value), 'Invalid '+name)
    return value


def observe_vllm(events, observation, started, model, prompt_ids, cap):
    observation['output_token_ids'] = []
    finished = usage = done = False
    request_id = None
    for data, timestamp in events:
        arrival(observation, timestamp, started)
        require(not done, 'Data after vLLM DONE')
        if data == b'[DONE]':
            require(finished and usage, 'vLLM DONE precedes finish or usage')
            done = True
            observation['terminal_ns'] = timestamp
            continue
        row = object_json(data)
        require('error' not in row, 'vLLM stream error: '+str(row.get('error')))
        require(row.get('model') == model and row.get('object') == 'text_completion'
                and type(row.get('id')) is str and row['id'], 'vLLM model/response identity differs')
        if request_id is None:
            request_id = row['id']
        require(row['id'] == request_id, 'vLLM request identity changed')
        choices = row.get('choices')
        require(type(choices) is list and len(choices) <= 1, 'Unexpected vLLM choices')
        if row.get('usage') is not None:
            require(not choices and finished and not usage, 'Duplicate or early vLLM usage')
            info = row['usage']
            require(type(info) is dict, 'Invalid vLLM usage')
            prompt = integer(info.get('prompt_tokens'), 'prompt_tokens', MAX_CONTEXT)
            output = integer(info.get('completion_tokens'), 'completion_tokens', cap)
            require(integer(info.get('total_tokens'), 'total_tokens', MAX_CONTEXT+cap) == prompt+output,
                    'vLLM total token count differs')
            require(output == len(observation['output_token_ids']) <= cap,
                    'vLLM output IDs and usage differ')
            require(observation['finish_reason'] != 'length' or output == cap,
                    'vLLM length stop does not match the admitted cap')
            details = info.get('prompt_tokens_details')
            if details is not None:
                require(type(details) is dict, 'Invalid prompt token details')
                cached = integer(details.get('cached_tokens'), 'cached_tokens')
                require(cached <= prompt, 'Invalid cached prompt count')
                observation['cached_prompt_tokens'] = cached
            observation.update(reported_prompt_tokens=prompt, reported_output_tokens=output)
            usage = True
        if not choices:
            continue
        require(not finished and not usage, 'vLLM choice follows terminal choice')
        choice = choices[0]
        require(type(choice) is dict and type(choice.get('index')) is int and choice['index'] == 0,
                'Invalid vLLM choice index')
        if choice.get('prompt_token_ids') is not None:
            require(observation['prompt_token_ids'] is None, 'Duplicate vLLM prompt IDs')
            observation['prompt_token_ids'] = token_ids(choice['prompt_token_ids'], 'prompt IDs')
        ids = token_ids(choice.get('token_ids'), 'requested output IDs')
        observation['output_token_ids'].extend(ids)
        require(len(observation['output_token_ids']) <= cap, 'vLLM output exceeds the admitted cap')
        if ids and observation['first_token_batch_ns'] is None:
            observation['first_token_batch_ns'] = timestamp
        append_text(observation, 'text', choice.get('text'), timestamp)
        reason = choice.get('finish_reason')
        stop = choice.get('stop_reason')
        require(stop is None or type(stop) in (int,str), 'Invalid vLLM stop reason type')
        if type(stop) is int:
            integer(stop, 'vLLM stop token ID', VOCAB-1)
        if reason is not None:
            require(reason in ('stop','length'), 'Unsupported vLLM finish reason')
            observation.update(finish_reason=reason, stop_reason=stop)
            finished = True
    require(done, 'vLLM stream ended without DONE')
    observation['stream_complete'] = True
    observation['unavailable'].update(
        content_only='Completion text is raw generated text; no qualified reasoning/content parser',
        engine_phase_timings='SSE completion events have no completed per-request engine phase clocks',
        per_token_decode_rate='Token IDs can arrive in batches; no per-token engine clock is exposed',
        pending_and_committed_tokens='Completion API has no KV publication journal')


def summarize(observation, started, prompt_ids):
    end = observation['terminal_ns']
    observation['client_metrics'] = {}
    for key, value in [('request_seconds', end if observation['stream_complete'] else None),
                       ('observed_terminal_event_seconds', end),
                       ('first_generated_text_seconds', observation['first_generated_text_ns']),
                       ('first_content_seconds', observation['first_content_ns']),
                       ('first_token_batch_seconds', observation['first_token_batch_ns'])]:
        observation['client_metrics'][key] = (value-started)/1e9 if value is not None else None
        if value is None:
            observation['unavailable'][key] = (
                'Stream did not complete validation' if key == 'request_seconds'
                else 'No corresponding qualified client event was observed')
    for key, message in [('reported_prompt_tokens','Prompt token count differs'),
                         ('cached_prompt_tokens','Zero prompt reuse is not established')]:
        expected = len(prompt_ids) if key == 'reported_prompt_tokens' else 0
        if observation[key] != expected:
            observation['workload_failures'].append(message)
        if observation[key] is None:
            observation['unavailable'][key] = 'Field absent from terminal API observations'
    observed_ids = observation['prompt_token_ids']
    if observation['engine'] == 'vllm' and observation['stop_reason'] not in (None,EOS):
        observation['workload_failures'].append('Unexpected vLLM stop reason')
    if observed_ids is None:
        observation['unavailable'].setdefault('prompt_token_ids', 'Requested prompt IDs were not returned')
    elif observed_ids != prompt_ids:
        observation['workload_failures'].append('Prompt IDs differ from the frozen native inputs')
    observation['count_and_cache_match'] = not observation['workload_failures']
    observation['exact_prompt_ids_match'] = observed_ids == prompt_ids
    observation['matched_input_qualified'] = (observation['stream_complete']
        and observation['count_and_cache_match'] and observation['exact_prompt_ids_match'])
    observation['text_sha256'] = hashlib.sha256(observation['text'].encode()).hexdigest()
    observation['thinking_sha256'] = hashlib.sha256(observation['thinking'].encode()).hexdigest()
    observation['clock_scope'] = 'Client monotonic timestamp after the delimiter line read and before JSON parsing; logging of earlier lines is included. No engine or display clock.'
    observation['engine_metrics_scope'] = 'Independent API counters/durations; no inferred clock enclosure, sampled/committed state or cross-engine denominator conversion'
    return observation


def request_body(engine, workload, trial, model):
    require(engine in ('ollama','vllm') and type(trial) is int and 0 <= trial < 13, 'Invalid engine or trial')
    require(type(model) is str and 0 < len(model) <= 256, 'Invalid served model name')
    require(workload.get('format') == 'toklet-lfm3-external-workloads-v1'
            and workload.get('prompt_tokens') == 512 and workload.get('maximum_sampled_output_tokens') == 32
            and workload.get('raw_text') is True and workload.get('chat_formatting') is False,
            'Frozen workload geometry differs')
    rows = workload['rows']
    require(len(rows) == 13 and rows[trial]['trial'] == trial, 'Workload trial differs')
    row = rows[trial]
    ids = token_ids(row['prompt_token_ids'], 'frozen prompt IDs')
    require(len(ids) == 512 and type(row['prompt']) is str and len(row['prompt'].encode()) <= 256*1024,
            'Frozen raw input differs')
    # Preserve the frozen numerical policy. These two additions concern raw
    # frontend tokenization and loaded-model residency, rather than sampling.
    body = json.loads(json.dumps(workload[engine+'_request'], allow_nan=False))
    require(body.get('stream') is True, 'Streaming observation is required')
    if engine == 'vllm':
        require(set(body) == {'stream','max_tokens','temperature','repetition_penalty',
                    'presence_penalty','frequency_penalty','top_p','top_k','min_p','seed',
                    'stop_token_ids','return_token_ids','skip_special_tokens','stream_options'}
                and body.get('max_tokens') == 32 and body.get('temperature') == 0
                and body.get('repetition_penalty') == 1 and body.get('presence_penalty') == 0
                and body.get('frequency_penalty') == 0 and body.get('top_p') == 1
                and body.get('top_k') == -1 and body.get('min_p') == body.get('seed') == 0
                and body.get('skip_special_tokens') is True
                and body.get('stop_token_ids') == [EOS] and body.get('return_token_ids') is True
                and body.get('stream_options') == {'include_usage':True}, 'vLLM sampling policy differs')
        body['add_special_tokens'] = False
    else:
        options = body.get('options', {})
        require(set(body) == {'raw','stream','options'}
                and set(options) == {'num_predict','num_ctx','temperature','repeat_penalty',
                    'repeat_last_n','presence_penalty','frequency_penalty','top_k','top_p','min_p','seed'}
                and body.get('raw') is True and options.get('num_predict') == 32
                and options.get('num_ctx') == 2048 and options.get('temperature') == 0
                and options.get('repeat_penalty') == 1 and options.get('repeat_last_n') == 0
                and options.get('presence_penalty') == options.get('frequency_penalty') == 0
                and options.get('top_k') == options.get('min_p') == options.get('seed') == 0
                and options.get('top_p') == 1, 'Ollama sampling policy differs')
        body['keep_alive'] = '10m'
    body.update(model=model, prompt=row['prompt'])
    return body, ids


def capture(engine, port, body, prompt_ids, output, timeout=1800, provenance=None):
    require(engine in ('ollama','vllm') and type(port) is int and 0 < port < 65536,
            'Invalid loopback engine endpoint')
    require(type(timeout) in (int,float) and math.isfinite(timeout) and timeout > 0, 'Invalid read timeout')
    output.mkdir()
    report = dict(format='toklet-lfm3-http-metrics-v1', completed=False, failure=None,
                  endpoint=dict(host='127.0.0.1',port=port), request=None,
                  socket_timeout_seconds=timeout, wall_timeout_owner='External server/process guard',
                  provenance=provenance, tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  server_weight_identity_verified=False, model_quality_qualified=False,
                  performance_qualified=False)
    observation = state(engine)
    report['observation'] = observation
    connection = http.client.HTTPConnection('127.0.0.1',port,timeout=timeout)
    started = None
    phase = 'send'
    with (output/'wire.jsonl').open('x') as raw:
        try:
            started = time.monotonic_ns()
            require(type(body) is dict and type(body.get('model')) is str, 'Invalid request model')
            payload = json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
            require(len(payload) <= MAX_WORKLOAD, 'Request body bound exceeded')
            report['request'] = json.loads(payload)
            path = '/api/generate' if engine == 'ollama' else '/v1/completions'
            connection.request('POST',path,body=payload,headers={'Content-Type':'application/json'})
            phase = 'headers'
            response = connection.getresponse()
            report['http_status'] = response.status
            if response.status != 200:
                data = response.read(MAX_FRAME+1)
                report['http_error_body'] = data[:MAX_FRAME].decode('utf-8',errors='replace')
                report['http_error_body_truncated'] = len(data) > MAX_FRAME
                report['http_error_body_base64'] = base64.b64encode(data).decode('ascii')
                report['http_error_body_bytes_captured'] = len(data)
            require(response.status == 200, 'HTTP request failed: '+str(response.status))
            report['response_content_type'] = response.getheader('Content-Type')
            phase = 'stream'
            observer = observe_ollama if engine == 'ollama' else observe_vllm
            observer(frames(response,engine,raw),observation,started,body['model'],prompt_ids,32)
            report['completed'] = True
        except Exception as error:
            observation['stream_complete'] = False
            report['failure'] = dict(phase=phase,error=repr(error))
        finally:
            connection.close()
    if started is not None:
        summarize(observation,started,prompt_ids)
    report['wire_sha256'] = hashlib.sha256((output/'wire.jsonl').read_bytes()).hexdigest()
    with (output/'report.json').open('x') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=True,allow_nan=False)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',choices=('ollama','vllm'),required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--workloads',type=Path,required=True)
    parser.add_argument('--workloads-sha256',required=True)
    parser.add_argument('--trial',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    with args.workloads.open('rb') as stream:
        data = stream.read(MAX_WORKLOAD+1)
    require(len(data) <= MAX_WORKLOAD and hashlib.sha256(data).hexdigest() == args.workloads_sha256,
            'Frozen workload size or hash differs')
    workload = object_json(data)
    body, ids = request_body(args.engine,workload,args.trial,args.model)
    provenance = dict(workload_sha256=args.workloads_sha256,trial=args.trial,
        expected_candidate_from_workload=workload.get('candidate'),
        identity_authentication_owner='External launcher/import receipt; server model tag alone is not weight authentication',
        effective_frontend_additions={'add_special_tokens':False} if args.engine=='vllm' else {'keep_alive':'10m'})
    report = capture(args.engine,args.port,body,ids,args.output,provenance=provenance)
    print(json.dumps(dict(completed=report['completed'],failure=report['failure'],
        matched_input_qualified=report['observation']['matched_input_qualified'])))
    return 0 if report['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
