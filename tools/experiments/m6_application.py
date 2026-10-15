"""Actual tokenizer-owning Online CLI at the frozen maximum context boundary.

One bounded generation is application memory/state qualification, not repeated
application latency or natural long-document task ability.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import m6_tasks as producer

context = producer.context
ROOT = producer.ROOT


class Worker(context.Worker):
    def __enter__(self):
        try:
            self.check()
            self.child = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            self.transport = context.shared.metal_engine.Transport(self.child, self.raw, self.stderr, stdout_limit=4*1024**2, event_limit=4096, request_timeout=7200, lifetime_timeout=86400)
            self.admission = self.transport.read(); self.record['ready'] = self.admission
            memory = producer.ready(self.admission, self.model, self.capacity)
            layers, dimension = (28, 128) if self.model == 'larger' else (24, 64)
            if memory['kv_bytes'] != (2*self.capacity*dimension+7)*4*layers or memory['history_bytes'] != self.capacity*4 or memory['logits_bytes'] != 151936*8:
                raise ValueError('Actual application capacity storage differs')
            self.usage('ready')
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise


def response(events, reset, initial, prompt, cap):
    """No EOS exclusion: --no-eos intentionally publishes all sampled IDs."""
    if not events or events[-1].get('event') != 'done' or len(events) != cap+1:
        raise ValueError('Full boundary output coverage differs')
    ids, texts, times = [], [], []
    for i, event in enumerate(events[:-1]):
        context.exact(event.get('event'), 'token', 'token event')
        context.exact(event.get('index'), i, 'token index')
        token = event.get('token_id')
        if type(token) is not int or not 0 <= token < 151936 or type(event.get('text')) is not str:
            raise ValueError('Invalid boundary token/text')
        ids.append(token); texts.append(event['text'])
        times.append(context.finite(event.get('elapsed_seconds'), 'boundary token clock'))
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError('Boundary token clock order differs')
    if len(reset) not in (1, 2) or reset[-1].get('event') != 'reset':
        raise ValueError('Boundary text reset did not complete')
    for event in reset[:-1]:
        if event.get('event') != 'text' or type(event.get('text')) is not str:
            raise ValueError('Boundary final text differs')
        texts.append(event['text'])
    visible = ''.join(texts)
    if len(visible.encode()) > producer.shared.MAX_RESPONSE_BYTES:
        raise ValueError('Boundary response text exceeds bound')
    generation = producer.shared.cli.stats(events[-1], initial)
    expected = dict(reused_tokens=0, prefill_tokens=len(prompt), sampled_tokens=cap, emitted_tokens=cap,
                    history_tokens=len(prompt)+cap, evaluated_tokens=len(prompt)+cap-1, pending_tokens=1)
    for key, value in expected.items():
        context.exact(generation.get(key), value, key)
    context.exact(generation.get('reason'), {'kind': 'output_limit'}, 'boundary stop')
    for key in ('prefill_seconds', 'decode_seconds', 'sampling_seconds', 'total_seconds'):
        context.finite(generation.get(key), key)
        if generation[key] > generation['total_seconds']:
            raise ValueError('Generation component outside completed request')
    if generation.get('first_token_seconds') != times[0] or times[-1] > generation['total_seconds']:
        raise ValueError('Boundary completed token clocks differ')
    stats = events[-1]['statistics']
    for key in ('input_preparation_seconds', 'input_formatting_seconds', 'tokenization_seconds',
                'detokenization_seconds', 'publication_seconds', 'load_seconds', 'conversion_seconds', 'tokenizer_load_seconds'):
        context.finite(stats.get(key), key)
    if stats['input_formatting_seconds'] != 0 or stats['tokenization_seconds'] != 0:
        raise ValueError('Fixed raw IDs unexpectedly formatted or tokenized')
    return dict(visible_token_ids=ids, visible_text=visible, visible_text_sha256=hashlib.sha256(visible.encode()).hexdigest(),
                statistics=stats, history_sha256=context.shared.history_sha(prompt+ids),
                history_hash_scope='Derived from admitted prompt and published IDs; CLI exposes counts, not an independent history hash')


def run(output, model, check=False):
    paths = [output, output.with_suffix('.raw.jsonl'), output.with_suffix('.stderr')]
    if any(p.exists() for p in paths):
        raise ValueError('Exclusive max-context application evidence exists')
    binary, source, packed, identities = producer.provenance(model)
    _, arrays, _, _, _ = context.inputs(model)
    ids = arrays[32768]; prompt, cap = (ids[:4], 2) if check else (ids[:-128], 128)
    identities[str(Path(__file__).relative_to(ROOT))] = context.shared.metal_engine.digest(Path(__file__))
    projection, extra = producer.arguments(model, packed)
    command = [str(binary), '--model', str(source), '--backend', 'metal', '--metal-projection', projection,
               '--interactive', '--json', '--context-capacity', '32768', '--prefill-chunk', '128', '--no-eos', *extra]
    result = dict(format='toklet-m6-max-context-application-v1', passed=False, collection_completed=False,
                  check_only=check, model=model, context_capacity=32768, prompt_tokens=len(prompt), output_tokens=cap,
                  identities=identities, revision=context.shared.metal_continue.observation(['git', 'rev-parse', 'HEAD']),
                  power_before=context.shared.metal_continue.endpoints(),
                  scope='Actual tokenizer-owning CLI memory/state/output boundary; one run, no latency ranking or natural long-document ability',
                  energy_joules=None, continuous_memory_pressure=None)
    with paths[0].open('x') as report, paths[1].open('x') as raw, paths[2].open('xb') as stderr:
        worker = Worker(binary, source, packed, 32768, 128, False, model, raw, stderr, lambda: producer.check_identities(identities))
        worker.command = command; worker.record['command'] = command
        result['worker'] = worker.record
        try:
            with worker:
                clock = time.monotonic()
                events = worker.call(dict(action='generate', tokens=prompt, max_new_tokens=cap), 'done')
                result['driver_submission_through_done_validation_seconds'] = time.monotonic()-clock
                reset = worker.call(dict(action='reset'), 'reset')
                result['response'] = response(events, reset, worker.admission, prompt, cap)
            producer.shared.finalized([result['response']], source/'tokenizer.json', model_exited=True)
            usage = producer.shared.native_usage.usage()
            charge = usage.get('lifetime_max_phys_footprint')
            if type(charge) is not int or not 0 < charge <= producer.shared.LIMIT:
                raise ValueError('Independent post-exit decoder native charge exceeds6GiB')
            result['decoder_phase'] = dict(after_owned_exit=True, admitted_bytes=247772624, driver=usage)
            result.update(passed=worker.record['passed'], collection_completed=worker.record['passed'] and not check,
                          independent_text_after_owned_exit=True)
        except BaseException as error:
            result.update(passed=False, error=f'{type(error).__name__}: {error}')
        finally:
            result['endpoint_errors'] = []
            try:
                producer.check_identities(identities); result['identities_unchanged'] = True
            except BaseException as error:
                result['identities_unchanged'] = False; result['endpoint_errors'].append(str(error))
            try:
                result['power_after'] = context.shared.metal_continue.endpoints()
            except BaseException as error:
                result['endpoint_errors'].append(str(error))
            if not result['identities_unchanged'] or result['endpoint_errors']:
                result.update(passed=False, collection_completed=False)
            json.dump(result, report, indent=2, allow_nan=False); report.write('\n')
    if not result['passed']:
        raise ValueError('Max-context CLI qualification failed; see retained evidence')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', choices=('small', 'larger'), required=True)
    p.add_argument('--check', action='store_true')
    a = p.parse_args(); run(a.output.resolve(), a.model, a.check)
