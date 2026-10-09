#!/usr/bin/env python3
"""Explicit short Qwen CLI qualification; no performance or quality ranking."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
sources = json.loads((root/'benchmarks/sources.json').read_text())
for item in sources['assets']:
    if item['id'].startswith('qwen.'):
        path = root/item['path']
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024*1024), b''):
                digest.update(chunk)
        assert path.stat().st_size == item['bytes'] and digest.hexdigest() == item['sha256']
manifest = json.loads((root/'tests/fixtures/qwen-v2/manifest.json').read_text())
command = [str(root/'target/release/toklet'), '--model',
           str(root/'models/Qwen2.5-0.5B-Instruct'), '--interactive',
           '--max-new-tokens', '4', '--prefill-chunk', '8', '--no-eos']
p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True)
# Bound an unexpected hang and always release this script's own child.
watchdog = threading.Timer(300, p.kill)
watchdog.start()
try:
    records=[]
    def read():
        line=p.stdout.readline()
        if not line:raise RuntimeError(p.stderr.read())
        value=json.loads(line);records.append(value);return value
    ready=read();assert ready['event']=='ready'
    def send(request,cancel=False):
        p.stdin.write(json.dumps(request,ensure_ascii=False)+'\n');p.stdin.flush()
        if cancel:time.sleep(.5);os.kill(p.pid,signal.SIGINT)
        events=[]
        while True:
            event=read();events.append(event)
            if event['event'] in ['done','error','reset']:return events
    checks=[]
    request={'tokens':[9707,11,358,2776],'max_new_tokens':4}
    events=send(request);ids=[e['token_id'] for e in events if e['event']=='token'];assert len(ids)==4 and ids[0]==manifest['argmax']['short'][-1]
    checks.append({'case':'raw_greedy','ids':ids,'statistics':events[-1]['statistics']})
    events=send({'action':'continue','max_new_tokens':2});more=[e['token_id'] for e in events if e['event']=='token'];assert len(more)==2
    r=events[-1]['statistics']['generation'];assert r['prefill_tokens']==1 and r['reused_tokens']==7
    checks.append({'case':'pending_continue','ids':more,'statistics':events[-1]['statistics']})
    history=request['tokens']+ids+more
    events=send({'tokens':history,'max_new_tokens':0});assert events[-1]['statistics']['generation']['prefill_tokens']==1
    checks.append({'case':'pending_full_prompt','statistics':events[-1]['statistics']})
    events=send({'tokens':history,'max_new_tokens':0});assert events[-1]['statistics']['generation']['prefill_tokens']==0 and events[-1]['statistics']['generation']['reused_tokens']==10
    checks.append({'case':'fully_cached_prompt','statistics':events[-1]['statistics']})
    events=send({'action':'append','prompt':'\n<|im_end|>\n<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n','max_new_tokens':1});assert events[-1]['event']=='done' and events[-1]['statistics']['generation']['reused_tokens']==10
    checks.append({'case':'raw_user_suffix','statistics':events[-1]['statistics']})
    events=send({'prompt':'Answer only with the digit: 2+2 =','max_new_tokens':2});assert events[-1]['event']=='done' and events[-1]['statistics']['generation']['reused_tokens']==0
    checks.append({'case':'chat_prompt_reset','text':''.join(e.get('text','') for e in events),'statistics':events[-1]['statistics']})
    assert send({'action':'reset'})[-1]['event']=='reset'
    events=send({'tokens':[9707,11,358,2776]*4,'max_new_tokens':1},cancel=True);r=events[-1]['statistics']['generation'];assert r['reason']['kind']=='cancelled' and r['sampled_tokens']==0 and r['evaluated_tokens']<16
    checks.append({'case':'prefill_ctrl_c','statistics':events[-1]['statistics']})
    events=send({'action':'continue','max_new_tokens':0});r=events[-1]['statistics']['generation'];assert r['evaluated_tokens']==16 and r['pending_tokens']==0
    checks.append({'case':'prefill_resume','statistics':events[-1]['statistics']})
    events=send({'tokens':[9707]*1921,'max_new_tokens':128});assert events[-1]['event']=='error'
    checks.append({'case':'over_capacity_rejected','error':events[-1]['message']})
    events=send({'action':'continue','max_new_tokens':0});assert events[-1]['statistics']['generation']['reused_tokens']==16 and events[-1]['statistics']['generation']['prefill_tokens']==0
    checks.append({'case':'after_rejection_prefix_retained','statistics':events[-1]['statistics']})
    peakpath=output/f'footprint-{p.pid}.json'
    subprocess.run(['/usr/bin/footprint','-p',str(p.pid),'-f','bytes','--noCategories','-j',str(peakpath)],check=True,capture_output=True)
    foot=json.loads(peakpath.read_text());peakpath.unlink();peak=next(x for x in foot['processes'] if x['pid']==p.pid)['auxiliary']['phys_footprint_peak'];assert peak<6*1024**3
    p.stdin.close();assert p.wait(timeout=10)==0
    report={'model':manifest['model'],'command':command,'cases':checks,'memory_admitted_bytes':ready['memory_admitted_bytes'],'physical_process_peak_bytes':peak,'scope':'functional CLI/session qualification; short real-model generation and cancellation; no benchmark or held-out quality claim'}
    report['driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['binary_sha256'] = hashlib.sha256((root/'target/release/toklet').read_bytes()).hexdigest()
    report['engine_base_revision'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    report['uncommitted_diff_sha256'] = hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=root)).hexdigest()
    (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (output/'events.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'cases':len(checks),'first_ids':ids,'peak':peak,
                     'admitted':ready['memory_admitted_bytes'],
                     'chat_text':next(c['text'] for c in checks if c['case']=='chat_prompt_reset')},ensure_ascii=False))
finally:
    watchdog.cancel()
    if p.poll() is None:
        p.kill()
    p.wait(timeout=10)
    for stream in (p.stdin,p.stdout,p.stderr):
        stream.close()
