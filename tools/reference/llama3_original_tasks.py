"""Cached original task collection plus mandatory stateless trajectory replay.

Only numerical inference runs here. Decoding and isolated grading happen after
both model workers exit. Hidden stops are sampled but never visible or cached.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import llama3_original_long as runtime  # Pin numerical threads before NumPy.
import llama3_original_cache as weights
import llama3_original_session as cached
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bench'))
import prepare_llama3_tasks as inputs


def margin(row):
    top=runtime.np.partition(row,-2)[-2:]
    return float(max(top)-min(top))


def generate(session, prompt, maximum, stops):
    if (type(maximum)is not int or maximum<=0 or not prompt or len(prompt)+maximum>session.capacity
            or session.position!=0 or session.history):
        raise ValueError('Empty admitted task session and full output reserve required')
    row=session.forward(prompt);visible=[];samples=[];reason=None
    for index in range(maximum):
        selected=cached.greedy(row)
        samples.append(dict(index=index,token_id=selected,margin=margin(row),
                            row_sha256=hashlib.sha256(memoryview(row).cast('B')).hexdigest()))
        if selected in stops:
            reason=dict(kind='stop_token',token_id=selected);break
        visible.append(selected)
        if len(visible)==maximum:
            reason=dict(kind='output_limit');break
        row=session.forward([selected])
    committed=len(prompt)+len(visible)-(reason['kind']=='output_limit')
    history=prompt+visible
    if reason['kind']=='stop_token':history=history+[reason['token_id']]
    if session.position!=committed or session.history!=history[:committed] or len(history)!=committed+1:
        raise ValueError('Original sampled/committed state differs')
    return dict(visible_token_ids=visible,stop_reason=reason,samples=samples,
                committed_tokens=committed,history_tokens=len(history),pending_tokens=1)


def trajectory(reader, prompt, result, stops):
    visible=result['visible_token_ids'];reason=result['stop_reason']
    if any(type(i)is not int or not 0<=i<128256 or i in stops for i in visible):
        raise ValueError('Visible task IDs differ')
    expected=list(visible)
    if reason['kind']=='stop_token':
        if reason['token_id']not in stops:raise ValueError('Unknown hidden stop')
        expected.append(reason['token_id'])
    elif reason['kind']!='output_limit':raise ValueError('Incomplete task reason')
    positions=list(range(len(prompt)-1,len(prompt)-1+len(expected)))
    logits=runtime.graph.quality_forward(reader,prompt+visible,positions)
    observed=[];mismatches=[]
    for index,(wanted,row)in enumerate(zip(expected,logits,strict=True)):
        actual=cached.greedy(row);item=dict(index=index,absolute_row=positions[index],
            expected_id=wanted,stateless_id=actual,incremental_margin=result['samples'][index]['margin'],stateless_margin=margin(row))
        observed.append(item)
        if actual!=wanted:mismatches.append(item)
    return dict(verified=not mismatches,visible_predictions=len(visible),
                terminal_stop_checked=reason['kind']=='stop_token',rows=observed,mismatches=mismatches)


def produce(model, folder, manifest_sha, output, probe=False):
    output=Path(output);output.mkdir()
    report=dict(format='toklet-llama3-quality-tasks-original-v1',completed=False,
                manifest_sha256=manifest_sha,mode='authored-probe'if probe else'full-tasks',full_quality_qualified=False)
    reader=None
    try:
        if importlib.metadata.version('numpy')!='2.3.2' or sys.byteorder!='little':raise ValueError('Reference runtime differs')
        manifest,cases=inputs.load(folder,manifest_sha)
        if probe:cases=[dict(id='authored-probe',category='probe',prompt_tokens=[128000,1,7,13],max_output_tokens=4)]
        paths=[Path(__file__),Path(cached.__file__),Path(weights.__file__),Path(runtime.graph.__file__),Path(runtime.__file__),Path(inputs.__file__),
               Path(runtime.short.__file__),runtime.graph.ROOT/'tools/reference/llama3_loss1024.py',runtime.graph.ROOT/'tools/bench/acquire_llama3_original.py']
        report['source_sha256']={str(p):runtime.short.sha(p)for p in paths}
        provider={str(p):runtime.short.sha(p)for p in sorted(Path(runtime.np.__file__).resolve().parent.rglob('*'))
                  if p.is_file()and p.suffix in ['.py','.so','.dylib']and '__pycache__'not in p.parts}
        pins=output/'runtime-pins.json';pins.write_text(json.dumps(provider,indent=2)+'\n')
        report['runtime_pins']=dict(file=pins.name,sha256=runtime.short.sha(pins),entries=len(provider))
        report['model']=manifest['models']['original'];runtime.resource_gate(output)
        reader=weights.Cache.load(model);session=cached.Session(reader,manifest['capacity']);runtime.resource_gate(output)
        report['cache_plan']=reader.plan;count=0;all_replayed=True
        with(output/'tasks.jsonl').open('x')as journal:
            for index,case in enumerate(cases):
                print('original case',index,case['id'],flush=True)
                session.reset();started=time.monotonic()
                result=generate(session,case['prompt_tokens'],case['max_output_tokens'],manifest['stop_ids'])
                replay=trajectory(reader,case['prompt_tokens'],result,manifest['stop_ids'])
                if not replay['verified']:all_replayed=False
                record=dict(id=case['id'],category=case['category'],visible_token_ids=result['visible_token_ids'],stop_reason=result['stop_reason'],prompt_tokens=len(case['prompt_tokens']),
                            output_reserve=case['max_output_tokens'],generation=result,trajectory=replay,elapsed_seconds=time.monotonic()-started)
                journal.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n');journal.flush()
                session.reset()
                if session.position or session.history or session.logits is not None:raise ValueError('Task reset failed')
                runtime.resource_gate(output);count+=1
        reader.close();reader=None
        if count!=(1 if probe else 100):raise ValueError('Incomplete task collection')
        if any(runtime.short.sha(Path(p))!=v for p,v in provider.items()):raise ValueError('Numerical provider changed')
        if any(runtime.short.sha(Path(p))!=v for p,v in report['source_sha256'].items()):raise ValueError('Source changed')
        inputs.load(folder,manifest_sha)
        report.update(completed=True,cases=count,all_original_trajectories_verified=all_replayed,
                      tasks_sha256=runtime.short.sha(output/'tasks.jsonl'),peak_rss_bytes=runtime.resource_gate(output))
    except BaseException as error:
        report.update(error=repr(error),peak_rss_bytes=runtime.resource.getrusage(runtime.resource.RUSAGE_SELF).ru_maxrss)
    finally:
        if reader is not None:reader.close()
        (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path);parser.add_argument('inputs',type=Path)
    parser.add_argument('manifest_sha');parser.add_argument('new_output',type=Path);parser.add_argument('--probe',action='store_true')
    args=parser.parse_args();report=produce(args.model,args.inputs,args.manifest_sha,args.new_output,args.probe)
    print(json.dumps(report,indent=2));raise SystemExit(0 if report['completed']else 1)
