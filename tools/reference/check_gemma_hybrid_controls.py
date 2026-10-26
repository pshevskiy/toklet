"""Independent ownership/admission transcript replay; no engine import."""
from pathlib import Path
import hashlib,json,re,struct,sys
import check_gemma_hybrid_tiny as check
TEST='model::gemma_packed::tests::hybrid::controls::admission_failure_replay_generation_and_source_ownership'
PREFIX='GEMMA_HYBRID_CONTROL_'

def expected_weights(data,binary):
    names=['model.embed_tokens.weight','model.embed_tokens_per_layer.weight','model.per_layer_model_projection.weight']
    for i in range(4):
        stages=['self_attn.q_proj.weight','self_attn.o_proj.weight']
        if i<2:stages+=['self_attn.k_proj.weight','self_attn.v_proj.weight']
        stages+=['mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','per_layer_input_gate.weight','per_layer_projection.weight']
        names += [f'model.layers.{i}.'+stage for stage in stages]
    out=[]
    for name in names:
        rows,columns=data['weights'][name]['shape'];components={}
        for label,source in [('codes','weight'),('scales','scales'),('biases','biases')]:
            spec=data['packed_original_bf16'][name][source]
            raw=binary[spec['offset']:spec['offset']+spec['bytes']]
            check.require(len(raw)==spec['bytes'] and hashlib.sha256(raw).hexdigest()==spec['sha256'],'complete original packed component')
            padded=raw+b'\0'*((-len(raw))%4)+struct.pack('<I',0x5a170031)*7
            components[label]={'words':len(padded)//4,'sha256':hashlib.sha256(padded).hexdigest()}
        out.append(dict(rows=rows,columns=columns,chunks=[dict(rows=rows,**components)]))
    return out

def records(raw):
    check.require(len(raw)<=1024**2,'bounded controls transcript');events=[];wrapper=[]
    for line in raw.decode().splitlines():
        check.require(len(line.encode())<=4096,'bounded controls record')
        if PREFIX in line:
            prefix,rest=line.split(PREFIX,1);check.require(prefix in ('','test '+TEST+' ... '),'exact test prefix')
            name,body=rest.split(' ',1);events.append((name,check.document(body)))
        elif line.strip():wrapper.append(line)
    check.require(len(wrapper)==3 and wrapper[:2]==['running 1 test','ok'] and re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s',wrapper[2]),'complete exact control wrapper')
    return events

def replay(events,data,binary):
    expected=expected_weights(data,binary);n=len(expected)
    check.require(len(events)==2*n+4,'complete control record count')
    identities=set();before=[]
    for i,wanted in enumerate(expected):
        label,record=events[i];check.require(label=='WEIGHT_BEFORE','weight before order')
        check.require(type(record)is dict and set(record)=={'index','matrix'},'weight record fields');check.same(record['index'],i)
        actual=record['matrix'];check.require(type(actual)is dict and set(actual)=={'rows','columns','chunks'},'matrix fields')
        cleaned=check.document(json.dumps(actual))
        check.require(type(cleaned['chunks'])is list and len(cleaned['chunks'])==1,'one tiny chunk')
        for component in ['codes','scales','biases']:
            value=cleaned['chunks'][0][component];check.require(type(value)is dict and set(value)=={'words','sha256','identity'},'buffer fields')
            identity=value.pop('identity');check.require(type(identity)is str and re.fullmatch('0x[0-9a-f]+',identity) and int(identity,16)!=0 and identity not in identities,'unique live buffer identity');identities.add(identity)
        check.same(cleaned,wanted);before.append(record)
    label,memory=events[n];check.require(label=='MEMORY','memory record order')
    check.require(type(memory)is dict and set(memory)=={'raw_admitted_bytes','model_bytes','generation_wrapper_bytes','generation','raw_minus_one_rust_allocations','generation_minus_one_rust_allocations','smaller_request_capacity','physical_capacity'},'memory fields')
    m=memory['generation'];categories=['weights_bytes','kv_bytes','scratch_bytes','history_bytes','logits_bytes','metadata_bytes','metadata_allowance_bytes']
    check.require(type(m)is dict and set(m)==set(categories)|{'admitted_bytes'},'memory categories')
    for value in [*m.values(),memory['raw_admitted_bytes'],memory['model_bytes'],memory['generation_wrapper_bytes']]:check.require(type(value)is int and value>0,'positive exact byte counts')
    check.require(sum(m[k] for k in categories)==m['admitted_bytes']<=512*1024**2,'memory sum and budget')
    check.same(m['admitted_bytes'],memory['raw_admitted_bytes']+memory['generation_wrapper_bytes'])
    check.same(m['history_bytes'],256);check.same(m['logits_bytes'],4352);check.same(m['metadata_allowance_bytes'],96*1024**2)
    for key,value in [('raw_minus_one_rust_allocations',0),('generation_minus_one_rust_allocations',0),('smaller_request_capacity',5),('physical_capacity',32)]:check.same(memory[key],value)
    extra=memory['raw_admitted_bytes']-memory['model_bytes']-64*1024**2;check.require(extra>0,'additional failed-session ownership')
    for i,(gather,calls) in enumerate([(False,12),(True,1)]):
        label,state=events[n+1+i];check.require(label=='FAILURE','failure record order')
        check.same(state,dict(gather=gather,successful_calls_before_injection=calls,committed_prefix=3,prefix_unchanged=True,reset_rejected=True,reuse_rejected=True,fresh_replay_tokens=5,extra_live_session_bytes=extra,in_flight_failure_qualified=False))
    for i,record in enumerate(before):
        label,after=events[n+3+i];check.require(label=='WEIGHT_AFTER','weight after order');check.same(after,record)
    label,state=events[-1];check.require(label=='COMPLETE','terminal control record')
    check.same(state,dict(generation_stop_cancel_output_failure_continue=True,foreign_source_rejected=True,weight_buffers_unchanged=True,different_device_tested=False,whole_e2b_qualified=False,performance_qualified=False,phone_qualified=False))
    return dict(passed=True,matrices=n,unique_unchanged_buffers=len(identities),memory=memory,injected_pre_command_failures=2,in_flight_failure_qualified=False,whole_e2b_qualified=False,performance_qualified=False,phone_qualified=False)

if __name__=='__main__':
    transcript,directory=map(Path,sys.argv[1:]);data,binary=check.fixture(directory)
    print(json.dumps(replay(records(check.read(transcript,1024**2)),data,binary)))
