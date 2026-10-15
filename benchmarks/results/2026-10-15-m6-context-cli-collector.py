import hashlib,json,pathlib,shutil,sys
ROOT=pathlib.Path('/Users/coloner/work/rust/toklet')
sys.path.insert(0,str(ROOT/'tools/experiments'))
import metal_cli as shared

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    out=ROOT/'artifacts/m6-context-cli-native-v1';out.mkdir(exist_ok=False)
    binary=out/'toklet';shutil.copyfile(ROOT/'artifacts/m5-checks/release/toklet',binary);binary.chmod(0o755)
    source=out/'tiny';source.mkdir()
    origin=ROOT/'artifacts/m5-tiny-source-v1'
    config=json.loads((origin/'config.json').read_bytes());config['max_position_embeddings']=32768
    (source/'config.json').write_text(json.dumps(config)+'\n')
    shutil.copyfile(origin/'model.safetensors',source/'model.safetensors')
    script_path=out/'collector.py';shutil.copyfile(pathlib.Path(__file__),script_path)
    files=[binary,source/'config.json',source/'model.safetensors',script_path,ROOT/'Cargo.toml',ROOT/'Cargo.lock']
    files+=list((ROOT/'src').rglob('*.rs'))+list((ROOT/'src').rglob('*.metal'))
    files += [pathlib.Path(m.__file__) for m in (shared,shared.metal_engine,shared.native_usage,shared.prepare)] if hasattr(shared,'prepare') else [pathlib.Path(m.__file__) for m in (shared,shared.metal_engine,shared.native_usage)]
    pins={str(p.relative_to(ROOT)):sha(p) for p in files}
    result={'scope':'Long admission/CLI component only; no long-prefix inference, quality or performance acceptance','passed':False,'pins':pins,'runs':[]}
    def script(request,record):
        shared.raw_script(request,record)
        cap=record['ready']['context_capacity']
        before=request({'action':'continue','max_new_tokens':1})[-1]['statistics']['generation']
        events=request({'tokens':[1]*cap,'max_new_tokens':1})
        assert events[-1]['event']=='error' and not any(e.get('event')=='token' for e in events)
        after=request({'action':'continue','max_new_tokens':1})[-1]['statistics']['generation']
        assert after['reused_tokens']==before['evaluated_tokens'] and after['prefill_tokens']==1
    def admission(event,mode,cap,chunk):
        assert event['execution'].get('attention_mode')=='Online'
        clone=dict(event);clone['execution']=dict(event['execution']);del clone['execution']['attention_mode']
        shared.ready(clone,mode,cap,chunk)
    try:
        with (out/'raw.jsonl').open('x') as raw,(out/'stderr').open('xb') as stderr:
            # Preserve exact legacy default schema and explicit Reference schema.
            for extra in [(),('--metal-attention','reference')]:
                result['runs'].append(shared.child(binary,source,'HalfCompensated',raw,stderr,shared.raw_script,capacity=32,chunk=3,extra=extra))
                assert result['runs'][-1]['passed'],result['runs'][-1].get('error')
            for cap in [4096,8192,16384,32768]:
                result['runs'].append(shared.child(binary,source,'HalfCompensated',raw,stderr,script,capacity=cap,chunk=129,
                    extra=('--metal-attention','online'),admission=admission))
                assert result['runs'][-1]['passed'],result['runs'][-1].get('error')
        result['passed']=True
    finally:
        result['endpoints_unchanged']=all(sha(ROOT/name)==value for name,value in pins.items())
        result['passed']=result['passed'] and result['endpoints_unchanged']
        with (out/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    assert result['passed']
    print('6 native CLI workers passed: default/reference compatibility and all four Online admission depths; output reservation rejection preserved state')
if __name__=='__main__':main()
