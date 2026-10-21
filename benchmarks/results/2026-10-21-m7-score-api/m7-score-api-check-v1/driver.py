"""Owned, source-bound validation of the additive score API in isolation."""
from pathlib import Path
import hashlib,json,re,sys
def validate_gpu_output(gpu_text):
 summaries=re.findall(r'^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out;',gpu_text,re.M)
 if len(summaries)!=1 or tuple(map(int,summaries[0][:4]))!=(1,0,0,0):raise ValueError('Exact selected GPU test did not run')
 if len(re.findall(r'^test native::tests::metal_native_selected_models_match_direct_sessions \.\.\. ok$',gpu_text,re.M))!=1:raise ValueError('Selected GPU test identity differs')
 rows=re.findall(r'native_metal_model=(\S+) ids=(\[[0-9, ]+\]) state=7/6/1 combined_admission_bytes=(\d+) passed',gpu_text)
 expected={'Qwen2.5-0.5B-Instruct':[1410,1492,752],'Qwen2.5-1.5B-Instruct':[614,264,3405]}
 if len(rows)!=2 or {row[0] for row in rows}!=set(expected):raise ValueError('Both selected model results required')
 for name,ids,admission in rows:
  if json.loads(ids)!=expected[name] or not 0<int(admission)<=6*1024**3:raise ValueError('Selected GPU oracle/state/admission differs')
 return rows

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/experiments'))
import m7_apple_package as p
import m7_apple_integration as integration
assert p.ROOT==ROOT
OUT=Path(__file__).resolve().parent;LOG=OUT/'logs';LOG.mkdir();(OUT/'tmp').mkdir()
def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
paths=[ROOT/'Cargo.toml',ROOT/'Cargo.lock',ROOT/'AGENTS.md',Path(__file__).resolve(),
       *[f for folder in ('src','tests','examples','apple/include') for f in (ROOT/folder).rglob('*') if f.is_file()],
       ROOT/'apple/tests/native_smoke.c',ROOT/'benchmarks/sources.json',
       *[f for f in (ROOT/'benchmarks/contract-v1').rglob('*') if f.is_file()],Path(p.__file__).resolve(),Path(integration.__file__).resolve()]
pins={str(path.relative_to(ROOT)):sha(path) for path in paths};commands=[];endpoints=[];primary=None;package_sha=None
def merge(incoming):
 for name,digest in incoming.items():
  if name in pins and pins[name]!=digest:raise ValueError('Conflicting input: '+name)
  pins[name]=digest
protected={'/Users/coloner/work/rust/toklet/target/release/toklet':'5250694e814ea112148a52cfdeaf5bc6286ff2804b3acfd8f89e80bc73122a59',
           '/Users/coloner/work/rust/toklet/target/release/examples/cpu_bench':'e3debabbae7db9e5801bc9bfba177275444ebe1d9849eb3ad0f2dff487ac8dea'}
def guard():
 for name,digest in pins.items():
  if sha(ROOT/name)!=digest:raise ValueError('Changed admitted input: '+name)
 for name,digest in protected.items():
  if sha(Path(name))!=digest:raise ValueError('Protected CPU binary differs')
def execute(command,label,timeout=900):
 guard();row=p.owned(command,LOG/(label+'.stdout'),LOG/(label+'.stderr'),env,timeout);commands.append(row)
 if row['primary_error'] or row['cleanup_errors']:raise RuntimeError(label+': '+str(row['primary_error'])+' '+repr(row['cleanup_errors']))
 return ROOT/row['stdout']
previous=p.signal.signal(p.signal.SIGTERM,p.interrupted)
try:
 env,absence=p.build_environment();env.update(TMPDIR=str(OUT/'tmp'),CARGO_TARGET_DIR=str(OUT/'target'),RUSTFLAGS='-Cpanic=unwind',CARGO_PROFILE_RELEASE_PANIC='unwind')
 execute(['rustup','run','1.90','rustc','-vV'],'rustc-version',120)
 cargo=['rustup','run','1.90','cargo']
 execute(cargo+['fmt','--all','--','--check'],'fmt')
 execute(cargo+['clippy','--offline','--locked','--all-targets','--','-D','warnings'],'clippy-cpu')
 execute(cargo+['test','--offline','--locked'],'test-cpu')
 execute(cargo+['clippy','--offline','--locked','--all-targets','--features','metal','--','-D','warnings'],'clippy-metal')
 execute(cargo+['test','--offline','--locked','--features','metal'],'test-metal-feature')
 proof=ROOT/'benchmarks/results/2026-10-19-m6-context-reviewed-closure-v1.json'
 with proof.open('rb') as stream:raw=stream.read(16*1024**2+1)
 if len(raw)>16*1024**2 or hashlib.sha256(raw).hexdigest()!='9873fe34fb887a6dac2936f5ea505ee57bc2bbe7ae0da742b6e1cfc52ddb9320':raise ValueError('Frozen model prerequisite differs')
 frozen=json.loads(raw);pins[str(proof.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
 wanted={f'models/{model}/{leaf}' for model in ('Qwen2.5-0.5B-Instruct','Qwen2.5-1.5B-Instruct') for leaf in ('config.json','model.safetensors')}
 wanted.update(name for name in frozen['bindings'] if name.startswith('artifacts/m5-larger-q4-v2/'))
 if len(wanted)!=201:raise ValueError('Frozen 201-file model inventory differs')
 merge({name:frozen['bindings'][name] for name in wanted})
 actual={str((ROOT/'artifacts/m5-larger-q4-v2'/path.name).relative_to(ROOT)) for path in (ROOT/'artifacts/m5-larger-q4-v2').iterdir() if path.is_file()}
 if actual!={name for name in wanted if name.startswith('artifacts/m5-larger-q4-v2/')}:raise ValueError('Packed file inventory differs')
 gpu_output=execute(cargo+['test','--offline','--locked','--release','--features','metal','--lib','native::tests::metal_native_selected_models_match_direct_sessions','--','--ignored','--exact','--nocapture'],'selected-metal-score',1200)
 validate_gpu_output(p.small(gpu_output))
 destination=ROOT/'artifacts/m7-apple-package-v5'
 execute([sys.executable,str(ROOT/'tools/experiments/m7_apple_package.py'),str(destination)],'apple-package',3600)
 build_path=destination/'build.json'
 with build_path.open('rb') as stream:raw=stream.read(16*1024**2+1)
 if len(raw)>16*1024**2:raise ValueError('Oversized package receipt')
 package_sha=hashlib.sha256(raw).hexdigest();build=json.loads(raw)
 merge({str(build_path.relative_to(ROOT)):package_sha})
 merge(integration.admit_build(destination,package_sha))
 if not build['passed'] or build['primary_error'] or build['endpoint_errors']:raise ValueError('Apple package did not pass')
 # Reuse the package's platform-checked C executable for both actual Mac paths.
 tiny=ROOT/'artifacts/m7-native-tiny-v2'
 for name,digest in {'config.json':'84b6e6078f35006f49ae2e2a8663093cdc8d2813e2b7f28ebfa6142113e9c1d9','model.safetensors':'b01a114bf2d44f0c6cb438035e85535b658e803c76691729facd78b15eefbce2'}.items():pins[str((tiny/name).relative_to(ROOT))]=digest
 exe=destination/'aarch64-apple-darwin-link-smoke'
 mac=next(row for row in build['slices'] if row['target']=='aarch64-apple-darwin')
 if str(exe.relative_to(ROOT))!=mac['link_executable'] or sha(exe)!=mac['link_sha256']:raise ValueError('Unadmitted Mac C executable')
 for label,extra in [('c-mac-cpu',[]),('c-mac-metal',['metal'])]:
  out=execute([str(exe),str(tiny),*extra],label,180)
  if json.loads(p.small(out))!={'ids':[47,47,47,47],'abi':1,'passed':True}:raise ValueError('Actual C score integration differs')
except BaseException as failure:primary=type(failure).__name__+': '+str(failure)
finally:
 p.signal.signal(p.signal.SIGTERM,previous)
 try:guard()
 except BaseException as failure:endpoints.append(type(failure).__name__+': '+str(failure))
 result=dict(format='toklet-m7-score-api-check-v1',passed=primary is None and not endpoints and len(commands)==10 and package_sha is not None,collection_root=str(ROOT),
             bindings=pins,protected_cpu_binaries=protected,commands=commands,controlled_environment=locals().get('env',{}),
             absent_cargo_configs=locals().get('absence',[]),package_build_sha256=package_sha,primary_error=primary,endpoint_errors=endpoints,
             physical_device_qualified=False,m7_complete=False)
 with (OUT/'checks.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
 print(json.dumps(dict(passed=result['passed'],primary_error=primary,endpoint_errors=endpoints,package_build_sha256=package_sha)))
sys.exit(0 if result['passed'] else 1)
