"""Offline evidence replay of the exact M6 changed-winner diagnostic."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'Cargo.toml').is_file() and (p/'tools/experiments').is_dir())
sys.path.insert(0,str(ROOT/'tools/experiments'))
import m6_winner_capture as capture
import m6_context_audit as audit
base=capture.base

def run():
 source=Path(__file__).resolve();source_sha=audit.read_bytes(source)[1]
 path=ROOT/'artifacts/m6-winner-capture-larger-4096-v1.json'
 record,record_sha=audit.read_json(path)
 expected=dict(build_record='artifacts/m6-winner-capture-build-v2.json',build_sha256='b93b18054d010c4f8c6b3bd3d31a9719d12c5f5ac4e8785ef57c130c17a91e97',original_quality_record='artifacts/m6-context-larger-quality-4096-v1.json',original_quality_sha256='1fc6cb52b8e19ce86dea5fedb9150e77bc2a8db5e276149cc8dbd4050bf42775',model='larger',depth=4096,indices=[90,127],capture_directory='artifacts/m6-winner-capture-larger-4096-v1')
 for key,value in expected.items():
  base.exact(record.get(key),value,'frozen diagnostic qualification '+key)
 if record.get('format')!='toklet-m6-winner-capture-collection-v1':raise ValueError('Wrong diagnostic format')
 for k in ('passed','collection_completed','identities_unchanged','capture_endpoints_unchanged'):
  if record.get(k) is not True:raise ValueError('Incomplete diagnostic: '+k)
 if record.get('investigation_complete') is not False or record.get('m6_complete') is not False or record.get('endpoint_errors')!=[]:raise ValueError('Diagnostic scope/endpoints differ')
 base.exact(record['worker']['ready'].get('format'),capture.FORMAT,'saved diagnostic ready format')
 binary,pins=capture.admission(capture.build.relative(record['build_record']),record['build_sha256'])
 anchor=capture.build.relative(record['original_quality_record'])
 original,anchor_pins=capture.anchor_admission(anchor,record['original_quality_sha256'])
 def merge(items):
  for p,sha in items.items():
   if p in pins and pins[p]!=sha:raise ValueError('Conflicting diagnostic identity')
   pins[p]=sha
 merge(anchor_pins)
 if (record['model'],record['depth'])!=(original['model'],original['depth']):raise ValueError('Original role mismatch')
 indices=capture.selected_indices(original,record['indices'])
 manifest,arrays=base.shared.contract()
 _,long_arrays,model,packed,fixed=base.inputs(record['model']);merge(fixed)
 modules=(base,audit,capture.analysis,capture.build,base.shared,base.shared.metal_engine,base.shared.native_usage,base.shared.metal_continue,base.loss)
 expected=set(pins)|{Path(capture.__file__).resolve(),*(Path(m.__file__).resolve() for m in modules),ROOT/'tools/experiments/m6_fp32_context_audit.py',ROOT/'tools/experiments/m6_fp32_values.py',ROOT/'tools/experiments/m6_typed_quality.py',ROOT/'tools/experiments/m6_typed_quality_audit.py',ROOT/'tools/experiments/m6_long_quality_audit.py',ROOT/'tools/experiments/m6_context_scope_audit.py'}
 identities=record['identities']
 if set(identities)!={str(p.relative_to(ROOT)) for p in expected}:raise ValueError('Consumed inventory differs')
 merge({ROOT/p:sha for p,sha in identities.items()})
 directory=capture.build.relative(record['capture_directory'])
 worker=record['worker']
 command=[str(binary),str(model),str(packed) if packed else '-',str(record['depth']),'128',str(directory)]
 if worker['command']!=command:raise ValueError('Diagnostic command differs')
 if worker.get('event_read_timeout_seconds')!=capture.TIMEOUT or worker.get('lifetime_timeout_seconds')!=capture.LIFETIME:raise ValueError('Diagnostic timeout differs')
 phases=worker['phases']
 if len(phases)!=1:raise ValueError('Diagnostic phase coverage differs')
 request=dict(original['worker']['phases'][0]['request'],captures=indices)
 if phases[0]['request']!=request:raise ValueError('Diagnostic request differs')
 raw,raw_sha=audit.read_json(path.with_suffix('.raw.jsonl'),lines=True)
 if raw!=[worker['ready'],*phases[0]['events']]:raise ValueError('Original raw coverage differs')
 ids=long_arrays[record['depth']]
 captures=capture.validate_events(phases[0]['events'],original,indices,ids)
 # Explicitly project already-verified diagnostic events back to the original
 # plan solely to reuse unchanged state/admission/usage/lifecycle checks.
 projected=copy.deepcopy(original)
 projected['worker']=copy.deepcopy(worker)
 projected['worker']['ready']['format']='toklet-m6-context-v1'
 projected['worker']['phases'][0]['request']=original['worker']['phases'][0]['request']
 projected['worker']['phases'][0]['events']=phases[0]['events'][len(indices):]
 projected_raw=[projected['worker']['ready'],*projected['worker']['phases'][0]['events']]
 replay=audit.replay(projected,projected_raw,manifest,arrays,ids)
 analyses,file_pins=capture.verify_files(directory,captures,original)
 if analyses!=record['analyses'] or {str(p.relative_to(ROOT)):sha for p,sha in file_pins.items()}!=record['capture_bindings']:raise ValueError('Saved independent analyses/bindings differ')
 peak=record['post_exit_analysis_usage'].get('lifetime_max_phys_footprint')
 if type(peak) is not int or not 0<peak<=base.LIMIT:raise ValueError('Postexit driver charge invalid')
 merge(file_pins)
 merge({path:record_sha,path.with_suffix('.raw.jsonl'):raw_sha,path.with_suffix('.stderr'):audit.read_bytes(path.with_suffix('.stderr'))[1],source:source_sha})
 bindings={str(p.relative_to(ROOT)):sha for p,sha in pins.items()}
 audit.verify_bindings(bindings)
 return dict(format='toklet-m6-winner-capture-replay-v1',passed=True,investigation_complete=False,m6_complete=False,raw_events=len(raw),full_distribution_pairs=len(captures),vocabulary_values_per_distribution=151936,bindings=bindings,replay=replay,analyses=analyses,post_exit_analysis_peak_bytes=peak,scope='Exact original full-distribution reproduction and independent arithmetic; final investigation assessment requires Astra review; no internal cause or task-quality inference')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 if args.output.exists():raise ValueError('Exclusive replay output exists')
 result=run()
 with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps({k:result[k] for k in ('passed','raw_events','full_distribution_pairs','post_exit_analysis_peak_bytes')}))
