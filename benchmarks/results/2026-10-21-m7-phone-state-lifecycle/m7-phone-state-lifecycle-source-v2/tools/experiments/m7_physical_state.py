"""Sealed physical session-state jobs; no numerical or ABI changes."""
from pathlib import Path
import json
import m7_physical_measurement as m
import m7_native_state_check as replay
ROOT=m.ROOT
NATIVE_SHA='06016a4e6eeea98c9c9c0f898cc4fd107c85415ed26cd3faa8de221386002971'
APP_PROOF='benchmarks/results/2026-10-21-m7-state-app-qualification-v1.json'
APP_PROOF_SHA='5320c1e7c48978ec4a14203a19b1675a80a854ec0b9f77bf5de0320653ca2ea1'
READY_FORMAT='toklet-m7-physical-state-ready-v1'
RESULT_FORMAT='toklet-m7-physical-state-v1'

def job(model,workload='512',verify_sources=True):
 if model not in ('small','larger') or workload!='512':raise ValueError('Explicit real-model state512 job required')
 base,host,pins=m.job(model,workload,verify_sources=verify_sources)
 proof=m.integration.metadata(ROOT/APP_PROOF,APP_PROOF_SHA)
 if (proof['passed'] is not True or proof['app_integration'] is not True or proof['review']!='Astra accepted source and completed app component evidence with no findings'
     or proof['actual_os_lifecycle_qualified'] is not False or proof['physical_device_qualified'] is not False):raise ValueError('Accepted current state app proof required')
 m.verify_inputs(proof['app_code_pins']);m.integration.merge(pins,proof['app_code_pins']);pins[APP_PROOF]=APP_PROOF_SHA
 path=Path(replay.__file__).resolve();m.integration.merge(pins,{str(path.relative_to(ROOT)):m.package.sha(path)})
 config={k:base[k] for k in ('version','modelFolder','packedFolder','files','metal','capacity','chunk','numericalBudget','processBudget','prompt','maximum')};config['vocabulary']=151936
 return config,host,pins

def stage(config,host,output,model,workload):
 expected,_,_=job(model,workload,verify_sources=False)
 if config!=expected:raise ValueError('Frozen state job differs')
 return m.stage(config,host,output)

def validate_console(text,config,signed):
 markers={'TOKLET_INSTALLATION_JSON=':'proof','TOKLET_STATE_JSON=':'result'};rows=[]
 for line in text.splitlines():
  for prefix,kind in markers.items():
   if line.startswith(prefix):rows.append((kind,json.loads(line[len(prefix):])));break
  else:
   if line.startswith('TOKLET_'):raise ValueError('Unexpected state marker')
 if [k for k,_ in rows]!=['proof','proof','result']:raise ValueError('Exact installed/state stream required')
 proofs=[rows[0][1],rows[1][1]];result=rows[2][1]
 if any(p['simulator'] is not False or p['files']!=signed for p in proofs) or [p['phase'] for p in proofs]!=['before','after']:raise ValueError('Installed state bundle proof differs')
 if config['metal'] is not True or config['capacity']!=1152 or config['chunk']!=128 or len(config['prompt'])!=512 or config['maximum']!=128 or config['vocabulary']!=151936:raise ValueError('Bounded physical state workload required')
 replay.validate_result(result,'metal','small',config)
 for row in result['rows']:
  env=row['environment']
  if (not m.uint(env['applicationState'],32) or env['applicationState']!=0 or not m.uint(env['thermalState'],32) or env['thermalState']>3
      or type(env['lowPowerMode']) is not bool or not m.uint(env['batteryState'],32) or env['batteryState'] not in range(1,4)
      or not m.nonnegative(env['batteryLevel']) or env['batteryLevel']>1):raise ValueError('Actual foreground/power/thermal state required')
 return dict(result=result,installed_proofs=proofs)
