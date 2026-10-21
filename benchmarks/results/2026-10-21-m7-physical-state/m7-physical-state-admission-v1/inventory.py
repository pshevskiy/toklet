from pathlib import Path
import json,sys
R=Path.cwd();sys.path.insert(0,str(R/'tools/experiments'));import m7_physical_state as p;import m7_physical_lifecycle as life
jobs={};replayed={}
for model in ('small','larger'):
 config,host,pins=p.job(model);other,otherhost,otherpins=life.job(model,verify_sources=False)
 assert config==other and host==otherhost and config['capacity']==1152 and len(config['prompt'])==512
 jobs[model]=dict(config=config,host=host,bindings=pins,lifecycle_bindings=otherpins)
base=R/'artifacts/m7-state-app-native-v4';d=json.loads((base/'checks.json').read_text())
for model,j in d['jobs'].items():
 checked=p.replay.validate_output((base/'logs'/(model+'.stdout')).read_text(),'cpu' if model=='tiny-cpu' else 'metal',j['kind'],j['config']);assert checked==d['checked'][model];replayed[model]=dict(rows=len(checked['diagnostic']['rows']),prefill=checked['diagnostic']['prefillWitness'])
print(json.dumps(dict(passed=True,jobs=jobs,replayed=replayed,device_commands=0,native_model_executions=0)))
