from pathlib import Path
import json,sys
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'tools/experiments'));import m7_physical_lifecycle as p
out=Path(__file__).resolve().parent;pins={};jobs={}
for model in ('small','larger'):
 config,host,extra=p.job(model);p.m.integration.merge(pins,extra)
 if set(config)!={'version','modelFolder','packedFolder','files','metal','capacity','chunk','numericalBudget','processBudget','prompt','maximum','vocabulary'} or config['capacity']!=1152 or len(config['prompt'])!=512 or config['maximum']!=128 or config['vocabulary']!=151936 or len(config['prompt'])+config['maximum']+3>config['capacity']:raise ValueError('Exact real lifecycle config differs')
 raw=(json.dumps(config,sort_keys=True,separators=(',',':'))+'\n').encode();jobs[model]=dict(config=config,host=host,canonical_job_sha256=p.m.package.hashlib.sha256(raw).hexdigest())
p.m.verify_inputs(pins)
with (out/'jobs.json').open('x') as f:json.dump(dict(passed=True,jobs=jobs,bindings=pins,native_models_loaded=False,physical_device_qualified=False,m7_complete=False),f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,models=2,native_models_loaded=False)))
