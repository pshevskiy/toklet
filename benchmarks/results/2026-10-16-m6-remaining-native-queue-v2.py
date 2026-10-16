"""Serial execution only: no acceptance decisions, no skipped boundary phases."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
roles=[('small','long',32768,2),('small','long-confirmation',32768,1)]
roles += [('larger',plan,depth,1) for depth in (4096,8192,16384,32768) for plan in ('long','long-confirmation')]
paths=[ROOT/f'artifacts/m6-context-{model}-{plan}-{depth}-v{version}.json' for model,plan,depth,version in roles]
status=ROOT/'artifacts/m6-remaining-native-queue-v2.status.jsonl'
log_path=ROOT/'artifacts/m6-remaining-native-queue-v2.log'
def run_collector(command,log):
 collector=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 try:
  return collector.wait()
 except KeyboardInterrupt as original:
  # Supported stop is SIGINT to the queue process group: the collector gets
  # the same single signal and runs its existing owned-worker finally cleanup.
  try:collector.wait(timeout=120)
  except BaseException as cleanup:original.add_note(f'Collector PID{collector.pid} did not finish cleanup: {cleanup!r}; verify owned worker absence before retry')
  raise

def main():
 for path in [status,log_path,*[item for p in paths for item in (p,p.with_suffix('.raw.jsonl'),p.with_suffix('.stderr'))]]:
  if path.exists():raise ValueError('Exclusive queue destination exists: '+str(path))
 with status.open('x') as events,log_path.open('xb') as log:
  def emit(**event):
   event['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
   text=json.dumps(event,allow_nan=False);events.write(text+'\n');events.flush();print(text,flush=True)
  emit(event='queue_begin',roles=roles,qualification=False)
  for index,((model,plan,depth,version),path) in enumerate(zip(roles,paths,strict=True)):
   collector='m6_typed_quality.py' if plan=='quality' else 'm6_context_long_budget.py'
   command=[str(ROOT/'.venv/bin/python'),str(ROOT/'tools/experiments'/collector),'--model',model,'--depth',str(depth),'--event-read-timeout','43200','--output',str(path.relative_to(ROOT))]
   if plan!='quality':command += ['--plan',plan]
   emit(event='role_begin',index=index,role=[model,plan,depth,version],command=command)
   returncode=run_collector(command,log)
   log.flush()
   if returncode:
    emit(event='role_failed',index=index,exit_code=returncode);raise SystemExit(returncode)
   with path.open() as stream:record=json.load(stream)
   if any(record.get(key) is not True for key in ('passed','collection_completed','identities_unchanged')) or record.get('endpoint_errors')!=[] or record['worker'].get('exit_code')!=0 or record['worker'].get('cleanup_errors')!=[]:
    emit(event='role_status_failed',index=index);raise SystemExit(1)
   emit(event='role_complete',index=index,path=str(path.relative_to(ROOT)),qualification=False)
  emit(event='queue_complete',qualification=False,scope='Collection only; independent full-scope replay and Astra acceptance remain required')

if __name__=="__main__":main()
