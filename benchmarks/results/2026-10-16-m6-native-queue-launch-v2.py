"""Detach one owned supervisor; its collectors keep ownership of GPU children.

Explicit pause sends one SIGINT to the verified queue process group and checks
all owned workers are gone. Publication/close failures use the same graceful
stop. No queued work or measured result is accepted by this launcher.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
ROOT=Path(__file__).resolve().parents[1]
queue=ROOT/'artifacts/m6-remaining-native-queue-v2.py'
owner=ROOT/'artifacts/m6-remaining-native-queue-v2.owner.json'
output=ROOT/'artifacts/m6-remaining-native-queue-v2.supervisor.log'
for path in (owner,output):
 if path.exists() or path.is_symlink():raise ValueError('Exclusive supervisor destination exists')
command=[str(ROOT/'.venv/bin/python'),str(queue)]
source_sha=hashlib.sha256(queue.read_bytes()).hexdigest()
child=None
try:
 with owner.open('x') as metadata,output.open('xb') as stream:
  child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  record=dict(format='toklet-m6-owned-native-queue-v1',pid=child.pid,pgid=child.pid,command=command,source_sha256=source_sha,launched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supervisor_output=str(output.relative_to(ROOT)),qualification=False,scope='Owned serial collector supervision; no measurement acceptance')
  json.dump(record,metadata,indent=2);metadata.write('\n');metadata.flush()
 # Both file closes finish within the cleanup boundary before success returns.
except BaseException as original:
 if child is not None:
  original.add_note(f'Owned queue PID/PGID{child.pid}; verify collector and numerical worker cleanup before retry')
  try:os.killpg(child.pid,signal.SIGINT)
  except ProcessLookupError:pass
  except BaseException as cleanup:original.add_note(f'Graceful queue interrupt failed: {cleanup!r}')
  try:child.wait(timeout=180)
  except BaseException as cleanup:original.add_note(f'Queue cleanup wait failed: {cleanup!r}')
 raise
print(json.dumps(record),flush=True)
