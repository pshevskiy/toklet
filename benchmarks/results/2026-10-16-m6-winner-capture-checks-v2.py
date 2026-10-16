"""Isolated formatting/lint/portable example checks; never execute a GPU test."""
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/experiments'))
import m6_winner_capture_build as build
record=json.loads((ROOT/'artifacts/m6-winner-capture-build-v2.json').read_text())
build.verify_record(record)
snapshot=ROOT/record['snapshot']
env={k:os.environ[k] for k in ('HOME','PATH','TMPDIR') if k in os.environ}
env.update(CARGO_TARGET_DIR=str(ROOT/'artifacts/m6-winner-capture-target-v2'),CARGO_INCREMENTAL='0',RUSTFLAGS='',CARGO_ENCODED_RUSTFLAGS='')
build.reject_ambient_configuration(snapshot,Path(env['HOME'])/'.cargo')
commands=[['cargo','fmt','--check'],
 ['cargo','clippy','--offline','--locked','--all-targets','--features','metal','--','-D','warnings'],
 ['cargo','test','--offline','--locked','--features','metal','--example','metal_context_capture']]
with (ROOT/'artifacts/m6-winner-capture-checks-v2.log').open('xb') as log:
 for command in commands:
  log.write((' '.join(command)+'\n').encode());log.flush()
  subprocess.run(command,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
build.verify_record(record)
print('Isolated formatting, strict all-target Metal lint and portable capture tests passed')
