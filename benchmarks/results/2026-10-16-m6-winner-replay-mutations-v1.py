"""Portable negative admission checks for the frozen diagnostic replay helper."""
import copy
import importlib.util
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('winner_replay',ROOT/'artifacts/m6-winner-capture-replay-v1.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)
record,_=replay.audit.read_json(ROOT/'artifacts/m6-winner-capture-larger-4096-v1.json')
mutations=[('build_record','artifacts/another-build.json'),('build_sha256','0'*64),('original_quality_record','artifacts/another-original.json'),('original_quality_sha256','0'*64),('model','small'),('depth',8192),('indices',[90]),('indices',[90,True]),('capture_directory','artifacts/another-capture')]
count=0
for key,value in mutations:
 changed=copy.deepcopy(record);changed[key]=value
 with patch.object(replay.audit,'read_json',return_value=(changed,'0'*64)),patch.object(replay.capture,'admission',side_effect=AssertionError('Invalid record reached build admission')) as admission:
  try:replay.run()
  except ValueError:pass
  else:raise AssertionError('Frozen identity mutation accepted: '+key)
  admission.assert_not_called()
 count+=1
for value in ('toklet-m6-context-v1','other-format',None):
 changed=copy.deepcopy(record);changed['worker']['ready']['format']=value
 # The mocked summary is self-consistent; rejection must precede all projection.
 with patch.object(replay.audit,'read_json',return_value=(changed,'0'*64)),patch.object(replay.capture,'admission',side_effect=AssertionError('Wrong ready format reached admission')) as admission:
  try:replay.run()
  except ValueError:pass
  else:raise AssertionError('Wrong diagnostic format accepted')
  admission.assert_not_called()
 count+=1
print('Passed',count,'independent frozen build/anchor/role/selection/directory/ready-format negative admission mutations')
