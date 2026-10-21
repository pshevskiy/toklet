from pathlib import Path
import hashlib,json,sys
ROOT=Path('/Users/coloner/work/rust/toklet');sys.path.insert(0,str(ROOT/'tools/experiments'))
import m7_physical_measurement as m
own=Path(__file__);before=hashlib.sha256(own.read_bytes()).hexdigest()
args=(ROOT/'artifacts/m7-apple-package-v4',ROOT/'artifacts/m7-physical-admission-v1/private-selection.json',
      '711ef5916ad223096bcb91cd93d7814f2c44056ab7494882d87af7d42a9804db',ROOT/'artifacts/m7-physical-prepared-v9',
      'c7f5be741d76259bce3ac03820a4bd0ecda915c6761c1263947b2a465c41fbf7')
proof=ROOT/'artifacts/m7-storage-proof-v3'
code=m.main(*args,'tiny-cpu','short',proof)
if code==0:
 assert hashlib.sha256(own.read_bytes()).hexdigest()==before
 digest=hashlib.sha256((proof/'measurement.json').read_bytes()).hexdigest()
 code=m.main(*args,'larger','short',ROOT/'artifacts/m7-larger-short-ready-v2',prepare_only=True,storage_proof=proof,storage_sha=digest)
assert hashlib.sha256(own.read_bytes()).hexdigest()==before
(own.parent/'execution.json').write_text(json.dumps(dict(driver_sha256=before,exit_code=code,proof=str(proof.relative_to(ROOT))),indent=2)+'\n')
sys.exit(code)
