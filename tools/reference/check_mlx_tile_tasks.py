"""Check exact literal-task identity against the admitted original Mac stream."""
import hashlib,json
from pathlib import Path
import sys

root=Path.cwd()
out=Path(sys.argv[1])
archive=root/'artifacts/m8-mlx8-full-task-driver-v1/mac-tasks-full.stdout'
with archive.open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()=='4d0add72f5a27efc80f78cf0b97ff6365223c92338ae8728dcdfb8dd4ee16032'
prefix='TOKLET_TASK_CASE_JSON='
old=[json.loads(line[len(prefix):]) for line in archive.read_text().splitlines() if line.startswith(prefix)]
assert len(old)==100
grade_path=root/'benchmarks/results/2026-10-22-native-mlx8-full-task-quality-v1.json'
grade_raw=grade_path.read_bytes()
assert hashlib.sha256(grade_raw).hexdigest()=='eafc5e1b1c6aea6b5d1b9ab3a1ef34d774b5455f06b77a3a9a1634a973440aa7'
grade=json.loads(grade_raw)
assert grade['passed'] and grade['task_quality_qualified']
mac_grade,=[entry['quality_gate'] for entry in grade['results'] if entry['platform']=='mac']
assert mac_grade['cases']==100 and mac_grade['gate_passed']
normalized='artifacts/m8-task-responses-driver-v3/normalized/mac.json'
accepted_raw=(root/normalized).read_bytes()
assert hashlib.sha256(accepted_raw).hexdigest()==grade['source_pins'][normalized]
accepted=json.loads(accepted_raw)
tokenizer=accepted['tokenizer']
assert tokenizer['skip_special_tokens'] is True
tokenizer_raw=(root/tokenizer['path']).read_bytes()
assert hashlib.sha256(tokenizer_raw).hexdigest()==tokenizer['sha256']
from tokenizers import Tokenizer
decode=Tokenizer.from_str(tokenizer_raw.decode()).decode
literal=accepted['responses']
assert len(literal)==100
events=[json.loads(line) for line in (out/'tasks.jsonl').read_text().splitlines()]
dispatch,=[row for row in events if row['event']=='dispatch']
assert (dispatch['rows'],dispatch['tile4_stages'])==(128,168)
ready,=[row for row in events if row['event']=='ready']
assert (ready['capacity'],ready['chunk'])==(562,128)
assert ready['combined_admitted_bytes']<=1536*1024**2
cases=[row for row in events if row['event']=='task']
assert len(cases)==100 and len({row['id'] for row in cases})==100
for index,(new,previous,text) in enumerate(zip(cases,old,literal,strict=True)):
    assert (new['index'],new['id'])==(index,previous['caseID'])
    assert all(new[key]==previous['response'][key] for key in ('ids','sampled','history','evaluated','pending')),new['id']
    assert new['id']==text['id'] and new['ids']==text['visible_token_ids']
    decoded=decode(new['ids'],skip_special_tokens=True)
    assert decoded==text['visible_text']
    assert hashlib.sha256(decoded.encode()).hexdigest()==text['visible_text_sha256']
    reason=new['reason']
    if reason['kind']=='stop_token':
        assert previous['response']['reason']==2 and reason['token_id']==previous['response']['stopToken']
    else:
        assert reason['kind']=='output_limit' and previous['response']['reason']==1
    assert previous['resetHistory']==previous['resetEvaluated']==previous['resetPending']==0
assert events[-1]==dict(event='done',tasks=100,bitwise_decision_rows=True)
print(json.dumps(dict(passed=True,exact_original_completions=100,full_distribution_pair=True,
                      grading='reuse after exact archived token, stop and literal identity',
                      accepted_original_score=mac_grade['passed'],
                      maximum_allowed_drop=mac_grade['maximum_drop_percentage_points']),indent=2))
