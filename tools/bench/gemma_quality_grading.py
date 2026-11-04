"""Gemma collection adapter to the unchanged qualified isolated task grader."""
from pathlib import Path
import hashlib
import json
import check_gemma_quality as inputs
import gemma_task_results as results
import task_cases
import paired_task_grading as isolated
from grading_support import require, verify_file, write_json
from llama3_quality_grading import bind

ROOT = inputs.ROOT
MANIFEST_SHA = 'c4e6fc5cce765a30c43c9ed2d3ee4bfda58a4b23bbfc3284e2160431e6fb6506'


def read_json(path, limit=16*1024**2):
    with Path(path).open('rb') as file:
        raw = file.read(limit+1)
    require(len(raw) <= limit, 'Bounded grading evidence required')
    return json.loads(raw), raw


def run(collection_file, expected_sha, output):
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT/'artifacts'), 'Fresh artifact destination required')
    output.mkdir()
    record = dict(format='toklet-gemma-full-task-grading-v1', completed=False, commands=[], runs=[], outcomes={},
                  started_by_this_task=False, restored_stopped=False, pipe_lifetime_finished=False,
                  collection_sha256=expected_sha, manifest_sha256=MANIFEST_SHA,
                  generated_code_runs_on_host=False, unresolved_case_ids=[],
                  scope='Literal source-test outcomes; qualified isolation unchanged; no timing acceptance')
    bindings = {}
    workers = []

    def guard():
        for name, pin in bindings.items():
            verify_file(ROOT/name, pin)
        results.require_idle(workers)

    try:
        bind(bindings, collection_file, expected_sha)
        collection, _ = read_json(collection_file)
        require(collection['format'] == 'toklet-gemma-paired-task-collection-v1'
                and collection['completed'] is True and collection['cases'] == 100
                and collection['manifest_sha256'] == MANIFEST_SHA, 'Complete Gemma paired collection required')
        folder = ROOT/collection['input_directory']
        manifest, gemma_cases, _, _ = inputs.structure(folder, MANIFEST_SHA)
        bind(bindings, folder/'manifest.json', MANIFEST_SHA)
        for name, spec in manifest['assets'].items():
            bind(bindings, folder/name, spec['sha256'])
        canonical = results.canonical_grading_cases(gemma_cases)
        # Canonical prompts are used only by the qualified grader's schema;
        # literal scoring fields match exactly. Gemma inference remains separate.
        canonical_manifest = task_cases.prepare.CONTRACT/'manifest.json'
        canonical_policy, _ = read_json(canonical_manifest)
        bind(bindings, canonical_manifest)
        bind(bindings, ROOT/canonical_policy['quality']['tasks_file'], task_cases.TASK_SHA)
        for role in ['original', 'native']:
            item = collection['outputs'][role]
            wrapper_path = ROOT/item['worker_receipt']
            bind(bindings, wrapper_path, item['worker_receipt_sha256'])
            worker, _ = read_json(wrapper_path)
            workers.append(worker)
        record['exited_owned_model_pids'] = results.require_idle(workers)
        admitted = {}
        for (role, _), worker in zip([('original', 'baseline'), ('native', 'native')], workers, strict=True):
            item = collection['outputs'][role]
            journal = ROOT/item['journal']
            bind(bindings, journal, item['journal_sha256'])
            with journal.open('rb') as file:
                raw = file.read(16*1024**2+1)
            require(len(raw) <= 16*1024**2, 'Bounded task journal required')
            require(worker['tasks_sha256'] == item['journal_sha256'] == hashlib.sha256(raw).hexdigest(),
                    'Journal is not the exited worker output')
            admitted[role] = results.collection(gemma_cases, [json.loads(line) for line in raw.splitlines()], role)
            for name, sha in item['evidence_bindings'].items():
                bind(bindings, ROOT/name, sha)
        decoded = {}
        for role, profile in [('original', 'baseline'), ('native', 'native')]:
            normalized = admitted[role]
            tokenizer_path = ROOT/collection['tokenizer_directory']/'tokenizer.json'
            spec = manifest['model_assets']['tokenizer.json']
            require(spec['sha256'] == manifest['unquantized_baseline']['tokenizer_sha256'],
                    'Original and candidate tokenizer identities differ')
            bind(bindings, tokenizer_path, spec['sha256'])
            with tokenizer_path.open('rb') as file:
                tokenizer_bytes = file.read(64*1024**2+1)
            require(len(tokenizer_bytes) <= 64*1024**2, 'Bounded tokenizer required')
            decoded[profile] = results.decode(gemma_cases, normalized, tokenizer_bytes, spec)
            write_json(output/(profile+'-decoded.json'), decoded[profile])
            bind(bindings, output/(profile+'-decoded.json'))
        record['exited_owned_model_pids'] = results.require_idle(workers)
        for path in [Path(__file__), Path(results.__file__), Path(inputs.__file__), Path(isolated.__file__),
                     ROOT/'tools/bench/grading_support.py', ROOT/'tools/bench/task_cases.py']:
            bind(bindings, path)
        import tokenizers
        package = Path(tokenizers.__file__).resolve().parent
        for path in package.rglob('*'):
            if path.is_file() and path.suffix in ['.py', '.so', '.dylib'] and '__pycache__' not in path.parts:
                bind(bindings, path)
        record['inference_manifest_sha256'] = MANIFEST_SHA
        record['canonical_grading_payload_sha256'] = task_cases.TASK_SHA
        guard()
    except BaseException as error:
        record['failure'] = dict(type=type(error).__name__, message=str(error)[:4096])
        record['source_and_evidence_bindings'] = bindings
        write_json(output/'result.json', record)
        return record
    return isolated.grade_paired_cases(canonical, decoded, guard, output, record, bindings,
                                      lambda outcomes: results.decision(canonical, outcomes))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection', type=Path)
    parser.add_argument('collection_sha256')
    parser.add_argument('fresh_output', type=Path)
    args = parser.parse_args()
    report = run(args.collection, args.collection_sha256, args.fresh_output)
    print(json.dumps({k: report.get(k) for k in ['completed', 'decision', 'failure', 'unresolved_case_ids']}, indent=2))
    raise SystemExit(0 if report['completed'] else 1)
