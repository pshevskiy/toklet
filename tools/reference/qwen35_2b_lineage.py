"""Authenticate and compare the selected text parent, streaming GGUF body away.

Run only under the external setup guard. This is prospective text equivalence,
not publisher provenance, model quality, full graph or performance qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import ssl
import urllib.parse
import urllib.request

import granite_convert as generic
import granite_quant_runtime as runtime
import ministral3_convert as files
import qwen35_2b_layout as layout
import qwen35_2b_parent as parent

need = layout.need


def sources():
    folder = Path(__file__).resolve().parent
    fixture = folder.parents[1]/'tests/fixtures/qwen35-2b-source-v1'
    return sorted({*generic.sources(), *files.sources(), *(folder/name for name in (
        'ministral3_convert_audit.py', 'test_ministral3_convert_audit.py',
        'qwen35_2b_layout.py', 'test_qwen35_2b_layout.py',
        'qwen35_2b_parent.py', 'test_qwen35_2b_parent.py',
        'qwen35_2b_lineage.py', 'test_qwen35_2b_lineage.py',
        'qwen35_2b_convert.py', 'test_qwen35_2b_convert.py',
        'qwen35_2b_audit.py', 'test_qwen35_2b_audit.py')),
        *(fixture/name for name in ('manifest.json','source-config.json','source-header.bin','parent-directory.bin'))})


class Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, url):
        need(urllib.parse.urlsplit(url).scheme == 'https', 'HTTPS-only parent redirects required')
        return super().redirect_request(request, file, code, message, headers, url)


def verified_context(context):
    need(context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is True,
         'Certificate and hostname verification must remain enabled')
    return context


def load_ca(path, expected_sha256):
    """Capture the external CA input before constructing a verified context."""
    need(path.is_absolute(), 'Absolute external CA path required')
    expected_sha256 = layout.identifier(expected_sha256)
    raw, identity = files.capture_named(path, parent.MAX_CA_BYTES, expected_sha256)
    need(0 < len(raw) <= parent.MAX_CA_BYTES, 'Bounded nonempty external CA PEM required')
    # cadata avoids loading an unpinned certificate file or modifying TLS globals.
    context = verified_context(ssl.create_default_context(cadata=raw.decode('ascii')))
    record = dict(path=str(path), bytes=len(raw), sha256=expected_sha256,
                  before_identity=list(identity), after_identity=list(identity))
    verify_ca(record)
    return context, record


def verify_ca(record):
    raw, identity = files.capture_named(Path(record['path']), parent.MAX_CA_BYTES, record['sha256'])
    need(len(raw) == record['bytes'] and list(identity) == record['before_identity']
         and list(identity) == record['after_identity'], 'External CA descriptor/path endpoint changed')
    return record


def open_parent(context):
    verified_context(context)
    p = parent.PARENT
    url = 'https://huggingface.co/'+p['repository']+'/resolve/'+p['revision']+'/'+p['file']
    request = urllib.request.Request(url, headers={'Accept-Encoding':'identity', 'User-Agent':'Toklet-parent-lineage/1'})
    response = urllib.request.build_opener(Redirects(), urllib.request.HTTPSHandler(context=context)).open(request, timeout=30)
    try:
        need(response.status == 200 and response.headers.get('Content-Encoding', 'identity') == 'identity',
             'Complete unencoded parent response required')
        length = response.headers.get('Content-Length')
        need(length is not None and int(length) == p['bytes'], 'Exact declared full parent extent required')
        need(urllib.parse.urlsplit(response.geturl()).scheme == 'https', 'HTTPS parent endpoint required')
        return response
    except BaseException:
        response.close()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('environment-root', 'runtime-manifest', 'runtime-sha256', 'source-dir',
                 'protocol', 'protocol-sha256', 'fresh-evidence', 'ca-file', 'ca-sha256'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    evidence, source_dir = Path(args.fresh_evidence), Path(args.source_dir)
    need(evidence.is_absolute() and source_dir.is_absolute() and not evidence.exists()
         and not evidence.is_symlink(), 'Fresh absolute evidence and source paths required')
    evidence.mkdir(mode=0o700)
    result = dict(format='toklet-qwen35-2b-parent-lineage-v1', passed=False,
                  protocol_sha256=args.protocol_sha256, historical_publisher_lineage_proved=False,
                  A_log_bitwise_reconstruction_claimed=False,
                  model_numerics_quality_fit_performance_adoption_qualified=False)
    journal_path = evidence/'records.jsonl'
    records = 0
    try:
        raw = layout.saved.capture(Path(args.protocol), 256*1024, layout.identifier(args.protocol_sha256))
        protocol = layout.saved.loads(raw)
        layout.saved.same(protocol, parent.protocol(), 'Frozen prospective parent protocol differs')
        result['protocol'] = protocol
        context, ca = load_ca(Path(args.ca_file), args.ca_sha256)
        result['transport'] = dict(https_only=True, certificate_verification='CERT_REQUIRED',
            hostname_verification=True, TLS_verified_for_completed_stream=False, ca=ca)
        with layout.saved.SavedFile(source_dir/'config.json', layout.SOURCE_CONFIG_BYTES) as config, \
                layout.saved.SavedFile(source_dir/'model.safetensors', layout.SOURCE_BYTES) as source:
            config_raw = config.read(layout.SOURCE_CONFIG_BYTES)
            layout.converted_config(config_raw)
            header_raw = source.read(layout.SOURCE_HEADER_BYTES, 8)
            spans = layout.source_spans(source.read(8), header_raw, source.before[2])
            selected = {row['name']:row for row in spans if row['selected']}
            need(source.digest() == layout.SOURCE_SHA256, 'Whole selected source SHA256 differs before parent stream')
            result['selected_source'] = dict(layout.SOURCE, path=str(source.path), bytes=source.before[2],
                before_identity=source.before, whole_SHA256_authenticated_before=True,
                whole_SHA256_authenticated_after=False)
            runtime.bind_environment(args.environment_root)
            admitted, endpoint = runtime.admit(Path(args.runtime_manifest), args.runtime_sha256, sources())
            import numpy as np
            finite, rates = parent.numerical_callbacks(np)
            result['admitted_runtime'] = admitted
            with journal_path.open('x') as journal:
                def event(value):
                    nonlocal records
                    journal.write(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')
                    journal.flush()
                    records += 1
                verify_ca(ca)
                with open_parent(context) as stream:
                    reports, receipt = parent.authenticate_stream(stream,
                        lambda count, offset: source.read(count, 8+len(header_raw)+offset),
                        selected, finite, rates, event)
                result.update(parent=receipt, tensors=reports)
                result['transport']['TLS_verified_for_completed_stream'] = True
                need(source.digest() == layout.SOURCE_SHA256, 'Whole selected source SHA256 differs after parent stream')
                source.check(); config.check()
                result['selected_source'].update(after_identity=source.check(), whole_SHA256_authenticated_after=True)
                result['endpoint_origins'] = endpoint()
                verify_ca(ca)
                os.fsync(journal.fileno())
            with layout.saved.SavedFile(journal_path) as journal:
                result['journal'] = dict(path=str(journal_path), bytes=journal.before[2], sha256=journal.digest(), records=records)
            result['passed'] = all(row['passed'] for row in reports)
            if result['passed']:
                parent.validate_certificate(result, args.protocol_sha256)
    except BaseException as error:
        result.update(passed=False, error_type=type(error).__name__, error=str(error)[:2048])
    finally:
        result['journal_records_observed'] = records
        files.write_exclusive(evidence/'manifest.json', (json.dumps(result, indent=2, allow_nan=False)+'\n').encode())
    print(json.dumps(dict(passed=result['passed'], journal_records=records,
                          parent_whole_SHA256_authenticated=result.get('parent',{}).get('whole_SHA256_authenticated',False),
                          historical_publisher_lineage_proved=False)))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
