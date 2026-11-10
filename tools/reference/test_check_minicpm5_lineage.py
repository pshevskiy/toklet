"""Portable retained-witness causal, coverage, native-origin and owner controls."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import check_minicpm5_lineage as reader
import minicpm5_parent as parent
import test_minicpm5_matrix_stream as fixtures


def converted():
    raw,*_=fixtures.fixture();stream,child,journal,control=fixtures.stream(raw)
    prefix,header,size,norms=control
    result=parent.authenticate(stream,prefix,header,size,hashlib.sha256(raw).hexdigest(),norms,io.StringIO())
    shard=('fixture',size,result['sha256'],prefix,header,norms)
    return stream.finish(size),journal.getvalue().encode(),shard,child,result


def owner():
    return dict(passed=True,owned_group_absent=True,source_inputs_unchanged=True,exit_code=0,failure=None,
                final_physical_peak_available=False,started_ns=1,completed_ns=1000000001,
                minimum_observed_free_bytes=3*1024**3,final_auditor_wait4_peak_rss_bytes=100,
                max_sampled_aggregate_rss_bytes=17,max_sampled_aggregate_physical_bytes=28,
                max_sampled_aggregate_lifetime_physical_bytes=39,sampled_observations=2,sample_errors=[])


def samples():
    process=lambda pid,a,b,c:dict(pid=pid,resident_size=a,phys_footprint=b,lifetime_max_phys_footprint=c)
    return [dict(observed_ns=2,processes=[process(1,10,20,30)]),
            dict(observed_ns=4,processes=[process(1,15,25,35),process(2,2,3,4)])]


class Witnesses(unittest.TestCase):
    def test_complete_row_and_hash_journal_matches_independent_child(self):
        result,journal,shard,child,_=converted()
        self.assertGreaterEqual(reader.validate_conversion(result,journal,shard,child),0)

    def test_single_tile_parent_hash_binds_but_multi_tile_membership_is_not_invented(self):
        result,journal,shard,child,source=converted()
        reader.bind_single_tile_sources(journal,source['tensors'])
        rows=[json.loads(line) for line in journal.splitlines()]
        single=next(row for row in rows if row['event']=='matrix_tile' and row['name']=='b.weight')
        single['parent_tile_sha256']='0'*64
        altered=b''.join(json.dumps(row).encode()+b'\n' for row in rows)
        with self.assertRaises(ValueError):reader.bind_single_tile_sources(altered,source['tensors'])
        rows=[json.loads(line) for line in journal.splitlines()]
        rows[0]['parent_tile_sha256']='0'*64
        altered=b''.join(json.dumps(row).encode()+b'\n' for row in rows)
        reader.bind_single_tile_sources(altered,source['tensors'])
        self.assertNotEqual(hashlib.sha256(altered).hexdigest(),hashlib.sha256(journal).hexdigest())

    def test_missing_duplicate_reordered_or_wrong_source_tiles_reject(self):
        result,journal,shard,child,_=converted();rows=[json.loads(line) for line in journal.splitlines()]
        variants=[rows[1:],rows+[rows[-1]],list(reversed(rows))]
        for change in [dict(first_row=1),dict(rows=True),dict(columns=128),dict(parent_offset=0),
                       dict(parent_bytes=1),dict(parent_tile_sha256='z'*64),dict(object_authenticated=True)]:
            changed=copy.deepcopy(rows);changed[0].update(change);variants.append(changed)
        for rows_ in variants:
            raw=b''.join(json.dumps(row).encode()+b'\n' for row in rows_)
            with self.assertRaises(ValueError):reader.validate_conversion(result,raw,shard,child)

    def test_code_scale_bias_digests_and_complete_matrix_aggregates_are_exact(self):
        result,journal,shard,child,_=converted()
        for key in ['weight','scales','biases']:
            for field,value in [('bytes',1),('mismatching_bytes',False),
                                ('reconstructed_sha256','0'*64),('expected_sha256','0'*64)]:
                rows=[json.loads(line) for line in journal.splitlines()];rows[0]['comparison'][key][field]=value
                raw=b''.join(json.dumps(row).encode()+b'\n' for row in rows)
                with self.assertRaises(ValueError):reader.validate_conversion(result,raw,shard,child)
        changed=copy.deepcopy(result);changed['matrices'][0]['comparison']['weight']['bytes']-=1
        with self.assertRaises(ValueError):reader.validate_conversion(changed,journal,shard,child)

    def test_conversion_flags_counts_and_typed_phase_clocks_cannot_be_promoted(self):
        result,journal,shard,child,_=converted()
        for change in [dict(passed=1),dict(object_authenticated=True),dict(matrix_count=True),
                       dict(tiles=2),dict(norm_tensors=0),dict(retained_payload_bytes=1),dict(extra=True)]:
            changed=dict(result,**change)
            with self.assertRaises(ValueError):reader.validate_conversion(changed,journal,shard,child)
        for value in [True,0,-1,1.0]:
            rows=[json.loads(line) for line in journal.splitlines()];rows[0]['completed_quantize_readback_ns']=value
            raw=b''.join(json.dumps(row).encode()+b'\n' for row in rows)
            with self.assertRaises(ValueError):reader.validate_conversion(result,raw,shard,child)

    def test_owner_limits_and_raw_sample_maxima_are_not_inferred(self):
        value=owner();reader.validate_owner(value)
        start=dict(pid=1,started_ns=1,command=['python'])
        reader.validate_samples(samples(),start,['python'],value,{2,3})
        for change in [dict(passed=1),dict(exit_code=False),dict(owned_group_absent=False),
                       dict(final_physical_peak_available=True),dict(started_ns=True),
                       dict(sample_errors=['missing process']),dict(sample_errors=None),
                       dict(final_auditor_wait4_peak_rss_bytes=reader.LIMIT+1),dict(minimum_observed_free_bytes=0)]:
            with self.assertRaises(ValueError):reader.validate_owner(dict(value,**change))
        for change in [dict(max_sampled_aggregate_rss_bytes=18),dict(sampled_observations=1)]:
            with self.assertRaises(ValueError):reader.validate_samples(samples(),start,['python'],dict(value,**change),{2,3})
        with self.assertRaises(ValueError):reader.validate_samples(samples(),dict(start,started_ns=True),['python'],value,{2,3})

    def test_sample_pid_identity_clock_types_duplicates_and_live_sum_reject(self):
        start=dict(pid=1,started_ns=1,command=['python'])
        for change in [dict(pid=4),dict(resident_size=True),dict(phys_footprint=reader.LIMIT+1)]:
            raw=samples();raw[0]['processes'][0].update(change)
            with self.assertRaises(ValueError):reader.validate_samples(raw,start,['python'],owner(),{2,3})
        for raw in [[],list(reversed(samples()))]:
            with self.assertRaises(ValueError):reader.validate_samples(raw,start,['python'],owner(),{2,3})
        raw=samples();raw[0]['processes']*=2
        with self.assertRaises(ValueError):reader.validate_samples(raw,start,['python'],owner(),{2,3})

    def test_exact_tls_command_and_complete_authenticated_source_are_required(self):
        result,journal,shard,child,source=converted();name,size,*_=shard
        command=['/usr/bin/curl','-q','--fail','--location','--proto','=https','--proto-redir','=https',
                 '--connect-timeout','20','--max-time','1200','--max-filesize',str(size),
                 '--no-progress-meter','-H','Accept-Encoding: identity','--dump-header','/tmp/fixture.headers',
                 '--output','-','--write-out','%{stderr}\nTOKLET_CURL_JSON%{json}\n',
                 f'https://huggingface.co/{parent.PARENT}/resolve/{parent.REVISION}/fixture?download=true&toklet_lineage='+('a'*32)]
        row=dict(passed=True,source_authentication_passed=True,name=name,pid=2,command=command,
                 started_ns=2,completed_ns=100,exit_code=0,error=None,http_code=200,size_download=size,
                 ssl_verify_result=0,final_wait4_peak_rss_bytes=100,payload_retained=False,result=source,conversion=result)
        reader.validate_transfer(row,shard,owner(),Path('/tmp'))
        for change in [dict(passed=1),dict(source_authentication_passed=1),dict(http_code=206),
                       dict(ssl_verify_result=False),dict(payload_retained=True),dict(completed_ns=2**62),
                       dict(size_download=size-1)]:
            with self.assertRaises(ValueError):reader.validate_transfer(dict(row,**change),shard,owner(),Path('/tmp'))
        changed=copy.deepcopy(row);changed['command'][5]='=http'
        with self.assertRaises(ValueError):reader.validate_transfer(changed,shard,owner(),Path('/tmp'))

    def test_runtime_pins_actual_native_images_and_endpoints_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);site=root/'.venv/lib/python3.12/site-packages';files={}
            for name in ['numpy/__init__.py','mlx/core.fixture.so','mlx/nn/__init__.py',
                         'mlx/lib/libmlx.dylib','mlx/lib/mlx.metallib']:
                p=site/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
                files[str(p)]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            imports={'numpy':str(site/'numpy/__init__.py'),'mlx.core':str(site/'mlx/core.fixture.so'),
                     'mlx.nn':str(site/'mlx/nn/__init__.py')}
            native={p:pin for p,pin in files.items() if Path(p).suffix in ['.so','.dylib']}
            origins=dict(modules=imports,non_system_native_images=native,system_native_images=['/usr/lib/libSystem.B.dylib'])
            external=dict(format='toklet-minicpm5-quant-runtime-v1',versions={'numpy':'2.3.2','mlx':'0.32.3'},files=files)
            admitted=dict(manifest_sha256='a'*64,files=len(files),versions=external['versions'],
                          pycache_prefix=str(root/'cache'),imports=imports,initial_origins=origins)
            reader.validate_runtime(admitted,origins,external,'a'*64,str(root/'cache'),root)
            for change in [dict(files=True),dict(manifest_sha256='b'*64),dict(pycache_prefix='/tmp/other')]:
                with self.assertRaises(ValueError):reader.validate_runtime(dict(admitted,**change),origins,external,'a'*64,str(root/'cache'),root)
            for change in [dict(non_system_native_images={}),dict(system_native_images=['/tmp/unadmitted.dylib'])]:
                with self.assertRaises(ValueError):reader.validate_runtime(admitted,dict(origins,**change),external,'a'*64,str(root/'cache'),root)
            (site/'mlx/core.fixture.so').write_bytes(b'changed')
            with self.assertRaises(ValueError):reader.validate_runtime(admitted,origins,external,'a'*64,str(root/'cache'),root)


if __name__=='__main__':unittest.main()
