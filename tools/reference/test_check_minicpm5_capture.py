"""Portable metadata/clock damage controls: stdlib only, no model or GPU."""
import copy
import unittest

import check_minicpm5_capture as checker
import minicpm5_checked as reference


def metrics():
    result=[]
    for case,spec in reference.CASES.items():
        for requested in [1,3,32]:
            chunk=min(requested,512-spec['position_offset'])
            for start in range(0,len(spec['tokens']),chunk):
                count=min(chunk,len(spec['tokens'])-start)
                memory=checker.expected_memory(requested,spec['position_offset'])
                timing={key:.000001 for key in ['command_creation_seconds','clock_calibration_seconds',
                        'commit_seconds','host_encode_seconds','wait_seconds','readback_seconds']}
                timing.update(gpu_start_seconds=1.,gpu_end_seconds=1.000001,command_count=1)
                result.append(dict(case=case,requested_chunk=requested,rows=count,
                    position_offset=spec['position_offset'],committed_before=start,
                    committed_after=start+count,pending_after=len(spec['tokens'])-start-count,
                    wall_nanoseconds=2000000,completed_timing=timing,session_memory=memory))
    return result


class CompletedMetrics(unittest.TestCase):
    def owned(self):return dict(started_ns=1,completed_ns=200000001)

    def test_complete_disjoint_chunks_and_ownership_ledger(self):
        result=checker.validate_chunk_metrics(metrics(),self.owned())
        self.assertEqual(result['completed_chunks'],51)
        self.assertEqual(result['completed_rows'],96)
        self.assertEqual(result['command_count'],51)
        self.assertEqual(result['gpu_timestamp_intervals_available'],51)
        self.assertEqual(result['completed_chunk_wall_nanoseconds'],102000000)
        self.assertFalse(result['ordinary_request_overhead_qualified'])
        self.assertEqual(len(checker.inventory()),2614)

    def test_owned_clocks_require_typed_positive_monotonic_interval(self):
        for change in [dict(started_ns=True),dict(started_ns=0),dict(started_ns=-1),
                       dict(started_ns=1.),dict(completed_ns=True),dict(completed_ns=1),
                       dict(completed_ns=0),dict(completed_ns=200000001.)]:
            owned=self.owned();owned.update(change)
            with self.assertRaises(ValueError):checker.validate_chunk_metrics(metrics(),owned)

    def test_inflated_disjoint_batch_sum_rejects_even_with_plausible_individual_durations(self):
        raw=metrics()
        for row in raw:row['wall_nanoseconds']=119000000000
        with self.assertRaisesRegex(ValueError,'exceed owned'):checker.validate_chunk_metrics(raw,self.owned())
        raw=metrics();owned=self.owned();owned['completed_ns']=102000000
        with self.assertRaisesRegex(ValueError,'exceed owned'):checker.validate_chunk_metrics(raw,owned)

    def test_missing_duplicate_state_and_typed_work_damage_rejects(self):
        for change in [dict(rows=True),dict(rows=1.),dict(committed_before=True),
                       dict(committed_after=999),dict(pending_after=-1),
                       dict(wall_nanoseconds=2000000.),dict(wall_nanoseconds=0)]:
            raw=metrics();raw[0].update(change)
            with self.assertRaises(ValueError):checker.validate_chunk_metrics(raw,self.owned())
        raw=metrics();raw.pop()
        with self.assertRaises(ValueError):checker.validate_chunk_metrics(raw,self.owned())
        raw=metrics();raw[-1]=copy.deepcopy(raw[0])
        with self.assertRaises(ValueError):checker.validate_chunk_metrics(raw,self.owned())

    def test_phase_clock_counter_and_memory_relabelling_rejects(self):
        for change in [dict(command_count=True),dict(command_count=0),dict(wait_seconds=float('nan')),
                       dict(host_encode_seconds=3.),dict(gpu_end_seconds=.5),dict(gpu_end_seconds=4.)]:
            raw=metrics();raw[0]['completed_timing'].update(change)
            with self.assertRaises(ValueError):checker.validate_chunk_metrics(raw,self.owned())
        for change in [dict(weights_bytes=True),dict(admitted_bytes=1),dict(admitted_bytes=6442450944),
                       dict(kv_bytes=-1),dict(scratch_bytes=2048.)]:
            raw=metrics();raw[0]['session_memory'].update(change)
            with self.assertRaises(ValueError):checker.validate_chunk_metrics(raw,self.owned())

    def test_capture_owners_are_shape_bound_even_when_resealed_totals_balance(self):
        self.assertEqual(checker.expected_memory(1,0), dict(weights_bytes=2674218150,
            kv_bytes=22021272,scratch_bytes=2786030,history_bytes=2048,logits_bytes=1044480,
            metadata_bytes=1464,metadata_allowance_bytes=67108864,admitted_bytes=2767182308))
        self.assertEqual(checker.expected_memory(32,0)['scratch_bytes'],80908138)
        self.assertEqual(checker.expected_memory(32,509),checker.expected_memory(3,509))
        self.assertEqual(checker.expected_memory(3,509)['admitted_bytes'],2750134120)
        for key in ['kv_bytes','scratch_bytes','history_bytes','logits_bytes',
                    'metadata_bytes','metadata_allowance_bytes']:
            raw=metrics();memory=raw[0]['session_memory'];memory[key]=0
            memory['admitted_bytes']=sum(v for k,v in memory.items() if k!='admitted_bytes')
            with self.assertRaisesRegex(ValueError,'owner ledger'):checker.validate_chunk_metrics(raw,self.owned())
        raw=metrics();raw[0]['session_memory']=checker.expected_memory(3,0)
        with self.assertRaisesRegex(ValueError,'owner ledger'):checker.validate_chunk_metrics(raw,self.owned())

    def test_disjoint_host_sum_rejects_but_gpu_overlap_is_not_added(self):
        raw=metrics();timing=raw[0]['completed_timing']
        for key in timing:
            if key.endswith('_seconds') and not key.startswith('gpu_'):timing[key]=.0004
        with self.assertRaisesRegex(ValueError,'Disjoint host'):checker.validate_chunk_metrics(raw,self.owned())
        raw=metrics();timing=raw[0]['completed_timing'];timing['wait_seconds']=.00199
        timing['gpu_end_seconds']=timing['gpu_start_seconds']+.00199
        self.assertEqual(checker.validate_chunk_metrics(raw,self.owned())['completed_chunks'],51)

    def test_disjoint_load_and_work_clocks_and_bounded_rounding(self):
        load=dict(load_seconds=.95,conversion_seconds=.5,compilation_seconds=.4)
        with self.assertRaisesRegex(ValueError,'Disjoint load'):checker.validate_load_timing(load,10**9,10**8)
        checker.validate_load_timing(load,10**9,50000000)
        for value in [0,1e-9,1.5e-9,2**20]:
            self.assertLessEqual(abs(checker.duration_ns(value)-value*10**9),.5)
        for value in [True,-1,2**20+1,float('inf'),float('nan'),10**1000]:
            with self.assertRaises(ValueError):checker.duration_ns(value)

    def test_only_exact_reviewed_eof_cleanup_can_reuse_the_old_producer_pin(self):
        current=reference.producer.producer_pins()
        source=(reference.producer.ROOT/'tools/reference/minicpm5.py').read_bytes()
        recorded=dict(current)
        recorded['tools/reference/minicpm5.py']='74557564fe62dd9836eb511f7db053bd598cd3d1e93d100450236a819e4167b9'
        self.assertTrue(reference.producer_pins_match(current,current,source))
        self.assertTrue(reference.producer_pins_match(recorded,current,source))
        for key in recorded:
            bad=dict(recorded);bad[key]='0'*64
            self.assertFalse(reference.producer_pins_match(bad,current,source))
        for bad_source in [source+b'\n',source+b'# comment\n',b'x'*65537]:
            self.assertFalse(reference.producer_pins_match(recorded,current,bad_source))
        bad=dict(current);bad['extra']='a'*64
        self.assertFalse(reference.producer_pins_match(recorded,bad,source))

    def test_typed_json_identity_and_finite_numbers(self):
        self.assertTrue(reference.same({'x':[0,1,1.]},{'x':[0,1,1.]}))
        for changed in [{'x':[False,1,1.]},{'x':[0,True,1.]},{'x':[0,1,1]}, {'x':[0,1,1.],'extra':0}]:
            self.assertFalse(reference.same(changed,{'x':[0,1,1.]}))
        for number in [True,False,float('nan'),float('inf'),'1']:
            self.assertFalse(reference.number(number))


if __name__=='__main__':unittest.main()
