import copy
import unittest

from o2_gemma_workloads import NAMES, paired_report, sustained_report, validate_row


class WorkloadReports(unittest.TestCase):
    def row(self, name='reset', mode='observed', trial=3):
        reuse, prefill = {'reference': (0,15),'edited_reference':(0,15),'reset': (0,15),'append': (14,1),'fully_cached': (15,0),
                          'pending': (15,1),'edited_prefix': (0,15),'sustained': (0,15)}[name]
        maximum = 3 if name == 'pending' else 4
        # Known independent timing: 1s prefill, 3s decode, .1s selection, 5s total.
        # Legacy takes 4s, observed 5s: the report must preserve the 25% cost.
        duration = 5 if mode == 'observed' else 4
        start = 10_000_000_000 + trial*5_000_000_000
        ids = [5279,529,7001] if name == 'pending' else [818,5279,529,7001]
        prompt = list(range(15))
        edited = prompt[:-1]+[prompt[2]]
        operation, tokens = {'append':(1,prompt[-1:]),'fully_cached':(0,prompt),'pending':(2,[]),
                             'edited_prefix':(0,edited),'edited_reference':(3,edited)}.get(name,(3,prompt))
        before = {'append':(14,14,0),'fully_cached':(15,15,0),'pending':(16,15,1),'reference':(0,0,0)}.get(name,(19,18,1))
        if name == 'sustained' and trial == 0:before=(0,0,0)
        row = dict(kind='request',workload=name,mode=mode,trial=trial,warmup=trial<3,
            submittedUptimeNanoseconds=start,completedUptimeNanoseconds=start+duration*1_000_000_000,
            callbackUptimeNanoseconds=[start+(i+1)*800_000_000 for i in range(maximum)],
            operation=operation,input=tokens,maximum=maximum,ids=ids,reason=1,status=0,
            before=dict(zip(('history','committed','pending'),before)),after=dict(history=19,committed=18,pending=1),
            footprint=dict(physicalBytes=100,lifetimePeakPhysicalBytes=200),thermalState=0)
        if mode == 'observed':
            row['native'] = dict(sessionInstance=1,reusedTokens=reuse,prefillTokens=prefill,
                sampledTokens=maximum,emittedTokens=maximum,firstTokenSeconds=.8,
                prefillSeconds=1 if prefill else 0,decodeSeconds=3,samplingSeconds=.1,totalSeconds=4.5)
            if prefill:
                row['native']['prefillTokensPerSecond'] = prefill
        return row

    def pairs(self):
        return [self.row('reference'),self.row('edited_reference')] + [self.row(name,mode,trial)
            for trial in range(13) for name in NAMES for mode in ('legacy','observed')]

    def test_known_phase_and_cost(self):
        report = paired_report(self.pairs())
        self.assertEqual(report['append']['measured_per_mode'],10)
        self.assertEqual(report['append']['native_phase_median_seconds']['decodeSeconds'],3)
        self.assertEqual(report['reset']['paired_overhead_median_pct']['total_ms'],25)
        self.assertEqual(report['fully_cached']['native_phase_median_seconds']['prefillSeconds'],0)

    def test_incomplete_work_reuse_or_clock_rejected(self):
        rows = self.pairs()
        bad = copy.deepcopy(rows); bad.pop()
        with self.assertRaises(ValueError): paired_report(bad)
        for change in (
            lambda r: r.update(after=dict(history=18,committed=18,pending=0)),
            lambda r: r['native'].update(prefillTokens=14),
            lambda r: r.update(callbackUptimeNanoseconds=[1]*4),
            lambda r: r['native'].update(totalSeconds=1000),
            lambda r: r.update(status=1),
        ):
            bad = copy.deepcopy(rows); change(bad[-1])
            with self.assertRaises(ValueError): paired_report(bad)

    def test_unknown_zero_work_and_nonfinite(self):
        row = self.row('fully_cached')
        validate_row(row)
        row['native']['prefillTokensPerSecond'] = 0
        with self.assertRaises(ValueError): validate_row(row)
        for key in ('decodeSeconds','totalSeconds'):
            row = self.row(); row['native'][key] = float('nan')
            with self.assertRaises(ValueError): validate_row(row)
        row = self.row(mode='legacy'); row['native'] = {}
        with self.assertRaises(ValueError): validate_row(row)

    def test_matching_but_impossible_setup_is_rejected(self):
        rows = self.pairs()
        for row in rows:
            if row['workload'] == 'append' and row['trial'] == 12:
                row['before'] = dict(history=15,committed=15,pending=0)
        with self.assertRaises(ValueError):paired_report(rows)

    def test_sustained_duration_and_duty(self):
        rows = [self.row('sustained',trial=i) for i in range(123)]
        report = sustained_report(rows)
        self.assertEqual(report['wall_seconds'],600)
        self.assertEqual(report['measured_requests'],120)
        self.assertFalse(report['tail_percentiles_qualified'])
        with self.assertRaises(ValueError): sustained_report(rows[:-1])
        bad = copy.deepcopy(rows)
        for row in bad[63:]:
            row['submittedUptimeNanoseconds'] += 100_000_000_000
            row['completedUptimeNanoseconds'] += 100_000_000_000
            row['callbackUptimeNanoseconds'] = [x+100_000_000_000 for x in row['callbackUptimeNanoseconds']]
        with self.assertRaises(ValueError): sustained_report(bad)

    def test_sustained_state_and_memory(self):
        for change in (
            lambda r: r.update(ids=[1,2,3,4]),
            lambda r: r['footprint'].update(lifetimePeakPhysicalBytes=7*1024**3),
            lambda r: r['native'].update(reusedTokens=1),
            lambda r: r.update(warmup=True),
            lambda r: r.update(before=dict(history=0,committed=0,pending=0)),
            lambda r: r['native'].update(sampledTokens=5),
        ):
            rows = [self.row('sustained',trial=i) for i in range(123)]
            change(rows[-1])
            with self.assertRaises(ValueError): sustained_report(rows)


if __name__ == '__main__':
    unittest.main()
