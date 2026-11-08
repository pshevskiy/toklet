import copy
import unittest
import check_qwen35_metal_benchmark as bench


def samples(cohort):
    return [dict(cohort=cohort,iteration=i,warmup=i<3,
        engine='scalar_cpu' if (s==0)==((i+cohort)%2==0) else 'metal_projection_cpu_state',
        elapsed_ns=1000 if (s==0)==((i+cohort)%2==0) else 100,tokens=17,position_offset=37,
        work=bench.context.work(17),maximum_cpu_gpu_logit_gate_ratio=.1)
        for i in range(13) for s in range(2)]


def environment():
    item=lambda command,output:dict(command=command,exit_code=0,output=output)
    start=dict(hardware=item(['/usr/sbin/sysctl','-n','hw.model','machdep.cpu.brand_string','hw.memsize'],'test machine'),
        os=item(['/usr/bin/sw_vers'],'test OS'),configuration=item(['/usr/bin/pmset','-g','custom'],'test configuration'),
        power=item(['/usr/bin/pmset','-g','batt'],"Now drawing from 'AC Power'\n"),memory_pressure=item(['/usr/bin/memory_pressure','-Q'],'System-wide memory free percentage: 50%'),
        thermal=item(['/usr/bin/pmset','-g','therm'],'Note: No thermal warning level has been recorded\nNote: No performance warning level has been recorded\nNote: No CPU power status has been recorded\n'))
    periodic={k:copy.deepcopy(start[k]) for k in ['power','memory_pressure','thermal']}
    return dict(start=start,end=copy.deepcopy(start),periodic=[dict(elapsed_seconds=t,observations=copy.deepcopy(periodic)) for t in [30.1,60.2,90.3]],
        sampling='External30s CLI snapshots, not engine instrumentation, energy or continuous thermal telemetry')


class Samples(unittest.TestCase):
    def test_independent_cohorts_preserve_alternation_counts_and_dispersion(self):
        for cohort in [0,1]:
            summary=bench.samples(samples(cohort),cohort)
            self.assertAlmostEqual(summary['improvement_fraction'],.9)
            self.assertEqual(summary['engines']['scalar_cpu']['mad_ns'],0)
            self.assertEqual(len(summary['engines']['metal_projection_cpu_state']['samples_ns']),10)

    def test_missing_duplicate_reordered_typed_or_wrong_work_rejected(self):
        original=samples(0);cases=[original[:-1],list(reversed(original)),original[:-1]+original[:1]]
        for key,value in [('iteration',True),('cohort',True),('elapsed_ns',True),('elapsed_ns',0),('tokens',16),('warmup',1),('maximum_cpu_gpu_logit_gate_ratio',True),('maximum_cpu_gpu_logit_gate_ratio',1.01),('work',bench.context.work(16))]:
            damaged=copy.deepcopy(original);damaged[0][key]=value;cases.append(damaged)
        for rows in cases:
            with self.subTest(first=rows[:1]),self.assertRaises(ValueError):bench.samples(rows,0)


class Conditions(unittest.TestCase):
    def test_complete_external_endpoints_and_periodic_observations(self):
        self.assertTrue(bench.conditions(environment(),100))

    def test_battery_warning_error_or_missing_pressure_not_qualified(self):
        for field,value in [('power',"Now drawing from 'Battery Power'\n"),('thermal','Thermal warning level: 2'),('memory_pressure','unavailable')]:
            data=environment();data['periodic'][0]['observations'][field]['output']=value
            with self.subTest(field=field):self.assertFalse(bench.conditions(data,100))
        for field in ['power','thermal','memory_pressure']:
            data=environment();data['start'][field]['exit_code']=False
            with self.subTest(field=field):self.assertFalse(bench.conditions(data,100))

    def test_configuration_drift_or_missing_typed_timeline_rejected(self):
        for field in ['hardware','os','configuration']:
            data=environment();data['end'][field]['output']='other'
            with self.subTest(field=field),self.assertRaises(ValueError):bench.conditions(data,100)
        for value in [True,-1,float('nan')]:
            data=environment();data['periodic'][0]['elapsed_seconds']=value
            with self.subTest(value=value),self.assertRaises(ValueError):bench.conditions(data,100)
        data=environment();data['periodic']*=2
        with self.assertRaises(ValueError):bench.conditions(data,100)

    def test_removed_observations_or_unobserved_tail_rejected(self):
        for rows in [[],environment()['periodic'][:1],environment()['periodic'][2:]]:
            data=environment();data['periodic']=rows
            with self.subTest(rows=rows),self.assertRaises(ValueError):bench.conditions(data,100)
        with self.assertRaises(ValueError):bench.conditions(environment(),151)

    def test_wrong_sampling_inventory_or_command_rejected(self):
        data=environment();data['sampling']='Only endpoints'
        with self.assertRaises(ValueError):bench.conditions(data,100)
        data=environment();del data['end']['power']
        with self.assertRaises(ValueError):bench.conditions(data,100)
        data=environment();del data['periodic'][0]['observations']['thermal']
        with self.assertRaises(ValueError):bench.conditions(data,100)
        data=environment();data['periodic'][0]['observations']['power']['command']=['test']
        with self.assertRaises(ValueError):bench.conditions(data,100)

    def test_out_of_run_timestamp_or_invalid_duration_rejected(self):
        for value in [100,101]:
            data=environment();data['periodic'][-1]['elapsed_seconds']=value
            with self.subTest(value=value),self.assertRaises(ValueError):bench.conditions(data,100)
        for value in [True,0,-1,float('nan')]:
            with self.subTest(value=value),self.assertRaises(ValueError):bench.conditions(environment(),value)

    def test_unparseable_duplicate_or_out_of_range_pressure_not_qualified(self):
        for value in ['System-wide memory free percentage: NaN%','System-wide memory free percentage: 101%',
            'System-wide memory free percentage: 50%\nSystem-wide memory free percentage: 50%']:
            data=environment();data['periodic'][0]['observations']['memory_pressure']['output']=value
            with self.subTest(value=value):self.assertFalse(bench.conditions(data,100))


if __name__=='__main__':unittest.main()
