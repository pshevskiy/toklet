import copy
import unittest
import metal_trace_analysis as analysis


def fixture():
    timing={'command_creation_seconds':1e-6,'clock_calibration_seconds':0.,'commit_seconds':1e-6,'host_encode_seconds':.001,'wait_seconds':.002,'gpu_start_seconds':1.,'gpu_end_seconds':1.002,'readback_seconds':1e-6,'command_count':1}
    header={'run':0,'operation':'decode','start':{'cpu_nanoseconds':1000000,'gpu_ticks':1000},'end':{'cpu_nanoseconds':3000000,'gpu_ticks':3000},'baseline_timing':timing,'profile_timing':{**timing,'clock_calibration_seconds':1e-6},'logits_sha256':'0'*64,'greedy_id':47,'host_clock_sample_bracket_seconds':1e-6}
    events=[]
    for i,name in enumerate(analysis.topology(2)):
        begin=1000+i*20;end=begin+10
        events.append({'sample':{'name':name,'host_begin_seconds':i*1e-6,'host_end_seconds':(i+1)*1e-6,'gpu_begin_ticks':begin,'gpu_end_ticks':end,'gpu_begin_seconds':(begin-1000)*1e-6,'gpu_end_seconds':(end-1000)*1e-6}})
    return header,events


class Analysis(unittest.TestCase):
    def test_interval_union_preserves_overlap_and_gaps(self):
        self.assertEqual(analysis.length([(0,2),(1,3)]),3)
        self.assertEqual(analysis.length([(0,1),(2,3)]),2)
        with self.assertRaises(ValueError):analysis.length([(2,1)])
    def test_manual_clock_and_stage_accounting(self):
        header,events=fixture();result=analysis.measure(header,events,2)
        self.assertEqual(result['events'],82)
        self.assertAlmostEqual(result['custom_encoder_union_seconds'],52*10e-6)
        self.assertAlmostEqual(result['marker_union_seconds'],30*10e-6)
        self.assertAlmostEqual(result['mps_enclosed_union_seconds'],15*10e-6)
        self.assertAlmostEqual(result['unattributed_transition_seconds'],660e-6)
        self.assertAlmostEqual(result['overlapping_attribution_seconds'],0.)
    def test_bad_conversion_topology_and_samples_are_rejected(self):
        for key,value in [('gpu_begin_seconds',.5),('gpu_begin_ticks',0),('gpu_end_ticks',(1<<64)-1),('name','wrong')]:
            header,events=fixture();events[0]['sample'][key]=value
            with self.assertRaises(ValueError):analysis.measure(header,events,2)
    def test_calibration_is_not_production_readback(self):
        header,events=fixture();header['baseline_timing']['clock_calibration_seconds']=1e-6
        with self.assertRaises(ValueError):analysis.measure(header,events,2)


if __name__=='__main__':unittest.main()
