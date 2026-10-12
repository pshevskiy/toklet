"""Independent manual/frozen gates and projection failure-policy checks."""
import copy
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import rust_metal_projection as projection


def fixture(inputs, anchors):
    cases=[]
    for item in inputs:
        rows,inputs_count,outputs=item['rows'],item['inputs'],item['outputs']
        count=rows*outputs;groups=math.ceil(count/32);storage_rows=math.ceil(groups*32/outputs)
        strides=[math.ceil(inputs_count*2/16)*8]*2+[math.ceil(outputs*4/16)*4,math.ceil(outputs*2/16)*8]
        a=(rows*strides[0]+7)*2;b=(outputs*strides[1]+7)*2;bias=(outputs+7)*2
        c=(storage_rows*strides[2]+7)*4;out=(storage_rows*strides[3]+7)*2
        bits=anchors[item['id']]['oracle_bits']
        floats=struct.unpack(f'<{count}e',struct.pack(f'<{count}H',*bits))
        fp32=list(struct.unpack(f'<{count}I',struct.pack(f'<{count}f',*floats)))
        cases.append({'id':item['id'],'rows_count':rows,'inputs':inputs_count,'outputs':outputs,
                      'threadgroup_width':32,'threadgroups':groups,'storage_rows':storage_rows,
                      'input_stride':strides[0],'weight_stride':strides[1],'fp32_stride':strides[2],'output_stride':strides[3],
                      'shared_buffer_bytes':a+b+bias+c+out,'host_baseline_readback_bytes':2*(a+b+bias)+c+out,
                      'admitted_bytes':3*(a+b+bias)+2*(c+out)+projection.rms.probe.ALLOWANCE,
                      'rows':[{'phase':'warmup' if i<3 else 'sample','repetition':i+1 if i<3 else i-2,
                               'command_completed':True,'inputs_and_guards_unchanged':True,'encode_commit_wait_seconds':.001,
                               'output_bits':bits.copy(),'fp32_bits':fp32.copy()} for i in range(13)]})
    return {'event':'idle_result','passed':True,'result':{'operation':'mps_fp16_inputs_fp32_bias_and_result_fp16_store','cases':cases,
            'metadata':{'unified_memory':True,'language_version':'2.3','math_mode':'safe','math_functions':'precise','minimum_macos_major':15,'mps_device_supported':True,
                        'execution_width':32,'pipeline_max_threads':1024,'bias_execution_width':32,'bias_max_threads':1024,'device_max_threads':[1024]*3,
                        'maximum_admitted_bytes':max(c['admitted_bytes'] for c in cases)}}}


class Projection(unittest.TestCase):
    def test_independent_bias_rounding_cancellation_and_frozen_source_gates(self):
        inputs,anchors,overflow=projection.cases()
        self.assertEqual(len(inputs),13)
        self.assertEqual(anchors['manual.bias_rounding']['oracle_bits'],[0x3c01])
        self.assertEqual(anchors['manual.finite_cancellation']['oracle_bits'],[0x7bff])
        self.assertEqual(overflow['weight_bits'],[0x4000])
        event=fixture(inputs,anchors);projection.validate(event,inputs,anchors)
        self.assertTrue(all(row['qualification']['single_half_store'] for c in event['result']['cases'] for row in c['rows']))
        bad=fixture(inputs,anchors)
        anchors[inputs[0]['id']]['source'][0]+=10
        with self.assertRaisesRegex(ValueError,'Frozen projection gate'):
            projection.validate(bad,inputs,anchors)

    def test_wrong_cast_bias_order_completion_guard_and_admission_are_rejected(self):
        all_inputs,anchors,_=projection.cases();inputs=all_inputs[-3:-2]
        for kind in ('cast','bias_round','completed','guard','stride','groups','storage','shared','host','budget','repetition','nan'):
            event=fixture(inputs,anchors);case=event['result']['cases'][0];row=case['rows'][-1]
            if kind=='cast':row['output_bits'][0]^=1
            elif kind=='bias_round':
                row['output_bits'][0]=0x3c00;row['fp32_bits'][0]=0x3f800000
            elif kind=='completed':row['command_completed']=False
            elif kind=='guard':row['inputs_and_guards_unchanged']=False
            elif kind=='stride':case['input_stride']=0
            elif kind=='groups':case['threadgroups']+=1
            elif kind=='storage':case['storage_rows']+=1
            elif kind=='shared':case['shared_buffer_bytes']-=1
            elif kind=='host':case['host_baseline_readback_bytes']-=1
            elif kind=='budget':case['admitted_bytes']-=1
            elif kind=='repetition':row['repetition']=9
            elif kind=='nan':row['fp32_bits'][0]=0x7fc00000
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                projection.validate(event,inputs,anchors)

    def test_completed_overflow_requires_discard_and_structured_failure(self):
        good={'passed':False,'error':{'stage':'validation','message':'completed projection nonfinite; result discarded'}}
        self.assertIs(projection.overflow_rejected(good),False)
        with self.assertRaises(ValueError):projection.overflow_rejected({**good,'passed':True})
        with self.assertRaises(ValueError):projection.overflow_rejected({**good,'error':{'stage':'admission','message':good['error']['message']}})

    def test_inputs_output_build_aliases_and_protected_roots_fail_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'result.json';build=root/'build'
            with patch.object(projection,'cases') as read:
                for inputs in (output,build,build/'inputs',root):
                    with self.assertRaises(ValueError):projection.run(output,build,inputs)
                for source in (projection.rms.FP16,projection.rms.FP32):
                    with self.assertRaises(ValueError):projection.run(source/'new.json',build,root/'inputs')
                read.assert_not_called()

if __name__=='__main__':unittest.main()
