"""End-to-end gates, direct GPU propagation and fixed greedy policy validation."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import rust_metal_decoder as decoder
from test_rust_metal_layer import fixture as layer_fixture


def fixture(item,anchor):
    n=item['rows'];frames=[];incoming=anchor['embedding']['bits']
    for index,a in enumerate(anchor['layers']):
        view=layer_fixture({**item,'id':f'{item["id"]}.layers.{index}.layer','input_bits':incoming},a)['result'];frames.append({'half':view['first_half_intermediates'],'single':view['first_fp32_intermediates']});incoming=frames[-1]['half'][17]
    def layout(r,c,size):return {'rows':r,'columns':c,'stride':c,'elements':r*c+7,'bytes':(r*c+7)*size}
    hl=view['half_layouts'];fl=view['float_layouts'];wl=view['input_layouts'][1:];outer=[layout(64,32,2),layout(1,32,2),layout(1,64,2),hl[0],hl[0],layout(n,64,4)]
    shared=2*sum(l['bytes']for group in(wl,hl,fl)for l in group)+sum(l['bytes']for l in outer)+(n+7)*4
    m=copy.deepcopy(view['metadata']);m['maximum_admitted_bytes']=3*shared+decoder.rms.probe.ALLOWANCE;m['embedding_pipeline_caps']={'execution_width':32,'max_threads':1024}
    logits=anchor['logits']['bits'];values=decoder.unpack(logits[-64:],'I');token=max(range(64),key=lambda i:values[i])
    return {'event':'idle_result','passed':True,'result':{'operation':'resident_tiny_decoder_half_boundaries','id':item['id'],'rows_count':n,'position':item['position'],'metadata':m,'threadgroup_width':32,'layer_weight_layouts':wl,'layer_half_layouts':hl,'layer_float_layouts':fl,'outer_layouts':outer,'token_buffer_bytes':(n+7)*4,'shared_buffer_bytes':shared,'host_snapshot_readback_bytes':2*shared,'admitted_bytes':3*shared+decoder.rms.probe.ALLOWANCE,'allocations_reused':True,'tied_embedding_allocation':True,'first_layers':frames,'first_embedding_bits':anchor['embedding']['bits'].copy(),'first_normalized_bits':anchor['norm']['bits'].copy(),'first_logits_bits':logits.copy(),'rows':[{**row,'last_logits_bits':logits[-64:].copy(),'token_id':token}for row in view['rows']]}}


class Decoder(unittest.TestCase):
    def test_five_prefix_gates_and_frozen_four_step_history(self):
        inputs,anchors,_=decoder.cases();self.assertEqual(len(inputs),9)
        for item in inputs[:5]:decoder.validate(fixture(item,anchors[item['id']]),item,anchors[item['id']])
        history=[1,7,13,3]
        for item in inputs[5:]:
            self.assertEqual(item['tokens'],history);self.assertNotIn('input_bits',item['layers'][0]);history.append(anchors[item['id']]['expected_token'])
        self.assertEqual(history[-4:],[47]*4)

    def test_closed_on_propagation_tied_memory_logits_and_completion_errors(self):
        inputs,anchors,_=decoder.cases();item=inputs[4];a=anchors[item['id']]
        for kind in ('embedding','second_layer','norm_gate','logit_gate','budget','tied','wait','token','outer','type'):
            event=fixture(item,a);r=event['result'];anchor=copy.deepcopy(a)
            if kind=='embedding':r['first_embedding_bits'][0]^=1
            elif kind=='second_layer':r['first_layers'][1]['half'][10][0]^=1
            elif kind=='norm_gate':anchor['norm']['source'][0]+=100
            elif kind=='logit_gate':anchor['logits']['source'][0]+=100
            elif kind=='budget':r['admitted_bytes']-=1
            elif kind=='tied':r['tied_embedding_allocation']=False
            elif kind=='wait':r['rows'][0]['commit_wait_pairs']=2
            elif kind=='token':r['rows'][0]['token_id']+=1
            elif kind=='outer':r['outer_layouts'][0]['columns']=64
            elif kind=='type':r['first_logits_bits'][0]=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):decoder.validate(event,item,anchor)

    def test_greedy_outer_and_intermediate_arrays_fail_closed(self):
        inputs,anchors,_=decoder.cases();prompt=inputs[3];greedy=inputs[5];a=anchors[greedy['id']]
        for kind in ('normalized_nonfinite','earlier_logit','normalized_type','layer_half_type','layer_float_count'):
            event=fixture(prompt,anchors[prompt['id']]);r=event['result'];r['id']=greedy['id']
            if kind=='normalized_nonfinite':r['first_normalized_bits'][0]=0x7c00
            elif kind=='earlier_logit':r['first_logits_bits'][0]=0x7f800000
            elif kind=='normalized_type':r['first_normalized_bits'][0]=True
            elif kind=='layer_half_type':r['first_layers'][1]['half'][0][0]=True
            elif kind=='layer_float_count':r['first_layers'][0]['single'][3].pop()
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'array shape/type/finite'):decoder.validate(event,greedy,a)

    def test_path_aliases_rejected_before_frozen_reads(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d)
            with patch.object(decoder,'cases')as read:
                for directory in(root,root/'out.json',root/'build',root/'build/inputs'):
                    with self.assertRaises(ValueError):decoder.run(root/'out.json',root/'build',directory)
                read.assert_not_called()

if __name__=='__main__':unittest.main()
