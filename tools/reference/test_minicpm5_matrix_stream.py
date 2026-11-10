"""Independent tiny byte fixtures exercise stream coverage, not quantizer math."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import minicpm5_matrix_stream as conversion
import minicpm5_parent as parent


class ShortReads(io.BytesIO):
    def read(self, count):
        return super().read(min(count,13))


class FixtureChild:
    def __init__(self, damage=False):
        self.calls=[];self.damage=damage
        self.weights={name:bytes(i%251 for i in range(rows*columns))
                      for name,rows,columns in [('a.weight',257,64),('b.weight',3,128)]}
        self.columns={'a.weight':64,'b.weight':128}

    def tile(self,name,first,rows):
        self.calls.append((name,first,rows));columns=self.columns[name]
        result=dict(weight=self.weights[name][first*columns:(first+rows)*columns],
                    scales=b'Sc'*(rows*columns//64),biases=b'Bi'*(rows*columns//64))
        if self.damage:result['weight']=bytes([result['weight'][0]^1])+result['weight'][1:]
        return result


def fixture():
    child=FixtureChild();norm=b'\x80\x3f'*8
    header={};payload=bytearray()
    for name,shape in [('a.weight',[257,64]),('norm.weight',[8]),('b.weight',[3,128])]:
        if len(shape)==1:raw=norm
        else:
            raw=bytes(value for code in child.weights[name] for value in [code,63])
        start=len(payload);payload.extend(raw)
        header[name]=dict(dtype='BF16',shape=shape,data_offsets=[start,len(payload)])
    encoded=json.dumps(header,separators=(',',':')).encode();prefix=len(encoded).to_bytes(8,'little')
    raw=prefix+encoded+payload
    return raw,prefix,encoded,{'norm.weight':norm}


def fake_quantizer(raw,rows,columns):
    conversion.quant.geometry(raw,rows,columns)
    return dict(weight=raw[::2],scales=b'Sc'*(rows*columns//64),biases=b'Bi'*(rows*columns//64))


def stream(raw,child=None,quantizer=fake_quantizer):
    original,prefix,header,norms=fixture();size=len(original)
    spans=parent.plan(prefix,header,size);journal=io.StringIO();child=FixtureChild() if child is None else child
    value=conversion.ConvertedStream(ShortReads(raw),len(header)+8,spans,child,quantizer,journal)
    return value,child,journal,(prefix,header,size,norms)


class MatrixStreams(unittest.TestCase):
    def test_partial_reads_preserve_all_rows_norms_and_exact_byte_hashes(self):
        raw,*_=fixture();value,child,journal,controls=stream(raw);prefix,header,size,norms=controls
        authenticated=parent.authenticate(value,prefix,header,size,hashlib.sha256(raw).hexdigest(),norms,io.StringIO())
        result=value.finish(size)
        self.assertEqual(authenticated['exact_child_norms'],1)
        self.assertEqual(child.calls,[('a.weight',0,256),('a.weight',256,1),('b.weight',0,3)])
        self.assertEqual((result['matrix_count'],result['tiles'],result['norm_tensors']),(2,3,1))
        self.assertTrue(result['passed']);self.assertFalse(result['object_authenticated'])
        events=[json.loads(line) for line in journal.getvalue().splitlines()]
        self.assertEqual([e['rows'] for e in events if e['event']=='matrix_tile'],[256,1,3])
        self.assertTrue(all(e['object_authenticated'] is False for e in events))
        for report in result['matrices']:
            self.assertEqual(report['comparison']['weight']['reconstructed_sha256'],
                             hashlib.sha256(child.weights[report['name']]).hexdigest())

    def test_changed_child_bytes_reject_reproduction_without_forging_source_authentication(self):
        raw,*_=fixture();value,child,journal,controls=stream(raw,FixtureChild(damage=True));prefix,header,size,norms=controls
        parent.authenticate(value,prefix,header,size,hashlib.sha256(raw).hexdigest(),norms,io.StringIO())
        result=value.finish(size)
        self.assertFalse(result['passed']);self.assertFalse(result['object_authenticated'])
        self.assertEqual(sum(r['comparison']['weight']['mismatching_bytes'] for r in result['matrices']),3)

    def test_source_truncation_trailing_bytes_and_wrong_digest_are_not_authenticated(self):
        raw,*_=fixture()
        for source,digest in [(raw[:-1],hashlib.sha256(raw).hexdigest()),
                              (raw+b'x',hashlib.sha256(raw).hexdigest()),(raw,'0'*64)]:
            value,child,journal,controls=stream(source);prefix,header,size,norms=controls
            with self.assertRaises(ValueError):parent.authenticate(value,prefix,header,size,digest,norms,io.StringIO())
        value,*_=stream(raw);value.read(1)
        with self.assertRaises(ValueError):value.finish(len(raw))

    def test_failed_quantizer_poison_requires_replay(self):
        raw,*_=fixture();calls=[]
        def fail_second(raw,rows,columns):
            calls.append(rows)
            if len(calls)==2:raise ValueError('injected quantizer failure')
            return fake_quantizer(raw,rows,columns)
        value,child,journal,controls=stream(raw,quantizer=fail_second);prefix,header,size,norms=controls
        with self.assertRaisesRegex(ValueError,'injected'):parent.authenticate(value,prefix,header,size,hashlib.sha256(raw).hexdigest(),norms,io.StringIO())
        self.assertTrue(value.failed)
        with self.assertRaisesRegex(ValueError,'replay'):value.read(1)
        with self.assertRaises(ValueError):value.finish(size)

    def test_child_layout_requires_complete_typed_contiguous_physical_buffers(self):
        logical={'a.weight':[2,64],'norm.weight':[8]}
        header={
            'a.weight':dict(dtype='U32',shape=[2,16],data_offsets=[0,128]),
            'a.scales':dict(dtype='BF16',shape=[2,1],data_offsets=[128,132]),
            'a.biases':dict(dtype='BF16',shape=[2,1],data_offsets=[132,136]),
            'norm.weight':dict(dtype='BF16',shape=[8],data_offsets=[136,152])}
        self.assertEqual(conversion.layout(header,logical,8,160)['a.weight']['fields']['weight']['row_stride'],64)
        for key,field,value in [('a.weight','dtype','BF16'),('a.scales','shape',[2,True]),
                                ('a.biases','data_offsets',[131,135]),('norm.weight','data_offsets',[136,153])]:
            changed=copy.deepcopy(header);changed[key][field]=value
            with self.assertRaises(ValueError):conversion.layout(changed,logical,8,160)
        changed=copy.deepcopy(header);del changed['a.biases']
        with self.assertRaises(ValueError):conversion.layout(changed,logical,8,160)
        for origin,size in [(False,160),(8,159),(8,161)]:
            with self.assertRaises(ValueError):conversion.layout(header,logical,origin,size)

    def test_child_rejects_wrong_regular_extent_directory_and_symlink_before_hashing(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'small';p.write_bytes(b'not a model')
            for path in [p,Path(folder)]:
                with self.assertRaises(ValueError):conversion.Child(path,{})
            link=Path(folder)/'link';link.symlink_to(p.name)
            with self.assertRaises(OSError):conversion.Child(link,{})


if __name__=='__main__':unittest.main()
