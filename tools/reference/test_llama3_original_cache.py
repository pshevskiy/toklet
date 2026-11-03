import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import llama3_original_cache as subject
from test_llama3_original_fp32 import TinyReader


class Source:
    def __init__(self,path):
        reference=TinyReader();self.shape=reference.shape;self.closed=False;self.fail_close=False
        self.tensors={'__metadata__':{'format':'pt'}};payload=bytearray();self.expected={}
        for name,dims in subject.graph.expected_tensors(self.shape).items():
            words=(reference.values[name].view(np.uint32)>>16).astype('<u2')
            raw=words.tobytes();first=len(payload);payload.extend(raw)
            self.tensors[name]=dict(dtype='BF16',shape=dims,data_offsets=[first,len(payload)])
            self.expected[name]=subject.graph.widen(words)
        self.tensors=subject.graph.validate_header(self.tensors,len(payload),self.shape)
        Path(path).write_bytes(payload);self.file=Path(path).open('r+b');self.offset=0
        self.identity=self.file_identity()
    def file_identity(self):
        s=os.fstat(self.file.fileno());return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
    def close(self):
        changed=self.file_identity()!=self.identity;self.file.close();self.closed=True
        if changed or self.fail_close:raise ValueError('fixture source close failure')


class CacheTests(unittest.TestCase):
    def test_exact_bits_tied_views_and_source_closed_before_publication(self):
        with tempfile.TemporaryDirectory()as name:
            source=Source(Path(name)/'weights');self.assertNotIn('__metadata__',source.tensors)
            edge=np.array([0,0x8000,1,0x7f7f,0xc000],dtype='<u2')
            source.file.seek(0);source.file.write(edge.tobytes());source.file.flush()
            source.expected['model.embed_tokens.weight'].reshape(-1)[:len(edge)]=subject.graph.widen(edge)
            source.identity=source.file_identity();expected=source.expected
            cache=subject.Cache._consume(source,chunk_elements=7)
            self.assertTrue(source.closed)
            self.assertEqual(cache.plan['weight_bytes'],sum(v.nbytes for v in expected.values()))
            for key,value in expected.items():
                np.testing.assert_array_equal(cache.weight(key).view(np.uint32),value.view(np.uint32))
                self.assertFalse(cache.weight(key).flags.writeable)
            table=cache.weight('model.embed_tokens.weight');tile=cache.weight('model.embed_tokens.weight',1,3)
            self.assertTrue(np.shares_memory(table,tile));self.assertFalse(tile.flags.writeable)
            gathered=cache.embedding([1,2]);gathered.fill(0)
            np.testing.assert_array_equal(table,expected['model.embed_tokens.weight'])
            cache.close()
            with self.assertRaises(ValueError):cache.weight('model.embed_tokens.weight')

    def test_admission_precedes_allocation_and_invalid_extents_close_source(self):
        with tempfile.TemporaryDirectory()as name:
            source=Source(Path(name)/'weights')
            with patch.object(subject.np,'empty',side_effect=AssertionError('unexpected allocation')):
                with self.assertRaisesRegex(ValueError,'exceed budget'):subject.Cache._consume(source,memory_limit_bytes=1)
            self.assertTrue(source.closed)
            source=Source(Path(name)/'weights2');source.tensors['model.norm.weight']['data_offsets'][1]-=2
            with self.assertRaises(ValueError):subject.Cache._consume(source)
            self.assertTrue(source.closed)
            for field,value in [('dtype','F16'),('shape',[7])]:
                source=Source(Path(name)/('bad-'+field));source.tensors['model.norm.weight'][field]=value
                with self.assertRaises(ValueError):subject.Cache._consume(source)
                self.assertTrue(source.closed)

    def test_short_read_nonfinite_mutation_and_close_fail_without_publishing(self):
        with tempfile.TemporaryDirectory()as name:
            source=Source(Path(name)/'short');source.fail_close=True
            with patch.object(subject.os,'pread',return_value=b''):
                with self.assertRaisesRegex(ValueError,'truncated')as caught:subject.Cache._consume(source)
            self.assertTrue(source.closed);self.assertTrue(caught.exception.__notes__)
            source=Source(Path(name)/'nonfinite');source.file.seek(0);source.file.write(b'\x80\x7f');source.file.flush();source.identity=source.file_identity()
            with self.assertRaisesRegex(ValueError,'Nonfinite'):subject.Cache._consume(source)
            self.assertTrue(source.closed)
            source=Source(Path(name)/'changed');real=os.pread;calls=0
            def changing(fd,size,offset):
                nonlocal calls
                raw=real(fd,size,offset);calls+=1
                if calls==1:
                    source.file.seek(0);source.file.write(b'\0\0');source.file.flush()
                    os.utime(source.file.name,ns=(source.identity[3]+1000000000,source.identity[3]+1000000000))
                return raw
            with patch.object(subject.os,'pread',side_effect=changing):
                with self.assertRaisesRegex(ValueError,'changed'):subject.Cache._consume(source,chunk_elements=7)
            self.assertTrue(source.closed)
            source=Source(Path(name)/'close');source.fail_close=True
            with self.assertRaisesRegex(ValueError,'close failure'):subject.Cache._consume(source)
            self.assertTrue(source.closed)
