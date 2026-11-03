"""One read-only FP32 reference model, streamed from the verified BF16 descriptor.

The source mmap is never accessed for tensor data. Bounded pread/widen buffers
fill row-major arrays; the descriptor and mmap close before publication. The
single tied table serves both input gathers and tiled output projections.
"""
import hashlib
import math
import os
import numpy as np
import llama3_original_fp32 as graph

MAX_MEMORY = 6*1024**3
RUNTIME_SCRATCH_RESERVE = 512*1024**2
CONVERSION_RESERVE = 64*1024**2
MAX_CHUNK_ELEMENTS = 1024**2


class Cache:
    @classmethod
    def load(cls, directory, memory_limit_bytes=MAX_MEMORY, capacity=566):
        return cls._consume(graph.Reader(directory), memory_limit_bytes, capacity)

    @classmethod
    def _consume(cls, source, memory_limit_bytes=MAX_MEMORY, capacity=566,
                 chunk_elements=MAX_CHUNK_ELEMENTS):
        weights={};records={};primary=None
        try:
            if (type(memory_limit_bytes) is not int or not 0<memory_limit_bytes<=MAX_MEMORY
                    or type(capacity) is not int or not 1<=capacity<=566
                    or type(chunk_elements) is not int or not 1<=chunk_elements<=MAX_CHUNK_ELEMENTS):
                raise ValueError('Invalid bounded FP32 cache options')
            shape=source.shape
            graph.validate_tensor_entries(source.tensors,source.identity[2]-source.offset,shape)
            expected=graph.expected_tensors(shape)
            weight_bytes=sum(math.prod(dims)*4 for dims in expected.values())
            kv_bytes=2*shape.layers*shape.kv_heads*capacity*shape.head_dim*4
            admitted=weight_bytes+kv_bytes+RUNTIME_SCRATCH_RESERVE+CONVERSION_RESERVE
            if admitted>memory_limit_bytes:
                raise ValueError('FP32 cache and reserved execution memory exceed budget')
            plan=dict(weight_bytes=weight_bytes,kv_bytes=kv_bytes,capacity=capacity,
                      runtime_scratch_reserve_bytes=RUNTIME_SCRATCH_RESERVE,
                      conversion_reserve_bytes=CONVERSION_RESERVE,
                      admitted_bytes=admitted,memory_limit_bytes=memory_limit_bytes,
                      chunk_elements=chunk_elements)
            entries=sorted(expected,key=lambda name:source.tensors[name]['data_offsets'][0])
            cursor=0
            for name in entries:
                row=source.tensors[name];count=math.prod(row['shape'])
                if row['data_offsets']!=[cursor,cursor+count*2]:
                    raise ValueError('Cache source extents are not complete contiguous BF16')
                if source.file_identity()!=source.identity:
                    raise ValueError('Cache source changed during conversion')
                target=np.empty(row['shape'],dtype=np.float32);flat=target.reshape(-1)
                for first in range(0,count,chunk_elements):
                    size=min(chunk_elements,count-first)
                    raw=os.pread(source.file.fileno(),size*2,source.offset+cursor+first*2)
                    if len(raw)!=size*2:raise ValueError('Cache source range is truncated')
                    words=np.frombuffer(raw,dtype='<u2');values=graph.widen(words)
                    flat[first:first+size]=values
                del raw,words,values,flat
                target.flags.writeable=False
                records[name]=dict(shape=list(target.shape),bytes=target.nbytes,
                                   fp32_sha256=hashlib.sha256(memoryview(target).cast('B')).hexdigest())
                weights[name]=target;cursor+=count*2
            if cursor!=source.identity[2]-source.offset or sum(v.nbytes for v in weights.values())!=weight_bytes:
                raise ValueError('Cache source coverage differs')
        except BaseException as error:
            primary=error
        finally:
            try:source.close()
            except BaseException as error:
                if primary is None:primary=error
                else:primary.add_note('Source close also failed: '+repr(error))
        if primary is not None:
            weights.clear()
            raise primary
        # Publish only after complete delivery and successful source close.
        owner=object.__new__(cls)
        owner.shape=shape;owner.plan=plan;owner.tensor_records=records
        owner._weights=weights;owner._closed=False
        return owner

    def weight(self,name,start=None,stop=None):
        if self._closed:raise ValueError('FP32 cache is closed')
        values=self._weights[name]
        return values if start is None else values[start:stop]

    def embedding(self,ids):
        # Advanced indexing returns session scratch, never a mutable model alias.
        return self.weight('model.embed_tokens.weight')[ids]

    def close(self):
        self._weights.clear();self._closed=True
