"""Exact source/child row coverage during the unchanged bounded parent audit."""
import hashlib
import json
import os
from pathlib import Path
import stat
import time

import minicpm5_parent as parent
import minicpm5_quantize as quant

need = quant.need


def layout(header, logical, origin, size):
    """Validate all physical spans before requesting any numerical operation."""
    need(type(header) is dict and type(logical) is dict and logical,
         'Child declarations required')
    expected = {};matrices = {}
    for name, shape in logical.items():
        need(type(name) is str and name.endswith('.weight') and type(shape) is list
             and 1 <= len(shape) <= 2 and all(type(n) is int and 0 < n <= 131072 for n in shape),
             'Invalid logical source shape')
        if len(shape) == 1:
            expected[name] = ('BF16', shape, 2*shape[0])
            continue
        rows, columns = shape
        need(columns % 64 == 0 and columns <= quant.MAX_COLUMNS,
             'Affine8 source columns differ')
        fields = {}
        for field in ['weight', 'scales', 'biases']:
            key = name if field == 'weight' else name.removesuffix('weight')+field
            physical = [rows, columns//(4 if field == 'weight' else 64)]
            stride = columns if field == 'weight' else columns//64*2
            expected[key] = ('U32' if field == 'weight' else 'BF16', physical, rows*stride)
            fields[field] = dict(name=key, row_stride=stride)
        matrices[name] = dict(shape=shape, fields=fields)
    need(set(header)-{'__metadata__'} == set(expected), 'Child physical inventory differs')
    spans = []
    for name, (dtype, shape, count) in expected.items():
        item = header[name]
        need(type(item) is dict and set(item) == {'dtype','shape','data_offsets'}
             and item['dtype'] == dtype and type(item['shape']) is list
             and all(type(n) is int for n in item['shape']) and item['shape'] == shape
             and type(item['data_offsets']) is list and len(item['data_offsets']) == 2
             and all(type(n) is int and 0 <= n <= size for n in item['data_offsets']),
             'Child dtype/shape/offset declaration differs')
        start, end = item['data_offsets']
        need(end-start == count, 'Child physical extent differs')
        spans.append((start,end))
    spans.sort()
    need(type(origin) is int and type(size) is int and 0 < origin < size
         and spans[0][0] == 0 and all(a[1] == b[0] for a,b in zip(spans,spans[1:]))
         and origin+spans[-1][1] == size, 'Child spans have holes/overlap/wrong extent')
    return matrices


class Child:
    """Read the one exact child through its consumed regular descriptor."""
    def __init__(self, path, logical):
        self.path = Path(path)
        self.fd = os.open(path, os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:
            self.verify()
            prefix = os.pread(self.fd,8,0);count = int.from_bytes(prefix,'little')
            need(len(prefix)==8 and count==110378, 'Pinned child header extent differs')
            raw = os.pread(self.fd,count,8)
            need(hashlib.sha256(raw).hexdigest() == 'b6fdf59a1fce7d5fb3caf76725156c53c222d1e5f481624981cc3cb3abcacf07',
                 'Pinned child header identity differs')
            self.header = json.loads(raw,object_pairs_hook=parent.unique)
            self.origin = 8+count
            self.matrices = layout(self.header,logical,self.origin,parent.CHILD_SIZE)
            need(len(logical)==381 and len(self.matrices)==296, 'Child model coverage differs')
        except BaseException:
            os.close(self.fd);raise

    def verify(self):
        info = os.fstat(self.fd)
        need(stat.S_ISREG(info.st_mode) and info.st_size==parent.CHILD_SIZE,
             'Child descriptor is not the exact regular object')
        digest = hashlib.sha256()
        for start in range(0,parent.CHILD_SIZE,parent.BLOCK):
            count = min(parent.BLOCK,parent.CHILD_SIZE-start)
            raw = os.pread(self.fd,count,start)
            need(len(raw)==count, 'Child descriptor shortened during hashing')
            digest.update(raw)
        need(digest.hexdigest()==parent.CHILD_SHA and parent.digest(self.path)==parent.CHILD_SHA,
             'Consumed child descriptor/path changed')

    def tile(self, name, first, rows):
        need(name in self.matrices and type(first) is int and type(rows) is int
             and first>=0 and 1<=rows<=quant.ROWS, 'Bounded child row range required')
        plan = self.matrices[name]
        need(first+rows<=plan['shape'][0], 'Child tile outside matrix')
        result = {}
        for field, item in plan['fields'].items():
            count = rows*item['row_stride']
            offset = self.origin+self.header[item['name']]['data_offsets'][0]+first*item['row_stride']
            result[field] = os.pread(self.fd,count,offset)
            need(len(result[field])==count, 'Child tile read shortened')
        return result

    def close(self):
        os.close(self.fd)


class ConvertedStream:
    """Coarse read boundary: at most one256-row tile, never a whole matrix.

    The original parent.authenticate still hashes every returned byte and checks
    copied norms. Each journal entry stays unauthenticated until full-object
    authentication and the transport receipt close successfully.
    """
    def __init__(self, stream, origin, spans, child, quantizer, journal):
        self.stream, self.origin, self.spans = stream, origin, spans
        self.child, self.quantizer, self.journal = child, quantizer, journal
        self.position = 0;self.index = 0;self.first = 0;self.failed = False
        self.buffer = bytearray();self.reports = [];self.tiles = 0;self.norms = 0
        self.hashes = None;self.expected_hashes = None;self.totals = None

    def read(self, count):
        need(not self.failed and type(count) is int and 0<=count<=parent.BLOCK,
             'Bounded conversion read required; failed streams require replay')
        try:
            raw = self.stream.read(count)
            need(type(raw) is bytes and len(raw)<=count, 'Invalid conversion stream read')
            cursor = 0
            while cursor<len(raw):
                absolute = self.position+cursor
                if absolute<self.origin:
                    cursor += min(len(raw)-cursor,self.origin-absolute);continue
                need(self.index<len(self.spans), 'Trailing conversion source bytes')
                span = self.spans[self.index];end = self.origin+span['end']
                need(self.origin+span['start']<=absolute<end, 'Conversion source coverage gap')
                take = min(len(raw)-cursor,end-absolute)
                if len(span['shape'])==2:
                    rows = min(quant.ROWS,span['shape'][0]-self.first)
                    size = rows*span['shape'][1]*2
                    take = min(take,size-len(self.buffer))
                    self.buffer.extend(raw[cursor:cursor+take])
                    if len(self.buffer)==size:
                        self._tile(span,rows)
                cursor += take
                if absolute+take==end:
                    need(not self.buffer, 'Incomplete matrix row/tile at source boundary')
                    if len(span['shape'])==2:
                        need(self.first==span['shape'][0], 'Matrix row coverage incomplete')
                        self._matrix(span)
                    else:
                        self.norms += 1
                    self.index += 1;self.first = 0
            self.position += len(raw)
            return raw
        except BaseException:
            self.failed = True;raise

    def _tile(self, span, rows):
        if self.hashes is None:
            self.hashes = {key:hashlib.sha256() for key in ['weight','scales','biases']}
            self.expected_hashes = {key:hashlib.sha256() for key in self.hashes}
            self.totals = {key:dict(bytes=0,mismatching_bytes=0) for key in self.hashes}
        raw = bytes(self.buffer);columns = span['shape'][1]
        started = time.monotonic_ns();expected = self.child.tile(span['name'],self.first,rows)
        child_read = time.monotonic_ns()-started
        started = time.monotonic_ns();actual = self.quantizer(raw,rows,columns)
        quantized = time.monotonic_ns()-started
        started = time.monotonic_ns();compared = quant.compare(actual,expected)
        for key, row in compared.items():
            self.hashes[key].update(actual[key]);self.expected_hashes[key].update(expected[key])
            self.totals[key]['bytes'] += row['bytes']
            self.totals[key]['mismatching_bytes'] += row['mismatching_bytes']
        compare_ns = time.monotonic_ns()-started
        event = dict(event='matrix_tile',object_authenticated=False,name=span['name'],
                     first_row=self.first,rows=rows,columns=columns,
                     parent_offset=self.origin+span['start']+self.first*columns*2,
                     parent_bytes=len(raw),parent_tile_sha256=hashlib.sha256(raw).hexdigest(),
                     child_read_ns=child_read,completed_quantize_readback_ns=quantized,
                     compare_hash_ns=compare_ns,comparison=compared)
        self.journal.write(json.dumps(event,separators=(',',':'))+'\n');self.journal.flush()
        self.buffer.clear();self.first += rows;self.tiles += 1

    def _matrix(self, span):
        fields = {key:dict(**self.totals[key],reconstructed_sha256=self.hashes[key].hexdigest(),
                          expected_sha256=self.expected_hashes[key].hexdigest()) for key in self.hashes}
        report = dict(name=span['name'],shape=span['shape'],comparison=fields,
                      passed=all(row['mismatching_bytes']==0 for row in fields.values()))
        self.reports.append(report)
        self.journal.write(json.dumps(dict(event='matrix_completed',object_authenticated=False,
                                           **report),separators=(',',':'))+'\n');self.journal.flush()
        self.hashes = None;self.expected_hashes = None;self.totals = None

    def finish(self, size):
        need(not self.failed and self.position==size and self.index==len(self.spans)
             and not self.buffer and self.hashes is None,
             'Conversion stream did not finish exact source coverage')
        need(len(self.reports)==sum(len(s['shape'])==2 for s in self.spans),
             'Matrix report coverage differs')
        return dict(matrix_count=len(self.reports),tiles=self.tiles,norm_tensors=self.norms,
                    matrices=self.reports,
                    passed=all(row['passed'] for row in self.reports),
                    object_authenticated=False,retained_payload_bytes=0)
