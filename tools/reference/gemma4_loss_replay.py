"""Bounded independent BF16 loss replay; NumPy is explicit, no Torch import."""
import hashlib
import math
import struct
import numpy as np


def replay(path, batches, ids, producer, *, block, global_start, vocab=262144):
    if (not 2<=len(ids)<=1025 or not 1<=vocab<=262144
            or any(type(i) is not int or not 0<=i<vocab for i in ids)):
        raise ValueError('Bounded next-token input required')
    records=[];offset=0;chain=hashlib.sha256();max_error=0.
    with path.open('rb') as file:
        for item in batches:
            start,end=item['start'],item['end']
            if (start!=len(records) or not start<end<=min(start+32,len(ids)-1)
                    or item['offset']!=offset or item['shape']!=[1,end-start,vocab]
                    or item['bytes']!=(end-start)*vocab*2):
                raise ValueError('Exact bounded position batch required')
            raw=file.read(item['bytes']);offset+=len(raw);chain.update(raw)
            if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
                raise ValueError('Complete batch identity differs')
            words=np.frombuffer(raw,dtype='<u2').reshape(end-start,vocab)
            values=(words.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError('Finite BF16 distributions required')
            maximum=values.max(axis=1)
            mass=np.exp(values-maximum[:,None]).sum(axis=1,dtype=np.float64)
            greedy=values.argmax(axis=1)
            # Stdlib anchor in every batch checks the independent vector reduction.
            anchor=math.fsum(math.exp(float(v)-float(maximum[0]))for v in values[0])
            if abs(anchor-float(mass[0]))>8*math.ulp(max(1.,anchor)):
                raise ValueError('Stdlib exponential-mass anchor differs')
            for local in range(end-start):
                position=start+local;target=ids[position+1]
                peak=float(maximum[local]);exponential=float(mass[local])
                target_value=float(values[local,target]);nll=(peak-target_value)+math.log(exponential)
                error=abs(nll-producer['nlls'][position]);max_error=max(max_error,error)
                if (not math.isfinite(nll) or nll<0 or error>1e-12
                        or int(greedy[local])!=producer['greedy'][position]):
                    raise ValueError('Independent NLL/greedy replay differs')
                records.append(dict(block=block,local_row=position,global_target_index=global_start+position+1,
                    target=target,vocab=vocab,target_fp32_bits=int(words[local,target])<<16,
                    maximum=peak,exp_mass=exponential,nll=nll,producer_nll=producer['nlls'][position],
                    greedy_lowest_id=int(greedy[local]),row_sha256=hashlib.sha256(raw[local*vocab*2:(local+1)*vocab*2]).hexdigest()))
        if file.read(1) or len(records)!=len(ids)-1 or producer['targets']!=len(records):
            raise ValueError('Exact complete next-token coverage required')
    return dict(passed=True,targets=len(records),mean_nll=math.fsum(r['nll']for r in records)/len(records),
                nll_max_replay_difference=max_error,bytes=offset,sha256=chain.hexdigest(),records=records)
