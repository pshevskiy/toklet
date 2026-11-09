"""Exact independent recurrence experiment; no engine imports or reassociation.

The staged profile accepts already-rounded FP32 pipeline operands. FP64 products
are materialized in their original order and reduced by the original math.fsum.
NumPy is imported only when numerical execution starts, after runtime admission.
"""
import math


def _operands(q,k,v,decay,beta,state,shape,staged):
    import numpy as np
    if type(staged) is not bool or type(shape) not in (tuple,list) or len(shape)!=4:
        raise ValueError('Explicit profile and four bounded recurrence dimensions required')
    if not all(type(n) is int and 0<n<=128 for n in shape):
        raise ValueError('Bounded integer recurrence dimensions required')
    hk,hv,dk,dv=shape
    if hv%hk:raise ValueError('Integral grouped head mapping required')
    dtype=np.float32 if staged else np.float64
    arrays=[]
    for values,size in [(q,hk*dk),(k,hk*dk),(v,hv*dv),(decay,hv),(beta,hv),(state,hv*dv*dk)]:
        raw=np.asarray(values)
        if raw.dtype not in (np.dtype('float32'),np.dtype('float64')) or raw.shape!=(size,) or not np.isfinite(raw).all():
            raise ValueError('Finite flat floating operands of exact extent required')
        array=np.asarray(raw,dtype=dtype)
        if not np.isfinite(array).all():raise ValueError('Finite representable operands required')
        if staged and not np.array_equal(array.astype(np.float64).view(np.uint64),raw.astype(np.float64).view(np.uint64)):
            raise ValueError('Staged recurrence operands must already be rounded FP32 values')
        arrays.append(array)
    return arrays


def _ordered_fp32_dot(left,right):
    import numpy as np
    products=np.multiply(left,right,dtype=np.float32)
    # The original loop starts from positive zero, even for signed-zero operands.
    products[...,0]=np.add(np.float32(0),products[...,0],dtype=np.float32)
    return np.add.accumulate(products,axis=-1,dtype=np.float32)[...,-1].copy()


def _fsum_dot(left,right):
    import numpy as np
    products=np.multiply(left,right,dtype=np.float64)
    width=products.shape[-1]
    rows=products.reshape(-1,width)
    # Only independent rows are traversed here. Products and fsum ordering within
    # each row are identical to the scalar generator, including signed zero.
    return np.asarray([math.fsum(row.tolist()) for row in rows],dtype=np.float64).reshape(products.shape[:-1])


def step(q,k,v,decay,beta,state,shape,staged):
    import numpy as np
    try:
        with np.errstate(over='raise',invalid='raise',under='ignore'):
            q,k,v,decay,beta,old=_operands(q,k,v,decay,beta,state,shape,staged)
            hk,hv,dk,dv=shape
            heads=np.arange(hv)//(hv//hk)
            query=q.reshape(hk,dk)[heads,None,:];key=k.reshape(hk,dk)[heads,None,:]
            dtype=np.float32 if staged else np.float64
            dot=_ordered_fp32_dot if staged else _fsum_dot
            decayed=np.multiply(old.reshape(hv,dv,dk),decay[:,None,None],dtype=dtype)
            prediction=dot(decayed,key)
            correction=np.multiply(np.subtract(v.reshape(hv,dv),prediction,dtype=dtype),beta[:,None],dtype=dtype)
            updated=np.add(decayed,np.multiply(key,correction[:,:,None],dtype=dtype),dtype=dtype)
            readout=dot(updated,query)
            if not np.isfinite(updated).all() or not np.isfinite(readout).all():
                raise ValueError('Finite recurrence result required')
    except (FloatingPointError,OverflowError) as error:
        raise ValueError('Nonfinite recurrence arithmetic') from error
    return updated.reshape(-1).tolist(),readout.reshape(-1).tolist()
