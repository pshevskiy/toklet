"""Bounded tiled FFI to an explicitly compiled independent projection oracle.

The caller supplies the compiled library; importing this module never builds or
loads executable code. No dense model expansion, source graph or target arrays
are consulted. This profile alone does not qualify Metal flush/provider behavior.
"""
import ctypes
import math
import struct
import sys


class Projection:
    def __init__(self, library):
        if sys.byteorder != 'little':
            raise ValueError('Little-endian oracle storage required')
        self.library = ctypes.CDLL(str(library))
        self.function = self.library.gemma_native_projection
        self.function.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_size_t]*4
        self.function.restype = ctypes.c_int

    def __call__(self, codes, scales, biases, inputs, *, tokens, rows, columns, lanes):
        for value, maximum in [(tokens, 32), (rows, 128), (columns, 12288), (lanes, 64)]:
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError('Bounded integer projection shape required')
        if columns % 64 or lanes & (lanes-1):
            raise ValueError('Group64 columns and power-of-two SIMD width required')
        blocks = [codes, scales, biases, inputs]
        sizes = [rows*columns//8*4, rows*columns//64*2, rows*columns//64*2, tokens*columns*4]
        if any(type(block) is not bytes or len(block) != size for block, size in zip(blocks, sizes, strict=True)):
            raise ValueError('Exact packed byte lengths required')
        # Validate at the public boundary; kernel errors leave discardable output.
        if any(not math.isfinite(v[0]) for v in struct.iter_unpack('<f', inputs)):
            raise ValueError('Finite FP32 inputs required')
        for block in [scales, biases]:
            if any((v[0] & 0x7f80) == 0x7f80 for v in struct.iter_unpack('<H', block)):
                raise ValueError('Finite BF16 metadata required')
        types = [ctypes.c_uint32, ctypes.c_uint16, ctypes.c_uint16, ctypes.c_float]
        owned = [(kind*(len(block)//ctypes.sizeof(kind))).from_buffer_copy(block)
                 for kind, block in zip(types, blocks, strict=True)]
        output = (ctypes.c_float*(tokens*rows))()
        status = self.function(*owned, output, tokens, rows, columns, lanes)
        if status:
            raise ValueError('Independent projection failed: '+str(status))
        return bytes(output)
