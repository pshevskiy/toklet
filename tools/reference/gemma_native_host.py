"""Explicitly loaded Darwin paired-sincos and live provider-image witness."""
import ctypes
import sys


class Host:
    def __init__(self, library):
        if sys.byteorder != 'little' or sys.platform != 'darwin':
            raise ValueError('Named little-endian Darwin host profile required')
        self.library = ctypes.CDLL(str(library))
        self.sincos_function = self.library.gemma_native_sincos
        self.sincos_function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.sincos_function.restype = ctypes.c_int
        self.provider_function = self.library.gemma_native_provider
        self.provider_function.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                                          ctypes.c_void_p, ctypes.c_void_p]
        self.provider_function.restype = ctypes.c_int

    def sincos(self, angles):
        if type(angles) is not bytes or not 8 <= len(angles) <= 65536*8 or len(angles) % 8:
            raise ValueError('Bounded binary64 angle bytes required')
        count = len(angles)//8
        inputs = (ctypes.c_double*count).from_buffer_copy(angles)
        output = (ctypes.c_float*(2*count))()
        status = self.sincos_function(inputs, output, count)
        if status:
            raise ValueError('Paired host sincos failed: '+str(status))
        return bytes(output)

    def providers(self):
        result = {}
        for index, name in enumerate(['tanh', 'exp', 'pow', 'sqrt', '__sincos_stret']):
            path = ctypes.create_string_buffer(1024)
            uuid = (ctypes.c_ubyte*16)()
            addresses = (ctypes.c_uint64*2)()
            status = self.provider_function(index, path, len(path), uuid, addresses)
            if status or not any(uuid):
                raise ValueError('Live host provider identity unavailable: '+name)
            result[name] = dict(path=path.value.decode(), uuid=bytes(uuid).hex(),
                                address=addresses[0], image_base=addresses[1])
        return result
