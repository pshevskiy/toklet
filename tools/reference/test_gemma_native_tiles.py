import io
import os
import struct
import threading
import unittest
from unittest.mock import patch
import affine4_tiles as storage
import gemma_native_projection as primitive
import gemma_native_tiles as subject


class OrderedSource(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.owner = threading.get_ident()
        self.reads = []

    def read(self, count=-1):
        if threading.get_ident() != self.owner:
            raise AssertionError('Worker touched shared source descriptor')
        self.reads.append((self.tell(), count))
        return super().read(count)


class TileControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = os.environ.get('TOKLET_GEMMA_NATIVE_PROJECTION_LIBRARY')
        if not cls.library:
            raise unittest.SkipTest('Explicit compiled projection oracle library required')

    def source(self):
        rows, columns = 513, 64
        codes = bytes((index*13+index//32) % 256 for index in range(rows*columns//2))
        scales = bytes.fromhex('003e')*rows
        biases = b''.join(struct.pack('<H', 0x3e00+row % 257) for row in range(rows))
        self.body = codes+scales+biases
        source = OrderedSource(self.body)
        matrix = storage.MatrixRows(source, len(self.body), rows, columns, 0, len(codes), len(codes)+len(scales))
        return matrix

    def test_ordered_source_partial_tile_and_single_four_worker_identity(self):
        expected_reads = None
        for tokens in [1, 32]:
            inputs = struct.pack('<'+'f'*(tokens*64), *((index % 31-15)/16 for index in range(tokens*64)))
            outputs = []
            for workers, normal_domain in [(1, False), (4, False), (1, True), (4, True)]:
                matrix = self.source()
                outputs.append(subject.project_rows(matrix, inputs, tokens=tokens, lanes=32,
                                                    library=self.library, workers=workers, normal_domain=normal_domain))
                if expected_reads is None:
                    expected_reads = matrix.stream.reads
                self.assertEqual(matrix.stream.reads, expected_reads)
                self.assertEqual(len(outputs[-1]), tokens*513*4)
            self.assertTrue(all(output == outputs[0] for output in outputs))
            # Independent primitive replay of the final one-row tile verifies
            # its token stride and row placement, including the partial tail.
            matrix = self.source()
            blocks = []
            for offset, stride in matrix.components():
                blocks.append(self.body[offset+512*stride:offset+513*stride])
            tail = primitive.Projection(self.library)(*blocks, inputs, tokens=tokens, rows=1, columns=64, lanes=32)
            for token in range(tokens):
                self.assertEqual(outputs[0][(token*513+512)*4:(token*513+513)*4], tail[token*4:(token+1)*4])

    def test_invalid_worker_admission_and_truncated_source_reject(self):
        matrix = self.source()
        for workers in [0, 2, 5, True]:
            with self.assertRaises(ValueError):
                subject.project_rows(matrix, bytes(64*4), tokens=1, lanes=32, library=self.library, workers=workers)
        matrix.stream.truncate(1)
        with self.assertRaises(ValueError):
            subject.project_rows(matrix, bytes(64*4), tokens=1, lanes=32, library=self.library, workers=4)

    def test_wrong_worker_environment_fails_before_projection(self):
        matrix = self.source()
        with patch.object(subject, 'Projection', return_value=lambda *args, **kwargs: bytes(4)):
            with self.assertRaisesRegex(ValueError, 'CPU worker subnormal environment differs'):
                subject.project_rows(matrix, bytes(64*4), tokens=1, lanes=32, library=self.library, workers=4)
        self.assertFalse(matrix.stream.closed)

    def test_strict_domain_reports_first_input_or_metadata_tile_before_dispatch(self):
        matrix = self.source()
        with patch.object(subject, 'Projection') as library:
            with self.assertRaises(subject.domain.DomainError) as caught:
                subject.project_rows(matrix, struct.pack('<I', 1)+bytes(63*4), tokens=1, lanes=32,
                    library=self.library, workers=4, normal_domain=True, matrix_name='test.matrix')
            self.assertEqual(caught.exception.evidence['matrix'], 'test.matrix')
            self.assertIsNone(caught.exception.evidence['tile_start'])
            matrix.stream.seek(matrix.bias_offset)
            matrix.stream.write(struct.pack('<H', 0x00c0))
            with self.assertRaises(subject.domain.DomainError) as caught:
                subject.project_rows(matrix, bytes(64*4), tokens=1, lanes=32,
                    library=self.library, workers=4, normal_domain=True, matrix_name='test.matrix')
            self.assertEqual(caught.exception.evidence['tile_start'], 0)
            self.assertEqual(caught.exception.evidence['status'], 'inconclusive')
            library.assert_not_called()
