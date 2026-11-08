import struct
import unittest
import check_qwen35_payload as check


class Controls(unittest.TestCase):
    def test_bf16_preserves_bits_and_signed_zero(self):
        self.assertEqual(check.bf16(bytes.fromhex('803f0080000080bf')), bytes.fromhex('0000803f0000008000000000000080bf'))

    def test_nonfinite_and_odd_bf16_reject(self):
        for raw in [b'\x00', bytes.fromhex('807f'), bytes.fromhex('c07f'), bytes.fromhex('80ff')]:
            with self.assertRaises(ValueError):check.bf16(raw)

    def test_independent_u32_nibble_order_and_negative_scale(self):
        codes=bytes.fromhex('1032547698badcfe')*4
        for scale,bias,want in [(bytes.fromhex('803f'),bytes.fromhex('00c0'),[float(i%16-2) for i in range(64)]),
                                (bytes.fromhex('00bf'),bytes.fromhex('4040'),[3.0-0.5*(i%16) for i in range(64)]),
                                (bytes.fromhex('0000'),bytes.fromhex('803f'),[1.0]*64)]:
            self.assertEqual(check.affine_row(codes,scale,bias,64),struct.pack('<64f',*want))

    def test_bad_row_geometry_and_overflow_reject(self):
        for columns in [0,63,True]:
            with self.assertRaises(ValueError):check.affine_row(b'',b'',b'',columns)
        with self.assertRaises(ValueError):check.affine_row(bytes(31),bytes(2),bytes(2),64)
        with self.assertRaises(OverflowError):check.affine_row(b'\xff'*32,bytes.fromhex('7f7f'),bytes(2),64)

    def test_component_duplicate_unknown_missing_and_boolean_reject(self):
        row=dict(name='a',stored='original',bytes=4,sha256='x')
        self.assertEqual(check.index_components([row],{'a':None}),{'a':row})
        for rows,expected in [([row,row],{'a':None,'b':None}),([row],{'b':None}),([] ,{'a':None}),([dict(row,bytes=True)],{'a':None})]:
            with self.assertRaises(ValueError):check.index_components(rows,expected)


if __name__ == '__main__':unittest.main()
