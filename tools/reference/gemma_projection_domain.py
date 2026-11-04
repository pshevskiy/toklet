"""Sufficient dyadic-grid proof excluding projection underflow intermediates.

Normal operands on grids 2^qm and 2^qx, with qm>=-126 and qm+qx>=-126,
cannot create a nonzero subnormal in affine reconstruction, products, exact FMA
residuals, compensated adds/subtracts or merges. FP32 RNE only coarsens a grid.
An insufficient bound is inconclusive, even when a particular dot is normal.
Finite/overflow, signed-zero and provider checks remain separate requirements.
"""
import struct


class DomainError(ValueError):
    def __init__(self, reason, **facts):
        self.evidence = dict(status='inconclusive', reason=reason, **facts)
        super().__init__(str(self.evidence))

    def __str__(self):
        return str(self.evidence)


def grid(raw, mantissa):
    if mantissa not in [7, 23]:
        raise ValueError('FP32 or BF16 significand required')
    size, code = (4, '<I') if mantissa == 23 else (2, '<H')
    if type(raw) is not bytes or not raw or len(raw) % size or len(raw) > 32*12288*4:
        raise ValueError('Bounded complete projection operand bytes required')
    minimum = None
    for index, (word,) in enumerate(struct.iter_unpack(code, raw)):
        exponent = (word >> mantissa) & 255
        fraction = word & ((1 << mantissa)-1)
        if exponent == 255:
            raise ValueError('Finite projection domain operands required')
        if exponent == 0:
            if fraction:
                raise DomainError('subnormal_operand', index=index, mantissa=mantissa)
            continue
        significand = (1 << mantissa) | fraction
        trailing = (significand & -significand).bit_length()-1
        quantum = exponent-127-mantissa+trailing
        minimum = quantum if minimum is None else min(minimum, quantum)
    return minimum


def input_grid(raw):
    return grid(raw, 23)


def admit_metadata(scales, biases, input_quantum):
    if type(scales) is not bytes or type(biases) is not bytes or len(scales) != len(biases) or len(scales) > 128*192*2:
        raise ValueError('Complete bounded paired BF16 metadata required')
    if input_quantum is not None and (type(input_quantum) is not int or not -149 <= input_quantum <= 127):
        raise ValueError('Validated FP32 input quantum required')
    values = [value for raw in [scales, biases] if (value := grid(raw, 7)) is not None]
    metadata_quantum = min(values) if values else None
    if metadata_quantum is not None and metadata_quantum < -126:
        raise DomainError('insufficient_reconstruction_grid', metadata_quantum=metadata_quantum)
    product_quantum = None if metadata_quantum is None or input_quantum is None else metadata_quantum+input_quantum
    if product_quantum is not None and product_quantum < -126:
        raise DomainError('insufficient_product_grid', metadata_quantum=metadata_quantum,
                          input_quantum=input_quantum, product_quantum=product_quantum)
    return dict(admitted=True, metadata_quantum=metadata_quantum, input_quantum=input_quantum,
                product_quantum=product_quantum, scope='projection underflow only')
