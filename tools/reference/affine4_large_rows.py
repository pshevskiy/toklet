"""Separate bounded large-file rows; the original4GiB reader stays unchanged.

File extent alone grants no source/model eligibility. Callers authenticate the
descriptor and its lifetime. Reconstruction, row reads and projections inherit
the original qualified arithmetic, with no tensor expansion or memory mapping.
"""
from dataclasses import dataclass
import affine4_tiles as tiles

MAX_FILE_BYTES = 6 * 1024**3


@dataclass(frozen=True)
class LargeMatrixRows(tiles.MatrixRows):
    def __post_init__(self):
        tiles.integer(self.file_bytes, 1, MAX_FILE_BYTES)
        tiles.integer(self.rows, 1, tiles.MAX_ROWS)
        tiles.integer(self.columns, 64, tiles.MAX_COLUMNS)
        if self.columns % 64:
            raise ValueError('Complete group64 rows required')
        spans = []
        for offset, stride in self.components():
            tiles.integer(offset, 0, self.file_bytes)
            end = offset + self.rows * stride
            if end > self.file_bytes:
                raise ValueError('Affine component exceeds source extent')
            spans.append((offset, end))
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError('Affine component spans overlap')
