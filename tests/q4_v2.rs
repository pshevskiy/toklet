use toklet::q4::{Format, Matrix};
const PACKED: &[u8] = include_bytes!("fixtures/q4-v2/rows.q4");

#[test]
fn group_eight_manual_bytes_reconstruct_and_multiply_without_expansion() {
    let matrix = Matrix::parse(PACKED).unwrap();
    assert_eq!(matrix.format(), Format::V2);
    assert_eq!(
        (matrix.rows(), matrix.columns(), matrix.row_stride()),
        (2, 17, 64)
    );
    let row = [
        0., 1., 2., 3., 4., 5., 6., 15., 0., 0., 0., 0., 0., 0., 0., 0., 3.25,
    ];
    for (i, &expected) in row.iter().enumerate() {
        assert_eq!(matrix.value(0, i).unwrap(), expected);
        assert_eq!(matrix.value(1, i).unwrap(), 0.);
    }
    let mut x = [1.; 17];
    x[16] = 2.;
    let mut y = [42.; 2];
    matrix.matvec(&x, &mut y).unwrap();
    assert_eq!(y, [42.5, 0.]);
    assert!(matrix.value(0, 17).is_err());
    assert!(matrix.value(2, 0).is_err());
    x[10] = f32::NAN;
    assert!(matrix.matvec(&x, &mut y).is_err());
    assert_eq!(y, [42.5, 0.]);
    x.fill(f32::MAX);
    assert!(matrix.matvec(&x, &mut y).is_err());
    assert_eq!(y, [42.5, 0.]);
    assert!(matrix.matvec(&[], &mut y).is_err());
    assert_eq!(y, [42.5, 0.]);
}

#[test]
fn group_eight_rejects_version_mixing_padding_metadata_and_overflow() {
    for offset in [0, 8, 12, 32, 40, 48, 85, 88] {
        let mut data = PACKED.to_vec();
        data[offset] ^= 1;
        assert!(Matrix::parse(&data).is_err(), "offset {offset}");
    }
    for offset in [16, 24] {
        for value in [0u64, u64::MAX] {
            let mut data = PACKED.to_vec();
            data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            assert!(Matrix::parse(&data).is_err());
        }
    }
    for (offset, bits) in [(64, 0x8000u16), (64, 0x7c00), (66, 0x7e00)] {
        let mut data = PACKED.to_vec();
        data[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
        assert!(Matrix::parse(&data).is_err());
    }
    let mut data = PACKED.to_vec();
    data[76] = 1;
    assert!(Matrix::parse(&data).is_err());
    for len in [0, 63, 64, 191] {
        assert!(Matrix::parse(&PACKED[..len]).is_err());
    }
    let mut data = PACKED.to_vec();
    data.push(0);
    assert!(Matrix::parse(&data).is_err());
    let mut data = PACKED.to_vec();
    data[64..66].copy_from_slice(&half::f16::from_f32(8736.).to_bits().to_le_bytes());
    data[66..68].copy_from_slice(&half::f16::from_f32(-65504.).to_bits().to_le_bytes());
    assert_eq!(Matrix::parse(&data).unwrap().value(0, 7).unwrap(), 65536.);
    assert_eq!(
        Matrix::parse(include_bytes!("fixtures/q4/rows.q4"))
            .unwrap()
            .format(),
        Format::V1
    );
}

#[test]
fn group_eight_manual_records_cover_group_and_alignment_boundaries() {
    for columns in [1usize, 7, 8, 9, 15, 16, 63, 64, 65] {
        let groups = columns.div_ceil(8);
        let stride = (groups * 8).div_ceil(64) * 64;
        let mut bytes = vec![0u8; 64 + stride];
        bytes[..8].copy_from_slice(b"TOKQ4V2\0");
        bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&8u32.to_le_bytes());
        for (offset, value) in [(16, 1usize), (24, columns), (32, stride), (40, stride)] {
            bytes[offset..offset + 8].copy_from_slice(&(value as u64).to_le_bytes());
        }
        for column in 0..columns {
            let record = 64 + column / 8 * 8;
            bytes[record..record + 2].copy_from_slice(&half::f16::ONE.to_bits().to_le_bytes());
            bytes[record + 4 + column % 8 / 2] |= ((column % 8) as u8) << (4 * (column % 2));
        }
        let matrix = Matrix::parse(&bytes).unwrap();
        assert_eq!(matrix.row_stride(), stride);
        for i in 0..columns {
            assert_eq!(matrix.value(0, i).unwrap(), (i % 8) as f32);
        }
        let mut y = [-1.];
        matrix.matvec(&vec![1.; columns], &mut y).unwrap();
        assert_eq!(y[0], (0..columns).map(|i| (i % 8) as f32).sum::<f32>());
        if columns % 8 != 0 {
            let lane = columns % 8;
            bytes[64 + (groups - 1) * 8 + 4 + lane / 2] |= 1 << (4 * (lane % 2));
            assert!(Matrix::parse(&bytes).is_err());
        }
    }
}
