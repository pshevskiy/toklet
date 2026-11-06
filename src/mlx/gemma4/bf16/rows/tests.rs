use super::*;
use std::{
    fs::{self, OpenOptions},
    os::unix::fs::OpenOptionsExt,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

const START: usize = SCAN_BYTES - 1;
const ROWS: usize = 8;
const COLUMNS: usize = 5000;
const WORDS: [u16; 8] = [
    0x3f80, 0x4000, 0xbf80, 0x0001, 0x4180, 0xc000, 0x0000, 0x8000,
];

struct TestFile(PathBuf);

impl TestFile {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "toklet-bf16-ple-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> PathBuf {
        self.0.join("source")
    }
}

impl Drop for TestFile {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixture() -> (TestFile, Vec<u8>, Table) {
    let mut raw = vec![0x13; START];
    for value in WORDS {
        for _ in 0..COLUMNS {
            raw.extend(value.to_le_bytes());
        }
    }
    raw.extend([0x88; 177]);
    let location = TestFile::new();
    fs::write(location.path(), &raw).unwrap();
    let table = Table::new(START as u64, ROWS, COLUMNS, raw.len() as u64).unwrap();
    (location, raw, table)
}

fn run(location: &TestFile, table: Table, expected: &str, budget: usize) -> Result<Source, Error> {
    let file = File::open(location.path()).unwrap();
    let before = file_identity(&file).unwrap();
    authenticate(file, before, table, expected, budget)
}

#[test]
fn exact_row_budgets_and_late_corruption_preserve_caller_output() {
    let (location, raw, table) = fixture();
    let expected = format!("{:x}", Sha256::digest(&raw));
    let need = memory(ROWS, SCAN_BYTES).unwrap();
    assert!(matches!(
        run(&location, table, &expected, need.load_bytes - 1),
        Err(Error::Budget { required, limit })
        if required == need.load_bytes && limit == need.load_bytes - 1
    ));
    let source = run(&location, table, &expected, need.load_bytes).unwrap();
    assert_eq!(source.memory(), need);
    let required = size_of::<Workspace>() + 3 * table.row_bytes;
    assert!(matches!(
        source.workspace(3, required - 1),
        Err(Error::Budget { required: r, limit })
        if r == required && limit == required - 1
    ));
    let mut workspace = source.workspace(3, required).unwrap();
    assert_eq!(workspace.owned_bytes(), required);
    let mut output = vec![0.; 3 * COLUMNS];
    source
        .gather(&mut workspace, &[6, 3, 6], &mut output)
        .unwrap();
    for (row, id) in output.chunks_exact(COLUMNS).zip([6, 3, 6]) {
        assert!(
            row.iter()
                .all(|v| v.to_bits() == u32::from(WORDS[id]) << 16)
        );
    }
    let writer = OpenOptions::new()
        .write(true)
        .open(location.path())
        .unwrap();
    writer
        .write_all_at(&[0x01], table.start + 7 * table.row_bytes as u64)
        .unwrap();
    output.fill(3.);
    assert!(
        source
            .gather(&mut workspace, &[0, 2, 7], &mut output)
            .is_err()
    );
    assert!(output.iter().all(|&v| v == 3.));
    assert!(
        source
            .gather(&mut workspace, &[0, 2, 8], &mut output)
            .is_err()
    );
    assert!(output.iter().all(|&v| v == 3.));

    fs::rename(location.path(), location.0.join("held")).unwrap();
    fs::write(location.path(), [0u8; 1]).unwrap();
    source
        .gather(&mut workspace, &[0], &mut output[..COLUMNS])
        .unwrap();
    assert!(output[..COLUMNS].iter().all(|&v| v == 1.));
    writer
        .set_len(table.start + 7 * table.row_bytes as u64)
        .unwrap();
    output.fill(3.);
    assert!(
        source
            .gather(&mut workspace, &[0, 2, 7], &mut output)
            .is_err()
    );
    assert!(output.iter().all(|&v| v == 3.));
}

#[test]
fn whole_file_pin_and_finite_row_checks_precede_publication() {
    let (location, mut raw, table) = fixture();
    let expected = format!("{:x}", Sha256::digest(&raw));
    raw[0] ^= 1;
    fs::write(location.path(), &raw).unwrap();
    assert!(matches!(
        run(&location, table, &expected, usize::MAX),
        Err(Error::Invalid(reason)) if reason.contains("source authentication")
    ));

    raw[START..START + 2].copy_from_slice(&0x7fc0_u16.to_le_bytes());
    fs::write(location.path(), &raw).unwrap();
    let malformed = format!("{:x}", Sha256::digest(&raw));
    assert!(matches!(
        run(&location, table, &malformed, usize::MAX),
        Err(Error::Invalid(reason)) if reason.contains("nonfinite")
    ));
}

#[test]
#[ignore = "explicit official BF16 full-source authentication and exact PLE rows"]
fn actual_official_bf16_ple_gather_matches_original_rows() {
    let root = PathBuf::from(std::env::var_os("TOKLET_GEMMA_BF16_MODEL").unwrap());
    let path = root.join("model.safetensors");
    let config = fs::read(root.join("config.json")).unwrap();
    let open = || {
        OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&path)
            .unwrap()
    };
    let required = memory(super::super::super::VOCAB, SCAN_BYTES).unwrap();
    assert!(matches!(
        Source::open_text_only(open(), &config, required.load_bytes - 1),
        Err(Error::Budget { required: r, limit })
        if r == required.load_bytes && limit == r - 1
    ));
    let source = Source::open_text_only(open(), &config, required.load_bytes).unwrap();
    assert_eq!(source.memory(), required);
    assert_eq!(source.table.start, 1_422_298_392);
    let mut workspace = source
        .workspace(32, size_of::<Workspace>() + 32 * 17920)
        .unwrap();
    let ids = [262143, 2, 257, 2, 0];
    let mut output = vec![0.; ids.len() * 8960];
    source.gather(&mut workspace, &ids, &mut output).unwrap();
    let original = open();
    let mut raw = vec![0u8; 17920];
    for (&id, row) in ids.iter().zip(output.chunks_exact(8960)) {
        original
            .read_exact_at(&mut raw, 1_422_298_392 + u64::from(id) * 17920)
            .unwrap();
        for (raw, got) in raw.chunks_exact(2).zip(row) {
            assert_eq!(got.to_bits().to_be_bytes(), [raw[1], raw[0], 0, 0]);
        }
    }
    println!(
        "BF16_PLE_REAL retained={} load={} workspace={} rows={}",
        required.retained_bytes,
        required.load_bytes,
        workspace.owned_bytes(),
        ids.len()
    );
}
