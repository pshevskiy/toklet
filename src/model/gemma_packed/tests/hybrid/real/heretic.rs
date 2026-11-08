//! Explicit quality-only owners. Every process selects one separately pinned source.
use super::*;
use crate::metal::gemma_affine::{Bf16Context, Bf16Weights, resident::heretic as mlx};
use crate::mlx::gemma4::{bf16::modified_parent as parent, uncensored};
use std::{fs::File, path::PathBuf};

pub(super) const BUDGET: usize = PROCESS - DIAGNOSTIC;

pub(super) struct Held {
    file: File,
    path: PathBuf,
    original: [u64; 7],
}
impl Held {
    fn open(environment: &str) -> Self {
        let path = PathBuf::from(
            std::env::var_os(environment).expect("explicit quality source directory"),
        )
        .join("model.safetensors");
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&path)
            .unwrap();
        let metadata = file.metadata().unwrap();
        assert!(metadata.is_file());
        let held = Self {
            file,
            path,
            original: identity(&metadata),
        };
        held.verify();
        held
    }
    pub(super) fn verify(&self) {
        assert_eq!(identity(&self.file.metadata().unwrap()), self.original);
        assert_eq!(
            identity(&std::fs::symlink_metadata(&self.path).unwrap()),
            self.original
        );
    }
}

pub(super) fn modified() -> bool {
    match std::env::var("TOKLET_HERETIC_QUALITY_SOURCE")
        .unwrap()
        .as_str()
    {
        "modified-bf16" => true,
        "quantized-mlx" => false,
        _ => panic!("named quality source required"),
    }
}

pub(super) fn load_parent() -> (Bf16Weights, Bf16Context, Held) {
    let held = Held::open("TOKLET_HERETIC_PARENT_DIRECTORY");
    let config = std::fs::read(held.path.with_file_name("config.json")).unwrap();
    let context = Bf16Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    let source = Bf16Weights::load_modified_parent_for_quality(
        &context,
        held.file.try_clone().unwrap(),
        &config,
        BUDGET,
    )
    .unwrap();
    held.verify();
    assert!(source.report().admitted_load_bytes + DIAGNOSTIC <= PROCESS);
    emit(
        "HERETIC_QUALITY_SOURCE",
        json!({"kind":"modified-bf16",
        "repository":parent::REPOSITORY,"revision":parent::REVISION,"sha256":parent::CONTAINER_SHA256,
        "load_admission_bytes":source.report().admitted_load_bytes,"quality_qualified":false}),
    );
    footprint();
    (source, context, held)
}

pub(super) fn load_mlx() -> (mlx::HereticWeights, mlx::HereticContext, Held) {
    let mut held = Held::open("TOKLET_GEMMA_HERETIC_DIRECTORY");
    let config = std::fs::read(held.path.with_file_name("config.json")).unwrap();
    let context = mlx::HereticContext::new(mlx::CONTEXT_BYTES).unwrap();
    let source = mlx::HereticWeights::load(&context, &config, &mut held.file, BUDGET).unwrap();
    held.verify();
    assert!(source.report().admitted_load_policy_bytes + DIAGNOSTIC <= PROCESS);
    emit(
        "HERETIC_QUALITY_SOURCE",
        json!({"kind":"quantized-mlx",
        "repository":uncensored::REPOSITORY,"revision":uncensored::REVISION,"sha256":uncensored::CONTAINER_SHA256,
        "load_admission_bytes":source.report().admitted_load_policy_bytes,"quality_qualified":false}),
    );
    footprint();
    (source, context, held)
}

pub(super) fn footprint() {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: flavor4 matches the owned writable Apple structure. There are no
    // concurrent references; success populates it before the read below.
    assert_eq!(
        unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V4,
                info.as_mut_ptr().cast(),
            )
        },
        0
    );
    // SAFETY: successful flavor4 output initialized the zeroed structure.
    let info = unsafe { info.assume_init() };
    assert!(
        info.ri_phys_footprint > 0
            && info.ri_phys_footprint <= info.ri_lifetime_max_phys_footprint
            && info.ri_lifetime_max_phys_footprint <= PROCESS as u64
    );
    emit(
        "HERETIC_QUALITY_MEMORY",
        json!({"physical_bytes":info.ri_phys_footprint,
        "lifetime_peak_bytes":info.ri_lifetime_max_phys_footprint,"limit_bytes":PROCESS}),
    );
}

enum Raw {
    Wrap,
    FirstLoss,
    RemainingLoss,
}

#[test]
#[ignore = "explicit pinned Heretic quality-only1025 wrap and state controls"]
fn quality1025_wrap() {
    raw(Raw::Wrap);
}
#[test]
#[ignore = "explicit pinned Heretic first1024 next-token losses"]
fn quality_first_loss_block() {
    raw(Raw::FirstLoss);
}
#[test]
#[ignore = "explicit pinned Heretic remaining8976 next-token losses"]
fn quality_remaining_loss_blocks() {
    raw(Raw::RemainingLoss);
}

fn raw(operation: Raw) {
    println!();
    if modified() {
        let (source, context, held) = load_parent();
        let before = source.test_retained_digest();
        let model = source.model_for_quality_probe(1025).unwrap();
        run_raw(
            &model,
            |budget| source.new_session(&model, &context, 32, budget),
            operation,
            || {
                held.verify();
                assert_eq!(source.test_retained_digest(), before);
            },
        );
    } else {
        let (source, context, held) = load_mlx();
        let norms = mlx::Norms::new(&source, BUDGET).unwrap();
        assert!(norms.model(513).is_err());
        assert!(norms.model_for_quality_probe(0).is_err());
        assert!(norms.model_for_quality_probe(1026).is_err());
        let model = norms.model_for_quality_probe(1025).unwrap();
        run_raw(
            &model,
            |budget| norms.new_session(&model, &context, 32, budget),
            operation,
            || held.verify(),
        );
    }
    footprint();
}

fn run_raw<'model, 'weights, M, B>(
    model: &'model Model<'weights, M>,
    new_session: impl Fn(usize) -> Result<Session<'model, 'weights, M, B>, hybrid::Error>,
    operation: Raw,
    verify: impl Fn(),
) where
    M: MatrixSource,
    B: Operations<M, Error = hybrid::Error>,
{
    assert!(model.content_identity().is_none());
    let warm = new_session(BUDGET).unwrap();
    let owned = warm.allocated_bytes().unwrap();
    let required = model.model_bytes() + owned;
    drop(warm);
    assert!(required + DIAGNOSTIC <= PROCESS);
    assert!(matches!(new_session(required - 1),
        Err(hybrid::Error::Metal(crate::metal::Error::Budget { required: r, limit }))
        if r == required && limit == required - 1));
    emit(
        "HERETIC_QUALITY_ADMISSION",
        json!({"required_bytes":required,"session_bytes":owned,
        "capacity":1025,"chunk_capacity":32,"diagnostic_bytes":DIAGNOSTIC,"minus_one_rejected":true}),
    );
    match operation {
        Raw::Wrap => super::wrap::run_wrap(model, || new_session(required).unwrap(), owned, verify),
        Raw::FirstLoss | Raw::RemainingLoss => {
            let blocks: Vec<_> = if matches!(operation, Raw::FirstLoss) {
                vec![0]
            } else {
                (1..10).collect()
            };
            let mut session = new_session(required).unwrap();
            super::loss_block::score_blocks(&blocks, &mut session, owned, verify);
        }
    }
}
