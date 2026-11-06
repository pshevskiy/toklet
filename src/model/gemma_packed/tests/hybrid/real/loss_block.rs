//! Frozen quality blocks; aggregate adoption quality is decided separately.
use super::*;
use sha2::{Digest, Sha256};

#[allow(
    clippy::duplicate_mod,
    reason = "Independent test harnesses reuse the same frozen NLL implementation"
)]
#[path = "../../../../../../examples/support/nll.rs"]
mod nll;

#[test]
#[ignore = "explicit real E2B diagnostic1025 next-token loss block"]
fn diagnostic1025_first_loss_block() {
    run_blocks(&[0]);
}

#[test]
#[ignore = "explicit real E2B remaining nine next-token loss blocks"]
fn diagnostic1025_remaining_loss_blocks() {
    run_blocks(&(1..10).collect::<Vec<_>>());
}

fn run_blocks(blocks: &[usize]) {
    let (source, file, path, original) = load_original();
    let owners = resident_snapshot(&source);
    let norms = hybrid::Norms::new(&source, PROCESS - DIAGNOSTIC).unwrap();
    let model = norms.model_for_context_probe(1025).unwrap();
    assert!(model.content_identity().is_none());
    let context = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    let mut session = norms
        .new_session(&model, &context, PROCESS - DIAGNOSTIC)
        .unwrap();
    score_blocks(blocks, &mut session, 271117548, || {
        assert_eq!(resident_snapshot(&source), owners);
    });
    assert_eq!(identity(&file.metadata().unwrap()), original);
    assert_eq!(
        identity(&std::fs::symlink_metadata(path).unwrap()),
        original
    );
}

#[test]
#[ignore = "explicit mixed QAT first original-loss block"]
fn mixed_qat_first_loss_block() {
    run_qat_blocks(&[0]);
}

#[test]
#[ignore = "explicit mixed QAT remaining nine original-loss blocks"]
fn mixed_qat_remaining_loss_blocks() {
    run_qat_blocks(&(1..10).collect::<Vec<_>>());
}

fn run_qat_blocks(blocks: &[usize]) {
    use crate::metal::gemma_affine::MixedContext;
    use hybrid::test_support::qat_resident_snapshot;

    println!();
    let (source, file, path, original) = super::qat::load_qat();
    let owners = qat_resident_snapshot(&source);
    let budget = PROCESS - DIAGNOSTIC;
    let norms = hybrid::QatNorms::new(&source, budget).unwrap();
    let model = norms.model_for_quality_probe(1025).unwrap();
    let context = MixedContext::new(super::qat::MIXED_ALLOWANCE).unwrap();
    let warm = norms.new_session(&model, &context, 32, budget).unwrap();
    let owned = warm.allocated_bytes().unwrap();
    let required = model.model_bytes() + owned;
    drop(warm);
    assert!(matches!(
        norms.new_session(&model, &context, 32, required - 1),
        Err(hybrid::Error::Metal(crate::metal::Error::Budget { required: r, limit }))
        if r == required && limit == required - 1
    ));
    let mut session = norms.new_session(&model, &context, 32, required).unwrap();
    emit(
        "QAT_LOSS_ADMISSION",
        json!({"required_bytes":required,"session_bytes":owned}),
    );
    score_blocks(blocks, &mut session, owned, || {});
    assert_eq!(qat_resident_snapshot(&source), owners);
    assert_eq!(identity(&file.metadata().unwrap()), original);
    assert_eq!(
        identity(&std::fs::symlink_metadata(path).unwrap()),
        original
    );
}

fn score_blocks(
    blocks: &[usize],
    session: &mut hybrid::Session<'_, '_, '_, '_>,
    expected_allocated: usize,
    verify_owners: impl Fn(),
) {
    let frozen =
        include_bytes!("../../../../../../benchmarks/gemma-quality-v1/held_out_loss.u32le");
    let prefix = include_bytes!("../../../../../../benchmarks/gemma-e2b-wrap-v1/tokens.u32le");
    assert_eq!(frozen.len(), 40004);
    assert_eq!(&frozen[..4100], prefix);
    let all_ids: Vec<u32> = frozen
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let mut capture = capture::Capture::bounded(DIAGNOSTIC);
    let pointers_before = pointers(session);
    let allocated = session.allocated_bytes().unwrap();
    assert_eq!(allocated, expected_allocated);
    let anchors = [0, 31, 255, 511, 512, 767, 1023, 1024];
    let mut total = nll::Sum::default();
    let mut scored = 0;
    for &block in blocks {
        assert!(block < 10);
        let start = block * 1024;
        let finish = ((block + 1) * 1024).min(10000);
        let ids = &all_ids[start..=finish];
        session.reset().unwrap();
        assert!(session.tokens().is_empty());
        if blocks != [0] {
            emit(
                "LOSS_BLOCK_START",
                json!({"block":block,"input_start":start,
                "targets":finish-start,"positions":ids.len(),"history_empty":true}),
            );
        }
        for offset in (0..ids.len()).step_by(32) {
            let end = (offset + 32).min(ids.len());
            let rows = session.forward(&ids[offset..end]).unwrap();
            assert_eq!(rows.len(), (end - offset) * 262144);
            for position in offset..end {
                let begin = (position - offset) * 262144;
                let row = &rows[begin..begin + 262144];
                let mut digest = Sha256::new();
                for &value in row {
                    assert!(value.is_finite());
                    digest.update(value.to_le_bytes());
                }
                let row_sha256 = format!("{:x}", digest.finalize());
                if block == 0 && anchors.contains(&position) {
                    capture.record("loss.anchor", row, position, 1);
                }
                if position == ids.len() - 1 {
                    emit(
                        "LOSS_UNSCORED",
                        if blocks == [0] {
                            json!({"position":position,"row_sha256":row_sha256})
                        } else {
                            json!({"block":block,"position":position,"row_sha256":row_sha256})
                        },
                    );
                    continue;
                }
                let target = ids[position + 1] as usize;
                let (maximum, exp_mass, value) = nll::statistics(row, target).unwrap();
                let greedy = row.iter().position(|&v| f64::from(v) == maximum).unwrap();
                emit(
                    "LOSS_TARGET",
                    json!({"block":block,"local_row":position,"global_target_index":start+position+1,
                "target":target,"vocab":262144,"target_fp32_bits":row[target].to_bits(),
                "maximum":maximum,"exp_mass":exp_mass,"nll":value,"greedy_lowest_id":greedy,
                "row_sha256":row_sha256}),
                );
                total.add(value);
                scored += 1;
            }
            assert_eq!(session.tokens(), &ids[..end]);
            assert_eq!(pointers(session), pointers_before);
            assert_eq!(session.allocated_bytes().unwrap(), allocated);
        }
        assert_eq!(session.tokens(), ids);
        verify_owners();
    }
    assert_eq!(
        scored,
        blocks
            .iter()
            .map(|&b| ((b + 1) * 1024).min(10000) - b * 1024)
            .sum::<usize>()
    );
    verify_owners();
    emit(
        "LOSS_SUMMARY",
        json!({"targets":scored,"mean_nll":total.value/scored as f64,
        "capacity":1025,"chunk_capacity":32,"session_allocated_bytes":allocated,
        "quality_qualified":false}),
    );
    capture.finish();
}
