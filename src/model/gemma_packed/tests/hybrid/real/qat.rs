//! Explicit pinned mixed-format source and raw diagnostic session.
use super::*;
use crate::metal::gemma_affine::{MixedContext, resident::QatWeights};
use sha2::{Digest, Sha256};
use std::{io::Read, path::PathBuf};

const VOCAB: usize = 262144;
pub(super) const MIXED_ALLOWANCE: usize = 2 * gemma_affine::CONTEXT_ALLOWANCE_BYTES;

pub(super) fn load_qat() -> (QatWeights, std::fs::File, PathBuf, [u64; 7]) {
    let root = PathBuf::from(std::env::var_os("TOKLET_GEMMA_QAT_METADATA").unwrap());
    let config = std::fs::read(root.join("config.json")).unwrap();
    let index = std::fs::read(root.join("model.safetensors.index.json")).unwrap();
    let path = root.join("model/model.safetensors");
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path)
        .unwrap();
    let before = identity(&file.metadata().unwrap());
    assert_eq!(before, identity(&std::fs::symlink_metadata(&path).unwrap()));
    let budget = PROCESS - DIAGNOSTIC;
    let source = autoreleasepool(|_| {
        let loader = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
        let affine8 =
            crate::mlx::affine::Layout::new(2, 64, 8, 64, crate::mlx::affine::MetadataDtype::BF16)
                .unwrap();
        assert!(loader.projection_session(affine8, 2, 1, budget).is_err());
        QatWeights::load_text_only(&loader, &config, &index, &mut file, budget).unwrap()
    });
    (source, file, path, before)
}

#[test]
#[ignore = "explicit mixed Gemma QAT real-model Metal session"]
fn mixed_qat_prefix_is_stable_across_chunks_reset_and_future() {
    let (source, file, path, before) = load_qat();
    let ids = std::env::var_os("TOKLET_GEMMA_QAT_PREFIX").map_or_else(
        || IDS.to_vec(),
        |path| {
            let mut bytes = Vec::new();
            std::fs::File::open(path)
                .unwrap()
                .take(4 * 513)
                .read_to_end(&mut bytes)
                .unwrap();
            assert!(bytes.len().is_multiple_of(4));
            bytes
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect()
        },
    );
    assert!((2..=510).contains(&ids.len()));
    let budget = PROCESS - DIAGNOSTIC;
    let warm = hybrid::QatNorms::new(&source, budget).unwrap();
    let norms_budget = source.report().resident_requested_bytes + warm.owned_bytes();
    drop(warm);
    assert!(matches!(
        hybrid::QatNorms::new(&source, norms_budget - 1),
        Err(crate::metal::Error::Budget { required, limit })
        if required == norms_budget && limit == norms_budget - 1
    ));
    let norms = hybrid::QatNorms::new(&source, norms_budget).unwrap();
    assert!(norms.model(0).is_err());
    assert!(norms.model(hybrid::QatNorms::MAX_POSITIONS + 1).is_err());
    let model = norms.model(hybrid::QatNorms::MAX_POSITIONS).unwrap();
    assert!(model.content_identity().is_none());
    assert!(matches!(
        MixedContext::new(MIXED_ALLOWANCE - 1),
        Err(crate::metal::Error::Budget { required: MIXED_ALLOWANCE, limit })
        if limit == MIXED_ALLOWANCE - 1
    ));
    let context = MixedContext::new(MIXED_ALLOWANCE).unwrap();
    let mut capture = capture::Capture::bounded(32 * 1024 * 1024);
    let mut expected = None;
    let mut extended = ids.clone();
    extended.extend([2, 100]);
    for (run, (chunk_rows, input)) in [
        (32, ids.as_slice()),
        (3, ids.as_slice()),
        (1, ids.as_slice()),
        (32, extended.as_slice()),
    ]
    .into_iter()
    .enumerate()
    {
        let warm = norms
            .new_session(&model, &context, chunk_rows, budget)
            .unwrap();
        let owned = warm.allocated_bytes().unwrap();
        let required = model.model_bytes() + owned;
        drop(warm);
        assert!(matches!(
            norms.new_session(&model, &context, chunk_rows, required - 1),
            Err(hybrid::Error::Metal(crate::metal::Error::Budget { required: r, limit }))
            if r == required && limit == required - 1
        ));
        let mut session = norms
            .new_session(&model, &context, chunk_rows, required)
            .unwrap();
        assert_eq!(session.allocated_bytes().unwrap(), owned);
        assert_eq!(
            (session.capacity(), session.chunk_capacity()),
            (512, chunk_rows)
        );
        let mut digest = Sha256::new();
        for (block, tokens) in input.chunks(chunk_rows).enumerate() {
            let offset = block * chunk_rows;
            let rows = session.forward(tokens).unwrap();
            for (i, row) in rows.chunks_exact(VOCAB).enumerate() {
                let position = offset + i;
                if position >= ids.len() {
                    break;
                }
                for value in row {
                    digest.update(value.to_le_bytes());
                }
                if run == 0 && (position == 0 || position == ids.len() - 1) {
                    capture.record("logits", row, position, 1);
                }
            }
            assert_eq!(session.tokens(), &input[..offset + tokens.len()]);
        }
        for (owner, cache) in session.caches.iter().enumerate() {
            let (keys, values) = cache.committed();
            let count = ids.len() * model.layout.layers[owner].head_dim;
            for (tag, part) in [
                ("k_rotary", &keys[..count]),
                ("self_attn.v_norm", &values[..count]),
            ] {
                for value in part {
                    digest.update(value.to_le_bytes());
                }
                if run == 0 {
                    capture.record(&format!("layer.{owner}.{tag}"), part, 0, ids.len());
                }
            }
        }
        let digest = digest.finalize();
        if let Some(first) = expected {
            assert_eq!(digest, first, "committed causal rows or KV changed");
        } else {
            expected = Some(digest);
        }
        assert_eq!(session.allocated_bytes().unwrap(), owned);
        let first = session.forward(&[]).err();
        assert!(first.is_some());
        assert_eq!(session.tokens(), input);
        session.reset().unwrap();
        assert!(session.tokens().is_empty());
        assert!(session.caches.iter().all(|cache| cache.offset() == 0));
        assert_eq!(session.allocated_bytes().unwrap(), owned);
        emit(
            "QAT_PREFIX",
            json!({
                "run":run,"tokens":input.len(),"causal_tokens":ids.len(),"chunk_rows":chunk_rows,
                "state_sha256":format!("{digest:x}"),"required_bytes":required,"session_bytes":owned
            }),
        );
    }
    capture.finish();
    assert_eq!(before, identity(&file.metadata().unwrap()));
    assert_eq!(before, identity(&std::fs::symlink_metadata(&path).unwrap()));
}
