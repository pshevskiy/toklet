//! Same-input local V boundaries, without earlier-layer tracing or promotion.
use super::*;

#[test]
#[ignore = "explicit original E2B same-input V projection and normalization capture"]
fn native_v_projection_norm_and_retained_rows() {
    let raw = include_bytes!("../../../../../../benchmarks/gemma-v-boundary-v1/tokens.u32le");
    let ids: Vec<u32> = raw
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(ids.len(), 651);
    let cases = [(13, 433), (12, 650), (14, 650)];
    let mut capture = capture::Capture::bounded(1024 * 1024);
    let (source, file, path, original) = load_original();
    let owners = resident_snapshot(&source);
    let norms = hybrid::Norms::new(&source, PROCESS - DIAGNOSTIC).unwrap();
    let model = norms.model_for_context_probe(1025).unwrap();
    assert!(model.content_identity().is_none());
    let context = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    let mut session = norms
        .new_session(&model, &context, PROCESS - DIAGNOSTIC)
        .unwrap();
    assert_eq!((session.capacity(), session.chunk_capacity()), (1025, 32));
    let allocated = session.allocated_bytes().unwrap();
    let pointers_before = pointers(&session);
    let mut saved = Vec::with_capacity(3);
    emit(
        "VBOUNDARY_CONTRACT",
        json!({"cases":cases,"input_tokens":651,"capacity":1025,"chunk_rows":32}),
    );
    for (chunk, input) in ids.chunks(32).enumerate() {
        let before = chunk * 32;
        for (owner, position) in cases {
            if !(before..before + input.len()).contains(&position) {
                continue;
            }
            let prior = snapshot(&session);
            session.fault = Some(owner);
            assert!(matches!(
                session.forward(input),
                Err(hybrid::Error::Model(Error::InvalidConfig(
                    "injected Gemma layer failure"
                )))
            ));
            assert_eq!(snapshot(&session), prior);
            assert_eq!(pointers(&session), pointers_before);
            assert_eq!(session.allocated_bytes().unwrap(), allocated);
            let row = position - before;
            let dim = model.layout.layers[owner].head_dim;
            emit(
                "VBOUNDARY_CASE",
                json!({"owner":owner,"position":position,"before":before,
                "count":input.len(),"head_dim":dim,"fault_preserved_prefix":true}),
            );
            // Producer finish uses dense_work/dense_norm and leaves both the
            // normalized projection input and raw V scratch intact at this hook.
            for (name, values) in [
                ("input", &session.scratch.norm[row * H..(row + 1) * H]),
                ("raw", &session.scratch.kv[row * dim..(row + 1) * dim]),
                (
                    "normalized",
                    &session.staging[owner].v[row * dim..(row + 1) * dim],
                ),
            ] {
                capture.record(&format!("vboundary.{owner}.{name}"), values, position, 1);
            }
            saved.push((
                owner,
                position,
                session.staging[owner].v[row * dim..(row + 1) * dim].to_vec(),
            ));
        }
        session.fault = None;
        session.forward(input).unwrap();
        assert_eq!(session.tokens(), &ids[..before + input.len()]);
        for cache in &session.caches {
            assert_eq!(
                (cache.offset(), cache.generation()),
                (before + input.len(), chunk as u64 + 1)
            );
        }
        assert_eq!(pointers(&session), pointers_before);
        assert_eq!(session.allocated_bytes().unwrap(), allocated);
    }
    assert_eq!(saved.len(), 3);
    for (owner, position, normalized) in saved {
        let dim = model.layout.layers[owner].head_dim;
        let cache = &session.caches[owner];
        let values = cache.committed().1;
        let capacity = values.len() / dim;
        let slot = position % capacity;
        assert_eq!(cache.position_in_slot(slot), Some(position));
        let row = &values[slot * dim..(slot + 1) * dim];
        assert!(
            row.iter()
                .zip(&normalized)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        capture.record(&format!("vboundary.{owner}.retained"), row, position, 1);
    }
    assert_eq!(resident_snapshot(&source), owners);
    assert_eq!(original, identity(&file.metadata().unwrap()));
    assert_eq!(
        original,
        identity(&std::fs::symlink_metadata(path).unwrap())
    );
    capture.finish();
    emit(
        "VBOUNDARY_COMPLETE",
        json!({"committed":651,"all3_stage_retained_exact":true,
        "allocated_bytes":allocated,"numerical_decision":"separate same-input replay",
        "earlier_failures_waived":false,"app_qualified":false,"phone_qualified":false}),
    );
}
