//! Same-input attention replay at the pre-eviction layer-13 V boundary.
use super::*;

#[test]
#[ignore = "explicit original E2B layer12 pre-eviction capture"]
fn native_layer12_attention_at_433() {
    const LAYER: usize = 12;
    const POSITION: usize = 433;
    const BEFORE: usize = 416;
    const ROWS: usize = 32;
    const DIM: usize = 256;
    let raw = include_bytes!("../../../../../../benchmarks/gemma-v-boundary-v1/tokens.u32le");
    let ids: Vec<_> = raw[..(BEFORE + ROWS) * 4]
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let (source, file, path, original) = load_original();
    let owners = resident_snapshot(&source);
    let norms = hybrid::Norms::new(&source, PROCESS - DIAGNOSTIC).unwrap();
    let model = norms.model_for_context_probe(1025).unwrap();
    assert_eq!(model.layout.layers[LAYER].head_dim, DIM);
    assert_eq!(model.layout.heads, HEADS);
    let context = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    let mut session = norms
        .new_session(&model, &context, PROCESS - DIAGNOSTIC)
        .unwrap();
    for input in ids[..BEFORE].chunks(ROWS) {
        session.forward(input).unwrap();
    }
    let old = snapshot(&session);
    let owned = pointers(&session);
    session.fault = Some(LAYER);
    assert!(matches!(
        session.forward(&ids[BEFORE..]),
        Err(hybrid::Error::Model(Error::InvalidConfig(
            "injected Gemma layer failure"
        )))
    ));
    assert_eq!(snapshot(&session), old);
    assert_eq!(pointers(&session), owned);
    // The producer finishes attention and its O projection before the fault.
    // Reborrow the identical cache/staging spans without committing the frame.
    let cache = &session.caches[LAYER];
    let stage = &session.staging[LAYER];
    let frame = cache
        .prepare(&stage.k[..ROWS * DIM], &stage.v[..ROWS * DIM], ROWS)
        .unwrap();
    let view = cache.view(&frame).unwrap();
    assert_eq!((view.start(), view.end()), (0, BEFORE + ROWS));
    let row = POSITION - BEFORE;
    let q = row * HEADS * DIM;
    let mut capture = capture::Capture::bounded(2 * 1024 * 1024);
    emit(
        "ATTENTION_BOUNDARY",
        json!({"layer":LAYER,"producer":LAYER,"position":POSITION,
        "chunk_start":BEFORE,"chunk_rows":ROWS,"head_dim":DIM,"heads":HEADS,"kv_heads":1,
        "window":model.layout.window,"start":view.start(),"end":view.end()}),
    );
    capture.record(
        "attention.q",
        &session.scratch.q[q..q + HEADS * DIM],
        POSITION,
        1,
    );
    capture.record(
        "attention.context",
        &session.scratch.context[q..q + HEADS * DIM],
        POSITION,
        1,
    );
    capture.record(
        "attention.output",
        &session.scratch.attention[row * H..(row + 1) * H],
        POSITION,
        1,
    );
    for (i, segment) in view.segments().iter().enumerate() {
        if segment.keys != 0 {
            assert_eq!(segment.k.len(), segment.keys * DIM);
            assert_eq!(segment.v.len(), segment.keys * DIM);
            capture.record(
                &format!("attention.{i}.k"),
                segment.k,
                segment.key_start,
                segment.keys,
            );
            capture.record(
                &format!("attention.{i}.v"),
                segment.v,
                segment.key_start,
                segment.keys,
            );
        }
    }
    let saved: Vec<_> = view
        .segments()
        .iter()
        .map(|segment| (segment.k.to_vec(), segment.v.to_vec()))
        .collect();
    session.fault = None;
    session.forward(&ids[BEFORE..]).unwrap();
    assert_eq!(session.tokens(), ids);
    let (keys, values) = session.caches[LAYER].committed();
    let mut offset = 0;
    for (k, v) in saved {
        assert_eq!(&keys[offset..offset + k.len()], k);
        assert_eq!(&values[offset..offset + v.len()], v);
        offset += k.len();
    }
    assert_eq!(offset, ids.len() * DIM);
    assert_eq!(pointers(&session), owned);
    assert_eq!(resident_snapshot(&source), owners);
    assert_eq!(original, identity(&file.metadata().unwrap()));
    assert_eq!(
        original,
        identity(&std::fs::symlink_metadata(path).unwrap())
    );
    emit(
        "ATTENTION_BOUNDARY_COMPLETE",
        json!({"committed":ids.len(),"fault_preserved_prefix":true,
        "committed_operands_equal":true}),
    );
    capture.finish();
}

#[test]
#[ignore = "explicit original E2B layer12 post-attention capture"]
fn native_layer12_tail_at_433() {
    const POSITION: usize = 433;
    const BEFORE: usize = 416;
    const ROWS: usize = 32;
    let raw = include_bytes!("../../../../../../benchmarks/gemma-v-boundary-v1/tokens.u32le");
    let ids: Vec<_> = raw[..(BEFORE + ROWS) * 4]
        .chunks_exact(4)
        .map(|value| u32::from_le_bytes(value.try_into().unwrap()))
        .collect();
    let (source, file, path, original) = load_original();
    let owners = resident_snapshot(&source);
    let norms = hybrid::Norms::new(&source, PROCESS - DIAGNOSTIC).unwrap();
    let model = norms.model_for_context_probe(1025).unwrap();
    let context = gemma_affine::Context::new(gemma_affine::CONTEXT_ALLOWANCE_BYTES).unwrap();
    let mut session = norms
        .new_session(&model, &context, PROCESS - DIAGNOSTIC)
        .unwrap();
    for input in ids[..BEFORE].chunks(ROWS) {
        session.forward(input).unwrap();
    }
    let committed = snapshot(&session);
    let owned = pointers(&session);
    let row = POSITION - BEFORE;
    let h = row * H..(row + 1) * H;
    let p = row * P..(row + 1) * P;
    let intermediate = model.layout.layers[12].intermediate;
    let ff = row * intermediate..(row + 1) * intermediate;
    let mut capture = capture::Capture::bounded(64 * 1024);
    emit(
        "LAYER_TAIL",
        json!({"layer":12,"position":POSITION,"chunk_start":BEFORE,
            "chunk_rows":ROWS,"context":1025,"hidden":H,"ple":P,"intermediate":intermediate}),
    );
    for layer in [11, 12] {
        session.fault = Some(layer);
        assert!(matches!(
            session.forward(&ids[BEFORE..]),
            Err(hybrid::Error::Model(Error::InvalidConfig(
                "injected Gemma layer failure"
            )))
        ));
        assert_eq!(snapshot(&session), committed);
        assert_eq!(pointers(&session), owned);
        let s = &session.scratch;
        if layer == 11 {
            capture.record("tail.input", &s.hidden[h.clone()], POSITION, 1);
        } else {
            for (name, values) in [
                ("attention", &s.attention[h.clone()]),
                ("ple", &s.layer_ple[p.clone()]),
                ("ff_up", &s.up[ff.clone()]),
                ("ple_gate", &s.gate[p.clone()]),
                ("ple_projection", &s.dense_work[h.clone()]),
                ("ple_norm", &s.dense_norm[h.clone()]),
                ("output", &s.hidden[h.clone()]),
            ] {
                capture.record(&format!("tail.{name}"), values, POSITION, 1);
            }
        }
    }
    session.fault = None;
    session.forward(&ids[BEFORE..]).unwrap();
    assert_eq!(session.tokens(), ids);
    assert_eq!(pointers(&session), owned);
    assert_eq!(resident_snapshot(&source), owners);
    assert_eq!(original, identity(&file.metadata().unwrap()));
    assert_eq!(
        original,
        identity(&std::fs::symlink_metadata(path).unwrap())
    );
    emit(
        "LAYER_TAIL_COMPLETE",
        json!({"committed":ids.len(),"both_faults_preserved_prefix":true}),
    );
    capture.finish();
}
