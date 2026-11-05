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

#[test]
#[ignore = "explicit original E2B earliest retained staged difference"]
fn native_prefix64_at_48() {
    capture_prefix64(false);
}

#[test]
#[ignore = "explicit original E2B layer3 same-input feed-forward replay"]
fn native_layer3_ff_at_48() {
    capture_prefix64(true);
}

fn capture_prefix64(replay_ff: bool) {
    const POSITION: usize = 48;
    const ROWS: usize = 32;
    let raw = include_bytes!("../../../../../../benchmarks/gemma-v-boundary-v1/tokens.u32le");
    let ids: Vec<_> = raw[..2 * ROWS * 4]
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
    session.forward(&ids[..ROWS]).unwrap();
    let committed = snapshot(&session);
    let owned = pointers(&session);
    let row = POSITION - ROWS;
    let at = |width| row * width..(row + 1) * width;
    let mut capture = capture::Capture::bounded(256 * 1024);
    emit(
        if replay_ff {
            "LAYER3_FF"
        } else {
            "FIRST_STAGED"
        },
        json!({"position":POSITION,"prefix":ids.len(),"chunk":ROWS,"context":1025}),
    );
    let layers: &[usize] = if replay_ff { &[3] } else { &[2, 3, 4] };
    for &layer in layers {
        session.fault = Some(layer);
        assert!(matches!(
            session.forward(&ids[ROWS..]),
            Err(hybrid::Error::Model(Error::InvalidConfig(
                "injected Gemma layer failure"
            )))
        ));
        assert_eq!(snapshot(&session), committed);
        assert_eq!(pointers(&session), owned);
        let s = &session.scratch;
        let g = &model.layout.layers[layer];
        if replay_ff {
            let w = &model.weights.layers[layer];
            let h = ROWS * H;
            let ff = ROWS * g.intermediate;
            let mut residual = vec![0.; h];
            let mut normalized = vec![0.; h];
            let mut gate = vec![0.; ff];
            let mut product = vec![0.; ff];
            let mut down = vec![0.; h];
            let mut after_ff = vec![0.; h];
            let mut ple_gate = vec![0.; ROWS * P];
            let ops = &mut session.operations;
            // After finish swaps the buffers, next holds the incoming hidden.
            cpu::rmsnorm(
                &s.attention[..h],
                w.post_attention_norm,
                &mut residual,
                ROWS,
                H,
                EPSILON,
            )
            .unwrap();
            cpu::residual_add(&mut residual, &s.next[..h]).unwrap();
            cpu::rmsnorm(&residual, w.pre_ff_norm, &mut normalized, ROWS, H, EPSILON).unwrap();
            ops.project(&normalized, &w.ff_gate, &mut gate, ROWS)
                .unwrap();
            cpu::gated_gelu_tanh(&gate, &s.up[..ff], &mut product).unwrap();
            ops.project(&product, &w.ff_down, &mut down, ROWS).unwrap();
            cpu::rmsnorm(&down, w.post_ff_norm, &mut after_ff, ROWS, H, EPSILON).unwrap();
            for (name, values) in [
                ("post_attention_residual", &residual[at(H)]),
                ("pre_feedforward_layernorm", &normalized[at(H)]),
                ("mlp.gate_proj", &gate[at(g.intermediate)]),
                ("ff_product", &product[at(g.intermediate)]),
                ("mlp.down_proj", &down[at(H)]),
                ("post_feedforward_layernorm", &after_ff[at(H)]),
            ] {
                capture.record(name, values, POSITION, 1);
            }
            cpu::residual_add(&mut after_ff, &residual).unwrap();
            ops.project(&after_ff, &w.ple_gate, &mut ple_gate, ROWS)
                .unwrap();
            // PLE only overwrites the first ROWS*P slots of gate/product.
            let ff_row = at(g.intermediate);
            assert!(ff_row.start >= ROWS * P);
            assert_eq!(gate[ff_row.clone()], s.gate[ff_row.clone()]);
            assert_eq!(product[ff_row.clone()], s.product[ff_row]);
            assert_eq!(ple_gate, s.gate[..ROWS * P]);
            capture.record("post_ff_residual", &after_ff[at(H)], POSITION, 1);
            capture.record("per_layer_input_gate", &ple_gate[at(P)], POSITION, 1);
            continue;
        }
        if layer == 3 {
            let q = HEADS * g.head_dim;
            for (stage, values) in [
                ("input_layernorm", &s.norm[at(H)]),
                ("self_attn.q_norm", &s.q_norm[at(q)]),
                ("q_rotary", &s.q[at(q)]),
                ("context", &s.context[at(q)]),
                ("self_attn.o_proj", &s.attention[at(H)]),
                ("mlp.up_proj", &s.up[at(g.intermediate)]),
                ("per_layer_input_gate", &s.gate[at(P)]),
                ("per_layer_projection", &s.dense_work[at(H)]),
                ("post_per_layer_input_norm", &s.dense_norm[at(H)]),
                ("ple_input", &s.layer_ple[at(P)]),
            ] {
                capture.record(&format!("layer.3.{stage}"), values, POSITION, 1);
            }
        }
        if layer == 4 {
            capture.record("layer.4.input_layernorm", &s.norm[at(H)], POSITION, 1);
            capture.record(
                "layer.4.self_attn.v_proj",
                &s.kv[at(g.head_dim)],
                POSITION,
                1,
            );
            capture.record(
                "layer.4.self_attn.k_norm",
                &s.kv_norm[at(g.head_dim)],
                POSITION,
                1,
            );
            for owner in 0..=layer {
                let dim = model.layout.layers[owner].head_dim;
                let stage = &session.staging[owner];
                capture.record(
                    &format!("layer.{owner}.k_rotary"),
                    &stage.k[at(dim)],
                    POSITION,
                    1,
                );
                capture.record(
                    &format!("layer.{owner}.self_attn.v_norm"),
                    &stage.v[at(dim)],
                    POSITION,
                    1,
                );
            }
        } else {
            capture.record(
                &format!("layer.{layer}.output"),
                &s.hidden[at(H)],
                POSITION,
                1,
            );
        }
    }
    session.fault = None;
    session.forward(&ids[ROWS..]).unwrap();
    assert_eq!(session.tokens(), ids);
    assert_eq!(pointers(&session), owned);
    assert_eq!(resident_snapshot(&source), owners);
    assert_eq!(original, identity(&file.metadata().unwrap()));
    assert_eq!(
        original,
        identity(&std::fs::symlink_metadata(path).unwrap())
    );
    emit(
        if replay_ff {
            "LAYER3_FF_COMPLETE"
        } else {
            "FIRST_STAGED_COMPLETE"
        },
        json!({"committed":ids.len(),"faults_preserved_prefix":true}),
    );
    capture.finish();
}
