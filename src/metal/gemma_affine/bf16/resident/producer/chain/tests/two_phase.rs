use super::*;
use rms::precision::two_phase as paired;

fn phase_capture(
    session: &ResidentSession<'_, '_>,
    frame: &GraphFrame,
    raw: &mut RawWriter,
) -> Value {
    let phase = session.rms_phase.as_ref().unwrap();
    let bank = phase.buffer.with_slice(|s| raw.words(s));
    json!({"ABI":1,"epoch":frame.epoch,"max_rows":session.graph.options.max_rows,"node_rows":frame.nodes.map(|n|n.rows),"norms":22,"context_bytes_per_row":32,"header_bytes_per_op":16,"guard_words":GUARD,"PENDING":paired::PENDING,"STRUCTURAL":paired::STRUCTURAL,"source_shader_sha256":"e7f00dc8888e700bb5a40b859b24924f9cb0ee535880de1cb563aa8578b59cce","bank":bank,"ordinary_default_selected":false})
}
#[test]
#[ignore = "explicit paired RMS/nonpooled genuine eight-cell full-state and commit qualification"]
fn paired_deep_chain_eight_cells_nonpooled() {
    nominal_cells_paired(false);
}
#[test]
#[ignore = "explicit paired RMS/pooled genuine eight-cell full-state and commit qualification"]
fn paired_deep_chain_eight_cells_pooled() {
    nominal_cells_paired(true);
}

fn finish_pair(mut raw: RawWriter) {
    let passed = true;
    raw.file.flush().unwrap();
    drop(raw.file);
    let bytes = read(&raw.root.join("arrays.bin"), raw.bytes);
    let report = json!({"format":raw.format,"passed":passed,"fixture_protocol_sha256":PROTOCOL_SHA,"fixture_data_sha256":DATA_SHA,"cases":raw.cases,"raw":{"path":raw.root.join("arrays.bin"),"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))},"decode_scope":"prefill-derived causal row3 views, not separately capturedT1","uncaptured_internals_are_independent_witnesses":false,"normal_route_selected":false,"deep_chain_rms_precision":"scaled-double-float-guard2-two-phase",
            "deep_chain_gelu_table":{"variant":"finer-affine-core-v2","table_sha256":gelu::core::TABLE_SHA,"certificate_sha256":gelu::core::CERT_SHA,"native_verification_sha256":gelu::core::NATIVE_SHA,"artifact_review_sha256":gelu::core::CORE_REVIEW_SHA,"parent_table_sha256":"78dfa84189353b4f9e31cc1a6e6e3a6476ef4818c7548c3c3c06e9227764dcc6","payload_bytes":5480520,"added_payload_bytes":2162736,"guard_bytes":56,"owned_bytes":gelu::CertifiedTable::planned_core_owned_bytes(),"parent_owned_bytes":gelu::CertifiedTable::planned_owned_bytes(),"added_owned_bytes":2162736,"combined_GPU_tables":1,"selected_GELU_pipelines":1},"whole_graph_or_quality_or_performance_qualified":false});
    let mut file = std::fs::File::create_new(raw.root.join("result.json")).unwrap();
    serde_json::to_writer_pretty(&mut file, &report).unwrap();
    file.write_all(b"\n").unwrap();
}

fn nominal_cells_paired(pooled: bool) {
    autoreleasepool(|_| {
        let fixture = Fixture::load();
        let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let source = source(&context);
        let table = table(&context);
        let pipelines = selected_pipelines(&context, &table, PIPELINE_POLICY_BYTES, pooled);
        let options = Options {
            capacity: 512,
            max_rows: 4,
        };
        let graph = ResidentGraph::new_two_phase(
            &source,
            &context,
            &pipelines,
            &table,
            options,
            ResidentGraph::planned_bytes(options).unwrap(),
        )
        .unwrap();
        let original_weights = weights_hash(&graph);
        let cancel = AtomicBool::new(false);
        let mut raw = RawWriter::new(if pooled {
            "toklet-g2-paired-pooled-deep-chain-cells-v1"
        } else {
            "toklet-g2-paired-deep-chain-cells-v1"
        });
        let mut original_prefix = None;
        let mut future_prefix = None;
        for case in fixture.protocol["cases"].as_array().unwrap() {
            autoreleasepool(|_| {
                let mut session = ResidentSession::new_storage(
                    &graph,
                    ResidentSession::planned_storage_bytes(&graph, pooled, pooled).unwrap(),
                    pooled,
                    pooled,
                )
                .unwrap();
                assert!(ledger(&graph, &session) <= LIMIT);
                fixture.seed(&mut session, case);
                let cache_before = session.caches.each_ref().map(cache_hash);
                let generations = session.caches.each_ref().map(|c| c.generation);
                let (hidden, ple) = fixture.checkpoint(case);
                let prefix = fixture.prefix(case);
                let output_rows = case["final_queries"].as_u64().unwrap() as usize;
                let mut output = vec![99.; output_rows * H];
                let mut tx = session
                    .begin(
                        &prefix,
                        &hidden,
                        [&ple[0], &ple[1], &ple[2]],
                        mode(case),
                        &cancel,
                    )
                    .unwrap();
                assert_eq!(
                    tx.frame.nodes[2].proposal_rows,
                    case["input_rows"].as_u64().unwrap() as usize
                );
                tx.encode_all(&cancel).unwrap();
                assert!(output.iter().all(|&x| x == 99.));
                tx.submit_and_wait().unwrap();
                let candidate_cache = tx.session.caches.each_ref().map(cache_hash);
                assert_eq!(candidate_cache, cache_before);
                let states = capture_states(&tx, &mut raw);
                let phase = phase_capture(tx.session, &tx.frame, &mut raw);
                println!(
                    "PAIRED_CHAIN_CASE_RAW {}",
                    json!({"name":case["name"],"nodes":states,"rms_phase":phase})
                );
                let masks = mask_count(&tx);
                if case["mode"] == "prefill" && mode(case) == OutputRows::All {
                    let hashes = prefix_hashes(&tx);
                    if case["sequence"] == "original" {
                        original_prefix = Some(hashes);
                    } else {
                        future_prefix = Some(hashes);
                    }
                }
                let completed = tx.complete(&cancel).unwrap();
                expected_states(&completed, &fixture, case);
                assert_eq!(completed.metrics().replay_count, 0);
                assert_eq!(completed.metrics().rms_phase_validated_operations, 22);
                assert_eq!(
                    completed.metrics().encoded_dispatches,
                    Some(if pooled { 127 } else { 124 })
                );
                let proposals: [(Vec<u32>, Vec<u32>); 2] = std::array::from_fn(|node| {
                    (
                        completed
                            .with_state(node, State::KProposal, |s| s.to_vec())
                            .unwrap(),
                        completed
                            .with_state(node, State::VProposal, |s| s.to_vec())
                            .unwrap(),
                    )
                });
                let metrics = completed.publish(Some(&mut output), &cancel).unwrap();
                assert_eq!(session.committed(), 4);
                assert_eq!(session.prefix(), prefix);
                assert_eq!(session.output().unwrap(), output);
                check(
                    &output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    &fixture.view(&case["expected_output"]),
                );
                assert_eq!(metrics.candidate_commands, 1);
                assert_eq!(metrics.candidate_waits, 1);
                assert_eq!(metrics.reference_commands, Some(0));
                assert_eq!(
                    metrics.new_kv_copy_bytes,
                    case["input_rows"].as_u64().unwrap() as usize * 6144
                );
                let offset = case["offset"].as_u64().unwrap() as usize;
                let mut kv = Vec::new();
                for node in 0..2 {
                    let c = &session.caches[node];
                    assert_eq!(c.offset, 4);
                    assert_eq!(c.generation, generations[node] + 1);
                    let n = proposals[node].0.len();
                    c.k.with_slice(|s| {
                        assert_eq!(&s[offset * c.d..offset * c.d + n], proposals[node].0)
                    });
                    c.v.with_slice(|s| {
                        assert_eq!(&s[offset * c.d..offset * c.d + n], proposals[node].1)
                    });
                    kv.push(json!({"layer":LAYERS[node],"generation":c.generation,"offset":c.offset,"K":c.k.with_slice(|s|raw.words(&s[..4*c.d])),"V":c.v.with_slice(|s|raw.words(&s[..4*c.d])),"guarded_sha256":cache_hash(c)}));
                }
                let audit = pooled::pool_capture(&session, &mut raw);
                let output_raw = raw.words(&output.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
                raw.cases.push(json!({"name":case["name"],"sequence":case["sequence"],"mode":case["mode"],"output_mode":case["output_mode"],"nodes":states,"rms_phase":phase,"pool_audit":audit,"exact_mask_zeros":masks,"cache_before_sha256":cache_before,"cache_after_candidate_sha256":candidate_cache,"cache_generations_before":generations,"committed_cache":kv,"output":output_raw,"metrics":metrics,"model_payload_authenticated_by_existing_loader":true,"deep_entry_input_bytes":hidden.len()*4,"checkpoint_PLE15_uses_full_All_view_before_Last":true}));
            });
        }
        assert_eq!(original_prefix.as_ref().unwrap().len(), 64);
        assert_eq!(original_prefix, future_prefix);
        assert_eq!(weights_hash(&graph), original_weights);
        finish_pair(raw);
    });
}

fn with_paired_graph(test: impl FnOnce(&Fixture, &ResidentGraph<'_>)) {
    autoreleasepool(|_| {
        let fixture = Fixture::load();
        let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let source = source(&context);
        let table = table(&context);
        let pipelines = selected_pipelines(&context, &table, PIPELINE_POLICY_BYTES, true);
        let options = Options {
            capacity: 512,
            max_rows: 4,
        };
        let graph = ResidentGraph::new_two_phase(
            &source,
            &context,
            &pipelines,
            &table,
            options,
            ResidentGraph::planned_bytes(options).unwrap(),
        )
        .unwrap();
        test(&fixture, &graph);
    });
}
fn session<'g, 'w>(
    graph: &'g ResidentGraph<'w>,
    fixture: &Fixture,
    case: &Value,
    pooled: bool,
) -> ResidentSession<'g, 'w> {
    let mut session = ResidentSession::new_storage(
        graph,
        ResidentSession::planned_storage_bytes(graph, pooled, false).unwrap(),
        pooled,
        false,
    )
    .unwrap();
    fixture.seed(&mut session, case);
    assert!(ledger(graph, &session) <= LIMIT);
    session
}
// Full expected_states traverses retired nodes: this one successful pooled
// control explicitly admits diagnostic snapshots. Warm/timing stay nondiagnostic.
fn diagnostic_control_session<'g, 'w>(
    graph: &'g ResidentGraph<'w>,
    fixture: &Fixture,
    case: &Value,
) -> ResidentSession<'g, 'w> {
    let mut session = ResidentSession::new_storage(
        graph,
        ResidentSession::planned_storage_bytes(graph, true, true).unwrap(),
        true,
        true,
    )
    .unwrap();
    fixture.seed(&mut session, case);
    assert!(ledger(graph, &session) <= LIMIT);
    session
}
fn inject_paired_reference(tx: &mut GraphTransaction<'_, '_, '_>) {
    tx.session.storage.status_mut(0).with_mut(|s| s[0] = 2);
    let region = tx.session.graph.plans[0].layout.rms_plan().0[0].unwrap();
    tx.session
        .storage
        .reasons_mut(0)
        .with_mut(|s| s[region.start] |= 16);
}
fn phase_offsets(tx: &GraphTransaction<'_, '_, '_>, node: usize, norm: usize) -> (usize, usize) {
    let recipe = tx
        .session
        .graph
        .layer_recipe(
            (&tx.session.storage, tx.session.rms_phase.as_ref()),
            (&tx.session.entry, &tx.session.entry_regions),
            &tx.session.caches,
            &tx.frame,
            node,
            false,
        )
        .unwrap();
    let recipe::NormKind::Paired {
        plan, phase, node, ..
    } = recipe.norm_kind
    else {
        panic!("paired test graph required")
    };
    let slot = plan.slots(phase, recipe.shape.rows, node).unwrap()[norm].unwrap();
    (slot.context.offset, slot.header.offset)
}
#[test]
#[ignore = "explicit paired context/counter/late-proof/ordered-reference/epoch/drop/partial-encode controls"]
fn paired_graph_receipt_lifecycle_and_publication_controls() {
    with_paired_graph(|fixture, graph| {
        let case = fixture.protocol["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "original-decode-all")
            .unwrap();
        let cancel = AtomicBool::new(false);
        let weights = weights_hash(graph);
        let mut raw = RawWriter::new("toklet-g2-paired-RMS-controls-v1");
        for pooled in [false, true] {
            for (name, replay, success) in [
                ("context-prefix", false, false),
                ("inactive-context", false, false),
                ("pending", false, false),
                ("header-op", false, false),
                ("row-count", false, false),
                ("later-structural-defeats-earlier-replay", true, false),
                ("later-numeric-allows-earlier-replay", true, true),
            ] {
                autoreleasepool(|_| {
                    let mut s = if pooled && success {
                        diagnostic_control_session(graph, fixture, case)
                    } else {
                        session(graph, fixture, case, pooled)
                    };
                    let before = s.caches.each_ref().map(cache_hash);
                    let prefix = s.prefix().to_vec();
                    let mut out = vec![99.; H];
                    let mut tx = run(&mut s, fixture, case, &cancel);
                    if replay {
                        inject_paired_reference(&mut tx);
                    }
                    let node = usize::from(replay);
                    let (context, header) = phase_offsets(&tx, node, 0);
                    match name {
                        "pending" => tx
                            .session
                            .storage
                            .status_mut(0)
                            .with_mut(|s| s[0] |= paired::PENDING),
                        "later-numeric-allows-earlier-replay" => {
                            tx.session.storage.status_mut(1).with_mut(|s| s[0] = 1);
                            let r = tx.session.graph.plans[1].layout.rms_plan().0[0].unwrap();
                            tx.session
                                .storage
                                .reasons_mut(1)
                                .with_mut(|s| s[r.start] |= 0x8000_0000);
                        }
                        _ => tx.session.rms_phase.as_mut().unwrap().buffer.with_mut(|s| {
                            let at = match name {
                                "context-prefix" => 0,
                                "inactive-context" => context + 8,
                                "row-count" => context + 7,
                                _ => header + 1,
                            };
                            s[at] ^= 1;
                        }),
                    }
                    let phase = phase_capture(tx.session, &tx.frame, &mut raw);
                    let statuses = status_capture(tx.session, &mut raw);
                    println!(
                        "PAIRED_CHAIN_CONTROL_RAW {}",
                        json!({"name":name,"pooled":pooled,"rms_phase":phase,"statuses":statuses})
                    );
                    let result = tx.complete(&cancel);
                    let metrics = if success {
                        let completed = result.unwrap();
                        assert_eq!(completed.metrics().replay_count, 1);
                        expected_states(&completed, fixture, case);
                        let m = completed.publish(Some(&mut out), &cancel).unwrap();
                        check(
                            &out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            &fixture.view(&case["expected_output"]),
                        );
                        m
                    } else {
                        assert!(result.is_err());
                        drop(result);
                        assert!(s.poisoned);
                        assert_eq!(s.prefix(), prefix);
                        assert_eq!(s.caches.each_ref().map(cache_hash), before);
                        assert!(out.iter().all(|&x| x == 99.));
                        assert_eq!(s.metrics.reference_commands, Some(0));
                        s.metrics
                    };
                    let caller = raw.words(&out.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
                    raw.cases.push(json!({"name":name,"pooled":pooled,"expected_success":success,"replay":replay,"rms_phase":phase,"status":statuses,"caller":caller,"cache_before_sha256":before,"cache_after_sha256":s.caches.each_ref().map(cache_hash),"prefix_before":prefix,"prefix_after":s.prefix(),"poisoned":s.poisoned,"metrics":metrics}));
                });
            }
            // Exact evidence is retained after validation; no late metadata
            // change may be hidden by a numerical reference upload.
            for replay in [false, true] {
                let mut s = session(graph, fixture, case, pooled);
                let before = s.caches.each_ref().map(cache_hash);
                let prefix = s.prefix().to_vec();
                let mut out = vec![99.; H];
                let mut tx = run(&mut s, fixture, case, &cancel);
                if replay {
                    inject_paired_reference(&mut tx);
                }
                let (context, _) = phase_offsets(&tx, 2, 0);
                let completed = tx.complete(&cancel).unwrap();
                let earlier = phase_capture(completed.0.session, &completed.0.frame, &mut raw);
                completed
                    .0
                    .session
                    .rms_phase
                    .as_mut()
                    .unwrap()
                    .buffer
                    .with_mut(|s| s[context + 7] ^= 1);
                let late = phase_capture(completed.0.session, &completed.0.frame, &mut raw);
                assert!(completed.publish(Some(&mut out), &cancel).is_err());
                assert!(s.poisoned);
                assert_eq!(s.prefix(), prefix);
                assert_eq!(s.caches.each_ref().map(cache_hash), before);
                assert!(out.iter().all(|&x| x == 99.));
                let caller = raw.words(&out.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
                raw.cases.push(json!({"name":"late-row-count-after-validation","pooled":pooled,"replay":replay,"before":earlier,"after":late,"cache_before_sha256":before,"cache_after_sha256":s.caches.each_ref().map(cache_hash),"caller":caller}));
            }
            for enqueued in [false, true] {
                let mut s = session(graph, fixture, case, pooled);
                let before = s.caches.each_ref().map(cache_hash);
                let (hidden, ple) = fixture.checkpoint(case);
                let prefix = fixture.prefix(case);
                let tx = s
                    .begin(
                        &prefix,
                        &hidden,
                        [&ple[0], &ple[1], &ple[2]],
                        mode(case),
                        &cancel,
                    )
                    .unwrap();
                let phase = phase_capture(tx.session, &tx.frame, &mut raw);
                if enqueued {
                    tx.command.enqueue();
                }
                drop(tx);
                assert!(!s.poisoned);
                assert_eq!(s.committed(), 3);
                assert_eq!(s.caches.each_ref().map(cache_hash), before);
                let discarded_epoch = s.epoch;
                let mut out = vec![99.; H];
                let m = s
                    .evaluate(
                        &prefix,
                        &hidden,
                        [&ple[0], &ple[1], &ple[2]],
                        mode(case),
                        &mut out,
                        &cancel,
                    )
                    .unwrap();
                assert_eq!(s.epoch, discarded_epoch + 1);
                assert_eq!(m.replay_count, 0);
                assert_eq!(m.rms_phase_validated_operations, 22);
                raw.cases.push(json!({"name":"unencoded-drop-and-reuse","enqueued":enqueued,"pooled":pooled,"rms_phase":phase,"metrics":m}));
            }
            // An actual A-only error after a successful43-dispatch producer
            // must discard the partial count, drain, poison and keep K/V private.
            {
                let mut s = session(graph, fixture, case, pooled);
                let before = s.caches.each_ref().map(cache_hash);
                let prefix_before = s.prefix().to_vec();
                let out = [99f32; H];
                let (hidden, ple) = fixture.checkpoint(case);
                let prefix = fixture.prefix(case);
                let mut tx = s
                    .begin(
                        &prefix,
                        &hidden,
                        [&ple[0], &ple[1], &ple[2]],
                        mode(case),
                        &cancel,
                    )
                    .unwrap();
                tx.encoding_started = true;
                tx.copy_node_inputs(0).unwrap();
                let (result, count) = {
                    let op = graph
                        .layer_recipe(
                            (&tx.session.storage, tx.session.rms_phase.as_ref()),
                            (&tx.session.entry, &tx.session.entry_regions),
                            &tx.session.caches,
                            &tx.frame,
                            0,
                            false,
                        )
                        .unwrap();
                    let r = op.encode(&tx.command);
                    (r, op.dispatches.get())
                };
                tx.session
                    .metrics
                    .record_recipe_encoding(&result, count, false);
                result.unwrap();
                assert_eq!(tx.session.metrics.encoded_dispatches, Some(43));
                tx.copy_node_inputs(1).unwrap();
                let (result, count) = {
                    let mut op = graph
                        .layer_recipe(
                            (&tx.session.storage, tx.session.rms_phase.as_ref()),
                            (&tx.session.entry, &tx.session.entry_regions),
                            &tx.session.caches,
                            &tx.frame,
                            1,
                            false,
                        )
                        .unwrap();
                    if let recipe::NormKind::Paired { fail_after_a, .. } = &mut op.norm_kind {
                        *fail_after_a = Some(0);
                    }
                    let r = op.encode(&tx.command);
                    (r, op.dispatches.get())
                };
                assert!(matches!(
                    &result,
                    Err(Error::Options(
                        "paired RMS synthetic B encoder failure after A"
                    ))
                ));
                tx.session
                    .metrics
                    .record_recipe_encoding(&result, count, false);
                assert_eq!(tx.session.metrics.encoded_dispatches, None);
                assert!(!tx.session.metrics.encoding_complete);
                drop(tx);
                assert!(s.poisoned);
                assert_eq!(s.prefix(), prefix_before);
                assert_eq!(s.caches.each_ref().map(cache_hash), before);
                assert_eq!(
                    (s.metrics.candidate_commands, s.metrics.candidate_waits),
                    (1, 1)
                );
                assert!(out.iter().all(|&x| x == 99.));
                let bank = s
                    .rms_phase
                    .as_ref()
                    .unwrap()
                    .buffer
                    .with_slice(|w| raw.words(w));
                raw.cases.push(json!({"name":"partial-A-after-counted-producer","pooled":pooled,"bank":bank,"metrics":s.metrics,"cache_before_sha256":before,"cache_after_sha256":s.caches.each_ref().map(cache_hash)}));
            }
            {
                let mut s = session(graph, fixture, case, pooled);
                let before = s.caches.each_ref().map(cache_hash);
                let prefix = s.prefix().to_vec();
                let mut out = vec![99.; H];
                let tx = run(&mut s, fixture, case, &cancel);
                let completed = tx.complete(&cancel).unwrap();
                cancel.store(true, Ordering::Release);
                assert!(completed.publish(Some(&mut out), &cancel).is_err());
                cancel.store(false, Ordering::Release);
                assert!(!s.poisoned);
                assert_eq!(s.prefix(), prefix);
                assert_eq!(s.caches.each_ref().map(cache_hash), before);
                assert!(out.iter().all(|&x| x == 99.));
                raw.cases.push(json!({"name":"cancel-after-validation","pooled":pooled,"epoch":s.epoch,"cache_before_sha256":before,"cache_after_sha256":s.caches.each_ref().map(cache_hash)}));
            }
            {
                let mut s = session(graph, fixture, case, pooled);
                let before = s.caches.each_ref().map(cache_hash);
                let prefix = fixture.prefix(case);
                let (hidden, ple) = fixture.checkpoint(case);
                let old_prefix = s.prefix().to_vec();
                let bank_before = s
                    .rms_phase
                    .as_ref()
                    .unwrap()
                    .buffer
                    .with_slice(digest_words);
                s.epoch = u64::from(paired::MAX_EPOCH);
                assert!(
                    s.begin(
                        &prefix,
                        &hidden,
                        [&ple[0], &ple[1], &ple[2]],
                        mode(case),
                        &cancel
                    )
                    .is_err()
                );
                assert!(!s.poisoned);
                assert_eq!(s.prefix(), old_prefix);
                assert_eq!(s.caches.each_ref().map(cache_hash), before);
                let bank_after = s
                    .rms_phase
                    .as_ref()
                    .unwrap()
                    .buffer
                    .with_slice(digest_words);
                assert_eq!(bank_before, bank_after);
                s.reset().unwrap();
                assert_eq!(s.epoch, u64::from(paired::MAX_EPOCH));
                raw.cases.push(json!({"name":"epoch-exhaustion-before-mutation","pooled":pooled,"bank_before_sha256":bank_before,"bank_after_sha256":bank_after,"epoch_after_reset":s.epoch}));
            }
        }
        assert_eq!(weights_hash(graph), weights);
        finish_pair(raw);
    });
}

#[test]
#[ignore = "explicit paired RMS pooled/nonpooled changing1..4 rows/32 warmed Rust allocation checks"]
fn paired_warm_changing_shapes_allocate_no_rust_memory() {
    with_paired_graph(|fixture, graph| {
        use crate::model::gemma_reference::tests::Measurement;
        let case = fixture.protocol["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "original-prefill-all")
            .unwrap();
        let (hidden, ple) = fixture.checkpoint(case);
        let prefix = fixture.prefix(case);
        let cancel = AtomicBool::new(false);
        let mut raw = RawWriter::new("toklet-g2-paired-RMS-warm-v1");
        for pooled in [false, true] {
            let mut s = session(graph, fixture, case, pooled);
            let mut out = vec![99.; H];
            for rows in 1..=4 {
                autoreleasepool(|_| {
                    s.reset().unwrap();
                    s.evaluate(
                        &prefix[..rows],
                        &hidden[..rows * H],
                        [
                            &ple[0][..rows * P],
                            &ple[1][..rows * P],
                            &ple[2][..rows * P],
                        ],
                        OutputRows::Last,
                        &mut out,
                        &cancel,
                    )
                    .unwrap();
                });
            }
            let measurement = Measurement::start();
            for iteration in 0..32 {
                autoreleasepool(|_| {
                    let rows = 1 + iteration % 4;
                    s.reset().unwrap();
                    let m = s
                        .evaluate(
                            &prefix[..rows],
                            &hidden[..rows * H],
                            [
                                &ple[0][..rows * P],
                                &ple[1][..rows * P],
                                &ple[2][..rows * P],
                            ],
                            OutputRows::Last,
                            &mut out,
                            &cancel,
                        )
                        .unwrap();
                    assert_eq!(m.replay_count, 0);
                    assert_eq!(m.reference_commands, Some(0));
                    assert_eq!((m.candidate_commands, m.candidate_waits), (1, 1));
                    assert_eq!(m.encoded_dispatches, Some(if pooled { 127 } else { 124 }));
                    assert_eq!(m.rms_phase_validated_operations, 22);
                    assert_eq!(
                        (m.rms_phase_gpu_bytes, m.rms_phase_snapshot_bytes),
                        (7116, 7116)
                    );
                });
            }
            let allocations = measurement.finish();
            assert_eq!(allocations, 0);
            check(
                &out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                &fixture.view(&case["expected_output"])[3 * H..4 * H],
            );
            let out = raw.words(&out.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            raw.cases.push(json!({"name":"paired-warm32-changing-shapes","pooled":pooled,"iterations":32,"Rust_allocations":allocations,"native_allocations_measured":false,"measured_commands":32,"measured_waits":32,"source_weights_shared":true,"output":out,"metrics_last":s.metrics().unwrap(),"committed":s.committed()}));
        }
        finish_pair(raw);
    });
}

#[test]
#[ignore = "explicit source-matched old/paired pooled whole-command diagnostic;3warm10samples,run1or2 env"]
fn paired_graph_whole_command_timing() {
    autoreleasepool(|_| {
        let run: usize = std::env::var("TOKLET_RMS_GRAPH_TIMING_RUN")
            .unwrap()
            .parse()
            .unwrap();
        assert!([1, 2].contains(&run));
        let fixture = Fixture::load();
        let context = Bf16Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        let source = source(&context);
        let table = table(&context);
        let pipelines = selected_pipelines(&context, &table, PIPELINE_POLICY_BYTES, true);
        let options = Options {
            capacity: 512,
            max_rows: 4,
        };
        let old = ResidentGraph::new(
            &source,
            &context,
            &pipelines,
            &table,
            options,
            ResidentGraph::planned_bytes(options).unwrap(),
        )
        .unwrap();
        let paired = ResidentGraph::new_two_phase(
            &source,
            &context,
            &pipelines,
            &table,
            options,
            ResidentGraph::planned_bytes(options).unwrap(),
        )
        .unwrap();
        let graphs = if run == 1 {
            [&old, &paired]
        } else {
            [&paired, &old]
        };
        let cancel = AtomicBool::new(false);
        let mut raw = RawWriter::new("toklet-g2-paired-RMS-whole-command-timing-v1");
        let names = if run == 1 {
            ["original-prefill-all", "original-decode-last"]
        } else {
            ["original-decode-last", "original-prefill-all"]
        };
        for name in names {
            let case = fixture.protocol["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["name"] == name)
                .unwrap();
            let (hidden, ple) = fixture.checkpoint(case);
            let prefix = fixture.prefix(case);
            let mut sessions = graphs.map(|graph| session(graph, &fixture, case, true));
            let session_bytes = sessions
                .iter()
                .map(ResidentSession::owned_bytes)
                .sum::<usize>();
            for phase in ["first-use", "warmup", "sample"] {
                let count = if phase == "first-use" {
                    1
                } else if phase == "warmup" {
                    3
                } else {
                    10
                };
                for iteration in 0..count {
                    let order = if iteration % 2 == 0 { [0, 1] } else { [1, 0] };
                    for index in order {
                        autoreleasepool(|_| {
                            let graph = graphs[index];
                            let variant = graph.phase_parts().unwrap().is_some();
                            let s = &mut sessions[index];
                            s.reset().unwrap();
                            fixture.seed(s, case);
                            assert!(
                                ledger(graph, s)
                                    + if variant {
                                        old.admitted_bytes()
                                    } else {
                                        paired.admitted_bytes()
                                    }
                                    + session_bytes
                                    - s.owned_bytes()
                                    <= LIMIT
                            );
                            let mut out =
                                vec![99.; case["final_queries"].as_u64().unwrap() as usize * H];
                            let m = s
                                .evaluate(
                                    &prefix,
                                    &hidden,
                                    [&ple[0], &ple[1], &ple[2]],
                                    mode(case),
                                    &mut out,
                                    &cancel,
                                )
                                .unwrap();
                            let record = json!({"name":case["name"],"run":run,"family":if variant{"paired"}else{"guard2"},"phase":phase,"iteration":iteration,"metrics":m,"same_weight_owner":true,"same_Gpu_and_shared_table":true,"shared_graph_policy_bytes_including_both":old.admitted_bytes()+paired.admitted_bytes(),"whole_graph_or_request_performance_claim":false});
                            println!("PAIRED_CHAIN_TIMING_RAW {record}");
                            raw.cases.push(record);
                            assert_eq!(m.replay_count, 0);
                            assert_eq!(m.reference_commands, Some(0));
                            assert_eq!((m.candidate_commands, m.candidate_waits), (1, 1));
                            assert_eq!(m.encoded_dispatches, Some(if variant { 127 } else { 105 }));
                            check(
                                &out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                &fixture.view(&case["expected_output"]),
                            );
                        });
                    }
                }
            }
        }
        assert_eq!(raw.cases.len(), 56);
        finish_pair(raw);
    });
}
