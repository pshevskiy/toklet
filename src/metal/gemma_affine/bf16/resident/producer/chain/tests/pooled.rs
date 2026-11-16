use super::*;

pub(super) fn pool_capture(session: &ResidentSession<'_, '_>, raw: &mut RawWriter) -> Value {
    session.storage.capture_audit(|words,chunks,spans,work,active,receipts| {
        let spans = raw.words(spans);
        let work = raw.words(work);
        let active = active.map(|s| raw.words(s));
        let receipts = receipts.map(|s| raw.words(s));
        json!({"ABI":1,"span_count":32,"state_spans":28,"opaque_GELU_spans":4,"guard_words":GUARD,"chunk_words_max":1024,"pool_words":words,"chunks":chunks,"spans":spans,"work":work,"active":active,"receipts":receipts,"diagnostics_are_explicit":true,"GPU_audits":3,"fill_after_nodes":[13,14],"finite_checked_after_ordered_replay":true})
    }).unwrap_or(Value::Null)
}
fn with_graph(run: impl FnOnce(&Fixture, &ResidentGraph<'_>)) {
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
        let graph = ResidentGraph::new(
            &source,
            &context,
            &pipelines,
            &table,
            options,
            ResidentGraph::planned_bytes(options).unwrap(),
        )
        .unwrap();
        run(&fixture, &graph);
    });
}
fn seeded_pool<'g, 'w>(
    graph: &'g ResidentGraph<'w>,
    fixture: &Fixture,
    case: &Value,
    diagnostic: bool,
) -> ResidentSession<'g, 'w> {
    let mut session = ResidentSession::new_pooled(
        graph,
        ResidentSession::planned_pooled_bytes(graph, diagnostic).unwrap(),
        diagnostic,
    )
    .unwrap();
    assert!(ledger(graph, &session) <= LIMIT);
    fixture.seed(&mut session, case);
    session
}
fn force_replay(tx: &mut GraphTransaction<'_, '_, '_>) {
    tx.session.storage.status_mut(0).with_mut(|s| s[0] = 2);
    let r = tx.session.graph.plans[0].layout.rms_plan().0[0].unwrap();
    // Shared tracked words are touched only after checked completion.
    tx.session
        .storage
        .reasons_mut(0)
        .with_mut(|s| s[r.start] |= 16);
}
#[test]
#[ignore = "explicit retirement audit/global contract/replay/postvalidation atomic prefix controls"]
fn pooled_retirement_and_publication_controls() {
    with_graph(|fixture, graph| {
        let decode = fixture.protocol["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "original-decode-all")
            .unwrap();
        let cancel = AtomicBool::new(false);
        let weights_before = weights_hash(graph);
        let mut raw = RawWriter::new("toklet-g2-pooled-deep-chain-controls-v1");
        // Faults are actual ordered GPU writes before retirement. FfDown is no
        // longer consumed by descendants, separating finite priority cleanly.
        for (name, node, guard, replay, success) in [
            ("retired13-guard-before-fill", 0, true, false, false),
            ("retired14-guard-before-fill", 1, true, false, false),
            ("retired15-guard-final-audit", 2, true, false, false),
            ("retired13-nonfinite-no-replay", 0, false, false, false),
            ("retired14-nonfinite-no-replay", 1, false, false, false),
            ("retired15-nonfinite-no-replay", 2, false, false, false),
            (
                "earlier-replay-discards-later-retired-NaN",
                1,
                false,
                true,
                true,
            ),
            (
                "later-retired-guard-defeats-earlier-replay",
                1,
                true,
                true,
                false,
            ),
        ] {
            autoreleasepool(|_| {
                let mut session = seeded_pool(graph, fixture, decode, true);
                session.retirement_fault = Some(pool::RetirementFault {
                    node,
                    state: State::FfDown,
                    inactive_guard: guard,
                    byte: if guard { 0 } else { 255 },
                });
                let before = session.caches.each_ref().map(cache_hash);
                let prefix_before = session.prefix().to_vec();
                let mut output = vec![99.; H];
                let mut tx = run(&mut session, fixture, decode, &cancel);
                if replay {
                    force_replay(&mut tx);
                }
                let audit = pool_capture(tx.session, &mut raw);
                if success {
                    let completed = tx.complete(&cancel).unwrap();
                    assert!(success);
                    assert_eq!(completed.metrics().replay_count, 1);
                    expected_states(&completed, fixture, decode);
                    let metrics = completed.publish(Some(&mut output), &cancel).unwrap();
                    check(
                        &output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        &fixture.view(&decode["expected_output"]),
                    );
                    let output = raw.words(&output.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
                    raw.cases.push(json!({"name":name,"passed":true,"success":true,"audit":audit,"metrics":metrics,"output":output}));
                } else {
                    // Consuming Result::err drops its possible CompletedGraph
                    // before any independent session assertion or capture.
                    let error = tx.complete(&cancel).err().unwrap();
                    assert!(!success);
                    assert!(session.poisoned);
                    assert_eq!(session.metrics.reference_commands, Some(0));
                    failure(
                        &mut raw,
                        name,
                        &session,
                        &before,
                        &prefix_before,
                        &output,
                        &error.to_string(),
                    );
                    raw.cases.last_mut().unwrap()["audit"] = audit;
                }
            });
        }
        for (name, kind, replay, after_complete) in [
            ("missing-audit-receipt", 0, false, false),
            ("unknown-audit-receipt", 1, true, false),
            ("changed-span-descriptor", 2, true, false),
            ("changed-work-descriptor", 3, true, false),
            ("changed-active-count", 4, true, false),
            ("consumer-reserved-status-precedes-replay", 8, true, false),
            ("GELU-proof-status-precedes-replay", 9, true, false),
            ("receipt-changed-after-validation", 5, false, true),
            ("final-pool-guard-after-validation", 6, false, true),
            ("proposal-NaN-after-validation", 7, false, true),
            ("final-pool-active-NaN-after-validation", 10, false, true),
        ] {
            autoreleasepool(|_| {
                let mut session = seeded_pool(graph, fixture, decode, false);
                let before = session.caches.each_ref().map(cache_hash);
                let prefix_before = session.prefix().to_vec();
                let mut output = vec![99.; H];
                let mut tx = run(&mut session, fixture, decode, &cancel);
                if replay {
                    force_replay(&mut tx);
                }
                let error = if after_complete {
                    let completed = tx.complete(&cancel).unwrap();
                    completed.0.session.storage.mutate_control(kind).unwrap();
                    completed.publish(Some(&mut output), &cancel).err().unwrap()
                } else {
                    match kind {
                        8 => tx.session.storage.status_mut(2).with_mut(|s| s[6] = 2),
                        9 => tx.session.storage.status_mut(2).with_mut(|s| s[23] |= 16),
                        _ => tx.session.storage.mutate_control(kind).unwrap(),
                    }
                    tx.complete(&cancel).err().unwrap()
                };
                assert!(session.poisoned);
                assert_eq!(session.metrics.reference_commands, Some(0));
                failure(
                    &mut raw,
                    name,
                    &session,
                    &before,
                    &prefix_before,
                    &output,
                    &error.to_string(),
                );
            });
        }
        autoreleasepool(|_| {
            let mut session = seeded_pool(graph, fixture, decode, false);
            let before = session.caches.each_ref().map(cache_hash);
            let prefix_before = session.prefix().to_vec();
            let mut output = vec![99.; H];
            let mut completed = run(&mut session, fixture, decode, &cancel)
                .complete(&cancel)
                .unwrap();
            completed.0.frame.cache_generations[1] ^= 1;
            let error = completed.publish(Some(&mut output), &cancel).err().unwrap();
            failure(
                &mut raw,
                "pooled-second-cache-stale-frame-blocks-both",
                &session,
                &before,
                &prefix_before,
                &output,
                &error.to_string(),
            );
        });
        // Immutable descriptor content is authenticated before any encoder or
        // staging write; a rejected cold frame leaves the prefix usable.
        autoreleasepool(|_| {
            let mut session = seeded_pool(graph, fixture, decode, false);
            session.storage.mutate_control(2).unwrap();
            let before = session.caches.each_ref().map(cache_hash);
            let (hidden, ple) = fixture.checkpoint(decode);
            let prefix = fixture.prefix(decode);
            let error = session
                .begin(
                    &prefix,
                    &hidden,
                    [&ple[0], &ple[1], &ple[2]],
                    mode(decode),
                    &cancel,
                )
                .err()
                .unwrap();
            assert!(!session.busy && !session.poisoned);
            assert_eq!(session.epoch, 0);
            assert_eq!(session.caches.each_ref().map(cache_hash), before);
            raw.cases.push(json!({"name":"pool-descriptor-rejected-before-staging","passed":true,"error":error.to_string(),"epoch":session.epoch,"committed":session.committed(),"poisoned":session.poisoned,"encoders_created":0}));
        });
        // Cancellation after successful validation discards all proposals and
        // leaves the exact committed prefix reusable; queued Drop drains first.
        for queued in [false, true] {
            autoreleasepool(|_| {
                let mut session = seeded_pool(graph, fixture, decode, false);
                let before = session.caches.each_ref().map(cache_hash);
                let prefix_before = session.prefix().to_vec();
                let (hidden, ple) = fixture.checkpoint(decode);
                let prefix = fixture.prefix(decode);
                if queued {
                    let mut tx = session
                        .begin(
                            &prefix,
                            &hidden,
                            [&ple[0], &ple[1], &ple[2]],
                            mode(decode),
                            &cancel,
                        )
                        .unwrap();
                    tx.encode_all(&cancel).unwrap();
                    tx.command.enqueue();
                    drop(tx);
                    assert!(session.poisoned);
                    assert_eq!(session.metrics.candidate_waits, 1);
                } else {
                    let completed = run(&mut session, fixture, decode, &cancel)
                        .complete(&cancel)
                        .unwrap();
                    cancel.store(true, Ordering::Release);
                    assert!(completed.publish(None, &cancel).is_err());
                    cancel.store(false, Ordering::Release);
                    assert!(!session.poisoned);
                }
                assert_eq!(session.prefix(), prefix_before);
                assert_eq!(session.caches.each_ref().map(cache_hash), before);
                raw.cases.push(json!({"name":if queued{"pooled-queued-Drop-drains"}else{"pooled-cancel-before-publish"},"passed":true,"cache_before_sha256":before,"cache_after_sha256":session.caches.each_ref().map(cache_hash),"committed":session.committed(),"poisoned":session.poisoned,"metrics":session.metrics}));
            });
        }
        assert_eq!(weights_hash(graph), weights_before);
        raw.finish();
    });
}
#[test]
#[ignore = "explicit pooled changing1..4row shapes/32 warmed Rust allocation measurements"]
fn pooled_warm_changing_shapes_allocate_no_rust_memory() {
    with_graph(|fixture, graph| {
        use crate::model::gemma_reference::tests::Measurement;
        let case = fixture.protocol["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "original-prefill-all")
            .unwrap();
        let mut session = seeded_pool(graph, fixture, case, false);
        let (hidden, ple) = fixture.checkpoint(case);
        let prefix = fixture.prefix(case);
        let cancel = AtomicBool::new(false);
        let mut output = vec![99.; H];
        for rows in 1..=4 {
            autoreleasepool(|_| {
                session.reset().unwrap();
                session
                    .evaluate(
                        &prefix[..rows],
                        &hidden[..rows * H],
                        [
                            &ple[0][..rows * P],
                            &ple[1][..rows * P],
                            &ple[2][..rows * P],
                        ],
                        OutputRows::Last,
                        &mut output,
                        &cancel,
                    )
                    .unwrap();
            });
        }
        let measurement = Measurement::start();
        for iteration in 0..32 {
            autoreleasepool(|_| {
                let rows = 1 + iteration % 4;
                session.reset().unwrap();
                let metrics = session
                    .evaluate(
                        &prefix[..rows],
                        &hidden[..rows * H],
                        [
                            &ple[0][..rows * P],
                            &ple[1][..rows * P],
                            &ple[2][..rows * P],
                        ],
                        OutputRows::Last,
                        &mut output,
                        &cancel,
                    )
                    .unwrap();
                assert_eq!(metrics.replay_count, 0);
                assert_eq!(metrics.candidate_commands, 1);
                assert_eq!(metrics.candidate_waits, 1);
                assert_eq!(metrics.encoded_dispatches, Some(105));
                assert_eq!(metrics.pool_audit_dispatches, Some(3));
                assert_eq!(metrics.pool_diagnostic_blit_copies, 0);
                assert_eq!(metrics.blit_copies, 3);
            });
        }
        let allocations = measurement.finish();
        assert_eq!(allocations, 0);
        let expected = fixture.view(&case["expected_output"]);
        check(
            &output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            &expected[3 * H..4 * H],
        );
        let mut raw = RawWriter::new("toklet-g2-pooled-deep-chain-warm-v1");
        let audit = pool_capture(&session, &mut raw);
        let output = raw.words(&output.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        raw.cases.push(json!({"name":"pool-warm32-changing-rows","iterations":32,"Rust_allocations":allocations,"native_allocations_measured":false,"rows_cycle":[1,2,3,4],"measured_commands":32,"measured_waits":32,"pool_diagnostic_allocations":false,"source_case":case["name"],"output":output,"audit":audit,"metrics_last":session.metrics().unwrap()}));
        raw.finish();
    });
}

#[test]
#[ignore = "explicit pooled caller-owned resident input checkpoint and Last donor lifetime"]
fn pooled_resident_entry_checkpoint_preserves_all_historical_states() {
    with_graph(|fixture, graph| {
        let case = fixture.protocol["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "original-prefill-last")
            .unwrap();
        let mut session = seeded_pool(graph, fixture, case, true);
        let (hidden, ple) = fixture.checkpoint(case);
        let values = [
            hidden.as_slice(),
            ple[0].as_slice(),
            ple[1].as_slice(),
            ple[2].as_slice(),
        ];
        let sources = values.map(|values| {
            let mut buffer = Buffer::new(&graph.context.0.gpu, values.len()).unwrap();
            buffer.with_mut(|s| {
                for (dst, value) in s.iter_mut().zip(values) {
                    *dst = value.to_bits();
                }
            });
            buffer
        });
        let inputs = sources
            .each_ref()
            .map(|b| BufferRange::new(b, 0, b.len).unwrap());
        let command = graph.context.0.gpu.command().unwrap();
        let prefix = fixture.prefix(case);
        let cancel = AtomicBool::new(false);
        let before = session.caches.each_ref().map(cache_hash);
        let mut output = vec![99.; H];
        let mut tx = session
            .begin_resident(&command, &prefix, inputs, OutputRows::Last, &cancel)
            .unwrap();
        tx.encode_all(&cancel).unwrap();
        tx.submit_and_wait().unwrap();
        assert_eq!(tx.session.caches.each_ref().map(cache_hash), before);
        let mut raw = RawWriter::new("toklet-g2-pooled-deep-chain-resident-v1");
        let states = capture_states(&tx, &mut raw);
        let completed = tx.complete(&cancel).unwrap();
        expected_states(&completed, fixture, case);
        let metrics = completed.publish(Some(&mut output), &cancel).unwrap();
        assert_eq!(metrics.replay_count, 0);
        assert_eq!(metrics.input_upload_bytes, 0);
        assert_eq!(metrics.blit_copies, 63);
        assert_eq!(metrics.pool_diagnostic_blit_copies, 56);
        assert_eq!(metrics.encoded_dispatches, Some(105));
        assert_eq!(metrics.candidate_commands, 1);
        assert_eq!(metrics.candidate_waits, 1);
        let audit = pool_capture(&session, &mut raw);
        let output = raw.words(&output.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        raw.cases.push(json!({"name":"pooled-resident-checkpoint","source_case":case["name"],"nodes":states,"output":output,"audit":audit,"metrics":metrics,"external_GPU_blit_copies":4,"external_source_upload_bytes":4*9216,"committed":session.committed(),"cache_generations":session.caches.each_ref().map(|c|c.generation)}));
        raw.finish();
    });
}
