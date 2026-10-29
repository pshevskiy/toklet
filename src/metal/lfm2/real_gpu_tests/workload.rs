use super::*;
use crate::model::gemma_reference::tests::Measurement;

impl Capture {
    fn history_stack(&mut self, s: &Session<'_>) -> Result<()> {
        s.cache.test_safe_read()?;
        require(self.history_stacks < 18, "history initial/reset count")?;
        let mut words = Vec::with_capacity(10 * 2 * 2055);
        for h in &s.histories {
            for b in [&h.old, &h.candidate] {
                b.with_slice(|v| words.extend_from_slice(v));
            }
        }
        require(
            words.len() == 41100 && words.capacity() * 2 <= 82200,
            "history stack owner bound",
        )?;
        let mut t = tag(
            self.session,
            self.attempt,
            HISTORY_STACK,
            self.history_stacks as u32,
            words.len(),
        );
        t.shape = [10, 2, 2055];
        t.rank = 3;
        self.writer
            .half(t, words.len(), words.iter().copied())
            .map_err(io)?;
        self.history_stacks += 1;
        Ok(())
    }
    fn session<'m>(
        &mut self,
        m: &'m Model,
        id: u32,
        diagnostics: bool,
    ) -> Result<(Session<'m>, State)> {
        self.session = id;
        self.attempt = 0;
        self.registry.clear();
        let s = m.session(options(diagnostics, diagnostics))?;
        let mode = if diagnostics { 2 } else { 0 };
        if let Some(old) = self.admissions[mode] {
            require(
                old == s.memory().admitted,
                "repeat session admission changed",
            )?;
        }
        self.admissions[mode] = Some(s.memory().admitted);
        if !diagnostics {
            // Same cold owner/layout metadata; all_logits only increases the two
            // Float32 logit banks from1 to32 rows. No speculative Session needed.
            self.admissions[1] = Some(add(s.memory().admitted, mul(2 * (B - 1) * 65536, 4)?)?);
        }
        self.event("session_begin",json!({"session":id,"options":{"diagnostics":diagnostics,"all_logits":diagnostics},
            "memory":memory(s.memory()),"capacity":512,"chunk":32,"query_heads":16,"kv_heads":8,"dim":64,
            "histories":10,"cache_owners":6,"diagnostic_slots":s.diagnostic_count()}),Class::Boundary)?;
        self.initial_cache(&s)?;
        self.history_stack(&s)?;
        let state = self.state(&s, "initial")?;
        Ok((s, state))
    }
    fn reset(&mut self, s: &mut Session<'_>, before: &State) -> Result<State> {
        let result = s.reset();
        self.event("reset_result",json!({"session":self.session,"attempt":self.attempt,"run":self.run,"result":format!("{result:?}")}),Class::Other)?;
        s.cache.test_safe_read()?;
        self.history_stack(s)?;
        let state = self.state(s, "reset")?;
        let mut logits = vec![0f32; 65536];
        let error = s.copy_logits(&mut logits).err().map(|e| format!("{e:?}"));
        self.event("reset",json!({"run":self.run,"session":self.session,"before_boundary":before.boundary,"after_boundary":state.boundary,
            "logits_error":error,"old_KV_unchanged":before.old_kv==state.old_kv}),Class::Other)?;
        result?;
        require(
            error.is_some() && s.cache.offset() == 0 && before.old_kv == state.old_kv,
            "reset prefix/publication/KV preservation",
        )?;
        Ok(state)
    }
    fn row_hashes(&mut self, s: &Session<'_>, start: usize, n: usize) -> Result<()> {
        s.cache.test_safe_read()?;
        let end = start + n;
        s.published.with_slice(|words|->Result<()> {
            for r in 0..s.published_rows {
                let absolute=if s.options.all_logits{start+r}else{end-1};let v=&words[r*65536..(r+1)*65536];
                let raw=output::hash32(v);let numeric=hash32_iter(v.iter().map(|&x|if x&0x7fff_ffff==0{0}else{x}));
                self.event("row_hash",json!({"session":self.session,"attempt":self.attempt,"run":self.run,"row":absolute,
                    "raw_word_sha256":raw,"numeric_sha256":numeric}),Class::Other)?;
                require(v.iter().all(|&x|f32::from_bits(x).is_finite()),"captured logits nonfinite")?;
                let prefix=s.tokens[..absolute+1].to_vec();
                if let Some(previous)=self.baseline.get(&prefix) {self.trajectory_failures+=usize::from(previous!=&numeric);}
                else {self.baseline.insert(prefix,numeric);}
                self.row_hashes+=1;if self.run<8 {self.primary_rows+=1;}
            }Ok(())
        })
    }
    fn actual_state(&mut self, s: &Session<'_>) -> Result<()> {
        let end = s.cache.offset();
        let mut kv = Vec::with_capacity(12);
        s.cache.test_visit_buffers(|_, role, _, _, v| {
            if role < 2 {
                kv.push(hash16_iter((0..8).flat_map(|h| {
                    v[h * 512 * 64..h * 512 * 64 + end * 64].iter().copied()
                })));
            }
            Ok(())
        })?;
        let history: Vec<_> = s
            .histories
            .iter()
            .map(|h| h.old.with_slice(output::hash16))
            .collect();
        let prefix = s.tokens[..end].to_vec();
        if let Some(previous) = self.physical_baseline.get(&prefix) {
            self.trajectory_failures += usize::from(previous != &(kv, history));
        } else {
            self.physical_baseline.insert(prefix, (kv, history));
        }
        Ok(())
    }
    fn attempt(
        &mut self,
        s: &mut Session<'_>,
        tokens: &[u32],
        selected: &[usize],
        before: &State,
        measured: bool,
    ) -> Result<(State, bool)> {
        require(
            self.attempts < 104 && self.rows + tokens.len() <= 367,
            "frozen native attempt bound",
        )?;
        self.attempts += 1;
        self.attempt = self.attempts as u32;
        self.rows += tokens.len();
        if self.run < 8 {
            self.primary_chunks += 1;
        }
        let start = s.cache.offset();
        self.event(
            "attempt_begin",
            json!({"session":self.session,"attempt":self.attempt,"run":self.run,"start":start,
            "tokens":tokens,"selected":selected,"prior_boundary":before.boundary}),
            Class::Other,
        )?;
        self.history(s, HISTORY_PRE)?;
        self.safely_terminal = false;
        let measurement = measured.then(Measurement::start);
        let result = s.forward(tokens);
        let allocation = measurement.map(Measurement::finish);
        if let Some(count) = allocation {
            self.measured += 1;
            self.event("allocation",json!({"session":self.session,"attempt":self.attempt,"call":self.measured,"count":count,
                "scope":"Rust forward only; capture and framework allocations excluded"}),Class::Other)?;
        }
        self.event("attempt_terminal",json!({"session":self.session,"attempt":self.attempt,"result":if result.is_ok(){"ok"}else{"error"},
            "error":result.as_ref().err().map(|e|format!("{e:?}")),"phase_a":s.test_phase_a,
            "cache_completion":s.cache.test_completion(),"poisoned":s.is_poisoned(),"offset":s.cache.offset()}),Class::Other)?;
        s.cache.test_safe_read()?;
        self.safely_terminal = true;
        if s.options.diagnostics {
            self.arrays(s, start, tokens.len(), selected, result.is_ok())?;
        }
        self.history(s, HISTORY_POST)?;
        self.active_cache(s)?;
        let after = self.state(
            s,
            if result.is_ok() {
                "committed"
            } else {
                "failed"
            },
        )?;
        if let Some(count) = allocation {
            require(count == 0, "ordinary forward Rust allocations")?;
        }
        if result.is_ok() {
            self.row_hashes(s, start, tokens.len())?;
            self.actual_state(s)?;
        }
        Ok((after, result.is_ok()))
    }
    fn extra_head(&mut self, s: &Session<'_>, index: u32) -> Result<(usize, Vec<u32>)> {
        s.cache.test_safe_read()?;
        let row = s.cache.offset() - 1;
        let offset = (s.published_rows - 1) * 65536;
        let words = s
            .published
            .with_slice(|v| v[offset..offset + 65536].to_vec());
        let mut t = tag(self.session, self.attempt, HEAD, index, 65536);
        t.query = [row, 1];
        let reference = self
            .writer
            .float(t, 65536, words.iter().copied())
            .map_err(io)?;
        // Compare ordinary public readback to the actual published owner only
        // after retaining raw bits. This is not an independent numerical oracle.
        let mut observed = vec![0f32; 65536];
        s.copy_logits(&mut observed)?;
        require(
            observed
                .iter()
                .map(|x| x.to_bits())
                .eq(words.iter().copied()),
            "public last-row getter differs",
        )?;
        Ok((reference.seq, words))
    }
    fn reject(
        &mut self,
        s: &mut Session<'_>,
        name: &'static str,
        tokens: &[u32],
        before: &State,
    ) -> Result<State> {
        self.safely_terminal = false;
        let error = s.forward(tokens).err().map(|e| format!("{e:?}"));
        self.event(
            "api_reject_result",
            json!({"session":self.session,"attempt":self.attempt,"kind":name,"error":error}),
            Class::Other,
        )?;
        // Even an unexpected preflight regression must establish terminality
        // before any later endpoint observation of shared model buffers.
        s.cache.test_safe_read()?;
        self.safely_terminal = true;
        let after = self.state(s, "preencode_reject")?;
        let unchanged = before.metadata == after.metadata
            && before.old_kv == after.old_kv
            && before.old_history == after.old_history
            && before.published == after.published
            && before.tokens == after.tokens
            && before.published_rows == after.published_rows;
        self.event("api_reject",json!({"session":self.session,"attempt":self.attempt,"kind":name,"error":error,
            "before_boundary":before.boundary,"after_boundary":after.boundary,"command_unchanged":unchanged}),Class::Other)?;
        require(
            error.is_some() && unchanged,
            "preencode rejection changed published state",
        )?;
        Ok(after)
    }
}
fn success(ok: bool) -> Result<()> {
    require(
        ok,
        "unexpected numerical forward failure; raw evidence retained",
    )
}
fn preserved(a: &State, b: &State, kv: bool) -> bool {
    a.old_history == b.old_history
        && a.published == b.published
        && a.tokens == b.tokens
        && a.published_rows == b.published_rows
        && a.metadata["offset"] == b.metadata["offset"]
        && a.metadata["generation"] == b.metadata["generation"]
        && (!kv || a.old_kv == b.old_kv)
}

pub(super) fn execute(c: &mut Capture, m: &Model, protocol: &Value) -> Result<()> {
    let (mut session, mut state) = c.session(m, 1, true)?;
    state = c.reject(&mut session, "empty", &[], &state)?;
    state = c.reject(&mut session, "token65536", &[65536], &state)?;
    state = c.reject(&mut session, "chunk33", &[1; 33], &state)?;
    let mut run = 0;
    for (case_id, case) in protocol["cases"].as_array().unwrap().iter().enumerate() {
        let tokens = token_list(&case["tokens"]);
        let selected: Vec<usize> = case["selected_rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        for partition in case["native_partitions"].as_array().unwrap() {
            c.run = run;
            if run != 0 {
                state = c.reset(&mut session, &state)?;
            }
            let parts: Vec<usize> = partition
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            c.event("run_begin",json!({"run":run,"session":1,"case_id":case_id,"mode":"primary","partition":parts,"selected":selected}),Class::Other)?;
            let mut at = 0;
            for n in parts {
                let (after, ok) =
                    c.attempt(&mut session, &tokens[at..at + n], &selected, &state, false)?;
                state = after;
                success(ok)?;
                at += n;
            }
            require(at == tokens.len(), "native partition extent")?;
            run += 1;
        }
    }
    require(run == 8, "primary schedule inventory")?;
    let plain = token_list(&protocol["cases"][1]["tokens"]);
    c.run = 8;
    state = c.reset(&mut session, &state)?;
    c.event(
        "run_begin",
        json!({"run":8,"session":1,"mode":"teacher","partition":[8,1,1,1],"selected":[]}),
        Class::Other,
    )?;
    for (i, teacher) in protocol["teacher"].as_array().unwrap().iter().enumerate() {
        let tokens = token_list(&teacher["tokens"]);
        let at = session.cache.offset();
        let (after, ok) = c.attempt(&mut session, &tokens[at..], &[], &state, false)?;
        state = after;
        success(ok)?;
        c.extra_head(&session, i as u32)?;
    }
    let mut repeats = Vec::new();
    for repeat in 0..2 {
        c.run = 9 + repeat;
        state = c.reset(&mut session, &state)?;
        c.event("run_begin",json!({"run":c.run,"session":1,"mode":"free","repeat":repeat,"partition":[8,1,1,1],"selected":[]}),Class::Other)?;
        let mut next = plain.clone();
        let mut samples = Vec::new();
        for sample in 0..4 {
            let (after, ok) = c.attempt(&mut session, &next, &[], &state, false)?;
            state = after;
            success(ok)?;
            let (reference, words) = c.extra_head(&session, 8 + repeat * 4 + sample)?;
            let id = greedy(&words)?;
            let stop = id == 7;
            let unmapped = id >= 64400;
            let pending = !stop;
            c.event("sample",json!({"run":c.run,"repeat":repeat,"sample":sample,"distribution_ref":reference,"id":id,
                "stop":stop,"unmapped":unmapped,"pending":pending,"committed_count":session.cache.offset()}),Class::Other)?;
            samples.push(id);
            c.codec_failures += usize::from(unmapped);
            if stop || unmapped {
                break;
            }
            next = vec![id];
        }
        repeats.push(samples);
    }
    c.event(
        "assertions",
        json!({"name":"free_repeat_ids","passed":repeats[0]==repeats[1]}),
        Class::Other,
    )?;
    require(repeats[0] == repeats[1], "free repeat mismatch")?;
    c.event(
        "session_end",
        json!({"session":1,"offset":session.cache.offset(),"poisoned":session.is_poisoned()}),
        Class::Other,
    )?;
    drop(session);
    for (i, fault) in [
        TestFault::OmitHistory(9),
        TestFault::Head,
        TestFault::FirstCopy,
    ]
    .into_iter()
    .enumerate()
    {
        c.run = 11 + i as u32;
        let (mut s, mut before) = c.session(m, 2 + i as u32, true)?;
        c.event("run_begin",json!({"run":c.run,"session":c.session,"mode":"fault","fault":format!("{fault:?}"),"partition":[3,2,3,5],"selected":[]}),Class::Other)?;
        let (after, ok) = c.attempt(&mut s, &plain[..3], &[], &before, false)?;
        before = after;
        success(ok)?;
        let t = tag(
            c.session,
            c.attempt,
            PUBLISHED_BEFORE_FAULT,
            0,
            s.published.len,
        );
        s.published
            .with_slice(|v| c.writer.float(t, v.len(), v.iter().copied()))
            .map_err(io)?;
        c.guard_bytes += GUARD * 4;
        require(
            c.guard_bytes <= GUARD_RESERVE,
            "prior publication guard reserve",
        )?;
        s.test_fault = fault;
        let (after, ok) = c.attempt(&mut s, &plain[3..5], &[], &before, false)?;
        if fault == TestFault::Head {
            let bits = s.candidate.with_slice(|v| v[0]);
            let hash = s.candidate.with_slice(output::hash32);
            c.event("fault_word",json!({"session":c.session,"attempt":c.attempt,"owner":61,"index":0,"bits":bits,"sha256":hash}),Class::Other)?;
            require(bits == u32::MAX, "head injection word absent")?;
        }
        let first_copy = fault == TestFault::FirstCopy;
        c.event("assertions",json!({"name":"fault_publication","fault":format!("{fault:?}"),"failed":!ok,
            "preserved":preserved(&before,&after,!first_copy),"poisoned":s.is_poisoned(),"firstKchanged":before.old_kv[0]!=after.old_kv[0]}),Class::Other)?;
        require(
            !ok && preserved(&before, &after, !first_copy) && s.is_poisoned() == first_copy,
            "fault publication semantics",
        )?;
        if first_copy {
            require(
                before.old_kv[0] != after.old_kv[0] && before.old_kv[1..] == after.old_kv[1..],
                "first-copy physical mutation",
            )?;
            let mut out = vec![0.; 65536];
            let getter = s.copy_logits(&mut out).is_err() && s.committed_tokens().is_err();
            c.event(
                "assertions",
                json!({"name":"poison_getters","refused":getter}),
                Class::Other,
            )?;
            require(getter, "poison getter admitted")?;
            before = c.reject(&mut s, "poisoned", &plain[5..6], &after)?;
        } else {
            before = after;
        }
        s.test_fault = TestFault::None;
        before = c.reset(&mut s, &before)?;
        for chunk in [&plain[..3], &plain[3..]] {
            let (after, ok) = c.attempt(&mut s, chunk, &[], &before, false)?;
            before = after;
            success(ok)?;
        }
        c.event(
            "session_end",
            json!({"session":c.session,"offset":s.cache.offset(),"poisoned":s.is_poisoned()}),
            Class::Other,
        )?;
        drop(s);
    }
    c.run = 14;
    let (mut ordinary, mut state) = c.session(m, 5, false)?;
    c.event(
        "run_begin",
        json!({"run":14,"session":5,"mode":"ordinary","partition":[32,32],"selected":[]}),
        Class::Other,
    )?;
    let journey = token_list(&protocol["cases"][3]["tokens"]);
    for i in 0..2 {
        let (after, ok) = c.attempt(
            &mut ordinary,
            &journey[i * 32..(i + 1) * 32],
            &[],
            &state,
            true,
        )?;
        state = after;
        success(ok)?;
        c.extra_head(&ordinary, 16 + i as u32)?;
    }
    c.event(
        "session_end",
        json!({"session":5,"offset":64,"poisoned":ordinary.is_poisoned()}),
        Class::Other,
    )?;
    drop(ordinary);
    budgets(c, m)?;
    c.event("assertions",json!({"name":"full_capture_inventory","attempts":c.attempts,"rows":c.rows,"primary_chunks":c.primary_chunks,
        "primary_rows":c.primary_rows,"selected_groups":c.selected_groups,"producer_rows":c.producer_rows,"history_stacks":c.history_stacks,
        "measured":c.measured,"same_native_failures":c.trajectory_failures,"codec_failures":c.codec_failures}),Class::Other)?;
    require(
        c.primary_chunks == 78
            && c.primary_rows == 231
            && c.selected_groups == 6739
            && c.history_stacks == 18
            && c.measured == 2
            && c.attempts <= 104
            && c.rows <= 367
            && c.producer_rows <= 303
            && c.trajectory_failures == 0
            && c.codec_failures == 0,
        "capture inventory/state/codec failure",
    )
}
fn budgets(c: &mut Capture, m: &Model) -> Result<()> {
    for (i, (diagnostics, all_logits, name)) in [
        (false, false, "ordinary_last_row"),
        (false, true, "all_logits"),
        (true, true, "diagnostic_all_logits"),
    ]
    .into_iter()
    .enumerate()
    {
        // Actual ordinary/diagnostic admissions and the exact two-logit-bank
        // delta fix the third bound, without another successful construction.
        let base = options(diagnostics, all_logits);
        let required =
            c.admissions[i].ok_or(Error::Options("missing observed session admission"))?;
        let mut exact = base;
        exact.memory_limit_bytes = required;
        let s = m.session(exact)?;
        c.event("session_begin",json!({"session":6+i,"options":{"diagnostics":diagnostics,"all_logits":all_logits},"memory":memory(s.memory()),"budget_only":true}),Class::Boundary)?;
        c.event("budget",json!({"session":6+i,"mode":name,"limit":required,"required":required,"accepted":true,"error":null}),Class::Other)?;
        require(
            s.memory().admitted == required,
            "exact admitted budget changed",
        )?;
        drop(s);
        let mut under = base;
        under.memory_limit_bytes = required - 1;
        let result = m.session(under);
        let error = result.as_ref().err().map(|e| format!("{e:?}"));
        c.event("budget",json!({"session":6+i,"mode":name,"limit":required-1,"required":required,"accepted":result.is_ok(),"error":error}),Class::Other)?;
        require(
            matches!(result,Err(Error::Budget{required:r,limit})if r==required&&limit==required-1),
            "one-byte-under budget admission",
        )?;
    }
    Ok(())
}
