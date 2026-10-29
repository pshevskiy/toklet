use super::*;

impl Capture {
    pub(super) fn event(&mut self, kind: &'static str, body: Value, class: Class) -> Result<()> {
        self.writer.event(kind, &body, class).map_err(io)
    }
    pub(super) fn history(&mut self, s: &Session<'_>, kind: u32) -> Result<()> {
        s.cache.test_safe_read()?;
        for (owner, h) in s.histories.iter().enumerate() {
            for (role, b) in [&h.old, &h.candidate].into_iter().enumerate() {
                let t = tag(
                    self.session,
                    self.attempt,
                    kind,
                    (2 * owner + role) as u32,
                    b.len,
                );
                b.with_slice(|words| self.writer.half(t, words.len(), words.iter().copied()))
                    .map_err(io)?;
            }
        }
        Ok(())
    }
    pub(super) fn initial_cache(&mut self, s: &Session<'_>) -> Result<()> {
        s.cache.test_visit_buffers(|owner, role, _, _, words| {
            self.writer
                .half(
                    tag(
                        self.session,
                        0,
                        KV_INITIAL,
                        (owner * 4 + role) as u32,
                        words.len(),
                    ),
                    words.len(),
                    words.iter().copied(),
                )
                .map_err(io)?;
            Ok(())
        })
    }
    pub(super) fn active_cache(&mut self, s: &Session<'_>) -> Result<()> {
        let end = s.cache.offset();
        // On poisoned partial copy the valid published prefix is unchanged;
        // full physical hashes and raw producers separately expose uncertain tail.
        if end == 0 {
            return Ok(());
        }
        s.cache.test_visit_buffers(|owner, role, _, _, words| {
            if role < 2 {
                let mut t = tag(
                    self.session,
                    self.attempt,
                    KV_ACTIVE,
                    (owner * 2 + role) as u32,
                    8 * end * 64,
                );
                t.shape = [8, end, 64];
                t.rank = 3;
                t.keys = [0, end];
                self.writer
                    .half(
                        t,
                        8 * end * 64,
                        (0..8).flat_map(|h| {
                            words[h * 512 * 64..h * 512 * 64 + end * 64].iter().copied()
                        }),
                    )
                    .map_err(io)?;
            }
            Ok(())
        })
    }
    pub(super) fn state(&mut self, s: &Session<'_>, label: &'static str) -> Result<State> {
        s.cache.test_safe_read()?;
        let metadata = s.cache.test_capture_metadata()?;
        let mut physical = BTreeSet::new();
        let mut guards = Vec::<u8>::with_capacity(6000);
        let mut guard_layout = Vec::<(usize, usize, usize, u32)>::new();
        let mut owners = Vec::<Value>::with_capacity(380);
        let mut register = Vec::with_capacity(380);
        let mut add_guard = |tail: &[u32], scalar: usize, expected: u32| {
            guard_layout.push((guards.len(), tail.len(), scalar, expected));
            for word in tail {
                guards.extend_from_slice(&word.to_le_bytes()[..scalar]);
            }
        };
        let mut old_kv = Vec::with_capacity(12);
        s.cache
            .test_visit_buffers(|owner, role, address, device, words| {
                let id = (owner * 4 + role) as u32;
                let hash = output::hash16(words);
                owners.push(json!([id, address, hash]));
                register.push((id, address, words.len() * 2, device));
                physical.insert(address);
                let tail: Vec<_> = words[words.len() - GUARD..]
                    .iter()
                    .map(|&x| u32::from(x))
                    .collect();
                add_guard(&tail, 2, u32::from(HALF_PAD));
                if role < 2 {
                    old_kv.push(hash);
                }
                Ok(())
            })?;
        let mut old_history = Vec::with_capacity(10);
        for (owner, h) in s.histories.iter().enumerate() {
            for (role, b) in [&h.old, &h.candidate].into_iter().enumerate() {
                let id = 24 + (owner * 2 + role) as u32;
                b.with_slice(|words| {
                    let hash = output::hash16(words);
                    owners.push(json!([id, b.raw.contents().as_ptr() as usize, hash]));
                    register.push((
                        id,
                        b.raw.contents().as_ptr() as usize,
                        words.len() * 2,
                        b.raw.device().registryID(),
                    ));
                    if role == 0 {
                        old_history.push(hash);
                    }
                    let tail: Vec<_> = words[words.len() - GUARD..]
                        .iter()
                        .map(|&x| u32::from(x))
                        .collect();
                    add_guard(&tail, 2, u32::from(HALF_PAD));
                });
                physical.insert(b.raw.contents().as_ptr() as usize);
            }
        }
        for (i, b) in [
            &s.scratch.hidden,
            &s.scratch.norm,
            &s.scratch.branch,
            &s.scratch.bcx,
            &s.scratch.u,
            &s.scratch.z,
            &s.scratch.q,
            &s.scratch.work,
            &s.scratch.k,
            &s.scratch.k_norm,
            &s.scratch.v,
            &s.scratch.gate,
            &s.scratch.up,
            &s.scratch.activated,
        ]
        .into_iter()
        .enumerate()
        {
            let id = 44 + i as u32;
            b.with_slice(|words| {
                owners.push(json!([
                    id,
                    b.raw.contents().as_ptr() as usize,
                    output::hash16(words)
                ]));
                register.push((
                    id,
                    b.raw.contents().as_ptr() as usize,
                    words.len() * 2,
                    b.raw.device().registryID(),
                ));
                let tail: Vec<_> = words[words.len() - GUARD..]
                    .iter()
                    .map(|&x| u32::from(x))
                    .collect();
                add_guard(&tail, 2, u32::from(HALF_PAD));
            });
            physical.insert(b.raw.contents().as_ptr() as usize);
        }
        for (i, b) in [
            &s.scratch.scores,
            &s.scratch.probabilities,
            &s.scratch.tokens,
            &s.candidate,
            &s.published,
        ]
        .into_iter()
        .enumerate()
        {
            let id = 58 + i as u32;
            b.with_slice(|words| {
                owners.push(json!([
                    id,
                    b.raw.contents().as_ptr() as usize,
                    output::hash32(words)
                ]));
                register.push((
                    id,
                    b.raw.contents().as_ptr() as usize,
                    words.len() * 4,
                    b.raw.device().registryID(),
                ));
                add_guard(&words[words.len() - GUARD..], 4, WORD_PAD);
            });
            physical.insert(b.raw.contents().as_ptr() as usize);
        }
        s.visit_capture_slots(|slot, words| {
            let id = 64 + slot.slot as u32;
            owners.push(json!([id, slot.address, output::hash32(words)]));
            register.push((id, slot.address, words.len() * 4, slot.device));
            physical.insert(slot.address);
            let n = if slot.half { 4 } else { 7 };
            add_guard(
                &words[words.len() - n..],
                4,
                if slot.half {
                    u32::from(HALF_PAD) * 0x10001
                } else {
                    WORD_PAD
                },
            );
            Ok(())
        })?;
        let status = &metadata["status"];
        let status_address = usize::try_from(status["address"].as_u64().unwrap()).unwrap();
        let status_words: Vec<u32> = status["words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| u32::try_from(x.as_u64().unwrap()).unwrap())
            .collect();
        add_guard(&status_words[1..], 4, WORD_PAD);
        physical.insert(status_address);
        owners.push(json!([63, status_address, output::hash32(&status_words)]));
        register.push((63, status_address, 32, status["device"].as_u64().unwrap()));
        // Exact mixed-width guard bytes, in the same owner-table order. Each
        // owner type fixes its tail extent; no expanded Half guard words.
        self.guard_bytes = self
            .guard_bytes
            .checked_add(guards.len())
            .ok_or(Error::Options("capture guard overflow"))?;
        if self.guard_bytes > GUARD_RESERVE {
            return Err(Error::Options("capture guard reserve"));
        }
        let range = self
            .writer
            .bytes(
                tag(
                    self.session,
                    self.attempt,
                    GUARDS,
                    self.guard_records as u32,
                    guards.len(),
                ),
                &guards,
            )
            .map_err(io)?;
        let actual_registry: BTreeMap<_, _> = register
            .iter()
            .map(|&(_, address, bytes, device)| (address, (bytes, device)))
            .collect();
        if self.registry.is_empty() {
            self.event(
                "session_owners",
                json!({"session":self.session,"owners":register,"status_id":63}),
                Class::Boundary,
            )?;
            self.registry = actual_registry.clone();
        }
        let registry_equal = self.registry == actual_registry;
        let boundary = self.writer.prefix().map_err(io)?.records;
        self.guard_records += 1;
        let published = s.published.with_slice(output::hash32);
        let state = State {
            boundary,
            metadata: metadata.clone(),
            old_kv,
            old_history,
            published: published.clone(),
            published_rows: s.published_rows,
            tokens: s.tokens.clone(),
            physical,
        };
        self.event(
            "boundary",
            json!({"session":self.session,"attempt":self.attempt,"label":label,
            "cache":metadata,"owners":owners,"guard_array":range.seq,
            "published_rows":s.published_rows,"diagnostic_rows":s.diagnostic_rows,
            "tokens":s.tokens,"model_table":"model_before"}),
            Class::Boundary,
        )?;
        // The raw aggregate and complete boundary precede any guard assertion.
        for (start, count, scalar, expected) in guard_layout {
            let expected = expected.to_le_bytes();
            if !guards[start..start + count * scalar]
                .chunks_exact(scalar)
                .all(|x| x == &expected[..scalar])
            {
                return Err(Error::Options("captured guard mismatch"));
            }
        }
        if !registry_equal {
            return Err(Error::Options("capture owner capacity/device changed"));
        }
        if state.physical.len() != owners.len() {
            return Err(Error::Options("captured physical owner alias"));
        }
        Ok(state)
    }
    pub(super) fn arrays(
        &mut self,
        s: &Session<'_>,
        start: usize,
        n: usize,
        selected: &[usize],
        success: bool,
    ) -> Result<()> {
        s.visit_capture_slots(|slot, words| {
            let stage_name = slot.stage;
            let producer = match stage_name {
                "u" => Some(U),
                "k_rotary" => Some(KROT),
                "v_projection" => Some(VPROJ),
                _ => None,
            };
            if let Some(kind) = producer {
                let mut t = tag(
                    self.session,
                    self.attempt,
                    kind,
                    slot.layer.unwrap() as u32,
                    n * slot.width,
                );
                t.shape = [n, slot.width, 1];
                t.rank = 2;
                t.query = [start, n];
                self.writer
                    .half(
                        t,
                        n * slot.width,
                        (0..n * slot.width).map(|i| (words[i / 2] >> (16 * (i % 2))) as u16),
                    )
                    .map_err(io)?;
            }
            if success {
                for &absolute in selected {
                    if !(start..start + n).contains(&absolute) {
                        continue;
                    }
                    let row = absolute - start;
                    let mut t = tag(
                        self.session,
                        self.attempt,
                        STAGE,
                        slot.slot as u32,
                        slot.width,
                    );
                    t.query = [absolute, 1];
                    if slot.scores {
                        let keys = start + n;
                        t.shape = [16, 1, keys];
                        t.rank = 3;
                        t.keys = [0, keys];
                        self.writer
                            .float(
                                t,
                                16 * keys,
                                (0..16).flat_map(|h| {
                                    words[(h * 32 + row) * 512..(h * 32 + row) * 512 + keys]
                                        .iter()
                                        .copied()
                                }),
                            )
                            .map_err(io)?;
                    } else if slot.half {
                        self.writer
                            .half(
                                t,
                                slot.width,
                                (row * slot.width..(row + 1) * slot.width)
                                    .map(|i| (words[i / 2] >> (16 * (i % 2))) as u16),
                            )
                            .map_err(io)?;
                    } else {
                        self.writer
                            .float(
                                t,
                                slot.width,
                                words[row * slot.width..(row + 1) * slot.width]
                                    .iter()
                                    .copied(),
                            )
                            .map_err(io)?;
                    }
                    self.selected_groups += 1;
                }
            }
            Ok(())
        })?;
        self.producer_rows += n;
        Ok(())
    }
}
