#[doc(hidden)]
pub struct SharedMlpSetup {
    workspace: crate::metal::SharedMlpWorkspace,
    output_published: [u64; 2],
    qkv_budget: usize,
    qkv_published: u64,
    cancelled: &'static std::sync::atomic::AtomicBool,
    tied_published: u64,
    attention_budget: usize,
    output_tag_seconds: f64,
    output_budget: usize,
    projection_tile_budget: usize,
    projection_tile_tag_seconds: f64,
    required: usize,
    weight_bytes: usize,
    padding_bytes: usize,
    allowance: usize,
    planning_seconds: f64,
    workspace_seconds: f64,
    load_seconds: f64,
    tag_seconds: f64,
    published: u64,
}

impl<'a> GenerationSession<'a> {
    #[doc(hidden)]
    pub fn load_metal(
        directory: &std::path::Path,
        budget: usize,
        cancelled: &'static std::sync::atomic::AtomicBool,
    ) -> Result<(crate::weights::LoadedLfm3, SharedMlpSetup), Box<dyn std::error::Error>> {
        use crate::{
            metal::SharedMlpWorkspace,
            weights::lfm3::{load_lfm3_shared, plan_lfm3_shared},
        };
        let (capacity, chunk) = (2048, 128);
        // SAFETY: this sysconf selector has no pointer arguments.
        let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .map_err(|_| super::super::Error::InvalidConfig("LFM3 Metal page size"))?;
        if !page.is_power_of_two() {
            return Err(super::super::Error::InvalidConfig("LFM3 Metal page geometry").into());
        }
        let aligned = |bytes: usize| add(bytes, page - 1).map(|n| n & !(page - 1));
        let padding = |count: usize| crate::cpu::product(page, count);
        const FRAMEWORK_AND_GUARD: usize = 128 << 20;
        let tied_payload = add(
            crate::cpu::product(10, super::VOCAB)?,
            2 * super::HIDDEN + 18,
        )?;
        let tied_budget = add(tied_payload, padding(6)?)?;
        let output_budget = add(
            SharedMlpWorkspace::OUTPUT_BYTES,
            padding(SharedMlpWorkspace::OUTPUT_BUFFERS)?,
        )?;
        let projection_tile_budget = add(
            SharedMlpWorkspace::PROJECTION_TILE_BYTES,
            padding(SharedMlpWorkspace::PROJECTION_TILE_BUFFERS)?,
        )?;
        let qkv_budget = SharedMlpWorkspace::QKV_BUFFER_BYTES
            .into_iter()
            .try_fold(0, |total, bytes| add(total, aligned(bytes)?))?;
        let attention_budget = SharedMlpWorkspace::LAST3_ATTENTION_BUFFERS
            .into_iter()
            .try_fold(
                SharedMlpWorkspace::LAST3_ATTENTION_NATIVE_ALLOWANCE,
                |total, bytes| add(total, aligned(bytes)?),
            )?;
        let clock = std::time::Instant::now();
        let device = objc2_metal::MTLCreateSystemDefaultDevice().ok_or(
            super::super::Error::InvalidConfig("LFM3 shared device unavailable"),
        )?;
        let extra = add(
            add(
                Self::planned_extra_bytes(capacity)?,
                SharedMlpWorkspace::BYTES
                    + SharedMlpWorkspace::DOMAIN_BYTES
                    + SharedMlpWorkspace::CONV_BYTES,
            )?,
            add(
                add(
                    add(FRAMEWORK_AND_GUARD, output_budget)?,
                    projection_tile_budget,
                )?,
                add(qkv_budget, tied_budget)?,
            )?,
        )?;
        let mut extra = add(extra, attention_budget)?;
        let mut plan = plan_lfm3_shared(&device, capacity, chunk, extra, budget)?;
        let limit = budget.min(crate::weights::INITIAL_MEMORY_LIMIT_BYTES);
        let resident_plan = aligned(SharedMlpWorkspace::RESIDENT_MLP_LUT_TEMP_BYTES)
            .and_then(|initial| {
                SharedMlpWorkspace::RESIDENT_MLP_BUFFERS
                    .into_iter()
                    .try_fold(initial, |total, bytes| add(total, aligned(bytes)?))
            })
            .ok()
            .and_then(|bytes| bytes.checked_add(SharedMlpWorkspace::RESIDENT_MLP_OWNER_BYTES))
            .and_then(|bytes| {
                let required = plan.required_bytes.checked_add(bytes)?;
                (required <= limit).then_some((bytes, required))
            });
        let planning_seconds = clock.elapsed().as_secs_f64();
        let clock = std::time::Instant::now();
        let mut workspace = SharedMlpWorkspace::new(&device)?;
        workspace.prepare_qkv()?;
        workspace.prepare_last3_attention(true)?;
        if let Some((bytes, required)) = resident_plan
            && workspace.prepare_resident_mlp(bytes).is_ok()
        {
            extra = add(extra, bytes)?;
            plan.required_bytes = required;
        }
        let workspace_seconds = clock.elapsed().as_secs_f64();
        let clock = std::time::Instant::now();
        let loaded = load_lfm3_shared(directory, &device, capacity, chunk, extra, budget)?;
        {
            let weights = &loaded.model.weights;
            if !weights
                .shared_bank(weights.embedding)
                .is_some_and(|bank| bank.aliases(weights.get(weights.embedding)))
            {
                return Err(
                    super::super::Error::InvalidConfig("LFM3 shared tied table alias").into(),
                );
            }
        }
        {
            let weights = &loaded.model.weights;
            let bank = weights.shared_bank(weights.embedding).ok_or(
                super::super::Error::InvalidConfig("LFM3 tied head bank missing"),
            )?;
            workspace.seal_tied(bank, weights.get(weights.embedding))?;
        }
        let load_seconds = clock.elapsed().as_secs_f64();
        let tag_seconds = {
            let clock = std::time::Instant::now();
            let weights = &loaded.model.weights;
            let mut views = [(None, None); super::LAYERS];
            for (layer, (gate, up)) in weights.layers.iter().zip(&mut views) {
                let view = |tensor| {
                    weights
                        .shared_bank(tensor)
                        .map(|bank| (bank, weights.get(tensor)))
                        .ok_or(super::super::Error::InvalidConfig(
                            "LFM3 missing sealed tag bank",
                        ))
                };
                *gate = Some(view(layer.gate)?);
                *up = Some(view(layer.up)?);
            }
            // Each entry was assigned by the complete typed 24-layer loop above.
            workspace.seal_pair_tags(views.map(|(gate, up)| [gate.unwrap(), up.unwrap()]))?;
            clock.elapsed().as_secs_f64()
        };
        let output_tag_seconds = {
            let clock = std::time::Instant::now();
            let weights = &loaded.model.weights;
            for (index, layer) in weights.layers.iter().enumerate() {
                let tensor = operator_output(layer.operator);
                let bank =
                    weights
                        .shared_bank(tensor)
                        .ok_or(super::super::Error::InvalidConfig(
                            "LFM3 output bank absent",
                        ))?;
                workspace.seal_output_tags(index, bank, weights.get(tensor))?;
            }
            clock.elapsed().as_secs_f64()
        };
        let projection_tile_tag_seconds = {
            let clock = std::time::Instant::now();
            let weights = &loaded.model.weights;
            for (index, layer) in weights.layers.iter().enumerate() {
                let bank = |tensor| {
                    weights
                        .shared_bank(tensor)
                        .ok_or(super::super::Error::InvalidConfig(
                            "LFM3 missing projection tile bank",
                        ))
                };
                workspace.seal_projection_tile(
                    1,
                    index,
                    bank(layer.down)?,
                    weights.get(layer.down),
                )?;
                if let crate::weights::lfm3::Operator::Convolution { bcx, .. } = layer.operator {
                    workspace.seal_projection_tile(0, index, bank(bcx)?, weights.get(bcx))?;
                }
            }
            clock.elapsed().as_secs_f64()
        };
        {
            let weights = &loaded.model.weights;
            for (i, layer) in weights.layers.iter().enumerate() {
                if let crate::weights::lfm3::Operator::Attention { q, k, v, .. } = layer.operator {
                    let view = |tensor| {
                        weights
                            .shared_bank(tensor)
                            .map(|bank| (bank, weights.get(tensor)))
                            .ok_or(super::super::Error::InvalidConfig(
                                "LFM3 QKV sealed bank absent",
                            ))
                    };
                    workspace.seal_qkv(i, [view(q)?, view(k)?, view(v)?])?;
                }
            }
        }
        Ok((
            loaded,
            SharedMlpSetup {
                workspace,
                required: plan.required_bytes,
                weight_bytes: plan.weights_bytes,
                padding_bytes: plan.weight_padding_bytes,
                allowance: plan.cold_allowance_bytes,
                planning_seconds,
                workspace_seconds,
                load_seconds,
                tag_seconds,
                published: 0,
                output_published: [0; 2],
                qkv_budget,
                qkv_published: 0,
                attention_budget,
                cancelled,
                tied_published: 0,
                output_tag_seconds,
                output_budget,
                projection_tile_budget,
                projection_tile_tag_seconds,
            },
        ))
    }

    pub fn attach_metal(&mut self, setup: SharedMlpSetup) -> Result<(), session::Error> {
        use super::super::Error;
        if self.shared.is_some()
            || !self.history.is_empty()
            || self.inner.model.capacity != 2048
            || self.inner.model.chunk != 128
        {
            return Err(
                Error::InvalidConfig("LFM3 shared MLP requires a fresh 2048/T128 owner").into(),
            );
        }
        let weights = &self.inner.model.weights;
        let mut identities =
            [0; super::LAYERS * 4 + super::CONV_LAYERS.len() + crate::weights::lfm3::QKV_BANKS];
        let (pair_down, conv_ids) = identities.split_at_mut(super::LAYERS * 3);
        for (layer, ids) in weights.layers.iter().zip(pair_down.chunks_exact_mut(3)) {
            for (tensor, id) in [layer.gate, layer.up, layer.down].into_iter().zip(ids) {
                let bank = weights
                    .shared_bank(tensor)
                    .ok_or(Error::InvalidConfig("LFM3 shared MLP bank absent"))?;
                setup.workspace.validate(bank)?;
                *id = bank.identity();
            }
        }
        let mut active_count = {
            for (index, identity) in super::CONV_LAYERS.into_iter().zip(conv_ids) {
                let crate::weights::lfm3::Operator::Convolution { bcx, .. } =
                    weights.layers[index].operator
                else {
                    return Err(Error::InvalidConfig("LFM3 shared ConvIn selected operator").into());
                };
                let bank = weights
                    .shared_bank(bcx)
                    .ok_or(Error::InvalidConfig("LFM3 shared ConvIn bank absent"))?;
                setup.workspace.validate_conv(bank)?;
                *identity = bank.identity();
            }
            super::LAYERS * 3 + super::CONV_LAYERS.len()
        };
        {
            for (layer, identity) in weights.layers.iter().zip(&mut identities[active_count..]) {
                *identity = weights
                    .shared_bank(operator_output(layer.operator))
                    .ok_or(Error::InvalidConfig("LFM3 output bank absent"))?
                    .identity();
            }
            active_count += super::LAYERS;
        }
        {
            for layer in &weights.layers {
                if let crate::weights::lfm3::Operator::Attention { q, k, v, .. } = layer.operator {
                    for (tensor, width) in
                        [q, k, v]
                            .into_iter()
                            .zip([super::HIDDEN, super::KV_WIDTH, super::KV_WIDTH])
                    {
                        let value = weights
                            .shared_bank(tensor)
                            .ok_or(Error::InvalidConfig("LFM3 shared QKV bank absent"))?;
                        setup.workspace.validate_qkv(value, width)?;
                        identities[active_count] = value.identity();
                        active_count += 1;
                    }
                }
            }
            if active_count != identities.len() {
                return Err(Error::InvalidConfig("LFM3 incomplete shared QKV owner").into());
            }
        }
        let active = &mut identities[..active_count];
        active.sort_unstable();
        if active[0] == 0 || active.windows(2).any(|p| p[0] == p[1]) {
            return Err(Error::InvalidConfig("LFM3 shared MLP bank aliases").into());
        }
        self.memory.admitted_bytes = setup.required;
        self.memory.metadata_allowance_bytes = setup
            .required
            .checked_sub(self.inner.model.memory().total_bytes)
            .and_then(|n| n.checked_sub(Self::planned_extra_bytes(self.inner.model.capacity).ok()?))
            .ok_or(Error::InvalidConfig("LFM3 shared MLP admission accounting"))?;
        self.shared = Some(setup);
        Ok(())
    }

    fn step_shared<'step>(
        inner: &'step mut Session<'a>,
        setup: &mut SharedMlpSetup,
        tokens: &[u32],
    ) -> Result<&'step [half::bf16], session::Error> {
        use super::super::Error;
        let weights = &inner.model.weights;
        let bank = |tensor| {
            weights
                .shared_bank(tensor)
                .ok_or(Error::InvalidConfig("LFM3 shared MLP bank absent"))
        };
        let visits = super::LAYERS as u64;
        let resident_next = if tokens.len() == 128 {
            setup
                .workspace
                .resident_mlp_published()
                .map(|published| {
                    published
                        .checked_add(visits)
                        .ok_or(crate::cpu::Error::SizeOverflow)
                })
                .transpose()
                .map_err(Error::from)?
        } else {
            None
        };
        let resident_mlp = resident_next.is_some();
        let next = setup
            .published
            .checked_add(if resident_mlp { 0 } else { visits })
            .ok_or(crate::cpu::Error::SizeOverflow)
            .map_err(Error::from)?;
        let mut output_next = setup.output_published;
        for (v, n) in output_next.iter_mut().zip([
            super::CONV_LAYERS.len(),
            super::LAYERS - super::CONV_LAYERS.len(),
        ]) {
            *v = v
                .checked_add(n as u64)
                .ok_or(crate::cpu::Error::SizeOverflow)
                .map_err(Error::from)?;
        }
        let gpu_qkv = tokens.len() == 128;
        let qkv_next = setup
            .qkv_published
            .checked_add(if gpu_qkv {
                (super::LAYERS - super::CONV_LAYERS.len()) as u64
            } else {
                0
            })
            .ok_or(crate::cpu::Error::SizeOverflow)
            .map_err(Error::from)?;
        let cancelled = setup.cancelled;
        let eligible_head = |count| matches!(count, 1 | 128);
        let tied_next = setup
            .tied_published
            .checked_add(u64::from(eligible_head(tokens.len())))
            .ok_or(crate::cpu::Error::SizeOverflow)
            .map_err(Error::from)?;
        let workspace = std::cell::RefCell::new(&mut setup.workspace);
        let logits = inner.step_placed::<false, true, _, _, _, _, _, _, _, _, _>(
            tokens,
            |_| unreachable!(),
            resident_mlp,
            |layer, input, output| {
                let ffn = &weights.layers[layer];
                workspace.borrow_mut().resident_mlp(
                    layer,
                    [bank(ffn.gate)?, bank(ffn.up)?, bank(ffn.down)?],
                    input,
                    output,
                    cancelled,
                )
            },
            |layer, input, out, count| {
                let crate::weights::lfm3::Operator::Convolution { bcx, .. } =
                    weights.layers[layer].operator
                else {
                    return Err(Error::InvalidConfig("LFM3 shared ConvIn selected operator"));
                };
                let mut workspace = workspace.borrow_mut();
                workspace.conv(layer, bank(bcx)?, input, out, count)?;
                Ok(())
            },
            |layer_index, input, gate, up, count| {
                let layer = &weights.layers[layer_index];
                let banks = [bank(layer.gate)?, bank(layer.up)?];
                let mut workspace = workspace.borrow_mut();
                workspace.pair(layer_index, banks, input, gate, up, count)?;
                Ok(())
            },
            |layer, input, out, count| {
                let down = weights.layers[layer].down;
                let mut workspace = workspace.borrow_mut();
                workspace.down(layer, bank(down)?, input, out, count)?;
                Ok(())
            },
            |layer, role, input, out, count| {
                workspace.borrow_mut().output(
                    layer,
                    role,
                    bank(operator_output(weights.layers[layer].operator))?,
                    input,
                    out,
                    count,
                )
            },
            |layer, input, out| {
                if !gpu_qkv {
                    return Ok(false);
                }
                let crate::weights::lfm3::Operator::Attention { q, k, v, .. } =
                    weights.layers[layer].operator
                else {
                    return Err(Error::InvalidConfig(
                        "LFM3 QKV selected non-attention layer",
                    ));
                };
                workspace
                    .borrow_mut()
                    .qkv(layer, [bank(q)?, bank(k)?, bank(v)?], input, out)?;
                Ok(true)
            },
            |input, out, count| {
                if !eligible_head(count) {
                    return Ok(false);
                }
                workspace.borrow_mut().tied(
                    bank(weights.embedding)?,
                    weights.get(weights.embedding),
                    input,
                    out,
                    cancelled,
                )?;
                if cancelled.load(std::sync::atomic::Ordering::Acquire) {
                    return Err(Error::Cancelled);
                }
                Ok(true)
            },
            |layer, q, spans, out, shape| {
                workspace
                    .borrow_mut()
                    .apply_last3_attention(layer, q, spans, out, shape)
            },
        )?;
        setup.tied_published = tied_next;
        setup.qkv_published = qkv_next;
        setup.output_published = output_next;
        setup.published = next;
        if let Some(published) = resident_next {
            workspace.into_inner().commit_resident_mlp(published);
        }
        Ok(logits)
    }

    #[doc(hidden)]
    pub fn metal_report(&self) -> serde_json::Value {
        self.shared
            .as_ref()
            .map_or(serde_json::Value::Null, |owner| self.shared_report(owner))
    }

    fn shared_report(&self, setup: &SharedMlpSetup) -> serde_json::Value {
        use sha2::{Digest, Sha256};
        let mut logits = Sha256::new();
        if let Some(values) = self.inner.logits() {
            for value in values {
                logits.update(value.to_le_bytes());
            }
        }
        let mut history = Sha256::new();
        for id in &self.history {
            history.update(id.to_le_bytes());
        }
        let (submitted, completed, clocks) = setup.workspace.telemetry();
        let domain = setup
            .workspace
            .domain_stats()
            .map(|(native, counts, possible)| {
                serde_json::json!({
                    "mode": if native { "row-certified" } else { "ordered" },
                    "tag_bytes": crate::metal::SharedMlpWorkspace::DOMAIN_BYTES,
                    "tag_seconds": setup.tag_seconds,
                    "possible_per_role": possible,
                    "native_by_role": counts,
                    "ordered_by_role": [possible-counts[0], possible-counts[1]],
                })
            });
        let mut report = serde_json::json!({
            "mode": "gate-up-down",
            "weight_admission":"sealed-input-only",
            "shared_mlp_weight_bytes": setup.weight_bytes,
            "shared_mlp_weight_padding_bytes": setup.padding_bytes,
            "cold_loader_allowance_bytes": setup.allowance,
            "numerical_admitted_bytes": setup.required,
            "shared_workspace_bytes": crate::metal::SharedMlpWorkspace::BYTES,
            "cold_plan_seconds": setup.planning_seconds,
            "cold_workspace_seconds": setup.workspace_seconds,
            "cold_load_seconds": setup.load_seconds,
            "submitted_mlp_pair_commands": submitted[0],
            "completed_mlp_pair_commands": completed[0],
            "published_mlp_pair_commands": setup.published,
            "submitted_mlp_down_commands": submitted[1],
            "completed_mlp_down_commands": completed[1],
            "published_mlp_down_commands": setup.published,
            "pair_completed_gpu_seconds": clocks[0][0],
            "pair_commit_wait_seconds": clocks[0][1],
            "down_completed_gpu_seconds": clocks[1][0],
            "down_commit_wait_seconds": clocks[1][1],
            "evaluated_len": self.evaluated_len(),
            "pending_len": self.pending_len(),
            "history_u32le_sha256": format!("{:x}",history.finalize()),
            "committed_logits_bf16le_sha256": self.inner.logits().map(|_|format!("{:x}",logits.finalize())),
        });
        if let Some(resident) = setup.workspace.resident_mlp_report() {
            report["resident_mlp"] = resident;
        }
        if let Some(domain) = domain {
            report["gate_up_domain"] = domain;
        }
        if let Some([attempted, submitted, completed, accepted, numeric_fallback]) =
            setup.workspace.last3_attention_stats()
        {
            report["last3_attention"] = serde_json::json!({
                "mode": "metal-t128",
                "admitted_bytes": setup.attention_budget,
                "workspace_payload_bytes": crate::metal::SharedMlpWorkspace::LAST3_ATTENTION_BUFFERS.iter().sum::<usize>(),
                "attempted": attempted, "submitted": submitted, "completed": completed,
                "accepted": accepted, "numeric_fallback": numeric_fallback
            });
        }
        let qkv_count = crate::weights::lfm3::QKV_BANKS;
        let qkv_bytes = crate::weights::lfm3::QKV_BYTES;
        {
            report["qkv_owner"] = serde_json::json!({
                "placement":"metal-t128",
                "banks":qkv_count,"payload_bytes":qkv_bytes,"admitted_bytes":setup.qkv_budget,
                "published_sets":setup.qkv_published,"roles":setup.workspace.qkv_stats(),
            });
        }
        let tied_count = 1;
        let tied_bytes = tied_count * crate::weights::lfm3::TIED_BYTES;
        {
            report["tied_table_owner"] = serde_json::json!({
                "placement":"metal-t1-t128",
                "banks":1, "payload_bytes":tied_bytes, "published":setup.tied_published,
                "workspace":setup.workspace.tied_report(),
            });
        }
        if let Some((submitted, completed, clocks)) = setup.workspace.conv_stats() {
            const CONV_BYTES: usize =
                super::CONV_LAYERS.len() * super::HIDDEN * 3 * super::HIDDEN * 2;
            let output_bytes = super::LAYERS * 2 * super::HIDDEN * super::HIDDEN;
            let banks_bytes =
                72 * 7680 * 2048 * 2 + CONV_BYTES + output_bytes + qkv_bytes + tied_bytes;
            report["convin_placement"] = serde_json::json!({
                "mode": "metal",
                "shared_bank_count": 72 + super::CONV_LAYERS.len() + super::LAYERS + qkv_count + tied_count,
                "shared_bank_payload_bytes": banks_bytes,
                "compact_cpu_weight_payload_bytes": setup.weight_bytes-banks_bytes,
                "convin_bank_payload_bytes": CONV_BYTES,
                "workspace_bytes": crate::metal::SharedMlpWorkspace::CONV_BYTES,
                "submitted": submitted, "completed": completed, "published": setup.output_published[0],
                "completed_gpu_seconds": clocks[0], "commit_wait_seconds": clocks[1],
            });
        }
        if let Some(mut output) = setup.workspace.output_stats(setup.output_published) {
            const OUTPUT_BYTES: usize = super::LAYERS * 2 * super::HIDDEN * super::HIDDEN;
            output["mode"] = serde_json::json!("metal");
            output["shared_bank_count"] = serde_json::json!(
                72 + super::CONV_LAYERS.len() + super::LAYERS + qkv_count + tied_count
            );
            output["output_bank_payload_bytes"] = serde_json::json!(OUTPUT_BYTES);
            output["compact_cpu_payload_bytes"] = serde_json::json!(
                setup.weight_bytes
                    - 72 * 7680 * 2048 * 2
                    - super::CONV_LAYERS.len() * super::HIDDEN * 3 * super::HIDDEN * 2
                    - OUTPUT_BYTES
                    - qkv_bytes
                    - tied_bytes
            );
            output["cold_tag_seconds"] = serde_json::json!(setup.output_tag_seconds);
            output["admitted_workspace_bytes"] = serde_json::json!(setup.output_budget);
            report["operator_output_placement"] = output;
        }
        if let Some(tile) = setup.workspace.tile_stats() {
            report["gate_up_tile"] = tile;
        }
        if let Some(mut tile) = setup.workspace.projection_tile_stats() {
            tile["tag_bytes"] =
                serde_json::json!(crate::metal::SharedMlpWorkspace::PROJECTION_TILE_BYTES);
            tile["admitted_bytes"] = serde_json::json!(setup.projection_tile_budget);
            tile["cold_tag_seconds"] = serde_json::json!(setup.projection_tile_tag_seconds);
            report["convin_down_tile"] = tile;
        }
        report
    }
}

fn operator_output(operator: crate::weights::lfm3::Operator) -> crate::weights::lfm3::Tensor {
    use crate::weights::lfm3::Operator;
    match operator {
        Operator::Convolution { output, .. } | Operator::Attention { output, .. } => output,
    }
}
