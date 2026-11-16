//! Whole-chain replay through unchanged CPU operators and synchronous BF16
//! tile helper. No expanded weight model and no reuse of speculative descendants.
use super::*;
pub(super) struct Runner {
    pub states: [[Vec<f32>; 32]; 3],
    old_k: [Vec<f32>; 2],
    old_v: [Vec<f32>; 2],
    input: Buffer<u32>,
    output: Buffer<u32>,
    status: Buffer<u32>,
    tile_output: Vec<f32>,
    pending: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    pub matrix_operations: usize,
    pub commands: usize,
    pub waits: usize,
    pub input_bytes: usize,
    pub readback_bytes: usize,
    pub checkpoint_bytes: usize,
    pub counts_known: bool,
}
pub(super) fn zero(n: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(n)
        .map_err(|_| Error::Native("chain reference reservation failed".into()))?;
    values.resize(n, 0.);
    Ok(values)
}
fn checked<T>(result: std::result::Result<T, cpu::Error>) -> Result<T> {
    result.map_err(|e| Error::Native(e.to_string()))
}
impl Runner {
    pub fn planned_bytes(graph: &ResidentGraph<'_>) -> Result<usize> {
        let states = graph
            .plans
            .iter()
            .flat_map(|p| p.layout.regions.iter())
            .map(|r| r.len)
            .sum::<usize>();
        let old = 2 * (512 * 256 + graph.options.capacity * 512);
        let rows = graph.options.max_rows;
        let words = states
            .checked_add(old)
            .and_then(|n| n.checked_add(rows * (12288 + 8192 + 8192) + 3 * GUARD + 1))
            .ok_or(Error::Options("chain reference storage overflow"))?;
        words
            .checked_mul(4)
            .ok_or(Error::Options("chain reference byte overflow"))
    }
    pub fn new(graph: &ResidentGraph<'_>) -> Result<Self> {
        let mut states = std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
        for (node, states) in states.iter_mut().enumerate() {
            for state in STATES {
                states[state as usize] = zero(graph.plans[node].layout.region(state).len)?;
            }
        }
        let rows = graph.options.max_rows;
        let gpu = &graph.context.0.gpu;
        Ok(Self {
            states,
            old_k: [zero(512 * 256)?, zero(graph.options.capacity * 512)?],
            old_v: [zero(512 * 256)?, zero(graph.options.capacity * 512)?],
            input: Buffer::new(gpu, rows * 12288 + GUARD)?,
            output: Buffer::new(gpu, rows * 8192 + GUARD)?,
            status: Buffer::new(gpu, 1 + GUARD)?,
            tile_output: zero(rows * 8192)?,
            pending: None,
            matrix_operations: 0,
            commands: 0,
            waits: 0,
            input_bytes: 0,
            readback_bytes: 0,
            checkpoint_bytes: 0,
            counts_known: true,
        })
    }
    pub fn allocation_bytes(&self) -> usize {
        (self
            .states
            .iter()
            .flat_map(|s| s.iter())
            .map(Vec::capacity)
            .sum::<usize>()
            + self
                .old_k
                .iter()
                .chain(&self.old_v)
                .map(Vec::capacity)
                .sum::<usize>()
            + self.input.len
            + self.output.len
            + self.status.len
            + self.tile_output.capacity())
            * 4
    }
    fn project(
        &mut self,
        graph: &ResidentGraph<'_>,
        node: usize,
        role: usize,
        from: State,
        to: State,
        rows: usize,
    ) -> Result<()> {
        let slot = graph.plans[node].matrix_slots[role]
            .ok_or(Error::Options("chain reference inactive matrix"))?;
        let m = graph.source.matrices[slot]
            .as_ref()
            .ok_or(Error::Options("chain reference matrix absent"))?;
        let [input, output] = self.states[node]
            .get_disjoint_mut([from as usize, to as usize])
            .map_err(|_| Error::Options("chain reference projection aliases"))?;
        let input = &input[..rows * m.columns];
        let output = &mut output[..rows * m.rows];
        admit(m.columns, m.maximum, input)?;
        self.matrix_operations += 1;
        for (index, tile) in m.tiles.iter().enumerate() {
            let first = index * 8192;
            let count = (m.rows - first).min(8192);
            let result = project::run(
                &graph.context.0,
                project::Matrix::bf16(count, m.columns, tile),
                project::Buffers {
                    input: &mut self.input,
                    output: &mut self.output,
                    status: &mut self.status,
                    pending: &mut self.pending,
                },
                input,
                &mut self.tile_output[..rows * count],
                rows,
            );
            if result.is_err() {
                self.counts_known = false;
            }
            result?;
            // Count actual successful helper calls, not one assumed command
            // per matrix: consumer gate/up each call twice. On helper failure
            // exact native/copy counts are explicitly unavailable, never guessed.
            self.commands += 1;
            self.waits += 1;
            self.input_bytes += input.len() * 4;
            self.readback_bytes += rows * count * 4;
            for token in 0..rows {
                output[token * m.rows + first..token * m.rows + first + count]
                    .copy_from_slice(&self.tile_output[token * count..(token + 1) * count]);
            }
        }
        Ok(())
    }
    fn norm(
        &mut self,
        graph: &ResidentGraph<'_>,
        node: usize,
        norm: Option<usize>,
        states: [State; 2],
        rows: usize,
        width: usize,
    ) -> Result<()> {
        let unit = [1_f32; 512];
        let weights = match norm {
            Some(n) => {
                let slot = graph.plans[node].norm_slots[n]
                    .ok_or(Error::Options("chain reference normalization absent"))?;
                &graph.source.vectors[slot][..]
            }
            None => &unit[..width],
        };
        let [input, output] = self.states[node]
            .get_disjoint_mut(states.map(|s| s as usize))
            .map_err(|_| Error::Options("chain reference norm aliases"))?;
        checked(cpu::rmsnorm(
            &input[..rows * width],
            weights,
            &mut output[..rows * width],
            rows,
            width,
            1e-6,
        ))
    }
    fn add(&mut self, node: usize, states: [State; 3], n: usize) -> Result<()> {
        let [a, b, out] = self.states[node]
            .get_disjoint_mut(states.map(|s| s as usize))
            .map_err(|_| Error::Options("chain reference residual aliases"))?;
        out[..n].copy_from_slice(&a[..n]);
        checked(cpu::residual_add(&mut out[..n], &b[..n]))
    }
    fn gelu(&mut self, node: usize, states: [State; 4], n: usize) -> Result<()> {
        let [gate, up, activation, product] = self.states[node]
            .get_disjoint_mut(states.map(|s| s as usize))
            .map_err(|_| Error::Options("chain reference GELU aliases"))?;
        checked(cpu::gelu_tanh(&gate[..n], &mut activation[..n]))?;
        checked(cpu::gated_gelu_tanh(
            &gate[..n],
            &up[..n],
            &mut product[..n],
        ))
    }
    fn rope(
        &mut self,
        graph: &ResidentGraph<'_>,
        node: usize,
        frame: NodeFrame,
        states: [State; 2],
        heads: usize,
    ) -> Result<()> {
        let plan = &graph.plans[node];
        let d = plan.layout.d;
        let [input, output] = self.states[node]
            .get_disjoint_mut(states.map(|s| s as usize))
            .map_err(|_| Error::Options("chain reference rotary aliases"))?;
        checked(cpu::proportional_rotary(
            &input[..frame.rows * heads * d],
            &mut output[..frame.rows * heads * d],
            &cpu::ProportionalRotaryShape {
                tokens: frame.rows,
                heads,
                head_dim: d,
                rotated_dims: plan.slots.geometry.rotated_dims,
                offset: frame.query_start,
                base: if d == 512 { 1000000. } else { 10000. },
                factor: 1.,
            },
        ))
    }
    fn attention(
        &mut self,
        graph: &ResidentGraph<'_>,
        node: usize,
        frame: NodeFrame,
        caches: &[CacheStorage; 2],
    ) -> Result<()> {
        use State::*;
        let plan = &graph.plans[node];
        let d = plan.layout.d;
        let t = frame.rows;
        let cache = match plan.kind {
            LayerKind::Producer { cache } | LayerKind::Consumer { donor: cache } => cache,
        };
        let old = window_rows::old_spans(
            caches[cache].capacity,
            frame.proposal_start,
            caches[cache].window,
        );
        let old_segment = |i: usize| {
            let span = old[i];
            cpu::AttentionKvSegment {
                k: &self.old_k[cache][span.slot * d..(span.slot + span.rows) * d],
                v: &self.old_v[cache][span.slot * d..(span.slot + span.rows) * d],
                keys: span.rows,
                head_stride: span.rows * d,
                key_start: span.position,
            }
        };
        let shape = cpu::SpanAttentionShape {
            queries: t,
            keys: frame.keys,
            query_heads: 8,
            kv_heads: 1,
            head_dim: d,
            query_start: frame.query_start,
            key_start: frame.key_start,
            window: caches[cache].window,
            scale: 1.,
        };
        let [old0, old1] = [old_segment(0), old_segment(1)];
        match plan.kind {
            LayerKind::Producer { .. } => {
                let [q, k, v, out, scores, prob] = self.states[node]
                    .get_disjoint_mut(
                        [
                            QRotary,
                            KProposal,
                            VProposal,
                            Context,
                            Scores,
                            Probabilities,
                        ]
                        .map(|s| s as usize),
                    )
                    .unwrap();
                checked(cpu::span_attention(
                    &q[..t * 8 * d],
                    &[
                        old0,
                        old1,
                        cpu::AttentionKvSegment {
                            k: &k[..frame.proposal_rows * d],
                            v: &v[..frame.proposal_rows * d],
                            keys: frame.proposal_rows,
                            head_stride: frame.proposal_rows * d,
                            key_start: frame.proposal_start,
                        },
                    ],
                    &mut out[..t * 8 * d],
                    &mut scores[..8 * t * frame.keys],
                    &mut prob[..8 * t * frame.keys],
                    &shape,
                ))
            }
            LayerKind::Consumer { donor } => {
                let [donor, current] = self.states.get_disjoint_mut([donor, node]).unwrap();
                let k = &donor[KProposal as usize][..frame.proposal_rows * d];
                let v = &donor[VProposal as usize][..frame.proposal_rows * d];
                let [q, out, scores, prob] = current
                    .get_disjoint_mut([QRotary, Context, Scores, Probabilities].map(|s| s as usize))
                    .unwrap();
                checked(cpu::span_attention(
                    &q[..t * 8 * d],
                    &[
                        old0,
                        old1,
                        cpu::AttentionKvSegment {
                            k,
                            v,
                            keys: frame.proposal_rows,
                            head_stride: frame.proposal_rows * d,
                            key_start: frame.proposal_start,
                        },
                    ],
                    &mut out[..t * 8 * d],
                    &mut scores[..8 * t * frame.keys],
                    &mut prob[..8 * t * frame.keys],
                    &shape,
                ))
            }
        }
    }
    pub fn run(
        &mut self,
        graph: &ResidentGraph<'_>,
        entry: &Buffer<u32>,
        entry_regions: &[Region; 4],
        caches: &[CacheStorage; 2],
        frame: &GraphFrame,
    ) -> Result<()> {
        use State::*;
        self.matrix_operations = 0;
        self.commands = 0;
        self.waits = 0;
        self.input_bytes = 0;
        self.readback_bytes = 0;
        self.checkpoint_bytes = 0;
        self.counts_known = true;
        // The ONLY hidden replay entry is the immutable chain entry. PLE rows
        // are independent inputs; every later hidden is recomputed topologically.
        entry.with_slice(|s| {
            let r = entry_regions[0];
            for (dst, &word) in self.states[0][Hidden as usize][..frame.rows * H]
                .iter_mut()
                .zip(&s[r.start..r.start + frame.rows * H])
            {
                *dst = f32::from_bits(word);
            }
            self.checkpoint_bytes += frame.rows * H * 4;
            for node in 0..3 {
                let f = frame.nodes[node];
                let dropped = if node == 2 && frame.mode == OutputRows::Last {
                    frame.rows - 1
                } else {
                    0
                };
                let r = entry_regions[node + 1];
                for (dst, &word) in self.states[node][PleInput as usize][..f.rows * P]
                    .iter_mut()
                    .zip(&s[r.start + dropped * P..r.start + (dropped + f.rows) * P])
                {
                    *dst = f32::from_bits(word);
                }
                self.checkpoint_bytes += f.rows * P * 4;
            }
        });
        for (i, cache) in caches.iter().enumerate() {
            let spans = window_rows::old_spans(cache.capacity, frame.offset, cache.window);
            let d = cache.d;
            let copy = |buffer: &Buffer<u32>, out: &mut [f32]| {
                let mut bytes = 0;
                buffer.with_slice(|s| {
                    for span in spans {
                        let first = span.slot * d;
                        let end = first + span.rows * d;
                        for (dst, &word) in out[first..end].iter_mut().zip(&s[first..end]) {
                            *dst = f32::from_bits(word);
                        }
                        bytes += span.rows * d * 4;
                    }
                });
                bytes
            };
            self.checkpoint_bytes += copy(&cache.k, &mut self.old_k[i]);
            self.checkpoint_bytes += copy(&cache.v, &mut self.old_v[i]);
        }
        for node in 0..3 {
            let p = &graph.plans[node];
            let f = frame.nodes[node];
            let t = f.rows;
            let d = p.layout.d;
            let h = t * H;
            if node != 0 {
                let dropped = if node == 2 && frame.mode == OutputRows::Last {
                    frame.rows - 1
                } else {
                    0
                };
                let [previous, current] = self.states.get_disjoint_mut([node - 1, node]).unwrap();
                current[Hidden as usize][..h]
                    .copy_from_slice(&previous[Output as usize][dropped * H..dropped * H + h]);
            }
            self.norm(graph, node, Some(0), [Hidden, InputNorm], t, H)?;
            self.project(graph, node, 0, InputNorm, QRaw, t)?;
            self.norm(graph, node, Some(1), [QRaw, QNorm], t * 8, d)?;
            self.rope(graph, node, f, [QNorm, QRotary], 8)?;
            if p.layout.producer {
                self.project(graph, node, 1, InputNorm, KRaw, t)?;
                self.norm(graph, node, Some(2), [KRaw, KNorm], t, d)?;
                self.rope(graph, node, f, [KNorm, KProposal], 1)?;
                self.project(graph, node, 2, InputNorm, VRaw, t)?;
                self.norm(graph, node, None, [VRaw, VProposal], t, d)?;
            }
            self.attention(graph, node, f, caches)?;
            self.project(graph, node, 3, Context, AttentionRaw, t)?;
            self.norm(graph, node, Some(3), [AttentionRaw, AttentionNorm], t, H)?;
            self.add(node, [AttentionNorm, Hidden, AttentionResidual], h)?;
            self.norm(graph, node, Some(4), [AttentionResidual, PreFf], t, H)?;
            self.project(graph, node, 4, PreFf, FfGate, t)?;
            self.project(graph, node, 5, PreFf, FfUp, t)?;
            self.gelu(
                node,
                [FfGate, FfUp, FfActivation, FfProduct],
                t * p.layout.ff,
            )?;
            self.project(graph, node, 6, FfProduct, FfDown, t)?;
            self.norm(graph, node, Some(5), [FfDown, FfNorm], t, H)?;
            self.add(node, [FfNorm, AttentionResidual, FfResidual], h)?;
            self.project(graph, node, 7, FfResidual, PleGate, t)?;
            self.gelu(node, [PleGate, PleInput, PleActivation, PleProduct], t * P)?;
            self.project(graph, node, 8, PleProduct, PleProjection, t)?;
            self.norm(graph, node, Some(6), [PleProjection, PleNorm], t, H)?;
            self.add(node, [FfResidual, PleNorm, PleResidual], h)?;
            let [input, out] = self.states[node]
                .get_disjoint_mut([PleResidual as usize, Output as usize])
                .unwrap();
            for (index, (dst, &value)) in out[..h].iter_mut().zip(&input[..h]).enumerate() {
                *dst = value * p.scalar;
                if !dst.is_finite() {
                    return checked(Err(cpu::Error::NonFinite {
                        tensor: "Gemma packed scalar output",
                        index,
                    }));
                }
            }
        }
        Ok(())
    }
    /// Pool replay validates all old logical states before discarding retired
    /// candidate-finite receipts. Arithmetic/order and the replay owner stay unchanged.
    pub fn validate_finite(&self, graph: &ResidentGraph<'_>, frame: &GraphFrame) -> Result<usize> {
        let mut valid = true;
        let mut bytes = 0;
        for (node, plan) in graph.plans.iter().enumerate() {
            for state in STATES {
                let n = plan.layout.active(state, frame.nodes[node]);
                let values = self.states[node][state as usize]
                    .get(..n)
                    .ok_or(Error::Options("chain reference active extent differs"))?;
                for value in values {
                    valid &= value.is_finite();
                }
                bytes += n * 4;
            }
        }
        if valid {
            Ok(bytes)
        } else {
            Err(Error::Nonfinite)
        }
    }
    pub fn upload(
        &self,
        graph: &ResidentGraph<'_>,
        storage: &mut [LayerStorage; 3],
        frame: &GraphFrame,
    ) -> usize {
        let mut bytes = 0;
        for (node, storage) in storage.iter_mut().enumerate() {
            storage.arena.with_mut(|s| {
                for state in STATES {
                    let r = graph.plans[node].layout.region(state);
                    let n = graph.plans[node].layout.active(state, frame.nodes[node]);
                    for (dst, value) in s[r.start..r.start + n]
                        .iter_mut()
                        .zip(&self.states[node][state as usize][..n])
                    {
                        *dst = value.to_bits();
                    }
                    bytes += n * 4;
                }
            });
        }
        bytes
    }
}
impl Drop for Runner {
    fn drop(&mut self) {
        if let Some(command) = self.pending.take() {
            match command.status() {
                MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                    command.commit();
                    command.waitUntilCompleted();
                }
                MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                    command.waitUntilCompleted()
                }
                _ => (),
            }
        }
    }
}
