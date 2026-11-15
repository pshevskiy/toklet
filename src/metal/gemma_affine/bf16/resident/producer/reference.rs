//! Whole-block fallback: original BF16 GPU projections and unchanged CPU recipe.
//! There is no all-CPU BF16 adapter and no second FP32 weight model here.
use super::*;
pub(super) struct Input<'a> {
    pub source: &'a Bf16Weights,
    pub context: &'a Bf16Context,
    pub slots: ProducerSlots,
    pub layer: usize,
    pub arena: &'a Buffer<u32>,
    pub layout: &'a Layout,
    pub cache: &'a ProducerKv,
    pub frame: &'a Frame,
}
pub(super) struct Runner {
    pub states: [Vec<f32>; 32],
    old_k: Vec<f32>,
    old_v: Vec<f32>,
    input: Buffer<u32>,
    output: Buffer<u32>,
    status: Buffer<u32>,
    pending: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    pub commands: usize,
    pub input_bytes: usize,
    pub readback_bytes: usize,
}
fn zero(n: usize) -> Result<Vec<f32>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| Error::Native("producer reference scratch reservation failed".into()))?;
    v.resize(n, 0.);
    Ok(v)
}
fn checked<T>(r: std::result::Result<T, cpu::Error>) -> Result<T> {
    r.map_err(|e| Error::Native(e.to_string()))
}
impl Runner {
    pub fn new(l: &Layout, g: &Gpu) -> Result<Self> {
        let mut states = std::array::from_fn(|_| Vec::new());
        for (v, r) in states.iter_mut().zip(l.regions) {
            *v = zero(r.len)?;
        }
        Ok(Self {
            states,
            old_k: zero(4 * l.d)?,
            old_v: zero(4 * l.d)?,
            input: Buffer::new(g, 4 * F + GUARD)?,
            output: Buffer::new(g, 4 * F + GUARD)?,
            status: Buffer::new(g, 1 + GUARD)?,
            pending: None,
            commands: 0,
            input_bytes: 0,
            readback_bytes: 0,
        })
    }
    pub fn allocation_bytes(&self) -> usize {
        (self.states.iter().map(Vec::capacity).sum::<usize>()
            + self.old_k.capacity()
            + self.old_v.capacity()
            + self.input.len
            + self.output.len
            + self.status.len)
            * 4
    }
    fn project(
        &mut self,
        source: &Bf16Weights,
        context: &Bf16Context,
        slot: usize,
        from: State,
        to: State,
        t: usize,
    ) -> Result<()> {
        let m = source.matrices[slot]
            .as_ref()
            .ok_or(Error::Options("producer reference matrix absent"))?;
        if m.tiles.len() != 1 {
            return Err(Error::Options(
                "producer reference requires one original tile",
            ));
        }
        let [input, output] = self
            .states
            .get_disjoint_mut([from as usize, to as usize])
            .map_err(|_| Error::Options("producer reference projection aliases"))?;
        let input = &input[..t * m.columns];
        let output = &mut output[..t * m.rows];
        admit(m.columns, m.maximum, input)?;
        project::run(
            &context.0,
            project::Matrix::bf16(m.rows, m.columns, &m.tiles[0]),
            project::Buffers {
                input: &mut self.input,
                output: &mut self.output,
                status: &mut self.status,
                pending: &mut self.pending,
            },
            input,
            output,
            t,
        )?;
        self.commands += 1;
        self.input_bytes += input.len() * 4;
        self.readback_bytes += output.len() * 4;
        Ok(())
    }
    fn norm(
        &mut self,
        source: &Bf16Weights,
        slot: Option<usize>,
        states: [State; 2],
        rows: usize,
        width: usize,
    ) -> Result<()> {
        let [a, b] = self
            .states
            .get_disjoint_mut(states.map(|s| s as usize))
            .map_err(|_| Error::Options("producer reference norm aliases"))?;
        let unit = [1_f32; 512];
        let weights = match slot {
            Some(i) => source.vectors[i].as_slice(),
            None => &unit[..width],
        };
        checked(cpu::rmsnorm(
            &a[..rows * width],
            weights,
            &mut b[..rows * width],
            rows,
            width,
            1e-6,
        ))
    }
    fn add(&mut self, states: [State; 3], n: usize) -> Result<()> {
        let [left, right, out] = self
            .states
            .get_disjoint_mut(states.map(|s| s as usize))
            .map_err(|_| Error::Options("producer reference add aliases"))?;
        out[..n].copy_from_slice(&left[..n]);
        checked(cpu::residual_add(&mut out[..n], &right[..n]))
    }
    fn gelu(&mut self, states: [State; 4], n: usize) -> Result<()> {
        let [gate, up, activation, product] = self
            .states
            .get_disjoint_mut(states.map(|s| s as usize))
            .map_err(|_| Error::Options("producer reference GELU aliases"))?;
        checked(cpu::gelu_tanh(&gate[..n], &mut activation[..n]))?;
        checked(cpu::gated_gelu_tanh(
            &gate[..n],
            &up[..n],
            &mut product[..n],
        ))
    }
    /// References every original CPU stage in queries/producer/finish order.
    /// The retained checkpoint and cache prefix are unchanged until publication.
    pub fn run(&mut self, input: Input<'_>) -> Result<()> {
        let Input {
            source,
            context,
            slots,
            layer,
            arena,
            layout,
            cache,
            frame: f,
        } = input;
        use State::*;
        let t = f.rows;
        let d = layout.d;
        let h = t * H;
        self.commands = 0;
        self.input_bytes = 0;
        self.readback_bytes = 0;
        arena.with_slice(|words| {
            for s in [Hidden, PleInput] {
                let r = layout.region(s);
                let n = layout.active(s, f);
                for (out, &bits) in self.states[s as usize][..n]
                    .iter_mut()
                    .zip(&words[r.start..r.start + n])
                {
                    *out = f32::from_bits(bits);
                }
            }
        });
        cache.k.with_slice(|s| {
            for (out, &bits) in self.old_k[..f.offset * d].iter_mut().zip(s) {
                *out = f32::from_bits(bits);
            }
        });
        cache.v.with_slice(|s| {
            for (out, &bits) in self.old_v[..f.offset * d].iter_mut().zip(s) {
                *out = f32::from_bits(bits);
            }
        });
        let rope = |heads| cpu::ProportionalRotaryShape {
            tokens: t,
            heads,
            head_dim: d,
            rotated_dims: slots.geometry.rotated_dims,
            offset: f.offset,
            base: if layer == 0 { 10000. } else { 1000000. },
            factor: 1.,
        };
        self.norm(source, Some(slots.norms[0]), [Hidden, InputNorm], t, H)?;
        self.project(source, context, slots.matrices[0], InputNorm, QRaw, t)?;
        self.norm(source, Some(slots.norms[1]), [QRaw, QNorm], t * 8, d)?;
        let [q, qr] = self
            .states
            .get_disjoint_mut([QNorm as usize, QRotary as usize])
            .unwrap();
        checked(cpu::proportional_rotary(
            &q[..t * 8 * d],
            &mut qr[..t * 8 * d],
            &rope(8),
        ))?;
        self.project(source, context, slots.matrices[1], InputNorm, KRaw, t)?;
        self.norm(source, Some(slots.norms[2]), [KRaw, KNorm], t, d)?;
        let [k, kr] = self
            .states
            .get_disjoint_mut([KNorm as usize, KProposal as usize])
            .unwrap();
        checked(cpu::proportional_rotary(
            &k[..t * d],
            &mut kr[..t * d],
            &rope(1),
        ))?;
        self.project(source, context, slots.matrices[2], InputNorm, VRaw, t)?;
        self.norm(source, None, [VRaw, VProposal], t, d)?;
        let [q, k, v, out, scores, prob] = self
            .states
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
        let empty = cpu::AttentionKvSegment {
            k: &[],
            v: &[],
            keys: 0,
            head_stride: 0,
            key_start: 0,
        };
        checked(cpu::span_attention(
            &q[..t * 8 * d],
            &[
                cpu::AttentionKvSegment {
                    k: &self.old_k[..f.offset * d],
                    v: &self.old_v[..f.offset * d],
                    keys: f.offset,
                    head_stride: f.offset * d,
                    key_start: 0,
                },
                empty,
                cpu::AttentionKvSegment {
                    k: &k[..t * d],
                    v: &v[..t * d],
                    keys: t,
                    head_stride: t * d,
                    key_start: f.offset,
                },
            ],
            &mut out[..t * 8 * d],
            &mut scores[..8 * t * f.end],
            &mut prob[..8 * t * f.end],
            &cpu::SpanAttentionShape {
                queries: t,
                keys: f.end,
                query_heads: 8,
                kv_heads: 1,
                head_dim: d,
                query_start: f.offset,
                key_start: 0,
                window: cache.window,
                scale: 1.,
            },
        ))?;
        self.project(source, context, slots.matrices[3], Context, AttentionRaw, t)?;
        self.norm(
            source,
            Some(slots.norms[3]),
            [AttentionRaw, AttentionNorm],
            t,
            H,
        )?;
        self.add([AttentionNorm, Hidden, AttentionResidual], h)?;
        self.norm(
            source,
            Some(slots.norms[4]),
            [AttentionResidual, PreFf],
            t,
            H,
        )?;
        self.project(source, context, slots.matrices[4], PreFf, FfGate, t)?;
        self.project(source, context, slots.matrices[5], PreFf, FfUp, t)?;
        self.gelu([FfGate, FfUp, FfActivation, FfProduct], t * F)?;
        self.project(source, context, slots.matrices[6], FfProduct, FfDown, t)?;
        self.norm(source, Some(slots.norms[5]), [FfDown, FfNorm], t, H)?;
        self.add([FfNorm, AttentionResidual, FfResidual], h)?;
        self.project(source, context, slots.matrices[7], FfResidual, PleGate, t)?;
        self.gelu([PleGate, PleInput, PleActivation, PleProduct], t * P)?;
        self.project(
            source,
            context,
            slots.matrices[8],
            PleProduct,
            PleProjection,
            t,
        )?;
        self.norm(source, Some(slots.norms[6]), [PleProjection, PleNorm], t, H)?;
        self.add([FfResidual, PleNorm, PleResidual], h)?;
        let [input, out] = self
            .states
            .get_disjoint_mut([PleResidual as usize, Output as usize])
            .unwrap();
        let scalar = source.vectors[slots.scalar][0];
        for (index, (out, &value)) in out[..h].iter_mut().zip(&input[..h]).enumerate() {
            *out = value * scalar;
            if !out.is_finite() {
                return checked(Err(cpu::Error::NonFinite {
                    tensor: "Gemma packed scalar output",
                    index,
                }));
            }
        }
        Ok(())
    }
    pub fn upload(&self, layout: &Layout, f: &Frame, arena: &mut Buffer<u32>) {
        arena.with_mut(|words| {
            for s in STATES {
                let r = layout.region(s);
                let n = layout.active(s, f);
                for (out, value) in words[r.start..r.start + n]
                    .iter_mut()
                    .zip(&self.states[s as usize][..n])
                {
                    *out = value.to_bits();
                }
            }
        });
    }
}
