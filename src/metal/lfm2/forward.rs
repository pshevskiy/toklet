//! One retained Phase-A command; every fallible encode precedes publication.
use super::*;
impl Session<'_> {
    pub(super) fn encode(&self, frames: &[cache::Frame], n: usize) -> Result<()> {
        let g = self.model.geometry;
        if frames.len() != g.attention {
            return Err(Error::Options("LFM2 exact KV frame count"));
        }
        let m = self.model;
        let s = &self.scratch;
        for frame in frames {
            self.cache.view(frame)?;
        }
        let view = self.cache.view(&frames[0])?;
        let command = &view.command;
        let status = view.status;
        // Stamps precede all producers on this same command; even an omitted
        // or partially written history is rejected by the final pair validator.
        for h in &self.histories {
            if h.candidate.len != 2 * g.h + GUARD {
                return Err(Error::Options("LFM2 candidate extent"));
            }
        }
        let fill = command
            .blitCommandEncoder()
            .ok_or_else(|| Error::Native("LFM2 candidate fill encoder absent".into()))?;
        for h in &self.histories {
            fill.fillBuffer_range_value(
                &h.candidate.raw,
                NSRange {
                    location: 0,
                    length: 2 * g.h * 2,
                },
                0x7e,
            );
        }
        fill.endEncoding();
        mlx::gather(command, &m.embedding, &s.tokens, &s.hidden, g.h, n, status)?;
        self.capture(command, 0, &s.hidden.raw)?;
        let rms = |input: &Buffer<u16>, scales: &Buffer<u32>, output: &Buffer<u16>, rows, width| {
            m.rms.encode(
                command,
                rms::Plan::new(rows, width, width, width, EPSILON)?,
                rms::Invocation {
                    input,
                    scales,
                    output,
                    status,
                },
            )
        };
        let affine = |weight: &Tensor, input: &Retained<MPSMatrix>, output: &Buffer<u16>, rows| {
            mlx::product(
                command,
                weight,
                &m.zero_bias,
                input,
                &output.raw,
                weight.layout.rows,
                rows,
                true,
                status,
            )
        };
        let half_product = |a: &Buffer<u16>,
                            a_offset,
                            a_stride,
                            b: &Buffer<u16>,
                            b_offset,
                            b_stride,
                            out: &Buffer<u16>| {
            m.product.encode(
                command,
                product::Plan::new(
                    n,
                    g.h,
                    product::View {
                        offset: a_offset,
                        stride: a_stride,
                    },
                    product::View {
                        offset: b_offset,
                        stride: b_stride,
                    },
                    product::View {
                        offset: 0,
                        stride: g.h,
                    },
                )?,
                product::Invocation {
                    a,
                    b,
                    output: out,
                    status,
                },
            )
        };
        for layer in &m.layers {
            let base = layer.stage_base;
            rms(&s.hidden, &layer.operator_norm, &s.norm, n, g.h)?;
            self.capture(command, base, &s.norm.raw)?;
            let ffn_base = match layer.kind {
                Kind::Convolution(owner) => {
                    let c = &m.convolution[owner];
                    let history = &self.histories[owner];
                    affine(&c.bcx, &self.views.norm[n - 1], &s.bcx, n)?;
                    self.capture(command, base + 1, &s.bcx.raw)?;
                    for (slot, offset) in [(2, 0), (3, g.h), (4, 2 * g.h)] {
                        self.capture_view(command, base + slot, &s.bcx.raw, offset, 3 * g.h)?;
                    }
                    half_product(&s.bcx, 0, 3 * g.h, &s.bcx, 2 * g.h, 3 * g.h, &s.u)?;
                    self.capture(command, base + 5, &s.u.raw)?;
                    #[cfg(test)]
                    let omit = self.test_fault == TestFault::OmitHistory(owner);
                    #[cfg(not(test))]
                    let omit = false;
                    if !omit {
                        m.convolution_op.encode(
                            command,
                            conv::Plan::new(n, g.h, g.h, g.h)?,
                            conv::Invocation {
                                input: &s.u,
                                scales: &c.coefficients,
                                output: &s.z,
                                old: &history.old,
                                candidate: &history.candidate,
                                status,
                            },
                        )?;
                    }
                    #[cfg(test)]
                    if self.test_fault == TestFault::PartialHistory(owner) {
                        // A terminal validator must reject the final untouched word.
                        inject_bytes(command, &history.candidate.raw, (2 * g.h - 1) * 2, 2, 0x7e)?;
                    }
                    self.capture(command, base + 6, &s.z.raw)?;
                    // u has already been consumed and its raw suffix copied into
                    // the candidate. Reuse its owner for C*z, disjoint from C/z.
                    half_product(&s.bcx, g.h, 3 * g.h, &s.z, 0, g.h, &s.u)?;
                    self.capture(command, base + 7, &s.u.raw)?;
                    affine(&c.output, &self.views.u[n - 1], &s.branch, n)?;
                    self.capture(command, base + 8, &s.branch.raw)?;
                    m.gpu.encode(
                        command,
                        Kernel::Residual,
                        &[&s.hidden.raw, &s.branch.raw],
                        &[(n * g.h) as u32],
                        n * g.h,
                        Some(status),
                    )?;
                    self.capture(command, base + 9, &s.hidden.raw)?;
                    base + 10
                }
                Kind::Attention(owner) => {
                    let view = self.cache.view(&frames[owner])?;
                    let a = &m.attention[owner];
                    for (slot, w, out) in [(1, &a.q, &s.q), (2, &a.k, &s.k), (3, &a.v, &s.v)] {
                        affine(w, &self.views.norm[n - 1], out, n)?;
                        self.capture(command, base + slot, &out.raw)?;
                    }
                    rms(&s.q, &a.q_norm, &s.work, n * g.query_heads, D)?;
                    self.capture(command, base + 4, &s.work.raw)?;
                    rms(&s.k, &a.k_norm, &s.k_norm, n * g.kv_heads, D)?;
                    self.capture(command, base + 5, &s.k_norm.raw)?;
                    for (heads, input, out, slot) in [
                        (g.query_heads as u32, &s.work, &s.q, 6),
                        (g.kv_heads as u32, &s.k_norm, &s.k, 7),
                    ] {
                        m.gpu.encode(
                            command,
                            Kernel::Rotary,
                            &[&input.raw, &out.raw],
                            &[
                                n as u32,
                                heads,
                                D as u32,
                                view.offset as u32,
                                THETA.to_bits(),
                            ],
                            n * heads as usize * D / 2,
                            Some(status),
                        )?;
                        self.capture(command, base + slot, &out.raw)?;
                    }
                    // Token-major producers to fixed32 per-head staging, no KV
                    // expansion and no whole committed-prefix concatenation.
                    m.gpu.encode(
                        command,
                        Kernel::Cache,
                        &[&s.k.raw, &s.v.raw, &view.new_k.raw, &view.new_v.raw],
                        &[n as u32, g.kv_heads as u32, D as u32, B as u32, 0],
                        n * g.k,
                        None,
                    )?;
                    m.attention_op.encode(
                        command,
                        attention::Plan::new_gqa(
                            g.capacity,
                            g.query_heads,
                            g.kv_heads,
                            None,
                            view.offset,
                            n,
                        )?,
                        attention::Invocation {
                            q: &s.q,
                            old_k: view.old_k,
                            old_v: view.old_v,
                            new_k: view.new_k,
                            new_v: view.new_v,
                            scores: &s.scores,
                            probabilities: &s.probabilities,
                            context: &s.work,
                            status,
                        },
                    )?;
                    self.capture(command, base + 8, &s.scores.raw)?;
                    self.capture(command, base + 9, &s.probabilities.raw)?;
                    self.capture(command, base + 10, &s.work.raw)?;
                    affine(&a.o, &self.views.context[n - 1], &s.branch, n)?;
                    self.capture(command, base + 11, &s.branch.raw)?;
                    m.gpu.encode(
                        command,
                        Kernel::Residual,
                        &[&s.hidden.raw, &s.branch.raw],
                        &[(n * g.h) as u32],
                        n * g.h,
                        Some(status),
                    )?;
                    self.capture(command, base + 12, &s.hidden.raw)?;
                    base + 13
                }
            };
            rms(&s.hidden, &layer.ffn_norm, &s.norm, n, g.h)?;
            self.capture(command, ffn_base, &s.norm.raw)?;
            for (slot, w, out) in [(1, &layer.gate, &s.gate), (2, &layer.up, &s.up)] {
                affine(w, &self.views.norm[n - 1], out, n)?;
                self.capture(command, ffn_base + slot, &out.raw)?;
            }
            m.gpu.encode(
                command,
                Kernel::Silu,
                &[&s.gate.raw, &s.activated.raw],
                &[(n * g.i) as u32],
                n * g.i,
                Some(status),
            )?;
            self.capture(command, ffn_base + 3, &s.activated.raw)?;
            m.gpu.encode(
                command,
                Kernel::Gating,
                &[&s.activated.raw, &s.up.raw, &s.gate.raw],
                &[(n * g.i) as u32],
                n * g.i,
                Some(status),
            )?;
            self.capture(command, ffn_base + 4, &s.gate.raw)?;
            affine(&layer.down, &self.views.gate[n - 1], &s.branch, n)?;
            self.capture(command, ffn_base + 5, &s.branch.raw)?;
            m.gpu.encode(
                command,
                Kernel::Residual,
                &[&s.hidden.raw, &s.branch.raw],
                &[(n * g.h) as u32],
                n * g.h,
                Some(status),
            )?;
            #[cfg(test)]
            if let TestFault::AfterLayer(index) = self.test_fault
                && m.layers
                    .get(index)
                    .is_some_and(|target| std::ptr::eq(layer, target))
            {
                inject_bytes(command, &s.hidden.raw, 0, 2, 0x7e)?;
            }
            self.capture(command, ffn_base + 6, &s.hidden.raw)?;
        }
        rms(&s.hidden, &m.final_norm, &s.norm, n, g.h)?;
        self.capture(command, g.stages() - 2, &s.norm.raw)?;
        let (input, rows) = if self.options.all_logits {
            (&self.views.norm[n - 1], n)
        } else {
            (&self.views.last[n - 1], 1)
        };
        mlx::product(
            command,
            &m.embedding,
            &m.zero_bias,
            input,
            &self.candidate.raw,
            g.v,
            rows,
            false,
            status,
        )?;
        #[cfg(test)]
        if self.test_fault == TestFault::Head {
            inject_bytes(command, &self.candidate.raw, 0, 4, 0xff)?;
        }
        m.gpu.encode(
            command,
            Kernel::Validate,
            &[&self.candidate.raw],
            &[rows as u32, g.v as u32, g.v as u32, 0],
            rows * g.v,
            Some(status),
        )?;
        self.capture(command, g.stages() - 1, &self.candidate.raw)?;
        // Every history candidate is unconditionally checked on the same
        // command/status before finish_math can mint a complete Ready.
        for pair in self.histories.chunks_exact(2) {
            m.cache.encode_half_pair(
                command,
                &pair[0].candidate,
                &pair[1].candidate,
                0,
                2 * g.h,
                status,
            )?;
        }
        Ok(())
    }
}
#[cfg(test)]
fn inject_bytes(
    command: &Command<'_>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    offset: usize,
    len: usize,
    value: u8,
) -> Result<()> {
    if offset.checked_add(len).is_none_or(|n| n > buffer.length()) {
        return Err(Error::Options("LFM2 test injection extent"));
    }
    let e = command
        .blitCommandEncoder()
        .ok_or_else(|| Error::Native("LFM2 test injection encoder absent".into()))?;
    e.fillBuffer_range_value(
        buffer,
        NSRange {
            location: offset,
            length: len,
        },
        value,
    );
    e.endEncoding();
    Ok(())
}
