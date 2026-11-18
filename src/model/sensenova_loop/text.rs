//! Scalar SenseNova text reference over authenticated borrowed W4 weights.

use super::{
    PHYSICAL_LAYERS, Plan, ResidualStreams, TextShape,
    mhc::Router,
    rotary::TextRotary,
    state::{Frame, State, Token, VisitState},
};
use crate::{
    cpu,
    mlx::sensenova_loop::{
        TEXT_BYTES,
        text_weights::{PhysicalLayer, TextWeights},
    },
    model::{
        self,
        qwen35::{
            Config, Cpu, Error, Execution, Memory, MixingWeights, Observation, Options, Work,
            forward::{Trace, attention, delta, mlp_normalized},
            state::Scratch,
        },
    },
};
use half::bf16;
use std::{mem::size_of, sync::atomic::AtomicBool};

type Result<T> = std::result::Result<T, Error>;

/// Packed scalar matvec and Qwen FP32 inner branches, with BF16 text boundaries.
pub const PRECISION_PROFILE: &str = "sensenova-mlx-w4-scalar-fp32-branches-bf16-boundaries-v1";

fn invalid(reason: &'static str) -> Error {
    model::Error::InvalidConfig(reason).into()
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(cpu::Error::SizeOverflow.into())
}

fn mul(a: usize, b: usize) -> Result<usize> {
    Ok(cpu::product(a, b)?)
}

fn config(shape: TextShape) -> Config {
    Config {
        hidden: shape.hidden_size,
        intermediate: shape.intermediate_size,
        vocab: shape.vocab_size,
        layers: PHYSICAL_LAYERS,
        query_heads: shape.num_attention_heads,
        kv_heads: shape.num_key_value_heads,
        head_dim: shape.head_dim,
        rotated_dims: shape.head_dim / 4,
        key_heads: shape.linear_num_key_heads,
        value_heads: shape.linear_num_value_heads,
        key_dim: shape.linear_key_head_dim,
        value_dim: shape.linear_value_head_dim,
        max_positions: shape.max_position_embeddings,
        rope_theta: super::YARN.rope_theta as f32,
        rms_epsilon: shape.rms_norm_eps,
    }
}

fn rounded(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn narrow(input: &[f32], output: &mut [bf16]) {
    for (out, &value) in output.iter_mut().zip(input) {
        *out = bf16::from_f32(value);
    }
}

fn norm(
    input: &[bf16],
    weight: &[f32],
    widened: &mut [f32],
    output: &mut [f32],
    epsilon: f32,
) -> Result<()> {
    for (out, value) in widened.iter_mut().zip(input) {
        *out = value.to_f32();
    }
    cpu::rmsnorm(widened, weight, output, 1, input.len(), epsilon)?;
    output.iter_mut().for_each(|value| *value = rounded(*value));
    Ok(())
}

fn residual_add(state: &mut [bf16], branch: &[f32]) {
    for (state, &value) in state.iter_mut().zip(branch) {
        *state = bf16::from_f32(state.to_f32() + rounded(value));
    }
}

fn expand(residual: &mut [Vec<bf16>; 2]) {
    let [first, second] = residual;
    second.copy_from_slice(first);
}

fn reduce(residual: &mut [Vec<bf16>; 2]) {
    let [first, second] = residual;
    for (a, b) in first.iter_mut().zip(second) {
        *a = bf16::from_f32((a.to_f32() + b.to_f32()) * 0.5);
    }
}

fn no_observe(_: Observation<'_>) -> Result<()> {
    Ok(())
}

struct Workspace {
    config: Config,
    capacity: usize,
    scratch: Scratch,
    residual: [Vec<bf16>; 2],
    merged: [Vec<bf16>; 2],
    branch: Vec<bf16>,
    rotary: TextRotary,
    execute: Cpu,
    work: Work,
}

impl Workspace {
    fn planned(c: Config, capacity: usize) -> Result<usize> {
        add(
            add(
                Scratch::planned(c, capacity)?,
                mul(c.hidden, 5 * size_of::<bf16>())?,
            )?,
            size_of::<ScalarSession<'_, '_>>() - size_of::<State>(),
        )
    }

    fn bytes(&self) -> Result<usize> {
        let words = self
            .residual
            .iter()
            .chain(&self.merged)
            .chain([&self.branch])
            .try_fold(0, |n, row| add(n, row.capacity()))?;
        add(
            add(self.scratch.bytes()?, mul(words, size_of::<bf16>())?)?,
            size_of::<ScalarSession<'_, '_>>() - size_of::<State>(),
        )
    }

    fn new(config: Config, capacity: usize) -> Result<Self> {
        let row = || {
            let mut data = Vec::new();
            data.try_reserve_exact(config.hidden)
                .map_err(|_| model::Error::Allocation)?;
            data.resize(config.hidden, bf16::ZERO);
            Ok::<_, Error>(data)
        };
        Ok(Self {
            config,
            capacity,
            scratch: Scratch::new(config, capacity)?,
            residual: [row()?, row()?],
            merged: [row()?, row()?],
            branch: row()?,
            rotary: TextRotary::new(),
            execute: Cpu,
            work: Work::default(),
        })
    }

    fn token(
        &mut self,
        weights: &TextWeights<'_>,
        token: Token,
        frame: &mut Frame<'_>,
        logits: &mut [f32],
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.execute
            .gather(&weights.embedding, token.id, &mut self.scratch.x)?;
        narrow(&self.scratch.x, &mut self.residual[0]);
        self.work = Work::default();
        let mut observe = no_observe;
        let mut trace = Trace {
            observe: &mut observe,
            token: token.position,
            position: token.absolute,
            layer: None,
            cancel,
        };
        while let Some((visit, state)) = frame.next_visit()? {
            let layer = &weights.layers[visit.physical_layer];
            trace.layer = Some(visit.physical_layer);
            match (visit.residual_streams, &layer.routers) {
                (ResidualStreams::Plain, None) => self.plain(layer, state, token, &mut trace)?,
                (
                    mode @ (ResidualStreams::Expand
                    | ResidualStreams::Carry
                    | ResidualStreams::Reduce),
                    Some(routers),
                ) => {
                    if mode == ResidualStreams::Expand {
                        expand(&mut self.residual);
                    }
                    self.routed(layer, routers, state, token, &mut trace)?;
                    if mode == ResidualStreams::Reduce {
                        reduce(&mut self.residual);
                    }
                }
                _ => return Err(invalid("SenseNova physical router and visit differ")),
            }
        }
        norm(
            &self.residual[0],
            weights.final_norm,
            &mut self.scratch.x,
            &mut self.scratch.final_norm,
            self.config.rms_epsilon,
        )?;
        self.execute
            .project(&weights.head, &self.scratch.final_norm, logits)?;
        logits.iter_mut().for_each(|value| *value = rounded(*value));
        Ok(())
    }

    fn mixing<F: for<'b> FnMut(Observation<'b>) -> Result<()>>(
        &mut self,
        layer: &PhysicalLayer<'_>,
        state: VisitState<'_>,
        token: Token,
        trace: &mut Trace<'_, F>,
    ) -> Result<()> {
        let s = &mut self.scratch;
        match (&layer.weights.mixing, state) {
            (
                MixingWeights::Delta(w),
                VisitState::Linear {
                    recurrent,
                    convolution,
                },
            ) => delta::<false, _, _, _>(
                self.config,
                w,
                &mut self.execute,
                convolution.prior,
                convolution.next,
                recurrent.prior,
                recurrent.next,
                &mut s.delta,
                &s.input_norm,
                &mut s.operator,
                &mut self.work,
                trace,
            ),
            (MixingWeights::Attention(w), VisitState::Full(mut kv)) => attention::<false, _, _, _>(
                self.config,
                w,
                &mut self.execute,
                kv.storage(),
                &mut s.attention,
                &s.input_norm,
                &mut s.operator,
                token.position,
                token.absolute,
                self.capacity,
                Some(&self.rotary),
                &mut self.work,
                trace,
            ),
            _ => Err(invalid("SenseNova mixing weights and visit state differ")),
        }
    }

    fn plain<F: for<'b> FnMut(Observation<'b>) -> Result<()>>(
        &mut self,
        layer: &PhysicalLayer<'_>,
        state: VisitState<'_>,
        token: Token,
        trace: &mut Trace<'_, F>,
    ) -> Result<()> {
        norm(
            &self.residual[0],
            layer.weights.input_norm,
            &mut self.scratch.x,
            &mut self.scratch.input_norm,
            self.config.rms_epsilon,
        )?;
        self.mixing(layer, state, token, trace)?;
        residual_add(&mut self.residual[0], &self.scratch.operator);
        norm(
            &self.residual[0],
            layer.weights.post_norm,
            &mut self.scratch.x,
            &mut self.scratch.post_norm,
            self.config.rms_epsilon,
        )?;
        mlp_normalized::<false, _, _, _>(
            &layer.weights,
            &mut self.execute,
            &mut self.scratch,
            &mut self.work,
            trace,
        )?;
        residual_add(&mut self.residual[0], &self.scratch.output);
        Ok(())
    }

    fn routed<F: for<'b> FnMut(Observation<'b>) -> Result<()>>(
        &mut self,
        layer: &PhysicalLayer<'_>,
        routers: &[Router<'_>; 2],
        state: VisitState<'_>,
        token: Token,
        trace: &mut Trace<'_, F>,
    ) -> Result<()> {
        let [attention, ffn] = routers;
        let route = attention.route(
            self.residual.each_ref().map(Vec::as_slice),
            &mut self.branch,
        )?;
        norm(
            &self.branch,
            layer.weights.input_norm,
            &mut self.scratch.x,
            &mut self.scratch.input_norm,
            self.config.rms_epsilon,
        )?;
        self.mixing(layer, state, token, trace)?;
        narrow(&self.scratch.operator, &mut self.branch);
        route.merge(
            self.residual.each_ref().map(Vec::as_slice),
            &self.branch,
            self.merged.each_mut().map(Vec::as_mut_slice),
        )?;
        std::mem::swap(&mut self.residual, &mut self.merged);

        let route = ffn.route(
            self.residual.each_ref().map(Vec::as_slice),
            &mut self.branch,
        )?;
        norm(
            &self.branch,
            layer.weights.post_norm,
            &mut self.scratch.x,
            &mut self.scratch.post_norm,
            self.config.rms_epsilon,
        )?;
        mlp_normalized::<false, _, _, _>(
            &layer.weights,
            &mut self.execute,
            &mut self.scratch,
            &mut self.work,
            trace,
        )?;
        narrow(&self.scratch.output, &mut self.branch);
        route.merge(
            self.residual.each_ref().map(Vec::as_slice),
            &self.branch,
            self.merged.each_mut().map(Vec::as_mut_slice),
        )?;
        std::mem::swap(&mut self.residual, &mut self.merged);
        Ok(())
    }
}

pub struct ScalarSession<'owner, 'weights> {
    weights: &'owner TextWeights<'weights>,
    state: State,
    workspace: Workspace,
}

impl<'owner, 'weights> ScalarSession<'owner, 'weights> {
    /// Exact preflight for scalar session owners and planned workspace.
    pub fn planned(plan: &Plan, options: Options, model_bytes: usize) -> Result<Memory> {
        State::planned(
            plan,
            options,
            model_bytes,
            Workspace::planned(config(plan.text_shape()), options.capacity)?,
        )
    }

    /// Borrow the complete immutable physical table. Charge the retained model,
    /// tokenizer if any, and other live caller-owned allocations in model_bytes.
    pub fn new(
        plan: &Plan,
        weights: &'owner TextWeights<'weights>,
        options: Options,
        model_bytes: usize,
    ) -> Result<Self> {
        let c = config(plan.text_shape());
        if c.max_positions != 524288
            || !plan.default_use_cache()
            || model_bytes < TEXT_BYTES as usize
            || weights.layers.len() != PHYSICAL_LAYERS
            || weights.final_norm.len() != c.hidden
            || [&weights.embedding, &weights.head]
                .iter()
                .any(|m| (m.layout().rows(), m.layout().columns()) != (c.vocab, c.hidden))
        {
            return Err(invalid(
                "SenseNova packed scalar binding or memory charge differs",
            ));
        }
        State::planned(
            plan,
            options,
            model_bytes,
            Workspace::planned(c, options.capacity)?,
        )?;
        let workspace = Workspace::new(c, options.capacity)?;
        let state = State::new(plan, options, model_bytes, workspace.bytes()?)?;
        Ok(Self {
            weights,
            state,
            workspace,
        })
    }

    pub fn forward(&mut self, tokens: &[u32], cancel: &AtomicBool) -> Result<&[f32]> {
        let workspace = &mut self.workspace;
        let weights = self.weights;
        self.state.evaluate(tokens, cancel, |token, frame, logits| {
            workspace.token(weights, token, frame, logits, cancel)
        })
    }

    pub fn memory(&self) -> Memory {
        self.state.memory()
    }
    pub fn history(&self) -> &[u32] {
        self.state.history()
    }
    pub fn logits(&self) -> Option<&[f32]> {
        self.state.logits()
    }
    pub fn reset(&mut self) {
        self.state.reset();
    }
    pub fn reuse_prefix(&mut self, tokens: &[u32]) -> Result<usize> {
        self.state.reuse_prefix(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_residual_boundaries_store_branch_before_merge_and_mean() {
        let mut residual = [
            vec![bf16::from_f32(1.), bf16::from_f32(4.)],
            vec![bf16::ZERO; 2],
        ];
        expand(&mut residual);
        assert_eq!(residual[0], residual[1]);
        let update = [513.0 / 131072.0, 2.];
        residual_add(&mut residual[1], &update);
        assert_eq!(residual[1], [bf16::from_f32(1.), bf16::from_f32(6.)]);
        reduce(&mut residual);
        assert_eq!(residual[0], [bf16::from_f32(1.), bf16::from_f32(5.)]);
    }
}
