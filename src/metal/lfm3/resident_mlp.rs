//! Single guarded command for the fixed T128 Pair, BF16 bridge and Down.
use super::{
    down::DownWorkspace,
    pair::{DomainLaunch, PairWorkspace, domain_tag},
    projection_tile::{Launch, record},
    *,
};
use crate::{
    cpu::lfm3_silu_bf16,
    metal::lfm3_shared_weights::BankRef,
    model::{Error, lfm3::reserve},
};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, Ordering};

const M: usize = crate::model::lfm3::INTERMEDIATE;
const T: usize = 128;
const PRODUCT: usize = T * M;
const OUT: usize = T * I;
const TILE: usize = 32;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

struct Bridge {
    admit: Pipeline,
    product: Pipeline,
    tags: Pipeline,
}

impl Bridge {
    fn new(gpu: &Gpu) -> Result<Self, Error> {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version2_3);
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        if options.languageVersion() != MTLLanguageVersion::Version2_3
            || options.mathMode() != MTLMathMode::Safe
            || options.mathFloatingPointFunctions() != MTLMathFloatingPointFunctions::Precise
        {
            return Err(Error::InvalidConfig("resident MLP compile options"));
        }
        let source = include_str!("resident_elementwise.metal")
            .replace("__LFM3_DYADIC__", include_str!("dyadic_shared.h"))
            + include_str!("resident_tags.metal");
        let lib = gpu
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
            .map_err(|_| Error::InvalidConfig("resident MLP bridge library"))?;
        let pipeline = |name| {
            let f = lib
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or(Error::InvalidConfig("resident MLP bridge function"))?;
            let p = gpu
                .device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|_| Error::InvalidConfig("resident MLP bridge pipeline"))?;
            if p.threadExecutionWidth() != TILE || p.maxTotalThreadsPerThreadgroup() < TILE {
                return Err(Error::InvalidConfig("resident MLP bridge SIMD width"));
            }
            Ok(p)
        };
        Ok(Self {
            admit: pipeline("lfm3_admit_elementwise")?,
            product: pipeline("lfm3_silu_product_witness")?,
            tags: pipeline("lfm3_resident_tags")?,
        })
    }
}

fn encoder(
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    pipeline: &Pipeline,
    groups: usize,
    bind: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
) -> Result<(), Error> {
    let e = command
        .computeCommandEncoder()
        .ok_or(Error::InvalidConfig("resident encoder"))?;
    e.setComputePipelineState(pipeline);
    bind(&e);
    let size = |width| MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    e.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(TILE));
    e.endEncoding();
    Ok(())
}

enum Tags<'a> {
    Pair(&'a Buffer<u32>, usize),
    Down(&'a Launch<'a>),
}

struct Linear<'a> {
    input: &'a Buffer<u16>,
    status: &'a Buffer<u32>,
    output: &'a Buffer<u16>,
    routes: &'a Buffer<u32>,
    input_tags: &'a Buffer<u32>,
    tags: Tags<'a>,
    dims: [u32; 4],
    bank: BankRef<'a>,
}

impl Linear<'_> {
    fn encode(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        pipeline: &Pipeline,
        groups: usize,
        projection: bool,
    ) -> Result<(), Error> {
        encoder(command, pipeline, groups, |e| {
            // SAFETY: same-device immutable BankRef and disjoint tracked staging
            // live through the command wait. Required encoders run in order.
            unsafe {
                self.bank.bind_read_only(e, 1);
                e.setBuffer_offset_atIndex(Some(&self.input.raw), 0, 0);
                e.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 2);
                e.setBytes_length_atIndex(
                    NonNull::from(&self.dims).cast(),
                    size_of::<[u32; 4]>(),
                    3,
                );
                if projection {
                    e.setBuffer_offset_atIndex(Some(&self.output.raw), 0, 4);
                    e.setBuffer_offset_atIndex(Some(&self.routes.raw), 0, 5);
                    e.setBuffer_offset_atIndex(Some(&self.input_tags.raw), 0, 6);
                    match &self.tags {
                        Tags::Pair(weights, offset) => {
                            e.setBuffer_offset_atIndex(Some(&weights.raw), *offset, 7);
                        }
                        Tags::Down(tile) => tile.bind_weight_tags(e),
                    }
                }
            }
        })
    }
}

pub(super) struct Workspace {
    bridge: Bridge,
    lut: Buffer<u16>,
    status: Buffer<u32>,
    activation: Buffer<u16>,
    product: Buffer<u16>,
    product_routes: Buffer<u32>,
    product_tags: Buffer<u32>,
    census: Buffer<u32>,
    admitted_bytes: usize,
    pub(super) published: u64,
    submitted: u64,
    completed: u64,
    clocks: [f64; 2],
    pair_census: [u64; 4],
    bridge_census: [u64; 3],
    down_census: [u64; 3],
}

impl Workspace {
    pub(super) const BUFFER_BYTES: [usize; 7] = [
        2 * 65536,
        8,
        2 * (PRODUCT + 1),
        2 * (PRODUCT + 1),
        4 * (PRODUCT + 1),
        4 * (T + 1),
        4 * (7 * T + 1),
    ];
    pub(super) const LUT_TEMP_BYTES: usize = 4 * 0x7f80;

    pub(super) fn new(gpu: &Gpu, admitted_bytes: usize) -> Result<Self, Error> {
        let words = |n| Buffer::<u16>::new(gpu, n).map_err(|_| Error::Allocation);
        let counters = |n| Buffer::<u32>::new(gpu, n).map_err(|_| Error::Allocation);
        let mut lut = words(65536)?;
        let mut input = reserve(0x7f80)?;
        let mut output = reserve(0x7f80)?;
        output.resize(0x7f80, bf16::ZERO);
        lut.with_mut(|table| {
            table.fill(SENTINEL);
            for sign in [0_u16, 0x8000] {
                input.clear();
                input.extend((0..0x7f80).map(|bits| bf16::from_bits(bits | sign)));
                lfm3_silu_bf16(&input, &mut output)?;
                for (dst, value) in table[usize::from(sign)..].iter_mut().zip(&output) {
                    *dst = value.to_bits();
                }
            }
            Ok::<_, Error>(())
        })?;
        let digest = lut.with_slice(|table| {
            let mut sha = Sha256::new();
            for word in table {
                sha.update(word.to_le_bytes());
            }
            format!("{:x}", sha.finalize())
        });
        if digest != "b64cf71ac1bbe8f4dfdfdcf10cb49008235662eba94f4b5297e24baf438afcfe" {
            return Err(Error::InvalidConfig("resident MLP canonical SiLU table"));
        }
        Ok(Self {
            bridge: Bridge::new(gpu)?,
            lut,
            status: counters(2)?,
            activation: words(PRODUCT + 1)?,
            product: words(PRODUCT + 1)?,
            product_routes: counters(PRODUCT + 1)?,
            product_tags: counters(T + 1)?,
            census: counters(7 * T + 1)?,
            admitted_bytes,
            published: 0,
            submitted: 0,
            completed: 0,
            clocks: [0.; 2],
            pair_census: [0; 4],
            bridge_census: [0; 3],
            down_census: [0; 3],
        })
    }

    pub(super) fn report(&self) -> serde_json::Value {
        serde_json::json!({
            "placement": "metal-t128", "admitted_bytes": self.admitted_bytes,
            "workspace_payload_bytes": Self::BUFFER_BYTES.iter().sum::<usize>(),
            "published": self.published, "submitted": self.submitted, "validated": self.completed,
            "completed_gpu_seconds": self.clocks[0], "commit_wait_seconds": self.clocks[1],
            "pair_census": self.pair_census, "bridge_census": self.bridge_census,
            "down_census": self.down_census,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run(
        &mut self,
        pair: &mut PairWorkspace,
        down: &mut DownWorkspace,
        input: &[bf16],
        banks: [BankRef<'_>; 3],
        dst: &mut [bf16],
        domain: DomainLaunch<'_>,
        tile: Launch<'_>,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        if input.len() != OUT
            || dst.len() != OUT
            || domain.tile_size != TILE
            || !domain.selected
            || tile.outputs_per_group != TILE
        {
            return Err(Error::InvalidConfig("resident MLP T128 shape or placement"));
        }
        domain.input.with_mut(|tags| {
            pair.staged.with_mut(|staged| {
                for ((source, slot), tag) in input
                    .chunks_exact(I)
                    .zip(staged.chunks_exact_mut(I))
                    .zip(tags)
                {
                    *tag = domain_tag(source.iter().zip(slot).map(|(w, v)| {
                        *v = w.to_bits();
                        *v
                    }));
                }
            });
        });
        for b in [&mut self.activation, &mut self.product, &mut down.output] {
            b.with_mut(|x| x.fill(SENTINEL));
        }
        for b in [
            &mut self.product_routes,
            &mut self.product_tags,
            &mut self.census,
            &mut down.routes,
        ] {
            b.with_mut(|x| x.fill(ROUTE_GUARD));
        }
        for b in [&mut pair.status, &mut self.status, &mut down.status] {
            b.with_mut(|x| x.copy_from_slice(&[0, STATUS_GUARD]));
        }
        for b in &mut pair.output {
            b.with_mut(|x| x.fill(SENTINEL));
        }
        for b in &mut pair.routes {
            b.with_mut(|x| x.fill(ROUTE_GUARD));
        }
        let command = pair
            .gpu
            .command()
            .map_err(|_| Error::InvalidConfig("resident command"))?;
        let projections = [0, 1].map(|i| {
            let (dims, groups) = Kernels::sealed_admission(banks[i], T, I, M);
            (
                Linear {
                    input: &pair.staged,
                    status: &pair.status,
                    output: &pair.output[i],
                    routes: &pair.routes[i],
                    input_tags: domain.input,
                    tags: Tags::Pair(
                        domain.weights,
                        (2 * domain.layer + i) * M * size_of::<u32>(),
                    ),
                    dims,
                    bank: banks[i],
                },
                groups,
            )
        });
        for (linear, groups) in &projections {
            linear.encode(&command, &pair.kernels.admission, *groups, false)?;
        }
        for (linear, _) in &projections {
            linear.encode(&command, &domain.kernels.projection, PRODUCT / TILE, true)?;
        }
        let count = PRODUCT as u32;
        for (pipeline, product) in [(&self.bridge.admit, false), (&self.bridge.product, true)] {
            encoder(&command, pipeline, PRODUCT / TILE, |e| {
                // SAFETY: the exact shader ABI covers PRODUCT private words.
                // Pair writes completed before both dependent encoders.
                unsafe {
                    e.setBuffer_offset_atIndex(Some(&pair.output[0].raw), 0, 0);
                    e.setBuffer_offset_atIndex(Some(&pair.output[1].raw), 0, 1);
                    e.setBytes_length_atIndex(NonNull::from(&count).cast(), size_of::<u32>(), 2);
                    e.setBuffer_offset_atIndex(Some(&self.status.raw), 0, 3);
                    if product {
                        e.setBuffer_offset_atIndex(Some(&self.lut.raw), 0, 4);
                        e.setBuffer_offset_atIndex(Some(&self.activation.raw), 0, 5);
                        e.setBuffer_offset_atIndex(Some(&self.product.raw), 0, 6);
                        e.setBuffer_offset_atIndex(Some(&self.product_routes.raw), 0, 7);
                    }
                }
            })?;
        }
        encoder(&command, &self.bridge.tags, T, |e| {
            // SAFETY: this T128 encoder reads completed producer outputs,
            // and owns disjoint tracked tag/census/status buffers until wait.
            unsafe {
                e.setBuffer_offset_atIndex(Some(&self.product.raw), 0, 0);
                e.setBuffer_offset_atIndex(Some(&pair.routes[0].raw), 0, 1);
                e.setBuffer_offset_atIndex(Some(&pair.routes[1].raw), 0, 2);
                e.setBuffer_offset_atIndex(Some(&self.product_routes.raw), 0, 3);
                e.setBuffer_offset_atIndex(Some(&self.product_tags.raw), 0, 4);
                e.setBuffer_offset_atIndex(Some(&self.census.raw), 0, 5);
                e.setBuffer_offset_atIndex(Some(&down.status.raw), 0, 6);
            }
        })?;
        let (dims, groups) = Kernels::sealed_admission(banks[2], T, M, I);
        let down_projection = Linear {
            input: &self.product,
            status: &down.status,
            output: &down.output,
            routes: &down.routes,
            input_tags: &self.product_tags,
            tags: Tags::Down(&tile),
            dims,
            bank: banks[2],
        };
        down_projection.encode(&command, &tile.kernels.admission, groups, false)?;
        down_projection.encode(&command, &tile.kernels.projection, OUT / TILE, true)?;
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let submitted = self
            .submitted
            .checked_add(1)
            .filter(|&n| n <= u64::MAX / PRODUCT as u64)
            .ok_or(Error::InvalidConfig("resident MLP submission count"))?;
        let clock = Instant::now();
        command.commit();
        self.submitted = submitted;
        command.waitUntilCompleted();
        let wait = clock.elapsed().as_secs_f64();
        if command.status() != MTLCommandBufferStatus::Completed || command.error().is_some() {
            return Err(Error::InvalidConfig(
                "resident command completed unsuccessfully",
            ));
        }
        let (start, end) = (command.GPUStartTime(), command.GPUEndTime());
        if !start.is_finite() || !end.is_finite() || start <= 0. || end <= start {
            return Err(Error::InvalidConfig("resident command timestamp"));
        }
        let flags =
            [&pair.status, &self.status, &down.status].map(|b| b.with_slice(|x| [x[0], x[1]]));
        let output_guarded = [
            &pair.output[0],
            &pair.output[1],
            &self.activation,
            &self.product,
        ]
        .into_iter()
        .all(|b| b.with_slice(|x| x[PRODUCT] == SENTINEL))
            && down.output.with_slice(|x| x[OUT] == SENTINEL);
        let census_guarded = [
            &pair.routes[0],
            &pair.routes[1],
            &self.product_routes,
            &self.product_tags,
            &self.census,
            &down.routes,
        ]
        .into_iter()
        .all(|b| b.with_slice(|x| x[x.len() - 1] == ROUTE_GUARD));
        if !output_guarded || !census_guarded || flags.iter().any(|f| f[1] != STATUS_GUARD) {
            return Err(Error::InvalidConfig("resident guard"));
        }
        if flags.map(|x| x[0]) == [0, 0, 4] {
            return Err(Error::InvalidConfig("resident Down overflow"));
        }
        if flags != [[0, STATUS_GUARD]; 3] {
            return Err(Error::InvalidConfig("resident producer or admission"));
        }
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let mut counts = [0_u64; 7];
        self.census.with_slice(|words| {
            for row in words[..T * 7].chunks_exact(7) {
                for (total, count) in counts.iter_mut().zip(row) {
                    *total += u64::from(*count);
                }
            }
        });
        let [gn, gp, un, up, zero, hardware, integer] = counts;
        if gn > PRODUCT as u64
            || un > PRODUCT as u64
            || gp > gn
            || up > un
            || gp % TILE as u64 != 0
            || up % TILE as u64 != 0
            || zero + hardware + integer != PRODUCT as u64
        {
            return Err(Error::InvalidConfig("resident intermediate census"));
        }
        let mut down_census = [0; 3];
        if !down.routes.with_slice(|words| {
            words[..OUT]
                .iter()
                .all(|&word| record(word, 16, M, true, &mut down_census))
        }) || !down
            .output
            .with_slice(|words| words[..OUT].iter().all(|&w| w & 0x7fff < 0x7f80))
        {
            return Err(Error::InvalidConfig("resident Down census or output"));
        }
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        down.output.with_slice(|words| {
            for (out, &bits) in dst.iter_mut().zip(words) {
                *out = bf16::from_bits(bits);
            }
        });
        self.completed += 1; // A completed visit always has a checked submission.
        self.clocks[0] += end - start;
        self.clocks[1] += wait;
        for (total, n) in self.pair_census.iter_mut().zip([gn, gp, un, up]) {
            *total += n;
        }
        for (total, n) in self.bridge_census.iter_mut().zip([zero, hardware, integer]) {
            *total += n;
        }
        for (total, n) in self.down_census.iter_mut().zip(down_census) {
            *total += n;
        }
        Ok(())
    }
}
