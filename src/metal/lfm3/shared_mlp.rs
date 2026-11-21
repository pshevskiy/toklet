use super::{
    down::DownWorkspace,
    pair::{DomainLaunch, PairWorkspace, domain_tag},
    *,
};
use crate::{metal::lfm3_shared_weights::BankRef, model::Error};

pub(crate) struct SharedMlpWorkspace {
    pair: PairWorkspace,
    down: DownWorkspace,
    resident: Option<Box<[super::resident_mlp::Workspace; 1]>>,
    domain: Option<DomainMetadata>,
    decode_ffn: Option<[Retained<ProtocolObject<dyn MTLComputePipelineState>>; 2]>,
    conv: Option<super::conv_workspace::ConvWorkspace>,
    output: Option<super::output_workspace::OutputWorkspace>,
    projection_tiles: Option<super::projection_tile::Tiles>,
    qkv: Option<super::qkv_workspace::Workspace>,
    last3_attention: Option<super::attention_workspace::Workspace>,
    tied: Option<super::shared_tied::Workspace>,
    completed: [u64; 2],
    clocks: [[f64; 2]; 2],
}

const TAG_K: usize = 2048;
const TAG_N: usize = 7680;
const TAG_LAYERS: usize = 24;
const TAG_T: usize = 128;

struct DomainMetadata {
    chosen: bool,
    kernels: Kernels,
    input: Buffer<u32>,
    weights: Buffer<u32>,
    bank_ids: [[usize; 2]; TAG_LAYERS],
    native: [u64; 2],
    possible: u64,
    t4: Option<TileTelemetry>,
}

struct TileTelemetry {
    selected: bool,
    tile_size: usize,
    pipeline: Kernels,
    shapes: [ShapeTelemetry; 3],
}

#[derive(Default)]
struct ShapeTelemetry {
    calls: u64,
    tiled: u64,
    groups: u64,
    output_rows: u64,
    native: [u64; 2],
    homogeneous: [u64; 2],
    clocks: [f64; 2],
}

impl SharedMlpWorkspace {
    pub(crate) const QKV_BUFFER_BYTES: [usize; 14] = super::qkv_workspace::Workspace::BYTE_SIZES;

    pub(crate) fn prepare_qkv(&mut self) -> std::result::Result<(), Error> {
        self.qkv = Some(super::qkv_workspace::Workspace::new(&self.pair.gpu)?);
        Ok(())
    }

    pub(crate) fn seal_qkv(
        &mut self,
        layer: usize,
        banks: [(BankRef<'_>, &[bf16]); 3],
    ) -> std::result::Result<(), Error> {
        self.qkv
            .as_mut()
            .ok_or(Error::InvalidConfig("LFM3 QKV workspace absent"))?
            .seal(&self.pair.gpu, layer, banks)
    }

    pub(crate) fn qkv(
        &mut self,
        layer: usize,
        banks: [BankRef<'_>; 3],
        input: &[bf16],
        out: [&mut [bf16]; 3],
    ) -> std::result::Result<(), Error> {
        let scalar = self
            .output
            .as_ref()
            .ok_or(Error::InvalidConfig("LFM3 output workspace"))?
            .scalar_kernels();
        self.qkv
            .as_mut()
            .ok_or(Error::InvalidConfig("LFM3 QKV workspace absent"))?
            .run(&self.pair.gpu, scalar, layer, banks, input, out)
    }

    // The released per-role buffers already back both shapes. Reserve
    // optional native framework/pipeline memory before constructing T1.
    pub(crate) const QKV_T1_NATIVE_ALLOWANCE: usize = 16 << 20;

    pub(crate) fn prepare_qkv_t1(&mut self) -> bool {
        self.qkv
            .as_mut()
            .is_some_and(|q| q.prepare_t1(&self.pair.gpu))
    }

    pub(crate) fn has_qkv_t1(&self) -> bool {
        self.qkv
            .as_ref()
            .is_some_and(super::qkv_workspace::Workspace::has_t1)
    }

    pub(crate) fn qkv_t1(
        &mut self,
        layer: usize,
        banks: [BankRef<'_>; 3],
        input: &[bf16],
        out: [&mut [bf16]; 3],
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> std::result::Result<(), Error> {
        self.qkv
            .as_mut()
            .ok_or(Error::InvalidConfig("LFM3 QKV workspace absent"))?
            .run_t1(&self.pair.gpu, layer, banks, input, out, cancelled)
    }

    pub(crate) fn qkv_t1_stats(&self) -> Option<serde_json::Value> {
        self.qkv
            .as_ref()
            .and_then(super::qkv_workspace::Workspace::t1_report)
    }

    pub(crate) fn qkv_stats(&self) -> Option<serde_json::Value> {
        self.qkv
            .as_ref()
            .map(super::qkv_workspace::Workspace::report)
    }

    pub(crate) const CONV_BYTES: usize = super::conv_workspace::ConvWorkspace::BYTES;
    pub(crate) const DOMAIN_BYTES: usize = (TAG_LAYERS * 2 * TAG_N + TAG_T) * size_of::<u32>();
    pub(crate) const BYTES: usize = PairWorkspace::BYTES + DownWorkspace::BYTES;
    pub(crate) const PROJECTION_TILE_BYTES: usize = super::projection_tile::Tiles::BYTES;
    pub(crate) const PROJECTION_TILE_BUFFERS: usize = 4;

    pub(crate) fn seal_projection_tile(
        &mut self,
        role: usize,
        layer: usize,
        bank: BankRef<'_>,
        data: &[bf16],
    ) -> std::result::Result<(), Error> {
        if role == 0 {
            self.validate_conv(bank)?;
        } else {
            self.validate(bank)?;
        }
        self.projection_tiles
            .as_mut()
            .ok_or(Error::InvalidConfig("LFM3 missing projection tile"))?
            .seal(role, layer, bank, data)
    }

    pub(crate) fn projection_tile_stats(&self) -> Option<serde_json::Value> {
        self.projection_tiles
            .as_ref()
            .map(super::projection_tile::Tiles::report)
    }

    pub(crate) const OUTPUT_BYTES: usize = super::output_workspace::OutputWorkspace::BYTES;
    pub(crate) const OUTPUT_BUFFERS: usize = super::output_workspace::OutputWorkspace::BUFFER_COUNT;
    pub(crate) const WORDS: usize = 2048 * 7680;

    pub(crate) fn new(device: &ProtocolObject<dyn MTLDevice>) -> std::result::Result<Self, Error> {
        let pair = PairWorkspace::try_new()?;
        if !std::ptr::eq::<ProtocolObject<dyn MTLDevice>>(&*pair.gpu.device, device) {
            return Err(Error::InvalidConfig("LFM3 shared workspace device"));
        }
        let down = DownWorkspace::new(&pair.gpu)?;
        let gpu = &pair.gpu;
        let domain = Some(DomainMetadata {
            chosen: true,
            kernels: Kernels::with_shader(gpu, include_str!("normal_domain.metal"))
                .ok_or(Error::InvalidConfig("LFM3 shared normal-domain pipeline"))?,
            input: Buffer::new(gpu, TAG_T).map_err(|_| Error::Allocation)?,
            weights: Buffer::new(gpu, TAG_LAYERS * 2 * TAG_N).map_err(|_| Error::Allocation)?,
            bank_ids: [[0; 2]; TAG_LAYERS],
            native: [0; 2],
            possible: 0,
            t4: Some(TileTelemetry {
                selected: true,
                tile_size: 32,
                pipeline: Kernels::with_shader(gpu, include_str!("panel84.metal"))
                    .ok_or(Error::InvalidConfig("LFM3 shared Panel84 pipeline"))?,
                shapes: Default::default(),
            }),
        });
        let conv = Some(super::conv_workspace::ConvWorkspace::new(gpu)?);
        let output = Some(super::output_workspace::OutputWorkspace::new(gpu)?);
        let projection_tiles = Some(super::projection_tile::Tiles::new(gpu)?);
        Ok(Self {
            projection_tiles,
            qkv: None,
            last3_attention: None,
            tied: None,
            output,
            conv,
            domain,
            decode_ffn: None,
            pair,
            down,
            resident: None,
            completed: [0; 2],
            clocks: [[0.; 2]; 2],
        })
    }

    pub(crate) const DECODE_FFN_PIPELINE_ALLOWANCE: usize = 32 << 20;

    pub(crate) fn prepare_decode_ffn(&mut self) -> bool {
        let gpu = &self.pair.gpu;
        // A partially constructed optional pair is dropped without changing
        // any live dispatch. The released input admission PSOs stay bound.
        let Some(pair) = Kernels::with_shader(gpu, include_str!("decode_pair_fma4.metal")) else {
            return false;
        };
        let Some(down) = Kernels::with_shader(gpu, include_str!("decode_down_fma4.metal")) else {
            return false;
        };
        self.decode_ffn = Some([pair.projection, down.projection]);
        true
    }

    pub(crate) fn has_decode_ffn(&self) -> bool {
        self.decode_ffn.is_some()
    }

    pub(crate) const LAST3_ATTENTION_BUFFERS: [usize; 6] =
        super::attention_workspace::Workspace::BUFFER_BYTES;
    pub(crate) const LAST3_ATTENTION_NATIVE_ALLOWANCE: usize =
        super::attention_workspace::Workspace::NATIVE_ALLOWANCE;

    pub(crate) fn prepare_last3_attention(
        &mut self,
        selected: bool,
    ) -> std::result::Result<(), Error> {
        self.last3_attention = Some(super::attention_workspace::Workspace::new(
            &self.pair.gpu,
            selected,
        )?);
        Ok(())
    }

    pub(crate) fn last3_attention_stats(&self) -> Option<[u64; 5]> {
        self.last3_attention
            .as_ref()
            .map(super::attention_workspace::Workspace::stats)
    }

    pub(crate) fn apply_last3_attention(
        &mut self,
        layer: usize,
        q: &[bf16],
        spans: &[crate::cpu::KvSegment<'_, bf16>; 3],
        out: &mut [bf16],
        shape: &crate::cpu::SpanAttentionShape,
    ) -> std::result::Result<bool, Error> {
        if let Some(owner) = &mut self.last3_attention {
            owner.apply(&self.pair.gpu, layer, q, spans, out, shape)
        } else {
            Ok(false)
        }
    }

    pub(crate) fn seal_tied(
        &mut self,
        bank: BankRef<'_>,
        words: &[bf16],
    ) -> std::result::Result<(), Error> {
        self.tied = Some(super::shared_tied::Workspace::new(
            &self.pair.gpu,
            bank,
            words,
        )?);
        Ok(())
    }

    pub(crate) fn tied(
        &mut self,
        bank: BankRef<'_>,
        words: &[bf16],
        input: &[bf16],
        output: &mut [bf16],
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> std::result::Result<(), Error> {
        self.tied
            .as_mut()
            .ok_or(Error::InvalidConfig("LFM3 tied head workspace absent"))?
            .project(&self.pair.gpu, bank, words, input, output, cancelled)
    }

    pub(crate) fn tied_report(&self) -> Option<serde_json::Value> {
        self.tied
            .as_ref()
            .map(super::shared_tied::Workspace::report)
    }

    pub(crate) fn validate(&self, bank: BankRef<'_>) -> std::result::Result<(), Error> {
        if !bank.device_matches(&self.pair.gpu.device) || bank.words() != Self::WORDS {
            return Err(Error::InvalidConfig(
                "LFM3 shared MLP bank device or extent",
            ));
        }
        Ok(())
    }

    pub(crate) fn seal_pair_tags(
        &mut self,
        views: [[(BankRef<'_>, &[bf16]); 2]; TAG_LAYERS],
    ) -> std::result::Result<(), Error> {
        for pair in views {
            for (bank, words) in pair {
                self.validate(bank)?;
                if words.len() != Self::WORDS {
                    return Err(Error::InvalidConfig("LFM3 shared tag extent"));
                }
            }
        }
        let Some(domain) = &mut self.domain else {
            return Ok(());
        };
        if domain.bank_ids[0][0] != 0 {
            return Err(Error::InvalidConfig("LFM3 shared tags already sealed"));
        }
        domain.weights.with_mut(|tags| {
            for (((bank, words), dst), identity) in views
                .into_iter()
                .flatten()
                .zip(tags.chunks_exact_mut(TAG_N))
                .zip(domain.bank_ids.iter_mut().flatten())
            {
                *identity = bank.identity();
                for (tag, row) in dst.iter_mut().zip(words.chunks_exact(TAG_K)) {
                    *tag = domain_tag(row.iter().map(|word| word.to_bits()));
                }
            }
        });
        Ok(())
    }

    pub(crate) fn domain_stats(&self) -> Option<(bool, [u64; 2], u64)> {
        self.domain
            .as_ref()
            .map(|d| (d.chosen, d.native, d.possible))
    }

    pub(crate) fn pair(
        &mut self,
        layer: usize,
        banks: [BankRef<'_>; 2],
        x: &[bf16],
        [gate, up]: [&mut [bf16]; 2],
        tokens: usize,
        decode: bool,
    ) -> std::result::Result<(), Error> {
        for bank in banks {
            self.validate(bank)?;
        }
        if banks[0].identity() == banks[1].identity() {
            return Err(Error::InvalidConfig("LFM3 shared pair aliases"));
        }
        let mut next_possible = 0;
        let mut tile_size = 1;
        let launch = if let Some(d) = self.domain.as_mut() {
            if d.bank_ids.get(layer) != Some(&banks.map(BankRef::identity))
                || !(1..=TAG_T).contains(&tokens)
            {
                return Err(Error::InvalidConfig("LFM3 shared tag role or shape"));
            }
            next_possible = d
                .possible
                .checked_add((tokens * TAG_N) as u64)
                .ok_or(Error::InvalidConfig("LFM3 domain output count"))?;
            let projection = match (&d.t4, self.decode_ffn.as_ref()) {
                (Some(t), _) if t.selected && tokens == TAG_T => {
                    tile_size = t.tile_size;
                    &t.pipeline.projection
                }
                (_, Some([pair, _])) if decode && tokens == 1 && d.chosen => {
                    tile_size = 4;
                    pair
                }
                _ => &d.kernels.projection,
            };
            Some(DomainLaunch {
                tile_size,
                selected: d.chosen,
                projection,
                input: &mut d.input,
                weights: &d.weights,
                layer,
            })
        } else {
            None
        };
        let (times, receipt) = self
            .pair
            .project_selected(x, gate, up, tokens, banks, launch)?;
        if let Some(d) = self.domain.as_mut() {
            for (total, increment) in d.native.iter_mut().zip(receipt.native) {
                // Selected output count is bounded by the checked possible count.
                *total += increment;
            }
            d.possible = next_possible;
            if let Some(t) = &mut d.t4 {
                let shape = &mut t.shapes[match tokens {
                    TAG_T => 0,
                    1 => 1,
                    _ => 2,
                }];
                // Each subtotal is bounded by the checked total output count.
                shape.calls += 1;
                shape.tiled += u64::from(tokens == TAG_T && tile_size != 1);
                shape.groups += receipt.projection_groups as u64;
                shape.output_rows += (tokens * TAG_N) as u64;
                for i in 0..2 {
                    shape.native[i] += receipt.native[i];
                    shape.homogeneous[i] += receipt.homogeneous[i] / tile_size as u64;
                }
                for (clock, elapsed) in shape.clocks.iter_mut().zip(times) {
                    *clock += elapsed;
                }
            }
        }
        self.completed[0] += 1;
        self.record(0, times);
        Ok(())
    }

    pub(crate) fn down(
        &mut self,
        layer: usize,
        bank: BankRef<'_>,
        x: &[bf16],
        output: &mut [bf16],
        tokens: usize,
        decode: bool,
    ) -> std::result::Result<(), Error> {
        self.validate(bank)?;
        let decode_projection = if decode && tokens == 1 {
            self.decode_ffn.as_ref().map(|pipelines| &pipelines[1])
        } else {
            None
        };
        let tile = self
            .projection_tiles
            .as_mut()
            .map(|tiles| tiles.launch(1, layer, bank, tokens, decode_projection))
            .transpose()?
            .flatten();
        let tiled = tile.is_some();
        let (times, _, census) =
            self.down
                .project_selected(&self.pair.gpu, x, output, tokens, bank, tile)?;
        if let Some(tiles) = &mut self.projection_tiles {
            tiles.completed(1, layer, tokens, tiled, census);
        }
        self.completed[1] += 1;
        self.record(1, times);
        Ok(())
    }

    pub(crate) const RESIDENT_MLP_BUFFERS: [usize; 7] =
        super::resident_mlp::Workspace::BUFFER_BYTES;
    pub(crate) const RESIDENT_MLP_LUT_TEMP_BYTES: usize =
        super::resident_mlp::Workspace::LUT_TEMP_BYTES;

    pub(crate) const RESIDENT_MLP_OWNER_BYTES: usize = size_of::<super::resident_mlp::Workspace>();

    pub(crate) fn prepare_resident_mlp(&mut self, admitted_bytes: usize) -> Result<(), Error> {
        // Keep optional state out of the inline session and reserve host storage
        // fallibly before allocating or compiling the optional Metal workspace.
        let mut owner = crate::model::lfm3::reserve(1)?;
        owner.push(super::resident_mlp::Workspace::new(
            &self.pair.gpu,
            admitted_bytes,
        )?);
        self.resident = Some(
            owner
                .into_boxed_slice()
                .try_into()
                .map_err(|_| Error::InvalidConfig("LFM3 resident MLP owner extent"))?,
        );
        Ok(())
    }

    pub(crate) fn resident_mlp_published(&self) -> Option<u64> {
        self.resident.as_ref().map(|owner| owner[0].published)
    }

    pub(crate) fn commit_resident_mlp(&mut self, published: u64) {
        // A publication was reserved only when this owner was present.
        self.resident.as_mut().unwrap()[0].published = published;
    }

    pub(crate) fn resident_mlp(
        &mut self,
        layer: usize,
        banks: [BankRef<'_>; 3],
        input: &[bf16],
        output: &mut [bf16],
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<(), Error> {
        for bank in banks {
            self.validate(bank)?;
        }
        let [gate, up, down] = banks.map(BankRef::identity);
        if gate == up || gate == down || up == down {
            return Err(Error::InvalidConfig("LFM3 resident MLP aliased banks"));
        }
        let domain = self
            .domain
            .as_mut()
            .ok_or(Error::InvalidConfig("LFM3 resident MLP Pair domain"))?;
        if domain.bank_ids.get(layer) != Some(&[gate, up]) {
            return Err(Error::InvalidConfig("LFM3 resident MLP Pair banks"));
        }
        let panel = domain
            .t4
            .as_ref()
            .filter(|p| p.selected)
            .ok_or(Error::InvalidConfig("LFM3 resident MLP Pair panel"))?;
        let pair_launch = DomainLaunch {
            tile_size: panel.tile_size,
            selected: domain.chosen,
            projection: &panel.pipeline.projection,
            input: &mut domain.input,
            weights: &domain.weights,
            layer,
        };
        let down_launch = self
            .projection_tiles
            .as_mut()
            .ok_or(Error::InvalidConfig("LFM3 resident MLP Down tile owner"))?
            .launch(1, layer, banks[2], TAG_T, None)?
            .ok_or(Error::InvalidConfig("LFM3 resident MLP Down tile"))?;
        self.resident
            .as_mut()
            .ok_or(Error::InvalidConfig("LFM3 resident MLP workspace absent"))?[0]
            .run(
                &mut self.pair,
                &mut self.down,
                input,
                banks,
                output,
                pair_launch,
                down_launch,
                cancelled,
            )
    }

    pub(crate) fn resident_mlp_report(&self) -> Option<serde_json::Value> {
        self.resident.as_ref().map(|owner| owner[0].report())
    }

    fn record(&mut self, role: usize, [gpu, wait]: [f64; 2]) {
        self.clocks[role][0] += gpu;
        self.clocks[role][1] += wait;
    }

    pub(crate) fn validate_qkv(
        &self,
        bank: BankRef<'_>,
        width: usize,
    ) -> std::result::Result<(), Error> {
        if !bank.device_matches(&self.pair.gpu.device) || bank.words() != 2048 * width {
            return Err(Error::InvalidConfig(
                "LFM3 shared QKV bank device or extent",
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_conv(&self, bank: BankRef<'_>) -> std::result::Result<(), Error> {
        super::conv_workspace::ConvWorkspace::validate(&self.pair.gpu, bank)
    }

    pub(crate) fn conv(
        &mut self,
        layer: usize,
        bank: BankRef<'_>,
        input: &[bf16],
        output: &mut [bf16],
        tokens: usize,
    ) -> std::result::Result<(), Error> {
        let tile = self
            .projection_tiles
            .as_mut()
            .map(|tiles| tiles.launch(0, layer, bank, tokens, None))
            .transpose()?
            .flatten();
        let tiled = tile.is_some();
        let census = self
            .conv
            .as_mut()
            .ok_or(Error::InvalidConfig("LFM3 missing shared ConvIn workspace"))?
            .project(&self.pair.gpu, bank, input, output, tokens, tile)?;
        if let Some(tiles) = &mut self.projection_tiles {
            tiles.completed(0, layer, tokens, tiled, census);
        }
        Ok(())
    }

    pub(crate) fn conv_stats(&self) -> Option<(u64, u64, [f64; 2])> {
        self.conv
            .as_ref()
            .map(super::conv_workspace::ConvWorkspace::telemetry)
    }

    pub(crate) fn tile_stats(&self) -> Option<serde_json::Value> {
        self.domain.as_ref()?.t4.as_ref().map(|tile| {
            let shapes = ["t128", "t1", "tail"]
                .into_iter()
                .zip(&tile.shapes)
                .map(|(shape, s)| {
                    let quad_calls = if shape == "t1" {
                        (s.output_rows - s.groups) / (TAG_N / 4 * 3) as u64
                    } else { 0 };
                    let tiled_groups = s.tiled * (TAG_T * TAG_N / tile.tile_size) as u64;
                    let quartet_groups = quad_calls * (TAG_N / 4) as u64;
                    let mut stats = serde_json::json!({
                        "shape": shape, "calls": s.calls, "tiled_calls": s.tiled,
                        "scalar_calls": s.calls - s.tiled - quad_calls, "projection_groups_per_role": s.groups,
                        "output_rows_per_role": s.output_rows, "native_by_role": s.native,
                        "ordered_by_role": s.native.map(|n| s.output_rows - n),
                        "homogeneous_panels_by_role": if shape == "t1" { [0; 2] } else { s.homogeneous },
                        "mixed_panels_by_role": if shape == "t1" { [0; 2] } else { s.homogeneous.map(|n| tiled_groups - n) },
                        "completed_gpu_seconds": s.clocks[0], "commit_wait_seconds": s.clocks[1],
                    });
                    if shape == "t1" {
                        stats["fma4_calls"] = serde_json::json!(quad_calls);
                        stats["fma4_quartets_by_role"] = serde_json::json!(s.homogeneous);
                        stats["fallback_quartets_by_role"] = serde_json::json!(s.homogeneous.map(|n| quartet_groups - n));
                    }
                    stats
                })
                .collect::<Vec<_>>();
            serde_json::json!({ "mode": "panel84",
                               "tile_simd_width": tile.pipeline.projection.threadExecutionWidth(),
                               "shapes": shapes })
        })
    }

    pub(crate) fn telemetry(&self) -> ([u64; 2], [u64; 2], [[f64; 2]; 2]) {
        (
            [self.pair.submitted, self.down.submissions()],
            self.completed,
            self.clocks,
        )
    }
}

impl SharedMlpWorkspace {
    pub(crate) fn seal_output_tags(
        &mut self,
        layer: usize,
        bank: BankRef<'_>,
        data: &[bf16],
    ) -> std::result::Result<(), Error> {
        self.output
            .as_mut()
            .ok_or(Error::InvalidConfig("LFM3 output workspace"))?
            .seal_tags(&self.pair.gpu, layer, bank, data)
    }

    pub(crate) fn output(
        &mut self,
        layer: usize,
        role: usize,
        bank: BankRef<'_>,
        input: &[bf16],
        output: &mut [bf16],
        count: usize,
    ) -> std::result::Result<(), Error> {
        let panel = self
            .qkv
            .as_ref()
            .and_then(super::qkv_workspace::Workspace::panel);
        self.output
            .as_mut()
            .ok_or(Error::InvalidConfig("LFM3 output workspace"))?
            .project(
                &self.pair.gpu,
                bank,
                (layer, role, panel),
                input,
                output,
                count,
            )?;
        Ok(())
    }

    pub(crate) fn output_stats(&self, published: [u64; 2]) -> Option<serde_json::Value> {
        self.output.as_ref().map(|output| output.stats(published))
    }
}
