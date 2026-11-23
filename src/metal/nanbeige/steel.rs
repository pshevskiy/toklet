//! Fixed one-shot Nanbeige prefill and decode with the pinned MLX wheel.
use super::{
    Error, HIDDEN, INTERMEDIATE, KV_WIDTH, LOGIT_ROWS, Model, Options, QUERY_WIDTH, Result, Sizes,
    add,
    dense::Prepacked,
    kernel::{Encoder, RawBuffer, encode_pipeline},
    plan::{HEAD_DIM, KV_HEADS, Matrix, QUERY_HEADS, VISITS},
    qmm,
    split_k::{PARTIAL_BYTES, ROWS, SplitK},
};
use dispatch2::DispatchData;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use sha2::{Digest, Sha256};
use std::{
    fs::OpenOptions,
    io::Read,
    mem::{align_of, offset_of, size_of},
    os::unix::fs::OpenOptionsExt,
    path::Path,
    ptr::NonNull,
};

const EXPORT: &str = "steel_attention_bfloat16_bq32_bk16_bd128_bv128_wm4_wn1_maskbfloat16";
const RMS_EXPORT: &str = "rmsbfloat16";
const VECTOR_EXPORT: &str = "sdpa_vector_bfloat16_t_128_128";
const VECTOR_THREADS: usize = 1024;
const ROTARY_THREADS: usize = 32;
const ROTARY_GROUPS_PER_ROW: usize = (QUERY_WIDTH + KV_WIDTH) / 2 / ROTARY_THREADS;
const _: () = assert!(ROTARY_GROUPS_PER_ROW * ROTARY_THREADS * 2 == QUERY_WIDTH + KV_WIDTH);
const VECTOR_POSITIONS: std::ops::RangeInclusive<usize> = 512..=542;
const HEAD_EXPORT: &str = "affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0_pr_0_hgs_0_r_4";
const HEAD_INPUT_OFFSET: usize = (ROWS - 1) * HIDDEN * 2;
const HEAD_SIMD_GROUPS: usize = 2;
const _: () = {
    assert!(HIDDEN == 3072 && LOGIT_ROWS == 166144);
    assert!(HEAD_INPUT_OFFSET == 780288);
    assert!(HEAD_INPUT_OFFSET.is_multiple_of(16));
    assert!(HIDDEN.is_multiple_of(512) && LOGIT_ROWS.is_multiple_of(8));
};
const RMS_THREADS: usize = 768;
const RMS_BYTES: usize = ROWS * HIDDEN * 2;
const RMS_GAIN_BYTES: usize = HIDDEN * 2;
pub(super) const PROMPT: usize = 512;
pub(super) const OUTPUT: usize = 32;
const CAPACITY: usize = PROMPT + OUTPUT;
const IMAGE_BYTES: usize = 214_156_448;
const IMAGE_SHA: &str = "c6d7a3facb83568f6e4cb2cf4a8c736ee29302571a48343cb9a3a4d1cd3e3893";
const KV_HEAD_WORDS: usize = CAPACITY * HEAD_DIM;
const SLOT_BYTES: usize = KV_HEADS * KV_HEAD_WORDS * 2;
const SCALE: f32 = 0.088_388_346;
const SPECIALIZATION: [(usize, bool); 6] = [
    (200, true),
    (201, true),
    (300, false),
    (301, true),
    (302, false),
    (303, false),
];

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub(super) struct Steel {
    rotate: Pipeline,
    attention: Pipeline,
    vector_attention: Pipeline,
    rms: Pipeline,
    head: Pipeline,
    swiglu: Pipeline,
    pub split: SplitK,
    pub prepacked: Prepacked,
    pub qmm_bm64: Pipeline,
    pub reserved_bytes: usize,
    pub physical_bytes: usize,
}

#[repr(C)]
struct Params {
    batch: i32,
    heads: i32,
    dims: i32,
    queries: i32,
    keys: i32,
    repeats: i32,
    scale: f32,
    query_blocks: i32,
    key_blocks: i32,
    aligned_query_blocks: i32,
    aligned_key_blocks: i32,
    query_remainder: i32,
    key_remainder: i32,
    query_offset: i32,
    query_strides: [i64; 3],
    key_strides: [i64; 3],
    value_strides: [i64; 3],
    output_strides: [i64; 3],
}

const _: () = {
    assert!(size_of::<Params>() == 152);
    assert!(align_of::<Params>() == 8);
    assert!(offset_of!(Params, scale) == 24);
    assert!(offset_of!(Params, query_offset) == 52);
    assert!(offset_of!(Params, query_strides) == 56);
    assert!(offset_of!(Params, key_strides) == 80);
    assert!(offset_of!(Params, value_strides) == 104);
    assert!(offset_of!(Params, output_strides) == 128);
};

fn wheel_image(path: &Path) -> Result<Box<[u8]>> {
    let io_error = |error: std::io::Error| Error::Native(format!("Nanbeige Steel wheel: {error}"));
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(io_error)?;
    let metadata = file.metadata().map_err(io_error)?;
    if !metadata.is_file() || metadata.len() != IMAGE_BYTES as u64 {
        return Err(Error::Unsupported("Nanbeige Steel exact MLX image extent"));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(IMAGE_BYTES)
        .map_err(|_| Error::Native("Nanbeige Steel wheel allocation failed".into()))?;
    bytes.resize(IMAGE_BYTES, 0);
    file.read_exact(&mut bytes).map_err(io_error)?;
    if file.read(&mut [0]).map_err(io_error)? != 0
        || format!("{:x}", Sha256::digest(&bytes)) != IMAGE_SHA
    {
        return Err(Error::Unsupported("Nanbeige Steel exact MLX image digest"));
    }
    Ok(bytes.into_boxed_slice())
}

impl Steel {
    pub fn validate_options(options: Options) -> Result<()> {
        if options.prefill_chunk_size != ROWS
            || options.capacity != CAPACITY
            || options.position_offset != 0
            || HIDDEN != RMS_THREADS * 4
            || (QUERY_HEADS, KV_HEADS, HEAD_DIM, VISITS) != (48, 8, 128, 44)
        {
            return Err(Error::Unsupported("Nanbeige Steel T128 geometry"));
        }
        Ok(())
    }

    pub fn validate_banks(banks: [&RawBuffer; 7]) -> Result<()> {
        let [query, key, value, cache_k, cache_v, output, status] = banks;
        if !qmm::disjoint_exact_ranges([
            (query, ROWS * QUERY_WIDTH * 2),
            (key, ROWS * KV_WIDTH * 2),
            (value, ROWS * KV_WIDTH * 2),
            (cache_k, SLOT_BYTES * VISITS),
            (cache_v, SLOT_BYTES * VISITS),
            (output, ROWS * QUERY_WIDTH * 2),
            (status, 4),
        ]) {
            return Err(Error::Unsupported("Nanbeige Steel T128 buffer layout"));
        }
        Ok(())
    }

    pub fn validate_rms(model: &Model, banks: [&RawBuffer; 3]) -> Result<()> {
        let [input, output, status] = banks;
        let gains = std::iter::once(model.plan.final_norm).chain(
            model
                .plan
                .blocks
                .iter()
                .flat_map(|block| [block.input_norm, block.post_attention_norm]),
        );
        for gain in gains {
            if !qmm::disjoint_exact_ranges([
                (input, RMS_BYTES),
                (&model.weights[gain].raw, RMS_GAIN_BYTES),
                (output, RMS_BYTES),
                (status, 4),
            ]) {
                return Err(Error::Unsupported("Nanbeige Steel RMSNorm buffer layout"));
            }
        }
        Ok(())
    }

    pub fn validate_head(model: &Model, banks: [&RawBuffer; 3]) -> Result<()> {
        let [input, output, status] = banks;
        let head = model.plan.head;
        let layout = head.layout;
        if (
            layout.columns(),
            layout.rows(),
            layout.bits(),
            layout.group_size(),
        ) != (HIDDEN, LOGIT_ROWS, 4, 64)
            || layout.metadata_dtype() != crate::mlx::affine::MetadataDtype::BF16
            || !qmm::disjoint_exact_ranges([
                (&model.weights[head.weights].raw, layout.weight_bytes()),
                (&model.weights[head.scales].raw, layout.metadata_bytes()),
                (&model.weights[head.biases].raw, layout.metadata_bytes()),
                (input, RMS_BYTES),
                (output, LOGIT_ROWS * 2),
                (status, 4),
            ])
        {
            return Err(Error::Unsupported(
                "Nanbeige Steel selective vocabulary head geometry",
            ));
        }
        Ok(())
    }

    pub fn validate_t1_affine(
        model: &Model,
        matrix: Matrix,
        [input, output, status]: [&RawBuffer; 3],
        output_rows: usize,
    ) -> Result<()> {
        let layout = matrix.layout;
        let (columns, outputs) = (layout.columns(), layout.rows());
        if !matches!(columns, HIDDEN | INTERMEDIATE | QUERY_WIDTH)
            || !columns.is_multiple_of(512)
            || !outputs.is_multiple_of(8)
            || (layout.bits(), layout.group_size()) != (4, 64)
            || layout.metadata_dtype() != crate::mlx::affine::MetadataDtype::BF16
            || !qmm::disjoint_exact_ranges([
                (&model.weights[matrix.weights].raw, layout.weight_bytes()),
                (&model.weights[matrix.scales].raw, layout.metadata_bytes()),
                (&model.weights[matrix.biases].raw, layout.metadata_bytes()),
                (input, ROWS * columns * 2),
                (output, output_rows * outputs * 2),
                (status, 4),
            ])
        {
            return Err(Error::Unsupported("Nanbeige Steel T1 affine geometry"));
        }
        Ok(())
    }

    pub fn prepare(model: &Model, options: Options, path: &Path) -> Result<Self> {
        Self::validate_options(options)?;
        let dense = model.dense.as_ref().ok_or_else(|| {
            Error::Native(format!(
                "Nanbeige Steel requires dense prefill: {}",
                model.dense_prefill_denial().unwrap_or("unavailable")
            ))
        })?;
        let qmm = model.qmm.as_ref().ok_or_else(|| {
            Error::Native(format!(
                "Nanbeige Steel requires the pinned QMM route: {}",
                model.qmm_denial().unwrap_or("unavailable")
            ))
        })?;
        let device = &model.context.device;
        let sizes = Sizes::new(options, false)?;
        let before = device.currentAllocatedSize();
        let logical_base = add(
            sizes.memory.admitted_bytes,
            add(dense.receipt.allocated_bytes, qmm.receipt.reserved_bytes)?,
        )?;
        let resident_base = add(
            before,
            add(
                sizes.memory.admitted_bytes - sizes.memory.weights_bytes,
                qmm.receipt.reserved_bytes - qmm.receipt.physical_delta_bytes,
            )?,
        )?;
        let device_limit = usize::try_from(device.recommendedMaxWorkingSetSize())
            .unwrap_or(usize::MAX)
            .min(options.memory_limit_bytes);
        let budget = |reserve| {
            for (base, limit) in [
                (logical_base, options.memory_limit_bytes),
                (resident_base, device_limit),
            ] {
                let required = add(base, add(PARTIAL_BYTES, reserve)?)?;
                if required > limit {
                    return Err(Error::Budget { required, limit });
                }
            }
            Ok(())
        };

        // The authenticated data itself is submitted to Metal. Budget one copy
        // for the input and one for Metal's image during pipeline preparation.
        budget(2 * IMAGE_BYTES)?;
        let image = DispatchData::from_boxed(wheel_image(path)?);
        let library = device
            .newLibraryWithData_error(&image)
            .map_err(|error| Error::Native(error.to_string()))?;
        let split = SplitK::prepare(model, &library)?;
        let constants = MTLFunctionConstantValues::new();
        for (index, value) in SPECIALIZATION {
            let value = u8::from(value);
            // SAFETY: Metal copies the one-byte MSL bool at the known constant index.
            unsafe {
                constants.setConstantValue_type_atIndex(
                    NonNull::from(&value).cast(),
                    MTLDataType::Bool,
                    index,
                );
            }
        }
        let function = library
            .newFunctionWithName_constantValues_error(&NSString::from_str(EXPORT), &constants)
            .map_err(|error| Error::Native(error.to_string()))?;
        let attention = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| Error::Native(error.to_string()))?;
        let constants = MTLFunctionConstantValues::new();
        for index in 20..=25 {
            let value = 0_u8;
            // SAFETY: the pinned vector SDPA export uses copied MSL bools at these indices.
            unsafe {
                constants.setConstantValue_type_atIndex(
                    NonNull::from(&value).cast(),
                    MTLDataType::Bool,
                    index,
                );
            }
        }
        let function = library
            .newFunctionWithName_constantValues_error(
                &NSString::from_str(VECTOR_EXPORT),
                &constants,
            )
            .map_err(|error| Error::Native(error.to_string()))?;
        let vector_attention = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| Error::Native(error.to_string()))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(RMS_EXPORT))
            .ok_or(Error::Unsupported("Nanbeige Steel RMSNorm exact export"))?;
        let rms = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| Error::Native(error.to_string()))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(HEAD_EXPORT))
            .ok_or(Error::Unsupported(
                "Nanbeige Steel selective head exact export",
            ))?;
        let head = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| Error::Native(error.to_string()))?;
        let library = super::super::mlx::precise_library_on(
            device,
            concat!(
                include_str!("../nanbeige.metal"),
                "\n",
                include_str!("steel_rotary.metal"),
                "\n",
                include_str!("steel_swiglu.metal"),
                "\n",
                include_str!("dense_prepacked.metal")
            ),
        )?;
        let local_pipeline = |name: &str| -> Result<Pipeline> {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or(Error::Unsupported("Nanbeige Steel local kernel export"))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| Error::Native(error.to_string()))
        };
        let rotate = local_pipeline("nanbeige_rotate_cache_steel")?;
        let swiglu = local_pipeline("nanbeige_swiglu_steel")?;
        let prepacked = Prepacked::new(
            local_pipeline("nanbeige_pack_dense_gate_up_input")?,
            local_pipeline("nanbeige_ffn_fused_gate_up_swiglu_prepacked_register_reuse_k64")?,
        )?;
        for (pipeline, threads) in [
            (&attention, 128),
            (&vector_attention, VECTOR_THREADS),
            (&rotate, ROTARY_THREADS),
            (&swiglu, 32),
            (&rms, RMS_THREADS),
            (&head, 32 * HEAD_SIMD_GROUPS),
        ] {
            if pipeline.threadExecutionWidth() != 32
                || pipeline.maxTotalThreadsPerThreadgroup() < threads
                || pipeline.staticThreadgroupMemoryLength() > device.maxThreadgroupMemoryLength()
            {
                return Err(Error::Unsupported("Nanbeige Steel wheel pipeline geometry"));
            }
        }
        const BM64_IMAGE: &[u8; 13484] =
            include_bytes!("../../../assets/nanbeige-qmm/qmm_bm64.metallib");
        const BM64_SHA: &str = "941137a5e36e1dbe293f3c1656b6e15be47e91f841489b4c01d5122bcb5f87f2";
        budget(add(
            IMAGE_BYTES,
            add(
                device.currentAllocatedSize().saturating_sub(before),
                2 * BM64_IMAGE.len(),
            )?,
        )?)?;
        if format!("{:x}", Sha256::digest(BM64_IMAGE)) != BM64_SHA {
            return Err(Error::Unsupported("Nanbeige BM64 exact sealed image"));
        }
        let data = DispatchData::from_boxed(BM64_IMAGE.to_vec().into_boxed_slice());
        let image = device
            .newLibraryWithData_error(&data)
            .map_err(|e| Error::Native(e.to_string()))?;
        if image.functionNames().count() != 1 {
            return Err(Error::Unsupported("Nanbeige BM64 single export"));
        }
        let export = image
            .newFunctionWithName(&NSString::from_str("toklet_qmm_bm64"))
            .ok_or(Error::Unsupported("Nanbeige BM64 exact export"))?;
        let qmm_bm64 = device
            .newComputePipelineStateWithFunction_error(&export)
            .map_err(|e| Error::Native(e.to_string()))?;
        if qmm_bm64.threadExecutionWidth() != 32
            || qmm_bm64.maxTotalThreadsPerThreadgroup() < 128
            || qmm_bm64.staticThreadgroupMemoryLength() != 7680
            || qmm_bm64.staticThreadgroupMemoryLength() > device.maxThreadgroupMemoryLength()
        {
            return Err(Error::Unsupported("Nanbeige BM64 exact geometry"));
        }
        let physical_bytes = device.currentAllocatedSize().saturating_sub(before);
        let reserved_bytes = add(add(IMAGE_BYTES, physical_bytes)?, BM64_IMAGE.len())?;
        budget(reserved_bytes)?;
        Ok(Self {
            rotate,
            attention,
            vector_attention,
            rms,
            head,
            swiglu,
            split,
            prepacked,
            qmm_bm64,
            reserved_bytes,
            physical_bytes,
        })
    }

    pub fn encode_decode_swiglu(&self, encoder: &Encoder, buffers: [&RawBuffer; 3]) {
        encode_pipeline(
            encoder,
            &self.swiglu,
            32,
            &buffers,
            &[1, INTERMEDIATE as u32],
            [INTERMEDIATE.div_ceil(32), 1],
        );
    }

    pub fn encode_prefill_head(&self, encoder: &Encoder, buffers: [&RawBuffer; 5]) {
        self.encode_affine(encoder, buffers, HIDDEN, LOGIT_ROWS, HEAD_INPUT_OFFSET);
    }

    pub fn encode_affine(
        &self,
        encoder: &Encoder,
        buffers: [&RawBuffer; 5],
        columns: usize,
        outputs: usize,
        input_offset: usize,
    ) {
        let groups = outputs / 8;
        let columns = columns as i32;
        let outputs = outputs as i32;
        encoder.setComputePipelineState(&self.head);
        // SAFETY: the authenticated fast QMV export takes W/scales/biases/X/Y
        // followed by two copied i32 constants. Admission checks exact aligned
        // independent allocation ranges and exact dense packed/metadata extents.
        // The active BF16 spans are one input and one output row. Decode uses
        // offset zero. The prefill head uses the last aligned row of its input.
        // The grid covers 8 outputs per group. The encoder retains the banks.
        unsafe {
            for (index, buffer) in buffers.into_iter().enumerate() {
                let offset = if index == 3 { input_offset } else { 0 };
                encoder.setBuffer_offset_atIndex(Some(buffer), offset, index);
            }
            encoder.setBytes_length_atIndex(NonNull::from(&columns).cast(), size_of::<i32>(), 5);
            encoder.setBytes_length_atIndex(NonNull::from(&outputs).cast(), size_of::<i32>(), 6);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            grid(1, groups),
            grid(32, HEAD_SIMD_GROUPS),
        );
    }

    pub fn encode_rms(&self, encoder: &Encoder, buffers: [&RawBuffer; 3], rows: usize) {
        debug_assert!(rows == 1 || rows == ROWS);
        let epsilon = 1e-5_f32;
        let axis = HIDDEN as u32;
        let gain_stride = 1_u32;
        encoder.setComputePipelineState(&self.rms);
        // SAFETY: the authenticated wheel ABI takes two BF16 input banks and
        // one BF16 output bank followed by the three copied scalar constants.
        // Session admission checks the exact extents and disjoint ranges.
        // A group covers one row at four contiguous words per thread. The
        // parent encoder owns all tracked banks through command completion.
        unsafe {
            for (index, buffer) in buffers.into_iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
            }
            encoder.setBytes_length_atIndex(NonNull::from(&epsilon).cast(), size_of::<f32>(), 3);
            encoder.setBytes_length_atIndex(NonNull::from(&axis).cast(), size_of::<u32>(), 4);
            encoder.setBytes_length_atIndex(
                NonNull::from(&gain_stride).cast(),
                size_of::<u32>(),
                5,
            );
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid(rows, 1), grid(RMS_THREADS, 1));
    }

    pub fn encode_rotary(
        &self,
        encoder: &Encoder,
        buffers: [&RawBuffer; 6],
        rows: usize,
        past: usize,
        slot: usize,
    ) {
        debug_assert!(rows == ROWS || (rows == 1 && Self::supports_t1_attention(past)));
        debug_assert!(past + rows <= CAPACITY && slot < VISITS);
        let shape = [rows as u32, past as u32, CAPACITY as u32, slot as u32, 0];
        encoder.setComputePipelineState(&self.rotate);
        // SAFETY: admission fixes exact, disjoint T128 scratch and cache banks.
        // The first row fits in the same banks for T1. Exactly one thread
        // handles each pair of Q/K words; the selected slot and position lie
        // within the admitted K/V banks. All resources outlive completion.
        unsafe {
            for (index, buffer) in buffers.into_iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
            }
            encoder.setBytes_length_atIndex(NonNull::from(&shape).cast(), size_of::<[u32; 5]>(), 6);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            grid(rows * ROTARY_GROUPS_PER_ROW, 1),
            grid(ROTARY_THREADS, 1),
        );
    }

    pub fn supports_t1_attention(position: usize) -> bool {
        VECTOR_POSITIONS.contains(&position)
    }

    pub fn encode_t1_attention(
        &self,
        encoder: &Encoder,
        buffers: [&RawBuffer; 4],
        position: usize,
        slot: usize,
    ) {
        debug_assert!(Self::supports_t1_attention(position) && slot < VISITS);
        let keys = (position + 1) as i32;
        let gqa_factor = (QUERY_HEADS / KV_HEADS) as i32;
        let slot_offset = slot * SLOT_BYTES;
        let head_stride = KV_HEAD_WORDS as u64;
        let sequence_stride = HEAD_DIM as u64;
        let scale = SCALE;
        encoder.setComputePipelineState(&self.vector_attention);
        // SAFETY: fixed session admission checks BF16 bank extents and disjoint,
        // aligned ranges. Each K/V binding starts at one of the 44 admitted
        // owner slots. Each head has CAPACITY rows and the kernel reads only
        // [0, keys), including this command's freshly appended position.
        // One contiguous Q/O row contains all 48 heads. The pinned ABI copies
        // the scalar arguments, and tracked banks live through completion.
        unsafe {
            for (index, buffer) in buffers.into_iter().enumerate() {
                let offset = if matches!(index, 1 | 2) {
                    slot_offset
                } else {
                    0
                };
                encoder.setBuffer_offset_atIndex(Some(buffer), offset, index);
            }
            encoder.setBytes_length_atIndex(NonNull::from(&gqa_factor).cast(), size_of::<i32>(), 4);
            encoder.setBytes_length_atIndex(NonNull::from(&keys).cast(), size_of::<i32>(), 5);
            for (index, stride) in [
                (6, &head_stride),
                (7, &sequence_stride),
                (8, &head_stride),
                (9, &sequence_stride),
            ] {
                encoder.setBytes_length_atIndex(
                    NonNull::from(stride).cast(),
                    size_of::<u64>(),
                    index,
                );
            }
            encoder.setBytes_length_atIndex(NonNull::from(&scale).cast(), size_of::<f32>(), 10);
        }
        // All visible keys precede or equal this single query's position.
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            grid(QUERY_HEADS, 1),
            grid(VECTOR_THREADS, 1),
        );
    }

    pub fn encode_attention(
        &self,
        encoder: &Encoder,
        buffers: [&RawBuffer; 4],
        past: usize,
        slot: usize,
    ) {
        let keys = (past + ROWS) as i32;
        let key_blocks = keys / 16;
        let q_strides = [
            (ROWS * QUERY_WIDTH) as i64,
            HEAD_DIM as i64,
            QUERY_WIDTH as i64,
        ];
        let kv_strides = [
            (SLOT_BYTES / 2) as i64,
            KV_HEAD_WORDS as i64,
            HEAD_DIM as i64,
        ];
        let params = Params {
            batch: 1,
            heads: QUERY_HEADS as i32,
            dims: HEAD_DIM as i32,
            queries: ROWS as i32,
            keys,
            repeats: (QUERY_HEADS / KV_HEADS) as i32,
            scale: SCALE,
            query_blocks: 4,
            key_blocks,
            aligned_query_blocks: 4,
            aligned_key_blocks: key_blocks,
            query_remainder: 0,
            key_remainder: 0,
            query_offset: past as i32,
            query_strides: q_strides,
            key_strides: kv_strides,
            value_strides: kv_strides,
            output_strides: q_strides,
        };
        encoder.setComputePipelineState(&self.attention);
        // SAFETY: bank extents are checked at admission and launch is confined
        // to the four full aligned cuts. The per-slot bases and head strides
        // leave all reads within that slot. Buffers are tracked through the
        // prior rotary/cache writes and subsequent finite scan in this encoder.
        unsafe {
            for (index, buffer) in buffers.into_iter().enumerate() {
                let offset = if index == 1 || index == 2 {
                    slot * SLOT_BYTES
                } else {
                    0
                };
                encoder.setBuffer_offset_atIndex(Some(buffer), offset, index);
            }
            encoder.setBytes_length_atIndex(NonNull::from(&params).cast(), size_of::<Params>(), 4);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid(4, QUERY_HEADS), grid(32, 4));
    }
}

fn grid(width: usize, height: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth: 1,
    }
}
