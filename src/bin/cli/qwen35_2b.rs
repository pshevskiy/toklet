//! Explicit hybrid raw-ID diagnostic; no tokenizer or ordinary model admission.
use super::invalid;
use std::io;

pub(super) fn sha256(text: &str) -> io::Result<[u8; 32]> {
    if text.len() != 64
        || !text
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(invalid(
            "SHA256 must contain exactly 64 lowercase hexadecimal digits",
        ));
    }
    let mut result = [0; 32];
    for (out, pair) in result.iter_mut().zip(text.as_bytes().chunks_exact(2)) {
        let digit = |c: u8| if c <= b'9' { c - b'0' } else { c - b'a' + 10 };
        *out = digit(pair[0]) * 16 + digit(pair[1]);
    }
    Ok(result)
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub(super) use native::{Inputs, run};

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod native {
    use super::super::*;
    use std::{
        fs::{self, File, Metadata, OpenOptions},
        os::unix::fs::{MetadataExt, OpenOptionsExt},
        path::Path,
    };
    use toklet::{
        metal::gemma_affine::{CONTEXT_ALLOWANCE_BYTES, Context},
        mlx::qwen35::{ModalityPolicy, two_b},
        model::qwen35::{GenerationSession, MetalModel},
    };

    pub(in crate::cli) struct Inputs<'a> {
        pub directory: &'a Path,
        pub manifest: &'a Path,
        pub manifest_sha256: [u8; 32],
        pub capacity: usize,
        pub chunk: usize,
        pub budget: usize,
    }
    type Identity = (u64, u64, u64, i64, i64, i64, i64, u32, u64);
    fn identity(m: &Metadata) -> Identity {
        (
            m.dev(),
            m.ino(),
            m.len(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec(),
            m.mode(),
            m.nlink(),
        )
    }
    fn open(path: &Path, limit: usize) -> io::Result<(File, Metadata)> {
        let before = fs::symlink_metadata(path)?;
        if !before.is_file() || before.len() == 0 || before.len() > limit as u64 {
            return Err(invalid(
                "Diagnostic input must be a nonempty bounded regular file",
            ));
        }
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)?;
        verify(path, &file, &before)?;
        Ok((file, before))
    }
    fn verify(path: &Path, file: &File, before: &Metadata) -> io::Result<()> {
        if identity(&fs::symlink_metadata(path)?) != identity(before)
            || identity(&file.metadata()?) != identity(before)
        {
            return Err(invalid("Diagnostic input descriptor/path changed"));
        }
        Ok(())
    }
    fn read(path: &Path, limit: usize) -> io::Result<Vec<u8>> {
        let (mut file, before) = open(path, limit)?;
        let mut bytes = Vec::new();
        Read::by_ref(&mut file)
            .take((limit + 1) as u64)
            .read_to_end(&mut bytes)?;
        verify(path, &file, &before)?;
        if bytes.len() as u64 != before.len() || bytes.len() > limit {
            return Err(invalid("Diagnostic input length changed"));
        }
        Ok(bytes)
    }

    pub(in crate::cli) fn run(
        inputs: Inputs<'_>,
        mut frontend: Frontend,
        initial: Option<UserRequest>,
    ) -> Result<u8, Box<dyn std::error::Error>> {
        if frontend.stops.iter().any(|&id| id >= 248_320) {
            return Err(invalid("Stop token is outside the2B model vocabulary").into());
        }
        if let Some(request) = &initial {
            let tokens = request
                .tokens
                .as_deref()
                .ok_or_else(|| invalid("2B diagnostic requires token IDs"))?;
            let max = request.max_new_tokens.unwrap_or(frontend.max);
            if tokens.is_empty()
                || tokens.iter().any(|&id| id >= 248_320)
                || tokens
                    .len()
                    .checked_add(max)
                    .is_none_or(|n| n > inputs.capacity)
            {
                return Err(
                    invalid("2B token IDs or prompt plus output exceed admitted bounds").into(),
                );
            }
        }
        let total = Instant::now();
        let manifest = read(inputs.manifest, 64 * 1024)?;
        let pins = two_b::artifact::pins(&manifest, inputs.manifest_sha256)?;
        let config = read(&inputs.directory.join("config.json"), pins.config_bytes)?;
        let model_path = inputs.directory.join("model.safetensors");
        let (mut input, before) = open(&model_path, usize::try_from(pins.container_bytes)?)?;
        if before.len() != pins.container_bytes {
            return Err(invalid("Child container length differs").into());
        }
        let start = Instant::now();
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES)?;
        let context_seconds = start.elapsed().as_secs_f64();
        let load_budget = inputs
            .budget
            .checked_sub(CONTEXT_ALLOWANCE_BYTES)
            .ok_or_else(|| invalid("Context allowance exceeds diagnostic budget"))?;
        let start = Instant::now();
        let store = two_b::payload::load(
            &config,
            &mut input,
            two_b::payload::LoadOptions {
                memory_limit_bytes: load_budget,
                modality: ModalityPolicy::TextOnly,
            },
            &pins,
        )?;
        verify(&model_path, &input, &before)?;
        let authenticated_load_seconds = start.elapsed().as_secs_f64();
        let r = store.report();
        let cpu_load = json!({"source_payload_bytes":r.source_payload_bytes,
            "packed_matrices":r.packed_matrices,"fp32_vectors":r.fp32_vectors,
            "fp32_convolutions":r.fp32_convolutions,
            "retained_tensor_capacity_bytes":r.retained_tensor_capacity_bytes,
            "metadata_reservation_bytes":r.metadata_reservation_bytes,
            "admitted_load_bytes":r.admitted_load_bytes});
        let start = Instant::now();
        let store = store.into_metal(&context, inputs.budget)?;
        let upload_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let model = MetalModel::from_two_b_metal_store(&store)?;
        let binding_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let numerical = model.session(
            &context,
            toklet::model::qwen35::Options {
                capacity: inputs.capacity,
                chunk: inputs.chunk,
                position_offset: 0,
                memory_limit_bytes: inputs.budget,
            },
        )?;
        let mut session = GenerationSession::new(numerical, inputs.budget)?;
        let session_initialization_seconds = start.elapsed().as_secs_f64();
        frontend.load_seconds = total.elapsed().as_secs_f64();
        frontend.execution = ExecutionMetadata {
            weight_dtype: "MLX affine4/G64 BF16 metadata; BF16 small weights + original F32 A_log",
            kv_dtype: "FP32",
            projection_mode: "MlxAffineQ4G64Gpu",
            normalization_mode: "CpuReferenceFp64ReductionFp32Store",
            readback_mode: "Fp32ProjectionRowsAndFullLogits",
            attention_mode: Some("CpuReferenceGatedGqa"),
            prefill_chunk_size: inputs.chunk,
            packed_format_version: None,
            quantization_group_size: Some(64),
            quantizer: Some("MLX affine4"),
            state_backend: Some("CPU; GPU projections/gather only"),
            state_dtype: Some("FP32 recurrent/convolution/KV; FP64 reference reductions"),
        };
        frontend.cpu_projection = "none";
        frontend.qwen35_2b_load = Some(json!({
            "artifact_manifest_sha256": inputs.manifest_sha256.iter().map(|b|format!("{b:02x}")).collect::<String>(),
            "manifest_binding_scope":"caller-selected bytes; not independent lineage/certificate authentication",
            "context_initialization_seconds":context_seconds,
            "authenticated_cpu_load_seconds":authenticated_load_seconds,
            "gpu_upload_seconds":upload_seconds,"binding_seconds":binding_seconds,
            "session_initialization_seconds":session_initialization_seconds,
            "compilation_seconds":Value::Null,"offline_conversion_seconds":Value::Null,
            "conversion_scope":"child conversion occurred offline; runtime conversion_seconds=0",
            "cpu_load":cpu_load,"gpu_upload":store.report(),
            "poison_recovery":"explicit repair API or CLI restart; reset does not repair poisoned execution",
            "quality_performance_adoption_qualified":false,
        }));
        let external = frontend.external_bytes;
        run_session(
            frontend,
            &mut session,
            initial,
            inputs.capacity,
            external,
            |s| {
                json!({
                    "memory":s.memory_report(),"generation_state":s.state(),
                    "work_scope":"generation_state.last_numerical_work is the last numerical call, not request-total work",
                    "requires_execution_repair":s.requires_execution_repair(),
                })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn manifest_hash_requires_exact_lowercase_hex() {
        assert_eq!(sha256(&"ab".repeat(32)).unwrap(), [0xab; 32]);
        for value in [
            "",
            "00",
            &"A".repeat(64),
            &"g".repeat(64),
            &"0".repeat(63),
            &"0".repeat(65),
        ] {
            assert!(sha256(value).is_err());
        }
    }
}
