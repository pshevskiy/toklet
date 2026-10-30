//! Cold selection from actual assembled owners; no directory/profile guessing.
use super::*;
use crate::mlx::qwen::{DenseSourceContentIdentity, SourceContentIdentity};

#[derive(Clone, Copy)]
pub(super) enum SourceReceipt {
    Smol(SourceContentIdentity),
    Qwen2(DenseSourceContentIdentity),
    Qwen3(DenseSourceContentIdentity),
    Mistral(DenseSourceContentIdentity),
}
impl SourceReceipt {
    pub(super) fn profile(self) -> Profile {
        match self {
            Self::Smol(_) => Profile::Smol,
            Self::Qwen2(_) => Profile::Qwen,
            Self::Qwen3(_) => Profile::Qwen3,
            Self::Mistral(_) => Profile::Mistral,
        }
    }
    pub(super) fn model(model: &ModelHandle) -> Result<(Self, usize)> {
        let (smol, dense, loaded): (
            Option<SourceContentIdentity>,
            Option<DenseSourceContentIdentity>,
            usize,
        ) = match model.model.as_ref() {
            NativeModel::Cpu(_) => (None, None, 0),
            #[cfg(all(
                feature = "metal",
                any(target_os = "macos", target_os = "ios"),
                target_arch = "aarch64"
            ))]
            NativeModel::Metal(value) => (
                value.mlx_source_content_identity().copied(),
                value.mlx_dense_source_content_identity().copied(),
                value.config().max_positions,
            ),
        };
        let source = match (smol, dense) {
            (Some(s), None) => Self::Smol(s),
            (None, Some(s)) => Self::dense(s)?,
            _ => {
                return Err(fail(
                    UNSUPPORTED,
                    "complete chat requires one audited assembled source",
                ));
            }
        };
        #[cfg(all(
            feature = "metal",
            any(target_os = "macos", target_os = "ios"),
            target_arch = "aarch64"
        ))]
        if let NativeModel::Metal(value) = model.model.as_ref() {
            use crate::metal::{AttentionMode, NormalizationMode, ProjectionMode};
            let architecture = match source {
                Self::Smol(_) => crate::mlx::dense::Architecture::Llama,
                Self::Qwen2(_) => crate::mlx::dense::Architecture::Qwen2,
                Self::Qwen3(_) => crate::mlx::dense::Architecture::Qwen3,
                Self::Mistral(_) => crate::mlx::dense::Architecture::Mistral,
            };
            if !source.profile().supports(value.config())
                || value.native_session_modes()
                    != (ProjectionMode::MlxAffine, NormalizationMode::Compensated)
                || value.native_attention_mode()
                    != if matches!(source, Self::Qwen2(_)) {
                        AttentionMode::Online
                    } else {
                        AttentionMode::Reference
                    }
            {
                return Err(fail(
                    UNSUPPORTED,
                    "unsupported complete identity execution policy",
                ));
            }
            crate::mlx::execution::metal_geometry(value.config(), architecture)
                .map_err(|e| fail(UNSUPPORTED, e.message()))?;
        }
        Ok((source, loaded))
    }
    fn dense(source: DenseSourceContentIdentity) -> Result<Self> {
        use crate::weights::content::{DenseProfile, QwenProfile};
        for profile in [
            DenseProfile::Qwen(QwenProfile::Qwen2),
            DenseProfile::Qwen(QwenProfile::Qwen3),
            DenseProfile::Mistral,
        ] {
            let manifest = profile
                .manifest()
                .map_err(|e| fail(ENGINE_ERROR, e.to_string()))?;
            if source.expected_manifest_sha256() == manifest.identity() {
                return Ok(match profile {
                    DenseProfile::Qwen(QwenProfile::Qwen2) => Self::Qwen2(source),
                    DenseProfile::Qwen(QwenProfile::Qwen3) => Self::Qwen3(source),
                    DenseProfile::Mistral => Self::Mistral(source),
                });
            }
        }
        Err(fail(
            UNSUPPORTED,
            "unsupported assembled dense source identity",
        ))
    }
    fn codec(self, directory: &Path, allowance: usize) -> Result<Codec> {
        match self {
            Self::Smol(s) => Codec::load_smol_complete_audited(directory, &s, allowance),
            Self::Qwen2(s) => Codec::load_qwen2_audited(directory, &s, allowance),
            Self::Qwen3(s) => Codec::load_qwen3_audited(directory, &s, allowance),
            Self::Mistral(s) => Codec::load_mistral_audited(directory, &s, allowance),
        }
        .map_err(|e| fail(LOAD_ERROR, e.to_string()))
    }
}

/// Construct a complete audited codec from the actual attached Smol/Qwen2/Qwen3/Mistral
/// model evidence. No caller-selected profile/path is authority. Reserve a
/// separate profile text allowance before loading the model (Smol/Mistral64 MiB,
/// Qwen2 128 MiB, Qwen3 256 MiB; at most6 GiB); combined process/device fit stays separate.
/// Legacy factories retain their existing contracts. A missing ancillary asset
/// fails this new route, and no partial codec or additional Rc is published.
/// # Safety
/// Follow the parent live-pointer, creating-thread and disjoint-output contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_chat_create_complete_audited(
    model: *const ModelHandle,
    allowance: u64,
    out: *mut *mut ChatHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(out)?;
            out.write(ptr::null_mut());
            pointer(model)?;
            let model = &*model;
            owner(model.owner)?;
            if allowance < crate::app_chat::TOKENIZER_ALLOWANCE as u64
                || allowance > weights::INITIAL_MEMORY_LIMIT_BYTES as u64
            {
                return Err(fail(
                    INVALID,
                    "complete text allowance must be 64 MiB..=6 GiB",
                ));
            }
            let (source, _) = SourceReceipt::model(model)?;
            if allowance < source.profile().allowance() as u64 {
                return Err(fail(
                    INVALID,
                    "complete text allowance is below model profile policy",
                ));
            }
            let codec = source.codec(&model.directory, allowance as usize)?;
            out.write(Box::into_raw(Box::new(ChatHandle {
                owner: thread::current().id(),
                codec,
                _model: Some(Rc::clone(&model.model)),
            })));
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod metal_tests;

#[cfg(all(test, feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[path = "complete/qwen2_metal_tests.rs"]
mod qwen2_metal_tests;

#[cfg(all(test, feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[path = "complete/mistral_metal_tests.rs"]
mod mistral_metal_tests;
