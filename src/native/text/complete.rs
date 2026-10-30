//! Cold selection from actual assembled owners; no directory/profile guessing.
use super::*;
use crate::mlx::qwen::{QwenSourceContentIdentity, SourceContentIdentity};

#[derive(Clone, Copy)]
pub(super) enum SourceReceipt {
    Smol(SourceContentIdentity),
    Qwen3(QwenSourceContentIdentity),
}
impl SourceReceipt {
    pub(super) fn profile(self) -> Profile {
        match self {
            Self::Smol(_) => Profile::Smol,
            Self::Qwen3(_) => Profile::Qwen3,
        }
    }
    pub(super) fn model(model: &ModelHandle) -> Result<(Self, usize)> {
        let receipts: (
            Option<SourceContentIdentity>,
            Option<QwenSourceContentIdentity>,
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
                value.mlx_qwen3_source_content_identity().copied(),
                value.config().max_positions,
            ),
        };
        let source = match (receipts.0, receipts.1) {
            (Some(s), None) => Self::Smol(s),
            (None, Some(s)) => Self::Qwen3(s),
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
                Self::Qwen3(_) => crate::mlx::dense::Architecture::Qwen3,
            };
            if !source.profile().supports(value.config())
                || value.native_session_modes()
                    != (ProjectionMode::MlxAffine, NormalizationMode::Compensated)
                || value.native_attention_mode() != AttentionMode::Reference
            {
                return Err(fail(
                    UNSUPPORTED,
                    "unsupported complete identity execution policy",
                ));
            }
            crate::mlx::execution::metal_geometry(value.config(), architecture)
                .map_err(|e| fail(UNSUPPORTED, e.message()))?;
        }
        Ok((source, receipts.2))
    }
    fn codec(self, directory: &Path, allowance: usize) -> Result<Codec> {
        match self {
            Self::Smol(s) => Codec::load_smol_complete_audited(directory, &s, allowance),
            Self::Qwen3(s) => Codec::load_qwen3_audited(directory, &s, allowance),
        }
        .map_err(|e| fail(LOAD_ERROR, e.to_string()))
    }
}

/// Construct a complete audited codec from the actual attached Smol/Qwen3
/// model evidence. No caller-selected profile/path is authority. Reserve a
/// separate profile text allowance before loading the model (Smol64 MiB,
/// Qwen3 256 MiB; at most6 GiB); combined process/device fit stays separate.
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
