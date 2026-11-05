//! Qwen-only profile selection and historical public compatibility spelling.
use super::{Manifest, qwen2_manifest, qwen3_manifest, qwen15_manifest};
use crate::weights::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QwenProfile {
    Qwen2,
    Qwen15,
    Qwen3,
}
impl QwenProfile {
    pub(crate) fn manifest(self) -> Result<Manifest<'static>, Error> {
        match self {
            Self::Qwen2 => qwen2_manifest(),
            Self::Qwen15 => qwen15_manifest(),
            Self::Qwen3 => qwen3_manifest(),
        }
    }
}

/// Compatibility spelling for the historical source receipt. The type name
/// is not Qwen admission: exact manifests and filtered source accessors decide
/// profile membership. No model assembly or codec pairing is implied.
pub type QwenSourceContentIdentity = super::dense::DenseSourceContentIdentity;
