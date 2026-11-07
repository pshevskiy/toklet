//! Pinned Mac engine and basic-text codec over the official BF16 source.
use super::{
    resident::{add, check_budget},
    *,
};
use crate::{
    app_chat::gemma4,
    metal::gemma_affine::resident::hybrid,
    mlx::gemma4::bf16::CONFIG_BYTES,
    session::{self, Options},
    weights::{INITIAL_MEMORY_LIMIT_BYTES, audit::open_regular},
};
use std::{io::Read, path::Path};

#[derive(Clone, Copy, Debug)]
pub struct OfficialBf16Memory {
    /// Whole owner admission during construction, including codec allowance.
    pub load_admitted_bytes: usize,
    /// Add to the numerical session report for whole owner admission.
    pub session_reserved_bytes: usize,
    pub numerical_load: Bf16LoadReport,
}

/// The authenticated weights and basic-text codec for one pinned checkpoint.
/// Graph and session borrows keep the model, context and PLE descriptor alive.
pub struct OfficialBf16 {
    weights: Bf16Weights,
    context: Bf16Context,
    codec: gemma4::Codec,
    memory: OfficialBf16Memory,
    position_limit: usize,
    memory_limit: usize,
}

impl OfficialBf16 {
    pub const REPOSITORY: &str = "google/gemma-4-E2B-it";
    pub const REVISION: &str = "70af34e20bd4b7a91f0de6b22675850c43922a03";
    pub const DECLARED_POSITIONS: usize = 131_072;
    pub const MAX_POSITIONS: usize = 512;
    pub(in crate::metal::gemma_affine::bf16) fn reserved_bytes() -> Result<usize> {
        let owner_bytes = size_of::<Self>()
            - size_of::<Bf16Weights>()
            - size_of::<Bf16Context>()
            - size_of::<gemma4::Codec>();
        add(owner_bytes, gemma4::ALLOWANCE_BYTES)
    }

    /// Reserve the codec alongside numerical construction, even before both
    /// owners are live. Caller-owned application buffers are separate.
    pub fn load(directory: &Path, position_limit: usize, memory_limit: usize) -> Result<Self> {
        if !(1..=Self::MAX_POSITIONS).contains(&position_limit) {
            return Err(Error::Options("Gemma BF16 Mac position limit"));
        }
        if memory_limit > INITIAL_MEMORY_LIMIT_BYTES {
            return Err(Error::Options("Gemma BF16 Mac memory limit exceeds 6 GiB"));
        }
        // The source, context and codec admission already charge their own
        // storage. Only the remaining fields and padding are additional.
        let reserved = Self::reserved_bytes()?;
        let load_reserved = add(reserved, CONFIG_BYTES)?;
        check_budget(add(load_reserved, CONTEXT_ALLOWANCE_BYTES)?, memory_limit)?;
        let numerical_budget = memory_limit - load_reserved;

        let mut config_file = open_regular(&directory.join("config.json"))?;
        let mut config = [0; CONFIG_BYTES];
        config_file
            .read_exact(&mut config)
            .map_err(|error| Error::Native(error.to_string()))?;
        if config_file
            .read(&mut [0])
            .map_err(|error| Error::Native(error.to_string()))?
            != 0
        {
            return Err(Error::Options("Gemma BF16 config exceeds its pinned size"));
        }
        let codec = gemma4::Codec::load_official_bf16_regular(directory, gemma4::ALLOWANCE_BYTES)
            .map_err(|error| Error::Native(error.to_string()))?;
        let file = open_regular(&directory.join("model.safetensors"))?;
        let context = Bf16Context::new(numerical_budget)?;
        let weights = match Bf16Weights::load_text_only(&context, file, &config, numerical_budget) {
            Err(Error::Budget { required, .. }) => {
                return Err(Error::Budget {
                    required: add(required, load_reserved)?,
                    limit: memory_limit,
                });
            }
            result => result?,
        };
        let numerical_load = *weights.report();
        let memory = OfficialBf16Memory {
            load_admitted_bytes: add(numerical_load.admitted_load_bytes, load_reserved)?,
            session_reserved_bytes: reserved,
            numerical_load,
        };
        Ok(Self {
            weights,
            context,
            codec,
            memory,
            position_limit,
            memory_limit,
        })
    }

    pub fn position_limit(&self) -> usize {
        self.position_limit
    }

    pub fn memory(&self) -> OfficialBf16Memory {
        self.memory
    }
    /// Explicit immutable-owner explanation; this performs no model evaluation.
    pub fn runtime_inventory(
        &self,
        chunk_rows: usize,
    ) -> Result<super::resident::planning::RuntimeInventory> {
        self.weights.runtime_inventory(chunk_rows)
    }

    pub fn codec(&self) -> &gemma4::Codec {
        &self.codec
    }

    pub fn model(&self) -> std::result::Result<Bf16Model<'_>, hybrid::Error> {
        self.weights.build_model(self.position_limit)
    }

    pub(crate) fn planned_session_admission(
        &self,
        model: &Bf16Model<'_>,
        options: Options,
    ) -> std::result::Result<usize, session::Error> {
        self.validate_session_budget(options)?;
        let numerical = self.weights.planned_generation_admission(model, options)?;
        add(numerical, self.memory.session_reserved_bytes)
            .map_err(hybrid::Error::from)
            .map_err(Into::into)
    }

    fn validate_session_budget(&self, options: Options) -> std::result::Result<(), session::Error> {
        if options.memory_limit_bytes == 0 || options.memory_limit_bytes > self.memory_limit {
            return Err(session::Error::InvalidOptions(
                "Gemma BF16 session budget exceeds its owner limit",
            ));
        }
        Ok(())
    }

    /// Session options use a whole owner budget. The returned numerical
    /// session reports its own admission, apart from `session_reserved_bytes`.
    pub fn new_session<'model, 'owner>(
        &'owner self,
        model: &'model Bf16Model<'owner>,
        options: Options,
    ) -> std::result::Result<Bf16GenerationSession<'model, 'owner, 'owner>, session::Error> {
        self.new_session_with_profile(model, options, None)
    }

    /// Diagnostic G2 session capturing one selected committed evaluation.
    /// Charges an additional64MiB before allocating counter resources.
    pub fn new_profiled_session<'model, 'owner>(
        &'owner self,
        model: &'model Bf16Model<'owner>,
        options: Options,
        profile: Bf16ProfileOptions,
    ) -> std::result::Result<Bf16GenerationSession<'model, 'owner, 'owner>, session::Error> {
        let mut session = self.new_session_with_profile(model, options, Some(profile))?;
        session.use_last_row_prefill();
        Ok(session)
    }

    fn new_session_with_profile<'model, 'owner>(
        &'owner self,
        model: &'model Bf16Model<'owner>,
        mut options: Options,
        profile: Option<Bf16ProfileOptions>,
    ) -> std::result::Result<Bf16GenerationSession<'model, 'owner, 'owner>, session::Error> {
        self.validate_session_budget(options)?;
        let limit = options.memory_limit_bytes;
        let reserved = self.memory.session_reserved_bytes;
        options.memory_limit_bytes = limit
            .checked_sub(reserved)
            .filter(|bytes| *bytes != 0)
            .ok_or(session::Error::MemoryBudget {
                required: reserved + 1,
                limit,
            })?;
        let result = match profile {
            Some(profile) => self.weights.new_generation_session_with_profile(
                model,
                &self.context,
                options,
                Some(profile),
            ),
            None => self
                .weights
                .new_generation_session(model, &self.context, options),
        };
        match result {
            Err(session::Error::MemoryBudget { required, .. }) => {
                Err(session::Error::MemoryBudget {
                    required: add(required, reserved).map_err(hybrid::Error::from)?,
                    limit,
                })
            }
            result => result,
        }
    }
}

#[cfg(test)]
mod tests;
