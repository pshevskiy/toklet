//! Private native lifetime bridge for the audited Mac BF16 owner.
use super::*;
use crate::metal::gemma_affine::{
    Bf16GenerationSession, Bf16Model, CONTEXT_ALLOWANCE_BYTES, OfficialBf16,
};

pub(super) struct Model {
    // The graph borrows the stable Rc allocation. Field order drops it first.
    graph: Bf16Model<'static>,
    owner: Rc<OfficialBf16>,
}

impl Model {
    pub(super) fn load(directory: &Path, options: session::Options) -> Result<Self> {
        let owner = Rc::new(
            OfficialBf16::load(directory, options.capacity, options.memory_limit_bytes)
                .map_err(|error| fail(LOAD_ERROR, error.to_string()))?,
        );
        // SAFETY: the Rc allocation is stable. The private graph is dropped
        // before the Rc and cannot escape Model. Native sessions retain this
        // Model through OwnedSession, which drops the session before its Rc.
        let borrowed = unsafe { &*Rc::as_ptr(&owner) };
        let graph = borrowed
            .model()
            .map_err(|error| fail(LOAD_ERROR, error.to_string()))?;
        let admitted = graph
            .model_bytes()
            .checked_add(CONTEXT_ALLOWANCE_BYTES)
            .and_then(|value| value.checked_add(owner.memory().session_reserved_bytes))
            .filter(|&value| value <= options.memory_limit_bytes);
        if admitted.is_none() {
            return Err(fail(
                LOAD_ERROR,
                "Gemma BF16 retained graph exceeds its owner budget",
            ));
        }
        Ok(Self { graph, owner })
    }

    pub(super) fn loaded_positions(&self) -> usize {
        self.owner.position_limit()
    }

    pub(super) fn memory(&self) -> crate::metal::gemma_affine::OfficialBf16Memory {
        self.owner.memory()
    }

    pub(super) fn codec(&self) -> &crate::app_chat::gemma4::Codec {
        self.owner.codec()
    }

    pub(super) fn planned_session_admission(&self, options: session::Options) -> Result<usize> {
        self.owner
            .planned_session_admission(&self.graph, options)
            .map_err(|error| fail(INVALID, error.to_string()))
    }

    pub(super) fn shared_model_bytes(&self) -> Result<usize> {
        self.graph
            .model_bytes()
            .checked_add(self.owner.memory().session_reserved_bytes)
            .ok_or_else(|| fail(ENGINE_ERROR, "official Gemma shared owner size overflow"))
    }

    pub(super) fn create_session(&'static self, options: session::Options) -> Result<Session> {
        let inference = self
            .owner
            .new_session(&self.graph, options)
            .map_err(|error| fail(ENGINE_ERROR, error.to_string()))?;
        let admitted_bytes = inference
            .memory()
            .admitted_bytes
            .checked_add(self.owner.memory().session_reserved_bytes)
            .filter(|&value| value <= options.memory_limit_bytes)
            .ok_or_else(|| fail(ENGINE_ERROR, "Gemma BF16 session owner accounting differs"))?;
        Ok(Session {
            inference,
            admitted_bytes,
        })
    }
}

pub(super) struct Session {
    pub(super) inference: Bf16GenerationSession<'static, 'static, 'static>,
    pub(super) admitted_bytes: usize,
}

#[cfg(test)]
mod tests;
