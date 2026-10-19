//! Versioned raw-token C boundary. Numerical/model behavior stays in the engine.
//!
//! # Foreign caller contract
//! Every non-null pointer must be aligned, live, initialized and valid for its
//! stated extent. Inputs must not overlap writable outputs/error records. A
//! handle must come from this library and must not be used after release.
//! Model/session calls and release run on their creating thread. Only operations
//! on the independent cancellation flag may run concurrently; its release must
//! wait until all users, including generation, have returned. Callbacks are
//! synchronous on the inference thread and must not unwind/throw across C.
//! Reentrant session calls return Busy. Rust unwinding poisons the session;
//! recreate it afterwards. OOM abort and foreign exceptions are not contained.

use crate::{generation, model, session, weights};
use std::{
    cell::{Cell, RefCell},
    ffi::c_void,
    io,
    mem::{align_of, size_of},
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    ptr,
    rc::Rc,
    sync::atomic::{AtomicBool, Ordering},
    thread::{self, ThreadId},
};

pub const ABI_VERSION: u32 = 1;
pub const OK: u32 = 0;
pub const INVALID: u32 = 1;
pub const LOAD_ERROR: u32 = 2;
pub const ENGINE_ERROR: u32 = 3;
pub const BUSY: u32 = 4;
pub const WRONG_THREAD: u32 = 5;
pub const PANIC: u32 = 6;
pub const POISONED: u32 = 7;
pub const CALLBACK_ERROR: u32 = 8;
const BRIDGE_BYTES: usize = 4096;

#[repr(C)]
pub struct NativeError {
    pub code: u32,
    /// NUL-terminated, possibly truncated UTF-8. Caller owns this fixed record.
    pub message: [u8; 256],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Options {
    pub abi_version: u32,
    pub struct_size: u32,
    pub capacity: u32,
    pub chunk: u32,
    /// Explicit whole single-model/session budget; no mobile budget is inferred.
    pub memory_limit_bytes: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TokenEvent {
    pub token_id: u32,
    pub index: u32,
    pub elapsed_seconds: f64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Report {
    pub abi_version: u32,
    pub struct_size: u32,
    /// 0 = state query, 1 = output limit, 2 = hidden stop ID, 3 = cancelled.
    pub reason: u32,
    pub stop_token: u32,
    pub history_tokens: u32,
    pub evaluated_tokens: u32,
    pub pending_tokens: u32,
    pub emitted_tokens: u32,
    pub admitted_bytes: u64,
}
pub type Emit = Option<unsafe extern "C" fn(*mut c_void, TokenEvent) -> u32>;

pub struct ModelHandle {
    owner: ThreadId,
    model: Rc<model::Model>,
}
// The borrow never escapes this private wrapper. Rc's allocation remains stable,
// immutable, and owned for the entire session lifetime. Field order drops the
// borrowed session BEFORE its retained model. No API exposes either field.
struct OwnedSession {
    session: session::Session<'static>,
    _model: Rc<model::Model>,
}
pub struct SessionHandle {
    owner: ThreadId,
    poisoned: Cell<bool>,
    state: RefCell<OwnedSession>,
}
pub struct CancelHandle {
    flag: AtomicBool,
}
struct Failure(u32, String);
type Result<T> = std::result::Result<T, Failure>;
fn fail(code: u32, message: impl Into<String>) -> Failure {
    Failure(code, message.into())
}
fn owner(id: ThreadId) -> Result<()> {
    if id == thread::current().id() {
        Ok(())
    } else {
        Err(fail(
            WRONG_THREAD,
            "handle belongs to a different worker thread",
        ))
    }
}
fn pointer<T>(p: *const T) -> Result<()> {
    if p.is_null() || !(p as usize).is_multiple_of(align_of::<T>()) {
        Err(fail(INVALID, "null or misaligned required pointer"))
    } else {
        Ok(())
    }
}
// Pointer provenance and live initialized extent remain the C caller's duty.
unsafe fn slice<'a, T>(p: *const T, len: usize, maximum: usize) -> Result<&'a [T]> {
    if len > maximum || len > isize::MAX as usize / size_of::<T>() {
        return Err(fail(INVALID, "input length exceeds boundary limit"));
    }
    if len == 0 {
        return Ok(&[]);
    }
    pointer(p)?;
    // SAFETY: caller provides a live initialized nonoverlapping input extent;
    // alignment, non-null and representable byte length were checked above.
    Ok(unsafe { std::slice::from_raw_parts(p, len) })
}
unsafe fn options(p: *const Options) -> Result<session::Options> {
    pointer(p)?;
    // SAFETY: caller supplies the entire initialized fixed-width Options record.
    let o = unsafe { *p };
    if o.abi_version != ABI_VERSION || o.struct_size as usize != size_of::<Options>() {
        return Err(fail(INVALID, "unsupported options version or size"));
    }
    let budget = usize::try_from(o.memory_limit_bytes)
        .ok()
        .and_then(|n| n.checked_sub(BRIDGE_BYTES))
        .filter(|n| *n > 0 && *n <= weights::INITIAL_MEMORY_LIMIT_BYTES - BRIDGE_BYTES)
        .ok_or_else(|| fail(INVALID, "explicit memory budget must be >4096 and <=6GiB"))?;
    Ok(session::Options {
        capacity: o.capacity as usize,
        prefill_chunk_size: o.chunk as usize,
        position_offset: 0,
        memory_limit_bytes: budget,
    })
}
unsafe fn boundary(error: *mut NativeError, work: impl FnOnce() -> Result<()>) -> u32 {
    if !error.is_null() && pointer(error).is_err() {
        return INVALID;
    }
    let result = catch_unwind(AssertUnwindSafe(work))
        .unwrap_or_else(|_| Err(fail(PANIC, "Rust panic contained at native boundary")));
    let (code, message) = match result {
        Ok(()) => (OK, String::new()),
        Err(Failure(code, message)) => (code, message),
    };
    if !error.is_null() {
        let mut record = NativeError {
            code,
            message: [0; 256],
        };
        let mut end = message.len().min(255);
        while !message.is_char_boundary(end) {
            end -= 1;
        }
        record.message[..end].copy_from_slice(&message.as_bytes()[..end]);
        // SAFETY: optional output record is aligned, writable and disjoint by
        // caller contract; it stays live through this synchronous call.
        unsafe { error.write(record) };
    }
    code
}
impl SessionHandle {
    fn access<T>(&self, work: impl FnOnce(&mut session::Session<'_>) -> Result<T>) -> Result<T> {
        owner(self.owner)?;
        if self.poisoned.get() {
            return Err(fail(POISONED, "session requires release and recreation"));
        }
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| fail(BUSY, "session is in a callback or another operation"))?;
        match catch_unwind(AssertUnwindSafe(|| work(&mut state.session))) {
            Ok(result) => result,
            Err(_) => {
                self.poisoned.set(true);
                Err(fail(PANIC, "session panic; recreate session"))
            }
        }
    }
}
fn report(s: &session::Session<'_>) -> Report {
    Report {
        abi_version: ABI_VERSION,
        struct_size: size_of::<Report>() as u32,
        stop_token: u32::MAX,
        history_tokens: s.history().len() as u32,
        evaluated_tokens: s.evaluated_len() as u32,
        pending_tokens: s.pending_len() as u32,
        admitted_bytes: (s.memory().admitted_bytes + BRIDGE_BYTES) as u64,
        ..Report::default()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn toklet_abi_version() -> u32 {
    ABI_VERSION
}

/// Load one immutable CPU model from a bounded UTF-8 local directory path.
/// # Safety
/// All pointers and handle use must obey this module's foreign caller contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_model_load(
    path: *const u8,
    len: usize,
    opts: *const Options,
    out: *mut *mut ModelHandle,
    error: *mut NativeError,
) -> u32 {
    // SAFETY: delegated caller contract applies to all pointers inside boundary.
    unsafe {
        boundary(error, || {
            pointer(out)?;
            out.write(ptr::null_mut());
            let o = options(opts)?;
            let path = std::str::from_utf8(slice(path, len, 16384)?)
                .map_err(|_| fail(INVALID, "path must be UTF-8"))?;
            if path.is_empty() || path.contains('\0') {
                return Err(fail(INVALID, "empty or NUL-containing path"));
            }
            let loaded = weights::load_qwen(
                Path::new(path),
                weights::LoadOptions {
                    context_limit: o.capacity,
                    memory_limit_bytes: o.memory_limit_bytes,
                },
            )
            .map_err(|e| fail(LOAD_ERROR, e.to_string()))?;
            out.write(Box::into_raw(Box::new(ModelHandle {
                owner: thread::current().id(),
                model: Rc::new(loaded.model),
            })));
            Ok(())
        })
    }
}
/// Release a model handle. Live sessions retain the model; NULL is a no-op.
/// # Safety
/// Follow the foreign caller contract; release each non-null handle once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_model_release(
    model: *mut ModelHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            if !model.is_null() {
                pointer(model)?;
                owner((*model).owner)?;
                drop(Box::from_raw(model));
            }
            Ok(())
        })
    }
}
/// Create fixed-capacity session state retaining the immutable model allocation.
/// # Safety
/// Follow the foreign caller contract; outputs must be disjoint from inputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_session_create(
    model: *const ModelHandle,
    opts: *const Options,
    out: *mut *mut SessionHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(out)?;
            out.write(ptr::null_mut());
            pointer(model)?;
            owner((*model).owner)?;
            let opts = options(opts)?;
            let retained = Rc::clone(&(*model).model);
            // SAFETY: private OwnedSession retains this immutable stable allocation
            // and drops the session first. No reference can escape the wrapper.
            let borrowed = &*Rc::as_ptr(&retained);
            let session = session::Session::new(borrowed, opts)
                .map_err(|e| fail(ENGINE_ERROR, e.to_string()))?;
            out.write(Box::into_raw(Box::new(SessionHandle {
                owner: thread::current().id(),
                poisoned: Cell::new(false),
                state: RefCell::new(OwnedSession {
                    session,
                    _model: retained,
                }),
            })));
            Ok(())
        })
    }
}
/// Release idle session state. Busy callback reentry returns Busy, retaining it.
/// # Safety
/// Follow the foreign caller contract; release once on the creating thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_session_release(
    session: *mut SessionHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            if !session.is_null() {
                pointer(session)?;
                owner((*session).owner)?;
                {
                    let _idle = (*session)
                        .state
                        .try_borrow_mut()
                        .map_err(|_| fail(BUSY, "session is busy"))?;
                }
                drop(Box::from_raw(session));
            }
            Ok(())
        })
    }
}
/// Snapshot committed/pending counts and complete single-session admission.
/// # Safety
/// Follow the foreign caller contract; report points to a complete writable record.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_session_state(
    session: *const SessionHandle,
    out: *mut Report,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(session)?;
            pointer(out)?;
            (*session).access(|s| {
                out.write(report(s));
                Ok(())
            })
        })
    }
}
/// Clear KV/history after any completed request, retaining resident resources.
/// # Safety
/// Follow the foreign caller contract; no concurrent/reentrant mutation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_session_reset(
    session: *const SessionHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(session)?;
            (*session).access(|s| {
                s.reset();
                Ok(())
            })
        })
    }
}
/// Generate raw token IDs: operation 0 prompt, 1 append, 2 continue, 3 reset+prompt.
/// Callback nonzero stops with CALLBACK_ERROR, retaining the pending sampled ID.
/// # Safety
/// Follow the foreign caller contract. Callback/context and cancellation remain
/// live until return. Callback must not throw or release active resources.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_session_generate(
    session: *const SessionHandle,
    operation: u32,
    tokens: *const u32,
    token_count: usize,
    max_new_tokens: u32,
    stops: *const u32,
    stop_count: usize,
    cancel: *const CancelHandle,
    emit: Emit,
    context: *mut c_void,
    out: *mut Report,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(session)?;
            pointer(cancel)?;
            pointer(out)?;
            let tokens = slice(tokens, token_count, 32768)?;
            let stops = slice(stops, stop_count, 32768)?;
            let input = match operation {
                0 => generation::Input::Prompt(tokens),
                1 => generation::Input::Append(tokens),
                2 if tokens.is_empty() => generation::Input::Continue,
                3 => generation::Input::Reset(tokens),
                _ => return Err(fail(INVALID, "invalid operation or continue input")),
            };
            (*session).access(|s| {
                let mut emitted = 0;
                let result = generation::generate(
                    s,
                    generation::Request {
                        input,
                        max_new_tokens: max_new_tokens as usize,
                        stop_tokens: stops,
                    },
                    &(*cancel).flag,
                    |event| {
                        if let Some(callback) = emit {
                            let event = TokenEvent {
                                token_id: event.token_id,
                                index: event.index as u32,
                                elapsed_seconds: event.elapsed_seconds,
                            };
                            // SAFETY: foreign synchronous callback/context obey the
                            // ABI contract; reentrant session operations are rejected.
                            if callback(context, event) != 0 {
                                return Err(io::Error::other("native callback stopped output"));
                            }
                        }
                        emitted += 1;
                        Ok(())
                    },
                );
                // Always publish recoverable state, including callback/session error.
                let mut summary = report(s);
                summary.emitted_tokens = emitted;
                if let Ok(ref r) = result {
                    summary.reason = match r.reason {
                        generation::StopReason::OutputLimit => 1,
                        generation::StopReason::StopToken { token_id } => {
                            summary.stop_token = token_id;
                            2
                        }
                        generation::StopReason::Cancelled => 3,
                    };
                    summary.emitted_tokens = r.emitted_tokens as u32;
                }
                out.write(summary);
                result.map(|_| ()).map_err(|e| match e {
                    generation::Error::Output(e) => fail(CALLBACK_ERROR, e.to_string()),
                    generation::Error::InvalidRequest(e) => fail(INVALID, e),
                    e => fail(ENGINE_ERROR, e.to_string()),
                })
            })
        })
    }
}
/// Create a cancellation flag, initially clear, independent of thread affinity.
/// # Safety
/// Follow the foreign caller contract for output/error storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_cancel_create(
    out: *mut *mut CancelHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(out)?;
            out.write(Box::into_raw(Box::new(CancelHandle {
                flag: AtomicBool::new(false),
            })));
            Ok(())
        })
    }
}
/// Set or clear the atomic flag. Only flag operations may overlap inference.
/// Clearing is an explicit idle-request action; never race it with a new cancel.
/// # Safety
/// The flag handle remains live until every concurrent caller has returned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_cancel_set(
    cancel: *const CancelHandle,
    value: u32,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            pointer(cancel)?;
            if value > 1 {
                return Err(fail(INVALID, "cancellation value must be zero or one"));
            }
            (*cancel).flag.store(value == 1, Ordering::Relaxed);
            Ok(())
        })
    }
}
/// Release an idle cancellation flag. NULL is a no-op.
/// # Safety
/// No generation/cancel caller may retain or use the flag during/after release.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn toklet_cancel_release(
    cancel: *mut CancelHandle,
    error: *mut NativeError,
) -> u32 {
    unsafe {
        boundary(error, || {
            if !cancel.is_null() {
                pointer(cancel)?;
                drop(Box::from_raw(cancel));
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests;
