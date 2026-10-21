//! Public Metal GPU trace capture; enable externally with MTL_CAPTURE_ENABLED=1.
use super::{Error, Model, Result};
use objc2::rc::Retained;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};
use std::{marker::PhantomData, path::Path};
/// Stops this capture on drop. Borrow keeps the captured queue/device alive.
#[must_use]
pub struct Capture<'m> {
    manager: Retained<MTLCaptureManager>,
    model: PhantomData<&'m Model>,
}
impl Drop for Capture<'_> {
    fn drop(&mut self) {
        self.manager.stopCapture();
    }
}
impl Model {
    /// Capture commands created during this guard's lifetime. The destination
    /// must be a new absolute `.gputrace` path; existing captures are rejected.
    pub fn capture(&self, path: &Path) -> Result<Capture<'_>> {
        if !path.is_absolute() || path.exists() || path.extension().is_none_or(|e| e != "gputrace")
        {
            return Err(Error::Options(
                "capture needs a new absolute .gputrace path",
            ));
        }
        let path = path
            .to_str()
            .ok_or(Error::Options("capture path must be UTF-8"))?;
        // SAFETY: Metal is initialized by the borrowed model; process singleton
        // is retained and capture is started through its checked API.
        let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
        if manager.isCapturing()
            || !manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument)
        {
            return Err(Error::Unsupported(
                "GPU trace unavailable or capture already active",
            ));
        }
        let descriptor = MTLCaptureDescriptor::new();
        // SAFETY: actual MTLCommandQueue, retained by model and descriptor;
        // guard borrows model through capture completion.
        unsafe {
            descriptor.setCaptureObject(Some((*self.gpu.queue).as_ref()));
        }
        descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
        descriptor.setOutputURL(Some(&NSURL::fileURLWithPath(&NSString::from_str(path))));
        manager
            .startCaptureWithDescriptor_error(&descriptor)
            .map_err(|e| Error::Native(e.to_string()))?;
        Ok(Capture {
            manager,
            model: PhantomData,
        })
    }
}
