//! Explicit bounded native checkpoint probe; see support/dense_probe.rs.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "support/dense_probe.rs"]
mod probe;
fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Err(error) = probe::run(true) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        eprintln!("Native checkpoint probe requires Apple Silicon macOS");
        std::process::exit(1);
    }
}
