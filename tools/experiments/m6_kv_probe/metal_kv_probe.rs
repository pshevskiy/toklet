//! Entry point for the isolated M6 KV component only; never a production CLI.
fn main() {
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    {
        let args: Vec<_> = std::env::args().collect();
        if args.len() != 3 {
            eprintln!("requires small|larger primary|confirmation");
            std::process::exit(1);
        }
        if let Err(error) = toklet::metal::kv_probe::run(&args[1], &args[2]) {
            eprintln!("component error: {error}");
            std::process::exit(1);
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
    {
        eprintln!("requires explicit Metal Apple Silicon component build");
        std::process::exit(1);
    }
}
