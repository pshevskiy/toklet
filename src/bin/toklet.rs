use std::{env, process::ExitCode};

const HELP: &str = "Toklet — a small Rust inference engine for Apple Silicon

Usage: toklet [OPTIONS]

Options:
  -h, --help     Print help
  -V, --version  Print version

Model loading and generation are not implemented yet.";

fn main() -> ExitCode {
    let mut args = env::args_os().skip(1);
    let first = args.next();

    if let Some(extra) = args.next() {
        eprintln!("error: unexpected argument {extra:?}\n\nUsage: toklet [OPTIONS]");
        return ExitCode::from(2);
    }

    match first.as_deref() {
        None => println!("{HELP}"),
        Some(arg) if arg == "-h" || arg == "--help" => println!("{HELP}"),
        Some(arg) if arg == "-V" || arg == "--version" => {
            println!("toklet {}", env!("CARGO_PKG_VERSION"));
        }
        Some(arg) => {
            eprintln!("error: unexpected argument {arg:?}\n\nUse 'toklet --help' for usage.");
            return ExitCode::from(2);
        }
    }

    ExitCode::SUCCESS
}
