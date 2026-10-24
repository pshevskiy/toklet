//! Cold metadata command; no tokenizer, session, GPU or signal handler is created.
use super::{invalid, number, utf8};
use std::{collections::HashSet, ffi::OsString, io, path::PathBuf};
use toklet::{
    mlx::compatibility::{self, Backend},
    weights::{self, LoadOptions},
};
const HELP: &str = "Usage: toklet inspect --model PATH [OPTIONS]

  --backend scalar-reference|metal   Reported backend (metal)
  --context-capacity N               Requested context (512)
  --memory-limit-bytes N             Numerical load budget (6442450944)

Print one schema-versioned JSON metadata report. No GPU or model expansion.
Exit 0 for every completed report, including unknown/unsupported/over-budget.
Exit 2 for malformed command arguments; exit 1 for output errors.
A closed output pipe exits 0.
Metadata never grants payload validity, device/session fit, Chat or qualification.";

fn report(values: Vec<OsString>) -> io::Result<compatibility::Report> {
    let mut model = None;
    let mut backend = Backend::Metal;
    let mut options = LoadOptions {
        context_limit: 512,
        memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES,
    };
    let mut seen = HashSet::new();
    let mut values = values.into_iter();
    while let Some(flag) = values.next() {
        let flag = utf8(flag)?;
        if !seen.insert(flag.clone()) {
            return Err(invalid(format!("duplicate {flag}")));
        }
        let value = values
            .next()
            .ok_or_else(|| invalid(format!("missing value for {flag}")))?;
        match flag.as_str() {
            "--model" if !value.is_empty() => model = Some(PathBuf::from(value)),
            "--backend" => {
                backend = match utf8(value)?.as_str() {
                    "scalar-reference" => Backend::ScalarReference,
                    "metal" => Backend::Metal,
                    _ => return Err(invalid("unknown inspection backend")),
                }
            }
            "--context-capacity" => options.context_limit = number(value)?,
            "--memory-limit-bytes" => options.memory_limit_bytes = number(value)?,
            _ => return Err(invalid(format!("unknown inspection option {flag}"))),
        }
    }
    let model = model.ok_or_else(|| invalid("inspection requires --model PATH"))?;
    Ok(compatibility::inspect(&model, backend, options))
}

pub(super) fn main(values: Vec<OsString>) -> u8 {
    if values.len() == 1 && (values[0] == "--help" || values[0] == "-h") {
        println!("{HELP}");
        return 0;
    }
    let report = match report(values) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("error: {error}\nUse 'toklet inspect --help' for usage.");
            return 2;
        }
    };
    write_report(&report)
}

pub(super) fn write_report(report: &impl serde::Serialize) -> u8 {
    use io::Write;
    let result = (|| -> io::Result<()> {
        let mut out = io::stdout().lock();
        serde_json::to_writer(&mut out, report)?;
        writeln!(out)?;
        out.flush()
    })();
    match result {
        Ok(()) => 0,
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => 0,
        Err(error) => {
            eprintln!("error: {error}");
            1
        }
    }
}
