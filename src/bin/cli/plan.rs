//! Explicit cold runtime inspection; unlike `inspect`, this initializes Metal.
use super::{invalid, number, utf8};
use std::{collections::HashSet, ffi::OsString, io, path::PathBuf};
use toklet::{session, weights};
const HELP: &str = "Usage: toklet plan --model PATH [OPTIONS]

  --route mlx-affine|official-gemma-bf16 (mlx-affine)
  --context-capacity N              Session capacity (512)
  --load-context-capacity N         Load context ceiling (offset + capacity)
  --prefill-chunk N                 Full chunk described (MLX:min(128,capacity); BF16:min(32,capacity))
  --position-offset N               Absolute initial position (0)
  --load-memory-limit-bytes N       Exact numerical load budget (6442450944)
  --session-memory-limit-bytes N    Exact numerical session budget (6442450944)
  --normalization original|compensated (compensated)
  --readback full-logits|greedy-token   (full-logits)
  --attention reference|online         (reference)

Explicit selected Metal runtime planning. Official Gemma uses the pinned Mac
BF16/G2 factory at offset0, full CPU logits and its CPU graph policies; MLX-only
normalization/readback/attention options are rejected for that route. Its session
budget cannot exceed its load owner budget. BF16 reports minimum admission;
allocator overreservation may increase it. Initializes device and precise
pipelines; allocates no numerical weight/KV/scratch/logit owners. Driver memory
uses the existing allowance estimate. Fit flags reserve nothing and do not
qualify payloads, Chat, quality, performance or a device. Loading revalidates.
Exit 0 for completed reports, including budget-fit failures; 2 for malformed
arguments; 1 for engine/unavailable-backend or output errors. Closed pipe: 0.";
struct Args {
    route: String,
    model: PathBuf,
    load: weights::LoadOptions,
    session: session::Options,
    normalization: String,
    readback: String,
    attention: String,
}
fn parse(values: Vec<OsString>) -> io::Result<Args> {
    let mut model = None;
    let mut route = "mlx-affine".to_owned();
    let mut capacity = 512;
    let mut load_capacity = None;
    let mut chunk = None;
    let mut offset: usize = 0;
    let mut load_budget = weights::INITIAL_MEMORY_LIMIT_BYTES;
    let mut session_budget = load_budget;
    let mut normalization = "compensated".to_owned();
    let mut readback = "full-logits".to_owned();
    let mut attention = "reference".to_owned();
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
            "--route" => {
                route = utf8(value)?;
                if !["mlx-affine", "official-gemma-bf16"].contains(&route.as_str()) {
                    return Err(invalid("unknown planning route"));
                }
            }
            "--model" if !value.is_empty() => model = Some(PathBuf::from(value)),
            "--context-capacity" => capacity = number(value)?,
            "--load-context-capacity" => load_capacity = Some(number(value)?),
            "--prefill-chunk" => chunk = Some(number(value)?),
            "--position-offset" => offset = number(value)?,
            "--load-memory-limit-bytes" => load_budget = number(value)?,
            "--session-memory-limit-bytes" => session_budget = number(value)?,
            "--normalization" => {
                normalization = utf8(value)?;
                if !["original", "compensated"].contains(&normalization.as_str()) {
                    return Err(invalid("unknown normalization"));
                }
            }
            "--readback" => {
                readback = utf8(value)?;
                if !["full-logits", "greedy-token"].contains(&readback.as_str()) {
                    return Err(invalid("unknown readback"));
                }
            }
            "--attention" => {
                attention = utf8(value)?;
                if !["reference", "online"].contains(&attention.as_str()) {
                    return Err(invalid("unknown attention"));
                }
            }
            _ => return Err(invalid(format!("unknown runtime planning option {flag}"))),
        }
    }
    let context_limit = match load_capacity {
        Some(n) => n,
        None => offset
            .checked_add(capacity)
            .ok_or_else(|| invalid("position/context overflow"))?,
    };
    if route == "official-gemma-bf16"
        && ["--normalization", "--readback", "--attention"]
            .iter()
            .any(|flag| seen.contains(*flag))
    {
        return Err(invalid(
            "official BF16 uses fixed factory policies; normalization/readback/attention flags are MLX-only",
        ));
    }
    let default_chunk = if route == "official-gemma-bf16" {
        32
    } else {
        128
    };
    Ok(Args {
        route,
        model: model.ok_or_else(|| invalid("planning requires --model PATH"))?,
        load: weights::LoadOptions {
            context_limit,
            memory_limit_bytes: load_budget,
        },
        session: session::Options {
            capacity,
            prefill_chunk_size: chunk.unwrap_or(capacity.min(default_chunk)),
            position_offset: offset,
            memory_limit_bytes: session_budget,
        },
        normalization,
        readback,
        attention,
    })
}
fn run(args: Args) -> io::Result<u8> {
    #[cfg(all(
        feature = "metal",
        any(target_os = "macos", target_os = "ios"),
        target_arch = "aarch64"
    ))]
    {
        use toklet::metal::{self, AttentionMode, NormalizationMode, ReadbackMode};
        if args.route == "official-gemma-bf16" {
            #[cfg(target_os = "macos")]
            {
                let report = metal::gemma_affine::inspect_official_bf16(
                    &args.model,
                    args.load,
                    args.session,
                )
                .map_err(io::Error::other)?;
                return Ok(super::inspect::write_report(&report));
            }
            #[cfg(not(target_os = "macos"))]
            return Err(io::Error::other("official BF16 planning is Mac-only"));
        }
        let report = metal::planning::inspect_mlx(
            &args.model,
            args.load,
            args.session,
            if args.normalization == "original" {
                NormalizationMode::Original
            } else {
                NormalizationMode::Compensated
            },
            if args.readback == "full-logits" {
                ReadbackMode::FullLogits
            } else {
                ReadbackMode::GreedyToken
            },
            if args.attention == "reference" {
                AttentionMode::Reference
            } else {
                AttentionMode::Online
            },
        )
        .map_err(io::Error::other)?;
        Ok(super::inspect::write_report(&report))
    }
    #[cfg(not(all(
        feature = "metal",
        any(target_os = "macos", target_os = "ios"),
        target_arch = "aarch64"
    )))]
    {
        let Args {
            route,
            model,
            load,
            session,
            normalization,
            readback,
            attention,
        } = args;
        let _ = (
            route,
            model,
            load,
            session,
            normalization,
            readback,
            attention,
        );
        Err(io::Error::other(
            "Runtime Metal planning unavailable in this build/platform",
        ))
    }
}
pub(super) fn main(values: Vec<OsString>) -> u8 {
    if values.len() == 1 && (values[0] == "--help" || values[0] == "-h") {
        println!("{HELP}");
        return 0;
    }
    let args = match parse(values) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("error: {error}\nUse 'toklet plan --help' for usage.");
            return 2;
        }
    };
    match run(args) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error}");
            1
        }
    }
}
