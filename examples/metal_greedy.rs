//! Paired complete 512-input/128-output generation, CPU versus GPU selection.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        io::{self, BufRead, Read, Write},
        path::PathBuf,
        sync::atomic::AtomicBool,
    };
    use toklet::{
        generation::{self, Input, Request},
        metal, session, weights,
    };
    const APP: usize = 64 * 1024 * 1024;
    fn handshake() -> Result<(), Box<dyn std::error::Error>> {
        let mut line = String::new();
        if io::stdin().lock().take(16).read_line(&mut line)? == 0 || line != "continue\n" {
            return Err("bounded greedy handshake required".into());
        }
        Ok(())
    }
    fn request(
        s: &mut metal::Session<'_>,
        base: &[u32],
    ) -> Result<(generation::Report, Vec<u32>), Box<dyn std::error::Error>> {
        let mut ids = Vec::with_capacity(128);
        let report = generation::generate(
            s,
            Request {
                input: Input::Reset(base),
                max_new_tokens: 128,
                stop_tokens: &[],
            },
            &AtomicBool::new(false),
            |e| {
                ids.push(e.token_id);
                Ok(())
            },
        )?;
        s.discard_pending();
        Ok((report, ids))
    }
    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let mut args = std::env::args_os().skip(1);
        let source = PathBuf::from(args.next().ok_or("missing model")?);
        let packed = PathBuf::from(args.next().ok_or("missing Q4 artifact")?);
        let contract = PathBuf::from(args.next().ok_or("missing contract")?);
        if args.next().is_some() {
            return Err("too many greedy arguments".into());
        }
        let mut bytes = Vec::new();
        std::fs::File::open(contract.join("prompt_512.u32le"))?
            .take(2049)
            .read_to_end(&mut bytes)?;
        if bytes.len() != 2048
            || format!("{:x}", Sha256::digest(&bytes))
                != "4ff74dae2ee5b778f9313f4c707f9ccb6415609a9c0653684cce7be63d457dd6"
        {
            return Err("frozen greedy prompt changed".into());
        }
        let base: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let model = metal::Model::load_q4(&source, &packed, weights::LoadOptions::default())?;
        if model.config().hidden_size != 1536
            || model.config().num_layers != 28
            || model.config().vocab_size != 151936
        {
            return Err("fixed larger greedy geometry required".into());
        }
        let projection = metal::ProjectionMode::Q4CompensatedTiled;
        let normalization = metal::NormalizationMode::Compensated;
        let opts = session::Options {
            capacity: 1024,
            prefill_chunk_size: 128,
            memory_limit_bytes: weights::INITIAL_MEMORY_LIMIT_BYTES - APP,
            ..Default::default()
        };
        let mut full =
            metal::Session::with_execution_modes(&model, opts, projection, normalization)?;
        let remaining = opts
            .memory_limit_bytes
            .checked_sub(full.memory().admitted_bytes - model.weights_bytes())
            .ok_or("shared greedy budget overflow")?;
        let mut token = metal::Session::with_readback_mode(
            &model,
            session::Options {
                memory_limit_bytes: remaining,
                ..opts
            },
            projection,
            normalization,
            metal::ReadbackMode::GreedyToken,
        )?;
        let combined = full
            .memory()
            .admitted_bytes
            .checked_add(token.memory().admitted_bytes)
            .and_then(|n| n.checked_sub(model.weights_bytes()))
            .and_then(|n| n.checked_add(APP))
            .ok_or("combined greedy budget overflow")?;
        if combined > weights::INITIAL_MEMORY_LIMIT_BYTES {
            return Err("combined greedy budget exceeds6GiB".into());
        }
        let mut out = io::stdout().lock();
        writeln!(
            out,
            "{}",
            json!({"event":"ready","format":"toklet-metal-greedy-v1","readback_modes":["FullLogits","GreedyToken"],"projection_mode":projection,"normalization_mode":normalization,"hidden_size":1536,"num_layers":28,"vocab_size":151936,"debug_assertions":cfg!(debug_assertions),"device":model.device_name(),"greedy_execution_width":model.greedy_execution_width(),"prompt_tokens":512,"output_tokens":128,"capacity":1024,"chunk":128,"warmups":3,"samples":10,"application_bytes":APP,"combined_admitted_bytes":combined,"full_memory":full.memory(),"token_memory":token.memory(),"load_seconds":model.load_timing.load_seconds})
        )?;
        out.flush()?;
        handshake()?;
        for run in 0..13 {
            let (a, b) = if run % 2 == 0 {
                (request(&mut full, &base)?, request(&mut token, &base)?)
            } else {
                let b = request(&mut token, &base)?;
                (request(&mut full, &base)?, b)
            };
            if a.1.len() != 128
                || a.1 != b.1
                || full.history() != token.history()
                || full.history().len() != 639
                || full.evaluated_len() != 639
                || token.evaluated_len() != 639
                || full.pending_len() != 0
                || token.pending_len() != 0
                || full.history()[..512] != base
                || full.history()[512..] != a.1[..127]
            {
                return Err("greedy generation IDs/committed history differ".into());
            }
            if full.greedy_token().ok_or("full distribution absent")?? != *a.1.last().unwrap()
                || token.greedy_token().ok_or("token distribution absent")?? != *a.1.last().unwrap()
                || token.logits().is_some()
            {
                return Err("greedy readback/publication differs".into());
            }
            writeln!(
                out,
                "{}",
                json!({"event":"sample","run":run,"order":if run%2==0 {"full_first"}else{"token_first"},"full_report":a.0,"token_report":b.0,"token_ids":a.1,"token_ids_equal":true,"full_memory":full.memory(),"token_memory":token.memory(),"full_last_timing":full.timing(),"token_last_timing":token.timing()})
            )?;
            out.flush()?;
            handshake()?;
        }
        writeln!(
            out,
            "{}",
            json!({"event":"done","pairs":13,"combined_admitted_bytes":combined})
        )?;
        out.flush()?;
        handshake()?;
        Ok(())
    }
}
fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Err(e) = apple::run() {
        eprintln!("metal greedy failed: {e}");
        std::process::exit(1);
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        eprintln!("Apple GPU required");
        std::process::exit(1);
    }
}
