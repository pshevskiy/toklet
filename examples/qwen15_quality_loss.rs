//! Explicit original-weight held-out loss collection for the pinned Qwen2.5-1.5B MLX8 model.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "support/nll.rs"]
mod nll;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use serde_json::json;
    use std::path::Path;
    use toklet::{
        metal::{self, AttentionMode, NormalizationMode, ProjectionMode, ReadbackMode},
        session::Options,
        weights::LoadOptions,
    };

    const BUDGET: usize = 4 * 1024 * 1024 * 1024;
    const VOCAB: usize = 151936;
    const BLOCK_TARGETS: usize = 1024;
    let args: Vec<_> = std::env::args().skip(1).collect();
    let (capacity, limit, chunks): (_, _, &[_]) = match args.as_slice() {
        [] => (512, 33, &[1, 32]),
        [mode] if mode == "--full" => (BLOCK_TARGETS + 1, 10_001, &[32]),
        _ => return Err("usage: qwen15_quality_loss [--full]".into()),
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("models/Qwen2.5-1.5B-Instruct");
    let candidate = root.join("models/MLX-Qwen2.5-1.5B-Instruct-8bit-c7e1ff2");
    let corpus = include_bytes!("../benchmarks/contract-v1/held_out_loss.u32le");
    let tokens: Vec<_> = corpus[..limit * 4]
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    let text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2 + 2? Reply with a single digit.<|im_end|>\n<|im_start|>assistant\n";
    let original_tokenizer = tokenizers::Tokenizer::from_file(source.join("tokenizer.json"))
        .map_err(|e| e.to_string())?;
    let candidate_tokenizer = tokenizers::Tokenizer::from_file(candidate.join("tokenizer.json"))
        .map_err(|e| e.to_string())?;
    let prompt = original_tokenizer
        .encode(text, false)
        .map_err(|e| e.to_string())?;
    let expected = [
        151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838,
        374, 220, 17, 488, 220, 17, 30, 17841, 448, 264, 3175, 15723, 13, 151645, 198, 151644,
        77091, 198,
    ];
    if prompt.get_ids() != expected
        || prompt.get_ids()
            != candidate_tokenizer
                .encode(text, false)
                .map_err(|e| e.to_string())?
                .get_ids()
        || original_tokenizer
            .decode(&[19], false)
            .map_err(|e| e.to_string())?
            != "4"
        || candidate_tokenizer
            .decode(&[19], false)
            .map_err(|e| e.to_string())?
            != "4"
    {
        return Err("Plain-chat tokenizer identity differs".into());
    }
    println!(
        "QWEN15_TEXT {}",
        json!({"prompt":prompt.get_ids(),"decoded_id":19,"decoded":"4"})
    );

    let model = metal::Model::load_mlx(
        &candidate,
        LoadOptions {
            context_limit: capacity,
            memory_limit_bytes: BUDGET,
        },
    )?;
    let c = model.config();
    if (c.num_layers, c.hidden_size, c.vocab_size, c.max_positions) != (28, 1536, VOCAB, capacity) {
        return Err("Loaded checkpoint geometry differs".into());
    }
    for &chunk in chunks {
        let mut session = metal::Session::with_attention_mode(
            &model,
            Options {
                capacity,
                prefill_chunk_size: chunk,
                position_offset: 0,
                memory_limit_bytes: BUDGET,
            },
            ProjectionMode::MlxAffine,
            NormalizationMode::Compensated,
            ReadbackMode::FullLogits,
            AttentionMode::Reference,
        )?;
        session.enable_diagnostics(&[])?;
        for start in (0..tokens.len() - 1).step_by(BLOCK_TARGETS) {
            let end = (start + BLOCK_TARGETS + 1).min(tokens.len());
            score_block(&mut session, &tokens[start..end], chunk, start)?;
        }
        session.reset();
        session.set_tokens(prompt.get_ids())?;
        session.evaluate()?;
        let answer = session.greedy_token().ok_or("Missing answer")??;
        session.append(&[answer])?;
        session.evaluate()?;
        let stop = session.greedy_token().ok_or("Missing stop")??;
        if (answer, stop) != (19, 151645) {
            return Err("Pinned native plain-chat trajectory differs".into());
        }
        println!(
            "QWEN15_COMPLETE {}",
            json!({"chunk":chunk,"scored":tokens.len()-1,
            "admitted_bytes":session.memory().admitted_bytes,"capacity":capacity,"answer":answer,"stop":stop})
        );
    }
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn score_block(
    session: &mut toklet::metal::Session<'_>,
    tokens: &[u32],
    chunk: usize,
    start: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use serde_json::json;
    use sha2::{Digest, Sha256};
    let vocab = session.vocab_size();
    session.reset();
    session.set_tokens(tokens)?;
    let mut position = 0;
    loop {
        let rows = session.evaluate_next_chunk()?;
        if rows == 0 {
            break;
        }
        let snapshots = session.diagnostics()?;
        let logits = snapshots
            .iter()
            .find(|d| d.stage == "logits")
            .ok_or("Missing all-row logits")?;
        if (logits.rows, logits.columns, logits.values.len()) != (rows, vocab, rows * vocab) {
            return Err("Loss row shape differs".into());
        }
        for values in logits.values.chunks_exact(vocab) {
            if let Some(&target) = tokens.get(position + 1) {
                let (maximum, exp_mass, loss) = nll::statistics(values, target as usize)?;
                let mut hash = Sha256::new();
                for value in values {
                    hash.update(value.to_le_bytes());
                }
                println!(
                    "QWEN15_LOSS {}",
                    json!({"chunk":chunk,"position":position,"global_target":start+position,"target":target,
                    "maximum":maximum,"exp_mass":exp_mass,"target_logit_bits":values[target as usize].to_bits(),"loss":loss,"row_sha256":format!("{:x}",hash.finalize())})
                );
            } else {
                let _ = nll::statistics(values, 0)?;
            }
            position += 1;
        }
    }
    if position != tokens.len()
        || session.history() != tokens
        || session.evaluated_len() != tokens.len()
    {
        return Err("Causal loss coverage differs".into());
    }
    println!(
        "QWEN15_BLOCK {}",
        json!({"chunk":chunk,"start":start,
        "targets":tokens.len()-1,"evaluated":session.evaluated_len(),"pending":session.pending_len()})
    );
    Ok(())
}

fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        eprintln!("Native Qwen quality alignment requires Apple Silicon macOS");
        std::process::exit(1);
    }
}
