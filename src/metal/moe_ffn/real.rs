//! Explicit process-owned qualification of the pinned original Granite BF16 model.
use super::*;
use crate::metal::{FeedForwardScratch, Session};
use half::f16;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{env, fs, io::Write, path::PathBuf};

const LIMIT: usize = 4 * 1024 * 1024 * 1024;
const REVISION: &str = "0da7a48b0276d500ce5922fd2b33944091fc6c09";
const MODEL_SHA: &str = "ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Inputs {
    prompt: Vec<u32>,
    expected: Vec<u32>,
    boundary: Vec<u32>,
}

fn digest(hex: &str) -> [u8; 32] {
    std::array::from_fn(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap())
}

fn row(session: &Session<'_>) -> Vec<u32> {
    let logits = session.logits().unwrap();
    assert_eq!(logits.len(), 49155);
    assert!(logits.iter().all(|x| x.is_finite()));
    logits.iter().map(|x| x.to_bits()).collect()
}

fn write_row(output: &Path, session: &Session<'_>, index: usize, last_row: usize) -> u32 {
    let logits = session.logits().unwrap();
    let id = crate::sampling::greedy(logits).unwrap();
    let mut file = fs::File::create(output.join(format!("row-{index}.f32le"))).unwrap();
    let raw = logits
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    file.write_all(&raw).unwrap();
    let FeedForwardScratch::Moe(ffn) = &session.scratch.feed_forward else {
        unreachable!()
    };
    let scores = ffn.scores.with_slice(|words| {
        words[last_row * 32..(last_row + 1) * 32]
            .iter()
            .map(|&bits| f16::from_bits(bits).to_f32())
            .collect::<Vec<_>>()
    });
    let route = last_row * 8..(last_row + 1) * 8;
    let ids = ffn.ids.with_slice(|ids| ids[route.clone()].to_vec());
    let gates = ffn.gates.with_slice(|gates| {
        gates[route]
            .iter()
            .map(|&bits| f32::from_bits(bits))
            .collect::<Vec<_>>()
    });
    println!(
        "{}",
        json!({"event":"decision","index":index,"id":id,"prefix_tokens":session.evaluated_len(),
               "sha256":format!("{:x}",Sha256::digest(&raw)),
               "router":{"scores":scores,"ids":ids,"gates":gates}})
    );
    id
}

#[test]
#[ignore = "Owned Mac GPU and pinned 2.669 GB Granite MoE original checkpoint"]
fn granite_moe_pinned_real_execution() {
    let model_dir = PathBuf::from(env::var_os("TOKLET_GRANITE_MOE_MODEL").unwrap());
    let input = PathBuf::from(env::var_os("TOKLET_GRANITE_MOE_INPUT").unwrap());
    let output = PathBuf::from(env::var_os("TOKLET_GRANITE_MOE_OUTPUT").unwrap());
    assert!(fs::metadata(&input).unwrap().len() < 65536);
    let input: Inputs = serde_json::from_slice(&fs::read(input).unwrap()).unwrap();
    assert!(!input.prompt.is_empty() && !input.expected.is_empty() && input.expected.len() <= 8);
    assert_eq!(input.boundary.len(), 512);
    assert!(input.prompt.len() + input.expected.len() <= 512);
    assert!(!input.expected[..input.expected.len() - 1].contains(&0));
    fs::create_dir(&output).unwrap();
    let model = Model::load_granite_moe(
        &model_dir,
        source::ExpectedArtifact {
            repository: "ibm-granite/granite-3.1-1b-a400m-instruct",
            revision: REVISION,
            config_bytes: 889,
            config_sha256: digest(
                "f1b64813debfabeb9cc20f1a13ed8248da6774fe54e532d19912d4b3f2ab335d",
            ),
            model_bytes: 2669283096,
            model_sha256: digest(MODEL_SHA),
        },
        weights::LoadOptions {
            context_limit: 512,
            memory_limit_bytes: LIMIT,
        },
    )
    .unwrap();
    let receipt = model.granite_moe_import_receipt().unwrap();
    assert_eq!(receipt.source_model_sha256(), digest(MODEL_SHA));
    let make = |chunk| {
        Session::new(
            &model,
            session::Options {
                capacity: 512,
                prefill_chunk_size: chunk,
                position_offset: 0,
                memory_limit_bytes: LIMIT,
            },
        )
        .unwrap()
    };
    let mut baseline = Vec::new();
    let mut boundary = Vec::new();
    let mut ids = Vec::new();
    for chunk in [32, 1] {
        let mut s = make(chunk);
        println!(
            "{}",
            json!({"event":"loaded","device":model.device_name(),"chunk":chunk,
                               "memory":s.memory(),"weights":model.weights_bytes(),
                               "converted_values":receipt.converted_values(),
                               "half_sha256":format!("{:x}",sha2::digest::Output::<Sha256>::from(receipt.half_payload_sha256()))})
        );
        s.append(&input.prompt).unwrap();
        for (index, &stock_id) in input.expected.iter().enumerate() {
            s.evaluate().unwrap();
            assert_eq!(s.evaluated_len(), input.prompt.len() + index);
            assert_eq!(s.history().len(), s.evaluated_len());
            let current = row(&s);
            if chunk == 32 {
                let last_row = if index == 0 {
                    (input.prompt.len() - 1) % chunk
                } else {
                    0
                };
                ids.push(write_row(&output, &s, index, last_row));
                baseline.push(current);
            } else {
                assert_eq!(current, baseline[index], "distribution at decision {index}");
            }
            if stock_id == 0 {
                break;
            }
            s.append(&[stock_id]).unwrap();
            assert!(s.logits().is_none());
        }
        s.reset();
        s.append(&input.boundary).unwrap();
        s.evaluate().unwrap();
        assert_eq!(s.evaluated_len(), 512);
        let current = row(&s);
        assert!(matches!(
            s.append(&[2]),
            Err(Error::Options("token ID/count outside session capacity"))
        ));
        assert_eq!(s.history(), input.boundary);
        assert_eq!(s.evaluated_len(), 512);
        assert_eq!(row(&s), current);
        if chunk == 32 {
            boundary = current;
        } else {
            assert_eq!(boundary, current);
        }
        s.reset();
        s.append(&input.prompt).unwrap();
        s.evaluate().unwrap();
        assert_eq!(row(&s), baseline[0]);
    }
    println!(
        "{}",
        json!({"event":"done","capacity":512,"chunks":[32,1],
                           "overflow_preserved":true,"reset_reproduced":true,"ids":ids})
    );
    assert_eq!(
        ids, input.expected,
        "stock/native greedy disagreement on frozen prefixes"
    );
}
