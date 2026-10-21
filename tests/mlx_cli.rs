#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use serde_json::{Value, json};
use std::{
    io::Write,
    path::Path,
    process::{Command, Output, Stdio},
};

fn command(attention: &str, chunk: &str) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_toklet"));
    c.arg("--model")
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlx-qwen-v1/model"))
        .args([
            "--backend",
            "metal",
            "--metal-projection",
            "mlx-affine",
            "--metal-attention",
            attention,
            "--context-capacity",
            "32",
            "--prefill-chunk",
            chunk,
            "--json",
        ]);
    c
}
fn records(out: &Output) -> Vec<Value> {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.stderr.is_empty());
    String::from_utf8(out.stdout.clone())
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}
fn execution(v: &Value, attention: &str, chunk: u32) {
    let mut expected = json!({"weight_dtype":"MLX affine + FP16 exclusions",
        "kv_dtype":"FP16","projection_mode":"MlxAffine","normalization_mode":"Compensated",
        "readback_mode":"FullLogits","prefill_chunk_size":chunk,
        "quantizer":"MLX affine (per-tensor bits/groups)"});
    if attention == "online" {
        expected["attention_mode"] = json!("Online");
    }
    assert_eq!(v, &expected);
}

#[test]
#[ignore = "Explicit Apple GPU native MLX CLI generation, selection, statistics, stop and persistent state"]
fn native_mlx_cli_runs_local_packed_model_with_truthful_metadata_and_state() {
    for attention in ["reference", "online"] {
        for chunk in ["1", "3"] {
            let out = command(attention, chunk)
                .args(["--tokens", "1,7,13,3", "--max-new-tokens", "4"])
                .output()
                .unwrap();
            let r = records(&out);
            assert_eq!(
                r.iter()
                    .filter(|v| v["event"] == "token")
                    .map(|v| v["token_id"].as_u64().unwrap())
                    .collect::<Vec<_>>(),
                [47; 4]
            );
            let stats = &r.last().unwrap()["statistics"];
            execution(&stats["execution"], attention, chunk.parse().unwrap());
            assert_eq!(stats["backend"], "metal");
            assert_eq!(stats["cpu_projection"], "none");
            assert_eq!(stats["generation"]["evaluated_tokens"], 7);
            assert_eq!(stats["generation"]["pending_tokens"], 1);
            let memory = &stats["logical_memory"];
            assert_eq!(memory["weights_bytes"], 18152);
            assert_eq!(
                memory
                    .as_object()
                    .unwrap()
                    .values()
                    .map(|v| v.as_u64().unwrap())
                    .sum::<u64>(),
                stats["memory_admitted_bytes"].as_u64().unwrap()
            );
            let out = command(attention, chunk)
                .args([
                    "--tokens",
                    "1,7,13,3",
                    "--max-new-tokens",
                    "4",
                    "--stop-token",
                    "47",
                ])
                .output()
                .unwrap();
            let r = records(&out);
            assert_eq!(r.len(), 1);
            let stopped = &r[0]["statistics"]["generation"];
            assert_eq!(stopped["sampled_tokens"], 1);
            assert_eq!(stopped["emitted_tokens"], 0);
            assert_eq!(stopped["evaluated_tokens"], 4);
            assert_eq!(stopped["pending_tokens"], 1);
            assert_eq!(
                r[0]["statistics"]["generation"]["reason"],
                json!({"kind":"stop_token","token_id":47})
            );
        }
        let mut child = command(attention, "3")
            .arg("--interactive")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut input = child.stdin.take().unwrap();
        for request in [
            json!({"tokens":[1,7,13,3],"max_new_tokens":0}),
            json!({"tokens":[1,7,13,3],"max_new_tokens":0}),
            json!({"action":"continue","max_new_tokens":4}),
            json!({"action":"continue","max_new_tokens":2}),
            json!({"tokens":[64],"max_new_tokens":1}),
            json!({"action":"reset"}),
            json!({"tokens":[1,7,13,3],"max_new_tokens":1}),
        ] {
            writeln!(input, "{request}").unwrap();
        }
        drop(input);
        let r = records(&child.wait_with_output().unwrap());
        assert_eq!(r[0]["event"], "ready");
        execution(&r[0]["execution"], attention, 3);
        let done: Vec<_> = r
            .iter()
            .filter(|e| e["event"] == "done")
            .map(|e| &e["statistics"]["generation"])
            .collect();
        assert_eq!(done.len(), 5);
        assert_eq!(
            r.iter()
                .filter(|e| e["event"] == "token")
                .map(|e| e["token_id"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            [47; 7]
        );
        for (result, evaluated, pending) in [
            (done[0], 4, 0),
            (done[1], 4, 0),
            (done[2], 7, 1),
            (done[3], 9, 1),
            (done[4], 4, 1),
        ] {
            assert_eq!(result["evaluated_tokens"], evaluated);
            assert_eq!(result["pending_tokens"], pending);
        }
        assert_eq!(done[0]["prefill_tokens"], 4);
        assert_eq!(done[1]["reused_tokens"], 4);
        assert_eq!(done[1]["prefill_tokens"], 0);
        assert_eq!(done[2]["sampled_tokens"], 4);
        assert_eq!(done[3]["reused_tokens"], 7);
        assert_eq!(done[3]["prefill_tokens"], 1);
        assert_eq!(done[4]["reused_tokens"], 0);
        assert_eq!(r.iter().filter(|e| e["event"] == "error").count(), 1);
        assert_eq!(r.iter().filter(|e| e["event"] == "reset").count(), 1);
        for e in r.iter().filter(|e| e["event"] == "done") {
            execution(&e["statistics"]["execution"], attention, 3);
        }
    }
    println!(
        "MLX_CLI {}",
        json!({"single_shot_processes":8,"interactive_processes":2,"expected_token_id":47,"weights_bytes":18152,"real_checkpoint_quality_qualified":false,"performance_qualified":false,"iphone_qualified":false})
    );
}
