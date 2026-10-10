use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_toklet"))
        .args(args)
        .output()
        .expect("Toklet CLI should run")
}

#[test]
fn help_is_available_without_a_model() {
    for args in [&[][..], &["--help"], &["-h"]] {
        let output = run(args);
        assert!(output.status.success());
        assert!(output.stderr.is_empty());
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("Usage: toklet [OPTIONS]"));
        assert!(stdout.contains("Generation is greedy"));
    }
}

#[test]
fn version_matches_the_package() {
    for arg in ["--version", "-V"] {
        let output = run(&[arg]);
        assert!(output.status.success());
        assert!(output.stderr.is_empty());
        assert_eq!(
            output.stdout,
            format!("toklet {}\n", env!("CARGO_PKG_VERSION")).as_bytes()
        );
    }
}

#[test]
fn invalid_arguments_fail_without_success_output() {
    for args in [
        &["generate"][..],
        &["--help", "extra"],
        &["--version", "extra"],
    ] {
        let output = run(args);
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8(output.stderr).unwrap().contains("error:"));
    }
}

#[cfg(unix)]
#[test]
fn non_unicode_arguments_return_an_error_instead_of_panicking() {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt};

    let output = Command::new(env!("CARGO_BIN_EXE_toklet"))
        .arg(OsString::from_vec(vec![0xff]))
        .output()
        .expect("Toklet CLI should run");
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8(output.stderr).unwrap().contains("error:"));
}

mod support;
use safetensors::{Dtype, tensor::TensorView};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::Stdio,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Artifact {
    root: PathBuf,
}
impl Artifact {
    fn tiny(capacity: usize) -> Self {
        let f = support::Fixture::tiny();
        let root = std::env::temp_dir().join(format!(
            "toklet-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let config = json!({"model_type":"qwen2","architectures":["Qwen2ForCausalLM"],"hidden_act":"silu","hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"vocab_size":64,"max_position_embeddings":capacity,"rms_norm_eps":1e-6,"rope_theta":1e6,"tie_word_embeddings":true,"torch_dtype":"float32"});
        fs::write(
            root.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let mut tensors = BTreeMap::new();
        for name in f.manifest["arrays"]
            .as_object()
            .unwrap()
            .keys()
            .filter(|name| name.starts_with("weights."))
        {
            let target = if name == "weights.embed_tokens" {
                "model.embed_tokens.weight".to_string()
            } else if name == "weights.norm" {
                "model.norm.weight".to_string()
            } else {
                let pieces: Vec<_> = name.split('.').collect();
                let index = pieces[2];
                let op = pieces[3];
                if ["input_layernorm", "post_attention_layernorm"].contains(&op) {
                    format!("model.layers.{index}.{op}.weight")
                } else {
                    let block = if ["q_proj", "k_proj", "v_proj", "o_proj"].contains(&op) {
                        "self_attn"
                    } else {
                        "mlp"
                    };
                    format!("model.layers.{index}.{block}.{op}.{}", pieces[4])
                }
            };
            tensors.insert(
                target,
                (
                    f.shape(name),
                    f.f32(name)
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            );
        }
        let views: Vec<_> = tensors
            .iter()
            .map(|(name, (shape, data))| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::F32, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(views, None, &root.join("model.safetensors")).unwrap();
        // Exercise the fixture's raw ID reader as well as its weight parser.
        assert_eq!(f.u32("cases.prompt.tokens"), [1, 7, 13, 3]);
        Self { root }
    }
    fn command(&self, capacity: usize) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_toklet"));
        c.arg("--model").arg(&self.root).args([
            "--context-capacity",
            &capacity.to_string(),
            "--json",
        ]);
        c
    }
}
impl Drop for Artifact {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}
fn records(output: &Output) -> Vec<Value> {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn local_token_input_streams_frozen_greedy_ids_and_honors_eos_and_stop_ids() {
    let a = Artifact::tiny(32);
    let out = a
        .command(32)
        .args([
            "--tokens",
            "1,7,13,3",
            "--max-new-tokens",
            "4",
            "--backend",
            "cpu",
        ])
        .output()
        .unwrap();
    let r = records(&out);
    assert_eq!(
        r.iter()
            .filter(|e| e["event"] == "token")
            .map(|e| e["token_id"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [47; 4]
    );
    assert_eq!(
        r.last().unwrap()["statistics"]["generation"]["evaluated_tokens"],
        7
    );
    let statistics = &r.last().unwrap()["statistics"];
    for field in [
        "input_formatting_seconds",
        "tokenization_seconds",
        "detokenization_seconds",
    ] {
        assert_eq!(statistics[field], 0.0, "token-only input has no {field}");
    }
    for field in [
        "input_preparation_seconds",
        "publication_seconds",
        "load_seconds",
        "conversion_seconds",
        "tokenizer_load_seconds",
    ] {
        assert!(statistics[field].as_f64().unwrap().is_finite());
        assert!(statistics[field].as_f64().unwrap() >= 0.0);
    }
    fs::write(
        a.root.join("generation_config.json"),
        br#"{"eos_token_id":[47]}"#,
    )
    .unwrap();
    let out = a
        .command(32)
        .args(["--tokens", "1,7,13,3", "--max-new-tokens", "4"])
        .output()
        .unwrap();
    let r = records(&out);
    assert_eq!(r.len(), 1);
    assert_eq!(
        r[0]["statistics"]["generation"]["reason"],
        json!({"kind":"stop_token","token_id":47})
    );
    assert_eq!(r[0]["statistics"]["generation"]["pending_tokens"], 1);
    let out = a
        .command(32)
        .args(["--tokens", "1,7,13,3", "--max-new-tokens", "4", "--no-eos"])
        .output()
        .unwrap();
    assert_eq!(records(&out).len(), 5);
    let path = a.root.join("prompt.u32le");
    fs::write(
        &path,
        [1_u32, 7, 13, 3]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let out = a
        .command(32)
        .arg("--token-file")
        .arg(&path)
        .args(["--max-new-tokens", "4", "--no-eos", "--stop-token", "47"])
        .output()
        .unwrap();
    assert_eq!(records(&out).len(), 1);
    fs::write(&path, [1, 2, 3]).unwrap();
    let out = a
        .command(32)
        .arg("--token-file")
        .arg(&path)
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(out.stdout.is_empty());
}

#[test]
fn interactive_cached_prompt_continuation_rejection_and_reset_keep_one_session() {
    let a = Artifact::tiny(32);
    let mut baseline = None;
    for (flags, projection) in [
        (&[][..], "scalar"),
        (&["--cpu-projection", "scalar"][..], "scalar"),
        (&["--cpu-projection", "tiled"][..], "tiled"),
    ] {
        let mut child = a
            .command(32)
            .args(flags)
            .args(["--interactive", "--max-new-tokens", "4"])
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
            json!({"tokens":[11],"max_new_tokens":1}),
        ] {
            writeln!(input, "{request}").unwrap();
        }
        drop(input);
        let output = child.wait_with_output().unwrap();
        let r = records(&output);
        assert_eq!(r[0]["cpu_projection"], projection);
        assert_eq!(r[0]["backend"], "cpu");
        let done: Vec<_> = r
            .iter()
            .filter(|e| e["event"] == "done")
            .map(|e| &e["statistics"]["generation"])
            .collect();
        assert_eq!(done.len(), 5);
        assert_eq!(done[0]["prefill_tokens"], 4);
        assert_eq!(done[1]["reused_tokens"], 4);
        assert_eq!(done[1]["prefill_tokens"], 0);
        assert_eq!(done[2]["sampled_tokens"], 4);
        assert_eq!(done[3]["reused_tokens"], 7);
        assert_eq!(done[3]["prefill_tokens"], 1);
        assert!(r.iter().any(|e| e["event"] == "error"));
        assert!(r.iter().any(|e| e["event"] == "reset"));
        assert_eq!(done[4]["reused_tokens"], 0);
        let mut semantics = Vec::new();
        for event in &r {
            match event["event"].as_str().unwrap() {
                "ready" => {
                    let mut ready = event.clone();
                    ready.as_object_mut().unwrap().remove("cpu_projection");
                    semantics.push(ready);
                }
                "done" => {
                    assert_eq!(event["statistics"]["cpu_projection"], projection);
                    let mut generation = event["statistics"]["generation"].clone();
                    generation
                        .as_object_mut()
                        .unwrap()
                        .retain(|name, _| !name.ends_with("_seconds"));
                    semantics.push(generation);
                }
                "token" => semantics.push(json!({
                    "event":"token", "token_id":event["token_id"],
                    "index":event["index"], "text":event["text"]
                })),
                _ => semantics.push(event.clone()),
            }
        }
        if let Some(expected) = &baseline {
            assert_eq!(&semantics, expected);
        } else {
            baseline = Some(semantics);
        }
    }
}

#[test]
fn invalid_projection_options_fail_before_model_access() {
    for tail in [
        vec!["--cpu-projection", "metal"],
        vec!["--cpu-projection", "Tiled"],
        vec!["--cpu-projection"],
        vec!["--cpu-projection", "scalar", "--cpu-projection", "tiled"],
    ] {
        let mut args = vec!["--model", "/no-such-toklet-model", "--tokens", "1"];
        args.extend(tail);
        let out = run(&args);
        assert_eq!(out.status.code(), Some(2));
        assert!(out.stdout.is_empty());
        assert!(String::from_utf8_lossy(&out.stderr).contains("--cpu-projection"));
    }
}

#[test]
fn invalid_cli_requests_and_over_capacity_fail_before_emitting_tokens() {
    let a = Artifact::tiny(32);
    for tail in [
        vec!["--tokens", "1,7,13,3", "--max-new-tokens", "29"],
        vec!["--tokens", "64", "--max-new-tokens", "1"],
        vec!["--tokens", "1", "--stop-token", "64"],
        vec!["--tokens", "1", "--backend", "metal"],
        vec!["--tokens", "1", "--prompt", "x"],
        vec!["--tokens", "1", "--max-new-tokens", "-1"],
    ] {
        let out = a.command(32).args(tail).output().unwrap();
        assert!(!out.status.success());
        assert!(
            out.stdout.is_empty(),
            "{}",
            String::from_utf8_lossy(&out.stdout)
        );
        assert!(!out.stderr.is_empty());
    }
}

#[cfg(unix)]
#[test]
fn actual_ctrl_c_cancels_decode_then_continues_and_exits_when_idle() {
    let a = Artifact::tiny(2048);
    let mut child = a
        .command(2048)
        .args(["--interactive", "--max-new-tokens", "2000"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    writeln!(input, "{}", json!({"tokens":[1,7,13,3]})).unwrap();
    input.flush().unwrap();
    fn read(output: &mut impl BufRead) -> Value {
        let mut line = String::new();
        assert!(output.read_line(&mut line).unwrap() > 0);
        serde_json::from_str(&line).unwrap()
    }
    assert_eq!(read(&mut output)["event"], "ready");
    let first = read(&mut output);
    assert_eq!(first["event"], "token");
    assert!(
        Command::new("/bin/kill")
            .args(["-INT", &child.id().to_string()])
            .status()
            .unwrap()
            .success()
    );
    let done = loop {
        let r = read(&mut output);
        if r["event"] == "done" {
            break r;
        }
    };
    let before = &done["statistics"]["generation"];
    assert_eq!(before["reason"]["kind"], "cancelled");
    assert!(before["sampled_tokens"].as_u64().unwrap() < 2000);
    writeln!(input, "{}", json!({"action":"continue","max_new_tokens":2})).unwrap();
    input.flush().unwrap();
    let after = loop {
        let r = read(&mut output);
        if r["event"] == "done" {
            break r;
        }
    };
    let after = &after["statistics"]["generation"];
    assert_eq!(after["sampled_tokens"], 2);
    assert_eq!(after["reused_tokens"], before["evaluated_tokens"]);
    assert_eq!(after["prefill_tokens"], before["pending_tokens"]);
    assert!(
        Command::new("/bin/kill")
            .args(["-INT", &child.id().to_string()])
            .status()
            .unwrap()
            .success()
    );
    assert_eq!(child.wait().unwrap().code(), Some(130));
    drop(input);
}

#[cfg(unix)]
#[test]
fn ctrl_c_during_prefill_can_resume_all_pending_input() {
    let a = Artifact::tiny(2048);
    let mut child = a
        .command(2048)
        .args(["--interactive", "--prefill-chunk", "1"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    fn read(output: &mut impl BufRead) -> Value {
        let mut line = String::new();
        assert!(output.read_line(&mut line).unwrap() > 0);
        serde_json::from_str(&line).unwrap()
    }
    assert_eq!(read(&mut output)["event"], "ready");
    let tokens: Vec<_> = (0..1920).map(|i| i % 64).collect();
    writeln!(input, "{}", json!({"tokens":tokens,"max_new_tokens":1})).unwrap();
    input.flush().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(10));
    assert!(
        Command::new("/bin/kill")
            .args(["-INT", &child.id().to_string()])
            .status()
            .unwrap()
            .success()
    );
    let before = read(&mut output);
    assert_eq!(before["event"], "done");
    assert_eq!(
        before["statistics"]["generation"]["reason"]["kind"],
        "cancelled"
    );
    let gen_before = &before["statistics"]["generation"];
    assert_eq!(gen_before["sampled_tokens"], 0);
    assert!(gen_before["evaluated_tokens"].as_u64().unwrap() < 1920);
    if gen_before["history_tokens"] == 0 {
        writeln!(input, "{}", json!({"tokens":tokens,"max_new_tokens":0})).unwrap();
    } else {
        writeln!(input, "{}", json!({"action":"continue","max_new_tokens":0})).unwrap();
    }
    input.flush().unwrap();
    let after = read(&mut output);
    assert_eq!(after["event"], "done");
    assert_eq!(after["statistics"]["generation"]["evaluated_tokens"], 1920);
    assert_eq!(after["statistics"]["generation"]["pending_tokens"], 0);
    drop(input);
    assert!(child.wait().unwrap().success());
}
