//! Portable command-boundary controls; no model download or GPU is used.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        let name = format!(
            "toklet-profile-cli-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let root = std::env::temp_dir().join(name);
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
    fn descriptor(&self, name: &str, value: &Value) -> PathBuf {
        let path = self.path(name);
        fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
        path
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn fixture() -> Value {
    let conversation = r#"[{"role":"user","content":"x"}]"#;
    json!({
        "schema_version":1,"workload_id":"portable-cli-control",
        "conversation_json":conversation,
        "conversation_sha256":format!("{:x}",Sha256::digest(conversation.as_bytes())),
        "input_token_count":1,"input_tokens_sha256":"0".repeat(64),
        "maximum_output_tokens":1,"capacity":512,"chunk":32,
        "cache_policy":"reset_chat_each_request",
        "expected_output":{"tokens":[47],"history":2,"committed":1,"pending":1,
                           "reason":1,"stop_token":u32::MAX}
    })
}
fn arguments(files: &Files, workload: &Path) -> Vec<OsString> {
    vec![
        "profile".into(),
        "--model".into(),
        files.path("model-must-not-be-opened").into_os_string(),
        "--workload".into(),
        workload.as_os_str().to_owned(),
        "--evaluation".into(),
        "1".into(),
        "--output".into(),
        files.path("new-output").into_os_string(),
    ]
}
fn invoke(args: &[OsString]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_toklet"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn profile_help_dispatches_without_model_io() {
    for flag in ["--help", "-h"] {
        let output = invoke(&["profile".into(), flag.into()]);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let help = String::from_utf8(output.stdout).unwrap();
        assert!(help.contains("--model") && help.contains("--workload"));
        assert!(help.contains("--evaluation") && help.contains("--output"));
    }
    let output = invoke(&["--help".into()]);
    assert!(output.status.success());
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("toklet profile")
    );
}

#[test]
fn profile_argument_errors_never_create_output() {
    let files = Files::new();
    let path = files.descriptor("valid.json", &fixture());
    let good = arguments(&files, &path);
    let mut cases = vec![vec!["profile".into(), "--unknown".into()]];
    let mut duplicate = good.clone();
    duplicate.extend(["--evaluation".into(), "1".into()]);
    cases.push(duplicate);
    let mut missing = good.clone();
    missing.pop();
    cases.push(missing);
    for value in ["0", "513", "+1", "１"] {
        let mut bad = good.clone();
        bad[6] = value.into();
        cases.push(bad);
    }
    for args in cases {
        let output = invoke(&args);
        assert_eq!(output.status.code(), Some(2));
        assert!(!files.path("new-output").exists());
    }
}

#[test]
fn profile_rejects_bad_descriptors_before_output_or_model_access() {
    let files = Files::new();
    let mut cases = Vec::new();
    let mut bad = fixture();
    bad["conversation_sha256"] = json!("f".repeat(64));
    cases.push(bad);
    let mut bad = fixture();
    bad["expected_output"]["committed"] = json!(0);
    cases.push(bad);
    let mut bad = fixture();
    bad["expected_output"] = Value::Null;
    cases.push(bad);
    let mut bad = fixture();
    bad["input_token_count"] = json!(true);
    cases.push(bad);
    for (index, value) in cases.into_iter().enumerate() {
        let path = files.descriptor(&format!("bad-{index}.json"), &value);
        let output = invoke(&arguments(&files, &path));
        assert_eq!(
            output.status.code(),
            Some(2),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!files.path("new-output").exists());
    }
    let path = files.path("oversize.json");
    fs::write(&path, vec![b' '; 16 * 1024 + 1]).unwrap();
    assert_eq!(invoke(&arguments(&files, &path)).status.code(), Some(2));
    assert!(!files.path("new-output").exists());
}

#[test]
fn invalid_profile_request_preserves_existing_output_files() {
    let files = Files::new();
    let target = files.path("new-output");
    fs::create_dir(&target).unwrap();
    fs::write(target.join("manifest.json"), b"existing user artifact\n").unwrap();
    let mut bad = fixture();
    bad["schema_version"] = json!(2);
    let path = files.descriptor("bad.json", &bad);
    assert_eq!(invoke(&arguments(&files, &path)).status.code(), Some(2));
    assert_eq!(
        fs::read(target.join("manifest.json")).unwrap(),
        b"existing user artifact\n"
    );
}

#[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
#[test]
fn profile_unsupported_build_is_explicit_and_does_not_fall_back() {
    let files = Files::new();
    let path = files.descriptor("valid.json", &fixture());
    let output = invoke(&arguments(&files, &path));
    assert_eq!(output.status.code(), Some(1));
    let error = String::from_utf8(output.stderr).unwrap().to_lowercase();
    assert!(
        error.contains("metal") || error.contains("apple silicon"),
        "{error}"
    );
    assert!(!files.path("new-output").exists());
}
