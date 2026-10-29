//! Explicit real Gemma3 diagnostic capture; no app admission or quality claim.
#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
mod contract {
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{fs::File, io::Read, path::Path};
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    pub const VOCAB: usize = 262144;
    pub const INPUT_SHA: &str = "30c67871d9de7086d32e8c250416ba2134a4c65c8f54173f6cd17401aebba4ac";
    pub const PROTOCOL_SHA: &str =
        "035bd42111d8e702a65a7442a805a3cffac6c04517c90a7cf95ccdc86e668c68";
    pub const REPO: &str = "mlx-community/gemma-3-270m-it-4bit";
    pub const REV: &str = "ff1143e3a10547c9f2129e94ca37059b096b23f4";
    pub const LAYERS: [&str; 22] = [
        "input_norm",
        "q_projection",
        "k_projection",
        "v_projection",
        "q_norm",
        "k_norm",
        "q_rotary",
        "k_rotary",
        "attention_scores",
        "attention_probabilities",
        "attention_context",
        "attention_projection",
        "post_attention_norm",
        "attention_residual",
        "pre_ff_norm",
        "gate_projection",
        "up_projection",
        "gate_gelu",
        "gated",
        "down_projection",
        "post_ff_norm",
        "output",
    ];
    pub fn pins() -> Value {
        serde_json::from_str(r#"{"README.md":{"bytes":1200,"sha256":"2f9fd1ee79d22ff16c563909ca7d42a93bbbb3658a0d24c8b1b41732957ec92d"},"added_tokens.json":{"bytes":35,"sha256":"50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946"},"chat_template.jinja":{"bytes":1532,"sha256":"7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"},"config.json":{"bytes":1653,"sha256":"8fa17cb1689800679eeb5b19ab2d2ef5d425e16cf9f36f59925f130ed976ea38"},"generation_config.json":{"bytes":173,"sha256":"f3d7bb125d6b8751b8fc57c7155c10d6716ebe5a64900a27f56ca236c1c0c241"},"model.safetensors":{"bytes":150939130,"sha256":"bf8b442cff5dd7c9ac6b5695310d5b66a4d0a4c3780f5c774fb90cd0c352bd9d"},"model.safetensors.index.json":{"bytes":34944,"sha256":"3ef9f6408eebccbe14976dd8db24d75d848f01c4430aa8b5924d97782a0930fb"},"special_tokens_map.json":{"bytes":662,"sha256":"2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397"},"tokenizer.json":{"bytes":33384568,"sha256":"4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795"},"tokenizer.model":{"bytes":4689074,"sha256":"1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c"},"tokenizer_config.json":{"bytes":1155373,"sha256":"db4e9d507644beaaf7e3da1ad8184057e9f661adeb67f9473c8b6ad45141f303"}}"#).unwrap()
    }
    pub fn hash_file(path: &Path, expected: &Value) -> Result<()> {
        let mut f = File::open(path)?;
        if f.metadata()?.len() != expected["bytes"].as_u64().ok_or("file bytes")? {
            return Err("file length differs".into());
        }
        let mut h = Sha256::new();
        let mut block = [0_u8; 65536];
        loop {
            let n = f.read(&mut block)?;
            if n == 0 {
                break;
            }
            h.update(&block[..n]);
        }
        if format!("{:x}", h.finalize()) != expected["sha256"].as_str().ok_or("file SHA")? {
            return Err("file digest differs".into());
        }
        Ok(())
    }
    pub fn document(path: &Path, expected: &str, cap: u64) -> Result<Value> {
        if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("expected SHA required".into());
        }
        let mut data = Vec::new();
        File::open(path)?.take(cap + 1).read_to_end(&mut data)?;
        if data.len() as u64 > cap || format!("{:x}", Sha256::digest(&data)) != expected {
            return Err("bounded document identity".into());
        }
        Ok(serde_json::from_slice(&data)?)
    }
    pub fn tokens(v: &Value) -> Result<Vec<u32>> {
        let a = v.as_array().ok_or("token list")?;
        if a.is_empty() || a.len() > 544 {
            return Err("token count".into());
        }
        a.iter()
            .map(|x| {
                x.as_u64()
                    .filter(|&x| x < VOCAB as u64)
                    .map(|x| x as u32)
                    .ok_or_else(|| "token ID".into())
            })
            .collect()
    }
    pub fn partition_b() -> Vec<usize> {
        let mut p = vec![31; 16];
        p.extend([15, 1, 1, 1, 30]);
        p
    }
    // Exact cold source-contract metadata, independent of model execution.
    // Future source file identity remains the explicit preregistered CLI SHA.
    pub fn expected_profile() -> Value {
        serde_json::from_str(r#"{"name":"NativeHalf_certified_projection_serial_attention_v3","main":"Original reconstructed FP32 weights and exactly Half inputs; certified binary64 projection stripes with exact dyadic direct-store fallback; FP32 serial Reference attention reductions/stores; existing FP32 Neumaier RMS/other explicit stores unchanged. Not GPU lane/SIMD emulation.","wide":"same reconstructed FP32 weights widened per operation; FP64 BLAS and sums; FP32 scores/probabilities/logits stores and same Half boundaries","rms":"main sequential column Neumaier FP32, vectorized only across independent rows; wide FP64 squares/reduction; stored FP32(1+rawBF16) scales","rotary":"main FP32 NumPy power/angle/trig/separate products; wide Decimal80 frequency/angle then FP64 trig; Half store","gamma":"selected card/0.26.3 converter no-fold provenance, raw BF16 +FP32 one once; not original-unquantized identity proof","embedding":"original FP32 reconstruction selected row ->Half ->scale25.25 ->Half; tied head original FP32","gelu":"explicit FP32 ufunc order main/FP64 wide thenHalf beforeHalf gating","residual":"Both sites: validate finite Half operands, FP32 add, clamp to [-65504,65504], then Half store; identical explicit FP32 operation in MAIN and WIDE","projection_certificate":{"stripe_outputs":1024,"input_width_bound":2048,"token_bound":544,"products":"finite binary16 times binary32 exact in binary64 (at most35 significand bits)","rounding_model":"audited ordinary binary64 BLAS, round-to-nearest; gamma_n=nu/(1-nu),u=2^-53; n<=2048; no reduced-precision BLAS mode","absolute_error":"outward rounded gamma_n * exact sumabs(Half input row) * maxabs(FP32 weight row)","certificate":"outward dot interval endpoints must round to same finite Half/F32 destination bits","fallback":"exact signed dyadic sum in units2^-173, direct nearest-ties-even destination word; no FP64 intermediary","finite":"active inputs/coefficients/output finite; overflow errors; exact zero canonical+0, nonzero underflow sign retained","witnesses":"first32 fallback exact integers/input+weight hashes, aggregate counts for all calls; no native target values"},"attention":"MAIN explicit increasing-feature FP32 dot then FP32 sqrt/divide and score store, visible max, FP32 exp, increasing-key FP32 mass/divide and probability store, visible increasing-key FP32 product+sum thenHalf. CPU transcendentals are not claimed bit-identical to Metal. Raw masked scores retained; masked probabilities zero. WIDE unchanged."}"#).unwrap()
    }
    pub fn expected_policy() -> Value {
        serde_json::from_str(r#"{"version":"legacy_fixed_limits_new_native_store_center_v1","fatal_reference":"main","scope":"each exact case/absolute selected row/stage and frozen teacher distribution","limits_sha256":"f40fa2275a9c9f193287f32f2764a3dbf9e4d8836a56706583df1d5555151bdd","wide":"diagnostic only; no native-WIDE/greedy/text gate","source_wide_identity":"every current WIDE array must have the historical WIDE hash; this proves the unchanged independent path, not native acceptance"}"#).unwrap()
    }
    pub fn expected_gate_basis() -> Value {
        serde_json::from_str(r#"{"format":"toklet-gemma3-legacy-numeric-limits-v1","file_pin":{"bytes":441159,"sha256":"f40fa2275a9c9f193287f32f2764a3dbf9e4d8836a56706583df1d5555151bdd"},"source_manifest_pin":{"bytes":2531015,"sha256":"60617b26647d2d4a6c96c41ac8675d978cfdec6dc251545ae308e52dd01c9033"},"group_count":3232,"scope":"same exactcase/row/stage names and teacher prefixes; MAIN center changes uniformly, literal old atol/rtol numbers do not change","comparison":"abs(native - successor_MAIN) <= legacy_atol + legacy_rtol*abs(successor_MAIN)","new_source_envelope":"diagnostic only; cannot update either literal numerical limit","history":"v2 failure92 MAIN/98 WIDE remains rejected; successor must be reported as a distinct explicit profile correction"}"#).unwrap()
    }
    pub fn validate(input: &Value, source: &Value) -> Result<()> {
        if input["format"] != "toklet-gemma3-real-input-contract-v1"
            || source["format"] != "toklet-mlx-gemma3-real-reference-v3"
            || source["source_reference_executed"] != true
            || source["target_gpu_executed"] != false
            || source["protocol_sha256"] != PROTOCOL_SHA
            || source["input_sha256"] != INPUT_SHA
            || source["model"]["repository"] != REPO
            || source["model"]["revision"] != REV
            || source["model"]["file_pins"] != pins()
            || source["profile"] != expected_profile()
            || source["policy"] != expected_policy()
            || source["gate_basis"] != expected_gate_basis()
            || source["cases"] != input["cases"]
        {
            return Err("fixed source contract differs".into());
        }
        let cases = input["cases"].as_array().ok_or("cases")?;
        if cases.len() != 3
            || cases[0]["name"] != "plain_france"
            || cases[1]["name"] != "plain_sequence"
            || cases[2]["name"] != "window544"
            || cases[0]["selected_rows"] != json!([0, 5])
            || cases[1]["selected_rows"] != json!([0, 10])
            || cases[2]["selected_rows"] != json!([511, 512, 513, 543])
        {
            return Err("case inventory".into());
        }
        for (c, n) in cases.iter().zip([6, 11, 544]) {
            let t = tokens(&c["tokens"])?;
            if t.len() != n || t[0] != 2 {
                return Err("plain input geometry/BOS".into());
            }
        }
        let teachers = source["greedy"].as_array().ok_or("teacher list")?;
        if teachers.len() != 2 {
            return Err("two teacher prompts required".into());
        }
        for (case, teacher) in cases[..2].iter().zip(teachers) {
            validate_teacher(case, teacher)?;
        }
        if teachers
            .iter()
            .map(|t| t["steps"].as_array().map_or(0, Vec::len))
            .sum::<usize>()
            != 32
        {
            return Err("fixed historical teacher group count".into());
        }
        if source["counts"]
            != json!({"graph_stage_groups":3200,"teacher_stage_groups":32,"total_arrays":6464})
        {
            return Err("source graph group count".into());
        }
        Ok(())
    }
    pub fn validate_teacher(case: &Value, teacher: &Value) -> Result<()> {
        if teacher["prompt"] != case["name"] {
            return Err("teacher prompt identity".into());
        }
        let steps = teacher["steps"].as_array().ok_or("teacher steps")?;
        if steps.is_empty() || steps.len() > 16 {
            return Err("bounded teacher count".into());
        }
        let mut prefix = tokens(&case["tokens"])?;
        let mut visible = Vec::new();
        let mut stopped = false;
        for (i, step) in steps.iter().enumerate() {
            if tokens(&step["prefix"])? != prefix {
                return Err("teacher prefix chain".into());
            }
            let selected = step["wide_teacher_token"]
                .as_u64()
                .filter(|&id| id < VOCAB as u64)
                .ok_or("teacher vocabulary")? as u32;
            step["main_greedy_diagnostic"]
                .as_u64()
                .filter(|&id| id < VOCAB as u64)
                .ok_or("main diagnostic vocabulary")?;
            stopped = matches!(selected, 1 | 106);
            if step["stop"] != json!(stopped) || stopped && i + 1 != steps.len() {
                return Err("teacher stop boundary".into());
            }
            if !stopped {
                visible.push(selected);
                prefix.push(selected);
            }
        }
        if teacher["visible_ids"] != json!(visible)
            || teacher["stop_reason"]
                != if stopped {
                    json!("stop_token")
                } else {
                    json!("length")
                }
            || (!stopped && steps.len() != 16)
        {
            return Err("teacher completion".into());
        }
        Ok(())
    }
    pub fn row_hash(v: &[f32]) -> Result<String> {
        if v.len() != VOCAB || v.iter().any(|x| !x.is_finite()) {
            return Err("finite vocabulary row required".into());
        }
        let mut h = Sha256::new();
        for &x in v {
            let bits = if x == 0. { 0_u32 } else { x.to_bits() };
            h.update(bits.to_le_bytes());
        }
        Ok(format!("{:x}", h.finalize()))
    }
    pub fn greedy(v: &[f32]) -> Result<u32> {
        row_hash(v)?;
        let mut best = 0;
        for i in 1..v.len() {
            if v[i] > v[best] {
                best = i;
            }
        }
        Ok(best as u32)
    }
    pub fn stage_name(index: usize) -> Result<String> {
        Ok(match index {
            0 => "embedding_lookup".into(),
            1 => "embedding".into(),
            398 => "final_norm".into(),
            399 => "logits".into(),
            2..=397 => format!("layers.{}.{}", (index - 2) / 22, LAYERS[(index - 2) % 22]),
            _ => return Err("400 stages required".into()),
        })
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn fixed_partitions_stages_and_budgeted_work() {
            assert_eq!(partition_b().iter().sum::<usize>(), 544);
            let mut c = 0;
            let ends: Vec<_> = partition_b()
                .iter()
                .map(|n| {
                    c += n;
                    c
                })
                .collect();
            for x in [512, 513, 514, 544] {
                assert!(ends.contains(&x));
            }
            assert_eq!(17 * 32, 544);
            let names: std::collections::BTreeSet<_> =
                (0..400).map(|i| stage_name(i).unwrap()).collect();
            assert_eq!(names.len(), 400);
            assert_eq!(
                4 * 544 + 2 * (6 + 11) + (6 + 11 + 30) + 2 * (6 + 11 + 30),
                2351
            );
        }
        #[test]
        fn teacher_corruption_is_rejected_before_model_work() {
            let case = json!({"name":"p","tokens":[2,7]});
            let good = json!({"prompt":"p","steps":[{"prefix":[2,7],"wide_teacher_token":106,"main_greedy_diagnostic":9,"stop":true}],"visible_ids":[],"stop_reason":"stop_token"});
            validate_teacher(&case, &good).unwrap();
            for (key, value) in [
                ("prefix", json!([2, 8])),
                ("wide_teacher_token", json!(262144)),
                ("stop", json!(false)),
            ] {
                let mut bad = good.clone();
                bad["steps"][0][key] = value;
                assert!(validate_teacher(&case, &bad).is_err());
            }
            let mut bad = good.clone();
            bad["steps"] = json!([]);
            assert!(validate_teacher(&case, &bad).is_err());
            let mut bad = good.clone();
            bad["steps"] = json!([good["steps"][0], good["steps"][0]]);
            assert!(validate_teacher(&case, &bad).is_err());
        }
        #[test]
        fn complete_source_profile_and_policy_corruption_is_rejected() {
            let cases = [
                json!({"name":"plain_france","tokens":[2,1,1,1,1,1],"selected_rows":[0,5]}),
                json!({"name":"plain_sequence","tokens":[2,1,1,1,1,1,1,1,1,1,1],"selected_rows":[0,10]}),
                json!({"name":"window544","tokens":std::iter::once(2).chain(std::iter::repeat_n(1,543)).collect::<Vec<_>>(),"selected_rows":[511,512,513,543]}),
            ];
            let input = json!({"format":"toklet-gemma3-real-input-contract-v1","cases":cases});
            let teachers: Vec<Value> = cases[..2].iter().map(|case| {
                let mut prefix=case["tokens"].as_array().unwrap().clone();
                let mut steps=Vec::new();
                for _ in 0..16 {
                    steps.push(json!({"prefix":prefix,"wide_teacher_token":7,"main_greedy_diagnostic":9,"stop":false}));
                    prefix.push(json!(7));
                }
                json!({"prompt":case["name"],"steps":steps,"visible_ids":vec![7;16],"stop_reason":"length"})
            }).collect();
            let good = json!({"format":"toklet-mlx-gemma3-real-reference-v3","source_reference_executed":true,"target_gpu_executed":false,
                "protocol_sha256":PROTOCOL_SHA,"input_sha256":INPUT_SHA,"model":{"repository":REPO,"revision":REV,"file_pins":pins()},
                "profile":expected_profile(),"policy":expected_policy(),"gate_basis":expected_gate_basis(),"cases":cases,"greedy":teachers,
                "counts":{"graph_stage_groups":3200,"teacher_stage_groups":32,"total_arrays":6464}});
            validate(&input, &good).unwrap();
            // Test nested policies, not only the advertised profile name.
            for (pointer, value) in [
                ("/format", json!("toklet-mlx-gemma3-real-reference-v2")),
                ("/protocol_sha256", json!("wrong")),
                ("/profile/main", json!("FP32 BLAS")),
                ("/profile/rms", json!("FP64 RMS")),
                ("/profile/attention", json!("FP32 BLAS attention")),
                (
                    "/profile/projection_certificate/stripe_outputs",
                    json!(2048),
                ),
                ("/profile/residual", json!("unchecked Half")),
                ("/policy/fatal_reference", json!("wide")),
                ("/policy/version", json!("recomputed_envelope")),
                ("/policy/limits_sha256", json!("wrong")),
                ("/gate_basis/file_pin/sha256", json!("wrong")),
                ("/gate_basis/source_manifest_pin/sha256", json!("wrong")),
                ("/gate_basis/group_count", json!(3231)),
                ("/counts/teacher_stage_groups", json!(31)),
                ("/source_reference_executed", json!(false)),
                ("/target_gpu_executed", json!(true)),
            ] {
                let mut bad = good.clone();
                *bad.pointer_mut(pointer).unwrap() = value;
                assert!(validate(&input, &bad).is_err(), "{pointer}");
            }
            let mut bad = good.clone();
            bad["policy"]["atol_floor"] = json!(0.5);
            assert!(validate(&input, &bad).is_err());
            let mut bad = good.clone();
            bad["profile"]
                .as_object_mut()
                .unwrap()
                .remove("projection_certificate");
            assert!(validate(&input, &bad).is_err());
        }
        #[test]
        fn stable_lowest_id_and_numeric_signed_zero() {
            let mut v = vec![0.; VOCAB];
            let h = row_hash(&v).unwrap();
            v[1] = -0.;
            assert_eq!(row_hash(&v).unwrap(), h);
            v[7] = 1.;
            v[13] = 1.;
            assert_eq!(greedy(&v).unwrap(), 7);
            v[0] = f32::NAN;
            assert!(greedy(&v).is_err());
            assert!(tokens(&json!([262144])).is_err());
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple {
    use super::contract::*;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::{
        fs::{self, File, OpenOptions},
        io::{BufWriter, Write},
        path::{Path, PathBuf},
        time::Instant,
    };
    use toklet::{
        metal::{
            Error,
            gemma3::{Memory, Model, Options, Session},
        },
        mlx::gemma3::payload::{self, ContentIdentity, LoadOptions},
    };
    const LIMIT: usize = 6 * 1024 * 1024 * 1024;
    const APP: usize = 512 * 1024 * 1024;
    const RAW_CAP: usize = 256 * 1024 * 1024;
    const RECORD_CAP: usize = 7000;
    struct Capture {
        raw: BufWriter<File>,
        journal: BufWriter<File>,
        bytes: usize,
        journal_bytes: usize,
        records: usize,
        stages: usize,
        commits: usize,
        calls: usize,
    }
    impl Capture {
        fn new(out: &Path) -> Result<Self> {
            fs::create_dir(out)?;
            Ok(Self {
                raw: BufWriter::new(
                    OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(out.join("native.bin"))?,
                ),
                journal: BufWriter::new(
                    OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(out.join("native.jsonl"))?,
                ),
                bytes: 0,
                journal_bytes: 0,
                records: 0,
                stages: 0,
                commits: 0,
                calls: 0,
            })
        }
        fn event(&mut self, v: Value) -> Result<()> {
            let bytes = serde_json::to_vec(&v)?;
            let end = self
                .journal_bytes
                .checked_add(bytes.len())
                .and_then(|n| n.checked_add(1))
                .ok_or("journal overflow")?;
            if bytes.len() > 1024 * 1024 || end > 16 * 1024 * 1024 {
                return Err("bounded journal exceeded".into());
            }
            self.journal.write_all(&bytes)?;
            self.journal.write_all(b"\n")?;
            self.journal.flush()?;
            self.journal_bytes = end;
            Ok(())
        }
        #[allow(clippy::too_many_arguments)] // Bounded diagnostic record fields, outside numerical/timed paths.
        fn save(
            &mut self,
            label: &str,
            kind: &str,
            row: usize,
            stage: &str,
            shape: &[usize],
            key_start: usize,
            values: &[f32],
        ) -> Result<()> {
            let bytes = values.len().checked_mul(4).ok_or("raw size overflow")?;
            if self.records >= RECORD_CAP
                || self.bytes.checked_add(bytes).is_none_or(|b| b > RAW_CAP)
                || shape.iter().product::<usize>() != values.len()
            {
                return Err("raw capture bound/shape".into());
            }
            let offset = self.bytes;
            self.event(json!({"kind":"capture_started","label":label,"capture_kind":kind,"row":row,"stage":stage,"shape":shape,"absolute_key_start":key_start,"offset":offset,"bytes":bytes}))?;
            let mut hash = Sha256::new();
            let mut finite = true;
            let mut block = [0_u8; 16384];
            for part in values.chunks(4096) {
                for (i, &x) in part.iter().enumerate() {
                    block[i * 4..i * 4 + 4].copy_from_slice(&x.to_bits().to_le_bytes());
                    finite &= x.is_finite();
                }
                let b = &block[..part.len() * 4];
                self.raw.write_all(b)?;
                hash.update(b);
            }
            self.raw.flush()?;
            self.bytes += bytes;
            self.records += 1;
            self.event(json!({"kind":"capture","label":label,"capture_kind":kind,"row":row,"stage":stage,"shape":shape,"absolute_key_start":key_start,"offset":offset,"bytes":bytes,"dtype":"f32le","sha256":format!("{:x}",hash.finalize()),"finite":finite}))?;
            if !finite {
                return Err("nonfinite saved native row".into());
            }
            Ok(())
        }
        fn step(&mut self, s: &mut Session<'_>, tokens: &[u32], label: &str) -> Result<()> {
            let before = s.committed_tokens()?.to_vec();
            let start = Instant::now();
            let result = s.forward(tokens);
            let elapsed = start.elapsed().as_secs_f64();
            self.calls += 1;
            let after = s.committed_tokens().ok().map(|history| history.to_vec());
            self.event(json!({"kind":"forward","label":label,"tokens":tokens,"before":before,"after":after,"poisoned":s.is_poisoned(),"error":result.as_ref().err().map(ToString::to_string),"completed_wall_seconds":elapsed,"timing_scope":"diagnostic, not performance qualification"}))?;
            result?;
            let mut expected = before;
            expected.extend_from_slice(tokens);
            if s.committed_tokens()? != expected {
                return Err("committed history differs".into());
            }
            self.commits += tokens.len();
            Ok(())
        }
    }
    fn memory(m: Memory) -> Value {
        json!({"packed_weights":m.packed_weights,"effective_norms":m.effective_norms,"zero_bias":m.zero_bias,"cache":m.cache,"scratch":m.scratch,"logits":m.logits,"tokens_and_history":m.tokens_and_history,"diagnostic_gpu":m.diagnostic_gpu,"diagnostic_readback_reserved":m.diagnostic_readback_reserved,"metadata":m.metadata,"shared_framework_allowance":m.shared_framework_allowance,"external_resident":m.external_resident,"admitted":m.admitted,"cold_peak":m.cold_peak})
    }
    fn options(diagnostic: bool) -> Options {
        Options {
            memory_limit_bytes: LIMIT,
            external_resident_bytes: APP,
            all_logits: diagnostic,
            diagnostics: diagnostic,
        }
    }
    fn digest(s: &str) -> Result<[u8; 32]> {
        if s.len() != 64 {
            return Err("digest length".into());
        }
        let mut out = [0; 32];
        for (i, b) in out.iter_mut().enumerate() {
            *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16)?;
        }
        Ok(out)
    }
    fn endpoints(folder: &Path) -> Result<()> {
        for (name, want) in pins().as_object().ok_or("file pins")? {
            hash_file(&folder.join(name), want)?;
        }
        Ok(())
    }
    fn model(folder: &Path) -> Result<Model> {
        endpoints(folder)?;
        let config = fs::read(folder.join("config.json"))?;
        let index = fs::read(folder.join("model.safetensors.index.json"))?;
        let identity = ContentIdentity {
            config_sha256: digest(
                "8fa17cb1689800679eeb5b19ab2d2ef5d425e16cf9f36f59925f130ed976ea38",
            )?,
            index_sha256: digest(
                "3ef9f6408eebccbe14976dd8db24d75d848f01c4430aa8b5924d97782a0930fb",
            )?,
            header_sha256: digest(
                "994efdf1aa3b4f6bc39d7935b1286380c6f9f49e926e02405fed6ed72f6e578f",
            )?,
            container_sha256: digest(
                "bf8b442cff5dd7c9ac6b5695310d5b66a4d0a4c3780f5c774fb90cd0c352bd9d",
            )?,
            container_bytes: 150939130,
        };
        let store = payload::load(
            &config,
            &index,
            File::open(folder.join("model.safetensors"))?,
            LoadOptions {
                identity,
                memory_limit_bytes: LIMIT - APP,
            },
        )?;
        let resident = Model::from_payload_raw_gamma(&store, 544, APP, LIMIT)?;
        drop(store);
        drop(config);
        drop(index);
        Ok(resident)
    }
    fn capture_selected(
        s: &Session<'_>,
        label: &str,
        selected: &[usize],
        scratch: &mut [f32],
        out: &mut Capture,
    ) -> Result<()> {
        if s.diagnostic_count() != 400 {
            return Err("native stage count differs".into());
        }
        for index in 0..400 {
            let info = s.diagnostic_info(index)?;
            let expected = stage_name(index)?;
            let name = info.layer.map_or_else(
                || info.stage.to_string(),
                |l| format!("layers.{l}.{}", info.stage),
            );
            if name != expected || info.values() > scratch.len() {
                return Err("native stage inventory/extent differs".into());
            }
            let rows =
                if info.stage == "attention_scores" || info.stage == "attention_probabilities" {
                    info.shape[1]
                } else {
                    info.shape[0]
                };
            let wanted: Vec<_> = selected
                .iter()
                .copied()
                .filter(|r| {
                    *r >= info.absolute_query_start && *r < info.absolute_query_start + rows
                })
                .collect();
            if wanted.is_empty() {
                continue;
            }
            s.copy_diagnostic(index, &mut scratch[..info.values()])?;
            for row in wanted {
                let local = row - info.absolute_query_start;
                let (shape, values) = if info.stage == "attention_scores"
                    || info.stage == "attention_probabilities"
                {
                    let keys = info.shape[2];
                    let mut v = Vec::with_capacity(4 * keys);
                    for h in 0..4 {
                        let at = (h * rows + local) * keys;
                        v.extend_from_slice(&scratch[at..at + keys]);
                    }
                    (vec![4, 1, keys], v)
                } else {
                    let width = info.shape[1] * info.shape[2];
                    let shape = if info.rank == 3 {
                        vec![1, info.shape[1], info.shape[2]]
                    } else {
                        vec![1, info.shape[1]]
                    };
                    (shape, scratch[local * width..(local + 1) * width].to_vec())
                };
                out.save(
                    label,
                    "source_stage",
                    row,
                    &name,
                    &shape,
                    info.absolute_key_start,
                    &values,
                )?;
                out.stages += 1;
            }
        }
        Ok(())
    }
    fn compare_row(
        label: &str,
        row: usize,
        values: &[f32],
        expected: &str,
        out: &mut Capture,
    ) -> Result<()> {
        let actual = match row_hash(values) {
            Ok(h) => h,
            Err(e) => {
                let _ = out.save(
                    label,
                    "nonfinite_current_row",
                    row,
                    "logits",
                    &[1, VOCAB],
                    0,
                    values,
                );
                return Err(e);
            }
        };
        if actual != expected {
            out.save(
                label,
                "equality_failure_current_row",
                row,
                "logits",
                &[1, VOCAB],
                0,
                values,
            )?;
            out.event(json!({"kind":"equality_failure","label":label,"row":row,"expected_numeric_sha256":expected,"actual_numeric_sha256":actual,"historical_expected_raw":"earlier authenticated row hash only unless selected-stage capture exists"}))?;
            return Err("exact numeric row equality differs".into());
        }
        Ok(())
    }
    fn reset(
        s: &mut Session<'_>,
        label: &str,
        scratch: &mut [f32],
        out: &mut Capture,
    ) -> Result<()> {
        s.reset()?;
        let history = s.committed_tokens()?.to_vec();
        let no_logits = s.copy_logits(&mut scratch[..VOCAB]);
        out.event(json!({"kind":"reset","label":label,"history":history,"poisoned":s.is_poisoned(),"copy_logits_error":no_logits.as_ref().err().map(ToString::to_string)}))?;
        if !history.is_empty()
            || s.is_poisoned()
            || !matches!(
                no_logits,
                Err(Error::Options("Gemma3 no logits/output shape"))
            )
        {
            return Err("reset prefix/publication differs".into());
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)] // Explicit frozen schedule inputs; no numerical dispatch abstraction.
    fn schedule(
        s: &mut Session<'_>,
        input: &[u32],
        parts: &[usize],
        selected: &[usize],
        label: &str,
        expected: Option<&[String]>,
        equal_until: usize,
        scratch: &mut [f32],
        out: &mut Capture,
    ) -> Result<Vec<String>> {
        if parts.iter().sum::<usize>() != input.len() || parts.iter().any(|&n| n == 0 || n > 32) {
            return Err("fixed schedule geometry".into());
        }
        out.event(json!({"kind":"schedule","label":label,"source_case":label.split('.').next(),"tokens":input,"partitions":parts,"selected_absolute_rows":selected,"exact_numeric_prefix_rows":equal_until}))?;
        reset(s, label, scratch, out)?;
        let mut start = 0;
        let mut hashes = Vec::with_capacity(input.len());
        for &n in parts {
            out.step(s, &input[start..start + n], label)?;
            s.copy_all_logits(&mut scratch[..n * VOCAB])?;
            for local in 0..n {
                let row = start + local;
                let values = &scratch[local * VOCAB..(local + 1) * VOCAB];
                let hash = match row_hash(values) {
                    Ok(h) => h,
                    Err(e) => {
                        let _ = out.save(
                            label,
                            "nonfinite_current_row",
                            row,
                            "logits",
                            &[1, VOCAB],
                            0,
                            values,
                        );
                        return Err(e);
                    }
                };
                out.event(json!({"kind":"row","label":label,"row":row,"numeric_sha256":hash,"vocabulary":VOCAB,"finite":true}))?;
                if let Some(expected) = expected
                    && row < equal_until
                {
                    compare_row(label, row, values, &expected[row], out)?;
                }
                hashes.push(hash);
            }
            capture_selected(s, label, selected, scratch, out)?;
            start += n;
        }
        Ok(hashes)
    }
    fn rejection(
        s: &mut Session<'_>,
        tokens: &[u32],
        label: &str,
        scratch: &mut [f32],
        out: &mut Capture,
    ) -> Result<()> {
        let before = s.committed_tokens()?.to_vec();
        let rows = s.diagnostic_info(399)?.shape[0];
        s.copy_all_logits(&mut scratch[..rows * VOCAB])?;
        let before_hashes: Vec<_> = scratch[..rows * VOCAB]
            .chunks(VOCAB)
            .map(row_hash)
            .collect::<Result<_>>()?;
        let result = s.forward(tokens);
        let after = s.committed_tokens()?.to_vec();
        s.copy_all_logits(&mut scratch[..rows * VOCAB])?;
        let after_hashes: Vec<_> = scratch[..rows * VOCAB]
            .chunks(VOCAB)
            .map(row_hash)
            .collect::<Result<_>>()?;
        out.event(json!({"kind":"rejection","label":label,"request":tokens,"before_history":before,"after_history":after,"before_published_row_hashes":before_hashes,"after_published_row_hashes":after_hashes,"published_rows":rows,"error":result.as_ref().err().map(ToString::to_string),"poisoned":s.is_poisoned()}))?;
        if !matches!(
            result,
            Err(Error::Options("Gemma3 token/count/position admission"))
        ) || before != after
            || before_hashes != after_hashes
            || s.is_poisoned()
        {
            out.save(
                label,
                "rejection_failure_current_publication",
                before.len() - rows,
                "logits",
                &[rows, VOCAB],
                0,
                &scratch[..rows * VOCAB],
            )?;
            return Err("input rejection mutated public state/wrong error".into());
        }
        Ok(())
    }
    fn session_budget(model: &Model, diagnostic: bool, out: &mut Capture) -> Result<()> {
        let required = {
            let s = model.session(options(diagnostic))?;
            out.event(
                json!({"kind":"session_plan","diagnostics":diagnostic,"memory":memory(s.memory())}),
            )?;
            s.memory().admitted
        };
        let mut under = options(diagnostic);
        under.memory_limit_bytes = required.checked_sub(1).ok_or("zero admission")?;
        let failure = model.session(under);
        out.event(json!({"kind":"session_budget_minus_one","diagnostics":diagnostic,"required":required,"limit":required-1,"error":failure.as_ref().err().map(ToString::to_string)}))?;
        if !matches!(failure,Err(Error::Budget{required:r,limit:l})if r==required&&l+1==required) {
            return Err("session exact budget rejection differs".into());
        }
        let mut exact = options(diagnostic);
        exact.memory_limit_bytes = required;
        let s = model.session(exact)?;
        out.event(json!({"kind":"session_exact_budget","diagnostics":diagnostic,"memory":memory(s.memory())}))?;
        if s.memory().admitted != required {
            return Err("session exact budget plan changed".into());
        }
        Ok(())
    }
    fn ordinary_window(
        model: &Model,
        tokens: &[u32],
        expected: &[String],
        scratch: &mut [f32],
        out: &mut Capture,
    ) -> Result<()> {
        let mut s = model.session(options(false))?;
        out.event(json!({"kind":"ordinary_plan","memory":memory(s.memory())}))?;
        for (start, chunk) in tokens.chunks(32).enumerate() {
            out.step(&mut s, chunk, "window544.ordinary_last")?;
            let row = (start + 1) * 32 - 1;
            s.copy_logits(&mut scratch[..VOCAB])?;
            compare_row(
                "window544.ordinary_last",
                row,
                &scratch[..VOCAB],
                &expected[row],
                out,
            )?;
            out.event(json!({"kind":"ordinary_last","row":row,"numeric_sha256":row_hash(&scratch[..VOCAB])?}))?;
        }
        Ok(())
    }
    fn codec(
        folder: &Path,
        inputs_path: &Path,
        input: &Value,
        out: &mut Capture,
    ) -> Result<tokenizers::Tokenizer> {
        let tokenizer = tokenizers::Tokenizer::from_file(folder.join("tokenizer.json"))
            .map_err(|e| e.to_string())?;
        for (text, id) in [
            ("<bos>", 2),
            ("<eos>", 1),
            ("<end_of_turn>", 106),
            ("<image_soft_token>", 262144),
        ] {
            if tokenizer.token_to_id(text) != Some(id) {
                return Err("fixed special-token mapping".into());
            }
        }
        let root = inputs_path.parent().ok_or("input parent")?;
        let probe_info = &input["tokenizer_probe"];
        let proof = document(
            &root.join(probe_info["path"].as_str().ok_or("probe path")?),
            probe_info["sha256"].as_str().ok_or("probe SHA")?,
            1024 * 1024,
        )?;
        for c in input["cases"].as_array().ok_or("cases")? {
            let path = root.join(c["token_file"]["path"].as_str().ok_or("token file")?);
            hash_file(&path, &c["token_file"])?;
            let ids = tokens(&c["tokens"])?;
            let raw = fs::read(path)?;
            let expected: Vec<_> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            if raw != expected {
                return Err("frozen token bytes".into());
            }
            if let Some(text) = c["text"].as_str() {
                let mut encoded = vec![2];
                encoded.extend_from_slice(
                    tokenizer
                        .encode(text, false)
                        .map_err(|e| e.to_string())?
                        .get_ids(),
                );
                let decoded = tokenizer.decode(&ids, false).map_err(|e| e.to_string())?;
                out.event(json!({"kind":"codec_plain","case":c["name"],"encoded":encoded,"decoded":decoded,"add_special_tokens":false,"manual_bos":2}))?;
                if encoded != ids || decoded != c["decoded"].as_str().ok_or("decoded proof")? {
                    return Err("plain codec observation differs".into());
                }
            }
        }
        let controls = proof["chat_controls"].as_array().ok_or("codec controls")?;
        if controls.len() != 4 {
            return Err("four serialized template controls".into());
        }
        for c in controls {
            let rendered = c["rendered"].as_str().ok_or("rendered proof")?;
            let ids = tokenizer
                .encode(rendered, false)
                .map_err(|e| e.to_string())?;
            let decoded = tokenizer
                .decode(ids.get_ids(), false)
                .map_err(|e| e.to_string())?;
            out.event(json!({"kind":"codec_serialized_template","rendered":rendered,"ids":ids.get_ids(),"decoded":decoded,"template_rendering_executed":false,"skip_special_tokens":false}))?;
            if ids.get_ids() != tokens(&c["tokens"])?
                || decoded != c["decoded"].as_str().ok_or("decoded proof")?
            {
                return Err("fixed serialized-template codec observation differs".into());
            }
        }
        Ok(tokenizer)
    }
    fn teacher(
        model: &Model,
        input: &Value,
        source: &Value,
        scratch: &mut [f32],
        out: &mut Capture,
    ) -> Result<()> {
        for (case, trial) in input["cases"].as_array().ok_or("cases")?[..2]
            .iter()
            .zip(source["greedy"].as_array().ok_or("teacher")?)
        {
            let name = case["name"].as_str().ok_or("case name")?;
            let mut s = model.session(options(false))?;
            let prompt = tokens(&case["tokens"])?;
            out.step(&mut s, &prompt, &format!("teacher.{name}.prefill"))?;
            let steps = trial["steps"].as_array().ok_or("teacher steps")?;
            for (i, step) in steps.iter().enumerate() {
                let history = tokens(&step["prefix"])?;
                if s.committed_tokens()? != history {
                    return Err("declared teacher input history differs".into());
                }
                s.copy_logits(&mut scratch[..VOCAB])?;
                let row = history.len() - 1;
                out.save(
                    &format!("teacher.{name}.step_{i}"),
                    "source_teacher",
                    row,
                    "logits",
                    &[1, VOCAB],
                    0,
                    &scratch[..VOCAB],
                )?;
                let native = greedy(&scratch[..VOCAB])?;
                let selected = step["wide_teacher_token"]
                    .as_u64()
                    .filter(|&v| v < VOCAB as u64)
                    .ok_or("teacher selected ID")? as u32;
                out.event(json!({"kind":"teacher","prompt":name,"step":i,"prefix":history,"wide_teacher_id":selected,"native_greedy_diagnostic":native,"agreement_diagnostic_only":native==selected}))?;
                if i + 1 < steps.len() {
                    if matches!(selected, 1 | 106) {
                        return Err("teacher continued past stop".into());
                    }
                    out.step(&mut s, &[selected], &format!("teacher.{name}.append"))?;
                }
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)] // Two fixed diagnostic trials; model and capture owners stay borrowed.
    fn free_once(
        model: &Model,
        prompt: &[u32],
        name: &str,
        trial: usize,
        tokenizer: &tokenizers::Tokenizer,
        scratch: &mut [f32],
        out: &mut Capture,
    ) -> Result<Value> {
        let mut s = model.session(options(false))?;
        let label = format!("free.{name}.trial_{trial}");
        out.step(&mut s, prompt, &label)?;
        let mut sampled = Vec::with_capacity(16);
        let mut visible = Vec::with_capacity(16);
        let mut hashes = Vec::with_capacity(16);
        let mut stopped = false;
        for step in 0..16 {
            s.copy_logits(&mut scratch[..VOCAB])?;
            let row = s.committed_tokens()?.len() - 1;
            out.save(
                &format!("{label}.step_{step}"),
                "native_free",
                row,
                "logits",
                &[1, VOCAB],
                0,
                &scratch[..VOCAB],
            )?;
            let token = greedy(&scratch[..VOCAB])?;
            if tokenizer.id_to_token(token).is_none() {
                return Err("sampled unmapped tokenizer ID".into());
            }
            hashes.push(row_hash(&scratch[..VOCAB])?);
            sampled.push(token);
            out.event(json!({"kind":"free_sample","label":label,"step":step,"sampled_id":token,"committed":s.committed_tokens()?}))?;
            if matches!(token, 1 | 106) {
                stopped = true;
                break;
            }
            visible.push(token);
            if step < 15 {
                out.step(&mut s, &[token], &label)?;
            }
        }
        let pending = usize::from(!stopped);
        let mut committed = prompt.to_vec();
        committed.extend_from_slice(&visible[..visible.len() - pending]);
        let decoded = tokenizer
            .decode(&visible, false)
            .map_err(|e| e.to_string())?;
        let value = json!({"sampled_ids":sampled,"visible_ids":visible,"decoded_visible":decoded,"skip_special_tokens":false,"stop_reason":if stopped{"stop_token"}else{"length"},"generated_pending":pending,"committed":s.committed_tokens()?,"row_hashes":hashes});
        out.event(json!({"kind":"free_complete","label":label,"record":value}))?;
        if s.committed_tokens()? != committed
            || sampled.is_empty()
            || sampled.len() > 16
            || visible.iter().any(|id| matches!(id, 1 | 106))
        {
            return Err("stop/pending/committed generation accounting".into());
        }
        Ok(value)
    }
    fn execute(
        model_path: &Path,
        reference: &Value,
        input: &Value,
        inputs_path: &Path,
        out: &mut Capture,
    ) -> Result<Value> {
        endpoints(model_path)?;
        let tokenizer = codec(model_path, inputs_path, input, out)?;
        let model = model(model_path)?;
        out.event(json!({"kind":"model_loaded","device":model.device_name(),"memory":memory(model.memory()),"source_cpu_payload_dropped":true,"capacity":544,"declared_positions_not_admitted":32768,"shared_tied_table":true}))?;
        if model.config().max_positions != 544
            || model.config().num_layers != 18
            || model.config().head_dim != 256
        {
            return Err("selected resident model geometry".into());
        }
        session_budget(&model, true, out)?;
        session_budget(&model, false, out)?;
        let cases = input["cases"].as_array().ok_or("cases")?;
        let window = tokens(&cases[2]["tokens"])?;
        let expected;
        {
            let mut session = model.session(options(true))?;
            if session.memory().admitted > LIMIT
                || session.memory().diagnostic_readback_reserved < 32 * VOCAB * 4
            {
                return Err("diagnostic admission/readback reserve".into());
            }
            out.event(json!({"kind":"diagnostic_plan","memory":memory(session.memory()),"host_readback_max_bytes":32*VOCAB*4,"extra_caller_allowance":APP}))?;
            let mut scratch = vec![0.; 32 * VOCAB];
            for c in &cases[..2] {
                let ids = tokens(&c["tokens"])?;
                let name = c["name"].as_str().ok_or("case name")?;
                let selected = [0, ids.len() - 1];
                let first = schedule(
                    &mut session,
                    &ids,
                    &[ids.len()],
                    &selected,
                    &format!("{name}.whole"),
                    None,
                    0,
                    &mut scratch,
                    out,
                )?;
                schedule(
                    &mut session,
                    &ids,
                    &vec![1; ids.len()],
                    &selected,
                    &format!("{name}.tokenwise"),
                    Some(&first),
                    ids.len(),
                    &mut scratch,
                    out,
                )?;
                rejection(&mut session, &[], "empty_request", &mut scratch, out)?;
                rejection(
                    &mut session,
                    &[262144],
                    "outside_head_image_token",
                    &mut scratch,
                    out,
                )?;
                rejection(&mut session, &[2; 33], "chunk33", &mut scratch, out)?;
            }
            expected = schedule(
                &mut session,
                &window,
                &[32; 17],
                &[511, 512, 513, 543],
                "window544.partition_a",
                None,
                0,
                &mut scratch,
                out,
            )?;
            rejection(&mut session, &[2], "capacity545", &mut scratch, out)?;
            schedule(
                &mut session,
                &window,
                &partition_b(),
                &[511, 512, 513, 543],
                "window544.partition_b",
                Some(&expected),
                544,
                &mut scratch,
                out,
            )?;
            let mut changed = window.clone();
            changed[543] = input["native_future_last_token"]
                .as_u64()
                .filter(|&id| id < VOCAB as u64)
                .ok_or("future-tail token")? as u32;
            if changed[543] == window[543] {
                return Err("future token unchanged".into());
            }
            schedule(
                &mut session,
                &changed,
                &[32; 17],
                &[],
                "window544.future_tail",
                Some(&expected),
                543,
                &mut scratch,
                out,
            )?;
            reset(&mut session, "after_window_replay", &mut scratch, out)?;
        }
        // No second session or diagnostic readback owner overlaps this phase.
        let mut scratch = vec![0.; VOCAB];
        ordinary_window(&model, &window, &expected, &mut scratch, out)?;
        teacher(&model, input, reference, &mut scratch, out)?;
        for (c, source_teacher) in cases[..2]
            .iter()
            .zip(reference["greedy"].as_array().ok_or("teachers")?)
        {
            let ids = tokens(&c["tokens"])?;
            let name = c["name"].as_str().ok_or("case name")?;
            let first = free_once(&model, &ids, name, 0, &tokenizer, &mut scratch, out)?;
            let second = free_once(&model, &ids, name, 1, &tokenizer, &mut scratch, out)?;
            out.event(json!({"kind":"free_replay","prompt":name,"exact_native_replay":first==second,"source_teacher_visible_ids":source_teacher["visible_ids"],"source_agreement_diagnostic_only":first["visible_ids"]==source_teacher["visible_ids"]}))?;
            if first != second {
                return Err(
                    "native free-generation replay differs; both raw trials retained".into(),
                );
            }
        }
        if out.stages != 6400 || out.commits > 2351 {
            return Err("fixed work/capture count differs".into());
        }
        Ok(
            json!({"capture_completed":true,"source_stage_records":out.stages,"raw_records":out.records,"raw_bytes":out.bytes,"native_committed_tokens":out.commits,"maximum_committed_tokens":2351,"forward_calls":out.calls,"source_reference_numerical_arrays_read":false,"numerical_qualification":false,"independent_checker_required":true,"memory_scope":"planned numerical owners plus 512MiB extra caller allowance; external process observations required","state_scope":"public rejection/history/publication/reset/replay only; inherited tiny numerical fault/poison tests remain separately scoped","text_scope":"fixed codec replay and raw bounded generation, no Chat or output-quality claim"}),
        )
    }
    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args().collect();
        if args.len() != 11 {
            return Err("usage: --model DIR --reference FILE --reference-sha256 SHA --inputs FILE --output NEW_DIR".into());
        }
        let mut map = std::collections::BTreeMap::new();
        for pair in args[1..].chunks(2) {
            if map.insert(pair[0].as_str(), pair[1].as_str()).is_some() {
                return Err("duplicate option".into());
            }
        }
        if map.keys().copied().collect::<Vec<_>>()
            != vec![
                "--inputs",
                "--model",
                "--output",
                "--reference",
                "--reference-sha256",
            ]
        {
            return Err("unknown options".into());
        }
        let folder = PathBuf::from(map["--model"]);
        let inputs_path = PathBuf::from(map["--inputs"]);
        let reference_path = PathBuf::from(map["--reference"]);
        let output = PathBuf::from(map["--output"]);
        let input = document(&inputs_path, INPUT_SHA, 1024 * 1024)?;
        let reference = document(&reference_path, map["--reference-sha256"], 16 * 1024 * 1024)?;
        validate(&input, &reference)?;
        let mut capture = Capture::new(&output)?;
        capture.event(json!({"kind":"start","format":"toklet-gemma3-real-native-v2","source_profile":"NativeHalf_certified_projection_serial_attention_v3","source_gate_basis":"legacy_fixed_limits_new_native_store_center_v1","legacy_limits_sha256":"f40fa2275a9c9f193287f32f2764a3dbf9e4d8836a56706583df1d5555151bdd","legacy_source_manifest_sha256":"60617b26647d2d4a6c96c41ac8675d978cfdec6dc251545ae308e52dd01c9033","model":folder,"repository":REPO,"revision":REV,"source_manifest":reference_path,"source_manifest_sha256":map["--reference-sha256"],"input_sha256":INPUT_SHA,"protocol_sha256":PROTOCOL_SHA,"caller_reserve_bytes":APP,"process_cap_bytes":LIMIT,"raw_cap_bytes":RAW_CAP,"maximum_records":RECORD_CAP}))?;
        let result = execute(&folder, &reference, &input, &inputs_path, &mut capture);
        let endpoint = endpoints(&folder)
            .and_then(|()| document(&inputs_path, INPUT_SHA, 1024 * 1024).map(|_| ()))
            .and_then(|()| {
                document(&reference_path, map["--reference-sha256"], 16 * 1024 * 1024).map(|_| ())
            });
        match (result, endpoint) {
            (Ok(summary), Ok(())) => {
                capture.event(json!({"kind":"complete","summary":summary}))?;
                fs::write(
                    output.join("summary.json"),
                    serde_json::to_vec_pretty(&summary)?,
                )?;
                println!("GEMMA3_REAL_CAPTURE {}", summary);
                Ok(())
            }
            (primary, endpoint) => {
                let error = format!(
                    "primary={:?}; endpoint={:?}",
                    primary.as_ref().err().map(ToString::to_string),
                    endpoint.as_ref().err().map(ToString::to_string)
                );
                let record = json!({"capture_completed":false,"error":error,"raw_records":capture.records,"raw_bytes":capture.bytes,"source_stage_records":capture.stages,"native_committed_tokens":capture.commits});
                let _ = capture.event(json!({"kind":"failure","record":record}));
                fs::write(
                    output.join("failure.json"),
                    serde_json::to_vec_pretty(&record)?,
                )?;
                Err(error.into())
            }
        }
    }
}
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    if let Err(error) = apple::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("This explicit diagnostic requires Apple Silicon macOS and --features metal");
    std::process::exit(2);
}
