use super::*;
use serde_json::json;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

#[test]
fn geometry_and_fatal_priority_are_hardware_independent() {
    for (rows, width) in [
        (0, 1),
        (1, 0),
        (33, 1),
        (1, 12289),
        (usize::MAX, 1),
        (1, usize::MAX),
    ] {
        assert!(Shape { rows, width }.lengths().is_err());
    }
    assert_eq!(
        Shape {
            rows: 32,
            width: 12288
        }
        .lengths()
        .unwrap(),
        (393216, 3145728)
    );
    assert_eq!(
        CompletionStatus::parse(0).unwrap(),
        CompletionStatus::CandidateReady
    );
    assert_eq!(
        CompletionStatus::parse(8).unwrap(),
        CompletionStatus::NeedsReference
    );
    for fatal in [1, 2, 4, 16, 23] {
        assert_eq!(
            CompletionStatus::parse(fatal | 8).unwrap(),
            CompletionStatus::Fatal(fatal)
        );
    }
    assert!(CompletionStatus::parse(32).is_err());
    assert!(CertifiedTable::planned_owned_bytes() < 4 * 1024 * 1024);
    assert!(
        validate_material(&ApprovedBytes {
            table: &[],
            certificate: &[],
            native_verification: &[],
            artifact_review: &[]
        })
        .is_err()
    );
}
#[test]
fn scalar_source_and_shader_are_exact_qualified_inputs() {
    assert_eq!(
        format!("{:x}", Sha256::digest(include_bytes!("../../cpu/gelu.rs"))),
        SOURCE_SHA
    );
    assert!(matches(
        include_bytes!("../gemma_gelu.metal"),
        9176,
        SHADER_SHA
    ));
}
fn read_exact_file(path: &Path, len: usize, sha: &str) -> Vec<u8> {
    let mut file = File::open(path).unwrap();
    assert_eq!(file.metadata().unwrap().len(), len as u64);
    let mut raw = vec![0; len];
    file.read_exact(&mut raw).unwrap();
    assert_eq!(file.read(&mut [0; 1]).unwrap(), 0);
    assert!(matches(&raw, len, sha));
    raw
}
fn env_path(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name).unwrap_or_else(|| panic!("explicit fixture requires {name}")),
    )
}
struct Material {
    table: Vec<u8>,
    certificate: Vec<u8>,
    native: Vec<u8>,
    review: Vec<u8>,
}
impl Material {
    fn load() -> Self {
        Self {
            table: read_exact_file(&env_path("TOKLET_GELU_TABLE"), TABLE_BYTES, TABLE_SHA),
            certificate: read_exact_file(&env_path("TOKLET_GELU_CERTIFICATE"), 2309, CERT_SHA),
            native: read_exact_file(
                &env_path("TOKLET_GELU_NATIVE_VERIFICATION"),
                635,
                NATIVE_SHA,
            ),
            review: read_exact_file(
                &env_path("TOKLET_GELU_ARTIFACT_REVIEW"),
                13881,
                ARTIFACT_REVIEW_SHA,
            ),
        }
    }
    fn borrowed(&self) -> ApprovedBytes<'_> {
        ApprovedBytes {
            table: &self.table,
            certificate: &self.certificate,
            native_verification: &self.native,
            artifact_review: &self.review,
        }
    }
}
#[test]
#[ignore = "explicit authenticated external table/evidence preflight; no GPU needed"]
fn cold_material_rejects_modified_table_certificate_source_profile_and_review() {
    let mut m = Material::load();
    validate_material(&m.borrowed()).unwrap();
    for index in [0, 8, 16, TABLE_BYTES - 4] {
        m.table[index] ^= 1;
        assert!(validate_material(&m.borrowed()).is_err());
        m.table[index] ^= 1;
    }
    for part in 0..3 {
        let raw = match part {
            0 => &mut m.certificate,
            1 => &mut m.native,
            _ => &mut m.review,
        };
        raw[0] ^= 1;
        assert!(validate_material(&m.borrowed()).is_err());
        let raw = match part {
            0 => &mut m.certificate,
            1 => &mut m.native,
            _ => &mut m.review,
        };
        raw[0] ^= 1;
    }
    validate_material(&m.borrowed()).unwrap();
}
struct Fixture {
    protocol: Value,
    input: File,
    expected: File,
}
fn n(v: &Value, key: &str) -> usize {
    v[key].as_u64().unwrap().try_into().unwrap()
}
fn authenticated_file(d: &Value, len: usize, sha: &str) -> File {
    assert_eq!(n(d, "bytes"), len);
    assert_eq!(d["sha256"], sha);
    let mut f = File::open(d["path"].as_str().unwrap()).unwrap();
    assert_eq!(f.metadata().unwrap().len(), len as u64);
    let mut digest = Sha256::new();
    let mut chunk = vec![0; 1024 * 1024];
    let mut read = 0usize;
    loop {
        let n = f.read(&mut chunk).unwrap();
        if n == 0 {
            break;
        }
        read += n;
        digest.update(&chunk[..n]);
    }
    assert_eq!(read, len);
    assert_eq!(format!("{:x}", digest.finalize()), sha);
    f
}
impl Fixture {
    fn load() -> Self {
        let raw = read_exact_file(
            &env_path("TOKLET_GELU_CASE_PROTOCOL"),
            93549,
            "e01222401622b10a5f04f16bcd6b6dc35939ed8501a2665870a0e1fbfa0b1e8d",
        );
        let p: Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(p["cases"].as_array().unwrap().len(), 42);
        assert_eq!(
            p["numerical_gate"]["absolute"].as_f64().unwrap().to_bits(),
            1e-5f64.to_bits()
        );
        assert_eq!(
            p["numerical_gate"]["relative"].as_f64().unwrap().to_bits(),
            1e-4f64.to_bits()
        );
        assert_eq!(p["numerical_gate"]["relaxed"].as_bool(), Some(false));
        let input = authenticated_file(
            &p["inputs"],
            3995352,
            "2b864814c889a370a92bc23b8ef8dbb0600650d3bd6e33518e66ee9c3da751a8",
        );
        let expected = authenticated_file(
            &p["frozen_outputs"],
            7993360,
            "47f40b1f16b5b5c12a689207981aa8ef2c467d4425457f23697affd20f72e450",
        );
        Self {
            protocol: p,
            input,
            expected,
        }
    }
}
fn region(file: &mut File, d: &Value, maximum: usize) -> Vec<u8> {
    let count = n(d, "bytes");
    assert!(count <= maximum);
    file.seek(SeekFrom::Start(n(d, "offset") as u64)).unwrap();
    let mut raw = vec![0; count];
    file.read_exact(&mut raw).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&raw)),
        d["sha256"].as_str().unwrap()
    );
    raw
}
fn input(file: &mut File, d: &Value, count: usize) -> Vec<f32> {
    assert_eq!(d["dtype"], "<f");
    let raw = region(file, d, 49152 * 4);
    assert_eq!(raw.len(), count * 4);
    raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn expected(file: &mut File, d: &Value, count: usize) -> Vec<f64> {
    let raw = region(file, d, 49152 * 8);
    if d["dtype"] == "<f" {
        assert_eq!(raw.len(), count * 4);
        raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64)
            .collect()
    } else {
        assert_eq!(d["dtype"], "<d");
        assert_eq!(raw.len(), count * 8);
        raw.chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
}
fn close(a: f32, e: f64) {
    assert!(
        a.is_finite() && e.is_finite() && (f64::from(a) - e).abs() <= 1e-5 + 1e-4 * e.abs(),
        "{a} vs {e}"
    );
}
fn fill(b: &mut Buffer<u32>, data: &[f32]) {
    b.with_mut(|s| {
        s.fill(SENTINEL);
        for (o, x) in s[GUARD..GUARD + data.len()].iter_mut().zip(data) {
            *o = x.to_bits();
        }
    });
}
fn clear(b: &mut Buffer<u32>) {
    b.with_mut(|s| s.fill(SENTINEL));
}
fn guarded(b: &Buffer<u32>, active: usize) -> bool {
    b.with_slice(|s| {
        s[..GUARD]
            .iter()
            .chain(&s[GUARD + active..])
            .all(|&x| x == SENTINEL)
    })
}
struct Scratch {
    gate: Buffer<u32>,
    up: Buffer<u32>,
    activated: Buffer<u32>,
    product: Buffer<u32>,
    status: Buffer<u32>,
    flags: Buffer<u32>,
    intervals: Buffer<u32>,
}
impl Scratch {
    fn new(gpu: &Gpu) -> Self {
        let make = |n| Buffer::new(gpu, n + 2 * GUARD).unwrap();
        let n = 49152;
        Self {
            gate: make(n),
            up: make(n),
            activated: make(n),
            product: make(n),
            status: make(1),
            flags: make(n),
            intervals: make(8 * n),
        }
    }
    fn buffers(&self, n: usize) -> DeviceBuffers<'_> {
        fn r(b: &Buffer<u32>, n: usize) -> BufferRange<'_> {
            BufferRange::new(b, GUARD, n).unwrap()
        }
        DeviceBuffers {
            gate: r(&self.gate, n),
            up: r(&self.up, n),
            activated: r(&self.activated, n),
            product: r(&self.product, n),
            status: r(&self.status, 1),
            lane_flags: r(&self.flags, n),
            intervals: r(&self.intervals, n * 8),
        }
    }
    fn stage(&mut self, gate: &[f32], up: &[f32]) {
        fill(&mut self.gate, gate);
        fill(&mut self.up, up);
        for b in [
            &mut self.activated,
            &mut self.product,
            &mut self.status,
            &mut self.flags,
            &mut self.intervals,
        ] {
            clear(b)
        }
        self.status.with_mut(|s| s[GUARD] = 0);
    }
    fn owned_bytes(&self) -> usize {
        [
            &self.gate,
            &self.up,
            &self.activated,
            &self.product,
            &self.status,
            &self.flags,
            &self.intervals,
        ]
        .iter()
        .map(|b| b.len * 4)
        .sum()
    }
    fn capture(&self, n: usize) -> Vec<u8> {
        let mut raw = Vec::new();
        for (b, count) in [
            (&self.gate, n),
            (&self.up, n),
            (&self.activated, n),
            (&self.product, n),
            (&self.status, 1),
            (&self.flags, n),
            (&self.intervals, 8 * n),
        ] {
            b.with_slice(|s| {
                for word in &s[..GUARD + count + GUARD] {
                    raw.extend_from_slice(&word.to_le_bytes());
                }
            });
        }
        raw
    }
}
fn shape(case: &Value) -> Shape {
    let n = n(case, "elements");
    let rows = case["origin"]["rows"].as_u64().unwrap_or(1) as usize;
    Shape {
        rows,
        width: n / rows,
    }
}
struct Publication<'a> {
    published: &'a [f32],
    activated: &'a [f32],
    status: u32,
    replayed: bool,
    seconds: f64,
}
fn capture(name: &str, force: bool, raw: &[u8], p: Publication<'_>) {
    let dir = env_path("TOKLET_GELU_CAPTURE_DIR");
    std::fs::create_dir_all(&dir).unwrap();
    let suffix = if force { "forced" } else { "normal" };
    let gpu = dir.join(format!("{name}-{suffix}.gpu.u32le"));
    let final_path = dir.join(format!("{name}-{suffix}.published.f32le"));
    std::fs::write(&gpu, raw).unwrap();
    let mut final_bytes = Vec::with_capacity(p.published.len() * 8);
    for a in [p.activated, p.published] {
        for x in a {
            final_bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(&final_path, &final_bytes).unwrap();
    println!(
        "GEMMA_GELU_ENCODE_CAPTURE {}",
        json!({"case":name,"forced":force,"status":p.status,"whole_case_replayed":p.replayed,"native_replay_seconds":p.seconds,
        "gpu_path":gpu,"gpu_bytes":raw.len(),"gpu_sha256":format!("{:x}",Sha256::digest(raw)),
        "publication_path":final_path,"publication_bytes":final_bytes.len(),"publication_sha256":format!("{:x}",Sha256::digest(&final_bytes)),
        "commands":1,"encoders":1,"waits":1,"graph_qualified":false,"performance_qualified":false})
    );
    std::io::stdout().flush().unwrap();
}
#[test]
#[ignore = "explicit release/Apple GPU certified same-command GELU42cases, normal+forced replay, bounded captures"]
fn borrowed_encoder_all_original_cases_reuses_table_and_replays_whole_case() {
    validate_compiled_reference().unwrap();
    let mut fixture = Fixture::load();
    let gpu = Gpu::new().unwrap();
    let table = {
        let material = Material::load();
        assert!(matches!(
            CertifiedTable::from_approved(
                &gpu,
                &material.borrowed(),
                CertifiedTable::planned_owned_bytes() - 1
            ),
            Err(Error::Budget { .. })
        ));
        CertifiedTable::from_approved(
            &gpu,
            &material.borrowed(),
            CertifiedTable::planned_owned_bytes(),
        )
        .unwrap()
    };
    let pipelines = Pipelines::new(&gpu).unwrap();
    table.validate_device(&gpu).unwrap();
    let mut scratch = Scratch::new(&gpu);
    assert!(table.owned_bytes() + scratch.owned_bytes() < 16 * 1024 * 1024);
    let mut cases = 0;
    let mut real = 0;
    let mut fallbacks = 0;
    for case in fixture.protocol["cases"].as_array().unwrap() {
        let count = n(case, "elements");
        let name = case["name"].as_str().unwrap();
        if count == 0 {
            assert_eq!(case["dispatch"], false);
            continue;
        }
        let gate = input(&mut fixture.input, &case["inputs"]["gate"], count);
        let up = input(&mut fixture.input, &case["inputs"]["up"], count);
        for force in [false, true] {
            scratch.stage(&gate, &up);
            let command = gpu.command().unwrap();
            if force {
                pipelines
                    .encode_forced_reference(&command, &table, &scratch.buffers(count), shape(case))
                    .unwrap();
            } else {
                pipelines
                    .encode(&command, &table, &scratch.buffers(count), shape(case))
                    .unwrap();
            }
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            table.check_completed_guards(&command).unwrap();
            for (b, active) in [
                (&scratch.gate, count),
                (&scratch.up, count),
                (&scratch.activated, count),
                (&scratch.product, count),
                (&scratch.status, 1),
                (&scratch.flags, count),
                (&scratch.intervals, count * 8),
            ] {
                assert!(guarded(b, active));
            }
            scratch.gate.with_slice(|s| {
                assert!(
                    s[GUARD..GUARD + count]
                        .iter()
                        .zip(&gate)
                        .all(|(&a, b)| a == b.to_bits())
                )
            });
            scratch.up.with_slice(|s| {
                assert!(
                    s[GUARD..GUARD + count]
                        .iter()
                        .zip(&up)
                        .all(|(&a, b)| a == b.to_bits())
                )
            });
            let status = scratch.status.with_slice(|s| s[GUARD]);
            let disposition = CompletionStatus::parse(status).unwrap();
            let mut published = vec![42.; count];
            let mut activated = published.clone();
            let mut replayed = false;
            let mut seconds = 0.;
            match disposition {
                CompletionStatus::Fatal(bits) => {
                    assert_eq!(bits as usize, n(case, "expected_status"))
                }
                CompletionStatus::NeedsReference => {
                    let clock = std::time::Instant::now();
                    let mut private = vec![0.; count];
                    let mut private_a = private.clone();
                    crate::cpu::gelu_tanh(&gate, &mut private_a).unwrap();
                    crate::cpu::gated_gelu_tanh(&gate, &up, &mut private).unwrap();
                    published.copy_from_slice(&private);
                    activated.copy_from_slice(&private_a);
                    seconds = clock.elapsed().as_secs_f64();
                    replayed = true;
                    fallbacks += 1;
                }
                CompletionStatus::CandidateReady => {
                    scratch.product.with_slice(|s| {
                        for (out, &x) in published.iter_mut().zip(&s[GUARD..GUARD + count]) {
                            *out = f32::from_bits(x)
                        }
                    });
                    scratch.activated.with_slice(|s| {
                        for (out, &x) in activated.iter_mut().zip(&s[GUARD..GUARD + count]) {
                            *out = f32::from_bits(x)
                        }
                    });
                }
            }
            if n(case, "expected_status") != 0 {
                assert!(published.iter().chain(&activated).all(|&x| x == 42.));
            } else {
                assert!(!matches!(disposition, CompletionStatus::Fatal(_)));
                for (key, a) in [
                    ("activated", &activated),
                    ("activated_fp32", &activated),
                    ("output", &published),
                    ("frozen_activated", &activated),
                    ("frozen_output", &published),
                ] {
                    if case["expected"][key].is_null() {
                        continue;
                    }
                    let e = expected(&mut fixture.expected, &case["expected"][key], count);
                    for (&a, &e) in a.iter().zip(&e) {
                        close(a, e)
                    }
                    let buffer = if key == "output" || key == "frozen_output" {
                        &scratch.product
                    } else {
                        &scratch.activated
                    };
                    scratch.flags.with_slice(|flags| {
                        buffer.with_slice(|raw| {
                            for i in 0..count {
                                if flags[GUARD + i] & 2 != 0 {
                                    close(f32::from_bits(raw[GUARD + i]), e[i]);
                                }
                            }
                        })
                    });
                    if replayed && (key == "activated_fp32" || key == "output") {
                        assert!(
                            a.iter()
                                .zip(&e)
                                .all(|(&a, &e)| a.to_bits() == (e as f32).to_bits())
                        );
                    }
                }
                assert!(
                    activated
                        .iter()
                        .zip(&up)
                        .zip(&published)
                        .all(|((&a, &u), &p)| (a * u).to_bits() == p.to_bits())
                );
                scratch.activated.with_slice(|a| {
                    scratch.product.with_slice(|p| {
                        assert!(
                            (0..count).all(|i| (f32::from_bits(a[GUARD + i]) * up[i]).to_bits()
                                == p[GUARD + i])
                        );
                    })
                });
                if case["family"] == "saved-original-bf16-own-input" && !force {
                    assert!(!replayed);
                    real += 1;
                }
            }
            let raw = scratch.capture(count);
            capture(
                name,
                force,
                &raw,
                Publication {
                    published: &published,
                    activated: &activated,
                    status,
                    replayed,
                    seconds,
                },
            );
            cases += 1;
        }
    }
    assert_eq!(cases, 82);
    assert_eq!(real, 24);
    println!(
        "GEMMA_GELU_ENCODE_SUMMARY {}",
        json!({"dispatches":cases,"real_normal_no_fallback":real,"fallback_transactions":fallbacks,"owned_numerical_bytes":table.owned_bytes()+scratch.owned_bytes(),"graph_qualified":false,"performance_qualified":false})
    );
}
#[test]
#[ignore = "explicit Apple GPU borrowed GELU rejection/alias/shape/guard checks"]
fn preflight_rejections_create_no_encoder_and_preserve_status() {
    validate_compiled_reference().unwrap();
    let gpu = Gpu::new().unwrap();
    let material = Material::load();
    let table = CertifiedTable::from_approved(
        &gpu,
        &material.borrowed(),
        CertifiedTable::planned_owned_bytes(),
    )
    .unwrap();
    let pipelines = Pipelines::new(&gpu).unwrap();
    table.validate_device(&gpu).unwrap();
    let mut scratch = Scratch::new(&gpu);
    scratch.stage(&[0.5; 17], &[1.; 17]);
    let command = gpu.command().unwrap();
    let mut b = scratch.buffers(17);
    let original = b.product;
    b.product = b.gate;
    assert!(
        pipelines
            .encode(&command, &table, &b, Shape { rows: 1, width: 17 })
            .is_err()
    );
    b.product = original;
    let flags = b.lane_flags;
    b.lane_flags.len -= 1;
    assert!(
        pipelines
            .encode(&command, &table, &b, Shape { rows: 1, width: 17 })
            .is_err()
    );
    b.lane_flags = flags;
    let intervals = b.intervals;
    b.intervals.offset = usize::MAX;
    assert!(
        pipelines
            .encode(&command, &table, &b, Shape { rows: 1, width: 17 })
            .is_err()
    );
    b.intervals = intervals;
    let status = b.status;
    b.status = BufferRange {
        raw: b.lane_flags.raw,
        offset: b.lane_flags.offset,
        len: 1,
    };
    assert!(
        pipelines
            .encode(&command, &table, &b, Shape { rows: 1, width: 17 })
            .is_err()
    );
    b.status = status;
    let product = b.product;
    let mut into_table = table.range().unwrap();
    into_table.len = 17;
    b.product = into_table;
    assert!(
        pipelines
            .encode(&command, &table, &b, Shape { rows: 1, width: 17 })
            .is_err()
    );
    b.product = product;
    assert!(
        pipelines
            .encode(&command, &table, &b, Shape { rows: 0, width: 17 })
            .is_err()
    );
    assert_eq!(command.status(), MTLCommandBufferStatus::NotEnqueued);
    assert_eq!(scratch.status.with_slice(|s| s[GUARD]), 0);
    // Read-only aliases are valid. One command remains usable after every
    // preflight rejection, and the real operation sees neither stale encoder nor writes.
    b.up = b.gate;
    pipelines
        .encode(&command, &table, &b, Shape { rows: 1, width: 17 })
        .unwrap();
    command.commit();
    command.waitUntilCompleted();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(
        pipelines
            .encode(&command, &table, &b, Shape { rows: 1, width: 17 })
            .is_err()
    );
    table.check_completed_guards(&command).unwrap();
    assert_eq!(
        CompletionStatus::parse(scratch.status.with_slice(|s| s[GUARD])).unwrap(),
        CompletionStatus::CandidateReady
    );
    let mut expected = [0.; 17];
    crate::cpu::gated_gelu_tanh(&[0.5; 17], &[0.5; 17], &mut expected).unwrap();
    scratch.product.with_slice(|s| {
        for (&a, &e) in s[GUARD..GUARD + 17].iter().zip(&expected) {
            close(f32::from_bits(a), f64::from(e))
        }
    });
    for (b, active) in [
        (&scratch.gate, 17),
        (&scratch.up, 17),
        (&scratch.activated, 17),
        (&scratch.product, 17),
        (&scratch.flags, 17),
        (&scratch.intervals, 17 * 8),
        (&scratch.status, 1),
    ] {
        assert!(guarded(b, active));
    }
}
