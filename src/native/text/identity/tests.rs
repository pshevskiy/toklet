use super::*;
use serde_json::{Value, json};
fn fixture() -> Evidence {
    let m = qwen3_manifest().unwrap();
    let f = m.files();
    Evidence {
        profile: IdentityProfile::Qwen3,
        manifest: m.identity(),
        digests: std::array::from_fn(|i| f[i].sha256),
        container_bytes: f[1].bytes,
        ranges: 706,
        reads: 706,
        text_policy: "qwen3-plain-nonthinking-v1",
    }
}
fn report(e: Evidence, instance: u64, loaded: usize) -> Value {
    serde_json::to_value(build(e, instance, loaded).unwrap()).unwrap()
}
#[test]
fn fixed_roles_and_canonical_namespace_exclude_live_correlation() {
    let a = report(fixture(), 1, 512);
    let b = report(fixture(), 99, 16);
    assert_eq!(a["preference_namespace"], b["preference_namespace"]);
    assert_ne!(a["model_instance"], b["model_instance"]);
    assert_ne!(a["loaded_position_limit"], b["loaded_position_limit"]);
    assert_eq!(
        a["preference_namespace"],
        json!({"version":1,"manifest_algorithm":"toklet-full-manifest-v1",
        "manifest_sha256":"5e9aafa305e0e48182a856cafe3ae2b597599033e61b16664f42b09a4af5ff51",
        "backend":"metal_native_dense_v1","load":"qwen3_audited_v1",
        "text":"qwen3-plain-nonthinking-v1","planning":"native_mlx_planning_v1"})
    );
    let roles = json!([
        ["source_metadata_consumed", "codec_config_verified"],
        ["source_bytes_delivered_to_assembled_model"],
        ["source_metadata_consumed"],
        ["tokenizer_parser_consumed"],
        ["fixed_text_policy_verified"],
        ["ancillary_verified"]
    ]);
    let assets = a["assets"].as_array().unwrap();
    assert_eq!(assets.len(), 6);
    for (i, asset) in assets.iter().enumerate() {
        assert_eq!(asset["roles"], roles[i]);
        assert_eq!(asset["sha256"].as_str().unwrap().len(), 64);
    }
    assert_eq!(
        assets
            .iter()
            .map(|f| f["bytes"].as_u64().unwrap())
            .sum::<u64>(),
        644927148
    );
    for field in [
        "current_disk_identity_evaluated",
        "resident_buffer_identity_evaluated",
        "qualification_evaluated",
        "whole_app_device_fit_evaluated",
    ] {
        assert_eq!(a[field], false);
    }
    assert_eq!(a["preferences_require_fresh_admission"], true);
    assert!(!a.as_object().unwrap().contains_key("path"));
    assert!(serde_json::to_vec(&a).unwrap().len() < 8 * 1024);
}
#[test]
fn every_identity_role_and_extent_fails_closed_on_change() {
    for i in 0..6 {
        let mut e = fixture();
        e.digests[i][0] ^= 1;
        assert!(build(e, 1, 512).is_err());
    }
    let mut e = fixture();
    e.manifest[0] ^= 1;
    assert!(build(e, 1, 512).is_err());
    let mut e = fixture();
    e.container_bytes += 1;
    assert!(build(e, 1, 512).is_err());
    for n in [0, 2, crate::weights::content::MAX_RANGES + 1] {
        let mut e = fixture();
        e.ranges = n;
        assert!(build(e, 1, 512).is_err());
    }
    let mut e = fixture();
    e.reads = e.ranges - 1;
    assert!(build(e, 1, 512).is_err());
    let mut e = fixture();
    e.text_policy = "other";
    assert!(build(e, 1, 512).is_err());
    for (instance, loaded) in [(0, 512), (1, 0), (1, 513)] {
        assert!(build(fixture(), instance, loaded).is_err());
    }
    let mut e = fixture();
    e.reads += 1;
    assert!(build(e, 1, 1).is_ok());
}
#[test]
fn source_receipts_compare_full_delivery_facts_not_only_manifest() {
    let f = crate::weights::content::SourceFixture::new();
    let mut source = f.source();
    source.visit_raw("a", |_| Ok(())).unwrap();
    source.visit_raw("b", |_| Ok(())).unwrap();
    let first = source.qwen_content_identity().unwrap();
    assert!(same_source(&first, &first).is_ok());
    source.visit_raw("a", |_| Ok(())).unwrap();
    let second = source.qwen_content_identity().unwrap();
    assert_eq!(
        first.expected_manifest_sha256(),
        second.expected_manifest_sha256()
    );
    assert_eq!(first.container_sha256(), second.container_sha256());
    assert_eq!(same_source(&first, &second).unwrap_err().0, INVALID);
}
#[test]
fn immutable_json_capture_short_copy_and_owner_release() {
    // Pure report-builder output, not an assembled-model getter qualification.
    let report = build(fixture(), 1, 512).unwrap();
    let handle = compatibility::capture(&report).unwrap();
    drop(report);
    unsafe {
        let mut count = 0;
        assert_eq!(
            toklet_mlx_inspect_json(handle, ptr::null_mut(), 0, &mut count, ptr::null_mut()),
            OK
        );
        assert!((1..8192).contains(&count));
        let expected = count;
        let mut short = vec![0x5a; expected - 1];
        assert_eq!(
            toklet_mlx_inspect_json(
                handle,
                short.as_mut_ptr(),
                short.len(),
                &mut count,
                ptr::null_mut()
            ),
            INVALID
        );
        assert_eq!(count, expected);
        assert!(short.iter().all(|&x| x == 0x5a));
        let mut full = vec![0; expected];
        assert_eq!(
            toklet_mlx_inspect_json(
                handle,
                full.as_mut_ptr(),
                full.len(),
                &mut count,
                ptr::null_mut()
            ),
            OK
        );
        let a: Value = serde_json::from_slice(&full).unwrap();
        assert_eq!(a["model_instance"], 1);
        let address = handle as usize;
        std::thread::spawn(move || {
            let mut count = 99;
            assert_eq!(
                toklet_mlx_inspect_json(
                    address as *const CompatibilityHandle,
                    ptr::null_mut(),
                    0,
                    &mut count,
                    ptr::null_mut()
                ),
                WRONG_THREAD
            );
            assert_eq!(count, 0);
            assert_eq!(
                toklet_mlx_inspect_release(address as *mut CompatibilityHandle, ptr::null_mut()),
                WRONG_THREAD
            );
        })
        .join()
        .unwrap();
        let mut second = vec![0; expected];
        assert_eq!(
            toklet_mlx_inspect_json(
                handle,
                second.as_mut_ptr(),
                second.len(),
                &mut count,
                ptr::null_mut()
            ),
            OK
        );
        assert_eq!(full, second);
        assert_eq!(toklet_mlx_inspect_release(handle, ptr::null_mut()), OK);
    }
}

#[test]
fn smol_complete_namespace_and_reference_role_are_profile_specific() {
    let codec = crate::app_chat::content::tests::synthetic_complete_identity(694, true);
    let e = smol_evidence(codec.source(), &codec).unwrap();
    let a = report(e.clone(), 1, 512);
    let b = report(e, 99, 16);
    assert_eq!(a["preference_namespace"], b["preference_namespace"]);
    assert_ne!(
        a["preference_namespace"],
        report(fixture(), 1, 512)["preference_namespace"]
    );
    assert_eq!(a["repository"], codec.source().repository());
    assert_eq!(a["revision"], codec.source().revision());
    assert_eq!(a["preference_namespace"]["load"], "smol_witness_audited_v1");
    assert_eq!(a["preference_namespace"]["text"], "smol-chatml-v1");
    assert_eq!(
        a["assets"][1]["roles"],
        json!([
            "source_bytes_delivered_to_assembled_model",
            "reference_container_digest_anchor"
        ])
    );
    assert_eq!(a["assets"][5]["roles"], json!(["ancillary_verified"]));
    assert_eq!(a["source_container_physical_ranges"], 694);
    assert!(serde_json::to_vec(&a).unwrap().len() < 8 * 1024);
}
#[test]
fn smol_legacy_cross_profile_and_changed_source_facts_reject() {
    let old = crate::app_chat::content::tests::synthetic_complete_identity(694, false);
    assert_eq!(
        smol_evidence(old.source(), &old).err().unwrap().0,
        UNSUPPORTED
    );
    let complete = crate::app_chat::content::tests::synthetic_complete_identity(694, true);
    let other = crate::app_chat::content::tests::synthetic_complete_identity(695, true);
    assert_eq!(
        smol_evidence(other.source(), &complete).err().unwrap().0,
        INVALID
    );
    let baseline = smol_evidence(complete.source(), &complete).unwrap();
    for i in 0..6 {
        let mut e = baseline.clone();
        e.digests[i][0] ^= 1;
        assert!(build(e, 1, 512).is_err());
    }
    let mut e = baseline.clone();
    e.profile = IdentityProfile::Qwen3;
    assert!(build(e, 1, 512).is_err());
    let mut e = baseline.clone();
    e.ranges = 693;
    assert!(build(e, 1, 512).is_err());
    let mut e = baseline.clone();
    e.reads = 693;
    assert!(build(e, 1, 512).is_err());
    let mut e = baseline.clone();
    e.text_policy = "qwen3-plain-nonthinking-v1";
    assert!(build(e, 1, 512).is_err());
    let mut e = baseline;
    e.manifest[0] ^= 1;
    assert!(build(e, 1, 512).is_err());
}
