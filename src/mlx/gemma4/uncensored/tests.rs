use super::*;

const CONFIG: &[u8] =
    include_bytes!("../../../../tests/fixtures/gemma4-heretic-metadata-v1/config.json");
const HEADER: &[u8] =
    include_bytes!("../../../../tests/fixtures/gemma4-heretic-metadata-v1/header.json");

fn plan() -> Plan {
    inspect(CONFIG, HEADER, CONTAINER_BYTES, ModalityPolicy::TextOnly).unwrap()
}

fn altered(edit: impl FnOnce(&mut Value)) -> Result<Plan, Error> {
    let mut header: Value = serde_json::from_slice(HEADER).unwrap();
    edit(&mut header);
    let bytes = serde_json::to_vec(&header).unwrap();
    // Exercise the semantic parser independently of public byte-pin rejection.
    let extent = CONTAINER_BYTES - HEADER_BYTES as u64 + bytes.len() as u64;
    layout(CONFIG, &bytes, extent)
}

#[test]
fn exact_inventory_preserves_mixed_projection_and_excluded_owners() {
    let p = plan();
    let oracle: Value = serde_json::from_slice(include_bytes!(
        "../../../../tests/fixtures/gemma4-heretic-metadata-v1/inventory.json"
    ))
    .unwrap();
    for (map, bytes, name) in [
        (p.text_tensors(), p.text_payload_bytes(), "text"),
        (
            p.unused_shared_kv_tensors(),
            p.unused_shared_kv_bytes(),
            "unused_shared_kv",
        ),
        (p.media_tensors(), p.media_bytes(), "media"),
    ] {
        assert_eq!(json!(map.len()), oracle["partitions"][name]["tensors"]);
        assert_eq!(json!(bytes), oracle["partitions"][name]["bytes"]);
        assert_eq!(map.values().map(|t| t.range.len()).sum::<usize>(), bytes);
    }
    assert_eq!(
        (p.text.len(), p.unused_shared_kv.len(), p.media.len()),
        (1094, 140, 1415)
    );
    assert_eq!(
        (p.text_bytes, p.unused_shared_kv_bytes, p.media_bytes),
        (2_623_765_062, 10_629_120, 946_370_944)
    );
    assert_eq!(p.payload_offset(), 335752);
    assert_eq!(
        p.payload_offset() + p.text_bytes + p.unused_shared_kv_bytes + p.media_bytes,
        CONTAINER_BYTES as usize
    );
    let bf16 = &p.text_tensors()[BF16_PROJECTION];
    assert_eq!(bf16.dtype, Dtype::Bf16);
    assert_eq!(bf16.shape, [8960, 1536]);
    assert_eq!(bf16.range, 3_484_155_792..3_511_680_912);
    assert_eq!(p.matrices().len(), 278);
    let mut packed = 0;
    let mut bf16 = 0;
    for m in p.matrices() {
        match m.storage {
            MatrixStorage::Affine(l) => {
                assert_eq!(
                    (l.bits(), l.group_size(), l.metadata_dtype()),
                    (4, 64, MetadataDtype::BF16)
                );
                assert_ne!(m.name, BF16_PROJECTION);
                packed += 1;
            }
            MatrixStorage::Bf16 { rows, columns } => {
                assert_eq!(m.name, BF16_PROJECTION);
                assert_eq!((rows, columns), (8960, 1536));
                bf16 += 1;
            }
        }
    }
    assert_eq!((packed, bf16), (277, 1));
    assert!(
        p.text_tensors()
            .keys()
            .all(|name| !p.unused_shared_kv_tensors().contains_key(name))
    );
}

#[test]
fn geometry_matches_official_and_prior_affine_route_remains_strict() {
    let official = super::super::bf16::inspect(
        include_bytes!("../../../../tests/fixtures/gemma-unquantized-header-v1/config.json"),
        include_bytes!("../../../../tests/fixtures/gemma-unquantized-header-v1/header.json"),
        super::super::bf16::CONTAINER_BYTES,
        ModalityPolicy::TextOnly,
    )
    .unwrap();
    assert_eq!(plan().layers(), official.layers());
    let header: Value = serde_json::from_slice(HEADER).unwrap();
    let map: BTreeMap<_, _> = header
        .as_object()
        .unwrap()
        .keys()
        .filter(|name| name.as_str() != "__metadata__")
        .map(|name| (name, "model.safetensors"))
        .collect();
    let index = serde_json::to_vec(&json!({"metadata":{"total_size":CONTAINER_BYTES - 8 - HEADER_BYTES as u64},"weight_map":map})).unwrap();
    assert!(
        super::super::inspect(
            CONFIG,
            &index,
            HEADER,
            CONTAINER_BYTES,
            ModalityPolicy::TextOnly
        )
        .is_err()
    );
}

#[test]
fn source_pins_reject_changed_bytes_extent_and_implicit_modalities() {
    for which in 0..2 {
        let mut c = CONFIG.to_vec();
        let mut h = HEADER.to_vec();
        let bytes = if which == 0 { &mut c } else { &mut h };
        for at in [0, bytes.len() / 2, bytes.len() - 1] {
            bytes[at] ^= 1;
            assert!(
                inspect(
                    if which == 0 { bytes } else { CONFIG },
                    if which == 1 { bytes } else { HEADER },
                    CONTAINER_BYTES,
                    ModalityPolicy::TextOnly
                )
                .is_err()
            );
            bytes[at] ^= 1;
        }
    }
    for (c, h, n, policy) in [
        (CONFIG, HEADER, CONTAINER_BYTES, ModalityPolicy::Reject),
        (
            CONFIG,
            HEADER,
            CONTAINER_BYTES - 1,
            ModalityPolicy::TextOnly,
        ),
        (
            CONFIG,
            HEADER,
            CONTAINER_BYTES + 1,
            ModalityPolicy::TextOnly,
        ),
        (
            &CONFIG[..CONFIG.len() - 1],
            HEADER,
            CONTAINER_BYTES,
            ModalityPolicy::TextOnly,
        ),
        (
            CONFIG,
            &HEADER[..HEADER.len() - 1],
            CONTAINER_BYTES,
            ModalityPolicy::TextOnly,
        ),
    ] {
        assert!(inspect(c, h, n, policy).is_err());
    }
}

#[test]
fn inactive_alias_and_graph_options_are_semantic_not_ignored() {
    for input in [
        b"null".as_slice(),
        b"[]",
        b"true",
        b"42",
        br#""string""#,
        b"{}",
    ] {
        assert!(configuration(input).is_err());
    }
    for (field, value) in [
        ("moe_intermediate_size", json!(123)),
        ("enable_moe_block", json!(true)),
        ("num_experts", json!(2)),
        ("hidden_size", json!(1537)),
        ("rms_norm_eps", json!(0.001)),
        ("unknown", Value::Null),
    ] {
        let mut c: Value = serde_json::from_slice(CONFIG).unwrap();
        c["text_config"][field] = value;
        assert!(
            configuration(&serde_json::to_vec(&c).unwrap()).is_err(),
            "{field}"
        );
    }
}

#[test]
fn mixed_matrix_and_unused_shared_kv_shapes_are_exact() {
    assert!(altered(|h| h[BF16_PROJECTION]["dtype"] = json!("U32")).is_err());
    assert!(altered(|h| h[BF16_PROJECTION]["shape"] = json!([1536, 8960])).is_err());
    let extra = "language_model.model.layers.34.self_attn.k_proj.weight";
    assert!(altered(|h| h[extra]["shape"] = json!([256, 192])).is_err());
    assert!(
        altered(|h| {
            h.as_object_mut().unwrap().remove(extra);
        })
        .is_err()
    );
    assert!(
        altered(|h| {
            let v = h.as_object_mut().unwrap().remove(extra).unwrap();
            h["unknown.payload"] = v;
        })
        .is_err()
    );
}

#[test]
fn malformed_spans_metadata_and_complete_coverage_reject() {
    for offsets in [
        json!([8, 4]),
        json!([0, CONTAINER_BYTES]),
        json!([3484155793_u64, 3511680913_u64]),
    ] {
        assert!(altered(|h| h[BF16_PROJECTION]["data_offsets"] = offsets).is_err());
    }
    assert!(altered(|h| h["__metadata__"] = json!({"format":"pt"})).is_err());
    assert!(altered(|h| h[BF16_PROJECTION]["extra"] = json!(true)).is_err());
    assert!(altered(|h| h[BF16_PROJECTION]["shape"] = json!([usize::MAX, 1536])).is_err());
}

#[test]
#[ignore = "explicit local Heretic MLX4 cold-file probe, no tensor payload read"]
fn actual_pinned_file_header_and_cold_samples() {
    use std::io::{Read, Seek, SeekFrom};
    let root =
        std::path::PathBuf::from(std::env::var_os("TOKLET_GEMMA_HERETIC_DIRECTORY").unwrap());
    let config = std::fs::read(root.join("config.json")).unwrap();
    let mut f = std::fs::File::open(root.join("model.safetensors")).unwrap();
    let extent = f.metadata().unwrap().len();
    let mut samples = Vec::with_capacity(10);
    for trial in 0..13 {
        f.seek(SeekFrom::Start(0)).unwrap();
        let clock = std::time::Instant::now();
        let mut n = [0; 8];
        f.read_exact(&mut n).unwrap();
        assert_eq!(u64::from_le_bytes(n), HEADER_BYTES as u64);
        let mut header = vec![0; HEADER_BYTES];
        f.read_exact(&mut header).unwrap();
        let p = inspect(&config, &header, extent, ModalityPolicy::TextOnly).unwrap();
        let seconds = clock.elapsed().as_secs_f64();
        assert_eq!(f.stream_position().unwrap(), p.payload_offset() as u64);
        assert_eq!(p.text_payload_bytes(), 2_623_765_062);
        if trial >= 3 {
            samples.push(seconds);
        }
    }
    println!(
        "HERETIC_COLD {}",
        json!({"passed":true,"warmups":3,"samples_seconds":samples,"bytes_read_each":8+HEADER_BYTES,"config_read_once_bytes":config.len(),"tensor_payload_read_bytes":0,"numerical_execution":false,"timing":"warm file-header read+SHA+parse+plan; excludes file open/config read; no cold-storage or inference overhead claim"})
    );
}
