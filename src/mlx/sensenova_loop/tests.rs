use super::*;

#[test]
fn missing_or_unpinned_metadata_is_rejected() {
    let file_bytes = SHARDS.map(|pin| pin.file_bytes);
    for (config, index) in [
        (&b""[..], &b""[..]),
        (&b"{}"[..], &b"{\"weight_map\":{}}"[..]),
    ] {
        assert!(matches!(
            inspect(config, index, [b"", b""], file_bytes),
            Err(Error::Invalid(message)) if message == "SenseNova pinned config or index differs"
        ));
    }
}

#[test]
#[ignore = "requires retained SenseNova release metadata, with no shard payloads"]
fn retained_release_metadata() {
    let metadata = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts/sensenova-loop-9b-2026-11-18/metadata");
    let read = |name| std::fs::read(metadata.join(name)).unwrap();
    let config = read("mlx-config.json");
    let index = read("mlx-index.json");
    let first = read("mlx-shard-1-header.json");
    let second = read("mlx-shard-2-header.json");
    let lengths = [5_362_366_858, 600_449_850];
    let plan = inspect(&config, &index, [&first, &second], lengths).unwrap();
    let text = plan.text_tensors();
    let affine = plan.affine_layouts();

    assert_eq!(text.len(), 1023);
    assert_eq!(affine.len(), 250);
    assert_eq!(
        text.values()
            .map(|t| t.file_range.end - t.file_range.start)
            .sum::<u64>(),
        5_050_626_048
    );
    assert!(text.keys().all(|name| name.starts_with("language_model.")));
    for (name, shard, range) in [
        (
            "language_model.model.embed_tokens",
            0,
            2_974_962_218..3_483_521_578,
        ),
        ("language_model.lm_head", 1, 33_358_650..541_918_010),
    ] {
        let layout = affine[name];
        assert_eq!(
            (layout.rows(), layout.columns(), layout.group_size()),
            (248_320, 4096, 64)
        );
        let tensor = &text[&format!("{name}.weight")];
        assert_eq!((tensor.shard, tensor.dtype), (shard, Dtype::U32));
        assert_eq!(tensor.shape, [248_320, 512]);
        assert_eq!(tensor.file_range, range);
    }
    for (name, shape, range) in [
        (
            "language_model.model.layers.4.mhc_attn.weight",
            &[8, 8192][..],
            2_297_778_762..2_298_040_906,
        ),
        (
            "language_model.model.layers.27.mhc_ffn.bias",
            &[8][..],
            202_602_218..202_602_250,
        ),
    ] {
        let tensor = &text[name];
        assert_eq!((tensor.shard, tensor.dtype), (0, Dtype::F32));
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.file_range, range);
    }

    let mut changed_config = config.clone();
    changed_config[0] ^= 1;
    let mut changed_index = index.clone();
    changed_index[0] ^= 1;
    assert!(inspect(&changed_config, &index, [&first, &second], lengths).is_err());
    assert!(inspect(&config, &changed_index, [&first, &second], lengths).is_err());
    for (headers, sizes) in [
        ([second.as_slice(), first.as_slice()], lengths),
        ([&first[..first.len() - 1], second.as_slice()], lengths),
        (
            [first.as_slice(), second.as_slice()],
            [lengths[0], lengths[1] - 1],
        ),
    ] {
        assert!(matches!(
            inspect(&config, &index, headers, sizes),
            Err(Error::Invalid(message)) if message == "SenseNova pinned shard header differs"
        ));
    }
}
