use super::*;
use serde::Deserialize;
use std::{io::Read, os::unix::fs::OpenOptionsExt};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Pin {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Control {
    pub format: String,
    pub artifact_directory: String,
    pub output_directory: String,
    pub binary: Pin,
    pub owner_manifest: Pin,
    pub protocol_sha256: String,
    pub inventory_sha256: String,
    pub gates_sha256: String,
    pub source_manifest_sha256: String,
    pub source_arrays_sha256: String,
    pub authorization: Value,
}
fn file(path: &Path) -> std::io::Result<File> {
    let f = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    if !f.metadata()?.is_file() {
        return Err(std::io::Error::other("regular file required"));
    }
    Ok(f)
}
fn small(path: &Path, cap: u64) -> std::io::Result<Vec<u8>> {
    let f = file(path)?;
    if f.metadata()?.len() > cap {
        return Err(std::io::Error::other("small file cap"));
    }
    let mut bytes = Vec::new();
    f.take(cap + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > cap {
        return Err(std::io::Error::other("small consumed file cap"));
    }
    Ok(bytes)
}
fn verify(pin: &Pin, cap: u64) -> std::io::Result<()> {
    let mut f = file(Path::new(&pin.path))?;
    if pin.bytes > cap || f.metadata()?.len() != pin.bytes {
        return Err(std::io::Error::other("pinned file extent"));
    }
    let mut block = [0; 65536];
    let mut hash = Sha256::new();
    let mut count = 0u64;
    loop {
        let n = f.read(&mut block)?;
        if n == 0 {
            break;
        }
        count += n as u64;
        if count > cap {
            return Err(std::io::Error::other("pinned consumed extent"));
        }
        hash.update(&block[..n]);
    }
    if count != pin.bytes || format!("{:x}", hash.finalize()) != pin.sha256 {
        return Err(std::io::Error::other("pinned consumed identity"));
    }
    Ok(())
}
pub(super) fn contract(c: &Control) -> Result<()> {
    if c.format != "toklet-lfm2-real-capture-control-v1"
        || c.protocol_sha256 != PROTOCOL_SHA
        || c.inventory_sha256 != INVENTORY_SHA
        || c.gates_sha256 != "7b1879ee54fd5a421d0b5f17f2d9192d291742bdc078ef2f5e90885eb293fbdf"
        || c.source_manifest_sha256
            != "2654f475aedb37024b66f6c9a9988369d8ea8138a34e078505adf6fe16635133"
        || c.source_arrays_sha256
            != "9db9e7cd82a1a15ef60f75b2cd91b04f20d30553bb64cb55be72d2c201d0837a"
    {
        return Err(Error::Options("capture fixed protocol/source identity"));
    }
    Ok(())
}
pub(super) fn control() -> Result<(Control, String)> {
    let path = std::env::var("TOKLET_LFM2_CAPTURE_CONTROL")
        .map_err(|_| Error::Options("explicit capture control required"))?;
    let expected = std::env::var("TOKLET_LFM2_CAPTURE_CONTROL_SHA256")
        .map_err(|_| Error::Options("explicit capture control digest required"))?;
    let bytes = small(Path::new(&path), 65536).map_err(io)?;
    if sha(&bytes) != expected {
        return Err(Error::Options("capture consumed control identity"));
    }
    let c: Control = serde_json::from_slice(&bytes).map_err(|e| Error::Native(e.to_string()))?;
    contract(&c)?;
    let current = std::env::current_exe()
        .map_err(io)?
        .canonicalize()
        .map_err(io)?;
    if current != Path::new(&c.binary.path).canonicalize().map_err(io)? {
        return Err(Error::Options("capture executed binary path"));
    }
    verify(&c.binary, 256 * 1024 * 1024).map_err(io)?;
    verify(&c.owner_manifest, 16 * 1024 * 1024).map_err(io)?;
    Ok((c, expected))
}
pub(super) fn load(c: &Control, capture: &mut Capture) -> Result<Model> {
    let root = Path::new(&c.artifact_directory);
    let config = small(&root.join("config.json"), 1024 * 1024).map_err(io)?;
    let index = small(&root.join("model.safetensors.index.json"), 1024 * 1024).map_err(io)?;
    let original = file(&root.join("model.safetensors")).map_err(io)?;
    let identity = ContentIdentity {
        config_sha256: digest("95e6906d77ac647e8bf93c2d2bbd2f37eb9bf3fac8732a563653236ad81072af"),
        index_sha256: digest("9921c33945951c541dfb7de7a11c6ff3b0bb8fb58009be615f851a7de89bcc38"),
        header_sha256: digest("7c5e241d377d693fac694426ed10fb79df470864a9ec31d0ca82f3044171506d"),
        container_sha256: digest(
            "558353ad098c1230a8e2ea508cd34ef311a1b97a2bc4f47498f4df8229dc5c0f",
        ),
        container_bytes: 199527820,
    };
    let store = payload::load(
        &config,
        &index,
        original,
        LoadOptions {
            identity,
            memory_limit_bytes: LIMIT,
            external_resident_bytes: EXTERNAL,
        },
    )
    .map_err(|e| Error::Native(e.to_string()))?;
    for capacity in [0, 513] {
        let error = store
            .prepare_direct_weights(capacity, EXTERNAL, LIMIT)
            .err()
            .map(|e| format!("{e:?}"));
        capture.event(
            "capacity_reject",
            json!({"capacity":capacity,"error":error,"scope":"cold_data_preparation"}),
            Class::Other,
        )?;
        if error.is_none() {
            return Err(Error::Options("unsupported capacity accepted"));
        }
    }
    let input = store
        .prepare_direct_weights(512, EXTERNAL, LIMIT)
        .map_err(|e| Error::Native(e.to_string()))?;
    let mut typed_lengths = vec![input.weights.final_gamma.len()];
    let mut typed_capacities = vec![input.weights.final_gamma.capacity()];
    for l in &input.weights.layers {
        typed_lengths.extend([l.operator_gamma.len(), l.ffn_gamma.len()]);
        typed_capacities.extend([l.operator_gamma.capacity(), l.ffn_gamma.capacity()]);
    }
    for a in &input.weights.attention {
        typed_lengths.extend([a.q_gamma.len(), a.k_gamma.len()]);
        typed_capacities.extend([a.q_gamma.capacity(), a.k_gamma.capacity()]);
    }
    for v in &input.weights.convolution {
        typed_lengths.push(v.coefficients.len());
        typed_capacities.push(v.coefficients.capacity());
    }
    let direct_words: usize = typed_lengths.iter().sum();
    let descriptor_sizes = [
        size_of::<crate::model::lfm2::LayerWeights<'_>>(),
        size_of::<crate::model::lfm2::AttentionWeights<'_>>(),
        size_of::<crate::model::lfm2::ConvWeights<'_>>(),
    ];
    let prep = json!({"source_resident_bytes":input.source_resident_bytes,
        "descriptor_capacities":[input.weights.layers.capacity(),input.weights.attention.capacity(),input.weights.convolution.capacity()],
        "descriptor_element_bytes":descriptor_sizes,"typed_BF16_capacities":typed_capacities,"typed_BF16_lengths":typed_lengths,"direct_words":direct_words});
    let r = store.report();
    let load = json!({"container_bytes":r.container_bytes,"header_bytes":r.header_bytes,"header_capacity_bytes":r.header_capacity_bytes,
        "payload_bytes":r.payload_bytes,"payload_capacity_bytes":r.payload_capacity_bytes,
        "metadata_reservation_bytes":r.metadata_reservation_bytes,"external_resident_bytes":r.external_resident_bytes,
        "admitted_load_bytes":r.admitted_load_bytes});
    let ModelInput {
        config,
        weights,
        source_resident_bytes,
    } = input;
    let model = Model::from_raw_weights(config, weights, source_resident_bytes, LIMIT)?;
    drop(store); // No PayloadStore/typed preparation owner survives upload.
    capture.event(
        "source_drop",
        json!({"load":load,"prepared":prep,"model":memory(model.memory()),
        "source_payload_retained":false,"whole_container_bytes":199527820}),
        Class::Boundary,
    )?;
    require(
        direct_words == 65280,
        "observed direct preparation word inventory",
    )?;
    Ok(model)
}
pub(super) fn model(
    c: &mut Capture,
    m: &Model,
    phase: &'static str,
) -> Result<BTreeMap<usize, (usize, String)>> {
    let mut all = BTreeMap::new();
    let mut index = 0usize;
    let mut matrix = |name: String, t: &Tensor| -> Result<()> {
        let address = t.buffer.raw.contents().as_ptr() as usize;
        let bytes = t.buffer.raw.length();
        let hash = t.buffer.with_slice(output::hash16);
        c.event(
            phase,
            json!({"index":index,"name":name,"role":"weight","address":address,"bytes":bytes,
            "device":t.buffer.raw.device().registryID(),"sha256":hash}),
            Class::Other,
        )?;
        if all.insert(address, (bytes, hash)).is_some() {
            return Err(Error::Options("immutable model owner alias"));
        }
        index += 1;
        let TensorStorage::Mlx(p) = &t.storage else {
            return Err(Error::Options("original packed model required"));
        };
        p.test_visit_metadata(|role, address, device, words| {
            let hash = output::hash32(words);
            let bytes = words.len() * 4;
            c.event(
                phase,
                json!({"index":index,"name":name,"role":(["scales","biases"][role]),"address":address,
                "bytes":bytes,"device":device,"sha256":hash}),
                Class::Other,
            )?;
            if all.insert(address, (bytes, hash)).is_some() {
                return Err(Error::Options("immutable metadata alias"));
            }
            index += 1;
            Ok(())
        })
    };
    matrix("model.embed_tokens".into(), &m.embedding)?;
    for (l, x) in m.layers.iter().enumerate() {
        for (name, t) in [("w1", &x.gate), ("w3", &x.up), ("w2", &x.down)] {
            matrix(format!("model.layers.{l}.feed_forward.{name}"), t)?;
        }
    }
    for (l, x) in m.layers.iter().enumerate() {
        if let Kind::Attention(owner) = x.kind {
            let a = &m.attention[owner];
            for (name, t) in [
                ("q_proj", &a.q),
                ("k_proj", &a.k),
                ("v_proj", &a.v),
                ("out_proj", &a.o),
            ] {
                matrix(format!("model.layers.{l}.self_attn.{name}"), t)?;
            }
        }
    }
    for (l, x) in m.layers.iter().enumerate() {
        if let Kind::Convolution(owner) = x.kind {
            let a = &m.convolution[owner];
            for (name, t) in [("in_proj", &a.bcx), ("out_proj", &a.output)] {
                matrix(format!("model.layers.{l}.conv.{name}"), t)?;
            }
        }
    }
    let mut direct = |name: String, role: &'static str, b: &Buffer<u32>| -> Result<()> {
        let address = b.raw.contents().as_ptr() as usize;
        let bytes = b.raw.length();
        let hash = b.with_slice(output::hash32);
        let tail = b.with_slice(|v| v[v.len() - GUARD..].to_vec());
        c.event(
            phase,
            json!({"index":index,"name":name,"role":role,"address":address,"bytes":bytes,
            "device":b.raw.device().registryID(),"sha256":hash,"tail":tail}),
            Class::Other,
        )?;
        if !tail.iter().all(|&x| x == WORD_PAD) || all.insert(address, (bytes, hash)).is_some() {
            return Err(Error::Options("immutable direct owner/guard"));
        }
        index += 1;
        Ok(())
    };
    direct(
        "model.embedding_norm.weight".into(),
        "direct_gain",
        &m.final_norm,
    )?;
    for (l, x) in m.layers.iter().enumerate() {
        direct(
            format!("model.layers.{l}.operator_norm.weight"),
            "direct_gain",
            &x.operator_norm,
        )?;
        direct(
            format!("model.layers.{l}.ffn_norm.weight"),
            "direct_gain",
            &x.ffn_norm,
        )?;
    }
    for (l, x) in m.layers.iter().enumerate() {
        if let Kind::Attention(owner) = x.kind {
            let a = &m.attention[owner];
            direct(
                format!("model.layers.{l}.self_attn.q_layernorm.weight"),
                "direct_gain",
                &a.q_norm,
            )?;
            direct(
                format!("model.layers.{l}.self_attn.k_layernorm.weight"),
                "direct_gain",
                &a.k_norm,
            )?;
        }
    }
    for (l, x) in m.layers.iter().enumerate() {
        if let Kind::Convolution(owner) = x.kind {
            direct(
                format!("model.layers.{l}.conv.conv.weight"),
                "convolution_tap",
                &m.convolution[owner].coefficients,
            )?;
        }
    }
    let b = &m.zero_bias;
    let address = b.raw.contents().as_ptr() as usize;
    let hash = b.with_slice(output::hash16);
    let tail = b.with_slice(|v| v[v.len() - GUARD..].to_vec());
    c.event(phase,json!({"index":index,"name":"zero_bias","role":"zero_bias","address":address,"bytes":b.raw.length(),
        "device":b.raw.device().registryID(),"sha256":hash,"tail":tail}),Class::Other)?;
    let active_zero = b.with_slice(|v| v[..v.len() - GUARD].iter().all(|&x| x == 0));
    if !active_zero
        || !tail.iter().all(|&x| x == 0)
        || all.insert(address, (b.raw.length(), hash)).is_some()
        || index != 334
    {
        return Err(Error::Options("immutable original model inventory"));
    }
    Ok(all)
}
