//! Explicit public held-out-prefix rows for the independent numerical oracle.
use super::*;

#[test]
#[ignore = "explicit Heretic512 raw full-row capture for independent FP64/stagedFP32 comparison"]
fn independent_context_rows() {
    use std::io::Write;
    let raw = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/benchmarks/gemma-e2b-context-v1/tokens.u32le"
    ));
    assert_eq!(raw.len(), 2048);
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "ade0775a55a1248340329388c5ba6fced9c12b42ef9974c6d10953fc5cca7f15"
    );
    let tokens: Vec<u32> = raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let positions = [0, 15, 30, 31, 63, 255, 511];
    let out = PathBuf::from(std::env::var_os("TOKLET_HERETIC_FIDELITY_OUTPUT").unwrap());
    std::fs::create_dir(&out).unwrap();
    let context = HereticContext::new(CONTEXT_BYTES).unwrap();
    let source = load(&context);
    let norms = Norms::new(&source, LIMIT - CALLER).unwrap();
    let model = norms.model(512).unwrap();
    let mut session = norms
        .new_session(&model, &context, 32, LIMIT - CALLER)
        .unwrap();
    let allocated = session.allocated_bytes().unwrap();
    assert!(model.model_bytes() + allocated + CALLER <= LIMIT);
    let start = std::time::Instant::now();
    let mut captured = Vec::new();
    for (chunk, input) in tokens.chunks_exact(32).enumerate() {
        let before = chunk * 32;
        let rows = session.forward(input).unwrap();
        assert_eq!(rows.len(), 32 * 262144);
        for &position in &positions {
            if (before..before + 32).contains(&position) {
                let row = &rows[(position - before) * 262144..(position - before + 1) * 262144];
                let name = format!("position-{position}.f32le");
                let file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(out.join(&name))
                    .unwrap();
                let mut file = std::io::BufWriter::new(file);
                for value in row {
                    file.write_all(&value.to_le_bytes()).unwrap();
                }
                file.flush().unwrap();
                captured.push(serde_json::json!({"position":position,"file":name,"bytes":262144*4,"sha256":digest(row),"greedy_id":greedy(row)}));
            }
        }
        assert_eq!(session.tokens(), &tokens[..before + 32]);
        assert_eq!(session.allocated_bytes().unwrap(), allocated);
    }
    assert_eq!(captured.len(), positions.len());
    assert!(session.forward(&[2]).is_err());
    assert_eq!(session.tokens(), tokens);
    assert_eq!(session.allocated_bytes().unwrap(), allocated);
    println!(
        "HERETIC_FIDELITY {}",
        serde_json::json!({"format":"toklet-heretic-context-fidelity-v1","passed":true,"tokens":tokens,"positions":positions,"rows":captured,
            "source":source.identity(),"load":source.report(),"model_bytes":model.model_bytes(),"session_allocated_bytes":allocated,"caller_reserve_bytes":CALLER,
            "collection_seconds":start.elapsed().as_secs_f64(),"memory":footprint(),"scope":"Raw All512 context with exact boundary rejection; numerical verdict belongs to independent reference. No quality/adoption or expanded context claim."})
    );
}
