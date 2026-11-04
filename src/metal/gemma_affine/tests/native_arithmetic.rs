//! Explicit, small probes on the production precise projection pipeline.
//! Subnormal observations characterize this pipeline; they do not define a
//! general flush emulator or qualify a coupled model graph.
use super::*;

#[test]
#[ignore = "explicit Apple GPU projection arithmetic environment probes"]
fn projection_width_subnormals_and_residual() {
    autoreleasepool(|_| {
        let context = Context::new(CONTEXT_ALLOWANCE_BYTES).unwrap();
        println!();
        report(
            "GEMMA_NATIVE_PROJECTION_META",
            json!({"device":context.device_name(),"simd_width":context.width,
                "math_mode":"Safe","functions":"Precise","language":"2.3",
                "cases":6,"coupled_graph_qualified":false}),
        );
        let cases = [
            ("subnormal-positive", 16777216., f32::from_bits(1), 0.),
            ("subnormal-negative", 16777216., -f32::from_bits(1), 0.),
            (
                "two-subnormal-products",
                0.5,
                f32::MIN_POSITIVE,
                f32::MIN_POSITIVE,
            ),
            ("single-subnormal-product", 0.5, f32::MIN_POSITIVE, 0.),
            (
                "normal-residual",
                1.0078125,
                f32::from_bits(0x3f80_0001),
                -1.,
            ),
            ("positive-zero", 1., 0., 0.),
        ];
        let layout = Layout::new(1, 64, 4, 64, MetadataDtype::BF16).unwrap();
        for (name, weight, first, second) in cases {
            let weight: f32 = weight;
            assert_eq!(weight.to_bits() & 65535, 0);
            let codes = [0; 32];
            let scales = [0; 2];
            let biases = ((weight.to_bits() >> 16) as u16).to_le_bytes();
            let sizes = Sizes::new(layout, 1).unwrap();
            let mut operator = context
                .affine(layout, &codes, &scales, &biases, 1, sizes.owned)
                .unwrap();
            let mut input = [0.; 64];
            input[0] = first;
            input[1] = second;
            let mut baseline = None;
            for repeat in 0..2 {
                let mut output = [42.];
                let result = operator.apply(&input, &mut output, 1);
                report(
                    "GEMMA_NATIVE_PROJECTION_CASE",
                    json!({"name":name,"repeat":repeat,"completed":result.is_ok(),
                        "error":result.as_ref().err().map(ToString::to_string),
                        "weight_bits":weight.to_bits(),"input_bits":bits(&input),
                        "output_bits":bits(&output),"device":device(&operator),
                        "owned_bytes":operator.owned_bytes()}),
                );
                result.unwrap();
                assert!(!operator.is_poisoned());
                assert!(output[0].is_finite());
                assert_eq!(operator.owned_bytes(), sizes.owned);
                if let Some(previous) = baseline {
                    assert_eq!(previous, output[0].to_bits());
                }
                baseline = Some(output[0].to_bits());
            }
        }
        report(
            "GEMMA_NATIVE_PROJECTION_COMPLETE",
            json!({"cases":6,"repeats":2,"coupled_graph_qualified":false}),
        );
    });
}
