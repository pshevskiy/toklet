//! Explicit fixed4096 diagnostic; shared mechanics keep the two wire protocols distinct.
#[path = "support/qwen3_context.rs"]
mod context;
fn main() {
    context::main(context::Endpoint::Context4096);
}
