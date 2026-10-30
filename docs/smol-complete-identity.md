# Complete Smol model and codec identity

The native identity path now supports complete historical evidence for the
pinned SmolLM2-135M eight-bit model and codec pair. One additive
`toklet_chat_create_complete_audited` factory selects Smol or Qwen3 from the
actual assembled model's attached source receipt. The existing identity getter
reuses exact-owner, thread, source-receipt and policy checks. A directory name
or caller-selected profile cannot supply that authority.

Smol retains its existing trusted 694-range audit and model factory. The complete
codec verifies the ancillary special-token map, releases that bounded buffer,
then uses the existing audited tokenizer constructor. Successful completion adds
the six-file manifest and ancillary evidence to its cold receipt. The legacy
three-asset codec API is unchanged and still works; its incomplete receipt is
rejected by the identity getter. Tokenizer bytes are hashed from the buffer
passed to the parser. Tokenizer configuration is verified fixed policy, and the
special map is verified ancillary data; neither enables arbitrary chat templates.

The shared bounded asset reader is extracted unchanged from Qwen3. Its report
schema, namespace and values stay unchanged. Smol uses
`smol_witness_audited_v1` and `smol-chatml-v1` with the existing backend/planning
versions. The model container's additional `reference_container_digest_anchor`
role distinguishes its trusted range-witness anchor from a recomputed whole-file
SHA. No extra native whole-container prepass or second model copy is introduced.
IDs, paths and loaded capacity remain outside the stable namespace.

The fixed receipt and 655+1-byte ancillary allocation fit the existing separate
64 MiB Smol text policy; the ancillary buffer is released before tokenizer
construction. This is policy accounting, not measured heap or whole-process fit.
Qwen3 keeps its 256 MiB text policy. Actual model geometry, modes and admission
remain the existing native factory's responsibility. The getter performs no
filesystem, session or GPU work and does not grant a larger context or quality
qualification.

The [verification record](../benchmarks/results/2026-10-30-smol-complete-identity-v1.json)
has twelve passing native/C jobs: formatting, strict default/Metal Clippy,
510 default and 618 Metal-feature tests (21/189 intentionally ignored),
release library/test builds, exactly 36 exports and the C ABI probe.

One separately reviewed cached Smol load passes all seven identity/ownership
frames. Its actual source receipt has 694 physical ranges and 694 verified
reads. Missing ancillary data rejects without adding an owner. The legacy codec
still constructs but cannot return complete identity; after its release, the
complete codec and getter succeed. The exact model owner remains alive after
external handle release, then expires when the codec releases it. Captured JSON
survives both owners and is released separately. No session or text is generated.
Astra authenticates the seven raw frames, canonical manifest, current source,
retained products and all 13 current plus five failed-attempt process closures.
Lifetime/stop/snapshot behavior also relies on completed compiled assertions;
this is not independent reconstruction of every native memory state.

The first default-test run passed 323 library tests
and failed two new missing-file expectations. The extracted reader intentionally
maps file errors to `InvalidInput`, preserving the path. A reviewed test-only
correction checks that kind and ancillary path, and strengthens malformed-size
checks to exact extent/digest diagnostics. Other 14 paths, production behavior
and numerical gates are unchanged; the failed receipt remains preserved.

This native task does not activate Smol preferences in Swift. The selected Apple
package remains the qualified 35-export bundle; adopting the new 36-export API
requires a coherent later package and consumer check. Smol's audited app preview
currently reports metadata only; numerical-capacity preview and Apply UI remain
separate work. Rendered UI, iPhone runtime, generation, quality and performance
are outside this identity task.
