# Swift accepted identity and coherent Apple bridge

Toklet's worker now consumes the accepted model/chat identity report through a
strict Swift decoder. Explicit managed Qwen3 selection uses the audited model
and codec factories; Expert, external-folder and other-profile defaults keep
ordinary loading. [Verification](../benchmarks/results/2026-10-30-swift-accepted-identity-v1.json)
includes the successful native getter/controller path on Mac. Preferences are
not stored by this change.

The decoder reuses `GenerationJSON` without widening its limits. It rejects
unknown keys, malformed numbers/booleans, duplicate or changed assets/roles,
unsupported policy versions and mismatched model/backend/loaded-limit facts.
It independently reconstructs the complete canonical manifest. Model instance
and loaded limit correlate the report; neither belongs to the stable namespace.
Requested managed routing is intent, not content authentication.

Managed intent, folder lease, worker/request and conversation are carried across
pending unload. Stale completion cannot dispatch a replacement load for another
request. Identity is published only after model/session/chat/stop/state setup
succeeds inside the cleanup transaction. Accepted identity belongs to the model
and codec, so session replay/replacement preserves it; retirement clears it.

The selected local XCFramework now uses
`artifacts/new-goal-b5d5d-native35-build-v1/Toklet.xcframework`.
All three Metal slices export the same 35 C APIs and pass header/link/platform
checks: Mac 15, iOS 18 and arm64 simulator 18. A fresh default 35 archive separately
passes export/C ABI checks for standalone CPU consumers. Native Rust/shader
inputs match the committed report implementation, reusing its 501 default and
609 Metal tests. The previous 32-export package is retained.

Swift verification has nineteen successful logical cold jobs: one authenticated
strict typed compile reused from the first attempt, then eighteen fresh jobs.
They run 51 wire and 37 routing controls, current transaction/configuration and
CPU preview regressions, and three complete artifact-local app build/link checks
(project 34/Mac 35 sources). Two Mac jobs then pass: the existing 21-barrier tiny
Metal preview/generation regression and one cached audited-model controller.

The real-model controller exercises the getter, with this actual lifecycle:

| Stage | Model | Session | Capacity/chunk |
| --- | ---: | ---: | --- |
| Loaded | 1 | 2 | 512/32 |
| Empty replay | 1 | 3 | 512/32 |
| Empty replacement | 1 | 4 | 256/16 |

The accepted namespace and loaded upper 512 stay exact. Replacement request
`C84D968D-C439-45A0-BE97-63B03AFE5732` passes actual coexistence accounting of
1,446,751,542 bytes and incremental requirement 125,957,028 against configured
headroom 5,589,007,112. Headroom is the existing Mac process policy, not free RAM
or a reservation. Workspace/file hashes and writes 2→2 stay exact from the
explicit baseline through replacement. The subsequent stale-conversation phase
intentionally writes 3, dispatches no replacement model load and clears identity;
close also leaves identity absent. The real controller generates zero tokens.
One model/getter is established by the reviewed source flow and records, not an
independent native-call counter or full binary-state reconstruction.

The first full worker/store compile failed on a duplicate local `code` binding.
Its receipt/logs remain failed and preserved. V2 changes only the chat status
name; all eight other files and all assertions stay exact. The independently
compiled typed executable is reused because its entire closure is unchanged.
Astra audits the final results and guarded atomic link selection; no successful
native build, model run or unaffected typed compile is repeated.

This qualifies source/controller/build integration and the successful Mac getter.
It does not qualify rendered UI, iPhone runtime, real-model generation after
resizing, broader accuracy/windows, speed or whole-app/device fit. No explicit
signing, account, provisioning or device-installation stage occurs. Mac/simulator
linked products can contain implicit `LC_CODE_SIGNATURE`. Next: bounded per-model
preferences, freshly admitted restoration and reply/edit/late-commit handling.
