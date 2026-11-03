# Smol download, verification and offline Chat

The ordinary pinned Smol profile now has a real two-process Mac journey using
production Hub and Chat controllers. The online process starts with an absent
cache, probes the pinned public Hugging Face metadata endpoint, downloads through
the default transfer, verifies six files totaling 146,607,552 bytes and their
snapshot receipt, opens the audited model and completes a “Hello!” response.
The process exits and leaves the verified cache for replay.

A separate child uses the same cache under this process-only sandbox profile:

```scheme
(version 1) (allow default) (deny network*)
```

It successfully writes a local probe marker, then gets `EPERM` from a numeric
loopback socket connection and the same metadata endpoint that succeeded online.
No system network settings change. The process then uses the saved model through
HubStore and FriendlyChatStore, publishes the expected audited Smol identity and
completes another “Hello!” response with normal stop reason 2. Metal initialization
and inference work inside the network-denied child.

A private cache copy exercises two rejections: a missing required file reports an
inventory mismatch; a one-byte weight mutation at the same file size reports a SHA-256 failure.
Neither triggers the open callback. The original accepted owner, session, saved
prefix and conversation remain unchanged, the original cache re-verifies, and
another send succeeds. The original cache and receipts are preserved.

The first strict compile caught a test-helper argument typo. The initial offline
probe correctly rejected raw POSIX `EPERM` under its URLError-only rule. The
corrected test accepts that exact error only with independent socket confinement
and successful local writing. These failures remain preserved. A small typed
Hub diagnostic exposes the existing transport code without NSError userInfo or
signed redirect URLs; production transient-retry classification is unchanged.
Existing Hub controls and strict Mac/iOS/simulator compilation pass.

[Evidence](../benchmarks/results/2026-11-03-smol-download-offline-v1.json).
The raw receipts retain child commands, sandbox policy, endpoint/probe errors,
file identities and the verified receipt digest. This is Mac controller/cache
qualification; physical iPhone, rendered UI, throughput and full quality remain
separate.
