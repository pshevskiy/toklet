import Foundation
import CryptoKit

private final class SmolRestoreGate: @unchecked Sendable {
    private let lock = NSLock()
    private let reached = DispatchSemaphore(value:0), resume = DispatchSemaphore(value:0)
    private var loads = 0, prepared = 0, committed = 0
    func command(_ command: NativeCommand) {
        guard case .load(let settings) = command else { return }
        lock.lock(); loads += 1; let count = loads; lock.unlock()
        precondition(count == 1 && settings.loadIdentity == .managedSmol,"Unexpected model load")
    }
    func before() {
        lock.lock(); prepared += 1; let count = prepared; lock.unlock()
        precondition(count == 1,"Unexpected restore candidate")
        reached.signal()
        guard resume.wait(timeout:.now()+30) == .success else {
            FileHandle.standardError.write(Data("Smol restore barrier deadline\n".utf8))
            preconditionFailure("Smol restore barrier deadline")
        }
    }
    func after() { lock.lock(); committed += 1; lock.unlock() }
    func wait(_ worker: NativeWorker) throws {
        guard reached.wait(timeout:.now()+30) == .success else {
            FileHandle.standardError.write(Data((String(reflecting:worker.snapshot()) + "\n").utf8.prefix(131072)))
            throw NativeContextFailure("Smol restore acceptance deadline")
        }
    }
    func release() { resume.signal() }
    var counts: (Int,Int,Int) { lock.lock(); defer { lock.unlock() }; return (loads,prepared,committed) }
}

@main
@MainActor
struct FriendlySmolPreferenceLoadedMain {
    static func require(_ value: Bool, _ message: String) throws { if !value { throw NativeContextFailure(message) } }
    static func digest(_ bytes: Data) -> String { SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined() }
    static func settle(_ harness: HarnessStore, _ chat: FriendlyChatStore, _ worker: NativeWorker) throws {
        for _ in 0..<4 {
            guard worker.waitUntilIdle(seconds:180) != nil else { throw NativeContextFailure("Smol worker idle deadline") }
            harness.poll(); chat.observe(harness.state)
            if !worker.snapshot().busy && !chat.restoringPreference { return }
        }
        throw NativeContextFailure("Smol restore did not settle")
    }
    static func frame(_ event: String, _ worker: NativeWorker, _ chat: FriendlyChatStore,
                      _ record: URL, _ qwenRecord: URL, _ workspace: URL, _ writes: Int) throws {
        let state = worker.snapshot(), encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        var row: [String:Any] = ["event":event,"snapshot":String(reflecting:state),"ready":chat.ready,
            "restoring":chat.restoringPreference,"preference_message":chat.preferenceMessage,"preference_writes":writes,
            "smol_record":try JSONSerialization.jsonObject(with:Data(contentsOf:record)),
            "smol_sha256":digest(try Data(contentsOf:record)),"qwen_sha256":digest(try Data(contentsOf:qwenRecord)),
            "workspace_sha256":digest(try encoder.encode(chat.workspace)),"file_sha256":digest(try Data(contentsOf:workspace)),
            "reply":chat.workspace.outputLimit,"draft":chat.workspace.current.draft,"messages":chat.workspace.current.messages.count,
            "history":state.history,"committed":state.evaluated,"pending":state.pending,"generated":state.ids.count]
        if let identity = state.acceptedIdentity {
            row["model"] = identity.modelInstance; row["manifest"] = identity.namespace.manifestSHA256
            row["load_policy"] = identity.namespace.load; row["text_policy"] = identity.namespace.text
            row["source_ranges"] = identity.sourceRanges; row["source_reads"] = identity.sourceReads
            if let asset = identity.assets.first(where:{ $0.name == "model.safetensors" }) { row["model_roles"] = asset.roles }
        }
        if let c = state.sessionConfiguration {
            row["configuration"] = ["model":c.modelInstance,"session":c.sessionInstance,"capacity":c.capacity,
                "chunk":c.chunk,"whole_budget":c.wholeSessionBudget,"accepted_request":c.acceptedRequestID?.uuidString as Any? ?? NSNull()]
        }
        if let h = state.transition?.headroom { row["headroom"] = ["required":h.requiredBytes,"available":h.availableBytes,"fits":h.fits,"instrument":h.instrument] }
        let raw = try JSONSerialization.data(withJSONObject:row,options:[.sortedKeys])
        try require(raw.count <= 131072,"Smol raw frame exceeds bound")
        FileHandle.standardOutput.write(raw + Data([10]))
    }
    static func main() throws {
        guard CommandLine.arguments.count == 3 else { throw NativeContextFailure("Expected cached Smol model and fresh output directory") }
        let model = URL(fileURLWithPath:CommandLine.arguments[1]), output = URL(fileURLWithPath:CommandLine.arguments[2])
        try FileManager.default.createDirectory(at:output,withIntermediateDirectories:false)
        let workspaceFile = output.appendingPathComponent("chat.json"), directory = output.appendingPathComponent("preferences-v1")
        let namespace = NativeAcceptedIdentity.Profile.smol.namespace, qwenNamespace = NativeAcceptedIdentity.Profile.qwen3.namespace
        let seed = FriendlyModelPreference(namespace:namespace,capacity:256,chunk:16,reply:64,wholeBudget:HubProfile.smol.numericalBudget)
        let qwenSeed = FriendlyModelPreference(namespace:qwenNamespace,capacity:128,chunk:8,reply:32,wholeBudget:HubProfile.qwen3.numericalBudget)
        let initialStore = FriendlyPreferenceStore(directory:directory)
        // Untrusted records only: restoration still waits for actual native identity.
        try initialStore.save(seed); try initialStore.save(qwenSeed)
        let record = try initialStore.url(namespace), qwenRecord = try initialStore.url(qwenNamespace)
        let seeded = try Data(contentsOf:record), qwenBytes = try Data(contentsOf:qwenRecord)
        var workspace = ChatWorkspace(); workspace.conversations[0].draft = "Keep this Smol unsent draft."
        let conversation = workspace.selected, draft = workspace.current.draft
        try JSONEncoder().encode(workspace).write(to:workspaceFile,options:.atomic)
        let gate = SmolRestoreGate(), harness = HarnessStore(enabled:true)
        guard let initial = harness.worker else { throw NativeContextFailure("Missing initial worker") }
        initial.close(); try require(initial.waitUntilClosed(seconds:30),"Initial worker did not close")
        let worker = try NativeWorker(beforeCommandDequeue:{ gate.command($0) },
            beforeTransitionAcceptance:{ _,_ in gate.before() },afterTransitionAcceptance:{ _ in gate.after() })
        harness.worker = worker; harness.state = worker.snapshot()
        var writes = 0
        let preferences = FriendlyPreferenceStore(directory:directory,persist:{ bytes,url in writes += 1; try bytes.write(to:url,options:.atomic) })
        let chat = FriendlyChatStore(harness:harness,fileURL:workspaceFile,preferences:preferences)
        var failure: Error?
        do {
            try require(chat.acceptedIdentity == nil && !worker.snapshot().hasModel && writes == 0,"Startup promoted stored path/record to identity")
            chat.openManaged(model,profile:.smol,ifSelected:conversation)
            guard worker.waitUntilIdle(seconds:180) != nil else { throw NativeContextFailure("Audited Smol load deadline") }
            harness.poll(); chat.observe(harness.state) // Starts exactly one fresh D1/D2 restore.
            try gate.wait(worker)
            try frame("restore-before-acceptance",worker,chat,record,qwenRecord,workspaceFile,writes)
            let before = worker.snapshot()
            try require(before.acceptedIdentity?.profile == .smol && chat.acceptedIdentity == before.acceptedIdentity
                && before.sessionConfiguration?.capacity == 512 && before.sessionConfiguration?.chunk == 32
                && chat.restoringPreference && !chat.ready && !chat.canSend && writes == 0
                && (try Data(contentsOf:record)) == seeded,"Readiness or saved settings escaped preacceptance gate")
            gate.release(); try settle(harness,chat,worker)
            try frame("restored",worker,chat,record,qwenRecord,workspaceFile,writes)
            let restored = worker.snapshot()
            guard let identity = restored.acceptedIdentity, let configuration = restored.sessionConfiguration else { throw NativeContextFailure("Missing restored identity/configuration") }
            try require(chat.ready && identity.profile == .smol && identity.loadedLimit == 512 && identity.sourceRanges == 694
                && restored.transition?.committed == true && restored.transition?.headroom?.fits == true
                && configuration.capacity == 256 && configuration.chunk == 16 && configuration.acceptedRequestID != nil
                && configuration.sessionInstance != before.sessionConfiguration?.sessionInstance
                && chat.workspace.outputLimit == 64 && writes == 0 && (try Data(contentsOf:record)) == seeded,"Actual Smol restore failed or rewrote record")
            func preserved() throws {
                let state = worker.snapshot()
                try require(state.acceptedIdentity == identity && state.modelContext?.modelInstance == identity.modelInstance
                    && state.modelContext?.loadedLimit == 512 && state.history == 0 && state.evaluated == 0 && state.pending == 0 && state.ids.isEmpty
                    && chat.workspace.selected == conversation && chat.workspace.current.draft == draft && chat.workspace.current.messages.isEmpty
                    && (try Data(contentsOf:qwenRecord)) == qwenBytes,"Owner, Qwen record, conversation or empty prefix changed")
            }
            try preserved()
            chat.setReplyLimit(128)
            try frame("reply-saved",worker,chat,record,qwenRecord,workspaceFile,writes)
            try require(writes == 1 && (try preferences.load(namespace))?.reply == 128
                && (try preferences.load(namespace))?.capacity == 256 && chat.workspace.outputLimit == 128,"Explicit Smol reply edit was not saved")
            try preserved()
            let acceptedBytes = try Data(contentsOf:record), savedWorkspace = try Data(contentsOf:workspaceFile)
            chat.setReplyLimit(512)
            try require(chat.workspace.outputLimit == 128 && writes == 1 && (try Data(contentsOf:record)) == acceptedBytes
                && (try Data(contentsOf:workspaceFile)) == savedWorkspace,"Rejected reply changed accepted record/workspace")
            chat.showPlanning(); chat.previewPlanning(); try settle(harness,chat,worker)
            try frame("metadata-only-preview",worker,chat,record,qwenRecord,workspaceFile,writes)
            try require(chat.planning.target?.profile == .smolAudited && chat.planning.result?.outcome == .metadataOnly
                && chat.planning.result?.limitation == .auditedProfileHasNoCapacityAPI && chat.planning.result?.capacity == nil
                && chat.planningApplyUnavailableReason != nil,"Smol was given unimplemented numerical Apply authority")
            chat.hidePlanning(); try preserved()
            worker.interrupt(unload:false); try settle(harness,chat,worker)
            try frame("suspended",worker,chat,record,qwenRecord,workspaceFile,writes)
            try require(!worker.snapshot().hasSession && worker.snapshot().acceptedIdentity == identity
                && worker.snapshot().sessionConfiguration == nil,"Suspend lost model evidence or retained session configuration")
            try preserved()
            try require(worker.submit(.replay),"Smol empty replay rejected")
            try settle(harness,chat,worker); try frame("replayed",worker,chat,record,qwenRecord,workspaceFile,writes)
            let replayed = worker.snapshot()
            try require(replayed.status == 0 && replayed.sessionConfiguration?.sessionInstance != configuration.sessionInstance
                && replayed.sessionConfiguration?.capacity == 256 && replayed.sessionConfiguration?.chunk == 16
                && replayed.sessionConfiguration?.acceptedRequestID == nil && writes == 1
                && (try Data(contentsOf:record)) == acceptedBytes && (try Data(contentsOf:workspaceFile)) == savedWorkspace,"Replay restored twice or changed records/workspace")
            try preserved()
        } catch { failure = error }
        gate.release(); worker.close()
        try require(worker.waitUntilClosed(seconds:60),"Smol owning worker did not close")
        harness.poll(); chat.observe(harness.state)
        try frame("closed",worker,chat,record,qwenRecord,workspaceFile,writes)
        if let failure { throw failure }
        let counts = gate.counts
        try require(counts.0 == 1 && counts.1 == 1 && counts.2 == 1 && writes == 1
            && worker.snapshot().acceptedIdentity == nil && chat.acceptedIdentity == nil,"Unexpected load/restore/save count or retained identity")
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject:["passed":true,"closed":true,"model_loads":counts.0,
            "prepared_candidates":counts.1,"accepted_replacements":counts.2,"preference_writes":writes,"real_generation_tokens":0,
            "qwen_record_unchanged":true,"smol_preview_metadata_only":true,"rendered_ui_checked":false,"phone_checked":false],options:[.sortedKeys]) + Data([10]))
    }
}
