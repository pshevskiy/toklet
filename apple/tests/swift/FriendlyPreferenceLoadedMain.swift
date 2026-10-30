import Foundation
import CryptoKit

private final class PreferenceGate: @unchecked Sendable {
    private let lock = NSLock()
    private let reached = DispatchSemaphore(value:0), resume = DispatchSemaphore(value:0)
    private var point: String?
    private var retirement: NativeTransitionRetirementFault?
    private var loads = 0
    private var prepared = 0, committed = 0
    func arm(_ point: String, quarantine: Bool = false) {
        lock.lock(); self.point = point; retirement = quarantine ? .busy : nil; lock.unlock()
    }
    func hit(_ point: String) {
        lock.lock(); let active = self.point == point
        if active { self.point = nil }; lock.unlock()
        if active {
            reached.signal()
            guard resume.wait(timeout:.now()+30) == .success else {
                FileHandle.standardError.write(Data("Preference callback deadline\n".utf8)); preconditionFailure("Preference callback deadline")
            }
        }
    }
    func command(_ command: NativeCommand) {
        if case .load(let settings) = command {
            lock.lock(); loads += 1; let count = loads; lock.unlock()
            precondition(count == 1 && settings.loadIdentity == .managedQwen3,"Unexpected model load")
        }
        if case .chat = command { hit("chat") }
    }
    func transition(_ point: String) {
        lock.lock()
        if point == "before" { prepared += 1 } else { committed += 1 }
        lock.unlock(); hit(point)
    }
    var transitions: (Int,Int) { lock.lock(); defer { lock.unlock() }; return (prepared,committed) }
    func fault() -> NativeTransitionRetirementFault? {
        lock.lock(); defer { lock.unlock() }; let value = retirement; retirement = nil; return value
    }
    func wait(_ worker: NativeWorker) throws {
        guard reached.wait(timeout:.now()+30) == .success else {
            FileHandle.standardError.write(Data((String(reflecting:worker.snapshot()) + "\n").utf8.prefix(131072)))
            throw NativeContextFailure("Preference barrier deadline; snapshot retained")
        }
    }
    func release() { resume.signal() }
    var modelLoads: Int { lock.lock(); defer { lock.unlock() }; return loads }
}

@main
@MainActor
struct FriendlyPreferenceLoadedMain {
    static func require(_ value: Bool, _ message: String) throws { if !value { throw NativeContextFailure(message) } }
    static func digest(_ bytes: Data) -> String { SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined() }
    static func namespace() -> NativeAcceptedIdentity.Namespace {
        .init(version:1,manifestAlgorithm:"toklet-full-manifest-v1",manifestSHA256:NativeAcceptedIdentity.manifestSHA256,
              backend:"metal_native_dense_v1",load:"qwen3_audited_v1",text:"qwen3-plain-nonthinking-v1",planning:"native_mlx_planning_v1")
    }
    static func settle(_ harness: HarnessStore, _ chat: FriendlyChatStore, _ worker: NativeWorker) throws {
        for _ in 0..<4 {
            guard worker.waitUntilIdle(seconds:180) != nil else { throw NativeContextFailure("Preference worker idle deadline") }
            harness.poll(); chat.observe(harness.state)
            if !worker.snapshot().busy && !chat.restoringPreference { return }
        }
        throw NativeContextFailure("Preference restore did not settle")
    }
    static func frame(_ label: String, _ worker: NativeWorker, _ chat: FriendlyChatStore,
                      _ record: URL, _ workspace: URL, _ writes: Int, required: UInt64? = nil) throws {
        let value = worker.snapshot(), encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let bytes = try Data(contentsOf:record)
        var row: [String:Any] = ["event":label,"snapshot":String(reflecting:value),"busy":value.busy,
            "restart_required":value.restartRequired,"status":value.status,"message":value.message,
            "preference_message":chat.preferenceMessage,"record":try JSONSerialization.jsonObject(with:bytes),
            "record_sha256":digest(bytes),"preference_writes":writes,"reply":chat.workspace.outputLimit,
            "draft":chat.workspace.current.draft,"messages":chat.workspace.current.messages.count,
            "workspace_sha256":digest(try encoder.encode(chat.workspace)),"file_sha256":digest(try Data(contentsOf:workspace)),
            "history":value.history,"committed":value.evaluated,"pending":value.pending,"generated":value.ids.count,
            "planning_draft":String(reflecting:chat.planning.draft),"restoring":chat.restoringPreference]
        if let identity = value.acceptedIdentity { row["manifest"] = identity.namespace.manifestSHA256; row["model"] = identity.modelInstance }
        if let c = value.sessionConfiguration {
            row["configuration"] = ["model":c.modelInstance,"session":c.sessionInstance,"capacity":c.capacity,
                "chunk":c.chunk,"whole_budget":c.wholeSessionBudget,"accepted_request":c.acceptedRequestID?.uuidString as Any? ?? NSNull()]
        }
        if let t = value.transition {
            row["transition"] = ["committed":t.committed,"status":t.status,"message":t.message]
            if let h = t.headroom { row["headroom"] = ["required":h.requiredBytes,"available":h.availableBytes,"fits":h.fits,"instrument":h.instrument] }
        }
        if let required { row["rendered_required_tokens"] = required; row["required_source"] = "actual worker preflight message" }
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject:row,options:[.sortedKeys]) + Data([10]))
    }
    static func main() throws {
        guard CommandLine.arguments.count == 3 else { throw NativeContextFailure("Expected cached model and fresh output directory") }
        let model = URL(fileURLWithPath:CommandLine.arguments[1]), output = URL(fileURLWithPath:CommandLine.arguments[2])
        try FileManager.default.createDirectory(at:output,withIntermediateDirectories:false)
        let file = output.appendingPathComponent("chat.json"), directory = output.appendingPathComponent("preferences-v1")
        // This is deliberately an untrusted persisted fixture. Only the actual
        // audited Use below may select its namespace and admit its settings.
        let seed = FriendlyModelPreference(namespace:namespace(),capacity:256,chunk:16,reply:64,wholeBudget:HubProfile.qwen3.numericalBudget)
        try FriendlyPreferenceStore(directory:directory).save(seed)
        var workspace = ChatWorkspace(); workspace.conversations[0].draft = "Keep this unsent preference draft."
        try JSONEncoder().encode(workspace).write(to:file,options:.atomic)
        let gate = PreferenceGate(), harness = HarnessStore(enabled:true)
        guard let initial = harness.worker else { throw NativeContextFailure("No initial worker") }
        initial.close(); try require(initial.waitUntilClosed(seconds:30),"Initial worker not closed")
        let worker = try NativeWorker(beforeCommandDequeue:{ gate.command($0) },
            beforeTransitionAcceptance:{ _,_ in gate.transition("before") },afterTransitionAcceptance:{ _ in gate.transition("after") },
            transitionRetirementFault:{ gate.fault() })
        harness.worker = worker; harness.state = worker.snapshot()
        var preferenceWrites = 0
        let preferences = FriendlyPreferenceStore(directory:directory,persist:{ bytes,url in
            preferenceWrites += 1; try bytes.write(to:url,options:.atomic)
        })
        // Reconstructed workspace and store, then explicit managed Use. There
        // is no automatic path-based reopening or managed promotion.
        let chat = FriendlyChatStore(harness:harness,fileURL:file,preferences:preferences)
        let record = try preferences.url(namespace()), seeded = try Data(contentsOf:record)
        var failure: Error?
        do {
            try require(chat.acceptedIdentity == nil && !worker.snapshot().hasModel && preferenceWrites == 0,"Startup used a path as identity")
            chat.openManaged(model,profile:.qwen3,ifSelected:chat.workspace.selected)
            try settle(harness,chat,worker)
            try frame("restored",worker,chat,record,file,preferenceWrites)
            let original = worker.snapshot()
            try require(chat.ready && original.acceptedIdentity != nil && original.sessionConfiguration?.capacity == 256
                && original.sessionConfiguration?.chunk == 16 && chat.workspace.outputLimit == 64
                && original.sessionConfiguration?.acceptedRequestID != nil && preferenceWrites == 0
                && (try Data(contentsOf:record)) == seeded && gate.modelLoads == 1,"Freshly admitted restore failed or rewrote its record")
            let owner = original.modelContext?.modelInstance, draft = chat.workspace.current.draft
            func stableOwner() throws {
                let v = worker.snapshot()
                try require(v.modelContext?.modelInstance == owner && v.acceptedIdentity == original.acceptedIdentity
                    && v.modelContext?.loadedLimit == 512 && v.history == 0 && v.evaluated == 0 && v.ids.isEmpty,"Owner/prefix changed unexpectedly")
            }
            func preview(_ upper: UInt32, _ chunk: UInt32) throws {
                chat.showPlanning(); chat.editPlanning(.upper,String(upper)); chat.editPlanning(.chunk,String(chunk))
                chat.editPlanning(.budgetMiB,"1536"); chat.previewPlanning(); try settle(harness,chat,worker)
                try require(chat.planning.result?.capacity?.capacity.selected?.capacity == upper && chat.planningApplyUnavailableReason == nil,"Actual preference preview failed")
            }
            // A newer valid reply edit before acceptance cancels the candidate,
            // then saves only the still-current configuration after cleanup.
            try preview(128,8); gate.arm("before"); chat.applyPlanning(); try gate.wait(worker)
            let beforeCancel = try Data(contentsOf:record), beforeCancelWrites = preferenceWrites
            chat.setReplyLimit(128)
            try frame("precommit-reply-edit-busy",worker,chat,record,file,preferenceWrites)
            try require(preferenceWrites == beforeCancelWrites && (try Data(contentsOf:record)) == beforeCancel,"Busy candidate wrote preferences")
            gate.release(); try settle(harness,chat,worker)
            try frame("precommit-reply-edit-terminal",worker,chat,record,file,preferenceWrites)
            try require(worker.snapshot().sessionConfiguration == original.sessionConfiguration
                && chat.workspace.outputLimit == 128 && (try preferences.load(namespace()))?.reply == 128
                && (try preferences.load(namespace()))?.capacity == 256,"Cancelled candidate changed accepted settings")
            try stableOwner()
            // A commit wins, then its transient result disappears while old
            // ownership is still being retired. Keep current facts/newer edits.
            chat.setReplyLimit(32); try preview(128,8)
            gate.arm("after"); chat.applyPlanning(); try gate.wait(worker)
            let beforeBusy = try Data(contentsOf:record), beforeBusyWrites = preferenceWrites
            chat.hidePlanning(); worker.cancel(); chat.showPlanning(); chat.editPlanning(.upper,"512"); chat.setReplyLimit(64)
            harness.poll(); chat.observe(harness.state)
            try frame("postcommit-edit-busy",worker,chat,record,file,preferenceWrites)
            try require(worker.snapshot().sessionConfiguration?.capacity == 128 && worker.snapshot().transition == nil
                && chat.planning.draft.upper == "512" && chat.workspace.outputLimit == 64
                && preferenceWrites == beforeBusyWrites && (try Data(contentsOf:record)) == beforeBusy,"Early save or newer edit lost")
            gate.release(); try settle(harness,chat,worker)
            try frame("postcommit-edit-terminal",worker,chat,record,file,preferenceWrites)
            try require((try preferences.load(namespace()))?.capacity == 128 && (try preferences.load(namespace()))?.reply == 64
                && chat.planning.draft.upper == "512" && chat.workspace.current.draft == draft,"Accepted settings/newer edits not retained")
            try stableOwner()
            // Minimum reply is mode-valid at33, but the actual template must
            // still fit. Gate the chat before dequeue to test newer-draft rollback.
            chat.setReplyLimit(32); try preview(33,1); chat.applyPlanning(); try settle(harness,chat,worker)
            try frame("small-window-mode-valid",worker,chat,record,file,preferenceWrites)
            try require(worker.snapshot().sessionConfiguration?.capacity == 33 && chat.outputLimitAvailable,"Minimum mode choice failed")
            chat.edit("a"); _ = chat.save(); gate.arm("chat"); chat.sendOrStop(); try gate.wait(worker)
            chat.edit("Keep this newer draft after rendered rejection."); _ = chat.save()
            gate.release(); try settle(harness,chat,worker)
            let rejected = worker.snapshot(), error = rejected.chatResult?.error ?? ""
            let prefix = "This request needs ", tokenText = error.hasPrefix(prefix) ? String(error.dropFirst(prefix.count)).components(separatedBy:" tokens").first ?? "" : ""
            let required = UInt64(tokenText.filter { $0.isASCII && $0.isNumber })
            try frame("small-window-rendered-rejection",worker,chat,record,file,preferenceWrites,required:required)
            try require(rejected.chatResult?.rejected == true && (required ?? 0) > 33 && error.contains("Increase context")
                && rejected.ids.isEmpty && rejected.history == 0 && rejected.evaluated == 0 && rejected.pending == 0
                && chat.workspace.current.messages.isEmpty && chat.workspace.current.draft == "Keep this newer draft after rendered rejection."
                && rejected.sessionConfiguration?.capacity == 33,"Rendered rejection/draft rollback failed")
            try stableOwner()
            // A committed-but-quarantined session must never replace the last
            // healthy file. The injected BUSY retains one old owner for close.
            try preview(64,8); gate.arm("after",quarantine:true); chat.applyPlanning(); try gate.wait(worker)
            let beforeQuarantine = try Data(contentsOf:record), beforeQuarantineWrites = preferenceWrites
            chat.editPlanning(.chunk,"4"); harness.poll(); chat.observe(harness.state)
            try frame("quarantine-commit-busy",worker,chat,record,file,preferenceWrites)
            try require(preferenceWrites == beforeQuarantineWrites && (try Data(contentsOf:record)) == beforeQuarantine,"Quarantine wrote before retirement")
            gate.release(); try settle(harness,chat,worker)
            try frame("quarantine-terminal",worker,chat,record,file,preferenceWrites)
            try require(worker.snapshot().restartRequired && worker.snapshot().sessionConfiguration?.capacity == 64
                && !chat.ready && chat.planning.draft.chunk == "4" && preferenceWrites == beforeQuarantineWrites
                && (try Data(contentsOf:record)) == beforeQuarantine,"Quarantined commit was saved or presented as rollback")
            try stableOwner()
        } catch { failure = error }
        gate.release(); worker.close()
        try require(worker.waitUntilClosed(seconds:60),"Preference worker not closed")
        harness.poll(); chat.observe(harness.state)
        try frame("closed",worker,chat,record,file,preferenceWrites)
        if let failure { throw failure }
        try require(gate.transitions.0 == 5 && gate.transitions.1 == 4 && preferenceWrites == 5,"Unexpected candidate/commit/write count")
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject:["passed":true,"closed":true,
            "actual_model_loads":gate.modelLoads,"preference_writes":preferenceWrites,"prepared_candidates":gate.transitions.0,"accepted_replacements":gate.transitions.1,
            "real_generation_tokens":0,"rendered_small_window_rejection":true,"quarantine_injected":true,
            "rendered_ui_checked":false,"phone_checked":false],options:[.sortedKeys]) + Data([10]))
    }
}
