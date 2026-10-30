import Foundation
import CryptoKit

private final class IdentityLoadCount: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    func observe(_ command: NativeCommand) {
        guard case .load(let settings) = command else { return }
        lock.lock(); count += 1; let value = count; lock.unlock()
        // Fail before a second load reaches the native factory. This workload
        // has one real model, not a fallback/reload qualification campaign.
        guard value == 1 && settings.loadIdentity == .managedQwen3 && settings.loadRequestID != nil else {
            FileHandle.standardError.write(Data("Unexpected additional or ordinary model load\n".utf8))
            preconditionFailure("Identity workload load boundary")
        }
    }
    var loads: Int { lock.lock(); defer { lock.unlock() }; return count }
}

@main
@MainActor
struct FriendlyIdentityLoadedMain {
    static func require(_ value: Bool, _ message: String) throws {
        if !value { throw NativeContextFailure(message) }
    }
    static func emit(_ value: [String:Any]) throws {
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys]) + Data([10]))
    }
    static func digest(_ data: Data) -> String { SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined() }
    static func identity(_ value: NativeAcceptedIdentity?) -> Any {
        guard let value else { return NSNull() }
        return ["namespace":["version":value.namespace.version,"manifest_algorithm":value.namespace.manifestAlgorithm,
                             "manifest_sha256":value.namespace.manifestSHA256,"backend":value.namespace.backend,
                             "load":value.namespace.load,"text":value.namespace.text,"planning":value.namespace.planning],
                "model_instance":value.modelInstance,"loaded_limit":value.loadedLimit,
                "source_ranges":value.sourceRanges,"source_reads":value.sourceReads,
                "assets":value.assets.map { ["name":$0.name,"bytes":$0.bytes,"sha256":$0.sha256,"roles":$0.roles] as [String:Any] }] as [String:Any]
    }
    static func frame(_ event: String, _ worker: NativeWorker, _ chat: FriendlyChatStore, _ file: URL, _ writes: Int) throws {
        let value = worker.snapshot(), encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        var row: [String:Any] = ["event":event,"snapshot":String(reflecting:value),"identity":identity(value.acceptedIdentity),
                  "friendly_identity":identity(chat.acceptedIdentity),"load_request_id":value.acceptedLoadRequestID?.uuidString as Any? ?? NSNull(),
                  "status":value.status,"message":value.message,"workspace_sha256":digest(try encoder.encode(chat.workspace)),
                  "file_sha256":digest(try Data(contentsOf:file)),"writes":writes]
        if let context = value.modelContext {
            row["model_context"] = ["model_instance":context.modelInstance,"loaded_limit":context.loadedLimit,
                "configured_capacity":context.configuredCapacity,"backend":context.backend]
        }
        if let configuration = value.sessionConfiguration {
            row["configuration"] = ["model_instance":configuration.modelInstance,"session_instance":configuration.sessionInstance,
                "capacity":configuration.capacity,"chunk":configuration.chunk,"whole_budget":configuration.wholeSessionBudget]
        }
        if let result = value.transition {
            row["transition"] = ["committed":result.committed,"status":result.status,"message":result.message,
                "request_id":result.request.id.uuidString,"plan":String(reflecting:result.plan)]
            if let plan = result.plan { row["aggregate_required_bytes"] = plan.aggregateRequiredBytes }
            if let headroom = result.headroom { row["headroom"] = ["required":headroom.requiredBytes,"available":headroom.availableBytes,
                "instrument":headroom.instrument,"fits":headroom.fits] }
        }
        try emit(row)
    }
    static func settle(_ harness: HarnessStore, _ chat: FriendlyChatStore, _ worker: NativeWorker) throws {
        guard worker.waitUntilIdle(seconds:180) != nil else { throw NativeContextFailure("Identity worker idle deadline") }
        harness.poll(); chat.observe(harness.state)
    }
    static func main() throws {
        guard CommandLine.arguments.count == 3 else { throw NativeContextFailure("Expected pinned model and fresh output directory") }
        let model = URL(fileURLWithPath:CommandLine.arguments[1]), output = URL(fileURLWithPath:CommandLine.arguments[2])
        try FileManager.default.createDirectory(at:output,withIntermediateDirectories:false)
        let file = output.appendingPathComponent("history.json"), counter = IdentityLoadCount()
        let harness = HarnessStore(enabled:true)
        guard let initial = harness.worker else { throw NativeContextFailure("Missing initial worker") }
        initial.close(); try require(initial.waitUntilClosed(seconds:30),"Initial worker did not close")
        let worker = try NativeWorker(beforeCommandDequeue:{ counter.observe($0) })
        harness.worker = worker; harness.state = worker.snapshot()
        var writes = 0
        let chat = FriendlyChatStore(harness:harness,fileURL:file,persist:{ bytes,url in
            writes += 1; try bytes.write(to:url,options:.atomic)
        })
        var failure: Error?, completed = false
        do {
            chat.openManaged(model,profile:.qwen3,ifSelected:chat.workspace.selected)
            try settle(harness,chat,worker)
            // Raw facts precede acceptance assertions, including rejected loads.
            if !FileManager.default.fileExists(atPath:file.path) { _ = chat.save() }
            try frame("loaded",worker,chat,file,writes)
            let original = worker.snapshot()
            guard let accepted = original.acceptedIdentity, let configuration = original.sessionConfiguration,
                  let context = original.modelContext, let request = original.acceptedLoadRequestID else {
                throw NativeContextFailure("Audited load did not publish complete identity: \(original.status) \(original.message)")
            }
            try require(chat.ready && original.status == 0 && chat.acceptedIdentity == accepted && accepted.matches(context)
                        && accepted.namespace.manifestSHA256 == NativeAcceptedIdentity.manifestSHA256
                        && accepted.loadedLimit == 512 && configuration.capacity == 512 && configuration.chunk == 32
                        && original.history == 0 && counter.loads == 1,"Audited accepted model identity mismatch")
            chat.edit("Keep this unsent identity-test draft."); try require(chat.save(),"Baseline persistence failed")
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            let before = try encoder.encode(chat.workspace), persisted = try Data(contentsOf:file), baselineWrites = writes
            try frame("baseline",worker,chat,file,writes)
            func unchanged() throws {
                try require(try encoder.encode(chat.workspace) == before && Data(contentsOf:file) == persisted && writes == baselineWrites,
                            "Identity observation changed workspace or persistence")
            }
            chat.showPlanning(); chat.editPlanning(.upper,"256"); chat.hidePlanning()
            worker.cancel(); try settle(harness,chat,worker)
            try frame("preview-cancel-preserved",worker,chat,file,writes)
            try require(worker.snapshot().acceptedIdentity == accepted && chat.acceptedIdentity == accepted
                        && worker.snapshot().acceptedLoadRequestID == request,"Preview invalidation/cancellation cleared retained owner identity")
            try unchanged()
            worker.interrupt(unload:false); try settle(harness,chat,worker)
            try frame("suspended",worker,chat,file,writes)
            let suspended = worker.snapshot()
            try require(!suspended.hasSession && suspended.hasModel && suspended.chatEnabled && suspended.acceptedIdentity == accepted
                        && chat.acceptedIdentity == accepted && suspended.sessionConfiguration == nil,"Suspend lost retained model/codec identity")
            try unchanged()
            try require(worker.submit(.replay),"Empty replay could not enqueue")
            try settle(harness,chat,worker); try frame("replayed",worker,chat,file,writes)
            let replayed = worker.snapshot()
            try require(replayed.status == 0 && replayed.hasSession && replayed.sessionConfiguration?.sessionInstance != configuration.sessionInstance
                        && replayed.acceptedIdentity == accepted && replayed.acceptedLoadRequestID == request
                        && replayed.history == 0 && replayed.evaluated == 0 && replayed.ids.isEmpty,"Empty replay changed model evidence or generated tokens")
            try unchanged()
            guard let live = replayed.liveContext else { throw NativeContextFailure("Missing replay context") }
            let selection = NativeTransitionSelection(id:UUID(),modelInstance:live.modelInstance,sessionInstance:live.sessionInstance)
            var replacement = NativeTransitionRequest(id:UUID(),selection:selection,capacity:256,chunk:16,
                wholeSessionBudget:HubProfile.qwen3.numericalBudget,coexistenceBudget:HubProfile.qwen3.processAllowance,
                replayReserve:0,textReserve:HubProfile.qwen3.textBudget,applicationReserve:NativeTransitionHeadroomCheck.applicationReserve)
            replacement.memoryPolicy = .friendly
            try require(worker.selectTransition(selection) && worker.submit(.replaceWithEmptySession(replacement)),"Identity replacement could not enqueue")
            try settle(harness,chat,worker); try frame("replaced",worker,chat,file,writes)
            let replaced = worker.snapshot()
            try require(replaced.transition?.request.id == replacement.id && replaced.transition?.committed == true
                        && replaced.status == 0 && !replaced.restartRequired && replaced.transition?.headroom?.fits == true
                        && replaced.sessionConfiguration?.capacity == 256 && replaced.sessionConfiguration?.chunk == 16
                        && replaced.sessionConfiguration?.sessionInstance != live.sessionInstance
                        && replaced.acceptedIdentity == accepted && chat.acceptedIdentity == accepted
                        && replaced.acceptedLoadRequestID == request && replaced.modelContext?.loadedLimit == 512
                        && replaced.modelContext?.modelInstance == accepted.modelInstance
                        && replaced.history == 0 && replaced.evaluated == 0,"Empty replacement changed model identity or failed admission")
            try unchanged()
            // Same path with explicit ordinary intent must not be declared an
            // already-open audited route. It queues unload. Change selection
            // before the pending load is dequeued; no second load may occur.
            chat.open(model)
            try require(chat.loadingModel,"Distinct explicit route was silently treated as already open")
            try chat.workspace.newChat(modelKey:"different-conversation")
            try settle(harness,chat,worker); try frame("stale-pending-discarded",worker,chat,file,writes)
            try require(!chat.loadingModel && !worker.snapshot().hasModel && chat.acceptedIdentity == nil
                        && worker.snapshot().acceptedIdentity == nil && worker.snapshot().acceptedLoadRequestID == nil
                        && counter.loads == 1,"Stale conversation redirected pending load or retained identity")
            completed = true
        } catch { failure = error }
        worker.close()
        try require(worker.waitUntilClosed(seconds:60),"Owning identity worker did not close")
        harness.poll(); chat.observe(harness.state)
        if FileManager.default.fileExists(atPath:file.path) { try frame("closed",worker,chat,file,writes) }
        try require(worker.snapshot().acceptedIdentity == nil && worker.snapshot().acceptedLoadRequestID == nil && chat.acceptedIdentity == nil,
                    "Closed owners retained accepted identity")
        if let failure { throw failure }
        try emit(["passed":completed,"closed":true,"actual_model_loads":counter.loads,"actual_getter_queries":1,"empty_session_replacements":1,
                  "generated_tokens":0,"rendered_ui_checked":false,"preferences_written":false])
    }
}
