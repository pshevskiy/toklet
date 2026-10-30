import Foundation
import CryptoKit

private final class Qwen2PreferenceGate: @unchecked Sendable {
    private let lock = NSLock()
    private let entered = DispatchSemaphore(value:0), released = DispatchSemaphore(value:0)
    private var loads = 0, prepared = 0, committed = 0, previews = 0
    private var request: NativeTransitionRequest?
    func command(_ command: NativeCommand) {
        lock.lock(); defer { lock.unlock() }
        if case .load(let settings) = command {
            loads += 1
            precondition(loads == 1 && settings.loadIdentity == .managedQwen2,"Unexpected Qwen2 load")
        }
        if case .preview = command { previews += 1 }
    }
    func before(_ request: NativeTransitionRequest) {
        lock.lock(); prepared += 1; let count = prepared; self.request = request; lock.unlock()
        precondition(count <= 2,"Unexpected candidate")
        entered.signal()
        guard released.wait(timeout:.now()+30) == .success else {
            FileHandle.standardError.write(Data("Qwen2 acceptance barrier deadline\n".utf8))
            preconditionFailure("Qwen2 acceptance barrier deadline")
        }
    }
    func after() { lock.lock(); committed += 1; lock.unlock() }
    func wait(_ worker: NativeWorker) throws -> NativeTransitionRequest {
        guard entered.wait(timeout:.now()+30) == .success else {
            FileHandle.standardError.write(Data((String(reflecting:worker.snapshot()) + "\n").utf8.prefix(131072)))
            throw NativeContextFailure("Qwen2 acceptance deadline")
        }
        lock.lock(); defer { lock.unlock() }
        guard let request else { throw NativeContextFailure("Missing staged request") }
        return request
    }
    func release() { released.signal() }
    var counts: (Int,Int,Int,Int) { lock.lock(); defer { lock.unlock() }; return (loads,prepared,committed,previews) }
}

@main
@MainActor
struct FriendlyQwen2PreferenceLoadedMain {
    static func check(_ value: Bool, _ message: String) throws { if !value { throw NativeContextFailure(message) } }
    static func hash(_ value: Data) -> String { SHA256.hash(data:value).map { String(format:"%02x",$0) }.joined() }
    static func terminal(_ worker: NativeWorker) throws -> NativeSnapshot {
        guard let state = worker.waitUntilIdle(seconds:180) else { throw NativeContextFailure("Qwen2 worker deadline") }
        return state
    }
    static func reconcile(_ harness: HarnessStore, _ chat: FriendlyChatStore) { harness.poll(); chat.observe(harness.state) }
    static func accepted(_ state: NativeSnapshot, request: NativeTransitionRequest, identity: NativeAcceptedIdentity,
                         capacity: UInt32, chunk: UInt32) throws -> NativeSessionConfiguration {
        guard let transition = state.transition, let plan = transition.plan, let candidate = transition.candidate,
              let configuration = state.sessionConfiguration else { throw NativeContextFailure("Missing coherent accepted evidence") }
        try check(!state.busy && !state.restartRequired && state.acceptedIdentity == identity && state.status == 0
            && transition.request == request && transition.committed && transition.outcome == .committed && transition.status == 0
            && transition.headroom?.fits == true && plan.executionModes.attention == "Online"
            && plan.modelInstance == identity.modelInstance && plan.sessionInstance == request.selection.sessionInstance
            && plan.fitsTransitionBudget && plan.fitsCandidateIndividualBudget && !plan.candidateAllocated
            && candidate.modelInstance == identity.modelInstance && candidate.loadedLimit == 512 && candidate.offset == 0
            && candidate.sessionInstance != request.selection.sessionInstance && candidate.capacity == UInt64(capacity)
            && configuration.modelInstance == identity.modelInstance && configuration.sessionInstance == candidate.sessionInstance
            && configuration.acceptedRequestID == request.id && configuration.capacity == capacity && configuration.chunk == chunk
            && configuration.wholeSessionBudget == request.wholeSessionBudget,"Qwen2 accepted owner/Online/accounting/configuration mismatch")
        return configuration
    }
    static func requestFields(_ request: NativeTransitionRequest) -> [String:Any] {
        ["id":request.id.uuidString,"selection":request.selection.id.uuidString,
         "model":request.selection.modelInstance,"old_session":request.selection.sessionInstance,
         "capacity":request.capacity,"chunk":request.chunk,"whole_budget":request.wholeSessionBudget,
         "coexistence_budget":request.coexistenceBudget,"replay_reserve":request.replayReserve,
         "text_reserve":request.textReserve,"application_reserve":request.applicationReserve]
    }
    static func frame(_ event: String, _ worker: NativeWorker, _ chat: FriendlyChatStore,
                      _ workspace: URL, _ record: URL, _ others: [URL], _ writes: Int, preview: NativePreviewResult? = nil,
                      captured: NativeSnapshot? = nil, submitted: NativeTransitionRequest? = nil) throws {
        let state = captured ?? worker.snapshot(), encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        var row: [String:Any] = ["event":event,"snapshot":String(reflecting:state),"preference_writes":writes,
            "workspace_sha256":hash(try encoder.encode(chat.workspace)),"file_sha256":hash(try Data(contentsOf:workspace)),
            "record_sha256":hash(try Data(contentsOf:record)),
            "record":try JSONSerialization.jsonObject(with:Data(contentsOf:record)),
            "other_record_hashes":try others.map { hash(try Data(contentsOf:$0)) },"draft":chat.workspace.current.draft,"reply":chat.workspace.outputLimit,
            "messages":chat.workspace.current.messages.count,"history":state.history,"committed":state.evaluated,"pending":state.pending,"generated":state.ids.count]
        if let i = state.acceptedIdentity { row["model"] = i.modelInstance; row["manifest"] = i.namespace.manifestSHA256; row["loaded_limit"] = i.loadedLimit; row["load_policy"] = i.namespace.load; row["text_policy"] = i.namespace.text; row["source_ranges"] = i.sourceRanges; row["source_reads"] = i.sourceReads }
        if let c = state.sessionConfiguration { row["configuration"] = ["model":c.modelInstance,"session":c.sessionInstance,"capacity":c.capacity,"chunk":c.chunk,"whole_budget":c.wholeSessionBudget,"accepted_request":c.acceptedRequestID?.uuidString as Any? ?? NSNull()] }
        if let value = preview ?? state.preview {
            row["estimator"] = value.request.estimator.rawValue; row["profile"] = value.request.selection.profile.rawValue
            row["requested_upper"] = value.request.upper; row["effective_upper"] = value.capacityRequest?.upper as Any? ?? NSNull()
            row["selected"] = value.capacity?.capacity.selected?.capacity as Any? ?? NSNull()
            row["preview_allocated"] = value.capacity?.resources.numericalBuffersAllocated as Any? ?? NSNull()
            row["preview_identity_authority"] = false; row["preview_attention"] = value.capacity?.resources.attention.rawValue as Any? ?? NSNull()
        }
        if let submitted { row["submitted_request"] = requestFields(submitted) }
        if let t = state.transition {
            row["transition_request"] = requestFields(t.request)
            row["transition"] = ["committed":t.committed,"status":t.status,"message":t.message,"outcome":t.outcome.rawValue]
            if let p = t.plan {
                row["actual_plan"] = ["model":p.modelInstance,"session":p.sessionInstance,"loaded_limit":p.loadedPositionLimit,
                    "shared":p.sharedModel.totalBytes,"learned":p.sharedModel.learnedBytes,"positional":p.sharedModel.positionalBytes,"model_table":p.sharedModel.modelTableBytes,
                    "current_owned":p.currentAllocated.totalBytes,"candidate_planned":p.candidatePlanned.totalBytes,
                    "reserves":p.reserves.totalBytes,"current_allowance":p.reserves.currentMetadataDriverBytes,
                    "candidate_allowance":p.reserves.candidateMetadataDriverBytes,"current_bridge":p.reserves.currentBridgeBytes,
                    "candidate_bridge":p.reserves.candidateBridgeBytes,"json":p.reserves.jsonSnapshotBytes,
                    "replay":p.reserves.replayBytes,"text":p.reserves.textBytes,"application":p.reserves.applicationBytes,
                    "aggregate":p.aggregateRequiredBytes,"transition_budget":p.transitionBudgetBytes,
                    "candidate_constructor":p.candidateConstructorRequiredBytes,"candidate_budget":p.candidateWholeBudgetBytes,
                    "fits_aggregate":p.fitsTransitionBudget,"fits_individual":p.fitsCandidateIndividualBudget,
                    "candidate_allocated_in_plan":p.candidateAllocated,"attention":p.executionModes.attention] as [String:Any]
            }
            if let c = t.candidate {
                row["actual_candidate_admission"] = c.numericalAdmission
                row["actual_candidate"] = ["model":c.modelInstance,"session":c.sessionInstance,
                    "capacity":c.capacity,"loaded_limit":c.loadedLimit,"offset":c.offset]
            }
            if let h = t.headroom { row["headroom"] = ["required":h.requiredBytes,"available":h.availableBytes,"instrument":h.instrument,"fits":h.fits] }
        }
        let raw = try JSONSerialization.data(withJSONObject:row,options:[.sortedKeys]);try check(raw.count<=131072,"Frame cap")
        FileHandle.standardOutput.write(raw+Data([10]))
    }
    static func main() throws {
        guard CommandLine.arguments.count == 3 else { throw NativeContextFailure("Expected cached Qwen2 model and exclusive output directory") }
        let model = URL(fileURLWithPath:CommandLine.arguments[1]), output = URL(fileURLWithPath:CommandLine.arguments[2])
        try FileManager.default.createDirectory(at:output,withIntermediateDirectories:false)
        let file = output.appendingPathComponent("chat.json"), directory = output.appendingPathComponent("preferences-v1")
        let namespace = NativeAcceptedIdentity.Profile.qwen2.namespace
        let seed = FriendlyModelPreference(namespace:namespace,capacity:256,chunk:16,reply:64,wholeBudget:HubProfile.qwen.numericalBudget)
        let initialStore = FriendlyPreferenceStore(directory:directory)
        try initialStore.save(seed)
        let record = try initialStore.url(namespace), seedBytes = try Data(contentsOf:record)
        let others = try [(NativeAcceptedIdentity.Profile.qwen3,HubProfile.qwen3),(.smol,.smol)].map { identity,profile -> URL in
            try initialStore.save(.init(namespace:identity.namespace,capacity:128,chunk:8,reply:32,wholeBudget:profile.numericalBudget))
            return try initialStore.url(identity.namespace)
        }
        let otherBytes = try others.map { try Data(contentsOf:$0) }
        var workspace = ChatWorkspace(); workspace.conversations[0].draft = "Keep this Qwen2 unsent draft."
        let conversation = workspace.selected, draft = workspace.current.draft
        try JSONEncoder().encode(workspace).write(to:file,options:.atomic)
        let gate = Qwen2PreferenceGate(), harness = HarnessStore(enabled:true)
        guard let initial = harness.worker else { throw NativeContextFailure("Missing initial worker") }
        initial.close(); try check(initial.waitUntilClosed(seconds:30),"Initial close deadline")
        let worker = try NativeWorker(beforeCommandDequeue:{ gate.command($0) },
            beforeTransitionAcceptance:{ request,_ in gate.before(request) },afterTransitionAcceptance:{ _ in gate.after() })
        harness.worker = worker; harness.state = worker.snapshot()
        var writes = 0
        let preferences = FriendlyPreferenceStore(directory:directory,persist:{ bytes,url in writes += 1; try bytes.write(to:url,options:.atomic) })
        let chat = FriendlyChatStore(harness:harness,fileURL:file,preferences:preferences)
        var failure: Error?
        do {
            try check(chat.acceptedIdentity == nil && !worker.snapshot().hasModel && writes == 0,"Stored preferences became identity before Use")
            chat.openManaged(model,profile:.qwen,ifSelected:conversation)
            let loaded = try terminal(worker)
            reconcile(harness,chat) // Native identity now allows one fresh automatic restore.
            let restoring = try gate.wait(worker)
            try frame("restore-before-acceptance",worker,chat,file,record,others,writes,submitted:restoring)
            guard let identity = loaded.acceptedIdentity, let original = loaded.sessionConfiguration else { throw NativeContextFailure("Audited Qwen2 load did not publish identity") }
            try check(identity.profile == .qwen2 && identity.loadedLimit == 512 && identity.sourceRanges == 630
                && identity.namespace == namespace && chat.acceptedIdentity == identity
                && original.capacity == 512 && original.chunk == 32 && original.acceptedRequestID == nil
                && restoring.selection.sessionInstance == original.sessionInstance
                && chat.restoringPreference && !chat.ready && !chat.canSend && writes == 0
                && (try Data(contentsOf:record)) == seedBytes,"Restore started from wrong/default/unauthenticated state")
            gate.release(); let restored = try terminal(worker)
            try frame("restore-accepted-before-reconcile",worker,chat,file,record,others,writes,captured:restored,submitted:restoring)
            let restoredConfiguration = try accepted(restored,request:restoring,identity:identity,capacity:256,chunk:16)
            reconcile(harness,chat)
            try frame("restored",worker,chat,file,record,others,writes)
            try check(chat.ready && !chat.restoringPreference && chat.currentSessionConfiguration == restoredConfiguration
                && chat.workspace.outputLimit == 64 && writes == 0 && (try Data(contentsOf:record)) == seedBytes,"Restore reconciliation changed record or failed")
            func preserved() throws {
                let state = worker.snapshot()
                try check(state.acceptedIdentity == identity && state.modelContext?.modelInstance == identity.modelInstance
                    && state.modelContext?.loadedLimit == 512 && state.history == 0 && state.evaluated == 0 && state.pending == 0 && state.ids.isEmpty
                    && chat.workspace.selected == conversation && chat.workspace.current.draft == draft && chat.workspace.current.messages.isEmpty
                    && (try others.map { try Data(contentsOf:$0) }) == otherBytes,"Identity, other records, conversation or empty prefix changed")
            }
            try preserved()
            chat.setReplyLimit(32)
            try frame("reply-saved",worker,chat,file,record,others,writes)
            try check(writes == 1 && chat.workspace.outputLimit == 32 && (try preferences.load(namespace))?.reply == 32
                && (try preferences.load(namespace))?.capacity == 256,"Explicit reply save failed")
            let replyBytes = try Data(contentsOf:record), workspaceBytes = try Data(contentsOf:file)
            chat.setReplyLimit(512)
            try check(chat.workspace.outputLimit == 32 && writes == 1 && (try Data(contentsOf:record)) == replyBytes
                && (try Data(contentsOf:file)) == workspaceBytes,"Rejected reply changed saved settings")
            chat.showPlanning(); chat.editPlanning(.upper,"128"); chat.editPlanning(.chunk,"8"); chat.previewPlanning()
            _ = try terminal(worker); reconcile(harness,chat)
            guard let preview = chat.planning.result, let plan = preview.capacity else { throw NativeContextFailure("Missing Qwen2 numerical estimate") }
            try frame("preview",worker,chat,file,record,others,writes,preview:preview)
            try check(preview.outcome == .planned && preview.status == 0 && preview.request.selection.profile == .ordinary
                && preview.request.estimator == .sourceProfile && preview.request.resident == nil
                && preview.request.upper == 128 && preview.capacityRequest?.upper == 128 && preview.capacityRequest?.chunk == 8
                && plan.resources.attention == .online && plan.capacity.selected?.capacity == 128
                && !plan.resources.numericalBuffersAllocated && plan.resources.payloadValidationRequired
                && chat.planningApplyUnavailableReason == nil && worker.snapshot().sessionConfiguration == restoredConfiguration
                && writes == 1 && (try Data(contentsOf:record)) == replyBytes && (try Data(contentsOf:file)) == workspaceBytes,
                "Advisory estimate changed current owners/records or Qwen2 mode")
            chat.applyPlanning()
            guard let applied = chat.applying.request else { throw NativeContextFailure("Apply request missing") }
            let applying = try gate.wait(worker)
            try check(applied.transition == applying && applied.previewID == preview.request.id,"Apply changed selected request")
            gate.release(); let terminalApply = try terminal(worker)
            try frame("apply-accepted-before-reconcile",worker,chat,file,record,others,writes,preview:preview,captured:terminalApply,submitted:applying)
            let configuration = try accepted(terminalApply,request:applying,identity:identity,capacity:128,chunk:8)
            try check(applying.selection.sessionInstance == restoredConfiguration.sessionInstance && writes == 1,"Unexpected old Apply owner or premature save")
            reconcile(harness,chat)
            try frame("applied",worker,chat,file,record,others,writes,preview:preview,submitted:applying)
            try check(worker.snapshot().sessionConfiguration == configuration && chat.currentSessionConfiguration == configuration
                && chat.applying.committed && !chat.applying.pending && writes == 2
                && (try preferences.load(namespace)) == FriendlyModelPreference(namespace:namespace,capacity:128,chunk:8,reply:32,wholeBudget:HubProfile.qwen.numericalBudget)
                && (try Data(contentsOf:file)) == workspaceBytes,"Durable Apply/save changed user state or lost accepted configuration")
            try preserved()
        } catch { failure = error }
        gate.release(); worker.close(); try check(worker.waitUntilClosed(seconds:60),"Qwen2 worker close deadline")
        reconcile(harness,chat); try frame("closed",worker,chat,file,record,others,writes)
        if let failure { throw failure }
        let counts = gate.counts
        try check(counts.0 == 1 && counts.1 == 2 && counts.2 == 2 && counts.3 == 1 && writes == 2
            && worker.snapshot().acceptedIdentity == nil && chat.acceptedIdentity == nil,"Unexpected work count or retained identity")
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject:["passed":true,"closed":true,"model_loads":counts.0,
            "prepared_candidates":counts.1,"accepted_replacements":counts.2,"previews":counts.3,"preference_writes":writes,
            "real_generation_tokens":0,"other_records_unchanged":true,"rendered_ui_checked":false,"phone_checked":false],options:[.sortedKeys])+Data([10]))
    }
}
